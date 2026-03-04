from __future__ import annotations

import base64
import json
import mimetypes
import sys
import time
import urllib.request
from pathlib import Path
from uuid import uuid4

import modal

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from modal_doc_parsing_vlm.cleanup import cleanup_expired_jobs
from modal_doc_parsing_vlm.config import (
    APP_NAME,
    ARTIFACT_ROOT,
    ARTIFACTS_VOLUME_NAME,
    CONTROL_PLANE_DEPENDENCIES,
    CONTROL_PLANE_PYTHON_VERSION,
    DEFAULT_RUNTIME_PROFILE,
    ENABLED_RUNTIME_PROFILES,
    HF_CACHE_VOLUME_NAME,
    IDEMPOTENCY_DICT_NAME,
    JOB_STATUS_DICT_NAME,
    ORCHESTRATOR_TIMEOUT_SECONDS,
    RETENTION_DAYS,
    VLLM_CACHE_VOLUME_NAME,
    get_runtime_profile,
)
from modal_doc_parsing_vlm.engine import create_engine_cls
from modal_doc_parsing_vlm.mcp_server import build_fastapi_app
from modal_doc_parsing_vlm.orchestrator import DocumentParseService, process_job
from modal_doc_parsing_vlm.storage import FileSystemStorageBackend
from modal_doc_parsing_vlm.types_api import (
    GetDocumentParseResultRequest,
    GetDocumentParseStatusRequest,
    SubmitDocumentParseRequest,
)
from modal_doc_parsing_vlm.types_result import MimeType

app = modal.App(APP_NAME)

control_plane_image = modal.Image.debian_slim(
    python_version=CONTROL_PLANE_PYTHON_VERSION
).uv_pip_install(*CONTROL_PLANE_DEPENDENCIES).add_local_python_source("modal_doc_parsing_vlm")

hf_cache_volume = modal.Volume.from_name(HF_CACHE_VOLUME_NAME, create_if_missing=True)
vllm_cache_volume = modal.Volume.from_name(VLLM_CACHE_VOLUME_NAME, create_if_missing=True)
artifacts_volume = modal.Volume.from_name(
    ARTIFACTS_VOLUME_NAME, create_if_missing=True, version=2
)
job_status_dict = modal.Dict.from_name(JOB_STATUS_DICT_NAME, create_if_missing=True)
idempotency_dict = modal.Dict.from_name(IDEMPOTENCY_DICT_NAME, create_if_missing=True)

ProdParserEngine = None
DevParserEngine = None
_ENGINE_CLASSES: dict[str, object] = {}

if "prod" in ENABLED_RUNTIME_PROFILES:
    ProdParserEngine = create_engine_cls(
        app,
        runtime_profile=get_runtime_profile("prod"),
        hf_cache_volume=hf_cache_volume,
        vllm_cache_volume=vllm_cache_volume,
        artifacts_volume=artifacts_volume,
        export_module=__name__,
    )
    _ENGINE_CLASSES["prod"] = ProdParserEngine

if "dev" in ENABLED_RUNTIME_PROFILES:
    DevParserEngine = create_engine_cls(
        app,
        runtime_profile=get_runtime_profile("dev"),
        hf_cache_volume=hf_cache_volume,
        vllm_cache_volume=vllm_cache_volume,
        artifacts_volume=artifacts_volume,
        export_module=__name__,
    )
    _ENGINE_CLASSES["dev"] = DevParserEngine


def build_storage() -> FileSystemStorageBackend:
    return FileSystemStorageBackend(
        ARTIFACT_ROOT,
        status_store=job_status_dict,
        idempotency_store=idempotency_dict,
        volume=artifacts_volume,
    )


def build_service(runtime_profile_name: str | None = None) -> DocumentParseService:
    runtime_profile = get_runtime_profile(runtime_profile_name or DEFAULT_RUNTIME_PROFILE)

    def schedule_job(job_id: str, selected_profile_name: str) -> None:
        run_orchestrator.spawn(job_id, selected_profile_name)

    return DocumentParseService(
        storage=build_storage(),
        runtime_profile=runtime_profile,
        schedule_job=schedule_job,
    )


def _engine_for_profile(runtime_profile_name: str):
    try:
        return _ENGINE_CLASSES[runtime_profile_name]()
    except KeyError as exc:
        raise RuntimeError(
            f"Runtime profile {runtime_profile_name!r} is not enabled. "
            f"Enabled profiles: {', '.join(ENABLED_RUNTIME_PROFILES) or '(none)'}"
        ) from exc


@app.function(
    image=control_plane_image,
    volumes={str(ARTIFACT_ROOT): artifacts_volume},
    timeout=ORCHESTRATOR_TIMEOUT_SECONDS,
)
def run_orchestrator(job_id: str, runtime_profile_name: str) -> dict:
    print(
        f"[app] run_orchestrator job_id={job_id} runtime_profile={runtime_profile_name}"
    )
    storage = build_storage()
    engine = _engine_for_profile(runtime_profile_name)

    def spawn_chunks(chunks):
        payloads = [chunk.model_dump(mode="json") for chunk in chunks]
        engine.parse_chunk.spawn_map(payloads)

    snapshot = process_job(
        storage,
        job_id,
        spawn_chunks=spawn_chunks,
    )
    return snapshot.model_dump(mode="json")


@app.function(image=control_plane_image, volumes={str(ARTIFACT_ROOT): artifacts_volume})
def submit_parse_request_remote(
    request_payload: dict,
    runtime_profile_name: str | None = None,
) -> dict:
    print(
        f"[app] submit_parse_request runtime_profile={runtime_profile_name or DEFAULT_RUNTIME_PROFILE} "
        f"mime_type={request_payload.get('mime_type')} mode={request_payload.get('mode')}"
    )
    service = build_service(runtime_profile_name)
    response = service.submit_document_parse(
        SubmitDocumentParseRequest.model_validate(request_payload)
    )
    return response.model_dump(mode="json")


@app.function(image=control_plane_image, volumes={str(ARTIFACT_ROOT): artifacts_volume})
def get_parse_status_remote(job_id: str) -> dict:
    service = build_service()
    response = service.get_document_parse_status(GetDocumentParseStatusRequest(job_id=job_id))
    return response.model_dump(mode="json")


@app.function(image=control_plane_image, volumes={str(ARTIFACT_ROOT): artifacts_volume})
def get_parse_result_remote(
    job_id: str,
    format: str = "markdown",
    include_pages: bool = False,
    include_debug: bool = False,
) -> dict:
    print(
        f"[app] get_parse_result job_id={job_id} format={format} "
        f"include_pages={include_pages} include_debug={include_debug}"
    )
    service = build_service()
    response = service.get_document_parse_result(
        GetDocumentParseResultRequest(
            job_id=job_id,
            format=format,
            include_pages=include_pages,
            include_debug=include_debug,
        )
    )
    return response.model_dump(mode="json")


@app.function(image=control_plane_image, volumes={str(ARTIFACT_ROOT): artifacts_volume})
def store_upload_remote(data: bytes, file_name: str, mime_type: str | None = None) -> str:
    upload_id = uuid4().hex
    build_storage().write_upload(upload_id, data, file_name=file_name, mime_type=mime_type)
    print(
        f"[app] store_upload upload_id={upload_id} file_name={file_name} "
        f"mime_type={mime_type or 'unknown'} bytes={len(data)}"
    )
    return upload_id


@app.function(image=control_plane_image, volumes={str(ARTIFACT_ROOT): artifacts_volume})
def cleanup_jobs_remote() -> list[str]:
    removed = cleanup_expired_jobs(build_storage(), RETENTION_DAYS)
    print(f"[app] cleanup_jobs removed={len(removed)} retention_days={RETENTION_DAYS}")
    return removed


@app.function(
    image=control_plane_image,
    volumes={str(ARTIFACT_ROOT): artifacts_volume},
    schedule=modal.Period(days=1),
)
def scheduled_cleanup() -> list[str]:
    return cleanup_expired_jobs(build_storage(), RETENTION_DAYS)


@app.function(image=control_plane_image, volumes={str(ARTIFACT_ROOT): artifacts_volume})
@modal.asgi_app()
def web():
    service = build_service()
    return build_fastapi_app(service)


def _infer_mime_type(path: str) -> MimeType:
    guessed, _ = mimetypes.guess_type(path)
    if guessed == MimeType.PDF.value:
        return MimeType.PDF
    if guessed == MimeType.PNG.value:
        return MimeType.PNG
    return MimeType.JPEG


def _default_smoke_sample() -> tuple[bytes, MimeType]:
    url = "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"
    with urllib.request.urlopen(url) as response:
        return response.read(), MimeType.PDF


def _format_error_message(message: str, *, limit: int = 320) -> str:
    compact = " ".join(message.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def _print_error_summary(status: dict) -> None:
    errors = status.get("error_summary") or []
    if not errors:
        return
    print("error_summary:")
    for error in errors:
        print(
            f"  page={error['page_id']} code={error['code']} "
            f"retries={error['retry_count']} stage={error.get('stage') or 'unknown'} "
            f"message={_format_error_message(error['message'])}"
        )


def _print_debug_info(result_payload: dict) -> None:
    debug = result_payload.get("debug") or {}
    pages = debug.get("pages") or {}
    if not pages:
        return
    print("debug_artifacts:")
    for page_id, page_info in sorted(pages.items(), key=lambda item: int(item[0])):
        raw_output_path = page_info.get("raw_output_path")
        prompt_path = page_info.get("prompt_path")
        if raw_output_path:
            print(f"  page={page_id} raw_output_path={raw_output_path}")
        if prompt_path:
            print(f"  page={page_id} prompt_path={prompt_path}")


def _modal_volume_paths(job_id: str) -> dict[str, str]:
    result_root = f"/jobs/{job_id}/result"
    return {
        "volume_name": ARTIFACTS_VOLUME_NAME,
        "job_root": f"/jobs/{job_id}",
        "result_root": result_root,
        "document_json": f"{result_root}/document_parse_result.json",
        "document_markdown": f"{result_root}/document.md",
        "document_text": f"{result_root}/document.txt",
    }


def _write_local_result_bundle(
    job_id: str,
    *,
    output_dir: str,
    include_pages: bool = True,
    include_debug: bool = True,
) -> Path:
    output_root = Path(output_dir).expanduser().resolve() / job_id
    output_root.mkdir(parents=True, exist_ok=True)

    json_result = get_parse_result_remote.remote(
        job_id,
        "json",
        include_pages=include_pages,
        include_debug=include_debug,
    )
    markdown_result = get_parse_result_remote.remote(job_id, "markdown")
    text_result = get_parse_result_remote.remote(job_id, "text")

    (output_root / "document_parse_result.json").write_text(
        json.dumps(json_result["result"], indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_root / "document.md").write_text(markdown_result["result"], encoding="utf-8")
    (output_root / "document.txt").write_text(text_result["result"], encoding="utf-8")
    (output_root / "result_envelope.json").write_text(
        json.dumps(json_result, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_root / "artifact_paths.json").write_text(
        json.dumps(_modal_volume_paths(job_id), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return output_root


def _print_artifact_locations(job_id: str, local_output_dir: Path | None = None) -> None:
    volume_paths = _modal_volume_paths(job_id)
    print("result_artifacts:")
    if local_output_dir is not None:
        print(f"  local_output_dir={local_output_dir}")
    print(f"  modal_volume={volume_paths['volume_name']}")
    print(f"  modal_result_root={volume_paths['result_root']}")
    print(f"  modal_document_json={volume_paths['document_json']}")
    print(f"  modal_document_markdown={volume_paths['document_markdown']}")
    print(f"  modal_document_text={volume_paths['document_text']}")


@app.local_entrypoint()
def stage_upload(path: str, mime_type: str | None = None):
    file_path = Path(path)
    data = file_path.read_bytes()
    upload_id = store_upload_remote.remote(
        data,
        file_path.name,
        mime_type or _infer_mime_type(path).value,
    )
    print(upload_id)


@app.local_entrypoint()
def cleanup_now():
    removed = cleanup_jobs_remote.remote()
    print("\n".join(removed))


@app.local_entrypoint()
def download_result(
    job_id: str,
    output_dir: str = "./tmp/job-results",
    include_pages: bool = True,
    include_debug: bool = True,
):
    local_output_dir = _write_local_result_bundle(
        job_id,
        output_dir=output_dir,
        include_pages=include_pages,
        include_debug=include_debug,
    )
    _print_artifact_locations(job_id, local_output_dir)


@app.local_entrypoint()
def smoke_test(
    sample_path: str | None = None,
    mode: str = "balanced",
    runtime_profile_name: str = "dev",
    save_raw_model_output: bool = True,
    save_prompt_text: bool = True,
    output_dir: str = "./tmp/job-results",
):
    if sample_path is None:
        data, mime_type = _default_smoke_sample()
    else:
        file_path = Path(sample_path)
        data = file_path.read_bytes()
        mime_type = _infer_mime_type(sample_path)

    request_payload = {
        "source": {"type": "bytes", "base64": base64.b64encode(data).decode("utf-8")},
        "mime_type": mime_type.value,
        "mode": mode,
        "output_formats": ["markdown", "text", "json"],
        "debug": {
            "persist_page_images": True,
            "save_raw_model_output": save_raw_model_output,
            "save_prompt_text": save_prompt_text,
        },
    }
    submission = submit_parse_request_remote.remote(request_payload, runtime_profile_name)
    job_id = submission["job_id"]
    print(f"job_id={job_id}")

    while True:
        status = get_parse_status_remote.remote(job_id)
        print(
            f"status={status['status']} "
            f"completed={status['pages_completed']} "
            f"failed={status['pages_failed']}"
        )
        if status["status"] in {"completed", "completed_with_errors", "failed"}:
            break
        time.sleep(5)

    _print_error_summary(status)
    local_output_dir = _write_local_result_bundle(
        job_id,
        output_dir=output_dir,
        include_pages=True,
        include_debug=True,
    )
    _print_artifact_locations(job_id, local_output_dir)
    result = get_parse_result_remote.remote(
        job_id,
        "json",
        include_pages=True,
        include_debug=True,
    )
    _print_debug_info(result)
    if status["status"] == "completed":
        print(Path(local_output_dir / "document.md").read_text(encoding="utf-8"))
        return

    raise RuntimeError(
        f"Smoke test finished with terminal status={status['status']} for job_id={job_id}"
    )
