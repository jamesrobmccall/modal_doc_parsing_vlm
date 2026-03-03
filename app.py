from __future__ import annotations

import base64
import mimetypes
import time
import urllib.request
from pathlib import Path
from uuid import uuid4

import modal

from modal_doc_parsing_vlm.cleanup import cleanup_expired_jobs
from modal_doc_parsing_vlm.config import (
    APP_NAME,
    ARTIFACT_ROOT,
    ARTIFACTS_VOLUME_NAME,
    CONTROL_PLANE_DEPENDENCIES,
    CONTROL_PLANE_PYTHON_VERSION,
    DEFAULT_RUNTIME_PROFILE,
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
).uv_pip_install(*CONTROL_PLANE_DEPENDENCIES)

hf_cache_volume = modal.Volume.from_name(HF_CACHE_VOLUME_NAME, create_if_missing=True)
vllm_cache_volume = modal.Volume.from_name(VLLM_CACHE_VOLUME_NAME, create_if_missing=True)
artifacts_volume = modal.Volume.from_name(
    ARTIFACTS_VOLUME_NAME, create_if_missing=True, version=2
)
job_status_dict = modal.Dict.from_name(JOB_STATUS_DICT_NAME, create_if_missing=True)
idempotency_dict = modal.Dict.from_name(IDEMPOTENCY_DICT_NAME, create_if_missing=True)

ProdParserEngine = create_engine_cls(
    app,
    runtime_profile=get_runtime_profile("prod"),
    hf_cache_volume=hf_cache_volume,
    vllm_cache_volume=vllm_cache_volume,
    artifacts_volume=artifacts_volume,
)
DevParserEngine = create_engine_cls(
    app,
    runtime_profile=get_runtime_profile("dev"),
    hf_cache_volume=hf_cache_volume,
    vllm_cache_volume=vllm_cache_volume,
    artifacts_volume=artifacts_volume,
)


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
    if runtime_profile_name == "prod":
        return ProdParserEngine()
    return DevParserEngine()


@app.function(
    image=control_plane_image,
    volumes={str(ARTIFACT_ROOT): artifacts_volume},
    timeout=ORCHESTRATOR_TIMEOUT_SECONDS,
)
def run_orchestrator(job_id: str, runtime_profile_name: str) -> dict:
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
    return upload_id


@app.function(image=control_plane_image, volumes={str(ARTIFACT_ROOT): artifacts_volume})
def cleanup_jobs_remote() -> list[str]:
    return cleanup_expired_jobs(build_storage(), RETENTION_DAYS)


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
def smoke_test(
    sample_path: str | None = None,
    mode: str = "balanced",
    runtime_profile_name: str = "dev",
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

    result = get_parse_result_remote.remote(job_id, "markdown")
    print(result["result"])
