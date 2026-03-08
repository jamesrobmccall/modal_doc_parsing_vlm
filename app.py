from __future__ import annotations

import base64
import inspect
import json
import mimetypes
import os
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

from modal_doc_parsing_vlm.cleanup import cleanup_expired_jobs, fail_stale_jobs
from modal_doc_parsing_vlm.config import (
    APP_NAME,
    ARTIFACT_ROOT,
    ARTIFACTS_VOLUME_NAME,
    CONTROL_PLANE_DEPENDENCIES,
    CONTROL_PLANE_PYTHON_VERSION,
    DEEPGEMM_CACHE_VOLUME_NAME,
    DEFAULT_RUNTIME_PROFILE,
    ENABLED_RUNTIME_PROFILES,
    EXTRACTION_BATCH_MAX_SIZE,
    EXTRACTION_ENGINE_TIMEOUT_SECONDS,
    EXTRACTION_HTTP_TIMEOUT_SECONDS,
    EXTRACTION_PER_PAGE_MAX_TOKENS,
    EXTRACTION_SUGGESTION_MAX_TOKENS,
    USE_DEDICATED_EXTRACTION_BATCH_ENGINE,
    EXTRACTION_WHOLE_DOCUMENT_MAX_TOKENS,
    HF_CACHE_ROOT,
    HF_CACHE_VOLUME_NAME,
    IDEMPOTENCY_DICT_NAME,
    JOB_STATUS_DICT_NAME,
    LOCAL_RESULT_OUTPUT_ROOT,
    MAX_UPLOAD_BYTES,
    ORCHESTRATOR_TIMEOUT_SECONDS,
    PADDLE_CACHE_VOLUME_NAME,
    RETENTION_DAYS,
    STALE_JOB_SWEEP_SECONDS,
    STALE_JOB_TIMEOUT_SECONDS,
    VLLM_CACHE_VOLUME_NAME,
    EXTRACTION_WHOLE_DOCUMENT_MAX_CHARS,
    FALLBACK_PAGES_PER_CHUNK,
    OCR_PAGES_PER_CHUNK,
    EXTRACTION_BATCH_WAIT_MS,
    get_runtime_profile,
)
from modal_doc_parsing_vlm.config import (
    EXTRACTION_MODEL_ID,
    EXTRACTION_MODEL_REVISION,
)
from modal_doc_parsing_vlm.engine import create_engine_cls
from modal_doc_parsing_vlm.engine_extraction import (
    create_extraction_batch_engine_cls,
    create_extraction_engine_cls,
)
from modal_doc_parsing_vlm.engine_ocr import create_ocr_engine_cls
from modal_doc_parsing_vlm.extraction_client import (
    build_entity_extraction_chat_request,
    build_extraction_headers,
    build_extraction_request_fingerprint,
    build_modal_session_id,
    build_suggestion_chat_request,
    build_suggestion_request_fingerprint,
    extract_chat_completion_content,
)
from modal_doc_parsing_vlm.mcp_server import build_fastapi_app
from modal_doc_parsing_vlm.model_cache import ensure_model_cached
from modal_doc_parsing_vlm.orchestrator import (
    DocumentParseService,
    process_job,
    process_refinement_job,
)
from modal_doc_parsing_vlm.storage import FileSystemStorageBackend
from modal_doc_parsing_vlm.types_api import (
    GetDocumentParseResultRequest,
    GetDocumentParseStatusRequest,
    SubmitDocumentParseRequest,
)
from modal_doc_parsing_vlm.types_result import MimeType

app = modal.App(APP_NAME)

FRONTEND_DIST_LOCAL = PROJECT_ROOT / "frontend" / "dist"
FRONTEND_DIST_REMOTE = Path("/frontend/dist")

control_plane_image = modal.Image.debian_slim(
    python_version=CONTROL_PLANE_PYTHON_VERSION
).uv_pip_install(*CONTROL_PLANE_DEPENDENCIES).add_local_python_source("modal_doc_parsing_vlm")
if FRONTEND_DIST_LOCAL.exists():
    control_plane_image = control_plane_image.add_local_dir(
        str(FRONTEND_DIST_LOCAL),
        remote_path=str(FRONTEND_DIST_REMOTE),
    )
cache_seed_image = modal.Image.debian_slim(
    python_version=CONTROL_PLANE_PYTHON_VERSION
).uv_pip_install(
    *CONTROL_PLANE_DEPENDENCIES,
    "huggingface-hub==0.36.0",
).add_local_python_source("modal_doc_parsing_vlm")

hf_cache_volume = modal.Volume.from_name(HF_CACHE_VOLUME_NAME, create_if_missing=True)
vllm_cache_volume = modal.Volume.from_name(VLLM_CACHE_VOLUME_NAME, create_if_missing=True)
deepgemm_cache_volume = modal.Volume.from_name(DEEPGEMM_CACHE_VOLUME_NAME, create_if_missing=True)
paddle_cache_volume = modal.Volume.from_name(PADDLE_CACHE_VOLUME_NAME, create_if_missing=True)
artifacts_volume = modal.Volume.from_name(
    ARTIFACTS_VOLUME_NAME, create_if_missing=True, version=2
)
job_status_dict = modal.Dict.from_name(JOB_STATUS_DICT_NAME, create_if_missing=True)
idempotency_dict = modal.Dict.from_name(IDEMPOTENCY_DICT_NAME, create_if_missing=True)

ProdFallbackEngine = None
DevFallbackEngine = None
# Backward-compatible aliases kept for internal references.
ProdParserEngine = None
DevParserEngine = None
OcrParserEngine = create_ocr_engine_cls(
    app,
    artifacts_volume=artifacts_volume,
    hf_cache_volume=hf_cache_volume,
    paddle_cache_volume=paddle_cache_volume,
    export_module=__name__,
)
_FALLBACK_ENGINE_CLASSES: dict[str, object] = {}

if "prod" in ENABLED_RUNTIME_PROFILES:
    ProdFallbackEngine = create_engine_cls(
        app,
        runtime_profile=get_runtime_profile("prod"),
        hf_cache_volume=hf_cache_volume,
        vllm_cache_volume=vllm_cache_volume,
        artifacts_volume=artifacts_volume,
        export_module=__name__,
    )
    ProdParserEngine = ProdFallbackEngine
    _FALLBACK_ENGINE_CLASSES["prod"] = ProdFallbackEngine

if "dev" in ENABLED_RUNTIME_PROFILES:
    DevFallbackEngine = create_engine_cls(
        app,
        runtime_profile=get_runtime_profile("dev"),
        hf_cache_volume=hf_cache_volume,
        vllm_cache_volume=vllm_cache_volume,
        artifacts_volume=artifacts_volume,
        export_module=__name__,
    )
    DevParserEngine = DevFallbackEngine
    _FALLBACK_ENGINE_CLASSES["dev"] = DevFallbackEngine

ExtractionEngine = create_extraction_engine_cls(
    app,
    hf_cache_volume=hf_cache_volume,
    deepgemm_cache_volume=deepgemm_cache_volume,
    export_module=__name__,
)
ExtractionBatchEngine = create_extraction_batch_engine_cls(
    app,
    hf_cache_volume=hf_cache_volume,
    deepgemm_cache_volume=deepgemm_cache_volume,
    export_module=__name__,
)

_EXTRACTION_BASE_URL: str | None = None


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

    def _suggest_entities_fn(job_id: str) -> dict:
        return suggest_entities_remote.remote(job_id)

    def _schedule_entity_extraction(job_id: str, request_payload: dict) -> None:
        run_entity_extraction.spawn(job_id, request_payload)

    service = DocumentParseService(
        storage=build_storage(),
        runtime_profile=runtime_profile,
        schedule_job=schedule_job,
    )
    service.suggest_entities_fn = _suggest_entities_fn  # type: ignore[attr-defined]
    service.schedule_entity_extraction = _schedule_entity_extraction  # type: ignore[attr-defined]
    return service


def _fallback_engine_for_profile(runtime_profile_name: str):
    try:
        return _FALLBACK_ENGINE_CLASSES[runtime_profile_name]()
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
    ocr_engine = OcrParserEngine()
    fallback_engine = _fallback_engine_for_profile(runtime_profile_name)

    def run_ocr_pages(payloads: list[dict]) -> list[dict]:
        flattened: list[dict] = []
        for chunk_results in ocr_engine.parse_pages.map(payloads):
            flattened.extend(chunk_results)
        return flattened

    def run_fallback_pages(payloads: list[dict]) -> list[dict]:
        flattened: list[dict] = []
        for chunk_results in fallback_engine.parse_pages.map(payloads):
            flattened.extend(chunk_results)
        return flattened

    def schedule_refinement(selected_job_id: str, selected_profile_name: str) -> None:
        run_refinement.spawn(selected_job_id, selected_profile_name)

    snapshot = process_job(
        storage,
        job_id,
        run_ocr_pages=run_ocr_pages,
        run_fallback_pages=run_fallback_pages,
        schedule_refinement=schedule_refinement,
    )
    return snapshot.model_dump(mode="json")


@app.function(
    image=control_plane_image,
    volumes={str(ARTIFACT_ROOT): artifacts_volume},
    timeout=ORCHESTRATOR_TIMEOUT_SECONDS,
)
def run_refinement(job_id: str, runtime_profile_name: str) -> dict:
    print(
        f"[app] run_refinement job_id={job_id} runtime_profile={runtime_profile_name}"
    )
    storage = build_storage()
    fallback_engine = _fallback_engine_for_profile(runtime_profile_name)

    def run_fallback_pages(payloads: list[dict]) -> list[dict]:
        flattened: list[dict] = []
        for chunk_results in fallback_engine.parse_pages.map(payloads):
            flattened.extend(chunk_results)
        return flattened

    snapshot = process_refinement_job(
        storage,
        job_id,
        run_fallback_pages=run_fallback_pages,
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
    result_level: str = "latest",
) -> dict:
    print(
        f"[app] get_parse_result job_id={job_id} format={format} "
        f"include_pages={include_pages} include_debug={include_debug} "
        f"result_level={result_level}"
    )
    service = build_service()
    response = service.get_document_parse_result(
        GetDocumentParseResultRequest(
            job_id=job_id,
            format=format,
            include_pages=include_pages,
            include_debug=include_debug,
            result_level=result_level,
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
def create_text_job_remote(text: str) -> str:
    return build_service().create_text_job(text)


def _coerce_entity(raw_entity: dict) -> dict | None:
    """Strip unknown fields so StrictModel validation doesn't reject LLM output."""
    ENTITY_KEYS = {"entity_name", "description", "fields"}
    FIELD_KEYS = {"name", "field_type", "description", "required", "examples"}
    cleaned = {k: v for k, v in raw_entity.items() if k in ENTITY_KEYS}
    if "fields" in cleaned and isinstance(cleaned["fields"], list):
        cleaned["fields"] = [
            {k: v for k, v in f.items() if k in FIELD_KEYS}
            for f in cleaned["fields"]
            if isinstance(f, dict)
        ]
    return cleaned if "entity_name" in cleaned and "fields" in cleaned else None


def _coerce_first_url(value: object) -> str | None:
    if isinstance(value, str) and value.startswith("http"):
        return value.rstrip("/")
    if isinstance(value, (list, tuple)):
        for item in value:
            candidate = _coerce_first_url(item)
            if candidate is not None:
                return candidate
    if isinstance(value, dict):
        for item in value.values():
            candidate = _coerce_first_url(item)
            if candidate is not None:
                return candidate
    return None


def _resolve_extraction_base_url() -> str | None:
    global _EXTRACTION_BASE_URL

    if _EXTRACTION_BASE_URL is not None:
        return _EXTRACTION_BASE_URL

    configured = os.environ.get("DOC_PARSE_EXTRACTION_BASE_URL", "").strip()
    if configured:
        _EXTRACTION_BASE_URL = configured.rstrip("/")
        return _EXTRACTION_BASE_URL

    try:
        resolved = ExtractionEngine.get_urls()
        if inspect.isawaitable(resolved):
            import asyncio

            resolved = asyncio.run(resolved)
    except Exception as exc:  # noqa: BLE001
        print(f"[app] failed to resolve extraction server url: {exc!r}")
        return None

    base_url = _coerce_first_url(resolved)
    if base_url is not None:
        _EXTRACTION_BASE_URL = base_url
        return _EXTRACTION_BASE_URL
    return None


def _call_extraction_chat_completion(
    payload: dict[str, object],
    *,
    session_id: str,
    base_url: str | None = None,
) -> tuple[dict[str, object], int]:
    import httpx

    headers = build_extraction_headers(session_id)
    base_url = base_url or _resolve_extraction_base_url()
    if base_url is None:
        raise RuntimeError(
            "Extraction server URL is unavailable. Set DOC_PARSE_EXTRACTION_BASE_URL "
            "or ensure the Modal extraction server is deployed."
        )

    started = time.perf_counter()
    last_exc: Exception | None = None
    for attempt in range(1, 5):
        try:
            response = httpx.post(
                f"{base_url}/v1/chat/completions",
                json=payload,
                headers=headers,
                timeout=EXTRACTION_HTTP_TIMEOUT_SECONDS,
            )
            if response.status_code in {502, 503, 504} and attempt < 4:
                time.sleep(float(attempt))
                continue
            if 400 <= response.status_code < 500:
                raise RuntimeError(
                    "Extraction server rejected the request: "
                    f"status={response.status_code} "
                    f"body={_format_error_message(response.text, limit=1200)}"
                )
            response.raise_for_status()
            return response.json(), int((time.perf_counter() - started) * 1000)
        except RuntimeError:
            raise
        except (httpx.HTTPStatusError, httpx.TimeoutException, httpx.TransportError) as exc:
            last_exc = exc
            if attempt < 4:
                time.sleep(float(attempt))
                continue
    raise RuntimeError(f"Extraction HTTP request failed after retries: {last_exc!r}")


def _validate_whole_document_extraction_size(document_markdown: str) -> None:
    char_count = len(document_markdown)
    if char_count <= EXTRACTION_WHOLE_DOCUMENT_MAX_CHARS:
        return
    approx_input_tokens = max(1, char_count // 3)
    raise ValueError(
        "Whole-document extraction is too large for the extraction model context window. "
        f"document_chars={char_count} "
        f"safe_limit_chars={EXTRACTION_WHOLE_DOCUMENT_MAX_CHARS} "
        f"approx_input_tokens={approx_input_tokens}. "
        "Use per-page extraction for parsed documents."
    )


def _wait_for_extraction_server(
    base_url: str,
    *,
    timeout_seconds: int,
) -> None:
    import httpx

    deadline = time.monotonic() + timeout_seconds
    last_exc: Exception | None = None
    while time.monotonic() < deadline:
        try:
            response = httpx.get(
                f"{base_url}/health",
                timeout=min(EXTRACTION_HTTP_TIMEOUT_SECONDS, 10),
            )
            if response.status_code == 200:
                return
            if response.status_code not in {502, 503, 504}:
                response.raise_for_status()
        except (httpx.HTTPError, httpx.TimeoutException) as exc:
            last_exc = exc
        time.sleep(2)
    raise RuntimeError(
        "Extraction server did not become externally ready before timeout. "
        f"last_error={last_exc!r}"
    )


def _ensure_extraction_server_ready(*, timeout_seconds: int) -> str:
    base_url = _resolve_extraction_base_url()
    if base_url is None:
        raise RuntimeError(
            "Extraction server URL is unavailable. Set DOC_PARSE_EXTRACTION_BASE_URL "
            "or ensure the Modal extraction server is deployed."
        )
    _wait_for_extraction_server(base_url, timeout_seconds=timeout_seconds)
    return base_url


def _warm_extraction_server() -> dict[str, object]:
    base_url = _ensure_extraction_server_ready(
        timeout_seconds=EXTRACTION_ENGINE_TIMEOUT_SECONDS,
    )
    payload = {
        "model": EXTRACTION_MODEL_ID,
        "messages": [
            {"role": "system", "content": "Return JSON only."},
            {"role": "user", "content": "Return {\"ok\": true}."},
        ],
        "temperature": 0.0,
        "max_tokens": 24,
        "chat_template_kwargs": {"enable_thinking": False},
        "response_format": {"type": "json_object"},
    }
    _response, elapsed_ms = _call_extraction_chat_completion(
        payload,
        session_id=build_modal_session_id("warmup", scope="warm"),
        base_url=base_url,
    )
    return {
        "status": "ok",
        "model_id": EXTRACTION_MODEL_ID,
        "elapsed_ms": elapsed_ms,
    }


def _set_extraction_status(
    storage,
    *,
    job_id: str,
    status,
    entities_requested: int,
    pages_total: int,
    pages_processed: int = 0,
    requests_total: int = 0,
    requests_completed: int = 0,
    error_message: str | None = None,
) -> None:
    from modal_doc_parsing_vlm.types_extraction import EntityExtractionStatusPayload

    storage.set_extraction_status(
        job_id,
        EntityExtractionStatusPayload(
            job_id=job_id,
            status=status,
            entities_requested=entities_requested,
            pages_processed=pages_processed,
            pages_total=pages_total,
            requests_total=requests_total,
            requests_completed=requests_completed,
            error_message=error_message,
        ),
    )


def _run_per_page_extraction_via_online_server(
    *,
    job_id: str,
    request,
    page_tasks: list[tuple[int, str]],
    storage,
) -> tuple[list[object], int]:
    from modal_doc_parsing_vlm.types_extraction import (
        ExtractedEntity,
        ExtractionStatus,
        entity_definition_to_json_schema,
    )

    base_url = _ensure_extraction_server_ready(
        timeout_seconds=EXTRACTION_ENGINE_TIMEOUT_SECONDS,
    )
    remaining_requests_by_page = {
        page_id: len(request.entities)
        for page_id, _page_markdown in page_tasks
    }
    requests_total = len(request.entities) * len(page_tasks)
    requests_completed = 0
    pages_processed = 0
    all_extracted: list[ExtractedEntity] = []
    total_inference_ms = 0

    for entity in request.entities:
        json_schema = entity_definition_to_json_schema(entity)
        session_id = build_modal_session_id(
            job_id,
            scope="extract-page",
            entity_name=entity.entity_name,
        )
        for page_id, page_markdown in page_tasks:
            payload = build_entity_extraction_chat_request(
                entity=entity,
                page_text=page_markdown,
                model_id=EXTRACTION_MODEL_ID,
                json_schema=json_schema,
                max_tokens=EXTRACTION_PER_PAGE_MAX_TOKENS,
            )
            raw_response, inference_ms = _call_extraction_chat_completion(
                payload,
                session_id=session_id,
                base_url=base_url,
            )
            all_extracted.append(
                ExtractedEntity(
                    entity_name=entity.entity_name,
                    page_id=page_id,
                    data=json.loads(extract_chat_completion_content(raw_response)),
                )
            )
            total_inference_ms += inference_ms
            requests_completed += 1
            remaining_requests_by_page[page_id] -= 1
            if remaining_requests_by_page[page_id] == 0:
                pages_processed += 1
            _set_extraction_status(
                storage,
                job_id=job_id,
                status=ExtractionStatus.EXTRACTING,
                entities_requested=len(request.entities),
                pages_processed=pages_processed,
                pages_total=len(page_tasks),
                requests_total=requests_total,
                requests_completed=requests_completed,
            )

    return all_extracted, total_inference_ms


@app.function(
    image=control_plane_image,
    volumes={str(ARTIFACT_ROOT): artifacts_volume},
)
def suggest_entities_remote(job_id: str) -> dict:
    from modal_doc_parsing_vlm.types_extraction import (
        EntityDefinition,
        EntitySuggestionResponse,
    )

    print(f"[app] suggest_entities job_id={job_id}")
    storage = build_storage()
    storage.reload()
    result = storage.read_final_result(job_id)
    markdown = result.derived.document_markdown
    page_count = max(len(result.document.pages), 1)
    fingerprint = build_suggestion_request_fingerprint(
        job_id=job_id,
        result_revision=result.metadata.result_revision,
        model_id=EXTRACTION_MODEL_ID,
    )
    cached = storage.read_cached_extraction_suggestion(job_id, fingerprint)
    if cached is not None:
        storage.write_extraction_suggestion(job_id, cached)
        return cached.model_dump(mode="json")

    base_url = _ensure_extraction_server_ready(
        timeout_seconds=EXTRACTION_ENGINE_TIMEOUT_SECONDS,
    )
    payload = build_suggestion_chat_request(
        document_markdown=markdown,
        page_count=page_count,
        model_id=EXTRACTION_MODEL_ID,
        max_tokens=EXTRACTION_SUGGESTION_MAX_TOKENS,
    )
    raw_response, _elapsed_ms = _call_extraction_chat_completion(
        payload,
        session_id=build_modal_session_id(job_id, scope="suggest"),
        base_url=base_url,
    )
    data = json.loads(extract_chat_completion_content(raw_response))

    entities: list[EntityDefinition] = []
    for raw_entity in data.get("entities", []):
        cleaned = _coerce_entity(raw_entity) if isinstance(raw_entity, dict) else None
        if cleaned is None:
            print(f"[app] suggest: skipping malformed entity: {raw_entity!r}")
            continue
        try:
            entities.append(EntityDefinition.model_validate(cleaned))
        except Exception as exc:
            print(f"[app] suggest: entity validation failed: {exc!r} raw={raw_entity!r}")
            continue

    suggestion = EntitySuggestionResponse(
        job_id=job_id,
        suggested_entities=entities,
        document_summary=data.get("document_summary", ""),
    )
    with storage.batch():
        storage.write_extraction_suggestion(job_id, suggestion)
        storage.write_cached_extraction_suggestion(job_id, fingerprint, suggestion)
    return suggestion.model_dump(mode="json")


@app.function(
    image=control_plane_image,
    volumes={str(ARTIFACT_ROOT): artifacts_volume},
    timeout=ORCHESTRATOR_TIMEOUT_SECONDS,
)
def run_entity_extraction(job_id: str, request_payload: dict) -> dict:
    from modal_doc_parsing_vlm.types_extraction import (
        EntityExtractionRequest,
        EntityExtractionResult,
        ExtractedEntity,
        ExtractionMode,
        ExtractionStatus,
        ExtractionWorkItem,
        ExtractionWorkResult,
        entity_definition_to_json_schema,
    )

    print(f"[app] run_entity_extraction job_id={job_id}")
    storage = build_storage()
    storage.reload()
    request = EntityExtractionRequest.model_validate(request_payload)

    result_doc = storage.read_final_result(job_id)
    page_markdowns = result_doc.derived.page_markdown or {}
    document_markdown = result_doc.derived.document_markdown
    page_tasks = [
        (page.id, page_markdowns.get(str(page.id), ""))
        for page in result_doc.document.pages
        if page_markdowns.get(str(page.id), "").strip()
    ]
    pages_total = len(page_tasks) if request.extraction_mode == ExtractionMode.PER_PAGE else 1
    requests_total = (
        len(request.entities) * len(page_tasks)
        if request.extraction_mode == ExtractionMode.PER_PAGE
        else len(request.entities)
    )
    fingerprint = build_extraction_request_fingerprint(
        job_id=job_id,
        result_revision=result_doc.metadata.result_revision,
        request_payload=request.model_dump(mode="json"),
        model_id=EXTRACTION_MODEL_ID,
    )
    cached_result = storage.read_cached_extraction_result(job_id, fingerprint)
    if cached_result is not None:
        with storage.batch():
            storage.write_extraction_result(job_id, cached_result)
            _set_extraction_status(
                storage,
                job_id=job_id,
                status=ExtractionStatus.COMPLETED,
                entities_requested=len(request.entities),
                pages_processed=pages_total,
                pages_total=pages_total,
                requests_total=requests_total,
                requests_completed=requests_total,
            )
        return cached_result.model_dump(mode="json")

    _set_extraction_status(
        storage,
        job_id=job_id,
        status=ExtractionStatus.EXTRACTING,
        entities_requested=len(request.entities),
        pages_total=pages_total,
        requests_total=requests_total,
        requests_completed=0,
    )

    requests_completed = 0
    pages_processed = 0
    try:
        all_extracted: list[ExtractedEntity] = []
        total_inference_ms = 0

        if request.extraction_mode == ExtractionMode.WHOLE_DOCUMENT:
            _validate_whole_document_extraction_size(document_markdown)
            base_url = _ensure_extraction_server_ready(
                timeout_seconds=EXTRACTION_ENGINE_TIMEOUT_SECONDS,
            )
            for entity in request.entities:
                json_schema = entity_definition_to_json_schema(entity)
                session_id = build_modal_session_id(
                    job_id,
                    scope="extract",
                    entity_name=entity.entity_name,
                )
                payload = build_entity_extraction_chat_request(
                    entity=entity,
                    page_text=document_markdown,
                    model_id=EXTRACTION_MODEL_ID,
                    json_schema=json_schema,
                    max_tokens=EXTRACTION_WHOLE_DOCUMENT_MAX_TOKENS,
                )
                raw_response, inference_ms = _call_extraction_chat_completion(
                    payload,
                    session_id=session_id,
                    base_url=base_url,
                )
                all_extracted.append(
                    ExtractedEntity(
                        entity_name=entity.entity_name,
                        page_id=None,
                        data=json.loads(extract_chat_completion_content(raw_response)),
                    )
                )
                total_inference_ms += inference_ms
                requests_completed += 1
                pages_processed = 1 if requests_completed == requests_total else 0
                _set_extraction_status(
                    storage,
                    job_id=job_id,
                    status=ExtractionStatus.EXTRACTING,
                    entities_requested=len(request.entities),
                    pages_processed=pages_processed,
                    pages_total=pages_total,
                    requests_total=requests_total,
                    requests_completed=requests_completed,
                )
        else:
            if USE_DEDICATED_EXTRACTION_BATCH_ENGINE:
                print(
                    f"[app] run_entity_extraction using dedicated batch engine "
                    f"job_id={job_id} requests_total={requests_total}"
                )
                batch_engine = ExtractionBatchEngine()
                work_items: list[ExtractionWorkItem] = []
                remaining_requests_by_page = {
                    page_id: len(request.entities)
                    for page_id, _page_markdown in page_tasks
                }
                for entity in request.entities:
                    json_schema = entity_definition_to_json_schema(entity)
                    session_id = build_modal_session_id(
                        job_id,
                        scope="extract-batch",
                        entity_name=entity.entity_name,
                    )
                    for page_id, page_markdown in page_tasks:
                        work_items.append(
                            ExtractionWorkItem(
                                job_id=job_id,
                                entity=entity,
                                page_id=page_id,
                                page_text=page_markdown,
                                json_schema=json_schema,
                                model_id=EXTRACTION_MODEL_ID,
                                max_tokens=EXTRACTION_PER_PAGE_MAX_TOKENS,
                                session_id=session_id,
                            )
                        )

                if work_items:
                    for result_payload in batch_engine.extract_pages.map(
                        [item.model_dump(mode="json") for item in work_items]
                    ):
                        result = ExtractionWorkResult.model_validate(result_payload)
                        all_extracted.append(
                            ExtractedEntity(
                                entity_name=result.entity_name,
                                page_id=result.page_id,
                                data=result.data,
                            )
                        )
                        total_inference_ms += result.inference_ms
                        requests_completed += 1
                        remaining_requests_by_page[result.page_id] -= 1
                        if remaining_requests_by_page[result.page_id] == 0:
                            pages_processed += 1
                        _set_extraction_status(
                            storage,
                            job_id=job_id,
                            status=ExtractionStatus.EXTRACTING,
                            entities_requested=len(request.entities),
                            pages_processed=pages_processed,
                            pages_total=pages_total,
                            requests_total=requests_total,
                            requests_completed=requests_completed,
                        )
            else:
                print(
                    f"[app] run_entity_extraction using online extraction server "
                    f"job_id={job_id} requests_total={requests_total}"
                )
                all_extracted, total_inference_ms = _run_per_page_extraction_via_online_server(
                    job_id=job_id,
                    request=request,
                    page_tasks=page_tasks,
                    storage=storage,
                )

        extraction_result = EntityExtractionResult(
            job_id=job_id,
            entities=all_extracted,
            schema_used=request.entities,
            extraction_mode=request.extraction_mode,
            model_id=EXTRACTION_MODEL_ID,
            inference_ms=total_inference_ms,
        )
        with storage.batch():
            storage.write_extraction_result(job_id, extraction_result)
            storage.write_cached_extraction_result(job_id, fingerprint, extraction_result)
            _set_extraction_status(
                storage,
                job_id=job_id,
                status=ExtractionStatus.COMPLETED,
                entities_requested=len(request.entities),
                pages_processed=pages_total,
                pages_total=pages_total,
                requests_total=requests_total,
                requests_completed=requests_total,
            )
        print(
            f"[app] run_entity_extraction complete job_id={job_id} "
            f"entities_extracted={len(all_extracted)} inference_ms={total_inference_ms}"
        )
        return extraction_result.model_dump(mode="json")
    except Exception as exc:  # noqa: BLE001
        error_message = _format_error_message(str(exc), limit=1200)
        _set_extraction_status(
            storage,
            job_id=job_id,
            status=ExtractionStatus.FAILED,
            entities_requested=len(request.entities),
            pages_processed=pages_processed,
            pages_total=pages_total,
            requests_total=requests_total,
            requests_completed=requests_completed,
            error_message=error_message,
        )
        print(
            f"[app] run_entity_extraction failed job_id={job_id} "
            f"error={error_message}"
        )
        return {
            "job_id": job_id,
            "status": ExtractionStatus.FAILED.value,
            "error_message": error_message,
        }


@app.function(image=control_plane_image, volumes={str(ARTIFACT_ROOT): artifacts_volume})
def cleanup_jobs_remote() -> list[str]:
    removed = cleanup_expired_jobs(build_storage(), RETENTION_DAYS)
    print(f"[app] cleanup_jobs removed={len(removed)} retention_days={RETENTION_DAYS}")
    return removed


@app.function(image=control_plane_image, volumes={str(ARTIFACT_ROOT): artifacts_volume})
def cleanup_stale_jobs_remote() -> list[str]:
    marked = fail_stale_jobs(build_storage(), STALE_JOB_TIMEOUT_SECONDS)
    print(
        f"[app] cleanup_stale_jobs marked={len(marked)} "
        f"stale_after_seconds={STALE_JOB_TIMEOUT_SECONDS}"
    )
    return marked


def _cache_model_specs(runtime_profile_name: str | None = None) -> list[tuple[str, str | None]]:
    selected_profiles = (
        [get_runtime_profile(runtime_profile_name or DEFAULT_RUNTIME_PROFILE)]
        if runtime_profile_name is not None
        else [get_runtime_profile(profile_name) for profile_name in ENABLED_RUNTIME_PROFILES]
    )
    model_specs: dict[str, str | None] = {EXTRACTION_MODEL_ID: EXTRACTION_MODEL_REVISION}
    for profile in selected_profiles:
        model_specs[profile.model_id] = profile.model_revision
        if profile.fallback_model_id:
            model_specs.setdefault(profile.fallback_model_id, None)
        if profile.deep_refine_model_id:
            model_specs.setdefault(profile.deep_refine_model_id, None)
    return sorted(model_specs.items())


@app.function(
    image=cache_seed_image,
    volumes={str(HF_CACHE_ROOT): hf_cache_volume},
    timeout=60 * 30,
)
def cache_model_weights_remote(
    runtime_profile_name: str | None = None,
) -> dict[str, object]:
    model_statuses = []
    for model_id, revision in _cache_model_specs(runtime_profile_name):
        status = ensure_model_cached(model_id, revision=revision)
        model_statuses.append(
            {
                "model_id": model_id,
                "revision": revision,
                "cache_root": str(HF_CACHE_ROOT),
                "model_root": str(status.model_root),
                "snapshot_count": status.snapshot_count,
                "blob_count": status.blob_count,
                "cache_populated": status.is_populated,
            }
        )
    hf_cache_volume.commit()
    extraction_warmup = _warm_extraction_server()
    ocr_warmup = OcrParserEngine().warmup.remote()
    result = {
        "runtime_profile": runtime_profile_name or "all_enabled_profiles",
        "models": model_statuses,
        "extraction_warmup": extraction_warmup,
        "ocr_warmup": ocr_warmup,
    }
    print(
        f"[app] cache_model_weights runtime_profile={runtime_profile_name or 'all'} "
        f"models={','.join(item['model_id'] for item in model_statuses)}"
    )
    return result


@app.function(
    image=control_plane_image,
    volumes={str(ARTIFACT_ROOT): artifacts_volume},
    schedule=modal.Period(days=1),
)
def scheduled_cleanup() -> list[str]:
    return cleanup_expired_jobs(build_storage(), RETENTION_DAYS)


@app.function(
    image=control_plane_image,
    volumes={str(ARTIFACT_ROOT): artifacts_volume},
    schedule=modal.Period(seconds=STALE_JOB_SWEEP_SECONDS),
)
def scheduled_stale_job_watchdog() -> list[str]:
    return fail_stale_jobs(build_storage(), STALE_JOB_TIMEOUT_SECONDS)


@app.function(image=control_plane_image, volumes={str(ARTIFACT_ROOT): artifacts_volume})
@modal.asgi_app()
def web():
    service = build_service()
    return build_fastapi_app(
        service,
        frontend_dist=FRONTEND_DIST_REMOTE,
        max_upload_bytes=MAX_UPLOAD_BYTES,
    )


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
        "document_json_latest": f"{result_root}/document_parse_result.json",
        "document_json_fast": f"{result_root}/document_parse_result.fast.json",
        "document_json_final": f"{result_root}/document_parse_result.final.json",
        "document_markdown_latest": f"{result_root}/document.md",
        "document_markdown_fast": f"{result_root}/document.fast.md",
        "document_markdown_final": f"{result_root}/document.final.md",
        "document_text_latest": f"{result_root}/document.txt",
        "document_text_fast": f"{result_root}/document.fast.txt",
        "document_text_final": f"{result_root}/document.final.txt",
    }


def _write_local_result_bundle(
    job_id: str,
    *,
    output_dir: str,
    include_pages: bool = True,
    include_debug: bool = True,
    result_level: str = "latest",
) -> Path:
    output_root = Path(output_dir).expanduser().resolve() / job_id
    output_root.mkdir(parents=True, exist_ok=True)

    json_result = get_parse_result_remote.remote(
        job_id,
        "json",
        include_pages=include_pages,
        include_debug=include_debug,
        result_level=result_level,
    )
    markdown_result = get_parse_result_remote.remote(
        job_id,
        "markdown",
        result_level=result_level,
    )
    text_result = get_parse_result_remote.remote(
        job_id,
        "text",
        result_level=result_level,
    )

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
    print(f"  modal_document_json_latest={volume_paths['document_json_latest']}")
    print(f"  modal_document_json_fast={volume_paths['document_json_fast']}")
    print(f"  modal_document_json_final={volume_paths['document_json_final']}")
    print(f"  modal_document_markdown_latest={volume_paths['document_markdown_latest']}")
    print(f"  modal_document_markdown_fast={volume_paths['document_markdown_fast']}")
    print(f"  modal_document_markdown_final={volume_paths['document_markdown_final']}")
    print(f"  modal_document_text_latest={volume_paths['document_text_latest']}")
    print(f"  modal_document_text_fast={volume_paths['document_text_fast']}")
    print(f"  modal_document_text_final={volume_paths['document_text_final']}")


def _build_request_payload(
    *,
    data: bytes,
    mime_type: MimeType,
    mode: str,
    latency_profile: str,
    language_hint: str | None,
    result_level: str,
    save_raw_model_output: bool,
    save_prompt_text: bool,
) -> dict[str, object]:
    return {
        "source": {"type": "bytes", "base64": base64.b64encode(data).decode("utf-8")},
        "mime_type": mime_type.value,
        "mode": mode,
        "latency_profile": latency_profile,
        "language_hint": language_hint,
        "output_formats": ["markdown", "text", "json"],
        "debug": {
            "persist_page_images": True,
            "save_raw_model_output": save_raw_model_output,
            "save_prompt_text": save_prompt_text,
        },
        "result_level": result_level,
    }


def _load_sample(sample_path: str | None) -> tuple[bytes, MimeType]:
    if sample_path is None:
        return _default_smoke_sample()
    file_path = Path(sample_path)
    return file_path.read_bytes(), _infer_mime_type(sample_path)


def _submit_document_job(
    *,
    sample_path: str | None,
    mode: str,
    latency_profile: str,
    language_hint: str | None,
    runtime_profile_name: str,
    result_level: str,
    save_raw_model_output: bool = False,
    save_prompt_text: bool = False,
) -> str:
    data, mime_type = _load_sample(sample_path)
    request_payload = _build_request_payload(
        data=data,
        mime_type=mime_type,
        mode=mode,
        latency_profile=latency_profile,
        language_hint=language_hint,
        result_level=result_level,
        save_raw_model_output=save_raw_model_output,
        save_prompt_text=save_prompt_text,
    )
    submission = submit_parse_request_remote.remote(request_payload, runtime_profile_name)
    return submission["job_id"]


def _wait_for_parse_job(
    job_id: str,
    *,
    require_final: bool,
    poll_interval_seconds: float = 5.0,
) -> tuple[dict[str, object], int, int | None]:
    started = time.perf_counter()
    first_fast_ms: int | None = None
    while True:
        status = get_parse_status_remote.remote(job_id)
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        if status["status"] == "completed_fast" and first_fast_ms is None:
            first_fast_ms = elapsed_ms
        if status["status"] in {
            "completed_fast",
            "completed_final",
            "completed",
            "completed_with_errors",
            "failed",
        }:
            if not require_final or status["status"] != "completed_fast":
                return status, elapsed_ms, first_fast_ms
        time.sleep(poll_interval_seconds)


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
def cleanup_stale_now():
    marked = cleanup_stale_jobs_remote.remote()
    print("\n".join(marked))


@app.local_entrypoint()
def cache_model_weights(runtime_profile_name: str = ""):
    selected_runtime_profile = runtime_profile_name or None
    result = cache_model_weights_remote.remote(selected_runtime_profile)
    print(json.dumps(result, indent=2, sort_keys=True))


@app.local_entrypoint()
def smoke_entity_extraction(
    text: str = "Invoice Number: INV-001\nInvoice Date: 2026-03-05\nTotal Amount: 42.50",
    extraction_mode: str = "whole_document",
):
    job_id = create_text_job_remote.remote(text)
    print(f"job_id={job_id}")
    suggestion = suggest_entities_remote.remote(job_id)
    suggested_entities = suggestion.get("suggested_entities") or []
    if not suggested_entities:
        suggested_entities = [
            {
                "entity_name": "Invoice",
                "description": "Invoice header fields",
                "fields": [
                    {
                        "name": "invoice_number",
                        "field_type": "string",
                        "description": "Invoice identifier",
                        "required": True,
                        "examples": [],
                    },
                    {
                        "name": "total_amount",
                        "field_type": "number",
                        "description": "Total amount due",
                        "required": False,
                        "examples": [],
                    },
                ],
            }
        ]
    result = run_entity_extraction.remote(
        job_id,
        {
            "job_id": job_id,
            "entities": suggested_entities,
            "extraction_mode": extraction_mode,
        },
    )
    print(json.dumps({"suggestion": suggestion, "result": result}, indent=2, sort_keys=True))


@app.local_entrypoint()
def benchmark_ocr_fast(
    sample_path: str = str(PROJECT_ROOT / "tmp" / "benchmark-5pages.pdf"),
    mode: str = "balanced",
    runtime_profile_name: str = "dev",
):
    job_id = _submit_document_job(
        sample_path=sample_path,
        mode=mode,
        latency_profile="fast",
        language_hint=None,
        runtime_profile_name=runtime_profile_name,
        result_level="latest",
    )
    status, wall_ms, _fast_ms = _wait_for_parse_job(job_id, require_final=False)
    pages_total = max(int(status.get("pages_total") or 0), 1)
    print(
        json.dumps(
            {
                "benchmark": "ocr_fast",
                "job_id": job_id,
                "status": status["status"],
                "pages_total": pages_total,
                "wall_ms": wall_ms,
                "pages_per_second": round(pages_total / max(wall_ms / 1000, 0.001), 3),
                "ocr_chunk_size": OCR_PAGES_PER_CHUNK.get(mode, OCR_PAGES_PER_CHUNK["balanced"]),
            },
            indent=2,
            sort_keys=True,
        )
    )


@app.local_entrypoint()
def benchmark_fallback_refinement(
    sample_path: str = str(PROJECT_ROOT / "tmp" / "benchmark-5pages.pdf"),
    mode: str = "balanced",
    runtime_profile_name: str = "dev",
):
    job_id = _submit_document_job(
        sample_path=sample_path,
        mode=mode,
        latency_profile="balanced",
        language_hint=None,
        runtime_profile_name=runtime_profile_name,
        result_level="latest",
    )
    status, wall_ms, fast_stage_ms = _wait_for_parse_job(job_id, require_final=True)
    pages_total = max(int(status.get("pages_total") or 0), 1)
    refinement_wall_ms = (
        wall_ms - fast_stage_ms
        if fast_stage_ms is not None and wall_ms >= fast_stage_ms
        else None
    )
    print(
        json.dumps(
            {
                "benchmark": "fallback_refinement",
                "job_id": job_id,
                "status": status["status"],
                "pages_total": pages_total,
                "wall_ms": wall_ms,
                "fast_stage_wall_ms": fast_stage_ms,
                "refinement_wall_ms": refinement_wall_ms,
                "pages_per_second": round(pages_total / max(wall_ms / 1000, 0.001), 3),
                "fallback_chunk_size": FALLBACK_PAGES_PER_CHUNK.get(
                    mode,
                    FALLBACK_PAGES_PER_CHUNK["balanced"],
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )


@app.local_entrypoint()
def benchmark_per_page_extraction(
    sample_path: str = str(PROJECT_ROOT / "tmp" / "benchmark-5pages.pdf"),
    mode: str = "balanced",
    runtime_profile_name: str = "dev",
):
    job_id = _submit_document_job(
        sample_path=sample_path,
        mode=mode,
        latency_profile="balanced",
        language_hint=None,
        runtime_profile_name=runtime_profile_name,
        result_level="latest",
    )
    status, _wall_ms, _fast_ms = _wait_for_parse_job(job_id, require_final=True)
    pages_total = int(status.get("pages_total") or 0)
    entities = [
        {
            "entity_name": "PageFacts",
            "description": "Basic document facts visible on a page.",
            "fields": [
                {
                    "name": "primary_identifier",
                    "field_type": "string",
                    "description": "Main identifying label or number on the page.",
                    "required": False,
                    "examples": [],
                }
            ],
        },
        {
            "entity_name": "Amounts",
            "description": "Amounts or totals visible on a page.",
            "fields": [
                {
                    "name": "amount_text",
                    "field_type": "string",
                    "description": "Raw amount text found on the page.",
                    "required": False,
                    "examples": [],
                }
            ],
        },
    ]
    started = time.perf_counter()
    result = run_entity_extraction.remote(
        job_id,
        {
            "job_id": job_id,
            "entities": entities,
            "extraction_mode": "per_page",
        },
    )
    wall_ms = int((time.perf_counter() - started) * 1000)
    requests_total = pages_total * len(entities)
    print(
        json.dumps(
            {
                "benchmark": "per_page_extraction",
                "job_id": job_id,
                "pages_total": pages_total,
                "entities_requested": len(entities),
                "requests_total": requests_total,
                "wall_ms": wall_ms,
                "requests_per_second": round(requests_total / max(wall_ms / 1000, 0.001), 3),
                "batch_max_size": EXTRACTION_BATCH_MAX_SIZE,
                "batch_wait_ms": EXTRACTION_BATCH_WAIT_MS,
                "reported_inference_ms": result.get("inference_ms"),
            },
            indent=2,
            sort_keys=True,
        )
    )


@app.local_entrypoint()
def download_result(
    job_id: str,
    output_dir: str = str(LOCAL_RESULT_OUTPUT_ROOT),
    include_pages: bool = True,
    include_debug: bool = True,
    result_level: str = "latest",
):
    local_output_dir = _write_local_result_bundle(
        job_id,
        output_dir=output_dir,
        include_pages=include_pages,
        include_debug=include_debug,
        result_level=result_level,
    )
    _print_artifact_locations(job_id, local_output_dir)


@app.local_entrypoint()
def smoke_test(
    sample_path: str | None = None,
    mode: str = "balanced",
    latency_profile: str = "balanced",
    language_hint: str | None = None,
    runtime_profile_name: str = "dev",
    save_raw_model_output: bool = True,
    save_prompt_text: bool = True,
    output_dir: str = str(LOCAL_RESULT_OUTPUT_ROOT),
    result_level: str = "latest",
):
    job_id = _submit_document_job(
        sample_path=sample_path,
        mode=mode,
        latency_profile=latency_profile,
        language_hint=language_hint,
        runtime_profile_name=runtime_profile_name,
        result_level=result_level,
        save_raw_model_output=save_raw_model_output,
        save_prompt_text=save_prompt_text,
    )
    print(f"job_id={job_id}")
    require_final = result_level != "fast"
    status, _wall_ms, _fast_ms = _wait_for_parse_job(job_id, require_final=require_final)
    print(
        f"status={status['status']} "
        f"completed={status['pages_completed']} "
        f"failed={status['pages_failed']}"
    )

    _print_error_summary(status)
    local_output_dir = _write_local_result_bundle(
        job_id,
        output_dir=output_dir,
        include_pages=True,
        include_debug=True,
        result_level=result_level,
    )
    _print_artifact_locations(job_id, local_output_dir)
    result = get_parse_result_remote.remote(
        job_id,
        "json",
        include_pages=True,
        include_debug=True,
        result_level=result_level,
    )
    _print_debug_info(result)
    if status["status"] in {"completed_fast", "completed_final", "completed"}:
        print(Path(local_output_dir / "document.md").read_text(encoding="utf-8"))
        return

    raise RuntimeError(
        f"Smoke test finished with terminal status={status['status']} for job_id={job_id}"
    )
