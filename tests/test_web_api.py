from __future__ import annotations

from types import SimpleNamespace

from fastapi.testclient import TestClient

from modal_doc_parsing_vlm.mcp_server import build_fastapi_app
from modal_doc_parsing_vlm.types_api import (
    GetDocumentParseResultResponse,
    GetDocumentParseStatusResponse,
    SubmitDocumentParseRequest,
    SubmitDocumentParseResponse,
)
from modal_doc_parsing_vlm.types_extraction import (
    EntityExtractionResult,
    EntityExtractionStatusPayload,
    EntitySuggestionResponse,
    ExtractionStatus,
)
from modal_doc_parsing_vlm.types_result import JobStatus, MimeType


class FakeStorage:
    def __init__(self):
        self._source_bytes = b"%PDF-1.7\n"
        self._extraction_status = None
        self._extraction_result = EntityExtractionResult(
            job_id="job-1",
            entities=[],
            schema_used=[],
            extraction_mode="whole_document",
            model_id="Qwen/Qwen3-4B-Thinking-2507-FP8",
            inference_ms=12,
        )

    def read_job_manifest(self, job_id: str):
        if job_id != "job-1":
            raise FileNotFoundError(job_id)
        return SimpleNamespace(
            file_metadata=SimpleNamespace(
                file_name="sample.pdf",
                mime_type=MimeType.PDF,
            )
        )

    def read_source_bytes(self, job_id: str) -> bytes:
        if job_id != "job-1":
            raise FileNotFoundError(job_id)
        return self._source_bytes

    def reload(self):
        return None

    def get_extraction_status(self, job_id: str):
        if job_id != "job-1":
            raise FileNotFoundError(job_id)
        return self._extraction_status

    def read_extraction_result(self, job_id: str):
        if job_id != "job-1":
            raise FileNotFoundError(job_id)
        return self._extraction_result


class FakeService:
    def __init__(self):
        self.storage = FakeStorage()
        self.last_submit_request: SubmitDocumentParseRequest | None = None
        self.last_suggest_job_id: str | None = None
        self.last_extraction_payload: dict | None = None
        self.raise_not_ready = False
        self.raise_missing_result = False
        self.suggest_entities_fn = self._suggest_entities
        self.schedule_entity_extraction = self._schedule_entity_extraction

    def submit_document_parse(self, request: SubmitDocumentParseRequest):
        self.last_submit_request = request
        return SubmitDocumentParseResponse(job_id="job-1")

    def get_document_parse_status(self, request):
        return GetDocumentParseStatusResponse(
            status=JobStatus.RUNNING,
            pages_total=2,
            pages_completed=1,
            pages_running=1,
            pages_failed=0,
            progress_percent=50.0,
            timings={
                "split_ms": 10,
                "submit_ms": 20,
                "aggregate_ms": 0,
                "elapsed_ms": 100,
            },
            result_revision=1,
            pending_refinement_pages=1,
        )

    def get_document_parse_result(self, request):
        if self.raise_not_ready:
            raise ValueError("Job job-1 is not terminal yet")
        if self.raise_missing_result:
            raise FileNotFoundError("No latest result available for job_id: job-1")
        return GetDocumentParseResultResponse.model_validate(
            {
                "job_id": "job-1",
                "status": "completed_fast",
                "format": request.format.value,
                "metadata": {
                    "job_id": "job-1",
                    "schema_version": "1.0",
                    "pipeline_mode": "balanced",
                    "quality_stage": "fast",
                    "result_revision": 1,
                    "models": {
                        "page_vlm": "Qwen/Qwen3-VL-8B-Instruct-FP8",
                        "fast_ocr": "PP-StructureV3",
                    },
                    "file_metadata": {
                        "file_name": "sample.pdf",
                        "mime_type": "application/pdf",
                        "pages_total": 1,
                        "bytes": 10,
                    },
                    "timings": {
                        "split_ms": 1,
                        "submit_ms": 2,
                        "aggregate_ms": 3,
                        "elapsed_ms": 4,
                    },
                },
                "result": "# ok" if request.format.value != "json" else {"ok": True},
            }
        )

    def _suggest_entities(self, job_id: str):
        self.last_suggest_job_id = job_id
        return EntitySuggestionResponse(
            job_id=job_id,
            suggested_entities=[],
            document_summary="summary",
        ).model_dump(mode="json")

    def _schedule_entity_extraction(self, job_id: str, request_payload: dict):
        self.last_extraction_payload = {"job_id": job_id, **request_payload}


def test_post_jobs_accepts_multipart_file_and_returns_202():
    service = FakeService()
    uploads: list[tuple[bytes, str, str | None]] = []

    def store_upload(data: bytes, file_name: str, mime_type: str | None = None) -> str:
        uploads.append((data, file_name, mime_type))
        return "upload-1"

    client = TestClient(
        build_fastapi_app(
            service,
            store_upload=store_upload,
            max_upload_bytes=1024,
        )
    )

    response = client.post(
        "/api/jobs",
        files={"file": ("sample.pdf", b"%PDF", "application/pdf")},
        data={"mode": "balanced", "latency_profile": "balanced", "result_level": "latest"},
    )

    assert response.status_code == 202
    body = response.json()
    assert body["job_id"] == "job-1"
    assert body["source_preview_url"] == "/api/jobs/job-1/source"
    assert uploads[0][1] == "sample.pdf"
    assert uploads[0][2] == "application/pdf"
    assert service.last_submit_request is not None
    assert service.last_submit_request.source.type == "upload_ref"


def test_post_jobs_rejects_invalid_mime_type():
    service = FakeService()
    client = TestClient(
        build_fastapi_app(
            service,
            store_upload=lambda *_args, **_kwargs: "upload-1",
            max_upload_bytes=1024,
        )
    )

    response = client.post(
        "/api/jobs",
        files={"file": ("bad.txt", b"hello", "text/plain")},
    )

    assert response.status_code == 400
    assert "Unsupported MIME type" in response.json()["detail"]


def test_get_job_status_returns_progress_payload():
    client = TestClient(
        build_fastapi_app(
            FakeService(),
            store_upload=lambda *_args, **_kwargs: "upload-1",
        )
    )

    response = client.get("/api/jobs/job-1/status")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "running"
    assert payload["pages_completed"] == 1
    assert payload["pending_refinement_pages"] == 1


def test_get_job_result_returns_409_when_not_ready():
    service = FakeService()
    service.raise_not_ready = True
    client = TestClient(
        build_fastapi_app(
            service,
            store_upload=lambda *_args, **_kwargs: "upload-1",
        )
    )

    response = client.get("/api/jobs/job-1/result?format=markdown&result_level=latest")

    assert response.status_code == 409
    payload = response.json()
    assert payload["error"] == "result_not_ready"
    assert payload["status"]["status"] == "running"


def test_get_job_result_returns_409_when_artifact_not_visible_yet():
    service = FakeService()
    service.raise_missing_result = True
    client = TestClient(
        build_fastapi_app(
            service,
            store_upload=lambda *_args, **_kwargs: "upload-1",
        )
    )

    response = client.get("/api/jobs/job-1/result?format=markdown&result_level=latest")

    assert response.status_code == 409
    payload = response.json()
    assert payload["error"] == "result_not_ready"
    assert "No latest result available" in payload["message"]


def test_get_job_source_streams_original_document():
    client = TestClient(
        build_fastapi_app(
            FakeService(),
            store_upload=lambda *_args, **_kwargs: "upload-1",
        )
    )

    response = client.get("/api/jobs/job-1/source")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/pdf")
    assert response.headers["content-disposition"].startswith("inline")
    assert response.content.startswith(b"%PDF")


def test_suggest_entities_ignores_legacy_model_backend():
    service = FakeService()
    client = TestClient(
        build_fastapi_app(
            service,
            store_upload=lambda *_args, **_kwargs: "upload-1",
        )
    )

    response = client.post(
        "/api/jobs/job-1/entities/suggest",
        json={"model_backend": "glm_hosted"},
    )

    assert response.status_code == 200
    assert response.json()["job_id"] == "job-1"
    assert service.last_suggest_job_id == "job-1"


def test_extract_entities_ignores_legacy_model_backend():
    service = FakeService()
    client = TestClient(
        build_fastapi_app(
            service,
            store_upload=lambda *_args, **_kwargs: "upload-1",
        )
    )

    response = client.post(
        "/api/jobs/job-1/entities/extract",
        json={
            "job_id": "job-1",
            "model_backend": "glm_hosted",
            "extraction_mode": "whole_document",
            "entities": [
                {
                    "entity_name": "Invoice",
                    "description": "",
                    "fields": [
                        {
                            "name": "invoice_number",
                            "field_type": "string",
                            "description": "",
                            "required": True,
                            "examples": [],
                        }
                    ],
                }
            ],
        },
    )

    assert response.status_code == 202
    assert service.last_extraction_payload is not None
    assert "model_backend" not in service.last_extraction_payload


def test_get_extraction_result_returns_failed_status_message():
    service = FakeService()
    service.storage._extraction_status = EntityExtractionStatusPayload(
        job_id="job-1",
        status=ExtractionStatus.FAILED,
        entities_requested=1,
        pages_total=1,
        error_message="Whole-document extraction is too large. Use per-page extraction.",
    )
    client = TestClient(
        build_fastapi_app(
            service,
            store_upload=lambda *_args, **_kwargs: "upload-1",
        )
    )

    response = client.get("/api/jobs/job-1/entities/result")

    assert response.status_code == 422
    payload = response.json()
    assert payload["error"] == "extraction_failed"
    assert "Use per-page extraction" in payload["message"]
