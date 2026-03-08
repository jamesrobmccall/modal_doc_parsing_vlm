import base64

import fitz

from modal_doc_parsing_vlm.orchestrator import DocumentParseService, process_job
from modal_doc_parsing_vlm.storage import FileSystemStorageBackend, InMemoryKVStore
from modal_doc_parsing_vlm.types_api import GetDocumentParseResultRequest, SubmitDocumentParseRequest
from modal_doc_parsing_vlm.types_result import (
    BoundingBox,
    DocumentElement,
    PageError,
    PageParseResult,
    PageResultStatus,
    ParseEngine,
)


def make_two_page_pdf() -> bytes:
    document = fitz.open()
    page_one = document.new_page()
    page_one.insert_text((72, 72), "Page 1")
    page_two = document.new_page()
    page_two.insert_text((72, 72), "Page 2")
    return document.tobytes()


def _completed_result(task_payload: dict, *, engine: str = "paddle_ocr") -> dict:
    task = task_payload
    return PageParseResult(
        job_id=task["job_id"],
        chunk_id=task["chunk_id"],
        page_id=task["page_id"],
        status=PageResultStatus.COMPLETED,
        page_markdown=f"# Page {task['page_id'] + 1}",
        elements=[
            DocumentElement(
                id=f"p{task['page_id']}-e1",
                page_id=task["page_id"],
                type="heading",
                content=f"Page {task['page_id'] + 1}",
                bbox=BoundingBox(coord=[0, 0, 10, 10], page_id=task["page_id"]),
                order=1,
                confidence=0.95,
            )
        ],
        attempts=1,
        valid_on_first_pass=True,
        result_revision=task.get("result_revision", 1),
        engine=engine,
        confidence_summary={
            "mean_ocr_confidence": 0.95,
            "text_coverage_ratio": 0.95,
            "table_confidence": 1.0,
        },
    ).model_dump(mode="json")


def _failed_result(task_payload: dict) -> dict:
    task = task_payload
    return PageParseResult(
        job_id=task["job_id"],
        chunk_id=task["chunk_id"],
        page_id=task["page_id"],
        status=PageResultStatus.FAILED,
        error=PageError(
            page_id=task["page_id"],
            code="synthetic_failure",
            message="parser failed",
        ),
        result_revision=task.get("result_revision", 1),
        engine=ParseEngine.PADDLE_OCR,
    ).model_dump(mode="json")


def _iter_task_payloads(payloads: list[dict]) -> list[dict]:
    tasks: list[dict] = []
    for payload in payloads:
        if "pages" in payload:
            tasks.extend(payload["pages"])
        else:
            tasks.append(payload)
    return tasks


def build_service(tmp_path, scheduled_jobs: list[tuple[str, str]]):
    storage = FileSystemStorageBackend(
        tmp_path,
        status_store=InMemoryKVStore(),
        idempotency_store=InMemoryKVStore(),
    )

    def schedule_job(job_id: str, runtime_profile_name: str):
        scheduled_jobs.append((job_id, runtime_profile_name))

    service = DocumentParseService(
        storage=storage,
        runtime_profile=type(
            "RuntimeProfile",
            (),
            {
                "name": "dev",
                "model_id": "Qwen/Qwen2.5-VL-7B-Instruct",
                "fallback_model_id": "Qwen/Qwen2.5-VL-7B-Instruct",
            },
        )(),
        schedule_job=schedule_job,
    )
    return storage, service


def test_submit_is_idempotent(tmp_path):
    scheduled_jobs: list[tuple[str, str]] = []
    storage, service = build_service(tmp_path, scheduled_jobs)
    payload = base64.b64encode(make_two_page_pdf()).decode("utf-8")
    request = SubmitDocumentParseRequest.model_validate(
        {
            "source": {"type": "bytes", "base64": payload},
            "mime_type": "application/pdf",
            "mode": "balanced",
            "output_formats": ["markdown"],
        }
    )

    first = service.submit_document_parse(request)
    second = service.submit_document_parse(request)

    assert first.job_id == second.job_id
    assert len(scheduled_jobs) == 1
    assert storage.read_job_manifest(first.job_id).runtime_profile == "dev"


def test_process_job_emits_fast_and_final_results(tmp_path):
    scheduled_jobs: list[tuple[str, str]] = []
    storage, service = build_service(tmp_path, scheduled_jobs)
    payload = base64.b64encode(make_two_page_pdf()).decode("utf-8")
    request = SubmitDocumentParseRequest.model_validate(
        {
            "source": {"type": "bytes", "base64": payload},
            "mime_type": "application/pdf",
            "mode": "balanced",
            "output_formats": ["markdown"],
        }
    )
    submission = service.submit_document_parse(request)

    def run_ocr_pages(payloads: list[dict]) -> list[dict]:
        # Force OCR on page 1 to fail so fallback gets scheduled.
        responses: list[dict] = []
        for payload in _iter_task_payloads(payloads):
            if payload["page_id"] == 1:
                responses.append(_failed_result(payload))
            else:
                responses.append(_completed_result(payload))
        return responses

    snapshot = process_job(
        storage,
        submission.job_id,
        run_ocr_pages=run_ocr_pages,
        schedule_refinement=None,
        run_fallback_pages=lambda payloads: [
            _completed_result(p, engine="vlm_fallback")
            for p in _iter_task_payloads(payloads)
        ],
    )
    assert snapshot.status == "completed_final"
    assert snapshot.pages_completed == 2
    assert snapshot.pages_failed == 0
    assert snapshot.result_revision == 2
    assert storage.read_job_manifest(submission.job_id).chunk_ids == [
        "ocr-0000",
        "fallback-r2-0000",
    ]

    result = service.get_document_parse_result(
        GetDocumentParseResultRequest(job_id=submission.job_id, format="json")
    )
    assert result.status == "completed_final"
    assert result.metadata.result_revision == 2


def test_result_request_rejects_non_terminal_job(tmp_path):
    scheduled_jobs: list[tuple[str, str]] = []
    _storage, service = build_service(tmp_path, scheduled_jobs)
    request = SubmitDocumentParseRequest.model_validate(
        {
            "source": {"type": "bytes", "base64": base64.b64encode(b"hello").decode()},
            "mime_type": "image/png",
            "mode": "balanced",
            "output_formats": ["markdown"],
        }
    )
    submission = service.submit_document_parse(request)

    try:
        service.get_document_parse_result(
            GetDocumentParseResultRequest(job_id=submission.job_id, format="markdown")
        )
    except ValueError as exc:
        assert "not terminal" in str(exc)
    else:
        raise AssertionError("Expected non-terminal result request to fail")
