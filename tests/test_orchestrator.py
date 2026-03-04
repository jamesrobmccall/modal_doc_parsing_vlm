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
)


def make_two_page_pdf() -> bytes:
    document = fitz.open()
    page_one = document.new_page()
    page_one.insert_text((72, 72), "Page 1")
    page_two = document.new_page()
    page_two.insert_text((72, 72), "Page 2")
    return document.tobytes()


class FakeParser:
    def __init__(self, storage, *, failing_pages: set[int] | None = None) -> None:
        self.storage = storage
        self.failing_pages = failing_pages or set()

    def parse_chunk(self, chunk):
        for task in chunk.pages:
            if task.page_id in self.failing_pages:
                self.storage.write_page_result(
                    PageParseResult(
                        job_id=chunk.job_id,
                        chunk_id=chunk.chunk_id,
                        page_id=task.page_id,
                        status=PageResultStatus.FAILED,
                        error=PageError(
                            page_id=task.page_id,
                            code="synthetic_failure",
                            message="parser failed",
                        ),
                    )
                )
                continue

            self.storage.write_page_result(
                PageParseResult(
                    job_id=chunk.job_id,
                    chunk_id=chunk.chunk_id,
                    page_id=task.page_id,
                    status=PageResultStatus.COMPLETED,
                    page_markdown=f"# Page {task.page_id + 1}",
                    elements=[
                        DocumentElement(
                            id=f"p{task.page_id}-e1",
                            page_id=task.page_id,
                            type="heading",
                            content=f"Page {task.page_id + 1}",
                            bbox=BoundingBox(coord=[0, 0, 10, 10], page_id=task.page_id),
                            order=1,
                        )
                    ],
                    attempts=1,
                    valid_on_first_pass=True,
                )
            )


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
            {"name": "dev", "model_id": "Qwen/Qwen2.5-VL-7B-Instruct"},
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


def test_process_job_supports_partial_completion(tmp_path):
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

    snapshot = process_job(
        storage,
        submission.job_id,
        page_parser=FakeParser(storage, failing_pages={1}),
        poll_interval_seconds=0.01,
    )

    assert snapshot.status == "completed_with_errors"
    assert snapshot.pages_completed == 1
    assert snapshot.pages_failed == 1

    result = service.get_document_parse_result(
        GetDocumentParseResultRequest(job_id=submission.job_id, format="json")
    )
    assert result.status == "completed_with_errors"
    assert result.result["error_status"][0]["page_id"] == 1


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
