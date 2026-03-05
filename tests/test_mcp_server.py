import asyncio

from fastapi.testclient import TestClient

from modal_doc_parsing_vlm.mcp_server import build_fastapi_app, make_mcp_server
from modal_doc_parsing_vlm.types_api import (
    GetDocumentParseResultResponse,
    GetDocumentParseStatusResponse,
    SubmitDocumentParseResponse,
)
from modal_doc_parsing_vlm.types_result import JobStatus


class FakeService:
    def submit_document_parse(self, request):
        return SubmitDocumentParseResponse(job_id="job-1")

    def get_document_parse_status(self, request):
        return GetDocumentParseStatusResponse(
            status=JobStatus.QUEUED,
            pages_total=1,
            pages_completed=0,
            pages_running=1,
            pages_failed=0,
            progress_percent=0.0,
            timings={"split_ms": 0, "submit_ms": 0, "aggregate_ms": 0, "elapsed_ms": 0},
        )

    def get_document_parse_result(self, request):
        return GetDocumentParseResultResponse.model_validate(
            {
                "job_id": "job-1",
                "status": "completed",
                "format": "markdown",
                "metadata": {
                    "job_id": "job-1",
                    "schema_version": "1.0",
                    "pipeline_mode": "balanced",
                    "models": {"page_vlm": "Qwen/Qwen2.5-VL-7B-Instruct"},
                    "file_metadata": {
                        "file_name": "doc.png",
                        "mime_type": "image/png",
                        "pages_total": 1,
                        "bytes": 4,
                    },
                    "timings": {
                        "split_ms": 1,
                        "submit_ms": 2,
                        "aggregate_ms": 3,
                        "elapsed_ms": 4,
                    },
                },
                "result": "# ok",
            }
        )


def test_mcp_server_registers_exactly_three_tools():
    mcp = make_mcp_server(FakeService())
    if hasattr(mcp, "list_tools"):
        tools = asyncio.run(mcp.list_tools(run_middleware=False))
        tool_names = sorted(tool.name for tool in tools)
    else:
        tools = asyncio.run(mcp.get_tools())
        tool_names = sorted(tools)
    assert tool_names == [
        "get_document_parse_result",
        "get_document_parse_status",
        "submit_document_parse",
    ]


def test_fastapi_wrapper_exposes_healthz():
    client = TestClient(build_fastapi_app(FakeService()))
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
