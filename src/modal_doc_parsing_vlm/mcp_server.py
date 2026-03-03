from __future__ import annotations

from fastapi import FastAPI

from .config import HEALTH_RESPONSE
from .types_api import (
    GetDocumentParseResultRequest,
    GetDocumentParseStatusRequest,
    SubmitDocumentParseRequest,
)


def make_mcp_server(service):
    from fastmcp import FastMCP

    mcp = FastMCP("Document Parse MCP Server")

    @mcp.tool()
    async def submit_document_parse(
        source: dict,
        mime_type: str,
        mode: str = "balanced",
        output_formats: list[str] | None = None,
        max_pages: int | None = None,
        page_range: str | None = None,
        language_hint: str | None = None,
        debug: dict | None = None,
    ):
        """Submit a document parse job and return a job id for polling."""
        request = SubmitDocumentParseRequest.model_validate(
            {
                "source": source,
                "mime_type": mime_type,
                "mode": mode,
                "output_formats": output_formats or ["json"],
                "max_pages": max_pages,
                "page_range": page_range,
                "language_hint": language_hint,
                "debug": debug,
            }
        )
        response = service.submit_document_parse(request)
        return response.model_dump(mode="json")

    @mcp.tool()
    async def get_document_parse_status(job_id: str):
        """Fetch progress information for a previously submitted document parse job."""
        request = GetDocumentParseStatusRequest(job_id=job_id)
        response = service.get_document_parse_status(request)
        return response.model_dump(mode="json")

    @mcp.tool()
    async def get_document_parse_result(
        job_id: str,
        format: str,
        include_pages: bool = False,
        include_debug: bool = False,
    ):
        """Fetch the final document parse result in JSON, Markdown, or plain text."""
        request = GetDocumentParseResultRequest(
            job_id=job_id,
            format=format,
            include_pages=include_pages,
            include_debug=include_debug,
        )
        response = service.get_document_parse_result(request)
        return response.model_dump(mode="json")

    return mcp


def build_fastapi_app(service) -> FastAPI:
    mcp = make_mcp_server(service)
    mcp_app = mcp.http_app(transport="streamable-http", stateless_http=True)

    app = FastAPI(lifespan=mcp_app.router.lifespan_context)

    @app.get("/healthz")
    async def healthz():
        return HEALTH_RESPONSE

    app.mount("/mcp", mcp_app, "mcp")
    return app
