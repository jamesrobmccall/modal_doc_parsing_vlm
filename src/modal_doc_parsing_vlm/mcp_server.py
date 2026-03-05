from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from .config import HEALTH_RESPONSE, MAX_UPLOAD_BYTES
from .types_api import (
    GetDocumentParseResultRequest,
    GetDocumentParseStatusRequest,
    SubmitDocumentParseRequest,
)
from .web_api import build_web_api_router


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
        result_level: str = "latest",
        latency_profile: str = "balanced",
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
                "result_level": result_level,
                "latency_profile": latency_profile,
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
        result_level: str = "latest",
    ):
        """Fetch the final document parse result in JSON, Markdown, or plain text."""
        request = GetDocumentParseResultRequest(
            job_id=job_id,
            format=format,
            include_pages=include_pages,
            include_debug=include_debug,
            result_level=result_level,
        )
        response = service.get_document_parse_result(request)
        return response.model_dump(mode="json")

    return mcp


def build_fastapi_app(
    service,
    *,
    store_upload=None,
    frontend_dist: str | Path | None = None,
    max_upload_bytes: int = MAX_UPLOAD_BYTES,
) -> FastAPI:
    mcp = make_mcp_server(service)
    mcp_app = mcp.http_app(transport="streamable-http", stateless_http=True)

    app = FastAPI(lifespan=mcp_app.router.lifespan_context)

    if store_upload is None:
        def store_upload(data: bytes, file_name: str, mime_type: str | None = None) -> str:
            upload_id = uuid4().hex
            service.storage.write_upload(
                upload_id,
                data,
                file_name=file_name,
                mime_type=mime_type,
            )
            return upload_id

    app.include_router(
        build_web_api_router(
            service,
            store_upload=store_upload,
            max_upload_bytes=max_upload_bytes,
        )
    )

    @app.get("/healthz")
    async def healthz():
        return HEALTH_RESPONSE

    app.mount("/mcp", mcp_app, "mcp")

    dist_path = Path(frontend_dist) if frontend_dist is not None else None
    if dist_path is None:
        return app
    if not dist_path.exists():
        @app.get("/", include_in_schema=False)
        async def frontend_not_built():
            return HTMLResponse(
                (
                    "<!doctype html><html><body>"
                    "<h1>Frontend bundle missing</h1>"
                    "<p>Run scripts/build_frontend.sh before deploying.</p>"
                    "</body></html>"
                ),
                status_code=503,
            )

        return app

    assets_dir = dist_path / "assets"
    if assets_dir.exists():
        app.mount(
            "/assets",
            StaticFiles(directory=str(assets_dir)),
            name="frontend-assets",
        )
    dist_real = dist_path.resolve()

    @app.get("/", include_in_schema=False)
    async def frontend_index():
        return FileResponse(dist_path / "index.html")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def frontend_spa(full_path: str):
        if full_path.startswith(("api/", "mcp", "healthz")):
            raise HTTPException(status_code=404, detail="Not found")
        candidate = (dist_path / full_path).resolve()
        try:
            candidate.relative_to(dist_real)
        except ValueError:
            return FileResponse(dist_path / "index.html")
        if candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(dist_path / "index.html")

    return app
