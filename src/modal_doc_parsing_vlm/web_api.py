from __future__ import annotations

import mimetypes
from pathlib import Path
from typing import Callable

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile, status
from fastapi.responses import JSONResponse, Response

from .types_api import (
    GetDocumentParseResultRequest,
    GetDocumentParseStatusRequest,
    SubmitDocumentParseRequest,
)
from .types_extraction import EntityExtractionRequest, ExtractionStatus
from .types_result import LatencyProfile, MimeType, OutputFormat, ParseMode, ResultLevel


UPLOAD_MIME_TYPES = {
    MimeType.PDF,
    MimeType.PNG,
    MimeType.JPEG,
}


def _resolve_upload_mime_type(upload: UploadFile) -> MimeType | None:
    declared = (upload.content_type or "").strip().lower()
    for mime_type in UPLOAD_MIME_TYPES:
        if declared == mime_type.value:
            return mime_type

    guessed, _ = mimetypes.guess_type(upload.filename or "")
    if guessed is None:
        return None
    for mime_type in UPLOAD_MIME_TYPES:
        if guessed == mime_type.value:
            return mime_type
    return None


def _normalize_file_name(upload: UploadFile, mime_type: MimeType) -> str:
    if upload.filename:
        return Path(upload.filename).name
    if mime_type == MimeType.PDF:
        return "upload.pdf"
    if mime_type == MimeType.PNG:
        return "upload.png"
    return "upload.jpg"


def _http_404(job_id: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Unknown job_id: {job_id}",
    )


def build_web_api_router(
    service,
    *,
    store_upload: Callable[[bytes, str, str | None], str],
    max_upload_bytes: int,
) -> APIRouter:
    router = APIRouter()

    def _normalize_extraction_body(body: dict | None, *, job_id: str | None = None) -> dict:
        normalized = dict(body or {})
        normalized.pop("model_backend", None)
        if job_id is not None:
            normalized.setdefault("job_id", job_id)
        return normalized

    @router.post("/api/jobs", status_code=status.HTTP_202_ACCEPTED)
    async def create_job(
        file: UploadFile = File(...),
        mode: ParseMode = Form(ParseMode.BALANCED),
        latency_profile: LatencyProfile = Form(LatencyProfile.BALANCED),
        result_level: ResultLevel = Form(ResultLevel.LATEST),
    ):
        data = await file.read()
        if not data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty.",
            )
        if len(data) > max_upload_bytes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Uploaded file exceeds {max_upload_bytes} byte limit.",
            )

        mime_type = _resolve_upload_mime_type(file)
        if mime_type is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    "Unsupported MIME type. Only application/pdf, image/png, "
                    "and image/jpeg are accepted."
                ),
            )

        file_name = _normalize_file_name(file, mime_type)
        upload_id = store_upload(data, file_name, mime_type.value)

        submission = service.submit_document_parse(
            SubmitDocumentParseRequest.model_validate(
                {
                    "source": {"type": "upload_ref", "upload_id": upload_id},
                    "mime_type": mime_type.value,
                    "mode": mode.value,
                    "latency_profile": latency_profile.value,
                    "result_level": result_level.value,
                    "output_formats": ["markdown", "text", "json"],
                }
            )
        )
        return {
            **submission.model_dump(mode="json"),
            "source_preview_url": f"/api/jobs/{submission.job_id}/source",
            "mime_type": mime_type.value,
        }

    @router.get("/api/jobs/{job_id}/status")
    def get_job_status(job_id: str):
        try:
            response = service.get_document_parse_status(
                GetDocumentParseStatusRequest(job_id=job_id)
            )
        except FileNotFoundError as exc:
            raise _http_404(job_id) from exc
        return response.model_dump(mode="json")

    @router.get("/api/jobs/{job_id}/result")
    def get_job_result(
        job_id: str,
        format: OutputFormat = Query(OutputFormat.MARKDOWN),
        include_pages: bool = Query(False),
        include_debug: bool = Query(False),
        result_level: ResultLevel = Query(ResultLevel.LATEST),
    ):
        try:
            status_payload = service.get_document_parse_status(
                GetDocumentParseStatusRequest(job_id=job_id)
            )
        except FileNotFoundError as exc:
            raise _http_404(job_id) from exc

        # Ensure this container sees the latest result artifacts written by workers.
        if hasattr(service.storage, "reload"):
            service.storage.reload()

        request = GetDocumentParseResultRequest(
            job_id=job_id,
            format=format,
            include_pages=include_pages,
            include_debug=include_debug,
            result_level=result_level,
        )
        try:
            response = service.get_document_parse_result(request)
            return response.model_dump(mode="json")
        except ValueError as exc:
            return JSONResponse(
                status_code=status.HTTP_409_CONFLICT,
                content={
                    "error": "result_not_ready",
                    "message": str(exc),
                    "status": status_payload.model_dump(mode="json"),
                },
            )
        except FileNotFoundError as exc:
            # Status exists but artifact file isn't visible yet (or not materialized yet).
            return JSONResponse(
                status_code=status.HTTP_409_CONFLICT,
                content={
                    "error": "result_not_ready",
                    "message": str(exc),
                    "status": status_payload.model_dump(mode="json"),
                },
            )

    @router.get("/api/jobs/{job_id}/source")
    def get_job_source(job_id: str):
        if hasattr(service.storage, "reload"):
            service.storage.reload()
        try:
            manifest = service.storage.read_job_manifest(job_id)
            source_bytes = service.storage.read_source_bytes(job_id)
        except FileNotFoundError as exc:
            raise _http_404(job_id) from exc

        file_name = manifest.file_metadata.file_name.replace('"', "")
        return Response(
            content=source_bytes,
            media_type=manifest.file_metadata.mime_type.value,
            headers={"Content-Disposition": f'inline; filename="{file_name}"'},
        )

    @router.post("/api/text-jobs", status_code=status.HTTP_201_CREATED)
    async def create_text_job(body: dict):
        text = (body.get("text") or "").strip()
        if not text:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="text is required and must be non-empty.",
            )
        if not hasattr(service, "create_text_job"):
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail="Text job creation is not available.",
            )
        job_id = service.create_text_job(text)
        return {"job_id": job_id}

    @router.post("/api/jobs/{job_id}/entities/suggest")
    async def suggest_entities(job_id: str, body: dict | None = None):
        if hasattr(service.storage, "reload"):
            service.storage.reload()
        try:
            service.storage.read_job_manifest(job_id)
        except FileNotFoundError as exc:
            raise _http_404(job_id) from exc

        if not hasattr(service, "suggest_entities_fn") or service.suggest_entities_fn is None:
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail="Entity extraction is not available.",
            )

        _normalize_extraction_body(body, job_id=job_id)
        result = service.suggest_entities_fn(job_id)
        return result

    @router.post("/api/jobs/{job_id}/entities/extract")
    async def extract_entities(job_id: str, body: dict):
        if hasattr(service.storage, "reload"):
            service.storage.reload()
        try:
            service.storage.read_job_manifest(job_id)
        except FileNotFoundError as exc:
            raise _http_404(job_id) from exc

        if (
            not hasattr(service, "schedule_entity_extraction")
            or service.schedule_entity_extraction is None
        ):
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail="Entity extraction is not available.",
            )

        request = EntityExtractionRequest.model_validate(
            _normalize_extraction_body(body, job_id=job_id)
        )
        service.schedule_entity_extraction(job_id, request.model_dump(mode="json"))
        return JSONResponse(
            status_code=status.HTTP_202_ACCEPTED,
            content={
                "job_id": job_id,
                "status": ExtractionStatus.EXTRACTING.value,
                "entities_requested": len(request.entities),
            },
        )

    @router.get("/api/jobs/{job_id}/entities/result")
    def get_extraction_result(job_id: str):
        if hasattr(service.storage, "reload"):
            service.storage.reload()
        try:
            service.storage.read_job_manifest(job_id)
        except FileNotFoundError as exc:
            raise _http_404(job_id) from exc

        extraction_status = service.storage.get_extraction_status(job_id)
        if extraction_status is not None and extraction_status.status == ExtractionStatus.EXTRACTING:
            return JSONResponse(
                status_code=status.HTTP_409_CONFLICT,
                content={
                    "error": "extraction_in_progress",
                    "message": "Entity extraction is still running.",
                    "status": extraction_status.model_dump(mode="json"),
                },
            )
        if extraction_status is not None and extraction_status.status == ExtractionStatus.FAILED:
            return JSONResponse(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                content={
                    "error": "extraction_failed",
                    "message": extraction_status.error_message
                    or "Entity extraction failed.",
                    "status": extraction_status.model_dump(mode="json"),
                },
            )

        result = service.storage.read_extraction_result(job_id)
        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No extraction result found for job_id: {job_id}",
            )
        return result.model_dump(mode="json")

    return router
