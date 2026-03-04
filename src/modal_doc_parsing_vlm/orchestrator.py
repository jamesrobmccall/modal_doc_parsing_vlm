from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Protocol
from uuid import uuid4

from .aggregator import aggregate_job
from .chunking import build_chunks
from .config import (
    ARTIFACT_ROOT,
    DEFAULT_POLL_AFTER_SECONDS,
    DEFAULT_STATUS_POLL_INTERVAL_SECONDS,
    PARSER_VERSION,
    SCHEMA_VERSION,
    TERMINAL_STATUSES,
)
from .rasterize import rasterize_document
from .source_ingest import compute_request_fingerprint, resolve_source_bytes
from .types_api import (
    GetDocumentParseResultRequest,
    GetDocumentParseResultResponse,
    GetDocumentParseStatusRequest,
    GetDocumentParseStatusResponse,
    SubmitDocumentParseRequest,
    SubmitDocumentParseResponse,
)
from .types_result import (
    DocumentPage,
    FileMetadata,
    IdempotencyRecord,
    JobManifest,
    JobProgressSnapshot,
    JobStatus,
    MimeType,
    OutputFormat,
    PageChunk,
    PageError,
    PageParseResult,
    PageResultStatus,
    PageTask,
    ResultEnvelope,
)


class PageParserBackend(Protocol):
    def parse_chunk(self, chunk: PageChunk): ...


def _elapsed_ms(start: float) -> int:
    return int((time.perf_counter() - start) * 1000)


def _snapshot(
    manifest: JobManifest,
    *,
    status: JobStatus,
    pages_total: int = 0,
    pages_completed: int = 0,
    pages_running: int = 0,
    pages_failed: int = 0,
    split_ms: int = 0,
    submit_ms: int = 0,
    aggregate_ms: int = 0,
    error_summary: list[PageError] | None = None,
) -> JobProgressSnapshot:
    terminal_pages = pages_completed + pages_failed
    progress_percent = (
        round((terminal_pages / pages_total) * 100, 2) if pages_total else 0.0
    )
    return JobProgressSnapshot(
        job_id=manifest.job_id,
        status=status,
        pages_total=pages_total,
        pages_completed=pages_completed,
        pages_running=pages_running,
        pages_failed=pages_failed,
        progress_percent=progress_percent,
        timings={
            "split_ms": split_ms,
            "submit_ms": submit_ms,
            "aggregate_ms": aggregate_ms,
            "elapsed_ms": int((time.time() - manifest.created_at.timestamp()) * 1000),
        },
        error_summary=error_summary,
        runtime_profile=manifest.runtime_profile,
        parser_version=manifest.parser_version,
    )


def _refresh_status_from_results(storage, manifest: JobManifest) -> JobProgressSnapshot:
    results = storage.list_page_results(manifest.job_id)
    completed = [
        result for result in results if result.status == PageResultStatus.COMPLETED
    ]
    failed = [
        result for result in results if result.status == PageResultStatus.FAILED
    ]
    existing = storage.get_status(manifest.job_id)
    split_ms = existing.timings.split_ms if existing is not None else 0
    submit_ms = existing.timings.submit_ms if existing is not None else 0
    aggregate_ms = existing.timings.aggregate_ms if existing is not None else 0
    return _snapshot(
        manifest,
        status=JobStatus.RUNNING,
        pages_total=len(manifest.pages),
        pages_completed=len(completed),
        pages_running=max(len(manifest.pages) - len(results), 0),
        pages_failed=len(failed),
        split_ms=split_ms,
        submit_ms=submit_ms,
        aggregate_ms=aggregate_ms,
        error_summary=[result.error for result in failed if result.error is not None] or None,
    )


def _build_page_tasks(manifest: JobManifest, rasterized_pages, debug) -> list[PageTask]:
    tasks: list[PageTask] = []
    for page in rasterized_pages:
        page_dir = ARTIFACT_ROOT / "jobs" / manifest.job_id / "pages" / str(page.page_id)
        tasks.append(
            PageTask(
                job_id=manifest.job_id,
                chunk_id="pending",
                page_id=page.page_id,
                mode=manifest.pipeline_mode,
                image_path=str(page.image_path),
                width=page.width,
                height=page.height,
                rotation=page.rotation,
                page_hash=page.page_hash,
                task_path=str(page_dir / "task.json"),
                result_path=str(page_dir / "result.json"),
                raw_output_path=str(page_dir / "raw_output.txt")
                if debug.save_raw_model_output
                else None,
                prompt_path=str(page_dir / "prompt.txt")
                if debug.save_prompt_text
                else None,
            )
        )
    return tasks


def process_job(
    storage,
    job_id: str,
    *,
    page_parser: PageParserBackend | None = None,
    spawn_chunks: Callable[[list[PageChunk]], Any] | None = None,
    poll_interval_seconds: float = DEFAULT_STATUS_POLL_INTERVAL_SECONDS,
):
    manifest = storage.read_job_manifest(job_id)
    print(
        f"[orchestrator] start job_id={job_id} runtime_profile={manifest.runtime_profile} "
        f"mode={manifest.pipeline_mode.value}"
    )
    split_started = time.perf_counter()
    storage.set_status(
        job_id,
        _snapshot(manifest, status=JobStatus.SPLITTING, split_ms=0),
    )

    source_bytes = storage.read_source_bytes(job_id)
    page_image_dir = storage.job_dir(job_id) / "pages"
    rasterized_pages = rasterize_document(
        source_bytes=source_bytes,
        mime_type=manifest.file_metadata.mime_type,
        mode=manifest.pipeline_mode,
        output_dir=page_image_dir,
        page_range=manifest.page_range,
        max_pages=manifest.max_pages,
    )
    manifest.pages = [
        DocumentPage(
            id=page.page_id,
            image_uri=f"jobs/{manifest.job_id}/pages/{page.page_id}/page.png",
            width=page.width,
            height=page.height,
            rotation=page.rotation,
            page_hash=page.page_hash,
        )
        for page in rasterized_pages
    ]
    manifest.file_metadata.pages_total = len(manifest.pages)
    page_tasks = _build_page_tasks(manifest, rasterized_pages, manifest.debug)
    chunks = build_chunks(
        job_id=manifest.job_id,
        mode=manifest.pipeline_mode,
        parser_version=manifest.parser_version,
        runtime_profile=manifest.runtime_profile,
        artifact_root=str(ARTIFACT_ROOT),
        model_id=manifest.model_id,
        language_hint=manifest.language_hint,
        debug=manifest.debug,
        page_tasks=page_tasks,
    )
    manifest.chunk_ids = [chunk.chunk_id for chunk in chunks]
    storage.write_job_manifest(manifest)
    for chunk in chunks:
        storage.write_chunk_manifest(chunk)
        for task in chunk.pages:
            storage.write_page_task(task)
    print(
        f"[orchestrator] split complete job_id={job_id} pages={len(manifest.pages)} "
        f"chunks={len(chunks)}"
    )

    split_ms = _elapsed_ms(split_started)
    storage.set_status(
        job_id,
        _snapshot(
            manifest,
            status=JobStatus.SUBMITTING,
            pages_total=len(manifest.pages),
            pages_running=len(manifest.pages),
            split_ms=split_ms,
        ),
    )

    submit_started = time.perf_counter()
    if spawn_chunks is not None:
        print(f"[orchestrator] submitting remote chunks job_id={job_id}")
        spawn_chunks(chunks)
    elif page_parser is not None:
        print(f"[orchestrator] parsing inline chunks job_id={job_id}")
        for chunk in chunks:
            page_parser.parse_chunk(chunk)
    else:
        raise ValueError("Either page_parser or spawn_chunks must be provided")

    submit_ms = _elapsed_ms(submit_started)
    storage.set_status(
        job_id,
        _snapshot(
            manifest,
            status=JobStatus.RUNNING,
            pages_total=len(manifest.pages),
            pages_running=len(manifest.pages),
            split_ms=split_ms,
            submit_ms=submit_ms,
        ),
    )

    while True:
        storage.reload()
        snapshot = _refresh_status_from_results(storage, manifest)
        snapshot.timings.split_ms = split_ms
        snapshot.timings.submit_ms = submit_ms
        storage.set_status(job_id, snapshot)
        print(
            f"[orchestrator] poll job_id={job_id} status={snapshot.status.value} "
            f"completed={snapshot.pages_completed} failed={snapshot.pages_failed} "
            f"running={snapshot.pages_running}"
        )
        if snapshot.pages_completed + snapshot.pages_failed >= len(manifest.pages):
            break
        time.sleep(poll_interval_seconds)

    aggregate_started = time.perf_counter()
    snapshot.status = JobStatus.AGGREGATING
    storage.set_status(job_id, snapshot)
    aggregate_job(storage, job_id)
    aggregate_ms = _elapsed_ms(aggregate_started)

    final_snapshot = _refresh_status_from_results(storage, manifest)
    final_snapshot.timings.split_ms = split_ms
    final_snapshot.timings.submit_ms = submit_ms
    final_snapshot.timings.aggregate_ms = aggregate_ms
    if final_snapshot.pages_completed == 0 and final_snapshot.pages_failed > 0:
        final_snapshot.status = JobStatus.FAILED
    elif final_snapshot.pages_failed > 0:
        final_snapshot.status = JobStatus.COMPLETED_WITH_ERRORS
    else:
        final_snapshot.status = JobStatus.COMPLETED
    storage.set_status(job_id, final_snapshot)
    print(
        f"[orchestrator] done job_id={job_id} status={final_snapshot.status.value} "
        f"completed={final_snapshot.pages_completed} failed={final_snapshot.pages_failed}"
    )
    return final_snapshot


class DocumentParseService:
    def __init__(
        self,
        *,
        storage,
        runtime_profile,
        schedule_job: Callable[[str, str], None],
    ) -> None:
        self.storage = storage
        self.runtime_profile = runtime_profile
        self.schedule_job = schedule_job

    def submit_document_parse(
        self, request: SubmitDocumentParseRequest
    ) -> SubmitDocumentParseResponse:
        resolved = resolve_source_bytes(request, self.storage)
        fingerprint = compute_request_fingerprint(request, resolved.content_hash)
        existing = self.storage.lookup_idempotency(fingerprint)
        if existing is not None and existing.parser_version == PARSER_VERSION:
            snapshot = self.storage.get_status(existing.job_id)
            if snapshot is not None:
                return SubmitDocumentParseResponse(
                    job_id=existing.job_id,
                    pages_total=snapshot.pages_total or (
                        1 if request.mime_type != MimeType.PDF else None
                    ),
                    poll_after_seconds=DEFAULT_POLL_AFTER_SECONDS,
                )

        job_id = uuid4().hex
        manifest = JobManifest(
            job_id=job_id,
            parser_version=PARSER_VERSION,
            schema_version=SCHEMA_VERSION,
            runtime_profile=self.runtime_profile.name,
            source_fingerprint=fingerprint,
            request_payload=request.model_dump(mode="json"),
            output_formats=request.output_formats,
            debug=request.debug,
            model_id=self.runtime_profile.model_id,
            pipeline_mode=request.mode,
            file_metadata=FileMetadata(
                file_name=resolved.file_name,
                mime_type=request.mime_type,
                pages_total=1 if request.mime_type != request.mime_type.PDF else 0,
                bytes=len(resolved.data),
            ),
            page_range=request.page_range,
            max_pages=request.max_pages,
            language_hint=request.language_hint,
        )
        self.storage.create_job_manifest(manifest)
        self.storage.write_source_bytes(job_id, resolved.data)
        self.storage.store_idempotency(
            IdempotencyRecord(
                request_fingerprint=fingerprint,
                job_id=job_id,
                parser_version=PARSER_VERSION,
            )
        )
        self.storage.set_status(
            job_id,
            _snapshot(
                manifest,
                status=JobStatus.QUEUED,
                pages_total=manifest.file_metadata.pages_total,
            ),
        )
        print(
            f"[service] queued job_id={job_id} runtime_profile={self.runtime_profile.name} "
            f"mime_type={request.mime_type.value} bytes={len(resolved.data)}"
        )
        self.schedule_job(job_id, self.runtime_profile.name)
        return SubmitDocumentParseResponse(
            job_id=job_id,
            pages_total=manifest.file_metadata.pages_total or None,
            poll_after_seconds=DEFAULT_POLL_AFTER_SECONDS,
        )

    def get_document_parse_status(
        self, request: GetDocumentParseStatusRequest
    ) -> GetDocumentParseStatusResponse:
        snapshot = self.storage.get_status(request.job_id)
        if snapshot is None:
            raise FileNotFoundError(f"Unknown job_id: {request.job_id}")
        return GetDocumentParseStatusResponse(
            status=snapshot.status,
            pages_total=snapshot.pages_total,
            pages_completed=snapshot.pages_completed,
            pages_running=snapshot.pages_running,
            pages_failed=snapshot.pages_failed,
            progress_percent=snapshot.progress_percent,
            timings=snapshot.timings,
            error_summary=snapshot.error_summary,
        )

    def get_document_parse_result(
        self, request: GetDocumentParseResultRequest
    ) -> GetDocumentParseResultResponse:
        snapshot = self.storage.get_status(request.job_id)
        if snapshot is None:
            raise FileNotFoundError(f"Unknown job_id: {request.job_id}")
        if snapshot.status.value not in TERMINAL_STATUSES:
            raise ValueError(
                f"Job {request.job_id} is not terminal yet; current status is {snapshot.status.value}"
            )
        result = self.storage.read_final_result(request.job_id)
        envelope: ResultEnvelope
        if request.format == OutputFormat.JSON:
            envelope = ResultEnvelope(
                job_id=request.job_id,
                status=snapshot.status,
                format=request.format,
                metadata=result.metadata,
                result=result.model_dump(mode="json"),
                pages=self.storage.list_page_results(request.job_id)
                if request.include_pages
                else None,
                debug=self.storage.collect_debug_info(request.job_id)
                if request.include_debug
                else None,
            )
        else:
            envelope = ResultEnvelope(
                job_id=request.job_id,
                status=snapshot.status,
                format=request.format,
                metadata=result.metadata,
                result=self.storage.read_result_text(request.job_id, request.format.value),
                pages=self.storage.list_page_results(request.job_id)
                if request.include_pages
                else None,
                debug=self.storage.collect_debug_info(request.job_id)
                if request.include_debug
                else None,
            )
        return GetDocumentParseResultResponse.model_validate(
            envelope.model_dump(mode="json")
        )
