from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from uuid import uuid4

from .aggregator import aggregate_job
from .config import (
    ARTIFACT_ROOT,
    DEFAULT_POLL_AFTER_SECONDS,
    ENABLE_ASYNC_REFINEMENT,
    FAST_PROFILE_RENDER_DPI,
    PARSER_VERSION,
    SCHEMA_VERSION,
    TERMINAL_STATUSES,
)
from .fallback_policy import fallback_reasons
from .page_router import classify_page
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
    BoundingBox,
    DerivedOutputs,
    DocumentBody,
    DocumentElement,
    DocumentPage,
    DocumentParseResult,
    FileMetadata,
    IdempotencyRecord,
    JobManifest,
    JobProgressSnapshot,
    JobStatus,
    JobTimings,
    MimeType,
    ModelMetadata,
    OutputFormat,
    PageChunk,
    PageError,
    PageParseResult,
    PageResultStatus,
    PageTask,
    ParseEngine,
    ParseMode,
    QualityStage,
    ResultEnvelope,
    ResultMetadata,
    LatencyProfile,
)


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
    refine_ms: int = 0,
    error_summary: list[PageError] | None = None,
    result_revision: int = 0,
    pending_refinement_pages: int = 0,
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
            "refine_ms": refine_ms,
            "elapsed_ms": int((time.time() - manifest.created_at.timestamp()) * 1000),
        },
        error_summary=error_summary,
        runtime_profile=manifest.runtime_profile,
        parser_version=manifest.parser_version,
        result_revision=result_revision,
        pending_refinement_pages=pending_refinement_pages,
    )


def _refresh_status_from_results(
    storage,
    manifest: JobManifest,
    *,
    result_revision: int,
    status: JobStatus,
) -> JobProgressSnapshot:
    results = storage.list_page_results(manifest.job_id, result_revision=result_revision)
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
    refine_ms = existing.timings.refine_ms if existing is not None else 0
    return _snapshot(
        manifest,
        status=status,
        pages_total=len(manifest.pages),
        pages_completed=len(completed),
        pages_running=max(len(manifest.pages) - len(results), 0),
        pages_failed=len(failed),
        split_ms=split_ms,
        submit_ms=submit_ms,
        aggregate_ms=aggregate_ms,
        refine_ms=refine_ms,
        error_summary=[result.error for result in failed if result.error is not None] or None,
        result_revision=result_revision,
        pending_refinement_pages=len(manifest.pending_refinement_pages),
    )


def _build_page_tasks(manifest: JobManifest, rasterized_pages, decisions, debug) -> list[PageTask]:
    tasks: list[PageTask] = []
    for page in rasterized_pages:
        decision = decisions[page.page_id]
        page_dir = ARTIFACT_ROOT / "jobs" / manifest.job_id / "pages" / str(page.page_id)
        tasks.append(
            PageTask(
                job_id=manifest.job_id,
                chunk_id="page",
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
            route_metrics={
                "extractable_char_count": float(decision.extractable_char_count),
                "printable_ratio": float(decision.printable_ratio),
                "common_word_ratio": float(decision.common_word_ratio),
            },
            source_text=decision.extracted_text,
            route_engine=decision.engine,
            latency_profile=manifest.latency_profile,
        )
    )
    return tasks


def _digital_page_result(task: PageTask, *, result_revision: int) -> PageParseResult:
    text = (task.source_text or "").strip()
    elements: list[DocumentElement] = []
    if text:
        elements.append(
            DocumentElement(
                id=f"p{task.page_id}-e1",
                page_id=task.page_id,
                type="text",
                content=text,
                bbox=BoundingBox(coord=[0, 0, task.width, task.height], page_id=task.page_id),
                order=1,
                confidence=0.99,
                attributes={"source": "pdf_text_layer"},
            )
        )
    coverage = 1.0 if text else 0.0
    return PageParseResult(
        job_id=task.job_id,
        chunk_id=task.chunk_id,
        page_id=task.page_id,
        status=PageResultStatus.COMPLETED,
        page_markdown=text,
        elements=elements,
        attempts=1,
        valid_on_first_pass=True,
        result_revision=result_revision,
        engine=ParseEngine.DIGITAL_TEXT,
        confidence_summary={
            "mean_ocr_confidence": 0.99 if text else 0.0,
            "text_coverage_ratio": coverage,
            "table_confidence": 1.0,
        },
    )


def _pending_fallback_pages(storage, manifest: JobManifest) -> list[int]:
    fast_results = storage.list_page_results(manifest.job_id, result_revision=1)
    pending: list[int] = []
    for result in fast_results:
        if result.status != PageResultStatus.COMPLETED:
            pending.append(result.page_id)
            continue
        if result.engine == ParseEngine.DIGITAL_TEXT:
            continue
        if fallback_reasons(result):
            pending.append(result.page_id)
    return sorted(set(pending))


def _run_page_map(run_pages: Callable[[list[dict]], list[dict]], tasks: list[PageTask]) -> list[PageParseResult]:
    if not tasks:
        return []
    payloads = [task.model_dump(mode="json") for task in tasks]
    raw_results = run_pages(payloads)
    return [PageParseResult.model_validate(item) for item in raw_results]


def process_refinement_job(
    storage,
    job_id: str,
    *,
    run_fallback_pages: Callable[[list[dict]], list[dict]],
) -> JobProgressSnapshot:
    manifest = storage.read_job_manifest(job_id)
    if not manifest.pending_refinement_pages:
        snapshot = storage.get_status(job_id)
        if snapshot is None:
            raise FileNotFoundError(f"Unknown job_id: {job_id}")
        return snapshot

    refine_started = time.perf_counter()
    next_revision = max(manifest.result_revision + 1, 2)
    fallback_tasks: list[PageTask] = []
    for page_id in manifest.pending_refinement_pages:
        task = storage.read_page_task(job_id, page_id)
        if task is None:
            continue
        fallback_tasks.append(task.model_copy(update={"result_revision": next_revision}))

    try:
        fallback_results = _run_page_map(run_fallback_pages, fallback_tasks)
    except Exception as exc:  # noqa: BLE001
        print(
            f"[orchestrator] refinement failed job_id={job_id} "
            f"error={' '.join(str(exc).split())[:200]}"
        )
        manifest.pending_refinement_pages = []
        storage.write_job_manifest(manifest)
        existing = storage.get_status(job_id)
        if existing is None:
            snapshot = _refresh_status_from_results(
                storage,
                manifest,
                result_revision=manifest.result_revision,
                status=JobStatus.COMPLETED_FAST,
            )
        else:
            snapshot = existing
        errors = list(snapshot.error_summary or [])
        errors.append(
            PageError(
                page_id=-1,
                code="fallback_refinement_failed",
                message=str(exc),
                retry_count=0,
                stage="refinement",
            )
        )
        snapshot.error_summary = errors
        snapshot.pending_refinement_pages = 0
        snapshot.result_revision = manifest.result_revision
        storage.set_status(job_id, snapshot)
        return snapshot

    with storage.batch():
        for result in fallback_results:
            storage.write_page_result(
                result.model_copy(update={"result_revision": next_revision, "fallback_triggered": True})
            )

    aggregate_started = time.perf_counter()
    aggregate_job(
        storage,
        job_id,
        quality_stage=QualityStage.FINAL,
        result_revision=next_revision,
    )
    aggregate_ms = _elapsed_ms(aggregate_started)
    refine_ms = _elapsed_ms(refine_started)

    manifest.pending_refinement_pages = []
    manifest.result_revision = next_revision
    storage.write_job_manifest(manifest)

    snapshot = _refresh_status_from_results(
        storage,
        manifest,
        result_revision=next_revision,
        status=JobStatus.COMPLETED_FINAL,
    )
    snapshot.timings.aggregate_ms = aggregate_ms
    snapshot.timings.refine_ms = refine_ms
    snapshot.pending_refinement_pages = 0
    snapshot.result_revision = next_revision
    storage.set_status(job_id, snapshot)
    return snapshot


def process_job(
    storage,
    job_id: str,
    *,
    run_ocr_pages: Callable[[list[dict]], list[dict]],
    run_fallback_pages: Callable[[list[dict]], list[dict]] | None = None,
    schedule_refinement: Callable[[str, str], None] | None = None,
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
    dpi_override = (
        FAST_PROFILE_RENDER_DPI
        if manifest.latency_profile == LatencyProfile.FAST
        else None
    )
    rasterized_pages = rasterize_document(
        source_bytes=source_bytes,
        mime_type=manifest.file_metadata.mime_type,
        mode=manifest.pipeline_mode,
        output_dir=page_image_dir,
        page_range=manifest.page_range,
        max_pages=manifest.max_pages,
        dpi_override=dpi_override,
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

    decisions = {
        page.page_id: classify_page(
            page_id=page.page_id,
            extracted_text=page.extracted_text,
        )
        for page in rasterized_pages
    }
    page_tasks = _build_page_tasks(manifest, rasterized_pages, decisions, manifest.debug)
    manifest.chunk_ids = [f"page-{task.page_id:04d}" for task in page_tasks]
    with storage.batch():
        storage.write_job_manifest(manifest)
        for task in page_tasks:
            storage.write_page_task(task)
    print(
        f"[orchestrator] split complete job_id={job_id} pages={len(manifest.pages)} "
        f"tasks={len(page_tasks)}"
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
    ocr_tasks: list[PageTask] = []
    ocr_results: list[PageParseResult] = []
    for task in page_tasks:
        if task.route_engine == ParseEngine.DIGITAL_TEXT:
            ocr_results.append(_digital_page_result(task, result_revision=1))
        else:
            ocr_tasks.append(task.model_copy(update={"result_revision": 1}))
    ocr_results.extend(_run_page_map(run_ocr_pages, ocr_tasks))
    with storage.batch():
        for result in ocr_results:
            storage.write_page_result(result.model_copy(update={"result_revision": 1}))
    submit_ms = _elapsed_ms(submit_started)

    aggregate_started = time.perf_counter()
    aggregate_job(
        storage,
        job_id,
        quality_stage=QualityStage.FAST,
        result_revision=1,
    )
    aggregate_ms = _elapsed_ms(aggregate_started)

    pending_refinement_pages = _pending_fallback_pages(storage, manifest)
    if manifest.latency_profile == LatencyProfile.FAST:
        pending_refinement_pages = []
    manifest.pending_refinement_pages = pending_refinement_pages
    manifest.result_revision = 1
    storage.write_job_manifest(manifest)

    status = JobStatus.COMPLETED_FAST
    snapshot = _refresh_status_from_results(
        storage,
        manifest,
        result_revision=1,
        status=status,
    )
    snapshot.timings.split_ms = split_ms
    snapshot.timings.submit_ms = submit_ms
    snapshot.timings.aggregate_ms = aggregate_ms
    snapshot.pending_refinement_pages = len(pending_refinement_pages)
    snapshot.result_revision = 1
    storage.set_status(job_id, snapshot)

    if not pending_refinement_pages:
        aggregate_job(
            storage,
            job_id,
            quality_stage=QualityStage.FINAL,
            result_revision=1,
        )
        snapshot = _refresh_status_from_results(
            storage,
            manifest,
            result_revision=1,
            status=JobStatus.COMPLETED_FINAL,
        )
        snapshot.timings.split_ms = split_ms
        snapshot.timings.submit_ms = submit_ms
        snapshot.timings.aggregate_ms = aggregate_ms
        snapshot.pending_refinement_pages = 0
        snapshot.result_revision = 1
        storage.set_status(job_id, snapshot)
        return snapshot

    if ENABLE_ASYNC_REFINEMENT and schedule_refinement is not None:
        print(
            f"[orchestrator] fast result ready job_id={job_id} "
            f"pending_refinement_pages={len(pending_refinement_pages)}"
        )
        schedule_refinement(job_id, manifest.runtime_profile)
        return snapshot

    if run_fallback_pages is not None:
        return process_refinement_job(
            storage,
            job_id,
            run_fallback_pages=run_fallback_pages,
        )

    return snapshot


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
                        1 if request.mime_type != "application/pdf" else None
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
            fallback_model_id=getattr(
                self.runtime_profile,
                "fallback_model_id",
                self.runtime_profile.model_id,
            ),
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
            latency_profile=request.latency_profile,
            result_level=request.result_level,
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
                result_revision=0,
                pending_refinement_pages=0,
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
            result_revision=snapshot.result_revision,
            pending_refinement_pages=snapshot.pending_refinement_pages,
        )

    def get_document_parse_result(
        self, request: GetDocumentParseResultRequest
    ) -> GetDocumentParseResultResponse:
        snapshot = self.storage.get_status(request.job_id)
        if snapshot is None:
            raise FileNotFoundError(f"Unknown job_id: {request.job_id}")

        if request.result_level.value == "final":
            final_statuses = {
                JobStatus.COMPLETED_FINAL,
                JobStatus.COMPLETED,
                JobStatus.COMPLETED_WITH_ERRORS,
                JobStatus.FAILED,
            }
            if snapshot.status not in final_statuses:
                raise ValueError(
                    f"Job {request.job_id} does not have a final result yet; "
                    f"current status is {snapshot.status.value}"
                )
        elif snapshot.status.value not in TERMINAL_STATUSES:
            raise ValueError(
                f"Job {request.job_id} is not terminal yet; current status is {snapshot.status.value}"
            )

        result = self.storage.read_final_result(
            request.job_id,
            result_level=request.result_level,
        )
        revision = result.metadata.result_revision
        envelope: ResultEnvelope
        if request.format == OutputFormat.JSON:
            envelope = ResultEnvelope(
                job_id=request.job_id,
                status=snapshot.status,
                format=request.format,
                metadata=result.metadata,
                result=result.model_dump(mode="json"),
                pages=self.storage.list_page_results(
                    request.job_id,
                    result_revision=revision,
                )
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
                result=self.storage.read_result_text(
                    request.job_id,
                    request.format.value,
                    result_level=request.result_level,
                ),
                pages=self.storage.list_page_results(
                    request.job_id,
                    result_revision=revision,
                )
                if request.include_pages
                else None,
                debug=self.storage.collect_debug_info(request.job_id)
                if request.include_debug
                else None,
            )
        return GetDocumentParseResultResponse.model_validate(
            envelope.model_dump(mode="json")
        )

    def create_text_job(self, text: str) -> str:
        """Create a minimal job seeded with raw text for entity extraction."""
        import hashlib

        job_id = uuid4().hex
        text_bytes = text.encode("utf-8")
        fingerprint = hashlib.sha256(text_bytes).hexdigest()

        manifest = JobManifest(
            job_id=job_id,
            parser_version=PARSER_VERSION,
            schema_version=SCHEMA_VERSION,
            runtime_profile="text",
            source_fingerprint=fingerprint,
            request_payload={"source": "raw_text"},
            output_formats=[OutputFormat.MARKDOWN, OutputFormat.TEXT],
            model_id="",
            pipeline_mode=ParseMode.BALANCED,
            file_metadata=FileMetadata(
                file_name="text_input.txt",
                mime_type=MimeType.TEXT,
                pages_total=0,
                bytes=len(text_bytes),
            ),
        )
        self.storage.create_job_manifest(manifest)

        result = DocumentParseResult(
            document=DocumentBody(),
            derived=DerivedOutputs(
                document_markdown=text,
                document_text=text,
            ),
            metadata=ResultMetadata(
                job_id=job_id,
                schema_version=SCHEMA_VERSION,
                pipeline_mode=ParseMode.BALANCED,
                quality_stage=QualityStage.FINAL,
                models=ModelMetadata(),
                file_metadata=FileMetadata(
                    file_name="text_input.txt",
                    mime_type=MimeType.TEXT,
                    pages_total=0,
                    bytes=len(text_bytes),
                ),
                timings=JobTimings(),
            ),
        )
        self.storage.write_final_result(
            job_id,
            result,
            text,
            text,
            quality_stage=QualityStage.FINAL,
            result_revision=1,
        )

        self.storage.set_status(
            job_id,
            _snapshot(
                manifest,
                status=JobStatus.COMPLETED_FINAL,
                pages_total=0,
                result_revision=1,
                pending_refinement_pages=0,
            ),
        )
        return job_id
