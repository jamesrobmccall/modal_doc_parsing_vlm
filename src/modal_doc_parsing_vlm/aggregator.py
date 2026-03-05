from __future__ import annotations

import re

from .config import PADDLE_OCR_ENGINE_NAME
from .types_result import (
    DerivedOutputs,
    DocumentBody,
    DocumentElement,
    DocumentParseResult,
    JobStatus,
    ModelMetadata,
    PageParseResult,
    PageResultStatus,
    QualityStage,
    ResultMetadata,
)


def _plain_text_from_markdown(markdown: str) -> str:
    text = re.sub(r"[#*_`>-]", "", markdown)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _page_text(page_result: PageParseResult) -> str:
    if page_result.page_markdown:
        return _plain_text_from_markdown(page_result.page_markdown)
    return "\n".join(element.content for element in page_result.elements).strip()


def aggregate_job(
    storage,
    job_id: str,
    *,
    quality_stage: QualityStage = QualityStage.FINAL,
    result_revision: int = 1,
) -> DocumentParseResult:
    manifest = storage.read_job_manifest(job_id)
    snapshot = storage.get_status(job_id)
    if snapshot is None:
        raise FileNotFoundError(f"Status missing for job_id: {job_id}")

    page_results = sorted(
        storage.list_page_results(job_id, result_revision=result_revision),
        key=lambda result: result.page_id,
    )
    completed_pages = [
        result for result in page_results if result.status == PageResultStatus.COMPLETED
    ]
    errors = [
        result.error
        for result in page_results
        if result.status == PageResultStatus.FAILED and result.error is not None
    ]

    elements: list[DocumentElement] = []
    page_markdown: dict[str, str] = {}
    document_text_parts: list[str] = []
    for page_result in completed_pages:
        page_result.elements.sort(key=lambda element: (element.page_id, element.order))
        elements.extend(page_result.elements)
        page_markdown[str(page_result.page_id)] = page_result.page_markdown
        page_text = _page_text(page_result)
        if page_text:
            document_text_parts.append(page_text)

    document_markdown = "\n\n".join(
        page_markdown[str(page.id)]
        for page in sorted(manifest.pages, key=lambda item: item.id)
        if page_markdown.get(str(page.id))
    ).strip()
    document_text = "\n\n".join(document_text_parts).strip()

    result = DocumentParseResult(
        document=DocumentBody(pages=manifest.pages, elements=elements),
        derived=DerivedOutputs(
            document_markdown=document_markdown,
            document_text=document_text,
            page_markdown=page_markdown,
        ),
        error_status=errors,
        metadata=ResultMetadata(
            job_id=job_id,
            schema_version=manifest.schema_version,
            pipeline_mode=manifest.pipeline_mode,
            quality_stage=quality_stage,
            result_revision=result_revision,
            models=ModelMetadata(
                page_vlm=manifest.model_id,
                fast_ocr=PADDLE_OCR_ENGINE_NAME,
                fallback_vlm=manifest.fallback_model_id or manifest.model_id,
            ),
            file_metadata=manifest.file_metadata,
            timings=snapshot.timings,
        ),
    )
    storage.write_final_result(
        job_id,
        result,
        document_markdown,
        document_text,
        quality_stage=quality_stage,
        result_revision=result_revision,
    )

    if quality_stage == QualityStage.FAST:
        snapshot.status = JobStatus.COMPLETED_FAST
    elif not completed_pages and errors:
        snapshot.status = JobStatus.FAILED
    elif errors:
        snapshot.status = JobStatus.COMPLETED_WITH_ERRORS
    else:
        snapshot.status = JobStatus.COMPLETED_FINAL
    snapshot.result_revision = result_revision
    storage.set_status(job_id, snapshot)
    return result
