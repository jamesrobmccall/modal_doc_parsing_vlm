from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class MimeType(str, Enum):
    PDF = "application/pdf"
    PNG = "image/png"
    JPEG = "image/jpeg"
    TEXT = "text/plain"


class ParseMode(str, Enum):
    BALANCED = "balanced"
    ACCURATE = "accurate"


class LatencyProfile(str, Enum):
    FAST = "fast"
    BALANCED = "balanced"
    MAX_QUALITY = "max_quality"


class ResultLevel(str, Enum):
    LATEST = "latest"
    FAST = "fast"
    FINAL = "final"


class QualityStage(str, Enum):
    FAST = "fast"
    FINAL = "final"


class OutputFormat(str, Enum):
    JSON = "json"
    MARKDOWN = "markdown"
    TEXT = "text"


class JobStatus(str, Enum):
    QUEUED = "queued"
    SPLITTING = "splitting"
    SUBMITTING = "submitting"
    RUNNING = "running"
    AGGREGATING = "aggregating"
    COMPLETED_FAST = "completed_fast"
    COMPLETED_FINAL = "completed_final"
    COMPLETED = "completed"
    COMPLETED_WITH_ERRORS = "completed_with_errors"
    FAILED = "failed"


class PageResultStatus(str, Enum):
    COMPLETED = "completed"
    FAILED = "failed"


class ElementType(str, Enum):
    TEXT = "text"
    HEADING = "heading"
    LIST_ITEM = "list_item"
    TABLE = "table"
    FIGURE = "figure"
    CAPTION = "caption"
    HEADER = "header"
    FOOTER = "footer"
    PAGE_NUMBER = "page_number"
    FORMULA = "formula"
    FOOTNOTE = "footnote"
    UNKNOWN = "unknown"


class ParseEngine(str, Enum):
    DIGITAL_TEXT = "digital_text"
    PADDLE_OCR = "paddle_ocr"
    VLM_FALLBACK = "vlm_fallback"


class DebugOptions(StrictModel):
    persist_page_images: bool = True
    save_raw_model_output: bool = False
    save_prompt_text: bool = False


class JobTimings(StrictModel):
    split_ms: int = 0
    submit_ms: int = 0
    aggregate_ms: int = 0
    refine_ms: int = 0
    elapsed_ms: int = 0


class PageError(StrictModel):
    page_id: int
    code: str
    message: str
    retry_count: int = 0
    stage: str | None = None


class BoundingBox(StrictModel):
    coord: list[int] = Field(min_length=4, max_length=4)
    page_id: int


class DocumentPage(StrictModel):
    id: int
    image_uri: str
    width: int
    height: int
    rotation: int = 0
    page_hash: str | None = None


class DocumentElement(StrictModel):
    id: str
    page_id: int
    type: ElementType
    content: str
    bbox: BoundingBox
    order: int
    confidence: float | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)


class DocumentBody(StrictModel):
    pages: list[DocumentPage] = Field(default_factory=list)
    elements: list[DocumentElement] = Field(default_factory=list)


class DerivedOutputs(StrictModel):
    document_markdown: str = ""
    document_text: str = ""
    page_markdown: dict[str, str] = Field(default_factory=dict)


class FileMetadata(StrictModel):
    file_name: str
    mime_type: MimeType
    pages_total: int = 0
    bytes: int


class ModelMetadata(StrictModel):
    page_vlm: str | None = None
    fast_ocr: str | None = None
    fallback_vlm: str | None = None


class ResultMetadata(StrictModel):
    job_id: str
    schema_version: str
    pipeline_mode: ParseMode
    quality_stage: QualityStage = QualityStage.FAST
    result_revision: int = 1
    models: ModelMetadata
    file_metadata: FileMetadata
    timings: JobTimings


class DocumentParseResult(StrictModel):
    document: DocumentBody
    derived: DerivedOutputs
    error_status: list[PageError] = Field(default_factory=list)
    metadata: ResultMetadata


class PageElementCandidate(StrictModel):
    type: ElementType | str
    content: str
    bbox: BoundingBox | Annotated[list[int], Field(min_length=4, max_length=4)] | None = None
    order: int | None = None
    confidence: float | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)


class PageModelOutput(StrictModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    page_markdown: str = ""
    elements: list[PageElementCandidate] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class PageParseResult(StrictModel):
    job_id: str
    chunk_id: str
    page_id: int
    status: PageResultStatus
    page_markdown: str = ""
    elements: list[DocumentElement] = Field(default_factory=list)
    attempts: int = 0
    valid_on_first_pass: bool = False
    error: PageError | None = None
    warnings: list[str] = Field(default_factory=list)
    inference_ms: int = 0
    raw_output_path: str | None = None
    prompt_path: str | None = None
    result_revision: int = 1
    engine: ParseEngine = ParseEngine.VLM_FALLBACK
    confidence_summary: dict[str, float] = Field(default_factory=dict)
    fallback_triggered: bool = False


class PageTask(StrictModel):
    job_id: str
    chunk_id: str
    page_id: int
    mode: ParseMode
    image_path: str
    width: int
    height: int
    rotation: int = 0
    page_hash: str
    task_path: str
    result_path: str
    raw_output_path: str | None = None
    prompt_path: str | None = None
    result_revision: int = 1
    route_metrics: dict[str, float] = Field(default_factory=dict)
    source_text: str | None = None
    route_engine: ParseEngine | None = None
    latency_profile: LatencyProfile = LatencyProfile.BALANCED


class PageChunk(StrictModel):
    job_id: str
    chunk_id: str
    mode: ParseMode
    parser_version: str
    runtime_profile: str
    artifact_root: str
    model_id: str
    language_hint: str | None = None
    debug: DebugOptions = Field(default_factory=DebugOptions)
    pages: list[PageTask] = Field(default_factory=list)


class ChunkParseSummary(StrictModel):
    job_id: str
    chunk_id: str
    pages_total: int
    pages_completed: int
    pages_failed: int


class JobManifest(StrictModel):
    job_id: str
    parser_version: str
    schema_version: str
    runtime_profile: str
    created_at: datetime = Field(default_factory=utc_now)
    source_fingerprint: str
    request_payload: dict[str, Any]
    output_formats: list[OutputFormat]
    debug: DebugOptions = Field(default_factory=DebugOptions)
    model_id: str
    pipeline_mode: ParseMode
    file_metadata: FileMetadata
    pages: list[DocumentPage] = Field(default_factory=list)
    page_range: str | None = None
    max_pages: int | None = None
    language_hint: str | None = None
    chunk_ids: list[str] = Field(default_factory=list)
    latency_profile: LatencyProfile = LatencyProfile.BALANCED
    result_level: ResultLevel = ResultLevel.LATEST
    pending_refinement_pages: list[int] = Field(default_factory=list)
    result_revision: int = 0
    fallback_model_id: str | None = None


class JobProgressSnapshot(StrictModel):
    job_id: str
    status: JobStatus
    pages_total: int = 0
    pages_completed: int = 0
    pages_running: int = 0
    pages_failed: int = 0
    progress_percent: float = 0.0
    timings: JobTimings = Field(default_factory=JobTimings)
    error_summary: list[PageError] | None = None
    updated_at: datetime = Field(default_factory=utc_now)
    runtime_profile: str | None = None
    parser_version: str | None = None
    chunk_function_call_id: str | None = None
    result_revision: int = 0
    pending_refinement_pages: int = 0


class IdempotencyRecord(StrictModel):
    request_fingerprint: str
    job_id: str
    parser_version: str
    created_at: datetime = Field(default_factory=utc_now)


class ResultEnvelope(StrictModel):
    job_id: str
    status: JobStatus
    format: OutputFormat
    metadata: ResultMetadata
    result: dict[str, Any] | str
    pages: list[PageParseResult] | None = None
    debug: dict[str, Any] | None = None
    quality_stage: QualityStage | None = None
    result_revision: int | None = None

    @model_validator(mode="after")
    def ensure_result_shape(self) -> "ResultEnvelope":
        if self.format == OutputFormat.JSON and not isinstance(self.result, dict):
            raise ValueError("JSON result envelopes must contain an object result")
        if self.format != OutputFormat.JSON and not isinstance(self.result, str):
            raise ValueError("Text and markdown envelopes must contain string results")
        if self.result_revision is None:
            self.result_revision = self.metadata.result_revision
        if self.quality_stage is None:
            self.quality_stage = self.metadata.quality_stage
        return self
