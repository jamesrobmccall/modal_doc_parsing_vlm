from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .config import DEFAULT_POLL_AFTER_SECONDS
from .types_result import (
    DebugOptions,
    JobStatus,
    JobTimings,
    LatencyProfile,
    MimeType,
    OutputFormat,
    PageError,
    ParseMode,
    ResultLevel,
    ResultEnvelope,
)


PAGE_RANGE_PATTERN = r"^\d+(-\d+)?(,\d+(-\d+)?)*$"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class UrlDocumentSource(StrictModel):
    type: Literal["url"]
    url: str


class UploadRefDocumentSource(StrictModel):
    type: Literal["upload_ref"]
    upload_id: str


class BytesDocumentSource(StrictModel):
    type: Literal["bytes"]
    base64: str


DocumentSource = Annotated[
    UrlDocumentSource | UploadRefDocumentSource | BytesDocumentSource,
    Field(discriminator="type"),
]


class SubmitDocumentParseRequest(StrictModel):
    source: DocumentSource
    mime_type: MimeType
    mode: ParseMode = ParseMode.BALANCED
    output_formats: list[OutputFormat]
    max_pages: int | None = None
    page_range: str | None = None
    language_hint: str | None = None
    debug: DebugOptions | None = None
    result_level: ResultLevel = ResultLevel.LATEST
    latency_profile: LatencyProfile = LatencyProfile.BALANCED

    @field_validator("output_formats")
    @classmethod
    def validate_output_formats(cls, value: list[OutputFormat]) -> list[OutputFormat]:
        if not value:
            raise ValueError("output_formats must contain at least one format")
        unique = list(dict.fromkeys(value))
        if OutputFormat.JSON not in unique:
            unique.append(OutputFormat.JSON)
        return unique

    @field_validator("page_range")
    @classmethod
    def validate_page_range(cls, value: str | None) -> str | None:
        if value is None:
            return value
        import re

        if not re.fullmatch(PAGE_RANGE_PATTERN, value):
            raise ValueError("page_range must look like 1-3,5,8-9")
        return value

    @field_validator("max_pages")
    @classmethod
    def validate_max_pages(cls, value: int | None) -> int | None:
        if value is not None and value <= 0:
            raise ValueError("max_pages must be greater than zero")
        return value

    @model_validator(mode="after")
    def ensure_debug_defaults(self) -> "SubmitDocumentParseRequest":
        if self.debug is None:
            self.debug = DebugOptions()
        return self


class SubmitDocumentParseResponse(StrictModel):
    job_id: str
    status: Literal["queued"] = "queued"
    pages_total: int | None = None
    poll_after_seconds: int = DEFAULT_POLL_AFTER_SECONDS


class GetDocumentParseStatusRequest(StrictModel):
    job_id: str


class GetDocumentParseStatusResponse(StrictModel):
    status: JobStatus
    pages_total: int
    pages_completed: int
    pages_running: int
    pages_failed: int
    progress_percent: float
    timings: JobTimings
    error_summary: list[PageError] | None = None
    result_revision: int = 0
    pending_refinement_pages: int = 0


class GetDocumentParseResultRequest(StrictModel):
    job_id: str
    format: OutputFormat
    include_pages: bool = False
    include_debug: bool = False
    result_level: ResultLevel = ResultLevel.LATEST


class GetDocumentParseResultResponse(ResultEnvelope):
    pass


class ToolError(StrictModel):
    code: str
    message: str
    details: dict[str, Any] | None = None
