from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import Field

from .types_result import StrictModel


class ExtractionFieldType(str, Enum):
    STRING = "string"
    NUMBER = "number"
    DATE = "date"
    BOOLEAN = "boolean"
    LIST_STRING = "list[string]"


class ExtractionMode(str, Enum):
    PER_PAGE = "per_page"
    WHOLE_DOCUMENT = "whole_document"


class ExtractionStatus(str, Enum):
    PENDING = "pending"
    SUGGESTING = "suggesting"
    EXTRACTING = "extracting"
    COMPLETED = "completed"
    FAILED = "failed"


class EntityFieldDefinition(StrictModel):
    """A single field within an entity schema."""

    name: str
    field_type: ExtractionFieldType
    description: str = ""
    required: bool = True
    examples: list[str] = Field(default_factory=list)


class EntityDefinition(StrictModel):
    """An entity type the user wants to extract."""

    entity_name: str
    description: str = ""
    fields: list[EntityFieldDefinition]


class EntitySuggestionResponse(StrictModel):
    job_id: str
    suggested_entities: list[EntityDefinition]
    document_summary: str = ""


class EntityExtractionRequest(StrictModel):
    job_id: str
    entities: list[EntityDefinition]
    extraction_mode: ExtractionMode = ExtractionMode.PER_PAGE


class ExtractedEntity(StrictModel):
    entity_name: str
    page_id: int | None = None
    data: dict[str, Any] = Field(default_factory=dict)
    confidence: float | None = None


class EntityExtractionResult(StrictModel):
    job_id: str
    entities: list[ExtractedEntity]
    schema_used: list[EntityDefinition]
    extraction_mode: ExtractionMode
    model_id: str = ""
    inference_ms: int = 0


class EntityExtractionStatusPayload(StrictModel):
    job_id: str
    status: ExtractionStatus
    entities_requested: int = 0
    pages_processed: int = 0
    pages_total: int = 0
    requests_total: int = 0
    requests_completed: int = 0
    error_message: str | None = None


class ExtractionWorkItem(StrictModel):
    job_id: str
    entity: EntityDefinition
    page_id: int
    page_text: str
    json_schema: dict[str, Any]
    model_id: str
    max_tokens: int
    session_id: str


class ExtractionWorkResult(StrictModel):
    entity_name: str
    page_id: int
    data: dict[str, Any] = Field(default_factory=dict)
    inference_ms: int = 0


_FIELD_TYPE_TO_JSON_SCHEMA: dict[ExtractionFieldType, dict[str, Any]] = {
    ExtractionFieldType.STRING: {"type": "string"},
    ExtractionFieldType.NUMBER: {"type": "number"},
    ExtractionFieldType.DATE: {"type": "string", "description": "ISO 8601 date string"},
    ExtractionFieldType.BOOLEAN: {"type": "boolean"},
    ExtractionFieldType.LIST_STRING: {"type": "array", "items": {"type": "string"}},
}


def entity_definition_to_json_schema(entity: EntityDefinition) -> dict[str, Any]:
    """Convert an EntityDefinition to a JSON Schema dict for vLLM guided decoding."""
    properties: dict[str, Any] = {}
    required: list[str] = []

    for field in entity.fields:
        prop = dict(_FIELD_TYPE_TO_JSON_SCHEMA[field.field_type])
        if field.description:
            prop["description"] = field.description
        # Allow null for optional fields
        if not field.required:
            prop = {"anyOf": [prop, {"type": "null"}]}
        properties[field.name] = prop
        if field.required:
            required.append(field.name)

    schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "additionalProperties": False,
    }
    if required:
        schema["required"] = required
    return schema
