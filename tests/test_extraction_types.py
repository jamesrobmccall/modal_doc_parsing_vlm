"""Tests for entity extraction Pydantic types and JSON schema generation."""

from __future__ import annotations

import json

import pytest

from modal_doc_parsing_vlm.types_extraction import (
    EntityDefinition,
    EntityExtractionRequest,
    EntityExtractionResult,
    EntityExtractionStatusPayload,
    EntityFieldDefinition,
    EntitySuggestionResponse,
    ExtractedEntity,
    ExtractionFieldType,
    ExtractionMode,
    ExtractionStatus,
    ExtractionWorkItem,
    ExtractionWorkResult,
    entity_definition_to_json_schema,
)


def _invoice_entity() -> EntityDefinition:
    return EntityDefinition(
        entity_name="Invoice",
        description="An invoice document",
        fields=[
            EntityFieldDefinition(
                name="invoice_number",
                field_type=ExtractionFieldType.STRING,
                description="Invoice ID",
                required=True,
            ),
            EntityFieldDefinition(
                name="total_amount",
                field_type=ExtractionFieldType.NUMBER,
                description="Total amount due",
                required=True,
            ),
            EntityFieldDefinition(
                name="issue_date",
                field_type=ExtractionFieldType.DATE,
                description="Date issued",
                required=False,
            ),
            EntityFieldDefinition(
                name="is_paid",
                field_type=ExtractionFieldType.BOOLEAN,
                description="Payment status",
                required=False,
            ),
            EntityFieldDefinition(
                name="line_items",
                field_type=ExtractionFieldType.LIST_STRING,
                description="List of item descriptions",
                required=False,
            ),
        ],
    )


class TestEntityDefinitionToJsonSchema:
    def test_produces_valid_json_schema(self):
        entity = _invoice_entity()
        schema = entity_definition_to_json_schema(entity)

        assert schema["type"] == "object"
        assert "properties" in schema
        assert schema["additionalProperties"] is False

    def test_required_fields_listed(self):
        entity = _invoice_entity()
        schema = entity_definition_to_json_schema(entity)

        assert "required" in schema
        assert "invoice_number" in schema["required"]
        assert "total_amount" in schema["required"]
        assert "issue_date" not in schema["required"]

    def test_field_types_mapped_correctly(self):
        entity = _invoice_entity()
        schema = entity_definition_to_json_schema(entity)
        props = schema["properties"]

        assert props["invoice_number"]["type"] == "string"
        assert props["total_amount"]["type"] == "number"
        # Optional fields use anyOf
        assert props["issue_date"]["anyOf"][0]["type"] == "string"
        assert props["is_paid"]["anyOf"][0]["type"] == "boolean"
        assert props["line_items"]["anyOf"][0]["type"] == "array"

    def test_schema_is_json_serializable(self):
        entity = _invoice_entity()
        schema = entity_definition_to_json_schema(entity)
        serialized = json.dumps(schema)
        assert isinstance(serialized, str)
        roundtrip = json.loads(serialized)
        assert roundtrip == schema

    def test_empty_entity_no_required(self):
        entity = EntityDefinition(
            entity_name="Empty",
            fields=[
                EntityFieldDefinition(
                    name="optional_field",
                    field_type=ExtractionFieldType.STRING,
                    required=False,
                )
            ],
        )
        schema = entity_definition_to_json_schema(entity)
        assert "required" not in schema


class TestPydanticModels:
    def test_entity_definition_roundtrip(self):
        entity = _invoice_entity()
        data = entity.model_dump(mode="json")
        restored = EntityDefinition.model_validate(data)
        assert restored.entity_name == entity.entity_name
        assert len(restored.fields) == len(entity.fields)

    def test_suggestion_response(self):
        response = EntitySuggestionResponse(
            job_id="test123",
            suggested_entities=[_invoice_entity()],
            document_summary="A test invoice.",
        )
        data = response.model_dump(mode="json")
        assert data["job_id"] == "test123"
        assert len(data["suggested_entities"]) == 1

    def test_extraction_request_default_mode(self):
        req = EntityExtractionRequest(
            job_id="test123",
            entities=[_invoice_entity()],
        )
        assert req.extraction_mode == ExtractionMode.PER_PAGE

    def test_extraction_result(self):
        result = EntityExtractionResult(
            job_id="test123",
            entities=[
                ExtractedEntity(
                    entity_name="Invoice",
                    page_id=0,
                    data={"invoice_number": "INV-001", "total_amount": 100.50},
                ),
            ],
            schema_used=[_invoice_entity()],
            extraction_mode=ExtractionMode.PER_PAGE,
            model_id="Qwen/Qwen2.5-3B-Instruct",
            inference_ms=1234,
        )
        data = result.model_dump(mode="json")
        assert data["entities"][0]["data"]["invoice_number"] == "INV-001"
        assert data["inference_ms"] == 1234

    def test_extraction_status_payload_tracks_request_progress(self):
        status = EntityExtractionStatusPayload(
            job_id="job-123",
            status=ExtractionStatus.EXTRACTING,
            entities_requested=2,
            pages_processed=1,
            pages_total=3,
            requests_total=6,
            requests_completed=2,
        )
        data = status.model_dump(mode="json")
        assert data["requests_total"] == 6
        assert data["requests_completed"] == 2

    def test_extraction_work_item_and_result_roundtrip(self):
        item = ExtractionWorkItem(
            job_id="job-123",
            entity=_invoice_entity(),
            page_id=0,
            page_text="Invoice Number: INV-001",
            json_schema=entity_definition_to_json_schema(_invoice_entity()),
            model_id="Qwen/Qwen3-4B-Thinking-2507-FP8",
            max_tokens=1024,
            session_id="doc-parse-session",
        )
        restored_item = ExtractionWorkItem.model_validate(item.model_dump(mode="json"))
        assert restored_item.page_id == 0

        result = ExtractionWorkResult(
            entity_name="Invoice",
            page_id=0,
            data={"invoice_number": "INV-001"},
            inference_ms=55,
        )
        restored_result = ExtractionWorkResult.model_validate(result.model_dump(mode="json"))
        assert restored_result.inference_ms == 55
