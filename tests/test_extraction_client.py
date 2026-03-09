from __future__ import annotations

from modal_doc_parsing_vlm.extraction_client import (
    build_entity_extraction_chat_request,
    build_extraction_headers,
    build_job_extraction_session_id,
    build_modal_session_id,
    build_suggestion_chat_request,
    extract_chat_completion_json,
    parse_chat_completion_json_content,
)
from modal_doc_parsing_vlm.types_extraction import (
    EntityDefinition,
    EntityFieldDefinition,
    ExtractionFieldType,
    entity_definition_to_json_schema,
)


def _entity() -> EntityDefinition:
    return EntityDefinition(
        entity_name="Invoice",
        description="Invoice header fields",
        fields=[
            EntityFieldDefinition(
                name="invoice_number",
                field_type=ExtractionFieldType.STRING,
                description="Invoice identifier",
                required=True,
            )
        ],
    )


def test_build_suggestion_chat_request_uses_json_schema_and_disables_thinking():
    payload = build_suggestion_chat_request(
        document_markdown="# Invoice\n\nInvoice Number: INV-1",
        page_count=1,
        model_id="Qwen/Qwen3-4B-Thinking-2507-FP8",
        max_tokens=512,
    )

    assert payload["response_format"]["type"] == "json_schema"
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert payload["model"] == "Qwen/Qwen3-4B-Thinking-2507-FP8"


def test_build_entity_extraction_chat_request_uses_schema_and_disables_thinking():
    entity = _entity()
    payload = build_entity_extraction_chat_request(
        entity=entity,
        page_text="Invoice Number: INV-1",
        model_id="Qwen/Qwen3-4B-Thinking-2507-FP8",
        json_schema=entity_definition_to_json_schema(entity),
        max_tokens=256,
    )

    assert payload["response_format"]["type"] == "json_schema"
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert payload["response_format"]["json_schema"]["schema"]["type"] == "object"


def test_modal_session_id_is_stable_and_added_to_headers():
    session_id = build_modal_session_id("job-123", scope="extract", entity_name="Invoice")

    assert session_id == build_modal_session_id(
        "job-123", scope="extract", entity_name="Invoice"
    )
    assert session_id != build_modal_session_id("job-123", scope="suggest")
    assert build_extraction_headers(session_id)["Modal-Session-ID"] == session_id


def test_job_extraction_session_id_is_shared_across_steps_but_not_warmup():
    session_id = build_job_extraction_session_id("job-123")

    assert session_id == build_job_extraction_session_id("job-123")
    assert session_id != build_modal_session_id("warmup", scope="warm")
    assert build_extraction_headers(session_id)["Modal-Session-ID"] == session_id


def test_parse_chat_completion_json_content_repairs_unterminated_strings():
    raw_text = '{\n  "title": "Modal Demo Ideas",\n  "summary": "A draft document\n'

    parsed = parse_chat_completion_json_content(raw_text)

    assert parsed["title"] == "Modal Demo Ideas"
    assert parsed["summary"] == "A draft document"


def test_extract_chat_completion_json_reads_and_repairs_message_content():
    payload = {
        "choices": [
            {
                "message": {
                    "content": '{\n  "entity_name": "Invoice",\n  "invoice_number": "INV-001"\n'
                }
            }
        ]
    }

    parsed = extract_chat_completion_json(payload)

    assert parsed == {"entity_name": "Invoice", "invoice_number": "INV-001"}
