"""Tests for entity extraction prompt builders."""

from __future__ import annotations

from modal_doc_parsing_vlm.prompts_extraction import (
    build_entity_extraction_prompt,
    build_entity_suggestion_prompt,
)
from modal_doc_parsing_vlm.types_extraction import (
    EntityDefinition,
    EntityFieldDefinition,
    ExtractionFieldType,
)


class TestBuildEntitySuggestionPrompt:
    def test_returns_messages_list(self):
        messages = build_entity_suggestion_prompt("Some document text", page_count=3)
        assert isinstance(messages, list)
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    def test_includes_document_text(self):
        messages = build_entity_suggestion_prompt("Invoice #12345", page_count=1)
        assert "Invoice #12345" in messages[1]["content"]

    def test_truncates_long_text(self):
        long_text = "x" * 20000
        messages = build_entity_suggestion_prompt(long_text, page_count=1)
        # Should be truncated to ~8000 chars
        assert len(messages[1]["content"]) < 15000

    def test_page_count_in_prompt(self):
        messages = build_entity_suggestion_prompt("doc", page_count=5)
        assert "5 pages" in messages[1]["content"]

    def test_singular_page(self):
        messages = build_entity_suggestion_prompt("doc", page_count=1)
        assert "1 page)" in messages[1]["content"]


class TestBuildEntityExtractionPrompt:
    def test_returns_messages_list(self):
        entity = EntityDefinition(
            entity_name="Person",
            description="A person mentioned in the document",
            fields=[
                EntityFieldDefinition(
                    name="full_name",
                    field_type=ExtractionFieldType.STRING,
                    description="Full name of the person",
                ),
            ],
        )
        messages = build_entity_extraction_prompt(entity, "John Doe is the CEO.")
        assert isinstance(messages, list)
        assert len(messages) == 2
        assert messages[0]["role"] == "system"

    def test_includes_entity_info(self):
        entity = EntityDefinition(
            entity_name="Address",
            description="A physical address",
            fields=[
                EntityFieldDefinition(
                    name="street",
                    field_type=ExtractionFieldType.STRING,
                    description="Street address",
                ),
                EntityFieldDefinition(
                    name="zip_code",
                    field_type=ExtractionFieldType.STRING,
                    description="ZIP code",
                    required=False,
                ),
            ],
        )
        messages = build_entity_extraction_prompt(entity, "123 Main St, ZIP 90210")
        user_content = messages[1]["content"]
        assert "Address" in user_content
        assert "street" in user_content
        assert "zip_code" in user_content
        assert "optional" in user_content
        assert "123 Main St" in user_content
