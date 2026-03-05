from __future__ import annotations

from .config import EXTRACTION_SUGGESTION_MAX_CHARS
from .types_extraction import EntityDefinition


SUGGESTION_SYSTEM = (
    "You are an entity extraction expert. "
    "Analyze documents and suggest structured entities that could be extracted."
)

SUGGESTION_USER_TEMPLATE = """Analyze the following document text and suggest structured entities that could be extracted from it.

Document ({page_count} page{page_plural}):
---
{document_text}
---

For each entity, provide:
- entity_name: A concise PascalCase name (e.g. "Invoice", "Person", "LineItem", "Address")
- description: What this entity represents in one sentence
- fields: Array of field definitions, each with:
  - name: snake_case field name
  - field_type: one of "string", "number", "date", "boolean", "list[string]"
  - description: what the field contains
  - required: true if the field is always expected to be present
  - examples: 1-2 example values from the document if visible

Suggest between 1 and 5 entities that are most relevant to this document type.
Focus on entities with concrete, extractable data — not abstract concepts.

Return JSON only with this exact shape:
{{"entities": [...], "document_summary": "brief one-sentence summary of the document"}}

Do not wrap the JSON in markdown fences.
Do not include commentary outside the JSON object."""


EXTRACTION_SYSTEM = (
    "You are a precise data extraction engine. "
    "Extract structured data from document text according to the provided schema. "
    "Use null for fields you cannot find. Do not fabricate data."
)

EXTRACTION_USER_TEMPLATE = """Extract the following entity from this document text.

Entity: {entity_name}
Description: {entity_description}

Expected fields:
{fields_description}

Document text:
---
{page_text}
---

Return a single JSON object with exactly the fields listed above.
Use null for any field whose value cannot be determined from the text.
Do not wrap the JSON in markdown fences."""


def build_entity_suggestion_prompt(
    document_markdown: str,
    page_count: int,
) -> list[dict[str, str]]:
    """Build chat messages for entity suggestion."""
    truncated = document_markdown[:EXTRACTION_SUGGESTION_MAX_CHARS]
    user_content = SUGGESTION_USER_TEMPLATE.format(
        page_count=page_count,
        page_plural="s" if page_count != 1 else "",
        document_text=truncated,
    )
    return [
        {"role": "system", "content": SUGGESTION_SYSTEM},
        {"role": "user", "content": user_content},
    ]


def _format_fields_description(entity: EntityDefinition) -> str:
    lines: list[str] = []
    for field in entity.fields:
        req = "required" if field.required else "optional"
        lines.append(f"- {field.name} ({field.field_type.value}, {req}): {field.description}")
    return "\n".join(lines)


def build_entity_extraction_prompt(
    entity: EntityDefinition,
    page_text: str,
) -> list[dict[str, str]]:
    """Build chat messages for entity extraction from a page."""
    user_content = EXTRACTION_USER_TEMPLATE.format(
        entity_name=entity.entity_name,
        entity_description=entity.description,
        fields_description=_format_fields_description(entity),
        page_text=page_text,
    )
    return [
        {"role": "system", "content": EXTRACTION_SYSTEM},
        {"role": "user", "content": user_content},
    ]
