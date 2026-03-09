from __future__ import annotations

import hashlib
import json
from typing import Any

from .json_output import repair_json_string
from .prompts_extraction import (
    build_entity_extraction_prompt,
    build_entity_suggestion_prompt,
)
from .types_extraction import EntityDefinition


_SUGGESTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "entity_name": {"type": "string"},
                    "description": {"type": "string"},
                    "fields": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "field_type": {
                                    "type": "string",
                                    "enum": ["string", "number", "date", "boolean", "list[string]"],
                                },
                                "description": {"type": "string"},
                                "required": {"type": "boolean"},
                                "examples": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                            },
                            "required": ["name", "field_type", "description", "required", "examples"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["entity_name", "description", "fields"],
                "additionalProperties": False,
            },
        },
        "document_summary": {"type": "string"},
    },
    "required": ["entities", "document_summary"],
    "additionalProperties": False,
}


def _stable_hash(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def build_modal_session_id(
    job_id: str,
    *,
    scope: str,
    entity_name: str | None = None,
) -> str:
    payload = {"job_id": job_id, "scope": scope, "entity_name": entity_name or ""}
    return f"doc-parse-{_stable_hash(payload)[:24]}"


def build_job_extraction_session_id(job_id: str) -> str:
    """Use one sticky extraction session per job to favor shared-server reuse."""
    return build_modal_session_id(job_id, scope="extract-job")


def build_suggestion_request_fingerprint(
    *,
    job_id: str,
    result_revision: int,
    model_id: str,
) -> str:
    return _stable_hash(
        {
            "kind": "suggestion",
            "job_id": job_id,
            "result_revision": result_revision,
            "model_id": model_id,
        }
    )


def build_extraction_request_fingerprint(
    *,
    job_id: str,
    result_revision: int,
    request_payload: dict[str, Any],
    model_id: str,
) -> str:
    return _stable_hash(
        {
            "kind": "extraction",
            "job_id": job_id,
            "result_revision": result_revision,
            "request_payload": request_payload,
            "model_id": model_id,
        }
    )


def build_extraction_headers(session_id: str) -> dict[str, str]:
    return {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Modal-Session-ID": session_id,
    }


def _json_schema_response_format(name: str, schema: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": name,
            "strict": True,
            "schema": schema,
        },
    }


def build_suggestion_chat_request(
    *,
    document_markdown: str,
    page_count: int,
    model_id: str,
    max_tokens: int,
) -> dict[str, Any]:
    return {
        "model": model_id,
        "messages": build_entity_suggestion_prompt(
            document_markdown=document_markdown,
            page_count=page_count,
        ),
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
        "response_format": _json_schema_response_format("entity_suggestion", _SUGGESTION_SCHEMA),
    }


def build_entity_extraction_chat_request(
    *,
    entity: EntityDefinition,
    page_text: str,
    model_id: str,
    json_schema: dict[str, Any],
    max_tokens: int,
) -> dict[str, Any]:
    return {
        "model": model_id,
        "messages": build_entity_extraction_prompt(entity, page_text),
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
        "response_format": _json_schema_response_format(
            f"{entity.entity_name.lower()}_extraction",
            json_schema,
        ),
    }


def extract_chat_completion_content(payload: dict[str, Any]) -> str:
    choices = payload.get("choices") or []
    if not choices:
        raise ValueError("Extraction server returned no choices.")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        ]
        if parts:
            return "".join(parts)
    raise ValueError("Extraction server returned an unsupported message payload.")


def parse_chat_completion_json_content(raw_text: str) -> dict[str, Any]:
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError:
        data = json.loads(repair_json_string(raw_text))
    if not isinstance(data, dict):
        raise ValueError("Extraction server returned JSON that is not an object.")
    return data


def extract_chat_completion_json(payload: dict[str, Any]) -> dict[str, Any]:
    return parse_chat_completion_json_content(extract_chat_completion_content(payload))
