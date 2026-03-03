from __future__ import annotations

import json
from typing import Any

from json_repair import repair_json

from .types_result import (
    BoundingBox,
    DocumentElement,
    ElementType,
    PageError,
    PageModelOutput,
)


def extract_json_object(raw_text: str) -> str:
    start = raw_text.find("{")
    end = raw_text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("No JSON object found in model output")
    return raw_text[start : end + 1]


def repair_json_string(raw_text: str) -> str:
    candidate = extract_json_object(raw_text)
    return repair_json(candidate, return_objects=False)


def parse_model_output(raw_text: str) -> PageModelOutput:
    candidate = repair_json_string(raw_text)
    data = json.loads(candidate)
    return PageModelOutput.model_validate(data)


def coerce_element_type(value: ElementType | str) -> ElementType:
    if isinstance(value, ElementType):
        return value
    try:
        return ElementType(value)
    except ValueError:
        return ElementType.UNKNOWN


def normalize_page_output(raw_output: PageModelOutput, page_id: int) -> list[DocumentElement]:
    elements: list[DocumentElement] = []
    for index, item in enumerate(raw_output.elements, start=1):
        bbox = item.bbox or BoundingBox(coord=[0, 0, 0, 0], page_id=page_id)
        bbox = BoundingBox(coord=bbox.coord, page_id=page_id)
        order = item.order or index
        elements.append(
            DocumentElement(
                id=f"p{page_id}-e{order}",
                page_id=page_id,
                type=coerce_element_type(item.type),
                content=item.content,
                bbox=bbox,
                order=order,
                confidence=item.confidence,
                attributes=item.attributes,
            )
        )
    elements.sort(key=lambda element: element.order)
    return elements


def page_error(page_id: int, code: str, message: str, retry_count: int, stage: str) -> PageError:
    return PageError(
        page_id=page_id,
        code=code,
        message=message,
        retry_count=retry_count,
        stage=stage,
    )


def parse_and_normalize_page_output(
    raw_text: str,
    page_id: int,
) -> tuple[PageModelOutput, list[DocumentElement]]:
    parsed = parse_model_output(raw_text)
    return parsed, normalize_page_output(parsed, page_id)


def build_debug_payload(raw_output: str | None = None, prompt: str | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if raw_output is not None:
        payload["raw_output"] = raw_output
    if prompt is not None:
        payload["prompt"] = prompt
    return payload
