from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import fitz

from .config import (
    ROUTING_EXTRACTABLE_CHAR_THRESHOLD,
    ROUTING_PRINTABLE_RATIO_THRESHOLD,
)
from .types_result import MimeType, ParseEngine


@dataclass(frozen=True)
class PageRouteDecision:
    page_id: int
    engine: ParseEngine
    extractable_char_count: int
    printable_ratio: float
    extracted_text: str | None = None


def _printable_ratio(text: str) -> float:
    visible = [char for char in text if not char.isspace()]
    if not visible:
        return 0.0
    printable = sum(1 for char in visible if char.isprintable())
    return printable / len(visible)


def _extractable_char_count(text: str) -> int:
    return sum(1 for char in text if not char.isspace())


def classify_page(
    *,
    page_id: int,
    extracted_text: str | None,
    extractable_char_threshold: int = ROUTING_EXTRACTABLE_CHAR_THRESHOLD,
    printable_ratio_threshold: float = ROUTING_PRINTABLE_RATIO_THRESHOLD,
) -> PageRouteDecision:
    text = extracted_text or ""
    char_count = _extractable_char_count(text)
    printable_ratio = _printable_ratio(text)
    if char_count >= extractable_char_threshold and printable_ratio >= printable_ratio_threshold:
        engine = ParseEngine.DIGITAL_TEXT
    else:
        engine = ParseEngine.PADDLE_OCR
        text = None

    return PageRouteDecision(
        page_id=page_id,
        engine=engine,
        extractable_char_count=char_count,
        printable_ratio=printable_ratio,
        extracted_text=text,
    )


def route_pages(
    *,
    source_bytes: bytes,
    mime_type: MimeType,
    page_ids: Iterable[int],
) -> dict[int, PageRouteDecision]:
    ordered_page_ids = list(page_ids)
    if mime_type != MimeType.PDF:
        return {
            page_id: PageRouteDecision(
                page_id=page_id,
                engine=ParseEngine.PADDLE_OCR,
                extractable_char_count=0,
                printable_ratio=0.0,
            )
            for page_id in ordered_page_ids
        }

    document = fitz.open(stream=source_bytes, filetype="pdf")
    decisions: dict[int, PageRouteDecision] = {}
    for page_id in ordered_page_ids:
        page = document.load_page(page_id)
        extracted_text = page.get_text("text") or ""
        decisions[page_id] = classify_page(
            page_id=page_id,
            extracted_text=extracted_text,
        )
    return decisions
