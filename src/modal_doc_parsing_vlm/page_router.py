from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

import fitz

from .config import (
    ROUTING_COMMON_WORD_RATIO_THRESHOLD,
    ROUTING_EXTRACTABLE_CHAR_THRESHOLD,
    ROUTING_MIN_WORDS_FOR_LANGUAGE_CHECK,
    ROUTING_PRINTABLE_RATIO_THRESHOLD,
)
from .types_result import MimeType, ParseEngine

_WORD_RE = re.compile(r"[A-Za-z]{2,}")
_COMMON_ENGLISH_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "but",
    "by",
    "for",
    "from",
    "had",
    "has",
    "have",
    "he",
    "her",
    "his",
    "in",
    "into",
    "is",
    "it",
    "its",
    "not",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "there",
    "they",
    "this",
    "to",
    "was",
    "were",
    "which",
    "with",
}


@dataclass(frozen=True)
class PageRouteDecision:
    page_id: int
    engine: ParseEngine
    extractable_char_count: int
    printable_ratio: float
    common_word_ratio: float
    extracted_text: str | None = None


def _printable_ratio(text: str) -> float:
    visible = [char for char in text if not char.isspace()]
    if not visible:
        return 0.0
    printable = sum(1 for char in visible if char.isprintable())
    return printable / len(visible)


def _extractable_char_count(text: str) -> int:
    return sum(1 for char in text if not char.isspace())


def _common_word_ratio(text: str) -> float:
    words = [match.group(0).lower() for match in _WORD_RE.finditer(text)]
    if not words:
        return 0.0
    hits = sum(1 for word in words if word in _COMMON_ENGLISH_WORDS)
    return hits / len(words)


def classify_page(
    *,
    page_id: int,
    extracted_text: str | None,
    extractable_char_threshold: int = ROUTING_EXTRACTABLE_CHAR_THRESHOLD,
    printable_ratio_threshold: float = ROUTING_PRINTABLE_RATIO_THRESHOLD,
    common_word_ratio_threshold: float = ROUTING_COMMON_WORD_RATIO_THRESHOLD,
    min_words_for_language_check: int = ROUTING_MIN_WORDS_FOR_LANGUAGE_CHECK,
) -> PageRouteDecision:
    text = extracted_text or ""
    char_count = _extractable_char_count(text)
    printable_ratio = _printable_ratio(text)
    words = [match.group(0) for match in _WORD_RE.finditer(text)]
    common_word_ratio = _common_word_ratio(text)
    language_quality_ok = (
        len(words) < min_words_for_language_check
        or common_word_ratio >= common_word_ratio_threshold
    )
    if (
        char_count >= extractable_char_threshold
        and printable_ratio >= printable_ratio_threshold
        and language_quality_ok
    ):
        engine = ParseEngine.DIGITAL_TEXT
    else:
        engine = ParseEngine.PADDLE_OCR
        text = None

    return PageRouteDecision(
        page_id=page_id,
        engine=engine,
        extractable_char_count=char_count,
        printable_ratio=printable_ratio,
        common_word_ratio=common_word_ratio,
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
                common_word_ratio=0.0,
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
