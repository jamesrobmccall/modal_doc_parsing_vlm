from __future__ import annotations

from .config import (
    FALLBACK_MEAN_OCR_CONFIDENCE_THRESHOLD,
    FALLBACK_MIN_ELEMENT_COUNT,
    FALLBACK_TABLE_CONFIDENCE_THRESHOLD,
    FALLBACK_TEXT_COVERAGE_THRESHOLD,
)
from .types_result import ElementType, PageParseResult


def _mean_confidence(page_result: PageParseResult) -> float:
    if page_result.confidence_summary.get("mean_ocr_confidence") is not None:
        return float(page_result.confidence_summary["mean_ocr_confidence"])
    scores = [element.confidence for element in page_result.elements if element.confidence is not None]
    if not scores:
        return 0.0
    return sum(scores) / len(scores)


def _text_coverage_ratio(page_result: PageParseResult) -> float:
    if page_result.confidence_summary.get("text_coverage_ratio") is not None:
        return float(page_result.confidence_summary["text_coverage_ratio"])
    text_length = len(page_result.page_markdown.strip())
    if text_length == 0:
        return 0.0
    return min(1.0, text_length / 2000.0)


def _table_confidence(page_result: PageParseResult) -> float:
    if page_result.confidence_summary.get("table_confidence") is not None:
        return float(page_result.confidence_summary["table_confidence"])
    table_scores = [
        element.confidence
        for element in page_result.elements
        if element.type == ElementType.TABLE and element.confidence is not None
    ]
    if not table_scores:
        return 1.0
    return sum(table_scores) / len(table_scores)


def _has_structure_anomaly(page_result: PageParseResult) -> bool:
    if len(page_result.elements) < FALLBACK_MIN_ELEMENT_COUNT:
        return True
    orders = [element.order for element in page_result.elements]
    return orders != sorted(orders)


def fallback_reasons(page_result: PageParseResult) -> list[str]:
    reasons: list[str] = []
    mean_ocr_confidence = _mean_confidence(page_result)
    text_coverage_ratio = _text_coverage_ratio(page_result)
    table_confidence = _table_confidence(page_result)
    table_detected = any(element.type == ElementType.TABLE for element in page_result.elements)

    if mean_ocr_confidence < FALLBACK_MEAN_OCR_CONFIDENCE_THRESHOLD:
        reasons.append("mean_ocr_confidence")
    if text_coverage_ratio < FALLBACK_TEXT_COVERAGE_THRESHOLD:
        reasons.append("text_coverage_ratio")
    if table_detected and table_confidence < FALLBACK_TABLE_CONFIDENCE_THRESHOLD:
        reasons.append("table_confidence")
    if _has_structure_anomaly(page_result):
        reasons.append("structure_anomaly")
    return reasons


def needs_fallback(page_result: PageParseResult) -> bool:
    return bool(fallback_reasons(page_result))
