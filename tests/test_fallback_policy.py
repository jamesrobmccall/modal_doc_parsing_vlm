from modal_doc_parsing_vlm.fallback_policy import fallback_reasons, needs_fallback
from modal_doc_parsing_vlm.types_result import BoundingBox, DocumentElement, PageParseResult, PageResultStatus


def _page_result(
    *,
    mean_ocr_confidence: float = 0.95,
    text_coverage_ratio: float = 0.9,
    table_confidence: float = 1.0,
) -> PageParseResult:
    return PageParseResult(
        job_id="job",
        chunk_id="page",
        page_id=0,
        status=PageResultStatus.COMPLETED,
        page_markdown="hello world",
        elements=[
            DocumentElement(
                id="p0-e1",
                page_id=0,
                type="text",
                content="hello world",
                bbox=BoundingBox(coord=[0, 0, 10, 10], page_id=0),
                order=1,
                confidence=mean_ocr_confidence,
            ),
            DocumentElement(
                id="p0-e2",
                page_id=0,
                type="text",
                content="line 2",
                bbox=BoundingBox(coord=[0, 10, 10, 20], page_id=0),
                order=2,
                confidence=mean_ocr_confidence,
            ),
            DocumentElement(
                id="p0-e3",
                page_id=0,
                type="text",
                content="line 3",
                bbox=BoundingBox(coord=[0, 20, 10, 30], page_id=0),
                order=3,
                confidence=mean_ocr_confidence,
            ),
        ],
        confidence_summary={
            "mean_ocr_confidence": mean_ocr_confidence,
            "text_coverage_ratio": text_coverage_ratio,
            "table_confidence": table_confidence,
        },
    )


def test_mean_confidence_rule_triggers_fallback():
    result = _page_result(mean_ocr_confidence=0.4)
    assert "mean_ocr_confidence" in fallback_reasons(result)
    assert needs_fallback(result) is True


def test_text_coverage_rule_triggers_fallback():
    result = _page_result(text_coverage_ratio=0.1)
    assert "text_coverage_ratio" in fallback_reasons(result)
    assert needs_fallback(result) is True


def test_high_quality_page_skips_fallback():
    result = _page_result()
    assert fallback_reasons(result) == []
    assert needs_fallback(result) is False
