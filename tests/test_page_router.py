from modal_doc_parsing_vlm.page_router import classify_page
from modal_doc_parsing_vlm.types_result import ParseEngine


def test_digital_page_routes_to_text_layer():
    decision = classify_page(
        page_id=0,
        extracted_text="A" * 600,
    )
    assert decision.engine == ParseEngine.DIGITAL_TEXT
    assert decision.extractable_char_count == 600
    assert decision.printable_ratio == 1.0


def test_scanned_like_page_routes_to_ocr():
    decision = classify_page(
        page_id=1,
        extracted_text="",
    )
    assert decision.engine == ParseEngine.PADDLE_OCR
    assert decision.extractable_char_count == 0
