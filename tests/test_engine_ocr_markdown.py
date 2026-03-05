from modal_doc_parsing_vlm.engine_ocr import (
    _html_table_to_markdown,
    _merge_layout_and_ocr_elements,
    _page_markdown_from_elements,
)
from modal_doc_parsing_vlm.types_result import BoundingBox, DocumentElement, ElementType


def _element(
    *,
    element_id: str,
    page_id: int,
    element_type: ElementType,
    content: str,
    bbox: list[int],
    order: int,
    source: str,
) -> DocumentElement:
    return DocumentElement(
        id=element_id,
        page_id=page_id,
        type=element_type,
        content=content,
        bbox=BoundingBox(coord=bbox, page_id=page_id),
        order=order,
        confidence=0.95,
        attributes={"source": source},
    )


def test_html_table_to_markdown_converts_table_html():
    html = """
    <table>
      <tr><th>Item</th><th>Qty</th></tr>
      <tr><td>Pen</td><td>2</td></tr>
    </table>
    """
    markdown = _html_table_to_markdown(html)

    assert "| Item | Qty |" in markdown
    assert "| Pen | 2 |" in markdown


def test_page_markdown_from_elements_keeps_structure():
    elements = [
        _element(
            element_id="h1",
            page_id=0,
            element_type=ElementType.HEADING,
            content="Invoice",
            bbox=[0, 0, 200, 40],
            order=1,
            source="layout",
        ),
        _element(
            element_id="p1",
            page_id=0,
            element_type=ElementType.TEXT,
            content="First sentence.",
            bbox=[0, 50, 200, 70],
            order=2,
            source="ocr",
        ),
        _element(
            element_id="p2",
            page_id=0,
            element_type=ElementType.TEXT,
            content="Second sentence.",
            bbox=[0, 75, 200, 95],
            order=3,
            source="ocr",
        ),
        _element(
            element_id="t1",
            page_id=0,
            element_type=ElementType.TABLE,
            content="| Item | Qty |\n| --- | --- |\n| Pen | 2 |",
            bbox=[0, 100, 300, 200],
            order=4,
            source="layout",
        ),
        _element(
            element_id="c1",
            page_id=0,
            element_type=ElementType.CAPTION,
            content="Totals",
            bbox=[0, 205, 300, 225],
            order=5,
            source="layout",
        ),
    ]

    markdown = _page_markdown_from_elements(elements)

    assert "## Invoice" in markdown
    assert "First sentence. Second sentence." in markdown
    assert "| Item | Qty |" in markdown
    assert "*Totals*" in markdown


def test_merge_layout_and_ocr_elements_drops_overlapping_duplicate_lines():
    layout_elements = [
        _element(
            element_id="layout-text",
            page_id=0,
            element_type=ElementType.TEXT,
            content="Layout line",
            bbox=[10, 10, 310, 60],
            order=1,
            source="layout",
        )
    ]
    ocr_elements = [
        _element(
            element_id="ocr-dup",
            page_id=0,
            element_type=ElementType.TEXT,
            content="Layout line",
            bbox=[15, 15, 305, 55],
            order=1,
            source="ocr",
        ),
        _element(
            element_id="ocr-unique",
            page_id=0,
            element_type=ElementType.TEXT,
            content="Outside line",
            bbox=[10, 80, 310, 120],
            order=2,
            source="ocr",
        ),
    ]

    merged = _merge_layout_and_ocr_elements(layout_elements, ocr_elements)
    contents = [element.content for element in merged]

    assert contents == ["Layout line", "Outside line"]
    assert [element.order for element in merged] == [1, 2]
