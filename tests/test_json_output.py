from modal_doc_parsing_vlm.json_output import parse_and_normalize_page_output


def test_parse_and_normalize_accepts_list_bbox():
    raw_output = """
    {
      "page_markdown": "# Title",
      "elements": [
        {
          "type": "heading",
          "content": "Title",
          "bbox": [10, 20, 30, 40],
          "order": 1,
          "confidence": 0.98,
          "attributes": {}
        }
      ],
      "notes": []
    }
    """

    parsed, elements = parse_and_normalize_page_output(raw_output, page_id=3)

    assert parsed.page_markdown == "# Title"
    assert len(elements) == 1
    assert elements[0].bbox.coord == [10, 20, 30, 40]
    assert elements[0].bbox.page_id == 3


def test_parse_and_normalize_repairs_truncated_json():
    raw_output = """
    {
      "page_markdown": "# Title",
      "elements": [
        {
          "type": "heading",
          "content": "Title",
          "bbox": [10, 20, 30, 40],
          "order": 1,
          "confidence": 0.98,
          "attributes": {}
        }
      ],
      "notes": []
    """

    parsed, elements = parse_and_normalize_page_output(raw_output, page_id=4)

    assert parsed.page_markdown == "# Title"
    assert len(elements) == 1
    assert elements[0].bbox.coord == [10, 20, 30, 40]
    assert elements[0].bbox.page_id == 4


def test_parse_and_normalize_ignores_extra_root_keys():
    raw_output = """
    {
      "page_markdown": "# Title",
      "elements": [
        {
          "type": "heading",
          "content": "Title",
          "bbox": [10, 20, 30, 40],
          "order": 1,
          "confidence": 0.98,
          "attributes": {}
        }
      ],
      "notes": [],
      "251": "spurious"
    }
    """

    parsed, elements = parse_and_normalize_page_output(raw_output, page_id=5)

    assert parsed.page_markdown == "# Title"
    assert len(elements) == 1
    assert elements[0].bbox.coord == [10, 20, 30, 40]
