from __future__ import annotations

from .config import PROMPT_VERSION
from .types_result import ParseMode


COMMON_OUTPUT_SPEC = """Return JSON only with this exact shape:
{
  "page_markdown": "markdown string for this page",
  "elements": [
    {
      "type": "text|heading|list_item|table|figure|caption|header|footer|page_number|formula|footnote|unknown",
      "content": "verbatim content for the element",
      "bbox": {
        "coord": [x0, y0, x1, y1],
        "page_id": 0
      },
      "order": 1,
      "confidence": 0.0,
      "attributes": {}
    }
  ],
  "notes": []
}
The bbox coordinates must be integers in image pixel space.
Do not wrap the JSON in markdown fences.
Do not include commentary outside the JSON object."""


BALANCED_TASK = """Parse this document page into structured layout elements.
Preserve reading order.
Capture text faithfully.
Prefer simple element types unless the structure is obvious."""


ACCURATE_TASK = """Parse this document page into high-fidelity structured layout elements.
Preserve reading order.
Separate headings, table-like regions, captions, headers, footers, and page numbers when visible.
Capture text faithfully.
If a region is ambiguous, prefer a conservative element split and mention uncertainty in notes."""


STRICT_JSON_REMINDER = """Your previous answer was not valid JSON.
Retry and output a single valid JSON object only.
Do not add prose, explanations, comments, markdown fences, or trailing text."""


def build_page_prompt(
    mode: ParseMode,
    *,
    page_id: int,
    language_hint: str | None = None,
    strict_json: bool = False,
) -> str:
    task = BALANCED_TASK if mode == ParseMode.BALANCED else ACCURATE_TASK
    language_line = (
        f"Primary language hint: {language_hint}.\n" if language_hint else ""
    )
    strict_line = STRICT_JSON_REMINDER + "\n" if strict_json else ""
    return (
        "<|im_start|>system\n"
        "You are a careful document parser.\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        "<|vision_start|><|image_pad|><|vision_end|>\n"
        f"Prompt version: {PROMPT_VERSION}\n"
        f"Page id: {page_id}\n"
        f"{language_line}"
        f"{task}\n"
        f"{strict_line}"
        f"{COMMON_OUTPUT_SPEC}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
