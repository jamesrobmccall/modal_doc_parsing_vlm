from __future__ import annotations

import re
import time
from html import unescape
from typing import Any

import modal

from .config import (
    ARTIFACT_ROOT,
    ENGINE_TIMEOUT_SECONDS,
    HF_CACHE_ROOT,
    OCR_PYTHON_VERSION,
    OCR_ALLOW_CONCURRENT_INPUTS,
    OCR_BUFFER_CONTAINERS,
    OCR_DEPENDENCIES,
    OCR_MAX_CONTAINERS,
    OCR_MIN_CONTAINERS,
    OCR_RUNTIME_PROFILE,
    OCR_SCALEDOWN_WINDOW_SECONDS,
    OCR_STARTUP_WARMUP_ENABLED,
    PADDLE_CACHE_ROOT,
    PADDLE_GPU_INDEX_URL,
    PADDLE_GPU_PACKAGE,
)
from .json_output import page_error
from .types_result import (
    BoundingBox,
    DocumentElement,
    ElementType,
    LatencyProfile,
    PageChunk,
    PageParseResult,
    PageResultStatus,
    PageTask,
    ParseEngine,
)


_WHITESPACE_RE = re.compile(r"\s+")
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_TABLE_ROW_RE = re.compile(r"<tr[^>]*>(.*?)</tr>", flags=re.IGNORECASE | re.DOTALL)
_TABLE_CELL_RE = re.compile(r"<t[hd][^>]*>(.*?)</t[hd]>", flags=re.IGNORECASE | re.DOTALL)


def build_ocr_image() -> modal.Image:
    return (
        modal.Image.debian_slim(python_version=OCR_PYTHON_VERSION)
        .apt_install("libgl1", "libglib2.0-0")
        .uv_pip_install(*OCR_DEPENDENCIES)
        .run_commands(
            f"python -m pip install {PADDLE_GPU_PACKAGE} -i {PADDLE_GPU_INDEX_URL}",
        )
        .env(
            {
                "HF_HOME": str(HF_CACHE_ROOT),
                "PADDLE_HOME": str(PADDLE_CACHE_ROOT),
                "PADDLEOCR_HOME": str(PADDLE_CACHE_ROOT),
                "FLAGS_use_mkldnn": "0",
            }
        )
        .add_local_python_source("modal_doc_parsing_vlm")
    )


def _bbox_from_quad(quad: list[list[float]], page_id: int) -> BoundingBox:
    xs = [point[0] for point in quad]
    ys = [point[1] for point in quad]
    return BoundingBox(
        coord=[int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))],
        page_id=page_id,
    )


def _element_type_from_layout(layout_type: str) -> ElementType:
    normalized = layout_type.lower()
    if normalized in {"title", "heading"}:
        return ElementType.HEADING
    if normalized in {"list", "list_item", "bullet_list"}:
        return ElementType.LIST_ITEM
    if normalized == "table":
        return ElementType.TABLE
    if normalized in {"figure", "image"}:
        return ElementType.FIGURE
    if normalized == "header":
        return ElementType.HEADER
    if normalized == "footer":
        return ElementType.FOOTER
    if normalized == "page_number":
        return ElementType.PAGE_NUMBER
    if normalized == "formula":
        return ElementType.FORMULA
    if normalized == "caption":
        return ElementType.CAPTION
    return ElementType.TEXT


def _normalize_text(value: str) -> str:
    return _WHITESPACE_RE.sub(" ", value).strip()


def _strip_html(value: str) -> str:
    without_scripts = re.sub(
        r"<(script|style)[^>]*>.*?</\1>",
        "",
        value,
        flags=re.IGNORECASE | re.DOTALL,
    )
    text = _HTML_TAG_RE.sub(" ", without_scripts)
    return _normalize_text(unescape(text))


def _html_table_to_markdown(table_html: str) -> str:
    rows: list[list[str]] = []
    for row_html in _TABLE_ROW_RE.findall(table_html):
        cells = [_strip_html(cell) for cell in _TABLE_CELL_RE.findall(row_html)]
        if any(cell for cell in cells):
            rows.append(cells)
    if not rows:
        return _strip_html(table_html)

    width = max(len(row) for row in rows)
    normalized_rows = [row + [""] * (width - len(row)) for row in rows]
    header = normalized_rows[0]
    body = normalized_rows[1:]
    if not body:
        body = [[""] * width]

    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * width) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in body)
    return "\n".join(lines).strip()


def _extract_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        text = value.strip()
        if "<" in text and ">" in text:
            return _strip_html(text)
        return _normalize_text(text)
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, (list, tuple)):
        parts = [_extract_text(item) for item in value]
        return "\n".join(part for part in parts if part).strip()
    if isinstance(value, dict):
        for key in ("text", "content", "label", "markdown"):
            extracted = _extract_text(value.get(key))
            if extracted:
                return extracted
        parts: list[str] = []
        for key in ("res", "items", "texts", "cells", "rows", "data"):
            extracted = _extract_text(value.get(key))
            if extracted:
                parts.append(extracted)
        if parts:
            return "\n".join(parts).strip()
        fallback = " ".join(_extract_text(v) for v in value.values())
        return _normalize_text(fallback)
    return ""


def _table_markdown_from_res(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        candidate = value.strip()
        if "<table" in candidate.lower():
            return _html_table_to_markdown(candidate)
        return ""
    if isinstance(value, dict):
        for key in ("html", "table_html"):
            candidate = value.get(key)
            if isinstance(candidate, str) and candidate.strip():
                return _html_table_to_markdown(candidate)
        for key in ("markdown", "table_markdown", "md"):
            candidate = value.get(key)
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
        for nested in value.values():
            nested_markdown = _table_markdown_from_res(nested)
            if nested_markdown:
                return nested_markdown
        return ""
    if isinstance(value, (list, tuple)):
        for item in value:
            nested_markdown = _table_markdown_from_res(item)
            if nested_markdown:
                return nested_markdown
    return ""


def _layout_block_content(block_type: str, block: dict[str, Any]) -> str:
    normalized_type = block_type.lower()
    if normalized_type == "table":
        table_markdown = _table_markdown_from_res(block.get("res"))
        if table_markdown:
            return table_markdown
    for key in ("text", "res"):
        extracted = _extract_text(block.get(key))
        if extracted:
            return extracted
    return ""


def _bbox_area(coord: list[int]) -> int:
    x0, y0, x1, y1 = coord
    return max(0, x1 - x0) * max(0, y1 - y0)


def _intersection_area(a: list[int], b: list[int]) -> int:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0
    return (ix1 - ix0) * (iy1 - iy0)


def _boxes_duplicate(a: list[int], b: list[int], *, threshold: float = 0.75) -> bool:
    shared = _intersection_area(a, b)
    if shared <= 0:
        return False
    smallest = min(_bbox_area(a), _bbox_area(b))
    if smallest <= 0:
        return False
    return (shared / smallest) >= threshold


def _element_sort_key(element: DocumentElement) -> tuple[int, int, int]:
    x0, y0, _x1, _y1 = element.bbox.coord
    return (y0, x0, element.order)


def _render_markdown_block(element: DocumentElement) -> str:
    if not element.content.strip():
        return ""
    if element.type == ElementType.TABLE:
        return element.content.strip()

    content = _normalize_text(element.content)
    if not content:
        return ""
    if element.type == ElementType.HEADING:
        level = "##"
        if str(element.attributes.get("layout_type", "")).lower() == "title":
            level = "#"
        return f"{level} {content}"
    if element.type == ElementType.LIST_ITEM:
        return f"- {content}"
    if element.type == ElementType.CAPTION:
        return f"*{content}*"
    if element.type == ElementType.HEADER:
        return f"> [header] {content}"
    if element.type == ElementType.FOOTER:
        return f"> [footer] {content}"
    if element.type == ElementType.PAGE_NUMBER:
        return f"> [page] {content}"
    return content


def _page_markdown_from_elements(elements: list[DocumentElement]) -> str:
    paragraph_types = {ElementType.TEXT, ElementType.UNKNOWN, ElementType.FOOTNOTE}
    blocks: list[str] = []
    paragraph_lines: list[str] = []

    def flush_paragraph() -> None:
        if paragraph_lines:
            blocks.append(" ".join(paragraph_lines))
            paragraph_lines.clear()

    for element in elements:
        block = _render_markdown_block(element)
        if not block:
            continue
        if element.type in paragraph_types:
            paragraph_lines.append(block)
            continue
        flush_paragraph()
        blocks.append(block)
    flush_paragraph()
    return "\n\n".join(blocks).strip()


def _ocr_elements_from_output(ocr_output: list[Any], *, page_id: int) -> list[DocumentElement]:
    elements: list[DocumentElement] = []
    lines: list[Any] = []
    if isinstance(ocr_output, list) and ocr_output:
        first_page = ocr_output[0]
        if isinstance(first_page, list):
            lines = first_page

    for line in lines:
        if not isinstance(line, (list, tuple)) or len(line) < 2:
            continue
        quad = line[0]
        text_info = line[1]
        if (
            not isinstance(text_info, (list, tuple))
            or len(text_info) < 2
            or not text_info[0]
        ):
            continue
        text = _normalize_text(str(text_info[0]))
        if not text:
            continue
        try:
            confidence = float(text_info[1])
        except (TypeError, ValueError):
            confidence = 0.0
        elements.append(
            DocumentElement(
                id=f"p{page_id}-ocr-{len(elements) + 1}",
                page_id=page_id,
                type=ElementType.TEXT,
                content=text,
                bbox=_bbox_from_quad(quad, page_id),
                order=len(elements) + 1,
                confidence=confidence,
                attributes={"source": "ocr"},
            )
        )
    return elements


def _layout_elements_from_output(layout_output: list[Any], *, page_id: int) -> list[DocumentElement]:
    elements: list[DocumentElement] = []
    for block in layout_output or []:
        if not isinstance(block, dict):
            continue
        block_type = str(block.get("type", "")).strip()
        bbox = block.get("bbox")
        if not block_type or bbox is None or not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        try:
            x0, y0, x1, y1 = [int(float(value)) for value in bbox]
        except (TypeError, ValueError):
            continue
        element_type = _element_type_from_layout(block_type)
        content = _layout_block_content(block_type, block)
        if not content and element_type in {
            ElementType.TEXT,
            ElementType.HEADING,
            ElementType.LIST_ITEM,
            ElementType.CAPTION,
        }:
            continue
        score = block.get("score", 0.9)
        try:
            confidence = float(score)
        except (TypeError, ValueError):
            confidence = 0.9
        attributes = {"layout_type": block_type, "source": "layout"}
        if element_type == ElementType.TABLE and content.startswith("|"):
            attributes["table_format"] = "markdown"
        elements.append(
            DocumentElement(
                id=f"p{page_id}-layout-{len(elements) + 1}",
                page_id=page_id,
                type=element_type,
                content=content,
                bbox=BoundingBox(coord=[x0, y0, x1, y1], page_id=page_id),
                order=len(elements) + 1,
                confidence=confidence,
                attributes=attributes,
            )
        )
    return elements


def _merge_layout_and_ocr_elements(
    layout_elements: list[DocumentElement],
    ocr_elements: list[DocumentElement],
) -> list[DocumentElement]:
    merged: list[DocumentElement] = list(layout_elements)
    if not layout_elements:
        merged = list(ocr_elements)
    else:
        layout_boxes = [element.bbox.coord for element in layout_elements]
        for ocr_element in ocr_elements:
            if any(_boxes_duplicate(ocr_element.bbox.coord, box) for box in layout_boxes):
                continue
            merged.append(ocr_element)

    merged.sort(key=_element_sort_key)
    return [
        element.model_copy(update={"order": index})
        for index, element in enumerate(merged, start=1)
    ]


def _coverage_ratio(elements: list[DocumentElement], *, width: int, height: int) -> float:
    if width <= 0 or height <= 0:
        return 0.0
    page_area = width * height
    covered = 0
    for element in elements:
        x0, y0, x1, y1 = element.bbox.coord
        area = max(0, x1 - x0) * max(0, y1 - y0)
        covered += area
    return min(1.0, covered / page_area)


def create_ocr_engine_cls(
    app: modal.App,
    *,
    artifacts_volume,
    hf_cache_volume,
    paddle_cache_volume,
    export_module: str,
):
    image = build_ocr_image()

    @modal.enter()
    def start(self) -> None:
        import numpy as np
        import paddle
        import threading
        from paddleocr import PPStructure, PaddleOCR

        self._volume_lock = threading.Lock()
        self._loaded_artifacts_job_id: str | None = None
        if not paddle.is_compiled_with_cuda():
            raise RuntimeError(
                "PaddleOCR container started without CUDA support. "
                "Install a CUDA-enabled paddlepaddle-gpu wheel."
            )
        self.ocr = PaddleOCR(use_angle_cls=False, lang="en", show_log=False)
        self.layout = PPStructure(show_log=False, lang="en")
        device = paddle.device.get_device()
        if not device.startswith("gpu"):
            raise RuntimeError(
                f"PaddleOCR is not using a GPU device (reported {device!r})."
            )
        if OCR_STARTUP_WARMUP_ENABLED:
            blank = np.full((128, 128, 3), 255, dtype=np.uint8)
            self.ocr.ocr(blank, cls=False)
            self.layout(blank)
        print(f"[engine:ocr] initialized paddle OCR + structure engines device={device}")

    def _ensure_artifacts_loaded(self, task: PageTask) -> None:
        with self._volume_lock:
            if self._loaded_artifacts_job_id != task.job_id:
                artifacts_volume.reload()
                self._loaded_artifacts_job_id = task.job_id

    def _parse_page_task(self, task: PageTask) -> PageParseResult:
        from PIL import Image
        import numpy as np

        started = time.perf_counter()
        try:
            self._ensure_artifacts_loaded(task)
            with self._volume_lock:
                with Image.open(task.image_path) as source_image:
                    image = source_image.convert("RGB")
            image_array = np.array(image)
            image.close()
            ocr_output = []
            ocr_warning: str | None = None
            try:
                ocr_output = self.ocr.ocr(image_array, cls=False)
            except Exception as exc:  # noqa: BLE001
                ocr_warning = str(exc)
                print(
                    f"[engine:ocr] text OCR degraded page_id={task.page_id} "
                    f"error={ocr_warning[:160]}"
                )
            layout_output = []
            layout_warning: str | None = None
            if task.latency_profile != LatencyProfile.FAST:
                try:
                    layout_output = self.layout(image_array)
                except Exception as exc:  # noqa: BLE001
                    layout_warning = str(exc)
                    print(
                        f"[engine:ocr] layout parse degraded page_id={task.page_id} "
                        f"error={layout_warning[:160]}"
                    )

            ocr_elements = _ocr_elements_from_output(ocr_output, page_id=task.page_id)
            layout_elements = _layout_elements_from_output(
                layout_output,
                page_id=task.page_id,
            )
            elements = _merge_layout_and_ocr_elements(layout_elements, ocr_elements)
            page_markdown = _page_markdown_from_elements(elements)
            confidences = [element.confidence for element in elements if element.confidence is not None]
            mean_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            table_confidences = [
                element.confidence
                for element in elements
                if element.type == ElementType.TABLE and element.confidence is not None
            ]
            table_confidence = (
                sum(table_confidences) / len(table_confidences)
                if table_confidences
                else 1.0
            )
            warnings: list[str] = []
            if ocr_warning:
                warnings.append(f"text_ocr_failed: {ocr_warning}")
            if layout_warning:
                warnings.append(f"layout_parse_failed: {layout_warning}")
            return PageParseResult(
                job_id=task.job_id,
                chunk_id=task.chunk_id,
                page_id=task.page_id,
                status=PageResultStatus.COMPLETED,
                page_markdown=page_markdown,
                elements=elements,
                warnings=warnings,
                attempts=1,
                valid_on_first_pass=True,
                inference_ms=int((time.perf_counter() - started) * 1000),
                raw_output_path=task.raw_output_path,
                prompt_path=task.prompt_path,
                result_revision=task.result_revision,
                engine=ParseEngine.PADDLE_OCR,
                confidence_summary={
                    "mean_ocr_confidence": float(mean_confidence),
                    "text_coverage_ratio": float(
                        _coverage_ratio(elements, width=task.width, height=task.height)
                    ),
                    "table_confidence": float(table_confidence),
                },
            )
        except Exception as exc:  # noqa: BLE001
            return PageParseResult(
                job_id=task.job_id,
                chunk_id=task.chunk_id,
                page_id=task.page_id,
                status=PageResultStatus.FAILED,
                error=page_error(
                    task.page_id,
                    code="ocr_failure",
                    message=str(exc),
                    retry_count=0,
                    stage="ocr",
                ),
                result_revision=task.result_revision,
                engine=ParseEngine.PADDLE_OCR,
            )

    @modal.method()
    def warmup(self) -> dict[str, object]:
        import numpy as np

        blank = np.full((128, 128, 3), 255, dtype=np.uint8)
        self.ocr.ocr(blank, cls=False)
        self.layout(blank)
        return {"status": "ok", "engine": OCR_RUNTIME_PROFILE.engine_name}

    @modal.method()
    def parse_page(self, task_payload: dict[str, Any]) -> dict[str, Any]:
        task = PageTask.model_validate(task_payload)
        return self._parse_page_task(task).model_dump(mode="json")

    @modal.method()
    def parse_pages(self, chunk_payload: dict[str, Any]) -> list[dict[str, Any]]:
        chunk = PageChunk.model_validate(chunk_payload)
        print(
            f"[engine:ocr] parse_pages chunk_id={chunk.chunk_id} "
            f"pages={len(chunk.pages)} mode={chunk.mode.value}"
        )
        return [
            self._parse_page_task(task).model_dump(mode="json")
            for task in chunk.pages
        ]

    cls_name = "OcrParserEngine"
    raw_cls = type(
        cls_name,
        (),
        {
            "__doc__": "Fast OCR parser engine using PaddleOCR PP-Structure.",
            "__module__": export_module,
            "start": start,
            "_ensure_artifacts_loaded": _ensure_artifacts_loaded,
            "_parse_page_task": _parse_page_task,
            "warmup": warmup,
            "parse_page": parse_page,
            "parse_pages": parse_pages,
        },
    )
    concurrent_cls = modal.concurrent(max_inputs=OCR_ALLOW_CONCURRENT_INPUTS)(raw_cls)
    cls = app.cls(
        image=image,
        gpu=OCR_RUNTIME_PROFILE.gpu,
        timeout=ENGINE_TIMEOUT_SECONDS,
        min_containers=OCR_MIN_CONTAINERS,
        max_containers=OCR_MAX_CONTAINERS,
        buffer_containers=OCR_BUFFER_CONTAINERS,
        scaledown_window=OCR_SCALEDOWN_WINDOW_SECONDS,
        volumes={
            str(HF_CACHE_ROOT): hf_cache_volume,
            str(ARTIFACT_ROOT): artifacts_volume,
            str(PADDLE_CACHE_ROOT): paddle_cache_volume,
        },
    )(concurrent_cls)
    return cls
