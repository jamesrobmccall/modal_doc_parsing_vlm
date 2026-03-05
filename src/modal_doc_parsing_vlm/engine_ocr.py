from __future__ import annotations

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
    OCR_MIN_CONTAINERS,
    OCR_RUNTIME_PROFILE,
    OCR_SCALEDOWN_WINDOW_SECONDS,
    PADDLE_CACHE_ROOT,
)
from .json_output import page_error
from .types_result import (
    BoundingBox,
    DocumentElement,
    ElementType,
    LatencyProfile,
    PageParseResult,
    PageResultStatus,
    PageTask,
    ParseEngine,
)


def build_ocr_image() -> modal.Image:
    return (
        modal.Image.debian_slim(python_version=OCR_PYTHON_VERSION)
        .apt_install("libgl1", "libglib2.0-0")
        .uv_pip_install(*OCR_DEPENDENCIES)
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
        import threading
        from paddleocr import PPStructure, PaddleOCR

        self._volume_lock = threading.Lock()
        self.ocr = PaddleOCR(use_angle_cls=False, lang="en", show_log=False)
        self.layout = PPStructure(show_log=False, lang="en")
        print("[engine:ocr] initialized paddle OCR + structure engines")

    @modal.method()
    def parse_page(self, task_payload: dict[str, Any]) -> dict[str, Any]:
        from PIL import Image
        import numpy as np

        task = PageTask.model_validate(task_payload)
        try:
            with self._volume_lock:
                artifacts_volume.reload()
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

            elements: list[DocumentElement] = []
            order = 1
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
                text = str(text_info[0]).strip()
                confidence = float(text_info[1])
                if not text:
                    continue
                elements.append(
                    DocumentElement(
                        id=f"p{task.page_id}-e{order}",
                        page_id=task.page_id,
                        type=ElementType.TEXT,
                        content=text,
                        bbox=_bbox_from_quad(quad, task.page_id),
                        order=order,
                        confidence=confidence,
                        attributes={},
                    )
                )
                order += 1

            for block in layout_output or []:
                block_type = str(block.get("type", "")).strip()
                bbox = block.get("bbox")
                text = str(block.get("res") or block.get("text") or "").strip()
                if not block_type or bbox is None:
                    continue
                x0, y0, x1, y1 = [int(value) for value in bbox]
                elements.append(
                    DocumentElement(
                        id=f"p{task.page_id}-e{order}",
                        page_id=task.page_id,
                        type=_element_type_from_layout(block_type),
                        content=text,
                        bbox=BoundingBox(coord=[x0, y0, x1, y1], page_id=task.page_id),
                        order=order,
                        confidence=float(block.get("score", 0.9)),
                        attributes={"layout_type": block_type},
                    )
                )
                order += 1

            elements.sort(key=lambda element: element.order)
            page_markdown = "\n\n".join(
                element.content for element in elements if element.content
            ).strip()
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
            result = PageParseResult(
                job_id=task.job_id,
                chunk_id=task.chunk_id,
                page_id=task.page_id,
                status=PageResultStatus.COMPLETED,
                page_markdown=page_markdown,
                elements=elements,
                warnings=warnings,
                attempts=1,
                valid_on_first_pass=True,
                inference_ms=0,
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
            return result.model_dump(mode="json")
        except Exception as exc:  # noqa: BLE001
            failed = PageParseResult(
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
            return failed.model_dump(mode="json")

    cls_name = "OcrParserEngine"
    raw_cls = type(
        cls_name,
        (),
        {
            "__doc__": "Fast OCR parser engine using PaddleOCR PP-Structure.",
            "__module__": export_module,
            "start": start,
            "parse_page": parse_page,
        },
    )
    cls = app.cls(
        image=image,
        gpu=OCR_RUNTIME_PROFILE.gpu,
        timeout=ENGINE_TIMEOUT_SECONDS,
        min_containers=OCR_MIN_CONTAINERS,
        buffer_containers=OCR_BUFFER_CONTAINERS,
        scaledown_window=OCR_SCALEDOWN_WINDOW_SECONDS,
        allow_concurrent_inputs=OCR_ALLOW_CONCURRENT_INPUTS,
        volumes={
            str(HF_CACHE_ROOT): hf_cache_volume,
            str(ARTIFACT_ROOT): artifacts_volume,
            str(PADDLE_CACHE_ROOT): paddle_cache_volume,
        },
    )(raw_cls)
    return cls
