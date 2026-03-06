from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from pathlib import Path

import fitz
from PIL import Image

from .config import RENDER_DPI
from .types_result import MimeType, ParseMode


@dataclass(frozen=True)
class RasterizedPage:
    page_id: int
    image_path: Path
    width: int
    height: int
    rotation: int
    page_hash: str
    extracted_text: str | None = None


def parse_page_range(page_range: str | None, total_pages: int) -> list[int]:
    if page_range is None:
        return list(range(total_pages))

    selected: list[int] = []
    for chunk in page_range.split(","):
        if "-" in chunk:
            start_text, end_text = chunk.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            if start <= 0 or end <= 0 or end < start:
                raise ValueError(f"Invalid page range segment: {chunk}")
            selected.extend(range(start - 1, end))
        else:
            value = int(chunk)
            if value <= 0:
                raise ValueError(f"Invalid page number: {chunk}")
            selected.append(value - 1)

    ordered = []
    seen = set()
    for page_id in selected:
        if page_id >= total_pages:
            raise ValueError(
                f"Requested page {page_id + 1} but document only has {total_pages} pages"
            )
        if page_id not in seen:
            ordered.append(page_id)
            seen.add(page_id)
    return ordered


def apply_max_pages(page_ids: list[int], max_pages: int | None) -> list[int]:
    if max_pages is None:
        return page_ids
    return page_ids[:max_pages]


def _save_png(image: Image.Image, destination: Path) -> bytes:
    destination.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    data = buffer.getvalue()
    destination.write_bytes(data)
    return data


def _hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def rasterize_document(
    *,
    source_bytes: bytes,
    mime_type: MimeType,
    mode: ParseMode,
    output_dir: Path,
    page_range: str | None = None,
    max_pages: int | None = None,
    dpi_override: int | None = None,
) -> list[RasterizedPage]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if mime_type == MimeType.PDF:
        document = fitz.open(stream=source_bytes, filetype="pdf")
        all_page_ids = parse_page_range(page_range, document.page_count)
        selected_page_ids = apply_max_pages(all_page_ids, max_pages)
        dpi = dpi_override or RENDER_DPI[mode.value]
        zoom = dpi / 72.0
        rasterized: list[RasterizedPage] = []
        for page_id in selected_page_ids:
            page = document.load_page(page_id)
            extracted_text = page.get_text("text") or ""
            pixmap = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
            image = Image.open(io.BytesIO(pixmap.tobytes("png"))).convert("RGB")
            image_path = output_dir / str(page_id) / "page.png"
            data = _save_png(image, image_path)
            rasterized.append(
                RasterizedPage(
                    page_id=page_id,
                    image_path=image_path,
                    width=image.width,
                    height=image.height,
                    rotation=page.rotation,
                    page_hash=_hash_bytes(data),
                    extracted_text=extracted_text,
                )
            )
        return rasterized

    image = Image.open(io.BytesIO(source_bytes)).convert("RGB")
    image_path = output_dir / "0" / "page.png"
    data = _save_png(image, image_path)
    return [
        RasterizedPage(
            page_id=0,
            image_path=image_path,
            width=image.width,
            height=image.height,
            rotation=0,
            page_hash=_hash_bytes(data),
            extracted_text=None,
        )
    ]
