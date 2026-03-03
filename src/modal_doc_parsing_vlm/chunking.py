from __future__ import annotations

from .config import MAX_PAGES_PER_CHUNK
from .types_result import DebugOptions, PageChunk, PageTask, ParseMode


def chunk_size_for_mode(mode: ParseMode) -> int:
    return MAX_PAGES_PER_CHUNK[mode.value]


def build_chunks(
    *,
    job_id: str,
    mode: ParseMode,
    parser_version: str,
    runtime_profile: str,
    artifact_root: str,
    model_id: str,
    language_hint: str | None,
    debug: DebugOptions,
    page_tasks: list[PageTask],
) -> list[PageChunk]:
    page_tasks = sorted(page_tasks, key=lambda task: task.page_id)
    size = chunk_size_for_mode(mode)
    chunks: list[PageChunk] = []
    for index in range(0, len(page_tasks), size):
        tasks = page_tasks[index : index + size]
        chunk_id = f"chunk-{index // size:04d}"
        chunks.append(
            PageChunk(
                job_id=job_id,
                chunk_id=chunk_id,
                mode=mode,
                parser_version=parser_version,
                runtime_profile=runtime_profile,
                artifact_root=artifact_root,
                model_id=model_id,
                language_hint=language_hint,
                debug=debug,
                pages=[
                    task.model_copy(update={"chunk_id": chunk_id})
                    for task in tasks
                ],
            )
        )
    return chunks
