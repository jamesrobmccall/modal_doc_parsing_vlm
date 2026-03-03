from modal_doc_parsing_vlm.chunking import build_chunks, chunk_size_for_mode
from modal_doc_parsing_vlm.rasterize import apply_max_pages, parse_page_range
from modal_doc_parsing_vlm.types_result import DebugOptions, PageTask, ParseMode


def make_task(page_id: int) -> PageTask:
    return PageTask(
        job_id="job-chunking",
        chunk_id="pending",
        page_id=page_id,
        mode=ParseMode.BALANCED,
        image_path=f"/artifacts/jobs/job-chunking/pages/{page_id}/page.png",
        width=100,
        height=200,
        rotation=0,
        page_hash=f"hash-{page_id}",
        task_path=f"/artifacts/jobs/job-chunking/pages/{page_id}/task.json",
        result_path=f"/artifacts/jobs/job-chunking/pages/{page_id}/result.json",
    )


def test_page_range_is_normalized_and_deduplicated():
    assert parse_page_range("1-3,2,5", 6) == [0, 1, 2, 4]
    assert apply_max_pages([0, 1, 2, 3], 2) == [0, 1]


def test_chunk_sizes_follow_mode_defaults():
    assert chunk_size_for_mode(ParseMode.BALANCED) == 16
    assert chunk_size_for_mode(ParseMode.ACCURATE) == 8


def test_build_chunks_is_deterministic():
    chunks = build_chunks(
        job_id="job-chunking",
        mode=ParseMode.ACCURATE,
        parser_version="1.0.0",
        runtime_profile="dev",
        artifact_root="/artifacts",
        model_id="Qwen/Qwen2.5-VL-7B-Instruct",
        language_hint=None,
        debug=DebugOptions(),
        page_tasks=[make_task(page_id) for page_id in range(20)],
    )

    assert [chunk.chunk_id for chunk in chunks] == [
        "chunk-0000",
        "chunk-0001",
        "chunk-0002",
    ]
    assert [len(chunk.pages) for chunk in chunks] == [8, 8, 4]
    assert [task.page_id for task in chunks[0].pages] == list(range(8))
