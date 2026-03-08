from modal_doc_parsing_vlm.aggregator import aggregate_job
from modal_doc_parsing_vlm.storage import FileSystemStorageBackend
from modal_doc_parsing_vlm.types_result import (
    BoundingBox,
    DebugOptions,
    DocumentElement,
    FileMetadata,
    JobManifest,
    JobProgressSnapshot,
    JobStatus,
    PageError,
    PageParseResult,
    PageResultStatus,
    ParseMode,
)


def make_manifest() -> JobManifest:
    return JobManifest(
        job_id="job-aggregate",
        parser_version="1.0.0",
        schema_version="1.0",
        runtime_profile="dev",
        source_fingerprint="fingerprint",
        request_payload={},
        output_formats=["json", "markdown", "text"],
        debug=DebugOptions(),
        model_id="Qwen/Qwen3-VL-8B-Instruct-FP8",
        pipeline_mode=ParseMode.BALANCED,
        file_metadata=FileMetadata(
            file_name="doc.pdf",
            mime_type="application/pdf",
            pages_total=2,
            bytes=10,
        ),
        pages=[
            {
                "id": 0,
                "image_uri": "jobs/job-aggregate/pages/0/page.png",
                "width": 100,
                "height": 200,
                "rotation": 0,
            },
            {
                "id": 1,
                "image_uri": "jobs/job-aggregate/pages/1/page.png",
                "width": 100,
                "height": 200,
                "rotation": 0,
            },
        ],
    )


def test_aggregate_job_merges_pages_and_errors(tmp_path):
    storage = FileSystemStorageBackend(tmp_path)
    manifest = make_manifest()
    storage.create_job_manifest(manifest)
    storage.set_status(
        manifest.job_id,
        JobProgressSnapshot(
            job_id=manifest.job_id,
            status=JobStatus.AGGREGATING,
            pages_total=2,
            pages_completed=1,
            pages_failed=1,
            timings={"split_ms": 1, "submit_ms": 2, "aggregate_ms": 0, "elapsed_ms": 3},
        ),
    )
    storage.write_page_result(
        PageParseResult(
            job_id=manifest.job_id,
            chunk_id="chunk-0000",
            page_id=0,
            status=PageResultStatus.COMPLETED,
            page_markdown="# Page 1",
            elements=[
                DocumentElement(
                    id="p0-e1",
                    page_id=0,
                    type="heading",
                    content="Page 1",
                    bbox=BoundingBox(coord=[0, 0, 10, 10], page_id=0),
                    order=1,
                )
            ],
        )
    )
    storage.write_page_result(
        PageParseResult(
            job_id=manifest.job_id,
            chunk_id="chunk-0000",
            page_id=1,
            status=PageResultStatus.FAILED,
            error=PageError(page_id=1, code="invalid", message="bad json"),
        )
    )

    result = aggregate_job(storage, manifest.job_id)
    snapshot = storage.get_status(manifest.job_id)

    assert result.derived.document_markdown == "# Page 1"
    assert result.derived.document_text == "Page 1"
    assert len(result.document.elements) == 1
    assert len(result.error_status) == 1
    assert snapshot is not None
    assert snapshot.status == JobStatus.COMPLETED_WITH_ERRORS
