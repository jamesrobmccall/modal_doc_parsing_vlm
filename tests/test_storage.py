from modal_doc_parsing_vlm.storage import FileSystemStorageBackend, InMemoryKVStore
from modal_doc_parsing_vlm.types_extraction import EntitySuggestionResponse
from modal_doc_parsing_vlm.types_result import (
    DebugOptions,
    DocumentBody,
    DocumentParseResult,
    FileMetadata,
    IdempotencyRecord,
    JobManifest,
    JobProgressSnapshot,
    JobStatus,
    ModelMetadata,
    PageError,
    PageParseResult,
    PageResultStatus,
    ParseMode,
    ResultMetadata,
)


class FakeVolume:
    def __init__(self) -> None:
        self.commits = 0
        self.reloads = 0

    def commit(self) -> None:
        self.commits += 1

    def reload(self) -> None:
        self.reloads += 1


def make_manifest(job_id: str = "job-storage") -> JobManifest:
    return JobManifest(
        job_id=job_id,
        parser_version="1.0.0",
        schema_version="1.0",
        runtime_profile="dev",
        source_fingerprint="fingerprint",
        request_payload={},
        output_formats=["json", "markdown", "text"],
        debug=DebugOptions(),
        model_id="Qwen/Qwen2.5-VL-7B-Instruct",
        pipeline_mode=ParseMode.BALANCED,
        file_metadata=FileMetadata(
            file_name="doc.png",
            mime_type="image/png",
            pages_total=1,
            bytes=4,
        ),
    )


def make_result(job_id: str = "job-storage") -> DocumentParseResult:
    return DocumentParseResult(
        document=DocumentBody(),
        derived={"document_markdown": "", "document_text": "", "page_markdown": {}},
        metadata=ResultMetadata(
            job_id=job_id,
            schema_version="1.0",
            pipeline_mode=ParseMode.BALANCED,
            models=ModelMetadata(page_vlm="Qwen/Qwen2.5-VL-7B-Instruct"),
            file_metadata=FileMetadata(
                file_name="doc.png",
                mime_type="image/png",
                pages_total=1,
                bytes=4,
            ),
            timings={"split_ms": 1, "submit_ms": 2, "aggregate_ms": 3, "elapsed_ms": 4},
        ),
    )


def test_storage_round_trip_status_and_idempotency(tmp_path):
    storage = FileSystemStorageBackend(
        tmp_path,
        status_store=InMemoryKVStore(),
        idempotency_store=InMemoryKVStore(),
    )
    manifest = make_manifest()
    storage.create_job_manifest(manifest)
    storage.write_source_bytes(manifest.job_id, b"data")

    snapshot = JobProgressSnapshot(job_id=manifest.job_id, status=JobStatus.QUEUED)
    storage.set_status(manifest.job_id, snapshot)
    storage.store_idempotency(
        IdempotencyRecord(
            request_fingerprint="fingerprint",
            job_id=manifest.job_id,
            parser_version="1.0.0",
        )
    )

    loaded_manifest = storage.read_job_manifest(manifest.job_id)
    loaded_status = storage.get_status(manifest.job_id)
    loaded_record = storage.lookup_idempotency("fingerprint")

    assert loaded_manifest.job_id == manifest.job_id
    assert storage.read_source_bytes(manifest.job_id) == b"data"
    assert loaded_status is not None
    assert loaded_status.status == JobStatus.QUEUED
    assert loaded_record is not None
    assert loaded_record.job_id == manifest.job_id


def test_storage_round_trip_page_result_and_final_result(tmp_path):
    storage = FileSystemStorageBackend(tmp_path)
    manifest = make_manifest()
    storage.create_job_manifest(manifest)
    page_result = PageParseResult(
        job_id=manifest.job_id,
        chunk_id="chunk-0000",
        page_id=0,
        status=PageResultStatus.FAILED,
        error=PageError(page_id=0, code="x", message="broken"),
    )
    storage.write_page_result(page_result)
    storage.write_final_result(manifest.job_id, make_result(), "markdown", "text")

    assert storage.read_page_result(manifest.job_id, 0) == page_result
    assert storage.read_final_result(manifest.job_id).metadata.job_id == manifest.job_id
    assert storage.read_result_text(manifest.job_id, "markdown") == "markdown"
    assert storage.read_result_text(manifest.job_id, "text") == "text"


def test_list_page_results_skips_transient_invalid_json(tmp_path):
    storage = FileSystemStorageBackend(tmp_path)
    manifest = make_manifest()
    storage.create_job_manifest(manifest)
    page_dir = storage.job_dir(manifest.job_id) / "pages" / "0"
    page_dir.mkdir(parents=True, exist_ok=True)
    (page_dir / "result.json").write_text("\x00\x00\x00\x00", encoding="utf-8")

    assert storage.list_page_results(manifest.job_id) == []


def test_storage_batch_commits_once_for_multiple_writes(tmp_path):
    volume = FakeVolume()
    storage = FileSystemStorageBackend(tmp_path, volume=volume)
    manifest = make_manifest()

    with storage.batch():
        storage.create_job_manifest(manifest)
        storage.write_source_bytes(manifest.job_id, b"data")
        storage.write_extraction_suggestion(
            manifest.job_id,
            EntitySuggestionResponse(job_id=manifest.job_id, suggested_entities=[]),
        )

    assert volume.commits == 1
