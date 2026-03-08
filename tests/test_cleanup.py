from datetime import datetime, timedelta, timezone

from modal_doc_parsing_vlm.cleanup import cleanup_expired_jobs, fail_stale_jobs
from modal_doc_parsing_vlm.storage import FileSystemStorageBackend
from modal_doc_parsing_vlm.types_result import (
    DebugOptions,
    FileMetadata,
    JobManifest,
    JobProgressSnapshot,
    JobStatus,
    ParseMode,
)


def _manifest(
    job_id: str,
    *,
    created_at: datetime | None = None,
    pending_refinement_pages: list[int] | None = None,
) -> JobManifest:
    return JobManifest(
        job_id=job_id,
        parser_version="1.0.0",
        schema_version="1.0",
        runtime_profile="dev",
        created_at=created_at or datetime.now(timezone.utc),
        source_fingerprint=f"fp-{job_id}",
        request_payload={},
        output_formats=["json"],
        debug=DebugOptions(),
        model_id="Qwen/Qwen3-VL-8B-Instruct-FP8",
        pipeline_mode=ParseMode.BALANCED,
        file_metadata=FileMetadata(
            file_name=f"{job_id}.png",
            mime_type="image/png",
            pages_total=5,
            bytes=10,
        ),
        pending_refinement_pages=pending_refinement_pages or [],
    )


def test_cleanup_expired_jobs_removes_old_jobs(tmp_path):
    storage = FileSystemStorageBackend(tmp_path)
    old_manifest = _manifest(
        "old-job",
        created_at=datetime.now(timezone.utc) - timedelta(days=10),
    )
    new_manifest = _manifest("new-job")

    storage.create_job_manifest(old_manifest)
    storage.create_job_manifest(new_manifest)

    removed = cleanup_expired_jobs(storage, retention_days=7)

    assert removed == ["old-job"]
    assert storage.list_job_ids() == ["new-job"]


def test_fail_stale_jobs_marks_active_job_failed_and_clears_refinement(tmp_path):
    storage = FileSystemStorageBackend(tmp_path)
    manifest = _manifest("stale-job", pending_refinement_pages=[2, 3])
    storage.create_job_manifest(manifest)
    storage.set_status(
        manifest.job_id,
        JobProgressSnapshot(
            job_id=manifest.job_id,
            status=JobStatus.RUNNING,
            pages_total=5,
            pages_completed=2,
            pages_failed=1,
            pages_running=2,
            updated_at=datetime.now(timezone.utc) - timedelta(hours=2),
        ),
    )

    marked = fail_stale_jobs(storage, stale_after_seconds=30 * 60)
    status = storage.get_status(manifest.job_id)
    updated_manifest = storage.read_job_manifest(manifest.job_id)

    assert marked == [manifest.job_id]
    assert status is not None
    assert status.status == JobStatus.FAILED
    assert status.pages_running == 0
    assert status.pages_failed == 3
    assert status.pending_refinement_pages == 0
    assert status.error_summary is not None
    assert status.error_summary[-1].code == "stale_job_timeout"
    assert updated_manifest.pending_refinement_pages == []


def test_fail_stale_jobs_does_not_touch_terminal_jobs(tmp_path):
    storage = FileSystemStorageBackend(tmp_path)
    manifest = _manifest("done-job")
    storage.create_job_manifest(manifest)
    snapshot = JobProgressSnapshot(
        job_id=manifest.job_id,
        status=JobStatus.COMPLETED_FINAL,
        updated_at=datetime.now(timezone.utc) - timedelta(days=2),
    )
    storage.set_status(manifest.job_id, snapshot)

    marked = fail_stale_jobs(storage, stale_after_seconds=30 * 60)
    status = storage.get_status(manifest.job_id)

    assert marked == []
    assert status == snapshot
