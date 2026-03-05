from __future__ import annotations

from datetime import datetime, timedelta, timezone

from .types_result import JobStatus, PageError


ACTIVE_JOB_STATUSES: set[JobStatus] = {
    JobStatus.QUEUED,
    JobStatus.SPLITTING,
    JobStatus.SUBMITTING,
    JobStatus.RUNNING,
    JobStatus.AGGREGATING,
}


def cleanup_expired_jobs(storage, retention_days: int) -> list[str]:
    cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)
    removed: list[str] = []
    for manifest in storage.iter_job_manifests():
        if manifest.created_at < cutoff:
            storage.remove_job(manifest.job_id)
            removed.append(manifest.job_id)
    return removed


def fail_stale_jobs(storage, stale_after_seconds: int) -> list[str]:
    cutoff = datetime.now(timezone.utc) - timedelta(seconds=stale_after_seconds)
    marked_failed: list[str] = []
    now = datetime.now(timezone.utc)

    for manifest in storage.iter_job_manifests():
        snapshot = storage.get_status(manifest.job_id)
        if snapshot is None:
            continue
        if snapshot.status not in ACTIVE_JOB_STATUSES:
            continue
        if snapshot.updated_at >= cutoff:
            continue

        pages_total = max(snapshot.pages_total, manifest.file_metadata.pages_total)
        pages_terminal = snapshot.pages_completed + snapshot.pages_failed
        unfinished = max(pages_total - pages_terminal, 0)

        error_summary = list(snapshot.error_summary or [])
        error_summary.append(
            PageError(
                page_id=-1,
                code="stale_job_timeout",
                message=(
                    "Marked failed by stale-job watchdog after "
                    f"{stale_after_seconds}s without status updates."
                ),
                retry_count=0,
                stage="watchdog",
            )
        )

        snapshot.status = JobStatus.FAILED
        snapshot.pages_total = pages_total
        snapshot.pages_running = 0
        snapshot.pages_failed = snapshot.pages_failed + unfinished
        snapshot.progress_percent = 100.0 if pages_total else snapshot.progress_percent
        snapshot.pending_refinement_pages = 0
        snapshot.error_summary = error_summary
        snapshot.updated_at = now
        storage.set_status(manifest.job_id, snapshot)

        if manifest.pending_refinement_pages:
            manifest.pending_refinement_pages = []
            storage.write_job_manifest(manifest)

        marked_failed.append(manifest.job_id)

    return marked_failed
