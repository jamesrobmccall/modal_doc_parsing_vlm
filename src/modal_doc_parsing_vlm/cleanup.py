from __future__ import annotations

from datetime import datetime, timedelta, timezone


def cleanup_expired_jobs(storage, retention_days: int) -> list[str]:
    cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)
    removed: list[str] = []
    for manifest in storage.iter_job_manifests():
        if manifest.created_at < cutoff:
            storage.remove_job(manifest.job_id)
            removed.append(manifest.job_id)
    return removed
