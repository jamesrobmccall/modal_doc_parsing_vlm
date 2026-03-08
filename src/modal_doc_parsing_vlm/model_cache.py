from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import HF_HUB_CACHE_ROOT


@dataclass(frozen=True)
class ModelCacheStatus:
    model_id: str
    model_root: Path
    snapshot_count: int
    blob_count: int

    @property
    def is_populated(self) -> bool:
        return self.snapshot_count > 0 and self.blob_count > 0


def hf_model_root(model_id: str, *, cache_root: Path = HF_HUB_CACHE_ROOT) -> Path:
    return cache_root / f"models--{model_id.replace('/', '--')}"


def describe_model_cache(
    model_id: str,
    *,
    cache_root: Path = HF_HUB_CACHE_ROOT,
) -> ModelCacheStatus:
    model_root = hf_model_root(model_id, cache_root=cache_root)
    snapshots_dir = model_root / "snapshots"
    blobs_dir = model_root / "blobs"
    snapshot_count = sum(1 for child in snapshots_dir.iterdir()) if snapshots_dir.exists() else 0
    blob_count = sum(1 for child in blobs_dir.iterdir()) if blobs_dir.exists() else 0
    return ModelCacheStatus(
        model_id=model_id,
        model_root=model_root,
        snapshot_count=snapshot_count,
        blob_count=blob_count,
    )


def ensure_model_cached(
    model_id: str,
    *,
    revision: str | None = None,
    cache_root: Path = HF_HUB_CACHE_ROOT,
) -> ModelCacheStatus:
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id=model_id, revision=revision, cache_dir=str(cache_root))
    return describe_model_cache(model_id, cache_root=cache_root)
