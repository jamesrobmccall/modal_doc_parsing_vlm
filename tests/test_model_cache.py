from pathlib import Path

from modal_doc_parsing_vlm.model_cache import describe_model_cache, hf_model_root


def test_hf_model_root_uses_hub_layout():
    model_root = hf_model_root(
        "Qwen/Qwen3.5-27B-FP8",
        cache_root=Path("/tmp/hf-cache"),
    )

    assert model_root == Path("/tmp/hf-cache/models--Qwen--Qwen3.5-27B-FP8")


def test_describe_model_cache_reports_population(tmp_path):
    model_root = tmp_path / "models--Qwen--Qwen3.5-27B-FP8"
    snapshots_dir = model_root / "snapshots"
    blobs_dir = model_root / "blobs"
    snapshots_dir.mkdir(parents=True)
    blobs_dir.mkdir(parents=True)
    (snapshots_dir / "123").mkdir()
    (blobs_dir / "abc").write_text("blob", encoding="utf-8")

    status = describe_model_cache(
        "Qwen/Qwen3.5-27B-FP8",
        cache_root=tmp_path,
    )

    assert status.model_root == model_root
    assert status.snapshot_count == 1
    assert status.blob_count == 1
    assert status.is_populated is True
