import importlib

import pytest

import modal_doc_parsing_vlm.config as config_module


def _reload_config():
    return importlib.reload(config_module)


def test_runtime_profile_uses_env_backed_fallback_defaults():
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setenv("DOC_PARSE_FALLBACK_MODEL_ID", "Qwen/Test-Qwen3-VL")
        monkeypatch.setenv("DOC_PARSE_FALLBACK_MODEL_REVISION", "rev-123")
        monkeypatch.setenv("DOC_PARSE_FALLBACK_TOKENIZER_REVISION", "tok-456")
        monkeypatch.setenv("DOC_PARSE_FALLBACK_GPU", "L40S")
        monkeypatch.setenv("DOC_PARSE_FALLBACK_GPU_MEMORY_UTILIZATION", "0.65")

        config = _reload_config()
        profile = config.get_runtime_profile("dev")

        assert profile.model_id == "Qwen/Test-Qwen3-VL"
        assert profile.model_revision == "rev-123"
        assert profile.tokenizer_revision == "tok-456"
        assert profile.gpu == "L40S"
        assert profile.trust_remote_code is True
        assert profile.gpu_memory_utilization == 0.65
        assert profile.fallback_model_id == "Qwen/Test-Qwen3-VL"

    _reload_config()


def test_extraction_enable_deepgemm_auto_disables_on_l4():
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setenv("DOC_PARSE_EXTRACTION_GPU", "L4")
        monkeypatch.delenv("DOC_PARSE_EXTRACTION_ENABLE_DEEPGEMM", raising=False)

        config = _reload_config()

        assert config.EXTRACTION_ENABLE_DEEPGEMM is False
        assert config.extraction_gpu_supports_deepgemm("H100:1") is True
        assert config.extraction_gpu_supports_deepgemm("L4") is False

    _reload_config()
