from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


APP_NAME = "modal-doc-parsing-vlm"
SCHEMA_VERSION = "1.0"
PARSER_VERSION = "1.0.0"
DEFAULT_RUNTIME_PROFILE = os.environ.get("DOC_PARSE_RUNTIME_PROFILE", "prod")

ARTIFACT_ROOT = Path("/artifacts")
HF_CACHE_ROOT = Path("/root/.cache/huggingface")
VLLM_CACHE_ROOT = Path("/root/.cache/vllm")

HF_CACHE_VOLUME_NAME = "doc-parse-hf-cache"
VLLM_CACHE_VOLUME_NAME = "doc-parse-vllm-cache"
ARTIFACTS_VOLUME_NAME = "doc-parse-artifacts"
JOB_STATUS_DICT_NAME = "doc-parse-job-status"
IDEMPOTENCY_DICT_NAME = "doc-parse-idempotency"

RETENTION_DAYS = 7
DEFAULT_POLL_AFTER_SECONDS = 2
DEFAULT_STATUS_POLL_INTERVAL_SECONDS = 2.0
ORCHESTRATOR_TIMEOUT_SECONDS = 60 * 60
ENGINE_TIMEOUT_SECONDS = 60 * 30
SCALEDOWN_WINDOW_SECONDS = 60 * 5

CONTROL_PLANE_PYTHON_VERSION = "3.12"
CUDA_IMAGE = "nvidia/cuda:12.9.0-devel-ubuntu22.04"

DEV_VLLM_PACKAGE = "vllm==0.13.0"
PROD_VLLM_PACKAGE = (
    "vllm @ git+https://github.com/vllm-project/vllm.git@"
    "8e1fd5baf0ff272936618bf578533d9aa7080a27"
)

CONTROL_PLANE_DEPENDENCIES = [
    "fastapi==0.121.1",
    "fastmcp==2.13.1",
    "httpx==0.28.1",
    "json-repair==0.54.1",
    "pillow==12.0.0",
    "pydantic==2.12.4",
    "pymupdf==1.26.6",
]

COMMON_VLLM_DEPENDENCIES = [
    "huggingface-hub==1.1.5",
    "json-repair==0.54.1",
    "pillow==12.0.0",
    "pydantic==2.12.4",
    "transformers==4.57.2",
]

HEALTH_RESPONSE = {"status": "ok"}
TERMINAL_STATUSES = {"completed", "completed_with_errors", "failed"}
ACTIVE_STATUSES = {"queued", "splitting", "submitting", "running", "aggregating"}

MIME_TO_SUFFIX = {
    "application/pdf": ".pdf",
    "image/png": ".png",
    "image/jpeg": ".jpg",
}

QWEN_MM_PROCESSOR_KWARGS = {
    "min_pixels": 28 * 28,
    "max_pixels": 1280 * 28 * 28,
    "fps": 1,
}

SAMPLING_MAX_TOKENS = {
    "balanced": 2048,
    "accurate": 3072,
}

RENDER_DPI = {
    "balanced": 200,
    "accurate": 300,
}

MAX_PAGES_PER_CHUNK = {
    "balanced": 16,
    "accurate": 8,
}

PROMPT_VERSION = "2026-03-03"


@dataclass(frozen=True)
class RuntimeProfile:
    name: str
    model_id: str
    gpu: str
    vllm_package: str
    tensor_parallel_size: int = 1
    trust_remote_code: bool = False


RUNTIME_PROFILES = {
    "prod": RuntimeProfile(
        name="prod",
        model_id="Qwen/Qwen3.5-27B-FP8",
        gpu="H100",
        vllm_package=PROD_VLLM_PACKAGE,
    ),
    "dev": RuntimeProfile(
        name="dev",
        model_id="Qwen/Qwen2.5-VL-7B-Instruct",
        gpu="L40S",
        vllm_package=DEV_VLLM_PACKAGE,
    ),
}


def get_runtime_profile(name: str | None = None) -> RuntimeProfile:
    selected = name or DEFAULT_RUNTIME_PROFILE
    try:
        return RUNTIME_PROFILES[selected]
    except KeyError as exc:
        raise ValueError(f"Unsupported runtime profile: {selected}") from exc
