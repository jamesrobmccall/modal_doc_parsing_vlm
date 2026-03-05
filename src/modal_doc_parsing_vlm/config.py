from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


APP_NAME = "modal-doc-parsing-vlm"
SCHEMA_VERSION = "1.0"
PARSER_VERSION = "1.2.1"
DEFAULT_RUNTIME_PROFILE = os.environ.get("DOC_PARSE_RUNTIME_PROFILE", "prod")
ENABLED_RUNTIME_PROFILES = tuple(
    profile.strip()
    for profile in os.environ.get("DOC_PARSE_ENABLED_RUNTIME_PROFILES", "prod,dev").split(",")
    if profile.strip()
)

ARTIFACT_ROOT = Path("/artifacts")
HF_CACHE_ROOT = Path("/root/.cache/huggingface")
HF_HUB_CACHE_ROOT = HF_CACHE_ROOT / "hub"
VLLM_CACHE_ROOT = Path("/root/.cache/vllm")
TORCHINDUCTOR_CACHE_ROOT = VLLM_CACHE_ROOT / "torchinductor"
PADDLE_CACHE_ROOT = Path("/root/.paddleocr")

HF_CACHE_VOLUME_NAME = "doc-parse-hf-cache"
VLLM_CACHE_VOLUME_NAME = "doc-parse-vllm-cache"
ARTIFACTS_VOLUME_NAME = "doc-parse-artifacts"
PADDLE_CACHE_VOLUME_NAME = "doc-parse-paddle-cache"
JOB_STATUS_DICT_NAME = "doc-parse-job-status"
IDEMPOTENCY_DICT_NAME = "doc-parse-idempotency"

STARTUP_WARMUP_ENABLED = os.environ.get("DOC_PARSE_STARTUP_WARMUP", "").lower() in {
    "1",
    "true",
    "yes",
}
STARTUP_WARMUP_MAX_TOKENS = int(
    os.environ.get("DOC_PARSE_STARTUP_WARMUP_MAX_TOKENS", "32")
)
ENABLE_ASYNC_REFINEMENT = os.environ.get("DOC_PARSE_ASYNC_REFINEMENT", "1").lower() in {
    "1",
    "true",
    "yes",
}
ENABLE_DEEP_REFINE = os.environ.get("DOC_PARSE_ENABLE_DEEP_REFINE", "").lower() in {
    "1",
    "true",
    "yes",
}

RETENTION_DAYS = 7
DEFAULT_POLL_AFTER_SECONDS = 2
DEFAULT_STATUS_POLL_INTERVAL_SECONDS = 2.0
ORCHESTRATOR_TIMEOUT_SECONDS = 60 * 60
ENGINE_TIMEOUT_SECONDS = 60 * 30
SCALEDOWN_WINDOW_SECONDS = int(
    os.environ.get("DOC_PARSE_FALLBACK_SCALEDOWN_WINDOW_SECONDS", str(60 * 5))
)
OCR_SCALEDOWN_WINDOW_SECONDS = int(
    os.environ.get("DOC_PARSE_OCR_SCALEDOWN_WINDOW_SECONDS", str(60 * 5))
)
OCR_MIN_CONTAINERS = int(os.environ.get("DOC_PARSE_OCR_MIN_CONTAINERS", "0"))
OCR_BUFFER_CONTAINERS = int(os.environ.get("DOC_PARSE_OCR_BUFFER_CONTAINERS", "0"))
OCR_ALLOW_CONCURRENT_INPUTS = int(
    os.environ.get("DOC_PARSE_OCR_ALLOW_CONCURRENT_INPUTS", "1")
)
FALLBACK_MIN_CONTAINERS = int(os.environ.get("DOC_PARSE_FALLBACK_MIN_CONTAINERS", "0"))
FALLBACK_BUFFER_CONTAINERS = int(
    os.environ.get("DOC_PARSE_FALLBACK_BUFFER_CONTAINERS", "0")
)
FALLBACK_ALLOW_CONCURRENT_INPUTS = int(
    os.environ.get("DOC_PARSE_FALLBACK_ALLOW_CONCURRENT_INPUTS", "1")
)
STALE_JOB_TIMEOUT_SECONDS = int(
    os.environ.get("DOC_PARSE_STALE_JOB_TIMEOUT_SECONDS", str(60 * 30))
)
STALE_JOB_SWEEP_SECONDS = int(
    os.environ.get("DOC_PARSE_STALE_JOB_SWEEP_SECONDS", str(60 * 10))
)

CONTROL_PLANE_PYTHON_VERSION = "3.12"
OCR_PYTHON_VERSION = os.environ.get("DOC_PARSE_OCR_PYTHON_VERSION", "3.10")
CUDA_IMAGE = "nvidia/cuda:12.9.0-devel-ubuntu22.04"
VLLM_NIGHTLY_EXTRA_INDEX_URL = "https://wheels.vllm.ai/nightly"
VLLM_UV_EXTRA_OPTIONS = (
    "--torch-backend=cu129 "
    "--index-strategy unsafe-best-match "
    "--prerelease=allow"
)

QWEN35_VLLM_PACKAGE = "vllm"
PADDLE_OCR_ENGINE_NAME = "PP-StructureV3"
PADDLE_OCR_GPU = os.environ.get("DOC_PARSE_OCR_GPU", "A10G")

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
    *CONTROL_PLANE_DEPENDENCIES,
    "huggingface-hub==0.36.0",
    "transformers==4.57.2",
]

OCR_DEPENDENCIES = [
    *CONTROL_PLANE_DEPENDENCIES,
    "numpy==2.2.6",
    "paddlepaddle==3.0.0",
    "paddleocr==2.10.0",
    "opencv-python-headless==4.12.0.88",
]

HEALTH_RESPONSE = {"status": "ok"}
TERMINAL_STATUSES = {
    "completed",
    "completed_fast",
    "completed_final",
    "completed_with_errors",
    "failed",
}
ACTIVE_STATUSES = {
    "queued",
    "splitting",
    "submitting",
    "running",
    "aggregating",
}

MIME_TO_SUFFIX = {
    "application/pdf": ".pdf",
    "image/png": ".png",
    "image/jpeg": ".jpg",
}

SAMPLING_MAX_TOKENS = {
    "balanced": 2048,
    "accurate": 3072,
}

RENDER_DPI = {
    "balanced": 200,
    "accurate": 300,
}
FAST_PROFILE_RENDER_DPI = int(os.environ.get("DOC_PARSE_FAST_PROFILE_RENDER_DPI", "150"))

MAX_PAGES_PER_CHUNK = {
    "balanced": 16,
    "accurate": 8,
}

PROMPT_VERSION = "2026-03-03"

ROUTING_EXTRACTABLE_CHAR_THRESHOLD = int(
    os.environ.get("DOC_PARSE_ROUTING_EXTRACTABLE_CHAR_THRESHOLD", "500")
)
ROUTING_PRINTABLE_RATIO_THRESHOLD = float(
    os.environ.get("DOC_PARSE_ROUTING_PRINTABLE_RATIO_THRESHOLD", "0.9")
)
FALLBACK_MEAN_OCR_CONFIDENCE_THRESHOLD = float(
    os.environ.get("DOC_PARSE_FALLBACK_MEAN_OCR_CONFIDENCE_THRESHOLD", "0.88")
)
FALLBACK_TEXT_COVERAGE_THRESHOLD = float(
    os.environ.get("DOC_PARSE_FALLBACK_TEXT_COVERAGE_THRESHOLD", "0.60")
)
FALLBACK_TABLE_CONFIDENCE_THRESHOLD = float(
    os.environ.get("DOC_PARSE_FALLBACK_TABLE_CONFIDENCE_THRESHOLD", "0.80")
)
FALLBACK_MIN_ELEMENT_COUNT = int(os.environ.get("DOC_PARSE_FALLBACK_MIN_ELEMENT_COUNT", "3"))


@dataclass(frozen=True)
class OcrRuntimeProfile:
    name: str
    engine_name: str
    gpu: str


OCR_RUNTIME_PROFILE = OcrRuntimeProfile(
    name="ocr-fast",
    engine_name=PADDLE_OCR_ENGINE_NAME,
    gpu=PADDLE_OCR_GPU,
)


@dataclass(frozen=True)
class RuntimeProfile:
    name: str
    model_id: str
    gpu: str
    vllm_package: str
    vllm_extra_index_url: str | None = None
    vllm_extra_options: str = ""
    tensor_parallel_size: int = 1
    trust_remote_code: bool = False
    disable_thinking: bool = False
    max_model_len: int = 16384
    enforce_eager: bool = False
    fallback_model_id: str | None = None
    deep_refine_model_id: str | None = None


RUNTIME_PROFILES = {
    "prod": RuntimeProfile(
        name="prod",
        model_id="Qwen/Qwen2.5-VL-7B-Instruct",
        gpu="A10G",
        vllm_package=QWEN35_VLLM_PACKAGE,
        vllm_extra_index_url=VLLM_NIGHTLY_EXTRA_INDEX_URL,
        vllm_extra_options=VLLM_UV_EXTRA_OPTIONS,
        disable_thinking=True,
        max_model_len=8192,
        fallback_model_id="Qwen/Qwen2.5-VL-7B-Instruct",
        deep_refine_model_id="Qwen/Qwen3.5-27B-FP8",
    ),
    "dev": RuntimeProfile(
        name="dev",
        model_id="Qwen/Qwen2.5-VL-7B-Instruct",
        gpu="A10G",
        vllm_package=QWEN35_VLLM_PACKAGE,
        vllm_extra_index_url=VLLM_NIGHTLY_EXTRA_INDEX_URL,
        vllm_extra_options=VLLM_UV_EXTRA_OPTIONS,
        disable_thinking=True,
        enforce_eager=True,
        max_model_len=8192,
        fallback_model_id="Qwen/Qwen2.5-VL-7B-Instruct",
        deep_refine_model_id="Qwen/Qwen3.5-27B-FP8",
    ),
}


def get_runtime_profile(name: str | None = None) -> RuntimeProfile:
    selected = name or DEFAULT_RUNTIME_PROFILE
    try:
        return RUNTIME_PROFILES[selected]
    except KeyError as exc:
        raise ValueError(f"Unsupported runtime profile: {selected}") from exc
