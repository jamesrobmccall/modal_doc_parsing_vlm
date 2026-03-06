from __future__ import annotations

import subprocess
import time
from typing import Any

import modal

from .config import (
    CONTROL_PLANE_DEPENDENCIES,
    EXTRACTION_ALLOW_CONCURRENT_INPUTS,
    EXTRACTION_BUFFER_CONTAINERS,
    EXTRACTION_ENGINE_TIMEOUT_SECONDS,
    EXTRACTION_GPU,
    EXTRACTION_MAX_MODEL_LEN,
    EXTRACTION_MAX_RUNNING_REQUESTS,
    EXTRACTION_MEM_FRACTION,
    EXTRACTION_MIN_CONTAINERS,
    EXTRACTION_MODEL_ID,
    EXTRACTION_MODEL_REVISION,
    EXTRACTION_REGION,
    EXTRACTION_SCALEDOWN_WINDOW_SECONDS,
    EXTRACTION_SERVER_PORT,
    EXTRACTION_TARGET_INPUTS,
    EXTRACTION_WARMUP_REQUESTS,
    HF_CACHE_ROOT,
    HF_HUB_CACHE_ROOT,
    SGLANG_IMAGE,
)


def _build_extraction_image(hf_cache_volume) -> modal.Image:
    base = (
        modal.Image.from_registry(SGLANG_IMAGE)
        .entrypoint([])
        .uv_pip_install(
            *CONTROL_PLANE_DEPENDENCIES,
            "huggingface-hub==0.36.0",
            "requests==2.32.5",
        )
        .env(
            {
                "HF_HOME": str(HF_CACHE_ROOT),
                "HF_HUB_CACHE": str(HF_HUB_CACHE_ROOT),
                "HF_XET_HIGH_PERFORMANCE": "1",
            }
        )
    )
    return base.add_local_python_source("modal_doc_parsing_vlm")


def _check_running(process: subprocess.Popen[str]) -> None:
    if (return_code := process.poll()) is not None:
        raise subprocess.CalledProcessError(return_code, process.args)


def _wait_ready(process: subprocess.Popen[str], *, timeout: int) -> None:
    import requests

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            _check_running(process)
            response = requests.get(
                f"http://127.0.0.1:{EXTRACTION_SERVER_PORT}/health",
                timeout=5,
            )
            response.raise_for_status()
            return
        except (
            subprocess.CalledProcessError,
            requests.exceptions.ConnectionError,
            requests.exceptions.HTTPError,
            requests.exceptions.Timeout,
        ):
            time.sleep(1)
    raise TimeoutError("Extraction server did not become healthy before the startup timeout.")


def _warmup_payload() -> dict[str, Any]:
    return {
        "model": EXTRACTION_MODEL_ID,
        "messages": [
            {
                "role": "system",
                "content": "You return short JSON objects.",
            },
            {
                "role": "user",
                "content": "Return a JSON object with key ok set to true.",
            },
        ],
        "max_tokens": 24,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
        "response_format": {"type": "json_object"},
    }


def _warmup_server() -> None:
    import requests

    for _ in range(EXTRACTION_WARMUP_REQUESTS):
        response = requests.post(
            f"http://127.0.0.1:{EXTRACTION_SERVER_PORT}/v1/chat/completions",
            json=_warmup_payload(),
            timeout=30,
        )
        response.raise_for_status()


def create_extraction_engine_cls(
    app: modal.App,
    *,
    hf_cache_volume,
    export_module: str,
):
    image = _build_extraction_image(hf_cache_volume)

    @modal.enter()
    def startup(self) -> None:
        cmd = [
            "python",
            "-m",
            "sglang.launch_server",
            "--model-path",
            EXTRACTION_MODEL_ID,
            "--revision",
            EXTRACTION_MODEL_REVISION,
            "--served-model-name",
            EXTRACTION_MODEL_ID,
            "--host",
            "0.0.0.0",
            "--port",
            str(EXTRACTION_SERVER_PORT),
            "--tp",
            "1",
            "--context-length",
            str(EXTRACTION_MAX_MODEL_LEN),
            "--max-running-requests",
            str(EXTRACTION_MAX_RUNNING_REQUESTS),
            "--cuda-graph-max-bs",
            str(EXTRACTION_ALLOW_CONCURRENT_INPUTS),
            "--grammar-backend",
            "xgrammar",
            "--enable-metrics",
            "--mem-fraction",
            str(EXTRACTION_MEM_FRACTION),
            "--enable-torch-compile",
            "--chunked-prefill-size",
            "2048",
            "--schedule-policy",
            "fcfs",
        ]
        print(
            f"[engine:extraction] starting sglang model={EXTRACTION_MODEL_ID} "
            f"revision={EXTRACTION_MODEL_REVISION} gpu={EXTRACTION_GPU}"
        )
        self.process = subprocess.Popen(cmd)
        _wait_ready(self.process, timeout=EXTRACTION_ENGINE_TIMEOUT_SECONDS)
        _warmup_server()
        print("[engine:extraction] extraction server ready")

    @modal.exit()
    def stop(self) -> None:
        self.process.terminate()
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=10)

    raw_cls = type(
        "ExtractionEngine",
        (),
        {
            "__doc__": f"SGLang extraction engine using {EXTRACTION_MODEL_ID}.",
            "__module__": export_module,
            "startup": startup,
            "stop": stop,
        },
    )
    server = app._experimental_server(
        image=image,
        gpu=EXTRACTION_GPU,
        startup_timeout=EXTRACTION_ENGINE_TIMEOUT_SECONDS,
        port=EXTRACTION_SERVER_PORT,
        proxy_regions=[EXTRACTION_REGION],
        exit_grace_period=15,
        target_concurrency=EXTRACTION_TARGET_INPUTS,
        region=EXTRACTION_REGION,
        min_containers=EXTRACTION_MIN_CONTAINERS,
        buffer_containers=EXTRACTION_BUFFER_CONTAINERS,
        scaledown_window=EXTRACTION_SCALEDOWN_WINDOW_SECONDS,
        volumes={
            str(HF_CACHE_ROOT): hf_cache_volume,
        },
    )(raw_cls)
    return server
