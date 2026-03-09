from __future__ import annotations

import concurrent.futures
import subprocess
import time
from typing import Any

import modal

from .extraction_client import (
    build_entity_extraction_chat_request,
    build_extraction_headers,
    extract_chat_completion_json,
)
from .config import (
    CONTROL_PLANE_DEPENDENCIES,
    DEEPGEMM_CACHE_ROOT,
    EXTRACTION_ALLOW_CONCURRENT_INPUTS,
    EXTRACTION_BATCH_MAX_CONTAINERS,
    EXTRACTION_BATCH_MAX_SIZE,
    EXTRACTION_BATCH_WAIT_MS,
    EXTRACTION_BUFFER_CONTAINERS,
    EXTRACTION_ENABLE_DEEPGEMM,
    EXTRACTION_ENGINE_TIMEOUT_SECONDS,
    EXTRACTION_GPU,
    EXTRACTION_MAX_CONTAINERS,
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
from .types_extraction import ExtractionWorkItem, ExtractionWorkResult


_EXTRACTION_JSON_PARSE_MAX_ATTEMPTS = 2


def _compile_deep_gemm() -> None:
    completed = subprocess.run(
        [
            "python3",
            "-m",
            "sglang.compile_deep_gemm",
            "--model-path",
            EXTRACTION_MODEL_ID,
            "--revision",
            EXTRACTION_MODEL_REVISION,
            "--tp",
            "1",
        ],
        check=False,
        text=True,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "DeepGEMM compilation failed: "
            f"{(completed.stderr or completed.stdout).strip()[:400]}"
        )


def _extraction_runtime_env() -> dict[str, str]:
    env = {
        "HF_HOME": str(HF_CACHE_ROOT),
        "HF_HUB_CACHE": str(HF_HUB_CACHE_ROOT),
        "HF_XET_HIGH_PERFORMANCE": "1",
    }
    if EXTRACTION_ENABLE_DEEPGEMM:
        env["SGLANG_ENABLE_JIT_DEEPGEMM"] = "1"
    return env


def _build_extraction_image(hf_cache_volume, deepgemm_cache_volume) -> modal.Image:
    base = (
        modal.Image.from_registry(SGLANG_IMAGE)
        .entrypoint([])
        .uv_pip_install(
            *CONTROL_PLANE_DEPENDENCIES,
            "huggingface-hub==0.36.0",
            "requests==2.32.5",
        )
        .env(_extraction_runtime_env())
    )
    if EXTRACTION_ENABLE_DEEPGEMM:
        base = base.run_function(
            _compile_deep_gemm,
            gpu=EXTRACTION_GPU,
            volumes={
                str(HF_CACHE_ROOT): hf_cache_volume,
                str(DEEPGEMM_CACHE_ROOT): deepgemm_cache_volume,
            },
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


def _server_command() -> list[str]:
    return [
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
    ]


def _call_local_chat_completion(payload: dict[str, Any], *, session_id: str) -> tuple[dict[str, Any], int]:
    import httpx

    started = time.perf_counter()
    response = httpx.post(
        f"http://127.0.0.1:{EXTRACTION_SERVER_PORT}/v1/chat/completions",
        json=payload,
        headers=build_extraction_headers(session_id),
        timeout=60,
    )
    response.raise_for_status()
    return response.json(), int((time.perf_counter() - started) * 1000)


def _call_local_chat_completion_json(
    payload: dict[str, Any],
    *,
    session_id: str,
) -> tuple[dict[str, Any], int]:
    total_inference_ms = 0
    last_exc: Exception | None = None
    for attempt in range(1, _EXTRACTION_JSON_PARSE_MAX_ATTEMPTS + 1):
        response, inference_ms = _call_local_chat_completion(
            payload,
            session_id=session_id,
        )
        total_inference_ms += inference_ms
        try:
            return extract_chat_completion_json(response), total_inference_ms
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            print(
                "[engine:extraction-batch] JSON parse failed "
                f"attempt={attempt}/{_EXTRACTION_JSON_PARSE_MAX_ATTEMPTS} "
                f"session_id={session_id} error={exc!r}"
            )
            if attempt < _EXTRACTION_JSON_PARSE_MAX_ATTEMPTS:
                time.sleep(float(attempt))
    raise RuntimeError(f"Extraction batch response contained invalid JSON after retries: {last_exc!r}")


def create_extraction_engine_cls(
    app: modal.App,
    *,
    hf_cache_volume,
    deepgemm_cache_volume,
    export_module: str,
):
    image = _build_extraction_image(hf_cache_volume, deepgemm_cache_volume)

    @modal.enter()
    def startup(self) -> None:
        print(
            f"[engine:extraction] starting sglang model={EXTRACTION_MODEL_ID} "
            f"revision={EXTRACTION_MODEL_REVISION} gpu={EXTRACTION_GPU} "
            f"deepgemm={EXTRACTION_ENABLE_DEEPGEMM}"
        )
        self.process = subprocess.Popen(_server_command())
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
        max_containers=EXTRACTION_MAX_CONTAINERS,
        buffer_containers=EXTRACTION_BUFFER_CONTAINERS,
        scaledown_window=EXTRACTION_SCALEDOWN_WINDOW_SECONDS,
        volumes={
            str(HF_CACHE_ROOT): hf_cache_volume,
            str(DEEPGEMM_CACHE_ROOT): deepgemm_cache_volume,
        },
    )(raw_cls)
    return server


def create_extraction_batch_engine_cls(
    app: modal.App,
    *,
    hf_cache_volume,
    deepgemm_cache_volume,
    export_module: str,
):
    image = _build_extraction_image(hf_cache_volume, deepgemm_cache_volume)

    @modal.enter()
    def startup(self) -> None:
        print(
            f"[engine:extraction-batch] starting sglang model={EXTRACTION_MODEL_ID} "
            f"revision={EXTRACTION_MODEL_REVISION} gpu={EXTRACTION_GPU} "
            f"deepgemm={EXTRACTION_ENABLE_DEEPGEMM}"
        )
        self.process = subprocess.Popen(_server_command())
        _wait_ready(self.process, timeout=EXTRACTION_ENGINE_TIMEOUT_SECONDS)
        _warmup_server()
        print("[engine:extraction-batch] extraction batch server ready")

    def _extract_one(self, item: ExtractionWorkItem) -> ExtractionWorkResult:
        payload = build_entity_extraction_chat_request(
            entity=item.entity,
            page_text=item.page_text,
            model_id=item.model_id,
            json_schema=item.json_schema,
            max_tokens=item.max_tokens,
        )
        data, inference_ms = _call_local_chat_completion_json(
            payload,
            session_id=item.session_id,
        )
        return ExtractionWorkResult(
            entity_name=item.entity.entity_name,
            page_id=item.page_id,
            data=data,
            inference_ms=inference_ms,
        )

    @modal.batched(max_batch_size=EXTRACTION_BATCH_MAX_SIZE, wait_ms=EXTRACTION_BATCH_WAIT_MS)
    def extract_pages(self, item_payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
        items = [ExtractionWorkItem.model_validate(payload) for payload in item_payloads]
        print(
            f"[engine:extraction-batch] extract_pages "
            f"batch_size={len(items)} max_batch_size={EXTRACTION_BATCH_MAX_SIZE}"
        )
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(len(items), 1)) as executor:
            futures = [executor.submit(self._extract_one, item) for item in items]
            results = [future.result() for future in futures]
        return [result.model_dump(mode="json") for result in results]

    @modal.exit()
    def stop(self) -> None:
        self.process.terminate()
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=10)

    raw_cls = type(
        "ExtractionBatchEngine",
        (),
        {
            "__doc__": f"Batched SGLang extraction engine using {EXTRACTION_MODEL_ID}.",
            "__module__": export_module,
            "startup": startup,
            "_extract_one": _extract_one,
            "extract_pages": extract_pages,
            "stop": stop,
        },
    )
    cls = app.cls(
        image=image,
        gpu=EXTRACTION_GPU,
        timeout=EXTRACTION_ENGINE_TIMEOUT_SECONDS,
        max_containers=EXTRACTION_BATCH_MAX_CONTAINERS,
        scaledown_window=EXTRACTION_SCALEDOWN_WINDOW_SECONDS,
        volumes={
            str(HF_CACHE_ROOT): hf_cache_volume,
            str(DEEPGEMM_CACHE_ROOT): deepgemm_cache_volume,
        },
    )(raw_cls)
    return cls
