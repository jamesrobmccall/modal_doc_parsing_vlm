from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import modal

from .config import (
    COMMON_VLLM_DEPENDENCIES,
    CUDA_IMAGE,
    ENGINE_TIMEOUT_SECONDS,
    FALLBACK_ALLOW_CONCURRENT_INPUTS,
    FALLBACK_BUFFER_CONTAINERS,
    FALLBACK_MIN_CONTAINERS,
    HF_CACHE_ROOT,
    HF_HUB_CACHE_ROOT,
    SAMPLING_MAX_TOKENS,
    SCALEDOWN_WINDOW_SECONDS,
    STARTUP_WARMUP_ENABLED,
    STARTUP_WARMUP_MAX_TOKENS,
    TORCHINDUCTOR_CACHE_ROOT,
    VLLM_CACHE_ROOT,
)
from .json_output import page_error, parse_and_normalize_page_output
from .model_cache import describe_model_cache
from .prompts import build_page_prompt
from .storage import FileSystemStorageBackend
from .types_result import (
    ChunkParseSummary,
    PageChunk,
    PageParseResult,
    PageResultStatus,
    PageTask,
    ParseEngine,
    ParseMode,
)


def build_vllm_image(runtime_profile) -> modal.Image:
    image = (
        modal.Image.from_registry(CUDA_IMAGE, add_python="3.12")
        .entrypoint([])
        .apt_install("git")
        .uv_pip_install(*COMMON_VLLM_DEPENDENCIES)
    )

    install_kwargs: dict[str, Any] = {}
    if runtime_profile.vllm_extra_index_url:
        install_kwargs["extra_index_url"] = runtime_profile.vllm_extra_index_url
    if runtime_profile.vllm_extra_options:
        install_kwargs["extra_options"] = runtime_profile.vllm_extra_options

    return (
        image.uv_pip_install(runtime_profile.vllm_package, **install_kwargs)
        .env(
            {
                "HF_HOME": str(HF_CACHE_ROOT),
                "HF_HUB_CACHE": str(HF_HUB_CACHE_ROOT),
                "TORCHINDUCTOR_CACHE_DIR": str(TORCHINDUCTOR_CACHE_ROOT),
                "HF_XET_HIGH_PERFORMANCE": "1",
            }
        )
        .add_local_python_source("modal_doc_parsing_vlm")
    )


def _prompt_to_messages(prompt: str, image) -> list[dict[str, Any]]:
    return [
        {
            "role": "system",
            "content": "You are a careful document parser. Return a single JSON object only.",
        },
        {
            "role": "user",
            "content": [
                {"type": "image_pil", "image_pil": image},
                {"type": "text", "text": prompt},
            ],
        },
    ]


def _preview_text(value: str, *, limit: int = 240) -> str:
    compact = " ".join(value.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def _extract_output_text(output: Any) -> str:
    if hasattr(output, "outputs") and output.outputs:
        return output.outputs[0].text.strip()
    if isinstance(output, str):
        return output.strip()
    raise TypeError(f"Unsupported vLLM output type: {type(output)!r}")


def _mean_confidence(elements) -> float:
    values = [element.confidence for element in elements if element.confidence is not None]
    if not values:
        return 0.0
    return sum(values) / len(values)


def create_engine_cls(
    app: modal.App,
    *,
    runtime_profile,
    hf_cache_volume,
    vllm_cache_volume,
    artifacts_volume,
    export_module: str,
):
    image = build_vllm_image(runtime_profile)

    @modal.enter()
    def start(self) -> None:
        import threading
        from PIL import Image
        from vllm import LLM

        self._volume_lock = threading.Lock()
        self.runtime_profile_name = runtime_profile.name
        self.model_id = runtime_profile.model_id
        self.disable_thinking = runtime_profile.disable_thinking
        cache_status = describe_model_cache(self.model_id)
        print(
            f"[engine:fallback:{self.runtime_profile_name}] loading model={self.model_id} "
            f"gpu={runtime_profile.gpu} disable_thinking={self.disable_thinking} "
            f"enforce_eager={runtime_profile.enforce_eager} "
            f"cache_populated={cache_status.is_populated} "
            f"snapshots={cache_status.snapshot_count} blobs={cache_status.blob_count}"
        )
        llm_kwargs = {
            "model": runtime_profile.model_id,
            "tensor_parallel_size": runtime_profile.tensor_parallel_size,
            "max_model_len": runtime_profile.max_model_len,
            "enforce_eager": runtime_profile.enforce_eager,
            "download_dir": str(HF_HUB_CACHE_ROOT),
        }
        if runtime_profile.trust_remote_code:
            llm_kwargs["trust_remote_code"] = True

        self.llm = LLM(**llm_kwargs)
        self.chat_template_kwargs = (
            {"enable_thinking": False} if runtime_profile.disable_thinking else None
        )
        if STARTUP_WARMUP_ENABLED:
            from vllm import SamplingParams

            print(
                f"[engine:fallback:{self.runtime_profile_name}] running optional startup warmup "
                f"max_tokens={STARTUP_WARMUP_MAX_TOKENS}"
            )
            blank = Image.new("RGB", (128, 128), color="white")
            warmup_prompt = build_page_prompt(
                mode=ParseMode.BALANCED,
                page_id=0,
                strict_json=True,
            )
            self.llm.chat(
                [_prompt_to_messages(warmup_prompt, blank)],
                sampling_params=SamplingParams(
                    temperature=0.0,
                    max_tokens=STARTUP_WARMUP_MAX_TOKENS,
                ),
                chat_template_content_format="openai",
                chat_template_kwargs=self.chat_template_kwargs,
                use_tqdm=False,
            )
            blank.close()
            print(
                f"[engine:fallback:{self.runtime_profile_name}] optional startup warmup complete"
            )
        else:
            print(
                f"[engine:fallback:{self.runtime_profile_name}] model initialized; "
                "skipping optional startup warmup"
            )

    def _sampling_params_for_mode(self, mode: ParseMode):
        from vllm import SamplingParams

        return SamplingParams(
            temperature=0.0,
            max_tokens=SAMPLING_MAX_TOKENS[mode.value],
        )

    @modal.method()
    def parse_page(self, task_payload: dict[str, Any]) -> dict[str, Any]:
        from PIL import Image

        task = PageTask.model_validate(task_payload)
        prompt = build_page_prompt(
            mode=task.mode,
            page_id=task.page_id,
            strict_json=False,
        )
        with self._volume_lock:
            artifacts_volume.reload()
            with Image.open(task.image_path) as source_image:
                image = source_image.convert("RGB")
        started = time.perf_counter()
        try:
            outputs = self.llm.chat(
                [_prompt_to_messages(prompt, image)],
                sampling_params=self._sampling_params_for_mode(task.mode),
                chat_template_content_format="openai",
                chat_template_kwargs=self.chat_template_kwargs,
                use_tqdm=False,
            )
            text = _extract_output_text(outputs[0])
            parsed, elements = parse_and_normalize_page_output(text, task.page_id)
            result = PageParseResult(
                job_id=task.job_id,
                chunk_id=task.chunk_id,
                page_id=task.page_id,
                status=PageResultStatus.COMPLETED,
                page_markdown=parsed.page_markdown.strip(),
                elements=elements,
                attempts=1,
                valid_on_first_pass=True,
                warnings=parsed.notes,
                inference_ms=int((time.perf_counter() - started) * 1000),
                raw_output_path=task.raw_output_path,
                prompt_path=task.prompt_path,
                result_revision=task.result_revision,
                engine=ParseEngine.VLM_FALLBACK,
                fallback_triggered=True,
                confidence_summary={
                    "mean_ocr_confidence": float(_mean_confidence(elements)),
                    "text_coverage_ratio": 1.0,
                    "table_confidence": float(
                        _mean_confidence(
                            [element for element in elements if element.type.value == "table"]
                        )
                        if any(element.type.value == "table" for element in elements)
                        else 1.0
                    ),
                },
            )
            if task.prompt_path:
                Path(task.prompt_path).write_text(prompt, encoding="utf-8")
            if task.raw_output_path:
                Path(task.raw_output_path).write_text(text, encoding="utf-8")
            artifacts_volume.commit()
            return result.model_dump(mode="json")
        except Exception as exc:  # noqa: BLE001
            print(
                f"[engine:fallback:{self.runtime_profile_name}] page_parse failure "
                f"page_id={task.page_id} error={_preview_text(str(exc))}"
            )
            failed = PageParseResult(
                job_id=task.job_id,
                chunk_id=task.chunk_id,
                page_id=task.page_id,
                status=PageResultStatus.FAILED,
                error=page_error(
                    task.page_id,
                    code="invalid_model_output",
                    message=str(exc),
                    retry_count=0,
                    stage="page_parse",
                ),
                result_revision=task.result_revision,
                engine=ParseEngine.VLM_FALLBACK,
                fallback_triggered=True,
            )
            return failed.model_dump(mode="json")
        finally:
            image.close()

    @modal.method()
    def parse_chunk(self, chunk_payload: dict[str, Any]) -> dict[str, Any]:
        chunk = PageChunk.model_validate(chunk_payload)
        storage = FileSystemStorageBackend(Path(chunk.artifact_root))
        completed = 0
        failed = 0
        for task in chunk.pages:
            result = PageParseResult.model_validate(self.parse_page(task.model_dump(mode="json")))
            storage.write_page_result(result)
            if result.status == PageResultStatus.COMPLETED:
                completed += 1
            else:
                failed += 1
        summary = ChunkParseSummary(
            job_id=chunk.job_id,
            chunk_id=chunk.chunk_id,
            pages_total=len(chunk.pages),
            pages_completed=completed,
            pages_failed=failed,
        )
        storage.write_chunk_summary(summary)
        return summary.model_dump(mode="json")

    @modal.exit()
    def stop(self) -> None:
        del self.llm

    cls_name = f"{runtime_profile.name.title()}FallbackEngine"
    raw_cls = type(
        cls_name,
        (),
        {
            "__doc__": f"{runtime_profile.name} fallback parser using {runtime_profile.model_id}.",
            "__module__": export_module,
            "start": start,
            "_sampling_params_for_mode": _sampling_params_for_mode,
            "parse_page": parse_page,
            "parse_chunk": parse_chunk,
            "stop": stop,
        },
    )
    concurrent_cls = modal.concurrent(max_inputs=FALLBACK_ALLOW_CONCURRENT_INPUTS)(raw_cls)
    cls = app.cls(
        image=image,
        gpu=runtime_profile.gpu,
        timeout=ENGINE_TIMEOUT_SECONDS,
        min_containers=FALLBACK_MIN_CONTAINERS,
        buffer_containers=FALLBACK_BUFFER_CONTAINERS,
        scaledown_window=SCALEDOWN_WINDOW_SECONDS,
        volumes={
            str(HF_CACHE_ROOT): hf_cache_volume,
            str(VLLM_CACHE_ROOT): vllm_cache_volume,
            "/artifacts": artifacts_volume,
        },
    )(concurrent_cls)
    return cls
