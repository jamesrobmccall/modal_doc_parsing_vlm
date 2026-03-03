from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import modal

from .config import (
    COMMON_VLLM_DEPENDENCIES,
    CUDA_IMAGE,
    ENGINE_TIMEOUT_SECONDS,
    HF_CACHE_ROOT,
    QWEN_MM_PROCESSOR_KWARGS,
    SAMPLING_MAX_TOKENS,
    SCALEDOWN_WINDOW_SECONDS,
    VLLM_CACHE_ROOT,
)
from .json_output import page_error, parse_and_normalize_page_output
from .prompts import build_page_prompt
from .storage import FileSystemStorageBackend
from .types_result import (
    ChunkParseSummary,
    PageChunk,
    PageParseResult,
    PageResultStatus,
    ParseMode,
)


def build_vllm_image(runtime_profile) -> modal.Image:
    return (
        modal.Image.from_registry(CUDA_IMAGE, add_python="3.12")
        .entrypoint([])
        .uv_pip_install(runtime_profile.vllm_package, *COMMON_VLLM_DEPENDENCIES)
        .env({"HF_XET_HIGH_PERFORMANCE": "1"})
    )


def _prompt_to_input(task, prompt: str, image) -> dict[str, Any]:
    return {
        "prompt": prompt,
        "multi_modal_data": {"image": image},
        "multi_modal_uuids": {"image": task.page_hash},
    }


def create_engine_cls(
    app: modal.App,
    *,
    runtime_profile,
    hf_cache_volume,
    vllm_cache_volume,
    artifacts_volume,
):
    image = build_vllm_image(runtime_profile)

    @modal.enter()
    def start(self) -> None:
        from PIL import Image
        from vllm import LLM, SamplingParams

        self.runtime_profile_name = runtime_profile.name
        self.model_id = runtime_profile.model_id
        llm_kwargs = {
            "model": runtime_profile.model_id,
            "tensor_parallel_size": runtime_profile.tensor_parallel_size,
            "max_model_len": 16384,
            "limit_mm_per_prompt": {"image": 1, "video": 0, "audio": 0},
            "attention_backend": "flashinfer",
            "async_scheduling": True,
            "mm_processor_kwargs": QWEN_MM_PROCESSOR_KWARGS,
        }
        if runtime_profile.trust_remote_code:
            llm_kwargs["trust_remote_code"] = True

        self.llm = LLM(**llm_kwargs)
        self.default_sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=SAMPLING_MAX_TOKENS["balanced"],
        )
        blank = Image.new("RGB", (128, 128), color="white")
        warmup_prompt = build_page_prompt(
            mode=ParseMode.BALANCED,
            page_id=0,
            strict_json=False,
        )
        self.llm.generate(
            [_prompt_to_input(type("Task", (), {"page_hash": "warmup"}), warmup_prompt, blank)],
            sampling_params=self.default_sampling_params,
        )

    def _sampling_params_for_chunk(self, chunk: PageChunk):
        from vllm import SamplingParams

        return SamplingParams(
            temperature=0.0,
            max_tokens=SAMPLING_MAX_TOKENS[chunk.mode.value],
        )

    def _attempt_parse(self, chunk: PageChunk, tasks, *, strict_json: bool) -> tuple[list[PageParseResult], list[tuple[Any, str, str]]]:
        from PIL import Image

        sampling_params = self._sampling_params_for_chunk(chunk)
        prompts: dict[int, str] = {}
        inputs = []
        opened_images = []
        for task in tasks:
            prompt = build_page_prompt(
                mode=chunk.mode,
                page_id=task.page_id,
                language_hint=chunk.language_hint,
                strict_json=strict_json,
            )
            prompts[task.page_id] = prompt
            image = Image.open(task.image_path).convert("RGB")
            opened_images.append(image)
            inputs.append(_prompt_to_input(task, prompt, image))

        started = time.perf_counter()
        outputs = self.llm.generate(inputs, sampling_params=sampling_params)
        inference_ms = int((time.perf_counter() - started) * 1000)

        successes: list[PageParseResult] = []
        failures: list[tuple[Any, str, str]] = []
        for task, output in zip(tasks, outputs):
            text = output.outputs[0].text.strip()
            if task.prompt_path:
                Path(task.prompt_path).write_text(prompts[task.page_id], encoding="utf-8")
            if task.raw_output_path:
                Path(task.raw_output_path).write_text(text, encoding="utf-8")
            try:
                parsed, elements = parse_and_normalize_page_output(text, task.page_id)
                successes.append(
                    PageParseResult(
                        job_id=chunk.job_id,
                        chunk_id=chunk.chunk_id,
                        page_id=task.page_id,
                        status=PageResultStatus.COMPLETED,
                        page_markdown=parsed.page_markdown.strip(),
                        elements=elements,
                        attempts=1,
                        valid_on_first_pass=not strict_json,
                        warnings=parsed.notes,
                        inference_ms=inference_ms,
                        raw_output_path=task.raw_output_path,
                        prompt_path=task.prompt_path,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                failures.append((task, text, str(exc)))

        for image in opened_images:
            image.close()
        return successes, failures

    @modal.method()
    def parse_chunk(self, chunk_payload: dict[str, Any]) -> dict[str, Any]:
        chunk = PageChunk.model_validate(chunk_payload)
        storage = FileSystemStorageBackend(Path(chunk.artifact_root))

        completed = 0
        failed = 0
        pending_tasks = list(chunk.pages)
        strict_json = False

        for attempt in range(1, 4):
            if not pending_tasks:
                break

            successes, failures = self._attempt_parse(
                chunk,
                pending_tasks,
                strict_json=strict_json,
            )
            next_pending = []
            for result in successes:
                result.attempts = attempt
                result.valid_on_first_pass = attempt == 1
                storage.write_page_result(result)
                completed += 1

            for task, _raw_text, error_message in failures:
                if attempt < 3:
                    next_pending.append(task)
                    continue
                storage.write_page_result(
                    PageParseResult(
                        job_id=chunk.job_id,
                        chunk_id=chunk.chunk_id,
                        page_id=task.page_id,
                        status=PageResultStatus.FAILED,
                        attempts=attempt,
                        valid_on_first_pass=False,
                        error=page_error(
                            task.page_id,
                            code="invalid_model_output",
                            message=error_message,
                            retry_count=attempt - 1,
                            stage="page_parse",
                        ),
                        raw_output_path=task.raw_output_path,
                        prompt_path=task.prompt_path,
                    )
                )
                failed += 1

            pending_tasks = next_pending
            strict_json = True

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

    cls_name = f"{runtime_profile.name.title()}ParserEngine"
    raw_cls = type(
        cls_name,
        (),
        {
            "__doc__": f"{runtime_profile.name} parser engine using {runtime_profile.model_id}.",
            "start": start,
            "_sampling_params_for_chunk": _sampling_params_for_chunk,
            "_attempt_parse": _attempt_parse,
            "parse_chunk": parse_chunk,
            "stop": stop,
        },
    )
    return app.cls(
        image=image,
        gpu=runtime_profile.gpu,
        timeout=ENGINE_TIMEOUT_SECONDS,
        scaledown_window=SCALEDOWN_WINDOW_SECONDS,
        volumes={
            str(HF_CACHE_ROOT): hf_cache_volume,
            str(VLLM_CACHE_ROOT): vllm_cache_volume,
            "/artifacts": artifacts_volume,
        },
    )(raw_cls)
