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
        from PIL import Image
        from vllm import LLM, SamplingParams

        self.runtime_profile_name = runtime_profile.name
        self.model_id = runtime_profile.model_id
        self.disable_thinking = runtime_profile.disable_thinking
        cache_status = describe_model_cache(self.model_id)
        print(
            f"[engine:{self.runtime_profile_name}] loading model={self.model_id} "
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
        self.default_sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=SAMPLING_MAX_TOKENS["balanced"],
        )
        if STARTUP_WARMUP_ENABLED:
            print(
                f"[engine:{self.runtime_profile_name}] running optional startup warmup "
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
            print(f"[engine:{self.runtime_profile_name}] optional startup warmup complete")
        else:
            print(
                f"[engine:{self.runtime_profile_name}] model initialized; "
                "skipping optional startup warmup"
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
        conversations = []
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
            conversations.append(_prompt_to_messages(prompt, image))

        started = time.perf_counter()
        outputs = self.llm.chat(
            conversations,
            sampling_params=sampling_params,
            chat_template_content_format="openai",
            chat_template_kwargs=self.chat_template_kwargs,
            use_tqdm=False,
        )
        inference_ms = int((time.perf_counter() - started) * 1000)

        successes: list[PageParseResult] = []
        failures: list[tuple[Any, str, str]] = []
        for task, output in zip(tasks, outputs):
            text = _extract_output_text(output)
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
        print(
            f"[engine:{self.runtime_profile_name}] parse_chunk start "
            f"job_id={chunk.job_id} chunk_id={chunk.chunk_id} pages={len(chunk.pages)}"
        )

        completed = 0
        failed = 0
        pending_tasks = list(chunk.pages)
        strict_json = False

        for attempt in range(1, 4):
            if not pending_tasks:
                break
            print(
                f"[engine:{self.runtime_profile_name}] parse_chunk attempt={attempt} "
                f"chunk_id={chunk.chunk_id} pending_pages={len(pending_tasks)} "
                f"strict_json={strict_json}"
            )

            successes, failures = self._attempt_parse(
                chunk,
                pending_tasks,
                strict_json=strict_json,
            )
            print(
                f"[engine:{self.runtime_profile_name}] parse_chunk attempt={attempt} "
                f"chunk_id={chunk.chunk_id} successes={len(successes)} failures={len(failures)}"
            )
            next_pending = []
            for result in successes:
                result.attempts = attempt
                result.valid_on_first_pass = attempt == 1
                storage.write_page_result(result)
                completed += 1

            for task, raw_text, error_message in failures:
                print(
                    f"[engine:{self.runtime_profile_name}] page_parse failure "
                    f"chunk_id={chunk.chunk_id} page_id={task.page_id} attempt={attempt} "
                    f"error={_preview_text(error_message, limit=320)} "
                    f"raw_output={_preview_text(raw_text)}"
                )
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
        print(
            f"[engine:{self.runtime_profile_name}] parse_chunk done "
            f"job_id={chunk.job_id} chunk_id={chunk.chunk_id} "
            f"completed={completed} failed={failed}"
        )
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
            "__module__": export_module,
            "start": start,
            "_sampling_params_for_chunk": _sampling_params_for_chunk,
            "_attempt_parse": _attempt_parse,
            "parse_chunk": parse_chunk,
            "stop": stop,
        },
    )
    cls = app.cls(
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
    return cls
