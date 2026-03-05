from __future__ import annotations

import json
import time
from typing import Any

import modal

from .config import (
    COMMON_VLLM_DEPENDENCIES,
    CUDA_IMAGE,
    EXTRACTION_ALLOW_CONCURRENT_INPUTS,
    EXTRACTION_BUFFER_CONTAINERS,
    EXTRACTION_ENGINE_TIMEOUT_SECONDS,
    EXTRACTION_GPU,
    EXTRACTION_MAX_MODEL_LEN,
    EXTRACTION_MIN_CONTAINERS,
    EXTRACTION_MODEL_ID,
    EXTRACTION_SAMPLING_MAX_TOKENS,
    EXTRACTION_SCALEDOWN_WINDOW_SECONDS,
    HF_CACHE_ROOT,
    HF_HUB_CACHE_ROOT,
    TORCHINDUCTOR_CACHE_ROOT,
    VLLM_CACHE_ROOT,
    VLLM_NIGHTLY_EXTRA_INDEX_URL,
    VLLM_UV_EXTRA_OPTIONS,
)


def _build_extraction_image() -> modal.Image:
    return (
        modal.Image.from_registry(CUDA_IMAGE, add_python="3.12")
        .entrypoint([])
        .apt_install("git")
        .uv_pip_install(*COMMON_VLLM_DEPENDENCIES)
        .uv_pip_install(
            "vllm",
            extra_index_url=VLLM_NIGHTLY_EXTRA_INDEX_URL,
            extra_options=VLLM_UV_EXTRA_OPTIONS,
        )
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


def _extract_output_text(output: Any) -> str:
    if hasattr(output, "outputs") and output.outputs:
        return output.outputs[0].text.strip()
    if isinstance(output, str):
        return output.strip()
    raise TypeError(f"Unsupported vLLM output type: {type(output)!r}")


def create_extraction_engine_cls(
    app: modal.App,
    *,
    hf_cache_volume,
    vllm_cache_volume,
    artifacts_volume,
    export_module: str,
):
    image = _build_extraction_image()

    @modal.enter()
    def start(self) -> None:
        from vllm import LLM

        self.model_id = EXTRACTION_MODEL_ID
        print(
            f"[engine:extraction] loading model={self.model_id} "
            f"gpu={EXTRACTION_GPU} max_model_len={EXTRACTION_MAX_MODEL_LEN}"
        )
        self.llm = LLM(
            model=self.model_id,
            tensor_parallel_size=1,
            max_model_len=EXTRACTION_MAX_MODEL_LEN,
            enforce_eager=False,
            download_dir=str(HF_HUB_CACHE_ROOT),
        )
        print(f"[engine:extraction] model loaded: {self.model_id}")

    @modal.method()
    def suggest_entities(self, payload: dict[str, Any]) -> dict[str, Any]:
        from json_repair import repair_json
        from vllm import SamplingParams

        from modal_doc_parsing_vlm.prompts_extraction import build_entity_suggestion_prompt

        messages = build_entity_suggestion_prompt(
            document_markdown=payload["document_markdown"],
            page_count=payload.get("page_count", 1),
        )
        started = time.perf_counter()
        outputs = self.llm.chat(
            [messages],
            sampling_params=SamplingParams(
                temperature=0.0,
                max_tokens=EXTRACTION_SAMPLING_MAX_TOKENS,
            ),
            use_tqdm=False,
        )
        raw_text = _extract_output_text(outputs[0])
        elapsed_ms = int((time.perf_counter() - started) * 1000)

        repaired = repair_json(raw_text, return_objects=False)
        data = json.loads(repaired)

        print(
            f"[engine:extraction] suggest_entities "
            f"entities={len(data.get('entities', []))} "
            f"elapsed_ms={elapsed_ms}"
        )
        return {
            "entities": data.get("entities", []),
            "document_summary": data.get("document_summary", ""),
            "inference_ms": elapsed_ms,
        }

    @modal.method()
    def extract_entity(self, payload: dict[str, Any]) -> dict[str, Any]:
        from vllm import SamplingParams

        from modal_doc_parsing_vlm.prompts_extraction import build_entity_extraction_prompt
        from modal_doc_parsing_vlm.types_extraction import EntityDefinition

        entity = EntityDefinition.model_validate(payload["entity_definition"])
        page_text = payload["page_text"]
        page_id = payload.get("page_id")
        json_schema = payload["json_schema"]

        messages = build_entity_extraction_prompt(entity, page_text)

        # Build guided decoding params – handle both old and new vLLM APIs.
        guided_kwargs: dict[str, Any] = {}
        try:
            from vllm.sampling_params import GuidedDecodingParams
            guided_kwargs["guided_decoding"] = GuidedDecodingParams(json=json_schema)
        except ImportError:
            from vllm.sampling_params import GuidedDecodingRequest
            guided_kwargs["guided_options_request"] = GuidedDecodingRequest(json=json_schema)

        started = time.perf_counter()
        outputs = self.llm.chat(
            [messages],
            sampling_params=SamplingParams(
                temperature=0.0,
                max_tokens=EXTRACTION_SAMPLING_MAX_TOKENS,
            ),
            **guided_kwargs,
            use_tqdm=False,
        )
        raw_text = _extract_output_text(outputs[0])
        elapsed_ms = int((time.perf_counter() - started) * 1000)

        data = json.loads(raw_text)

        print(
            f"[engine:extraction] extract_entity "
            f"entity={entity.entity_name} page_id={page_id} "
            f"elapsed_ms={elapsed_ms}"
        )
        return {
            "entity_name": entity.entity_name,
            "page_id": page_id,
            "data": data,
            "inference_ms": elapsed_ms,
        }

    @modal.exit()
    def stop(self) -> None:
        del self.llm

    raw_cls = type(
        "ExtractionEngine",
        (),
        {
            "__doc__": f"Entity extraction engine using {EXTRACTION_MODEL_ID}.",
            "__module__": export_module,
            "start": start,
            "suggest_entities": suggest_entities,
            "extract_entity": extract_entity,
            "stop": stop,
        },
    )
    concurrent_cls = modal.concurrent(max_inputs=EXTRACTION_ALLOW_CONCURRENT_INPUTS)(raw_cls)
    cls = app.cls(
        image=image,
        gpu=EXTRACTION_GPU,
        timeout=EXTRACTION_ENGINE_TIMEOUT_SECONDS,
        min_containers=EXTRACTION_MIN_CONTAINERS,
        buffer_containers=EXTRACTION_BUFFER_CONTAINERS,
        scaledown_window=EXTRACTION_SCALEDOWN_WINDOW_SECONDS,
        volumes={
            str(HF_CACHE_ROOT): hf_cache_volume,
            str(VLLM_CACHE_ROOT): vllm_cache_volume,
            "/artifacts": artifacts_volume,
        },
    )(concurrent_cls)
    return cls
