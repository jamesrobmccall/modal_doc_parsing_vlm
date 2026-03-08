# Configuration

## Runtime Defaults

| Component | Model / engine | GPU | Key throughput settings |
| --- | --- | --- | --- |
| OCR | `PP-StructureV3` | `A10G` | chunk size `balanced=4`, `accurate=2` |
| Fallback VLM (`dev`) | `Qwen/Qwen3-VL-8B-Instruct-FP8` | `L4` | chunk size `balanced=2`, `accurate=1`, `enforce_eager=True`, `gpu_memory_utilization=0.70` |
| Fallback VLM (`prod`) | `Qwen/Qwen3-VL-8B-Instruct-FP8` | `L4` | chunk size `balanced=2`, `accurate=1`, `enforce_eager=False`, `gpu_memory_utilization=0.70` |
| Extraction online server | `Qwen/Qwen3-4B-Thinking-2507-FP8` via SGLang | `L4` | `target_concurrency=4`, `max_running_requests=4`, `min_containers=0`, `max_containers=1`, `mem_fraction=0.70` |
| Extraction batch worker | same SGLang stack | `L4` | opt-in only, `max_batch_size=4`, `wait_ms=20`, `max_containers=1` |

Pinned revisions are configured for both fallback and extraction models so deploys stay reproducible. DeepGEMM compilation is enabled automatically only on Hopper or Blackwell-class extraction GPUs; the default `L4` extraction profile skips that path.

## Runtime Profiles

The fallback stack supports `dev` and `prod` runtime profiles. Both pin the same model and tokenizer revisions; the main behavior difference is startup mode:

- `dev`: `enforce_eager=True` for easier iteration and more predictable debug behavior
- `prod`: `enforce_eager=False` for better warm-throughput behavior

Both use one `L4`, no tensor parallelism, `trust_remote_code=True`, and `gpu_memory_utilization=0.70`.

## Environment Variables

```bash
# OCR worker pool
export DOC_PARSE_OCR_MIN_CONTAINERS=0
export DOC_PARSE_OCR_MAX_CONTAINERS=1
export DOC_PARSE_OCR_BUFFER_CONTAINERS=0
export DOC_PARSE_OCR_SCALEDOWN_WINDOW_SECONDS=120

# Fallback VLM worker pool
export DOC_PARSE_FALLBACK_MIN_CONTAINERS=0
export DOC_PARSE_FALLBACK_MAX_CONTAINERS=1
export DOC_PARSE_FALLBACK_BUFFER_CONTAINERS=0
export DOC_PARSE_FALLBACK_MODEL_ID=Qwen/Qwen3-VL-8B-Instruct-FP8
export DOC_PARSE_FALLBACK_GPU=L4
export DOC_PARSE_FALLBACK_GPU_MEMORY_UTILIZATION=0.70
export DOC_PARSE_FALLBACK_SCALEDOWN_WINDOW_SECONDS=120

# Extraction server
export DOC_PARSE_EXTRACTION_MIN_CONTAINERS=0
export DOC_PARSE_EXTRACTION_MAX_CONTAINERS=1
export DOC_PARSE_EXTRACTION_GPU=L4
export DOC_PARSE_EXTRACTION_TARGET_INPUTS=4
export DOC_PARSE_EXTRACTION_MAX_RUNNING_REQUESTS=4
export DOC_PARSE_EXTRACTION_ENABLE_DEEPGEMM=auto
export DOC_PARSE_EXTRACTION_SCALEDOWN_WINDOW_SECONDS=120

# Opt-in dedicated batch extraction engine
export DOC_PARSE_USE_DEDICATED_EXTRACTION_BATCH_ENGINE=0

# Cleanup / watchdog
export DOC_PARSE_STALE_JOB_TIMEOUT_SECONDS=1800
export DOC_PARSE_STALE_JOB_SWEEP_SECONDS=600
```

## Result Artifacts

Result artifacts are written under `/artifacts/jobs/<job_id>/result/` in the Modal volume. Standard outputs:

- `document_parse_result.fast.json` / `document_parse_result.final.json` / `document_parse_result.json`
- `document.fast.md` / `document.final.md` / `document.md`
- `document.fast.txt` / `document.final.txt` / `document.txt`

Download a result bundle locally:

```bash
./.venv/bin/modal run app.py::download_result --job-id <job_id> --result-level latest
```

Local downloads default to `~/.cache/modal-doc-parsing-vlm/job-results`. Override with:

```bash
export DOC_PARSE_LOCAL_OUTPUT_ROOT="$HOME/Documents/modal-doc-parse-results"
```
