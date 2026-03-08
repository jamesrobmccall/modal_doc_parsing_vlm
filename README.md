# modal_doc_parsing_vlm

OCR-first document parsing on Modal with staged results, selective VLM fallback, and a split extraction stack for online serving and batched per-page work.

The deployed app is a single Modal app named `modal-doc-parsing-vlm`. One deploy publishes:

- the FastAPI web app
- the control-plane orchestrator functions
- the PaddleOCR worker pool
- the fallback vLLM worker pools (`dev` and `prod`)
- the online SGLang extraction server
- the batched per-page extraction worker
- scheduled cleanup/watchdog jobs

Detailed architecture and lifecycle notes live in [docs/architecture.md](/Users/jmccall/Documents/Coding/modal_doc_parse/modal_doc_parsing_vlm/docs/architecture.md).

## Pipeline Overview

1. `submit_parse_request_remote` stores source bytes, creates a job manifest, and enforces request idempotency.
2. `run_orchestrator` rasterizes pages, routes digital-text pages inline, and builds OCR chunks.
3. `OcrParserEngine.parse_pages` processes OCR chunks on `A10G` with PaddleOCR + PP-Structure.
4. The control plane aggregates the fast result and writes `completed_fast`.
5. Hard pages are selected by the fallback policy and regrouped into fallback chunks.
6. `DevFallbackEngine.parse_pages` or `ProdFallbackEngine.parse_pages` runs one batched `LLM.chat(...)` call per fallback chunk.
7. The control plane aggregates the refined result and writes `completed_final`.

Entity extraction is separate from document parsing:

- entity suggestion and whole-document extraction use the online `ExtractionEngine` SGLang server
- per-page extraction uses the online extraction server by default to avoid a second H100; the dedicated `ExtractionBatchEngine` path is now opt-in via `DOC_PARSE_USE_DEDICATED_EXTRACTION_BATCH_ENGINE=1`

## Runtime Defaults

| Component | Model / engine | GPU | Key throughput settings |
| --- | --- | --- | --- |
| OCR | `PP-StructureV3` | `A10G` | chunk size `balanced=4`, `accurate=2` |
| Fallback VLM (`dev`) | `Qwen/Qwen2.5-VL-7B-Instruct` | `A10G` | chunk size `balanced=2`, `accurate=1`, `enforce_eager=True` |
| Fallback VLM (`prod`) | `Qwen/Qwen2.5-VL-7B-Instruct` | `A10G` | chunk size `balanced=2`, `accurate=1`, `enforce_eager=False` |
| Extraction online server | `Qwen/Qwen3-4B-Thinking-2507-FP8` via SGLang | `H100:1` | `target_concurrency=8`, `max_running_requests=8`, `min_containers=0`, `max_containers=1` |
| Extraction batch worker | same SGLang stack | `H100:1` | opt-in only, `max_batch_size=8`, `wait_ms=20`, `max_containers=1` |

Pinned revisions are configured for both fallback and extraction models so deploys stay reproducible.

## Result Lifecycle

- active: `queued`, `splitting`, `submitting`, `running`, `aggregating`
- staged complete: `completed_fast`, `completed_final`
- terminal errors: `completed_with_errors`, `failed`

Parse status also reports:

- `result_revision`
- `pending_refinement_pages`

Extraction status also reports:

- `requests_total`
- `requests_completed`

## Local Setup

Use Python `3.12` for the project environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

If you use the pipx-installed Modal CLI, keep using `modal` from `$(command -v modal)`. For local Python commands and tests, use `./.venv/bin/python`.

Build the frontend bundle before local serve/deploy if you want the web UI at `/`:

```bash
./scripts/build_frontend.sh
```

## Deploy

Deploy everything together:

```bash
./scripts/build_frontend.sh
./.venv/bin/modal deploy app.py
```

For a non-default environment, add `-e <env>` to every Modal command:

```bash
./.venv/bin/modal deploy app.py -e codex-doc-parse-perf
```

Useful post-deploy commands:

```bash
./.venv/bin/modal app logs modal-doc-parsing-vlm --timestamps
./.venv/bin/modal app dashboard modal-doc-parsing-vlm
./.venv/bin/modal app history modal-doc-parsing-vlm
```

## Warmup And Smoke Tests

Warm caches and startup paths:

```bash
./.venv/bin/modal run app.py::cache_model_weights
```

Smoke the extraction stack:

```bash
./.venv/bin/modal run app.py::smoke_entity_extraction
```

Smoke the document parser:

```bash
./.venv/bin/modal run app.py::smoke_test --runtime-profile-name dev --latency-profile fast --result-level latest
./.venv/bin/modal run app.py::smoke_test --runtime-profile-name dev --latency-profile balanced --result-level latest
```

Environment-scoped smoke helper:

```bash
MODAL_ENVIRONMENT=codex-doc-parse-perf ./scripts/safe_modal_smoke.sh
```

## Benchmarks

Built-in CLI entrypoints:

```bash
./.venv/bin/modal run app.py::benchmark_ocr_fast --sample-path tmp/benchmark-5pages.pdf
./.venv/bin/modal run app.py::benchmark_fallback_refinement --sample-path tmp/benchmark-5pages.pdf
./.venv/bin/modal run app.py::benchmark_per_page_extraction --sample-path tmp/benchmark-5pages.pdf
```

These commands run against real Modal infra, but they use ephemeral `modal run` apps. That means the numbers can include cold-start work such as model startup, OCR/table asset downloads, and first-run compilation. For steadier service-path measurements, benchmark against a deployed environment by calling the deployed functions or HTTP API directly.

The parser also uses request idempotency. If you rerun a benchmark with the exact same input bytes and request payload, the job may be reused. For repeat benchmark runs, use fresh sample copies or change the request payload so you get a new job fingerprint.

## Results

Result artifacts are written under `/artifacts/jobs/<job_id>/result/` in the Modal volume. The standard outputs are:

- `document_parse_result.fast.json`
- `document_parse_result.final.json`
- `document_parse_result.json`
- `document.fast.md`
- `document.final.md`
- `document.md`
- `document.fast.txt`
- `document.final.txt`
- `document.txt`

Download a result bundle locally:

```bash
./.venv/bin/modal run app.py::download_result --job-id <job_id> --result-level latest
```

Local downloads default to `~/.cache/modal-doc-parsing-vlm/job-results`. Override with:

```bash
export DOC_PARSE_LOCAL_OUTPUT_ROOT="$HOME/Documents/modal-doc-parse-results"
```

## Testing

Local tests:

```bash
./.venv/bin/python -m pytest
```

Live Modal integration tests:

```bash
RUN_MODAL_TESTS=1 MODAL_BIN=./.venv/bin/modal MODAL_ENVIRONMENT=codex-doc-parse-perf ./.venv/bin/python -m pytest tests/integration/test_modal_smoke.py
```

## Cleanup

Mark stale jobs failed now:

```bash
./.venv/bin/modal run app.py::cleanup_stale_now
```

Stop only this app in a specific environment:

```bash
./.venv/bin/modal app stop modal-doc-parsing-vlm -e codex-doc-parse-perf
```

Inspect remaining containers in that environment:

```bash
./.venv/bin/modal container list -e codex-doc-parse-perf --json
```

Delete a disposable environment after testing:

```bash
./.venv/bin/modal environment delete codex-doc-parse-perf --yes
```

## Key Environment Knobs

```bash
export DOC_PARSE_OCR_MIN_CONTAINERS=0
export DOC_PARSE_OCR_MAX_CONTAINERS=1
export DOC_PARSE_OCR_BUFFER_CONTAINERS=0
export DOC_PARSE_FALLBACK_MIN_CONTAINERS=0
export DOC_PARSE_FALLBACK_MAX_CONTAINERS=1
export DOC_PARSE_FALLBACK_BUFFER_CONTAINERS=0
export DOC_PARSE_EXTRACTION_MIN_CONTAINERS=0
export DOC_PARSE_EXTRACTION_MAX_CONTAINERS=1
export DOC_PARSE_OCR_SCALEDOWN_WINDOW_SECONDS=120
export DOC_PARSE_FALLBACK_SCALEDOWN_WINDOW_SECONDS=120
export DOC_PARSE_EXTRACTION_SCALEDOWN_WINDOW_SECONDS=120
export DOC_PARSE_USE_DEDICATED_EXTRACTION_BATCH_ENGINE=0
export DOC_PARSE_STALE_JOB_TIMEOUT_SECONDS=1800
export DOC_PARSE_STALE_JOB_SWEEP_SECONDS=600
```
