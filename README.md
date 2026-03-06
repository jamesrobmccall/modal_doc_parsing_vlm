# modal_doc_parsing_vlm

OCR-first document parsing on Modal with staged results:

- fast stage: digital text extraction + PaddleOCR (`completed_fast`)
- refine stage: selective VLM fallback on hard pages (`completed_final`)

## Pipeline Overview

1. Ingest + rasterize pages.
2. Route each page:
   - digital-native text layer => `digital_text` engine
   - otherwise => `paddle_ocr` engine
   - if extracted PDF text appears low-language-quality (e.g., encoded/garbled layer), force `paddle_ocr`
3. Aggregate fast result (`result_revision=1`, `quality_stage=fast`).
4. Detect hard pages via fallback policy.
5. Refine only those pages with VLM fallback.
6. Aggregate final result (`result_revision>=2`, `quality_stage=final`).

## Runtime Profiles

- OCR runtime profile:
  - engine: `PP-StructureV3` (PaddleOCR image path)
  - GPU: `A10G`
  - structured markdown generation:
    - headings/list/caption formatting
    - HTML table -> Markdown table conversion
    - OCR/layout dedupe by bbox overlap
  - Modal scaling defaults:
    - `min_containers=0`
    - `buffer_containers=1`
    - `allow_concurrent_inputs=1`
    - `scaledown_window=900s`
- Extraction runtime:
  - model: `Qwen/Qwen3-4B-Thinking-2507-FP8`
  - serving: SGLang low-latency OpenAI-compatible endpoint on `H100:1`
  - reasoning disabled per request for faster structured extraction
  - Modal scaling defaults:
    - `min_containers=1`
    - `target_inputs=4`
    - `scaledown_window=900s`
- Fallback VLM runtime profiles (`prod`, `dev`):
  - model: `Qwen/Qwen2.5-VL-7B-Instruct`
  - GPU: `A10G`
  - async refinement only for triggered pages
  - fast boot defaults to eager mode for lower cold-start latency
  - optional deep refine model ID retained in config (`Qwen/Qwen3.5-27B-FP8`)

## API Additions

- `SubmitDocumentParseRequest`
  - `result_level`: `latest|fast|final` (default: `latest`)
  - `latency_profile`: `fast|balanced|max_quality` (default: `balanced`)
- `GetDocumentParseResultRequest`
  - `result_level`: `latest|fast|final` (default: `latest`)
- `GetDocumentParseStatusResponse`
  - `result_revision`
  - `pending_refinement_pages`

## Job Status Lifecycle

- active: `queued`, `splitting`, `submitting`, `running`, `aggregating`
- staged complete: `completed_fast`, `completed_final`
- terminal errors: `completed_with_errors`, `failed`

## Local Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Main Commands

Serve:

```bash
PATH="$HOME/.local/bin:$PATH" modal serve app.py
```

Build frontend bundle (required before deploy/serve if you want the web UI at `/`):

```bash
./scripts/build_frontend.sh
```

## Deploy to Modal

This repo is a single Modal app named `modal-doc-parsing-vlm`. Running
`modal deploy app.py` deploys all of the updated pieces together:

- the FastAPI web app
- the OCR worker
- the fallback VLM worker(s)
- the dedicated SGLang extraction server
- the schedules / background cleanup functions

If you use a non-default Modal environment, add `--env <name>` to every
`modal deploy`, `modal run`, and `modal app ...` command below.

One-time setup on a new machine:

```bash
python -m pip install --upgrade modal
PATH="$HOME/.local/bin:$PATH" modal setup
```

Simple update flow after you change code:

```bash
./scripts/build_frontend.sh
PATH="$HOME/.local/bin:$PATH" modal deploy app.py
PATH="$HOME/.local/bin:$PATH" modal run app.py::cache_model_weights
PATH="$HOME/.local/bin:$PATH" modal run app.py::smoke_entity_extraction
```

What each step does:

- `./scripts/build_frontend.sh`
  - rebuilds the static UI bundle that gets embedded into the deployed web app
- `modal deploy app.py`
  - publishes the latest code to Modal and updates the deployed app in place
- `modal run app.py::cache_model_weights`
  - warms the Hugging Face cache plus OCR/extraction startup paths
- `modal run app.py::smoke_entity_extraction`
  - quick sanity check that the deployed extraction stack is healthy

Notes:

- You do not need a separate deploy command for the extraction server. It is part of the same `app.py` deployment.
- Volumes and Dicts are created automatically on first deploy because the code uses `modal.Volume.from_name(..., create_if_missing=True)` and `modal.Dict.from_name(..., create_if_missing=True)`.
- If you skip `./scripts/build_frontend.sh`, the API still deploys, but the root `/` UI may serve the “Frontend bundle missing” placeholder.
- Modal prints the deployed web URL during `modal deploy app.py`; you can also open it later with `modal app dashboard modal-doc-parsing-vlm`.
- The first `modal run app.py::cache_model_weights` can take a while. It may spend several minutes building the Paddle GPU image and then several more minutes doing the first SGLang/DeepGEMM cold start. That is expected on the first run; later runs are much faster because the image layers, model weights, and DeepGEMM cache are reused.

Useful post-deploy commands:

```bash
PATH="$HOME/.local/bin:$PATH" modal app logs modal-doc-parsing-vlm --timestamps
PATH="$HOME/.local/bin:$PATH" modal app dashboard modal-doc-parsing-vlm
PATH="$HOME/.local/bin:$PATH" modal app history modal-doc-parsing-vlm
```

Rollback / stop:

```bash
PATH="$HOME/.local/bin:$PATH" modal app rollback modal-doc-parsing-vlm
PATH="$HOME/.local/bin:$PATH" modal app stop modal-doc-parsing-vlm
```

`modal app rollback` requires a Modal plan that supports deployment rollbacks.

Seed runtime caches and warm OCR + extraction assets for all enabled profiles
(or pass `--runtime-profile-name` to limit it):

```bash
PATH="$HOME/.local/bin:$PATH" modal run app.py::cache_model_weights
```

Smoke the dedicated extraction flow:

```bash
PATH="$HOME/.local/bin:$PATH" modal run app.py::smoke_entity_extraction
```

Smoke test (fast/final selectable):

```bash
PATH="$HOME/.local/bin:$PATH" modal run app.py::smoke_test --runtime-profile-name dev --latency-profile fast --result-level latest
```

For best structure quality (layout + tables), use `balanced` or `max_quality` latency profile:

```bash
PATH="$HOME/.local/bin:$PATH" modal run app.py::smoke_test --runtime-profile-name dev --latency-profile balanced --result-level latest
```

Smoke test with automatic cleanup trap (recommended for ad-hoc testing):

```bash
./scripts/safe_modal_smoke.sh
```

Download result:

```bash
PATH="$HOME/.local/bin:$PATH" modal run app.py::download_result --job-id <job_id> --result-level latest
```

Run frontend tests:

```bash
npm --prefix frontend test
```

Local result bundle default path (outside repo): `~/.cache/modal-doc-parsing-vlm/job-results`.
Override with:

```bash
export DOC_PARSE_LOCAL_OUTPUT_ROOT="$HOME/Documents/modal-doc-parse-results"
```

Mark stale jobs failed now (watchdog action):

```bash
PATH="$HOME/.local/bin:$PATH" modal run app.py::cleanup_stale_now
```

Hard-stop deployed app + active containers (manual emergency cleanup):

```bash
PATH="$HOME/.local/bin:$PATH" modal app list --json | jq -r '.[] | select(.State=="running" or .State=="deployed") | .["App ID"]' | xargs -r -n1 "$HOME/.local/bin/modal" app stop
PATH="$HOME/.local/bin:$PATH" modal container list --json | jq -r '.[].container_id' | xargs -r -n1 "$HOME/.local/bin/modal" container stop
```

## Result Artifacts (Modal volume)

`/jobs/<job_id>/result/` includes:

- `document_parse_result.fast.json`
- `document_parse_result.final.json`
- `document_parse_result.json` (latest alias)
- `document.fast.md`, `document.final.md`, `document.md`
- `document.fast.txt`, `document.final.txt`, `document.txt`

## Testing

```bash
python3 -m pytest
```

Live Modal tests:

```bash
RUN_MODAL_TESTS=1 python3 -m pytest tests/integration/test_modal_smoke.py
```

Cost-safe env defaults (override only if you explicitly want warm pools):

```bash
export DOC_PARSE_OCR_MIN_CONTAINERS=0
export DOC_PARSE_OCR_BUFFER_CONTAINERS=1
export DOC_PARSE_FALLBACK_MIN_CONTAINERS=0
export DOC_PARSE_FALLBACK_BUFFER_CONTAINERS=0
export DOC_PARSE_EXTRACTION_MIN_CONTAINERS=1
export DOC_PARSE_OCR_SCALEDOWN_WINDOW_SECONDS=900
export DOC_PARSE_EXTRACTION_SCALEDOWN_WINDOW_SECONDS=900
export DOC_PARSE_FALLBACK_SCALEDOWN_WINDOW_SECONDS=300
export DOC_PARSE_STALE_JOB_TIMEOUT_SECONDS=1800
export DOC_PARSE_STALE_JOB_SWEEP_SECONDS=600
```
