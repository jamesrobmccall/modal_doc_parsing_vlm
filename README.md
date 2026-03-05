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
    - `buffer_containers=0`
    - `allow_concurrent_inputs=1`
    - `scaledown_window=300s`
- Fallback VLM runtime profiles (`prod`, `dev`):
  - model: `Qwen/Qwen2.5-VL-7B-Instruct`
  - GPU: `A10G`
  - async refinement only for triggered pages
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

Seed HF cache volume:

```bash
PATH="$HOME/.local/bin:$PATH" modal run app.py::cache_model_weights --runtime-profile-name dev
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
pytest
```

Live Modal tests:

```bash
RUN_MODAL_TESTS=1 pytest tests/integration/test_modal_smoke.py
```

Cost-safe env defaults (override only if you explicitly want warm pools):

```bash
export DOC_PARSE_OCR_MIN_CONTAINERS=0
export DOC_PARSE_OCR_BUFFER_CONTAINERS=0
export DOC_PARSE_FALLBACK_MIN_CONTAINERS=0
export DOC_PARSE_FALLBACK_BUFFER_CONTAINERS=0
export DOC_PARSE_OCR_SCALEDOWN_WINDOW_SECONDS=300
export DOC_PARSE_FALLBACK_SCALEDOWN_WINDOW_SECONDS=300
export DOC_PARSE_STALE_JOB_TIMEOUT_SECONDS=1800
export DOC_PARSE_STALE_JOB_SWEEP_SECONDS=600
```
