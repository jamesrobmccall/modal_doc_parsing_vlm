# Architecture

## High-Level Topology

`modal-doc-parsing-vlm` is a single Modal app with six major roles:

1. FastAPI web app exposed through `web()`
2. Control-plane functions for submission, orchestration, result reads, extraction, and cleanup
3. `OcrParserEngine` for OCR/layout parsing on `A10G`
4. `DevFallbackEngine` and `ProdFallbackEngine` for fallback VLM parsing on `L4`
5. `ExtractionEngine`, an on-demand online SGLang server on `L4`
6. `ExtractionBatchEngine`, an optional batched per-page extraction worker on `L4`

Persistent state is split across Modal Volumes and Dicts:

- `doc-parse-artifacts`: source files, page images, page results, and final result bundles
- `doc-parse-hf-cache`: Hugging Face snapshots
- `doc-parse-vllm-cache`: vLLM and TorchInductor caches
- `doc-parse-deepgemm-cache`: SGLang DeepGEMM compilation cache when enabled
- `doc-parse-paddle-cache`: PaddleOCR assets
- `doc-parse-job-status`: job status snapshots
- `doc-parse-idempotency`: request fingerprint dedupe records

## Document Parse Flow

### 1. Submission

`submit_parse_request_remote` validates the request, stores source bytes, computes a request fingerprint, and returns a `job_id`. If the same bytes and payload are submitted again, the service can reuse the existing job.

### 2. Split And Route

`run_orchestrator` drives `process_job()`:

- rasterize source pages
- run `classify_page()` on each page
- keep digital-text pages on the control plane
- build OCR `PageChunk`s only for pages that need OCR

Chunking is engine-specific:

- OCR chunk size: `balanced=4`, `accurate=2`
- fallback chunk size: `balanced=2`, `accurate=1`

Chunk IDs are persisted in the manifest so result bundles reflect the actual worker grouping.

### 3. Fast Stage

Digital-text pages are converted directly into `PageParseResult`s. OCR pages are sent to `OcrParserEngine.parse_pages`.

`OcrParserEngine`:

- runs PaddleOCR text detection/recognition
- runs PP-Structure layout parsing for non-`fast` latency profiles
- merges layout elements and OCR lines
- emits Markdown plus structured elements
- records real OCR `inference_ms`

After all fast-stage page results are written, the control plane aggregates revision `1` and publishes `completed_fast`.

### 4. Selective Refinement

The fallback policy inspects the fast-stage results and marks only hard pages for refinement. `latency_profile=fast` disables this stage completely.

### 5. Fallback Refinement

`run_refinement` rebuilds only the pending pages into fallback chunks and calls `parse_pages` on the selected fallback engine.

Each fallback worker:

- loads a pinned `Qwen/Qwen3-VL-8B-Instruct-FP8` revision
- builds multimodal prompts for every page in the chunk
- runs one batched `LLM.chat(...)` call for the chunk
- maps outputs back to the original page order

The control plane writes revision `>=2`, aggregates the final document, and publishes `completed_final`.

## Runtime Profiles

The fallback stack supports `prod` and `dev` runtime profiles. Both pin the same model and tokenizer revisions; the main behavior difference is startup mode:

- `dev`: `enforce_eager=True` for easier iteration and more predictable debug behavior
- `prod`: `enforce_eager=False` for better warm-throughput behavior

Both currently use one `L4`, no tensor parallelism, `trust_remote_code=True`, and `gpu_memory_utilization=0.70`.

## Extraction Architecture

Extraction is intentionally split into two paths.

### Online Path

`ExtractionEngine` is the default SGLang server used for:

- entity suggestion
- whole-document extraction

It uses sticky `Modal-Session-ID` routing and is tuned around:

- `min_containers=0`
- `max_containers=1`
- `target_concurrency=4`
- `max_running_requests=4`
- `mem_fraction=0.70`

Token budgets:

- suggestion: `1536`
- whole-document extraction: `2048`

### Batched Per-Page Path

By default, per-page extraction stays on the online extraction server so the system does not need a second L4 just for batch work.

If `DOC_PARSE_USE_DEDICATED_EXTRACTION_BATCH_ENGINE=1` is set, `run_entity_extraction` flattens per-page extraction into `(entity, page)` work items and sends them to `ExtractionBatchEngine.extract_pages`.

When enabled, `ExtractionBatchEngine`:

- starts its own local SGLang server in `@modal.enter`
- accepts a list of work items
- is wrapped in `@modal.batched(max_batch_size=4, wait_ms=20)`
- is capped to one container by default
- returns list-aligned results so the control plane can reconstruct final output

The extraction image now enables DeepGEMM only when the selected extraction GPU is Hopper or Blackwell-class, or when `DOC_PARSE_EXTRACTION_ENABLE_DEEPGEMM=true` is set explicitly. The default `L4` profile skips DeepGEMM compilation and uses the standard SGLang runtime path.

Per-page token budget:

- `1024`

Status reporting is additive and now includes:

- `requests_total`
- `requests_completed`

## Deployment Model

`modal deploy app.py` updates the entire system in one step. There is no separate extraction-server deploy command. The deployed app contains:

- the ASGI web app
- all worker classes
- all background jobs and schedules

That means a single Modal environment can host a full isolated stack, which is useful for perf testing in disposable environments such as `codex-doc-parse-perf`.

## Benchmarking Notes

There are two valid ways to benchmark this project, and they measure different things.

### Ephemeral CLI Benchmarks

`modal run app.py::benchmark_*` exercises real Modal infra and is convenient for smoke/perf validation from the CLI. These runs create ephemeral apps, so wall time may include:

- container cold starts
- OCR/table asset downloads
- first-run SGLang compilation
- model warmup

Cost note: ephemeral `modal run` apps are separate from the deployed app. If you keep warm pools enabled, each app can reserve its own GPUs. The cost-safe defaults in this repo cap worker container counts and keep warm pools at zero to avoid duplicated spend in dev and perf environments.

### Deployed-App Benchmarks

For steadier numbers, benchmark a deployed environment by calling the deployed functions or HTTP endpoints directly. This reuses the long-lived deployed topology and is a better fit for throughput measurements once the service is warm.

Because submission is idempotent, benchmark reruns need unique input bytes or a changed payload to avoid accidentally reusing an existing `job_id`.

## Result Lifecycle

Job status values:

- **Active**: `queued`, `splitting`, `submitting`, `running`, `aggregating`
- **Staged complete**: `completed_fast` (OCR result ready), `completed_final` (VLM refinement done)
- **Terminal errors**: `completed_with_errors`, `failed`

Parse status also reports `result_revision` and `pending_refinement_pages`.

Extraction status also reports `requests_total` and `requests_completed`.

## Smoke Tests And Warmup

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

These run against real Modal infra using ephemeral `modal run` apps, so wall time may include cold starts, asset downloads, and first-run compilation. For steadier numbers, benchmark against a deployed environment directly.

Because submission is idempotent, benchmark reruns need unique input bytes or a changed payload to avoid reusing an existing `job_id`.

## Testing And Cleanup

Recommended live workflow in a disposable environment:

1. `./scripts/build_frontend.sh`
2. `./.venv/bin/modal deploy app.py -e <env>`
3. `./.venv/bin/modal run app.py::cache_model_weights -e <env>`
4. smoke tests and benchmarks
5. `./.venv/bin/python -m pytest`
6. `RUN_MODAL_TESTS=1 MODAL_BIN=./.venv/bin/modal MODAL_ENVIRONMENT=<env> ./.venv/bin/python -m pytest tests/integration/test_modal_smoke.py`

Safe cleanup is environment-scoped:

1. `./.venv/bin/modal run app.py::cleanup_stale_now -e <env>`
2. `./.venv/bin/modal app stop modal-doc-parsing-vlm -e <env>`
3. `./.venv/bin/modal container list -e <env> --json`
4. `./.venv/bin/modal environment delete <env> --yes`

Avoid workspace-wide cleanup commands when testing in shared Modal workspaces.

Useful post-deploy commands:

```bash
./.venv/bin/modal app logs modal-doc-parsing-vlm --timestamps
./.venv/bin/modal app dashboard modal-doc-parsing-vlm
./.venv/bin/modal app history modal-doc-parsing-vlm
```

Mark stale jobs failed immediately:

```bash
./.venv/bin/modal run app.py::cleanup_stale_now
```
