# modal_doc_parsing_vlm

Batch document parsing using a multimodal LLM on Modal.

## Scope

This scaffold implements the core backend from the PRD:

- stateless FastMCP server mounted at `/mcp`
- Modal-based document splitting, chunking, and background orchestration
- warmed vLLM parser engines via `modal.Cls`
- durable artifact storage in a Modal Volume
- job status and idempotency tracking in Modal Dicts
- versioned `1.0` JSON output plus markdown and text artifacts

Out of scope in this initial scaffold:

- web UI
- structured extraction stage
- benchmark harness

## Runtime profiles

- `prod`
  - model: `Qwen/Qwen3.5-27B-FP8`
  - GPU: `H100`
  - vLLM: nightly wheels from `https://wheels.vllm.ai/nightly`
  - boot mode: standard startup
- `dev`
  - model: `Qwen/Qwen3.5-27B-FP8`
  - GPU: `H100`
  - vLLM: nightly wheels from `https://wheels.vllm.ai/nightly`
  - boot mode: `enforce_eager=True` for faster cold-start debugging

The public API does not expose profile selection. The deployed web endpoint defaults to `prod`. Local smoke testing can target `dev`.

The vLLM image setup follows Modal's `vllm_inference` example closely:

- CUDA base image: `nvidia/cuda:12.9.0-devel-ubuntu22.04`
- `modal.Image.from_registry(...).entrypoint([]).uv_pip_install(...)`
- Hugging Face and vLLM cache volumes mounted into the container
- explicit `HF_HOME` / `HF_HUB_CACHE` paths inside the mounted volumes
- `enforce_eager` used as the fast-boot tradeoff in the debug profile

For `Qwen/Qwen3.5-27B-FP8`, the worker does not use the older `vllm==0.13.0` example literally. It keeps the same Modal image pattern, but installs nightly `vllm` with an explicit CUDA torch backend (`cu129`) so the build does not fall back to CPU-only `torch`.

## Local setup

The project targets Python `3.12`.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

If `modal` is installed with `pipx`, either add it to `PATH`:

```bash
export PATH="$HOME/.local/bin:$PATH"
```

or call it directly:

```bash
/Users/jmccall/.local/bin/modal --version
```

## Main commands

Serve the MCP endpoint locally on Modal:

```bash
PATH="$HOME/.local/bin:$PATH" modal serve app.py
```

Run the end-to-end smoke test against the `dev` profile:

```bash
PATH="$HOME/.local/bin:$PATH" modal run app.py::smoke_test --runtime-profile-name dev
```

Download a completed job's result bundle locally:

```bash
PATH="$HOME/.local/bin:$PATH" modal run app.py::download_result --job-id <job_id>
```

Stage a local file for `upload_ref` usage:

```bash
PATH="$HOME/.local/bin:$PATH" modal run app.py::stage_upload --path ./sample.pdf
```

Seed the Hugging Face cache volume before running inference:

```bash
PATH="$HOME/.local/bin:$PATH" modal run app.py::cache_model_weights --runtime-profile-name dev
```

Run retention cleanup once:

```bash
PATH="$HOME/.local/bin:$PATH" modal run app.py::cleanup_now
```

## MCP tools

The server exposes exactly three tools:

- `submit_document_parse`
- `get_document_parse_status`
- `get_document_parse_result`

The ASGI wrapper also exposes `GET /healthz`.

## Testing

Run unit tests:

```bash
pytest
```

Run live Modal tests explicitly:

```bash
RUN_MODAL_TESTS=1 pytest tests/integration/test_modal_smoke.py
```

## Notes

- JSON output is always persisted, even if the request only asks for markdown or text.
- The parser engine no longer performs a heavyweight startup `chat()` warmup by default. vLLM still performs its own engine initialization; set `DOC_PARSE_STARTUP_WARMUP=1` only if you explicitly want an extra startup probe.
- The orchestrator uses `spawn_map` for background chunk dispatch and then polls storage until all page results are terminal.
- Page images are currently persisted for all jobs because the parser workers consume them from the artifacts volume.
- Canonical result artifacts are stored in the Modal volume `doc-parse-artifacts` under `/jobs/<job_id>/result/`.
- `smoke_test` also writes a local copy of the final JSON, markdown, text, and artifact paths under `./tmp/job-results/<job_id>/`.
- The parser worker uses vLLM's offline `LLM.chat(...)` API for page batching, but the container and boot configuration are intentionally aligned with Modal's `vllm_inference` example.
- The HF cache volume avoids repeated downloads, but each cold container still has to load the model into process and GPU memory. If you invoke `modal run ...` repeatedly, expect disk cache reuse but not reuse of a warm in-memory model from a previous app run.
