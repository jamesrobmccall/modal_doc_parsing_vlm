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
  - vLLM: pinned to `vllm` main commit `8e1fd5baf0ff272936618bf578533d9aa7080a27`
- `dev`
  - model: `Qwen/Qwen2.5-VL-7B-Instruct`
  - GPU: `L40S`
  - vLLM: `0.13.0`

The public API does not expose profile selection. The deployed web endpoint defaults to `prod`. Local smoke testing can target `dev`.

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

Stage a local file for `upload_ref` usage:

```bash
PATH="$HOME/.local/bin:$PATH" modal run app.py::stage_upload --path ./sample.pdf
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
- The orchestrator uses `spawn_map` for background chunk dispatch and then polls storage until all page results are terminal.
- Page images are currently persisted for all jobs because the parser workers consume them from the artifacts volume.
