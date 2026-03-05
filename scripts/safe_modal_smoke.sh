#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODAL_BIN="${MODAL_BIN:-.venv/bin/modal}"
if [[ ! -x "$MODAL_BIN" ]]; then
  MODAL_BIN="$(command -v modal)"
fi

if [[ -z "${MODAL_BIN:-}" ]]; then
  echo "modal CLI not found. Set MODAL_BIN or install modal." >&2
  exit 1
fi

cleanup() {
  echo "[cleanup] marking stale jobs failed..."
  "$MODAL_BIN" run app.py::cleanup_stale_now >/dev/null 2>&1 || true

  echo "[cleanup] stopping running apps..."
  "$MODAL_BIN" app list --json 2>/dev/null \
    | python3 -c 'import json,sys; data=json.load(sys.stdin); print("\n".join(item["App ID"] for item in data if item.get("State") in {"running", "deployed"}))' \
    | while IFS= read -r app_id; do
        [[ -n "$app_id" ]] || continue
        "$MODAL_BIN" app stop "$app_id" >/dev/null 2>&1 || true
      done

  echo "[cleanup] stopping running containers..."
  "$MODAL_BIN" container list --json 2>/dev/null \
    | python3 -c 'import json,sys; data=json.load(sys.stdin); print("\n".join(item["container_id"] for item in data))' \
    | while IFS= read -r container_id; do
        [[ -n "$container_id" ]] || continue
        "$MODAL_BIN" container stop "$container_id" >/dev/null 2>&1 || true
      done
}

trap cleanup EXIT INT TERM

RUNTIME_PROFILE="${RUNTIME_PROFILE:-dev}"
LATENCY_PROFILE="${LATENCY_PROFILE:-fast}"
RESULT_LEVEL="${RESULT_LEVEL:-latest}"

echo "[run] smoke_test runtime_profile=${RUNTIME_PROFILE} latency_profile=${LATENCY_PROFILE} result_level=${RESULT_LEVEL}"
"$MODAL_BIN" run app.py::smoke_test \
  --runtime-profile-name "$RUNTIME_PROFILE" \
  --latency-profile "$LATENCY_PROFILE" \
  --result-level "$RESULT_LEVEL" \
  "$@"
