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

MODAL_ENVIRONMENT="${MODAL_ENVIRONMENT:-main}"
MODAL_APP_NAME="${MODAL_APP_NAME:-modal-doc-parsing-vlm}"

cleanup() {
  echo "[cleanup] marking stale jobs failed..."
  "$MODAL_BIN" run -e "$MODAL_ENVIRONMENT" app.py::cleanup_stale_now >/dev/null 2>&1 || true

  echo "[cleanup] stopping ${MODAL_APP_NAME} in ${MODAL_ENVIRONMENT}..."
  "$MODAL_BIN" app stop "$MODAL_APP_NAME" -e "$MODAL_ENVIRONMENT" >/dev/null 2>&1 || true

  echo "[cleanup] stopping running containers in ${MODAL_ENVIRONMENT}..."
  "$MODAL_BIN" container list -e "$MODAL_ENVIRONMENT" --json 2>/dev/null \
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

echo "[run] smoke_test env=${MODAL_ENVIRONMENT} runtime_profile=${RUNTIME_PROFILE} latency_profile=${LATENCY_PROFILE} result_level=${RESULT_LEVEL}"
"$MODAL_BIN" run -e "$MODAL_ENVIRONMENT" app.py::smoke_test \
  --runtime-profile-name "$RUNTIME_PROFILE" \
  --latency-profile "$LATENCY_PROFILE" \
  --result-level "$RESULT_LEVEL" \
  "$@"
