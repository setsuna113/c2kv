#!/usr/bin/env bash
set -Eeuo pipefail

: "${CKPT:?set CKPT=/absolute/path/to/gist/checkpoint}"

GU_BASE="${GU_BASE:-/inspire/hdd/global_user/yanjunchi-24040}"
BENCH_ROOT="${BENCH_ROOT:-$GU_BASE/bench-sglang-h200}"
SGLANG_VENV="${SGLANG_VENV:-$BENCH_ROOT/venv-sglang}"
SGLANG_PYTHON="${SGLANG_PYTHON:-$SGLANG_VENV/bin/python}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen/Qwen3-4B-Instruct-2507-FC}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-34000}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.8}"
C2KV_POOL_FRACTION="${C2KV_POOL_FRACTION:-0.01}"
C2KV_MAX_TOKENS="${C2KV_MAX_TOKENS:-4096}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-qwen25}"
HEALTH_TIMEOUT_SEC="${HEALTH_TIMEOUT_SEC:-1800}"
SGLANG_LOG="${SGLANG_LOG:-/tmp/sglang-c2kv-${PORT}.log}"

# Qwen3 C2KV attention reads a prefix-length scalar inside a traced region.
# Torch needs this mode enabled; otherwise default piecewise CUDA graph warmup
# fails on Tensor.item() before the HTTP server becomes healthy.
export TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS="${TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS:-1}"

CKPT_PATH=$(realpath "$CKPT")
if [[ ! -f "$CKPT_PATH/config.json" || ! -f "$CKPT_PATH/model.safetensors" ]]; then
  echo "FATAL: CKPT does not look like a complete checkpoint: $CKPT_PATH" >&2
  exit 2
fi
if [[ ! -x "$SGLANG_PYTHON" ]]; then
  echo "FATAL: SGLang python not executable: $SGLANG_PYTHON" >&2
  echo "Run benchmarks/run_matrix_h200.sh once to create the isolated serving venv." >&2
  exit 2
fi

mkdir -p "$(dirname "$SGLANG_LOG")"
echo "[sglang] CKPT=$CKPT_PATH"
echo "[sglang] model=$SERVED_MODEL_NAME endpoint=$HOST:$PORT log=$SGLANG_LOG"

"$SGLANG_PYTHON" -m sglang.launch_server \
  --model-path "$CKPT_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --device cuda \
  --enable-c2kv \
  --c2kv-pool-fraction "$C2KV_POOL_FRACTION" \
  --c2kv-max-tokens "$C2KV_MAX_TOKENS" \
  --tool-call-parser "$TOOL_CALL_PARSER" \
  --mem-fraction-static "$MEM_FRACTION_STATIC" \
  --disable-piecewise-cuda-graph \
  --host "$HOST" \
  --port "$PORT" \
  >"$SGLANG_LOG" 2>&1 &

SERVER_PID=$!
cleanup() {
  if kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

started=$SECONDS
while true; do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    wait "$SERVER_PID" || true
    echo "FATAL: sglang exited before becoming healthy; log=$SGLANG_LOG" >&2
    tail -100 "$SGLANG_LOG" >&2 || true
    exit 2
  fi
  if "$SGLANG_PYTHON" - "$HOST" "$PORT" <<'PY' >/dev/null 2>&1
import sys
import urllib.request
urllib.request.urlopen(f"http://{sys.argv[1]}:{sys.argv[2]}/health", timeout=3).close()
PY
  then
    echo "[sglang] healthy after $(( SECONDS - started )) seconds (pid=$SERVER_PID)"
    break
  fi
  if (( SECONDS - started >= HEALTH_TIMEOUT_SEC )); then
    echo "FATAL: sglang health timeout after ${HEALTH_TIMEOUT_SEC}s; log=$SGLANG_LOG" >&2
    tail -100 "$SGLANG_LOG" >&2 || true
    exit 2
  fi
  sleep 5
done

wait "$SERVER_PID"
