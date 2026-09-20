#!/bin/bash
# Generality experiment NPU engine launcher (sglang-gen = sglang-paper@d2ca37175 lineage).
# Usage: launch_engine.sh <card> <port> <logtag> [extra sglang args...]
set -eo pipefail
CARD=${1:?card}; PORT=${2:?port}; TAG=${3:?logtag}; shift 3
if [[ ! $CARD =~ ^[0-9]+$ || ! $PORT =~ ^[0-9]+$ ]]; then
  echo "engine launch refused: card and port must be integers" >&2
  exit 64
fi
CARD=$((10#$CARD))
PORT=$((10#$PORT))
if (( CARD > 7 || PORT < 1 || PORT > 65535 )); then
  echo "engine launch refused: card or port is out of range" >&2
  exit 64
fi
for arg in "$@"; do
  case "$arg" in
    --port|--port=*|--base-gpu-id|--base-gpu-id=*)
      echo "engine launch refused: extra arguments cannot override card or port" >&2
      exit 64
      ;;
  esac
done
ROOT=/home/liuyancheng/c2kv-generality-20260918
CKPT=/home/liuyancheng/c2kv-b-final-20260912/checkpoints/b_history/arm-C/seed-42/checkpoint-1000

# Keep both locks across exec so another launcher cannot start on this card or port.
LOCK_DIR=$ROOT/logs/engines
mkdir -p "$LOCK_DIR"
exec 9>"$LOCK_DIR/.port-$PORT.lock"
if ! flock -n 9; then
  echo "engine launch refused: port $PORT has an active launcher" >&2
  exit 75
fi
exec 8>"$LOCK_DIR/.card-$CARD.lock"
if ! flock -n 8; then
  echo "engine launch refused: card $CARD has an active launcher" >&2
  exit 75
fi

# Older engines did not hold these locks. Reject their ports before model load.
if ss -H -ltn "sport = :$PORT" | grep -q .; then
  echo "engine launch refused: port $PORT already has a listener" >&2
  exit 75
fi
if pgrep -af 'sglang.launch_server' | grep -Eq -- "(^|[[:space:]])--port(=|[[:space:]])${PORT}([[:space:]]|$)"; then
  echo "engine launch refused: port $PORT has an engine starting" >&2
  exit 75
fi
while read -r pid _; do
  if tr '\0' '\n' <"/proc/$pid/environ" 2>/dev/null | grep -Fxq "ASCEND_RT_VISIBLE_DEVICES=$CARD"; then
    echo "engine launch refused: card $CARD is held by engine pid $pid" >&2
    exit 75
  fi
done < <(pgrep -af 'sglang.launch_server' || true)

source /usr/local/Ascend/cann-8.5.0/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY || true
export no_proxy=127.0.0.1,localhost
export NO_PROXY=127.0.0.1,localhost
export PYTHONUNBUFFERED=1
export ASCEND_RT_VISIBLE_DEVICES=$CARD
export PYTHONPATH="${C2KV_SGLANG_SOURCE:-$ROOT/src/sglang-gen}/python"
export C2KV_PAPER_TELEMETRY=1
export C2KV_PAPER_TELEMETRY_LOG=$ROOT/logs/engines/${TAG}_server_telemetry.jsonl
exec /home/liuyancheng/envs/sgl/bin/python -m sglang.launch_server \
  --model-path "$CKPT" --served-model-name gen-c1000 --model-impl sglang \
  --device npu --attention-backend ascend --dtype bfloat16 \
  --enable-c2kv --c2kv-gist-type dynamic-interleave --c2kv-gist-param qkv \
  --c2kv-query-proj base --c2kv-pool-fraction 0.05 \
  --c2kv-shadow-feature-layer -2 --enable-return-hidden-states \
  --mem-fraction-static 0.55 --max-total-tokens 65536 --context-length 131072 \
  --max-running-requests 1 --page-size 128 --chunked-prefill-size 256 \
  --disable-radix-cache --disable-cuda-graph --enable-streaming-session \
  --host 127.0.0.1 --port "$PORT" "$@"
