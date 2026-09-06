#!/bin/bash
# launch_sgl_proxy.sh <arm> <port> <upstream_port> [suffix] [extra proxy args...]
# Arm proxy in front of the SGLang server (backend sglang is explicit).
# Vendored from ~/bench_logs/launch_sgl_proxy.sh (2026-09-03).
REPO_DIR=${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
LOG_DIR=${LOG_DIR:-/home/liuyancheng/bench_logs}
PY=${PY:-/home/liuyancheng/envs/bench/bin/python}
if [ "$#" -lt 3 ]; then
  echo "usage: $0 <arm> <port> <upstream_port> [suffix] [extra proxy args...]" >&2
  exit 2
fi
ARM=$1; PORT=$2; UP=$3; shift 3
SUF=$ARM
if [ "$#" -gt 0 ] && [[ "$1" != --* ]]; then
  SUF=$1; shift
fi
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost
"$PY" -c 'import socket,sys; s=socket.socket(); s.bind(("127.0.0.1", int(sys.argv[1]))); s.close()' "$PORT" || exit 2
cd "$REPO_DIR" || exit 2
mkdir -p "$LOG_DIR" || exit 2
STAMP=$(date +%H%M%S)
setsid nohup "$PY" -u benchmarks/proxy.py \
  --upstream "http://127.0.0.1:$UP" --backend sglang --arm "$ARM" --port "$PORT" \
  --request-log "$LOG_DIR/proxy_task_$SUF.jsonl" \
  "$@" >> "$LOG_DIR/proxy_${PORT}_${STAMP}.log" 2>&1 &
echo "sgl-proxy $ARM pid $! :$PORT -> :$UP log=proxy_${PORT}_${STAMP}.log extra=$*"
