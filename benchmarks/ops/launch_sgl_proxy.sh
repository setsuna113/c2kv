#!/bin/bash
# launch_sgl_proxy.sh <arm> <port> <upstream_port> [suffix] [extra proxy args...]
# Arm proxy in front of the SGLang server (backend sglang is explicit).
# Vendored from ~/bench_logs/launch_sgl_proxy.sh (2026-09-03).
REPO_DIR=${REPO_DIR:-/home/liuyancheng/c2kv-bench}
LOG_DIR=${LOG_DIR:-/home/liuyancheng/bench_logs}
PY=${PY:-/home/liuyancheng/envs/bench/bin/python}
ARM=$1; PORT=$2; UP=$3; SUF=${4:-$ARM}; shift 4 2>/dev/null
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost
for p in $(ps -ef | grep "benchmarks/proxy.py" | grep "port $PORT" | grep -v grep | awk "{print \$2}"); do kill $p 2>/dev/null; done
cd "$REPO_DIR"
STAMP=$(date +%H%M%S)
setsid nohup "$PY" -u benchmarks/proxy.py \
  --upstream http://127.0.0.1:$UP --backend sglang --arm $ARM --port $PORT \
  --request-log "$LOG_DIR/proxy_task_$SUF.jsonl" \
  "$@" >> "$LOG_DIR/proxy_${PORT}_${STAMP}.log" 2>&1 &
echo "sgl-proxy $ARM pid $! :$PORT -> :$UP log=proxy_${PORT}_${STAMP}.log extra=$*"
