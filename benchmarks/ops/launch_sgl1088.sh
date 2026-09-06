#!/bin/bash
# Boot the SGLang c2kv server (the default eval backend).
# Vendored from ~/bench_logs/sgl_deploy/launch_sgl1088.sh (2026-09-03);
# absolute paths overridable via env so any checkout/serve target works.
#
# Requires the consolidated SGLang task/bdf-pilot source tree.
# base is the original lowercase-qkv rule; gist is the later local fork
# rule. Choose explicitly for checkpoints trained under the latter.
SGLANG_DIR=${SGLANG_DIR:?Set SGLANG_DIR to the consolidated SGLang checkout}
PYTHON_BIN=${PYTHON_BIN:-/home/liuyancheng/envs/sgl/bin/python}
MODEL_PATH=${MODEL_PATH:-/home/liuyancheng/checkpoints_upstream/checkpoint-1088}
PORT=${PORT:-35000}
DEVICE=${DEVICE:-3}
QUERY_PROJECTION=${QUERY_PROJECTION:-base}
case "$QUERY_PROJECTION" in
  gist|base) ;;
  *) echo "QUERY_PROJECTION must be gist or base" >&2; exit 2 ;;
esac
if [ ! -f "$SGLANG_DIR/python/sglang/srt/models/qwen3.py" ]; then
  echo "SGLANG_DIR does not contain the serving source: $SGLANG_DIR" >&2
  exit 2
fi
# pool tuning validated 2026-09-03 on dev3 (the 0.30/no-pool config of the
# 22fbf31 era OOMs the c2kv pool under the b0817204 layout)
MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.20}
C2KV_POOL_FRACTION=${C2KV_POOL_FRACTION:-0.06}
CONTEXT_LENGTH=${CONTEXT_LENGTH:-16384}
export http_proxy= https_proxy= HTTP_PROXY= HTTPS_PROXY=
export NO_PROXY=127.0.0.1,localhost,::1 no_proxy=127.0.0.1,localhost,::1
source /usr/local/Ascend/cann-8.5.0/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
export PYTHONPATH="${SGLANG_DIR}/python"
export HF_HUB_OFFLINE=1
cd "${SGLANG_DIR}"
SGLANG_DEBUG_MEMORY_POOL=1 \
SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=0 \
SGLANG_EMPTY_CACHE_INTERVAL=1 \
ASCEND_LAUNCH_BLOCKING=1 \
TASK_QUEUE_ENABLE=1 \
ASCEND_RT_VISIBLE_DEVICES=$DEVICE \
exec "${PYTHON_BIN}" -m sglang.launch_server \
  --model-path "${MODEL_PATH}" \
  --served-model-name c2kv-agent \
  --model-impl sglang \
  --device npu \
  --attention-backend ascend \
  --tool-call-parser qwen25 \
  --enable-c2kv \
  --c2kv-query-proj "$QUERY_PROJECTION" \
  --dtype bfloat16 \
  --mem-fraction-static "$MEM_FRACTION_STATIC" \
  --c2kv-pool-fraction "$C2KV_POOL_FRACTION" \
  --context-length "$CONTEXT_LENGTH" \
  --disable-cuda-graph \
  --host 127.0.0.1 \
  --port "$PORT"
