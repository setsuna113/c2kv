#!/bin/bash
# Boot the SGLang c2kv server (the default eval backend).
# Vendored from ~/bench_logs/sgl_deploy/launch_sgl1088.sh (2026-09-03);
# absolute paths overridable via env so any checkout/serve target works.
#
# Requires the SGLang fork branch task/c2kv-serve-align (see
# benchmarks/backends/sglang_patches/README.md for the deployment recipe:
# codeload tarball + in-repo patches, PYTHONPATH precedence over the
# editable install in the sgl venv).  Default tree: the b0817204 codeload
# extract (includes the detokenizer kv_runtime_stats fix 425cd6573, applied
# in ~/kvoffload-sglang-c2kv and the tarball).
SGLANG_DIR=${SGLANG_DIR:-/home/liuyancheng/sgl-b0817204}
PYTHON_BIN=${PYTHON_BIN:-/home/liuyancheng/envs/sgl/bin/python}
MODEL_PATH=${MODEL_PATH:-/home/liuyancheng/checkpoints_upstream/checkpoint-1088}
PORT=${PORT:-35000}
DEVICE=${DEVICE:-3}
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
"${PYTHON_BIN}" -m sglang.launch_server \
  --model-path "${MODEL_PATH}" \
  --served-model-name c2kv-agent \
  --model-impl sglang \
  --device npu \
  --attention-backend ascend \
  --tool-call-parser qwen25 \
  --enable-c2kv \
  --c2kv-query-proj gist \
  --dtype bfloat16 \
  --mem-fraction-static "$MEM_FRACTION_STATIC" \
  --c2kv-pool-fraction "$C2KV_POOL_FRACTION" \
  --context-length "$CONTEXT_LENGTH" \
  --disable-cuda-graph \
  --host 127.0.0.1 \
  --port "$PORT"
