#!/bin/bash
# Boot the SGLang c2kv server (the default eval backend).
# Vendored from ~/bench_logs/sgl_deploy/launch_sgl1088.sh (2026-09-03);
# absolute paths overridable via env so any checkout/serve target works.
#
# Requires the SGLang fork branch task/c2kv-serve-align (see
# benchmarks/backends/sglang_patches/README.md for the deployment recipe:
# codeload tarball + in-repo patches, PYTHONPATH precedence over the
# editable install in the sgl venv).
SGLANG_DIR=${SGLANG_DIR:-/home/liuyancheng/sgl-22fbf3146}
PYTHON_BIN=${PYTHON_BIN:-/home/liuyancheng/envs/sgl/bin/python}
MODEL_PATH=${MODEL_PATH:-/home/liuyancheng/checkpoints_upstream/checkpoint-1088}
PORT=${PORT:-35000}
DEVICE=${DEVICE:-3}
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
  --mem-fraction-static 0.30 --disable-cuda-graph \
  --host 127.0.0.1 \
  --port "$PORT"
