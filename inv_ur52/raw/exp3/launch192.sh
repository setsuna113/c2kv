#!/usr/bin/env bash
# 192fork server: instrumented T2 copy, graph mode, same params as tools-norm52 run
T2=$(cat /tmp/c2kv-192fork.T2)
source /usr/local/Ascend/cann-8.5.0/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
cd "$T2"
mkdir -p "$T2/out"
SGLANG_DEBUG_MEMORY_POOL=1 \
SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=0 \
SGLANG_EMPTY_CACHE_INTERVAL=1 \
ASCEND_LAUNCH_BLOCKING=1 \
TASK_QUEUE_ENABLE=1 \
no_proxy='*' NO_PROXY='*' http_proxy='' https_proxy='' HTTP_PROXY='' HTTPS_PROXY='' \
EXP_HOOKS=1 EXP_OUT_DIR="$T2/out" \
C2KV_REPAIR_EXTRACT_ATTN_IMPL=prompt_flash \
C2KV_DEBUG_POSITIONS=1 \
ASCEND_RT_VISIBLE_DEVICES=6 \
PYTHONPATH="$T2/sglang/python" \
exec /home/zhuyuhan/miniconda3/envs/sglang/bin/python -m sglang.launch_server \
  --model-path /home/zhuyuhan/project/c2kv/checkpoints/qwen3-4b-agent-history-c2kv-toolcall-npu-v2/checkpoint-1088 \
  --served-model-name Qwen/Qwen3-4B-Instruct-2507-FC \
  --model-impl sglang \
  --device npu \
  --attention-backend ascend \
  --tool-call-parser qwen25 \
  --enable-c2kv \
  --c2kv-pool-fraction 0.06 \
  --dtype bfloat16 \
  --mem-fraction-static 0.55 \
  --host 127.0.0.1 \
  --port 34780
