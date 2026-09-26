#!/bin/bash
# Launch one SGLang C2KV engine pinned to a physical NPU card.
# Usage (second engine per card): engine_card2.sh <card> <port>
set -e
CARD=$1
PORT=$2
source /usr/local/Ascend/cann-8.5.0/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=$CARD
export HCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4
export TORCHINDUCTOR_COMPILE_THREADS=1
export PYTHONPATH=/home/liuyancheng/gp_search_v1/sglang/python
export NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
exec /home/liuyancheng/envs/sgl/bin/python -m sglang.launch_server \
  --model-path /home/liuyancheng/c2kv-b-final-20260912/checkpoints/b_history/arm-C/seed-42/checkpoint-1000 \
  --served-model-name d3-c1000 --model-impl sglang \
  --device npu --attention-backend ascend --dtype bfloat16 \
  --enable-c2kv --c2kv-gist-type dynamic-interleave --c2kv-gist-param qkv \
  --c2kv-query-proj base --c2kv-pool-fraction 0.05 \
  --c2kv-shadow-feature-layer -2 --enable-return-hidden-states \
  --mem-fraction-static 0.40 --max-total-tokens 65536 --context-length 131072 \
  --max-running-requests 1 --page-size 128 --chunked-prefill-size 256 \
  --disable-radix-cache --disable-cuda-graph --host 127.0.0.1 --port $PORT
