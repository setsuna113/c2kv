#!/usr/bin/env bash
# $1 = tag, $2 = port
TAG="$1"; PORT="$2"
ROOT=/home/zhuyuhan/project/gorilla/berkeley-function-call-leaderboard
ARM_ROOT=/tmp/zh_exp/replay_${TAG}/d_corr_w2
mkdir -p "${ARM_ROOT}"/result "${ARM_ROOT}"/score "${ARM_ROOT}"/logs
printf 'multi_turn_base_110\nmulti_turn_base_122\nmulti_turn_base_136\n' > /tmp/zh_exp/ids3.txt
cd "${ROOT}"
exec /home/zhuyuhan/miniconda3/envs/bfcl/bin/python -m c2kv_eval.adapters.bfcl_history_kv_repair \
  --arm d_corr_w2 \
  --category multi_turn_base \
  --max-examples 3 \
  --ids-path /tmp/zh_exp/ids3.txt \
  --reference-details-path /home/zhuyuhan/project/gorilla/bfcl_runs/history_full_temp0_stability_20260819_172725/frozen_reference/details.jsonl \
  --plan-path "" \
  --base-url "http://127.0.0.1:${PORT}" \
  --model Qwen/Qwen3-4B-Instruct-2507-FC \
  --served-model-name Qwen/Qwen3-4B-Instruct-2507-FC \
  --tokenizer-path /home/zhuyuhan/project/c2kv/models/Qwen3-4B-Instruct-2507 \
  --ratio 4 \
  --checkpoint-interval 4 \
  --repair-window 1 \
  --repair-extract-source auto \
  --c2kv-append-position-frame wrapper \
  --repair-trigger oracle \
  --max-completion-tokens 4096 \
  --result-dir "${ARM_ROOT}/result" \
  --details-path "${ARM_ROOT}/logs/details.jsonl" \
  --metrics-path "${ARM_ROOT}/logs/metrics.jsonl" \
  --summary-path "${ARM_ROOT}/logs/summary.json" \
  --temperature 0
