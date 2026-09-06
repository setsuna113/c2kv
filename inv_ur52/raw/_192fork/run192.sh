#!/usr/bin/env bash
# $1 = arm (d_corr_w2 | d_corr_replace_w2)
T2=$(cat /tmp/c2kv-192fork.T2)
ARM="$1"
ARM_ROOT="$T2/runs/$ARM"
mkdir -p "${ARM_ROOT}"/result "${ARM_ROOT}"/score "${ARM_ROOT}"/logs
cd "$T2/client"
exec /home/zhuyuhan/miniconda3/envs/bfcl/bin/python -m c2kv_eval.adapters.bfcl_history_kv_repair \
  --arm "${ARM}" \
  --category multi_turn_base \
  --max-examples 1 \
  --ids-path "$T2/ids192.txt" \
  --reference-details-path "$T2/inputs/details.jsonl" \
  --plan-path "" \
  --base-url "http://127.0.0.1:34780" \
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
