#!/usr/bin/env bash
# S4 diagnostic arms: forced action-prefix on the agent history eval.
# Complements eval_agent_history_c2kv_npu.sh with per-arm invocation; new flags default off.
# Usage: S4_ARM=C bash agent/eval_agent_history_s4_npu.sh
# Arms: A=full+free  B=c2kv@4+free  C=c2kv@4+forced  D=full+forced
# (Arm E = negative-target slice of C, computed at analysis time.)
set -euo pipefail

export PYTHONPATH="$(pwd)/python:$(pwd)/python/inference:$(pwd)/agent:${PYTHONPATH:-}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"

MODEL_PATH="${MODEL_PATH:-./checkpoints/qwen3-4b-agent-history-c2kv-npu}"
BASE_MODEL="${BASE_MODEL:-./models/Qwen3-4B-Instruct-2507}"
TOKENIZER_PATH="${HISTORY_TOKENIZER_PATH:-${BASE_MODEL}}"
DATASET_PATH="${DATASET_PATH:-./datasets/agent-llm-traces}"
SPLIT="${SPLIT:-eval}"
SPLIT_NAME="${SPLIT_NAME:-subset_disjoint}"
SPLIT_MANIFEST_FILE="${SPLIT_MANIFEST_FILE:-}"
SPLIT_SEED="${SPLIT_SEED:-42}"
EVAL_RATIO="${EVAL_RATIO:-0.1}"
MAX_SAMPLES_PER_SESSION="${MAX_SAMPLES_PER_SESSION:-4}"
INCLUDE_TOOLS="${INCLUDE_TOOLS:-True}"
MAX_EXAMPLES="${MAX_EXAMPLES:-0}"
SAMPLE_SEED="${SAMPLE_SEED:-}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
RATIO="${RATIO:-4}"
NPU_ATTN_IMPL="${NPU_ATTN_IMPL:-eager}"

ARM="${S4_ARM:?set S4_ARM to A/B/C/D}"
case "${ARM}" in
  A) COMPARE_MODE="full"; FORCE=0 ;;
  B) COMPARE_MODE="c2kv"; FORCE=0 ;;
  C) COMPARE_MODE="c2kv"; FORCE=1 ;;
  D) COMPARE_MODE="full"; FORCE=1 ;;
  *) echo "unknown S4_ARM=${ARM}" >&2; exit 2 ;;
esac

FORCE_SUFFIX=""
if [[ "${FORCE}" == "1" ]]; then
  FORCE_SUFFIX="_forced"
fi
OUTPUT_FILE="${OUTPUT_FILE:-./outputs/s4_arm${ARM}_${COMPARE_MODE}${FORCE_SUFFIX}.jsonl}"
mkdir -p "$(dirname "${OUTPUT_FILE}")"

FORCE_ARGS=()
if [[ "${FORCE}" == "1" ]]; then
  FORCE_ARGS+=(--force_action_prefix)
fi
SAMPLE_ARGS=()
if [[ -n "${SAMPLE_SEED}" ]]; then
  SAMPLE_ARGS+=(--sample_seed "${SAMPLE_SEED}")
fi
SPLIT_ARGS=(--split_manifest_name "${SPLIT_NAME}")
if [[ -n "${SPLIT_MANIFEST_FILE}" ]]; then
  SPLIT_ARGS=(--split_manifest_file "${SPLIT_MANIFEST_FILE}" --split_manifest_name "${SPLIT_NAME}")
fi

python agent/eval_agent_history_c2kv.py \
  --device_type npu \
  --model "${MODEL_PATH}" \
  --base_model "${BASE_MODEL}" \
  --tokenizer "${TOKENIZER_PATH}" \
  --dataset_path "${DATASET_PATH}" \
  --split "${SPLIT}" \
  "${SPLIT_ARGS[@]}" \
  --split_seed "${SPLIT_SEED}" \
  --eval_ratio "${EVAL_RATIO}" \
  --max_samples_per_session "${MAX_SAMPLES_PER_SESSION}" \
  --include_tools "${INCLUDE_TOOLS}" \
  --compare_modes "${COMPARE_MODE}" \
  --ratios "${RATIO}" \
  --max_examples "${MAX_EXAMPLES}" \
  "${SAMPLE_ARGS[@]}" \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  --system_attn_impl "${NPU_ATTN_IMPL}" \
  --gist_attn_impl "${NPU_ATTN_IMPL}" \
  --generate_attn_impl "${NPU_ATTN_IMPL}" \
  "${FORCE_ARGS[@]}" \
  --output_file "${OUTPUT_FILE}"
