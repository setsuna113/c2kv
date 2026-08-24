#!/usr/bin/env bash
# Big-tool-pool baseline harness: raises the tool-definition budget to 32k tokens
# (MAX_DOC_NUM=64) and runs the two non-compression baselines under the
# toolset_disjoint split. Compression arms (c2kv/hybrid) are intentionally not
# part of this harness.
# Usage: BIGPOOL_ARM=full bash agent/eval_agent_tool_definition_bigpool_npu.sh
#        BIGPOOL_ARM=topk_only TOP_K=3 bash agent/eval_agent_tool_definition_bigpool_npu.sh
set -euo pipefail

export PYTHONPATH="$(pwd)/python:$(pwd)/python/inference:$(pwd)/agent:${PYTHONPATH:-}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"

MODEL_PATH="${MODEL_PATH:-./checkpoints/qwen3-4b-agent-tooldef-npu}"
BASE_MODEL="${BASE_MODEL:-./models/Qwen3-4B-Instruct-2507}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${MODEL_PATH}}"
DATASET_PATH="${DATASET_PATH:-./datasets/agent-llm-traces}"
SPLIT="${SPLIT:-eval}"
SPLIT_MANIFEST_FILE="${SPLIT_MANIFEST_FILE:-./configs/agent_tooldef_split_manifests.json}"
SPLIT_NAME="${SPLIT_NAME:-toolset_disjoint}"
EVAL_RATIO="${EVAL_RATIO:-0.1}"
SPLIT_SEED="${SPLIT_SEED:-42}"
MAX_SAMPLES_PER_SESSION="${MAX_SAMPLES_PER_SESSION:-16}"
MIN_TARGET_TOKENS="${MIN_TARGET_TOKENS:-128}"
MAX_EXAMPLES="${MAX_EXAMPLES:-0}"
MAX_DOC_LENGTH="${MAX_DOC_LENGTH:-1024}"
MAX_DOC_NUM="${MAX_DOC_NUM:-64}"
MAX_TOOL_DEFINITION_TOKENS="${MAX_TOOL_DEFINITION_TOKENS:-32000}"
MAX_BASELINE_INPUT_TOKENS="${MAX_BASELINE_INPUT_TOKENS:-34000}"
TOOL_DOCUMENT_EVAL_MODE="${TOOL_DOCUMENT_EVAL_MODE:-full}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
NPU_ATTN_IMPL="${NPU_ATTN_IMPL:-eager}"
# 32k-context full prefills OOM with eager attention (fp32 q×kv softmax matrix);
# the full arm therefore defaults to the NPU fusion attention kernel.
FULL_ATTN_IMPL="${FULL_ATTN_IMPL:-npu_fusion_attention}"
TOP_K="${TOP_K:-3}"
RATIO="${RATIO:-4}"

# R2 S1/S2 arms (default off): prior = topk_only with TOP_K=0 (no tool
# documents retained — pure model-prior floor); topk_semantic = topk_only with
# the CPU BM25 ranker. topk_only keeps its lexical default unchanged.
ARM="${BIGPOOL_ARM:?set BIGPOOL_ARM to full, c2kv, topk_only, topk_semantic, or prior}"
ROUTER_STRATEGY="${ROUTER_STRATEGY:-lexical}"
case "${ARM}" in
  topk_semantic) ROUTER_STRATEGY=bm25 ;;
  prior) TOP_K=0 ;;
esac
case "${ARM}" in
  full|c2kv) SUFFIX="" ;;
  *) SUFFIX="_k${TOP_K}" ;;
esac
OUTPUT_FILE="${OUTPUT_FILE:-./outputs/bigpool_${ARM}${SUFFIX}.jsonl}"
mkdir -p "$(dirname "${OUTPUT_FILE}")"

SPLIT_ARGS=(--split "${SPLIT}" --split_manifest_name "${SPLIT_NAME}")
if [[ -n "${SPLIT_MANIFEST_FILE}" ]]; then
  SPLIT_ARGS+=(--split_manifest_file "${SPLIT_MANIFEST_FILE}")
fi

echo "ARM=${ARM} MAX_TOOL_DEFINITION_TOKENS=${MAX_TOOL_DEFINITION_TOKENS} MAX_DOC_NUM=${MAX_DOC_NUM}"
echo "SPLIT_MANIFEST_FILE=${SPLIT_MANIFEST_FILE} SPLIT_NAME=${SPLIT_NAME}"
echo "MAX_SAMPLES_PER_SESSION=${MAX_SAMPLES_PER_SESSION} MAX_EXAMPLES=${MAX_EXAMPLES} OUTPUT_FILE=${OUTPUT_FILE}"

if [[ "${ARM}" == "full" ]]; then
  python agent/eval_agent_tool_definition_c2kv.py \
    --device_type npu \
    --model "${MODEL_PATH}" \
    --base_model "${BASE_MODEL}" \
    --tokenizer "${TOKENIZER_PATH}" \
    --dataset_path "${DATASET_PATH}" \
    --output_file "${OUTPUT_FILE}" \
    "${SPLIT_ARGS[@]}" \
    --eval_ratio "${EVAL_RATIO}" \
    --split_seed "${SPLIT_SEED}" \
    --max_samples_per_session "${MAX_SAMPLES_PER_SESSION}" \
    --min_target_tokens "${MIN_TARGET_TOKENS}" \
    --max_examples "${MAX_EXAMPLES}" \
    --max_doc_length "${MAX_DOC_LENGTH}" \
    --max_doc_num "${MAX_DOC_NUM}" \
    --max_tool_definition_tokens "${MAX_TOOL_DEFINITION_TOKENS}" \
    --tool_document_eval_mode "${TOOL_DOCUMENT_EVAL_MODE}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --max_baseline_input_tokens "${MAX_BASELINE_INPUT_TOKENS}" \
    --mode full \
    --system_attn_impl "${FULL_ATTN_IMPL}" \
    --gist_attn_impl "${FULL_ATTN_IMPL}" \
    --generate_attn_impl "${FULL_ATTN_IMPL}" \
    --truncate_tool_definition False \
    --require_tool_call True
elif [[ "${ARM}" == "c2kv" ]]; then
  # R2 退化阶梯臂：c2kv 压缩（ratio 默认 4）在当前预算 cap 下评测。
  python agent/eval_agent_tool_definition_c2kv.py \
    --device_type npu \
    --model "${MODEL_PATH}" \
    --base_model "${BASE_MODEL}" \
    --tokenizer "${TOKENIZER_PATH}" \
    --dataset_path "${DATASET_PATH}" \
    --output_file "${OUTPUT_FILE}" \
    "${SPLIT_ARGS[@]}" \
    --eval_ratio "${EVAL_RATIO}" \
    --split_seed "${SPLIT_SEED}" \
    --max_samples_per_session "${MAX_SAMPLES_PER_SESSION}" \
    --min_target_tokens "${MIN_TARGET_TOKENS}" \
    --max_examples "${MAX_EXAMPLES}" \
    --max_doc_length "${MAX_DOC_LENGTH}" \
    --max_doc_num "${MAX_DOC_NUM}" \
    --max_tool_definition_tokens "${MAX_TOOL_DEFINITION_TOKENS}" \
    --tool_document_eval_mode "${TOOL_DOCUMENT_EVAL_MODE}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --max_baseline_input_tokens "${MAX_BASELINE_INPUT_TOKENS}" \
    --mode c2kv \
    --ratios "${RATIO}" \
    --system_attn_impl "${NPU_ATTN_IMPL}" \
    --gist_attn_impl "${NPU_ATTN_IMPL}" \
    --generate_attn_impl "${NPU_ATTN_IMPL}" \
    --truncate_tool_definition False \
    --require_tool_call True
elif [[ "${ARM}" == "topk_only" || "${ARM}" == "topk_semantic" || "${ARM}" == "prior" ]]; then
  python agent/eval_agent_tool_definition_hybrid_router.py \
    --device_type npu \
    --model "${MODEL_PATH}" \
    --base_model "${BASE_MODEL}" \
    --tokenizer "${TOKENIZER_PATH}" \
    --dataset_path "${DATASET_PATH}" \
    --output_file "${OUTPUT_FILE}" \
    "${SPLIT_ARGS[@]}" \
    --eval_ratio "${EVAL_RATIO}" \
    --split_seed "${SPLIT_SEED}" \
    --max_samples_per_session "${MAX_SAMPLES_PER_SESSION}" \
    --min_target_tokens "${MIN_TARGET_TOKENS}" \
    --max_examples "${MAX_EXAMPLES}" \
    --max_doc_length "${MAX_DOC_LENGTH}" \
    --max_doc_num "${MAX_DOC_NUM}" \
    --max_tool_definition_tokens "${MAX_TOOL_DEFINITION_TOKENS}" \
    --tool_document_eval_mode "${TOOL_DOCUMENT_EVAL_MODE}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --hybrid_cases "${TOP_K}:${RATIO}" \
    --hybrid_mode topk_only \
    --router_strategy "${ROUTER_STRATEGY}" \
    --router_hit_filter all \
    --system_attn_impl "${NPU_ATTN_IMPL}" \
    --gist_attn_impl "${NPU_ATTN_IMPL}" \
    --generate_attn_impl "${NPU_ATTN_IMPL}" \
    --truncate_tool_definition False \
    --require_tool_call True
else
  echo "unknown BIGPOOL_ARM=${ARM}" >&2
  exit 2
fi
