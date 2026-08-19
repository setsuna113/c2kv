#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONPATH="$(pwd)/python:$(pwd)/python/inference:${PYTHONPATH:-}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-4,5,6,7}"
export ASCEND_VISIBLE_DEVICES="${ASCEND_VISIBLE_DEVICES:-${ASCEND_RT_VISIBLE_DEVICES}}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-max_split_size_mb:128}"

MODEL_PATH="${MODEL_PATH:-/home/zhuyuhan/project/c2kv/checkpoints/qwen3-4b-mixed-mdoc-c2kv-r4-8-16-npu-10k-30k/checkpoint-1800}"
BASE_MODEL="${BASE_MODEL:-/home/zhuyuhan/project/c2kv/models/Qwen3-4B-Instruct-2507}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${MODEL_PATH}}"
DATASET="${DATASET:-hotpotqa}"
DATASET_PATH="${DATASET_PATH:-/home/zhuyuhan/project/c2kv/datasets/longbench_raw}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/zhuyuhan/project/c2kv/outputs/hybridkv_mdoc}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MAX_EXAMPLES="${MAX_EXAMPLES:-200}"
RATIO="${RATIO:-16}"
HYBRID_TOP_K="${HYBRID_TOP_K:-3}"
MAX_DOC_LENGTH="${MAX_DOC_LENGTH:-2048}"
MAX_DOC_NUM="${MAX_DOC_NUM:-0}"
MAX_CONTEXT_TOKENS="${MAX_CONTEXT_TOKENS:-0}"
MAX_QUERY_TOKENS="${MAX_QUERY_TOKENS:-1024}"
DOC_SELECTION="${DOC_SELECTION:-head}"
DEVICE_TYPE="${DEVICE_TYPE:-npu}"
if [[ "${DEVICE_TYPE}" == "cuda" ]]; then
  DEVICES_CSV="${DEVICES:-${CUDA_VISIBLE_DEVICES:-0}}"
else
  DEVICES_CSV="${DEVICES:-${ASCEND_RT_VISIBLE_DEVICES}}"
fi
SYSTEM_ATTN_IMPL="${SYSTEM_ATTN_IMPL:-eager}"
if [[ "${DEVICE_TYPE}" == "cuda" ]]; then
  GIST_ATTN_IMPL="${GIST_ATTN_IMPL:-eager}"
  GENERATE_ATTN_IMPL="${GENERATE_ATTN_IMPL:-eager}"
else
  GIST_ATTN_IMPL="${GIST_ATTN_IMPL:-npu_fusion_attention}"
  GENERATE_ATTN_IMPL="${GENERATE_ATTN_IMPL:-npu_fusion_attention}"
fi
DTYPE="${DTYPE:-bf16}"
MODES="${MODES:-full,c2kv,hybrid}"
RANK_PLANS="${RANK_PLANS:-}"
TARGET_COMPRESSION_RATIO="${TARGET_COMPRESSION_RATIO:-8}"
RECOVERY_CANDIDATE_DOCS="${RECOVERY_CANDIDATE_DOCS:-4}"
RECOVERY_SPAN_TOKENS="${RECOVERY_SPAN_TOKENS:-256,128,64}"
RECOVERY_MAX_SPANS="${RECOVERY_MAX_SPANS:-2}"
HYBRID_CHUNK_TOP_K="${HYBRID_CHUNK_TOP_K:-8}"
HYBRID_CHUNK_TOKENS="${HYBRID_CHUNK_TOKENS:-256}"
HYBRID_CHUNK_OVERLAP="${HYBRID_CHUNK_OVERLAP:-64}"
HYBRID_CHUNK_RANKER="${HYBRID_CHUNK_RANKER:-lexical}"
ATTENTION_ROUTER_LAYERS="${ATTENTION_ROUTER_LAYERS:-4}"
ATTENTION_ROUTER_ATTN_IMPL="${ATTENTION_ROUTER_ATTN_IMPL:-eager}"
ATTENTION_ROUTER_MAX_QUERY_TOKENS="${ATTENTION_ROUTER_MAX_QUERY_TOKENS:-512}"
ATTENTION_ROUTER_SCORE_MODE="${ATTENTION_ROUTER_SCORE_MODE:-top4_mean}"
ATTENTION_ROUTER_SPAN_TOP_TOKENS="${ATTENTION_ROUTER_SPAN_TOP_TOKENS:-4}"

if [[ -d "${DATASET_PATH}" && -f "${DATASET_PATH}/data/${DATASET}.jsonl" ]]; then
  DATASET_PATH="${DATASET_PATH}/data/${DATASET}.jsonl"
elif [[ -d "${DATASET_PATH}" && -f "${DATASET_PATH}/${DATASET}.jsonl" ]]; then
  DATASET_PATH="${DATASET_PATH}/${DATASET}.jsonl"
fi

mkdir -p "${OUTPUT_DIR}"

echo "ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES}"
echo "MODEL_PATH=${MODEL_PATH}"
echo "BASE_MODEL=${BASE_MODEL}"
echo "TOKENIZER_PATH=${TOKENIZER_PATH}"
echo "DATASET=${DATASET}"
echo "DATASET_PATH=${DATASET_PATH}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "PYTHON_BIN=${PYTHON_BIN}"
echo "MAX_EXAMPLES=${MAX_EXAMPLES}"
echo "RATIO=${RATIO}"
echo "HYBRID_TOP_K=${HYBRID_TOP_K}"
echo "MAX_DOC_LENGTH=${MAX_DOC_LENGTH}"
echo "MAX_DOC_NUM=${MAX_DOC_NUM}"
echo "MAX_CONTEXT_TOKENS=${MAX_CONTEXT_TOKENS}"
echo "DEVICE_TYPE=${DEVICE_TYPE}"
echo "DEVICES=${DEVICES_CSV}"
echo "RANK_PLANS=${RANK_PLANS}"
echo "TARGET_COMPRESSION_RATIO=${TARGET_COMPRESSION_RATIO}"
echo "RECOVERY_CANDIDATE_DOCS=${RECOVERY_CANDIDATE_DOCS}"
echo "RECOVERY_SPAN_TOKENS=${RECOVERY_SPAN_TOKENS}"
echo "RECOVERY_MAX_SPANS=${RECOVERY_MAX_SPANS}"
echo "HYBRID_CHUNK_TOP_K=${HYBRID_CHUNK_TOP_K}"
echo "HYBRID_CHUNK_TOKENS=${HYBRID_CHUNK_TOKENS}"
echo "HYBRID_CHUNK_OVERLAP=${HYBRID_CHUNK_OVERLAP}"
echo "HYBRID_CHUNK_RANKER=${HYBRID_CHUNK_RANKER}"
echo "ATTENTION_ROUTER_LAYERS=${ATTENTION_ROUTER_LAYERS}"
echo "ATTENTION_ROUTER_ATTN_IMPL=${ATTENTION_ROUTER_ATTN_IMPL}"
echo "ATTENTION_ROUTER_SCORE_MODE=${ATTENTION_ROUTER_SCORE_MODE}"

IFS=',' read -ra VISIBLE_DEVICES <<< "${DEVICES_CSV}"
IFS=',' read -ra RUN_MODES <<< "${MODES}"
IFS=';' read -ra RUN_RANK_PLANS <<< "${RANK_PLANS}"

if [[ "${#VISIBLE_DEVICES[@]}" -eq 0 ]]; then
  echo "DEVICES is empty" >&2
  exit 1
fi

PIDS=()
INDEX=0
RANK_PLAN_INDEX=0
for MODE in "${RUN_MODES[@]}"; do
  MODE="${MODE// /}"
  RANK_PLAN=""
  MODE_NAME="${MODE}"
  CHUNK_TOP_K="${HYBRID_CHUNK_TOP_K}"
  CHUNK_RANKER="${HYBRID_CHUNK_RANKER}"
  if [[ "${MODE}" == rank_plan:* ]]; then
    MODE_NAME="${MODE#rank_plan:}"
    if (( RANK_PLAN_INDEX < ${#RUN_RANK_PLANS[@]} )); then
      RANK_PLAN="${RUN_RANK_PLANS[${RANK_PLAN_INDEX}]}"
    fi
    RANK_PLAN_INDEX=$((RANK_PLAN_INDEX + 1))
    MODE="rank_plan"
  fi
  if [[ "${MODE}" == chunk_hybrid:* ]]; then
    MODE_NAME="${MODE#chunk_hybrid:}"
    MODE="chunk_hybrid"
    if [[ "${MODE_NAME}" == *attention_fullkv* ]] || [[ "${MODE_NAME}" == *att_fullkv* ]]; then
      CHUNK_RANKER="attention_fullkv"
    elif [[ "${MODE_NAME}" == *attention_c2kv* ]] || [[ "${MODE_NAME}" == *att_c2kv* ]] || [[ "${MODE_NAME}" == *compressedkv* ]]; then
      CHUNK_RANKER="attention_c2kv"
    elif [[ "${MODE_NAME}" == *bm25* ]]; then
      CHUNK_RANKER="bm25"
    elif [[ "${MODE_NAME}" == *lex* ]] || [[ "${MODE_NAME}" == *chunk* ]]; then
      CHUNK_RANKER="lexical"
    fi
    if [[ "${MODE_NAME}" =~ top([0-9]+) ]]; then
      CHUNK_TOP_K="${BASH_REMATCH[1]}"
    fi
  fi
  DEVICE="${VISIBLE_DEVICES[$((INDEX % ${#VISIBLE_DEVICES[@]}))]}"
  OUTPUT_FILE="${OUTPUT_DIR}/${DATASET}_${MODE_NAME}_r${RATIO}.jsonl"
  LOG_FILE="${OUTPUT_DIR}/${DATASET}_${MODE_NAME}_r${RATIO}.log"
  echo "[launch] mode=${MODE} name=${MODE_NAME} rank_plan=${RANK_PLAN} device=${DEVICE} output=${OUTPUT_FILE}"
  (
    if [[ "${DEVICE_TYPE}" == "cuda" ]]; then
      export CUDA_VISIBLE_DEVICES="${DEVICE}"
      unset ASCEND_RT_VISIBLE_DEVICES || true
      unset ASCEND_VISIBLE_DEVICES || true
    else
      export ASCEND_RT_VISIBLE_DEVICES="${DEVICE}"
      export ASCEND_VISIBLE_DEVICES="${DEVICE}"
    fi
    "${PYTHON_BIN}" python/inference/expr_mdoc_c2kv_baselines.py \
      --model "${MODEL_PATH}" \
      --base_model "${BASE_MODEL}" \
      --tokenizer "${TOKENIZER_PATH}" \
      --dataset "${DATASET}" \
      --dataset_path "${DATASET_PATH}" \
      --mode "${MODE}" \
      --override_ratio "${RATIO}" \
      --hybrid_top_k "${HYBRID_TOP_K}" \
      --rank_plan "${RANK_PLAN:-1:full,2-:c2kv${RATIO}}" \
      --target_compression_ratio "${TARGET_COMPRESSION_RATIO}" \
      --recovery_candidate_docs "${RECOVERY_CANDIDATE_DOCS}" \
      --recovery_span_tokens "${RECOVERY_SPAN_TOKENS}" \
      --recovery_max_spans "${RECOVERY_MAX_SPANS}" \
      --max_examples "${MAX_EXAMPLES}" \
      --output_file "${OUTPUT_FILE}" \
      --max_doc_length "${MAX_DOC_LENGTH}" \
      --max_doc_num "${MAX_DOC_NUM}" \
      --max_context_tokens "${MAX_CONTEXT_TOKENS}" \
      --max_query_tokens "${MAX_QUERY_TOKENS}" \
      --doc_selection "${DOC_SELECTION}" \
      --hybrid_chunk_top_k "${CHUNK_TOP_K}" \
      --hybrid_chunk_tokens "${HYBRID_CHUNK_TOKENS}" \
      --hybrid_chunk_overlap "${HYBRID_CHUNK_OVERLAP}" \
      --hybrid_chunk_ranker "${CHUNK_RANKER}" \
      --attention_router_layers "${ATTENTION_ROUTER_LAYERS}" \
      --attention_router_attn_impl "${ATTENTION_ROUTER_ATTN_IMPL}" \
      --attention_router_max_query_tokens "${ATTENTION_ROUTER_MAX_QUERY_TOKENS}" \
      --attention_router_score_mode "${ATTENTION_ROUTER_SCORE_MODE}" \
      --attention_router_span_top_tokens "${ATTENTION_ROUTER_SPAN_TOP_TOKENS}" \
      --device_type "${DEVICE_TYPE}" \
      --system_attn_impl "${SYSTEM_ATTN_IMPL}" \
      --gist_attn_impl "${GIST_ATTN_IMPL}" \
      --generate_attn_impl "${GENERATE_ATTN_IMPL}" \
      --dtype "${DTYPE}"
  ) > "${LOG_FILE}" 2>&1 &
  PIDS+=("$!")
  INDEX=$((INDEX + 1))
done

FAILED=0
for PID in "${PIDS[@]}"; do
  if ! wait "${PID}"; then
    FAILED=1
  fi
done

echo "==== summaries ===="
for MODE in "${RUN_MODES[@]}"; do
  MODE="${MODE// /}"
  MODE_NAME="${MODE}"
  if [[ "${MODE}" == rank_plan:* ]]; then
    MODE_NAME="${MODE#rank_plan:}"
  elif [[ "${MODE}" == chunk_hybrid:* ]]; then
    MODE_NAME="${MODE#chunk_hybrid:}"
  fi
  SUMMARY_FILE="${OUTPUT_DIR}/${DATASET}_${MODE_NAME}_r${RATIO}.summary.json"
  echo "---- ${SUMMARY_FILE} ----"
  if [[ -f "${SUMMARY_FILE}" ]]; then
    cat "${SUMMARY_FILE}"
  else
    echo "missing summary; check ${OUTPUT_DIR}/${DATASET}_${MODE}_r${RATIO}.log"
  fi
done

exit "${FAILED}"
