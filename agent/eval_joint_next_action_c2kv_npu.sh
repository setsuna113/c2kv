#!/usr/bin/env bash
# True-joint C2KV next-action eval on Ascend NPU
# (agent/eval_joint_next_action_c2kv.py).  Mirrors
# agent/eval_agent_history_c2kv_npu.sh / eval_unified_next_action_c2kv_npu.sh
# conventions: Ascend env, HF_HUB_OFFLINE=1, per-(condition, mode, ratio)
# shards fanned out across ASCEND_RT_VISIBLE_DEVICES, then a condition-aware
# merge (--merge_only).  Checkpoint-latest resolution (checkpoint-*) happens
# in the python entry (_resolve_model_checkpoint).
#
# Env knobs (all optional):
#   MODEL_PATH             joint C2KV checkpoint   (./checkpoints/qwen3-4b-joint-c2kv-npu)
#   BASE_MODEL             base model dir          (./models/Qwen3-4B-Instruct-2507)
#   TOKENIZER_PATH         tokenizer dir           (BASE_MODEL)
#   DATASET_PATH           traces parquet dir      (./datasets/agent-llm-traces)
#   OUTPUT_FILE            merged jsonl            (./outputs/joint_next_action_eval_npu.jsonl)
#   SPLIT                  train | eval            (eval)
#   CONDITIONS             joint,tool_only,history_only (joint)
#   COMPARE_MODES          c2kv,c2kv_untrained,truncate,full (c2kv,full)
#   RATIOS                 compression ratios      (8)
#   SEPARATE               True -> J-separate arm (needs CHECKPOINT_TOOL /
#                          CHECKPOINT_HISTORY; ignores CONDITIONS/COMPARE_MODES)
#   SEPARATE_GENERATOR     tool | history          (tool)
#   PARALLEL_EVAL          True -> shard across visible NPUs (True)
set -euo pipefail

# Ascend toolkit env (NPU server): provides torch_npu runtime libs (libhccl...).
if [[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]]; then
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi

export PYTHONPATH="$(pwd)/python:$(pwd)/python/inference:$(pwd)/agent:${PYTHONPATH:-}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-max_split_size_mb:128}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,3,4,5,6,7}"

MODEL_PATH="${MODEL_PATH:-./checkpoints/qwen3-4b-joint-c2kv-npu}"
BASE_MODEL="${BASE_MODEL:-./models/Qwen3-4B-Instruct-2507}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${BASE_MODEL}}"
DATASET_PATH="${DATASET_PATH:-./datasets/agent-llm-traces}"
OUTPUT_FILE="${OUTPUT_FILE:-./outputs/joint_next_action_eval_npu.jsonl}"
SPLIT="${SPLIT:-eval}"

CONDITIONS="${CONDITIONS:-joint}"
COMPARE_MODES="${COMPARE_MODES:-c2kv,full}"
RATIOS="${RATIOS:-8}"
SEPARATE="${SEPARATE:-False}"
CHECKPOINT_TOOL="${CHECKPOINT_TOOL:-}"
CHECKPOINT_HISTORY="${CHECKPOINT_HISTORY:-}"
SEPARATE_GENERATOR="${SEPARATE_GENERATOR:-tool}"
MAX_EXAMPLES="${MAX_EXAMPLES:-100}"
MAX_SOURCE_EXAMPLES="${MAX_SOURCE_EXAMPLES:-}"

SPLIT_MANIFEST_FILE="${SPLIT_MANIFEST_FILE:-}"
SPLIT_NAME="${SPLIT_NAME:-subset_disjoint}"
SPLIT_SEED="${SPLIT_SEED:-42}"
EVAL_RATIO="${EVAL_RATIO:-0.1}"
MAX_SAMPLES_PER_SESSION="${MAX_SAMPLES_PER_SESSION:-4}"
REQUIRE_TOOL_CALL="${REQUIRE_TOOL_CALL:-True}"

MAX_DOC_LENGTH="${MAX_DOC_LENGTH:-1024}"
MAX_DOC_NUM="${MAX_DOC_NUM:-24}"
MAX_TOOL_CHUNKS="${MAX_TOOL_CHUNKS:-}"
MIN_DOC_NUM="${MIN_DOC_NUM:-2}"
MAX_TOOL_DEFINITION_TOKENS="${MAX_TOOL_DEFINITION_TOKENS:-32000}"
MAX_SYSTEM_LENGTH="${MAX_SYSTEM_LENGTH:-512}"
MAX_PROMPT_TOKENS="${MAX_PROMPT_TOKENS:-1920}"
MAX_BASELINE_INPUT_TOKENS="${MAX_BASELINE_INPUT_TOKENS:-16000}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
HISTORY_SELECTION="${HISTORY_SELECTION:-tail}"
SPLIT_OVERSIZED_HISTORY_DOCS="${SPLIT_OVERSIZED_HISTORY_DOCS:-True}"
MAX_INPUT_CHARS="${MAX_INPUT_CHARS:-}"
MAX_ANSWER_CHARS="${MAX_ANSWER_CHARS:-}"
PREFIX_HISTORY_DOC_NUM="${PREFIX_HISTORY_DOC_NUM:-}"
PREFIX_HISTORY_EXACT="${PREFIX_HISTORY_EXACT:-False}"
NPU_ATTN_IMPL="${NPU_ATTN_IMPL:-eager}"
PARALLEL_EVAL="${PARALLEL_EVAL:-True}"
OUTPUT_STEM="${OUTPUT_FILE%.jsonl}"
TMP_DIR="${TMP_DIR:-${OUTPUT_STEM}.parts}"

SPLIT_ARGS=(--split_manifest_name "${SPLIT_NAME}")
if [[ -n "${SPLIT_MANIFEST_FILE}" ]]; then
  SPLIT_ARGS=(--split_manifest_file "${SPLIT_MANIFEST_FILE}" --split_manifest_name "${SPLIT_NAME}")
fi

OPTIONAL_ARGS=()
if [[ -n "${MAX_SOURCE_EXAMPLES}" ]]; then
  OPTIONAL_ARGS+=(--max_source_examples "${MAX_SOURCE_EXAMPLES}")
fi
if [[ -n "${MAX_TOOL_CHUNKS}" ]]; then
  OPTIONAL_ARGS+=(--max_tool_chunks "${MAX_TOOL_CHUNKS}")
fi
# Pre-fix doc budgets, for diffing against the pre-fix small arms only.
# Value semantics (not mere presence): false/0/no keep the fixed budgets.
case "${LEGACY_MODE_CAPS:-}" in
  1|true|True|yes) OPTIONAL_ARGS+=(--legacy_mode_caps) ;;
  ""|0|false|False|no) ;;
  *) echo "Unrecognized LEGACY_MODE_CAPS=${LEGACY_MODE_CAPS} (use true/false)" >&2; exit 1 ;;
esac
if [[ -n "${MAX_INPUT_CHARS}" ]]; then
  OPTIONAL_ARGS+=(--max_input_chars "${MAX_INPUT_CHARS}")
fi
if [[ -n "${MAX_ANSWER_CHARS}" ]]; then
  OPTIONAL_ARGS+=(--max_answer_chars "${MAX_ANSWER_CHARS}")
fi
if [[ -n "${PREFIX_HISTORY_DOC_NUM}" ]]; then
  OPTIONAL_ARGS+=(--prefix_history_doc_num "${PREFIX_HISTORY_DOC_NUM}")
fi

COMMON_ARGS=(
  --device_type npu
  --base_model "${BASE_MODEL}"
  --tokenizer "${TOKENIZER_PATH}"
  --dataset_path "${DATASET_PATH}"
  --split "${SPLIT}"
  "${SPLIT_ARGS[@]}"
  --max_examples "${MAX_EXAMPLES}"
  --eval_ratio "${EVAL_RATIO}"
  --split_seed "${SPLIT_SEED}"
  --max_samples_per_session "${MAX_SAMPLES_PER_SESSION}"
  --require_tool_call "${REQUIRE_TOOL_CALL}"
  --max_doc_length "${MAX_DOC_LENGTH}"
  --max_doc_num "${MAX_DOC_NUM}"
  --min_doc_num "${MIN_DOC_NUM}"
  --max_tool_definition_tokens "${MAX_TOOL_DEFINITION_TOKENS}"
  --max_system_length "${MAX_SYSTEM_LENGTH}"
  --max_prompt_tokens "${MAX_PROMPT_TOKENS}"
  --max_baseline_input_tokens "${MAX_BASELINE_INPUT_TOKENS}"
  --max_new_tokens "${MAX_NEW_TOKENS}"
  --history_selection "${HISTORY_SELECTION}"
  --split_oversized_history_docs "${SPLIT_OVERSIZED_HISTORY_DOCS}"
  --prefix_history_exact "${PREFIX_HISTORY_EXACT}"
  --system_attn_impl "${NPU_ATTN_IMPL}"
  --gist_attn_impl "${NPU_ATTN_IMPL}"
  --generate_attn_impl "${NPU_ATTN_IMPL}"
  "${OPTIONAL_ARGS[@]}"
)

echo "ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES}"
echo "MODEL_PATH=${MODEL_PATH}"
echo "BASE_MODEL=${BASE_MODEL}"
echo "TOKENIZER_PATH=${TOKENIZER_PATH}"
echo "DATASET_PATH=${DATASET_PATH}"
echo "OUTPUT_FILE=${OUTPUT_FILE}"
echo "SPLIT=${SPLIT}"
echo "CONDITIONS=${CONDITIONS}"
echo "COMPARE_MODES=${COMPARE_MODES}"
echo "RATIOS=${RATIOS}"
echo "SEPARATE=${SEPARATE}"
echo "CHECKPOINT_TOOL=${CHECKPOINT_TOOL}"
echo "CHECKPOINT_HISTORY=${CHECKPOINT_HISTORY}"
echo "SEPARATE_GENERATOR=${SEPARATE_GENERATOR}"
echo "MAX_EXAMPLES=${MAX_EXAMPLES}"
echo "MAX_DOC_LENGTH=${MAX_DOC_LENGTH}"
echo "MAX_DOC_NUM=${MAX_DOC_NUM}"
echo "MAX_SYSTEM_LENGTH=${MAX_SYSTEM_LENGTH}"
echo "MAX_PROMPT_TOKENS=${MAX_PROMPT_TOKENS}"
echo "MAX_BASELINE_INPUT_TOKENS=${MAX_BASELINE_INPUT_TOKENS}"
echo "HISTORY_SELECTION=${HISTORY_SELECTION}"
echo "PARALLEL_EVAL=${PARALLEL_EVAL}"

run_case() {
  local case_output="$1"; shift
  python agent/eval_joint_next_action_c2kv.py \
    --output_file "${case_output}" \
    "$@" \
    "${COMMON_ARGS[@]}"
}

if [[ "${PARALLEL_EVAL}" != "True" && "${PARALLEL_EVAL}" != "true" && "${PARALLEL_EVAL}" != "1" ]]; then
  if [[ "${SEPARATE}" == "True" || "${SEPARATE}" == "true" || "${SEPARATE}" == "1" ]]; then
    run_case "${OUTPUT_FILE}" \
      --separate \
      --checkpoint_tool "${CHECKPOINT_TOOL}" \
      --checkpoint_history "${CHECKPOINT_HISTORY}" \
      --separate_generator "${SEPARATE_GENERATOR}" \
      --ratios "${RATIOS}"
  else
    run_case "${OUTPUT_FILE}" \
      --model "${MODEL_PATH}" \
      --conditions "${CONDITIONS}" \
      --compare_modes "${COMPARE_MODES}" \
      --ratios "${RATIOS}"
  fi
  exit 0
fi

mkdir -p "${TMP_DIR}"
IFS=',' read -ra _visible_npus <<< "${ASCEND_RT_VISIBLE_DEVICES}"
IFS=',' read -ra _ratios <<< "${RATIOS}"

CASE_OUTPUTS=()
SUMMARY_FILES=()
CASE_INDEX=0
BATCH_SIZE="${#_visible_npus[@]}"

launch_case() {
  local case_name="$1"; shift
  local device="${_visible_npus[$((CASE_INDEX % BATCH_SIZE))]}"
  local case_output="${TMP_DIR}/${case_name}.jsonl"
  local case_summary="${TMP_DIR}/${case_name}.summary.json"
  local case_log="${TMP_DIR}/${case_name}.log"
  rm -f "${case_output}" "${case_summary}" "${case_log}"
  CASE_OUTPUTS+=("${case_output}")
  SUMMARY_FILES+=("${case_summary}")
  echo "[launch] case=${case_name} device=${device} output=${case_output}"
  (
    export ASCEND_RT_VISIBLE_DEVICES="${device}"
    run_case "${case_output}" "$@"
  ) > "${case_log}" 2>&1 &

  CASE_INDEX=$((CASE_INDEX + 1))
  if (( CASE_INDEX % BATCH_SIZE == 0 )); then
    wait
  fi
}

if [[ "${SEPARATE}" == "True" || "${SEPARATE}" == "true" || "${SEPARATE}" == "1" ]]; then
  for ratio in "${_ratios[@]}"; do
    ratio="${ratio// /}"
    launch_case "separate_${SEPARATE_GENERATOR}_r${ratio}" \
      --separate \
      --checkpoint_tool "${CHECKPOINT_TOOL}" \
      --checkpoint_history "${CHECKPOINT_HISTORY}" \
      --separate_generator "${SEPARATE_GENERATOR}" \
      --ratios "${ratio}"
  done
else
  IFS=',' read -ra _conditions <<< "${CONDITIONS}"
  IFS=',' read -ra _modes <<< "${COMPARE_MODES}"
  for condition in "${_conditions[@]}"; do
    condition="${condition// /}"
    for mode in "${_modes[@]}"; do
      mode="${mode// /}"
      case_ratios=("${_ratios[@]}")
      if [[ "${mode}" == "full" ]]; then
        case_ratios=("1")
      fi
      for ratio in "${case_ratios[@]}"; do
        ratio="${ratio// /}"
        launch_case "${condition}_${mode}_r${ratio}" \
          --model "${MODEL_PATH}" \
          --conditions "${condition}" \
          --compare_modes "${mode}" \
          --ratios "${ratio}"
      done
    done
  done
fi

wait

MERGE_ARGS=(--model "${MODEL_PATH}")
if [[ "${SEPARATE}" == "True" || "${SEPARATE}" == "true" || "${SEPARATE}" == "1" ]]; then
  MERGE_ARGS=(--separate --checkpoint_tool "${CHECKPOINT_TOOL}" --checkpoint_history "${CHECKPOINT_HISTORY}" --separate_generator "${SEPARATE_GENERATOR}")
fi

python agent/eval_joint_next_action_c2kv.py \
  --merge_only \
  --output_file "${OUTPUT_FILE}" \
  --base_model "${BASE_MODEL}" \
  --dataset_path "${DATASET_PATH}" \
  --split "${SPLIT}" \
  "${MERGE_ARGS[@]}" \
  --input_files "${CASE_OUTPUTS[@]}"

echo "Shard summaries:"
for summary in "${SUMMARY_FILES[@]}"; do
  echo "==== ${summary} ===="
  if [[ -f "${summary}" ]]; then
    cat "${summary}"
  else
    summary_log="${summary%.summary.json}.log"
    echo "MISSING summary file. Check log: ${summary_log}"
  fi
done
