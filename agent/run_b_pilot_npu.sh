#!/usr/bin/env bash
# Experiment B Stage-1 chunking pilot on Ascend NPU (eval-only, no training).
# Runs the four policy arms of configs/bdf_pilot/b_prereg.md against ONE shared
# full/truncate reference on the frozen eval-200 qid manifest, merges per arm,
# then calls agent/analyze_b_pilot.py.  Mirrors
# agent/eval_joint_next_action_c2kv_npu.sh conventions (Ascend env,
# HF_HUB_OFFLINE=1, one shard per NPU, condition-aware merge).
#
# Env knobs (all optional):
#   NAME                   run name / output subdir  (b_pilot)
#   MODEL_PATH             joint C2KV checkpoint     (./checkpoints/qwen3-4b-joint-c2kv-npu)
#   BASE_MODEL             base model dir            (./models/Qwen3-4B-Instruct-2507)
#   TOKENIZER_PATH         tokenizer dir             (BASE_MODEL)
#   DATASET_PATH           traces parquet dir        (./datasets/agent-llm-traces)
#   SPLIT_MANIFEST_FILE    session split manifest    ("")
#   SPLIT_NAME             split manifest name       (subset_disjoint)
#   SPLIT                  train | eval              (eval)
#   QID_MANIFEST           frozen eval-200 qid list  (configs/bdf_pilot/b_eval200_qids.json)
#   OUTPUT_DIR             results dir               (./outputs/b_pilot)
#   ARMS                   comma list of arm names   (P-fixed,P-turn,P-struct,P-delay)
#   RATIOS                 compression ratios        (8)
#   RUN_REF                True -> run the shared full/truncate reference (True)
#   MAX_EXAMPLES           cap before the manifest filter (0 = all)  (0)
#   PARALLEL_EVAL          True -> shard arms across visible NPUs     (True)
#   TMP_DIR                per-case shards + logs    (OUTPUT_DIR/parts)
#   SPLIT_SEED             session split seed        (42)
#   EVAL_RATIO             eval fraction of sessions (0.1)
#   MAX_SAMPLES_PER_SESSION  examples per session    (4)
#   MAX_TOOL_DEFINITION_TOKENS  tool-side skip cap   (32000)
#   MAX_DOC_LENGTH / MAX_DOC_NUM / MAX_TOOL_CHUNKS / MIN_DOC_NUM
#   MAX_SYSTEM_LENGTH / MAX_PROMPT_TOKENS / MAX_BASELINE_INPUT_TOKENS / MAX_NEW_TOKENS
#   HISTORY_SELECTION / SPLIT_OVERSIZED_HISTORY_DOCS / REQUIRE_TOOL_CALL
#   DO_SAMPLE              True -> sampled decode    (False, i.e. greedy)
#   TEMPERATURE            sampled-pass T            (0.7; only sent when DO_SAMPLE)
#   TOP_P                  sampled-pass top_p        (0.95; only sent when DO_SAMPLE)
#   GEN_SEED               per-row seed base         (0)
#   KV_BYTES_PER_TOKEN     analyzer byte unit        (147456)
#   NPU_ATTN_IMPL          attention impl            (eager)
#
# Arm -> flag mapping (the ONLY place it is defined; echoed below):
#   P-fixed   --chunk_policy fixed-1024                          (P3, gist reference)
#   P-turn    --chunk_policy agent-turn                          (P5, incumbent / in-dist)
#   P-struct  --chunk_policy structural                          (P6)
#   P-delay   --chunk_policy agent-turn --delay_recent_turns 1   (P5+L)
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

NAME="${NAME:-b_pilot}"
MODEL_PATH="${MODEL_PATH:-./checkpoints/qwen3-4b-joint-c2kv-npu}"
BASE_MODEL="${BASE_MODEL:-./models/Qwen3-4B-Instruct-2507}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${BASE_MODEL}}"
DATASET_PATH="${DATASET_PATH:-./datasets/agent-llm-traces}"
SPLIT_MANIFEST_FILE="${SPLIT_MANIFEST_FILE:-}"
SPLIT_NAME="${SPLIT_NAME:-subset_disjoint}"
SPLIT="${SPLIT:-eval}"
QID_MANIFEST="${QID_MANIFEST:-configs/bdf_pilot/b_eval200_qids.json}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/${NAME}}"
ARMS="${ARMS:-P-fixed,P-turn,P-struct,P-delay}"
RATIOS="${RATIOS:-8}"
RUN_REF="${RUN_REF:-True}"
MAX_EXAMPLES="${MAX_EXAMPLES:-0}"
PARALLEL_EVAL="${PARALLEL_EVAL:-True}"

SPLIT_SEED="${SPLIT_SEED:-42}"
EVAL_RATIO="${EVAL_RATIO:-0.1}"
MAX_SAMPLES_PER_SESSION="${MAX_SAMPLES_PER_SESSION:-4}"
REQUIRE_TOOL_CALL="${REQUIRE_TOOL_CALL:-True}"
MAX_DOC_LENGTH="${MAX_DOC_LENGTH:-1024}"
MAX_DOC_NUM="${MAX_DOC_NUM:-24}"
MAX_TOOL_CHUNKS="${MAX_TOOL_CHUNKS:-16}"
MIN_DOC_NUM="${MIN_DOC_NUM:-2}"
MAX_TOOL_DEFINITION_TOKENS="${MAX_TOOL_DEFINITION_TOKENS:-32000}"
MAX_SYSTEM_LENGTH="${MAX_SYSTEM_LENGTH:-512}"
MAX_PROMPT_TOKENS="${MAX_PROMPT_TOKENS:-1920}"
MAX_BASELINE_INPUT_TOKENS="${MAX_BASELINE_INPUT_TOKENS:-16000}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
HISTORY_SELECTION="${HISTORY_SELECTION:-tail}"
SPLIT_OVERSIZED_HISTORY_DOCS="${SPLIT_OVERSIZED_HISTORY_DOCS:-True}"
DO_SAMPLE="${DO_SAMPLE:-False}"
# Explicit defaults (same values as agent/run_f_pilot_npu.sh): a sampled run
# must never inherit the checkpoint's generation_config, or the run summary
# records temperature=null and the decode configuration is untraceable.
# eval_joint_next_action_c2kv.py rejects --do_sample true without --temperature.
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.95}"
GEN_SEED="${GEN_SEED:-0}"
KV_BYTES_PER_TOKEN="${KV_BYTES_PER_TOKEN:-147456}"
NPU_ATTN_IMPL="${NPU_ATTN_IMPL:-eager}"

TMP_DIR="${TMP_DIR:-${OUTPUT_DIR}/parts}"

SPLIT_ARGS=(--split_manifest_name "${SPLIT_NAME}")
if [[ -n "${SPLIT_MANIFEST_FILE}" ]]; then
  SPLIT_ARGS=(--split_manifest_file "${SPLIT_MANIFEST_FILE}" --split_manifest_name "${SPLIT_NAME}")
fi

OPTIONAL_ARGS=()
if [[ -n "${QID_MANIFEST}" ]]; then
  if [[ ! -f "${QID_MANIFEST}" ]]; then
    echo "QID_MANIFEST=${QID_MANIFEST} not found — the frozen eval-200 set is mandatory " \
         "(24号 1-7/1-8); set QID_MANIFEST='' only for a smoke run" >&2
    exit 1
  fi
  OPTIONAL_ARGS+=(--qid_manifest "${QID_MANIFEST}")
fi
if [[ -n "${MAX_TOOL_CHUNKS}" ]]; then
  OPTIONAL_ARGS+=(--max_tool_chunks "${MAX_TOOL_CHUNKS}")
fi
# Value semantics, not mere presence.  T/top_p ride along ONLY on the sampled
# branch: passing them under greedy decode would stamp a temperature into the
# run summary that no token was ever drawn with.
case "${DO_SAMPLE}" in
  1|true|True|yes)
    if [[ -z "${TEMPERATURE}" || -z "${TOP_P}" ]]; then
      echo "DO_SAMPLE=${DO_SAMPLE} needs non-empty TEMPERATURE and TOP_P (defaults 0.7/0.95); " \
           "an unset temperature would silently fall back to the checkpoint generation_config" >&2
      exit 1
    fi
    OPTIONAL_ARGS+=(--do_sample true --temperature "${TEMPERATURE}" --top_p "${TOP_P}")
    ;;
  ""|0|false|False|no) OPTIONAL_ARGS+=(--do_sample false) ;;
  *) echo "Unrecognized DO_SAMPLE=${DO_SAMPLE} (use true/false)" >&2; exit 1 ;;
esac

COMMON_ARGS=(
  --device_type npu
  --base_model "${BASE_MODEL}"
  --tokenizer "${TOKENIZER_PATH}"
  --dataset_path "${DATASET_PATH}"
  --split "${SPLIT}"
  "${SPLIT_ARGS[@]}"
  --conditions joint
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
  --gen_seed "${GEN_SEED}"
  --system_attn_impl "${NPU_ATTN_IMPL}"
  --gist_attn_impl "${NPU_ATTN_IMPL}"
  --generate_attn_impl "${NPU_ATTN_IMPL}"
  "${OPTIONAL_ARGS[@]}"
)

arm_flags() {
  case "$1" in
    P-fixed)  echo "--chunk_policy fixed-1024" ;;
    P-turn)   echo "--chunk_policy agent-turn" ;;
    P-struct) echo "--chunk_policy structural" ;;
    P-delay)  echo "--chunk_policy agent-turn --delay_recent_turns 1" ;;
    *) echo "UNKNOWN" ;;
  esac
}

echo "ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES}"
echo "NAME=${NAME}"
echo "MODEL_PATH=${MODEL_PATH}"
echo "BASE_MODEL=${BASE_MODEL}"
echo "TOKENIZER_PATH=${TOKENIZER_PATH}"
echo "DATASET_PATH=${DATASET_PATH}"
echo "SPLIT=${SPLIT}"
echo "SPLIT_MANIFEST_FILE=${SPLIT_MANIFEST_FILE}"
echo "QID_MANIFEST=${QID_MANIFEST}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "TMP_DIR=${TMP_DIR}"
echo "SPLIT_SEED=${SPLIT_SEED} EVAL_RATIO=${EVAL_RATIO} MAX_SAMPLES_PER_SESSION=${MAX_SAMPLES_PER_SESSION}"
echo "MAX_TOOL_DEFINITION_TOKENS=${MAX_TOOL_DEFINITION_TOKENS}"
echo "ARMS=${ARMS}"
echo "RATIOS=${RATIOS}"
echo "RUN_REF=${RUN_REF}"
echo "MAX_EXAMPLES=${MAX_EXAMPLES}"
echo "MAX_DOC_LENGTH=${MAX_DOC_LENGTH}"
echo "MAX_DOC_NUM=${MAX_DOC_NUM}"
echo "MAX_TOOL_CHUNKS=${MAX_TOOL_CHUNKS}"
echo "HISTORY_SELECTION=${HISTORY_SELECTION}"
echo "DO_SAMPLE=${DO_SAMPLE} TEMPERATURE=${TEMPERATURE} TOP_P=${TOP_P} GEN_SEED=${GEN_SEED}"
echo "PARALLEL_EVAL=${PARALLEL_EVAL}"
echo "--- arm -> flag map ---"
IFS=',' read -ra _arms <<< "${ARMS}"
for arm in "${_arms[@]}"; do
  arm="${arm// /}"
  flags="$(arm_flags "${arm}")"
  if [[ "${flags}" == "UNKNOWN" ]]; then
    echo "Unrecognized arm '${arm}' (known: P-fixed, P-turn, P-struct, P-delay)" >&2
    exit 1
  fi
  echo "  ${arm}: ${flags}"
done
echo "-----------------------"

mkdir -p "${OUTPUT_DIR}" "${TMP_DIR}"
IFS=',' read -ra _visible_npus <<< "${ASCEND_RT_VISIBLE_DEVICES}"
IFS=',' read -ra _ratios <<< "${RATIOS}"
CASE_INDEX=0
BATCH_SIZE="${#_visible_npus[@]}"

run_case() {
  local case_output="$1"; shift
  # Per-case flags go LAST: argparse lets the later occurrence win, so this
  # ordering guarantees the arm's --chunk_policy / --delay_recent_turns /
  # --compare_modes / --ratios can never be shadowed by a same-named knob in
  # COMMON_ARGS.  (No such collision exists today — the two sets are disjoint —
  # but a silently collapsed arm table is not a failure mode worth risking.)
  python agent/eval_joint_next_action_c2kv.py \
    --output_file "${case_output}" \
    --model "${MODEL_PATH}" \
    "${COMMON_ARGS[@]}" \
    "$@"
}

launch_case() {
  local case_name="$1"; shift
  local case_output="${TMP_DIR}/${case_name}.jsonl"
  local case_log="${TMP_DIR}/${case_name}.log"
  rm -f "${case_output}" "${case_output%.jsonl}.summary.json" "${case_log}"
  if [[ "${PARALLEL_EVAL}" != "True" && "${PARALLEL_EVAL}" != "true" && "${PARALLEL_EVAL}" != "1" ]]; then
    echo "[run] case=${case_name} output=${case_output}"
    run_case "${case_output}" "$@" 2>&1 | tee "${case_log}"
    return 0
  fi
  local device="${_visible_npus[$((CASE_INDEX % BATCH_SIZE))]}"
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

# --- 1. one shared full/truncate reference (incumbent chunking) -------------
REF_SHARDS=()
case "${RUN_REF}" in
  1|true|True|yes)
    for ratio in "${_ratios[@]}"; do
      ratio="${ratio// /}"
      launch_case "reference_r${ratio}" \
        --compare_modes full,truncate \
        --chunk_policy agent-turn \
        --ratios "${ratio}"
      REF_SHARDS+=("${TMP_DIR}/reference_r${ratio}.jsonl")
    done
    ;;
  ""|0|false|False|no) ;;
  *) echo "Unrecognized RUN_REF=${RUN_REF} (use true/false)" >&2; exit 1 ;;
esac

# --- 2. one c2kv run per arm per ratio --------------------------------------
for arm in "${_arms[@]}"; do
  arm="${arm// /}"
  read -ra _arm_flags <<< "$(arm_flags "${arm}")"
  for ratio in "${_ratios[@]}"; do
    ratio="${ratio// /}"
    launch_case "${arm}_r${ratio}" \
      --compare_modes c2kv \
      "${_arm_flags[@]}" \
      --ratios "${ratio}"
  done
done

wait

# --- 3. per-arm merge -------------------------------------------------------
merge_shards() {
  local out_file="$1"; shift
  python agent/eval_joint_next_action_c2kv.py \
    --merge_only \
    --output_file "${out_file}" \
    --model "${MODEL_PATH}" \
    --base_model "${BASE_MODEL}" \
    --dataset_path "${DATASET_PATH}" \
    --split "${SPLIT}" \
    --input_files "$@"
}

ANALYZE_ARGS=()
for arm in "${_arms[@]}"; do
  arm="${arm// /}"
  ARM_SHARDS=()
  for ratio in "${_ratios[@]}"; do
    ratio="${ratio// /}"
    ARM_SHARDS+=("${TMP_DIR}/${arm}_r${ratio}.jsonl")
  done
  merge_shards "${OUTPUT_DIR}/${arm}.jsonl" "${ARM_SHARDS[@]}"
  ANALYZE_ARGS+=(--arm "${arm}=${OUTPUT_DIR}/${arm}.jsonl")
done

if (( ${#REF_SHARDS[@]} > 0 )); then
  merge_shards "${OUTPUT_DIR}/reference.jsonl" "${REF_SHARDS[@]}"
  ANALYZE_ARGS+=(--full "${OUTPUT_DIR}/reference.jsonl")
fi

# --- 4. paired analysis -----------------------------------------------------
# The analyzer re-checks the frozen qid set itself: any arm not covering the
# manifest stamps an INCOMPLETE COMMON-QID SET banner on analysis.md
# (b_prereg.md §2 — such a round enters no paired table).
if [[ -n "${QID_MANIFEST}" ]]; then
  ANALYZE_ARGS+=(--qid_manifest "${QID_MANIFEST}")
fi
python agent/analyze_b_pilot.py \
  "${ANALYZE_ARGS[@]}" \
  --out_prefix "${OUTPUT_DIR}/${NAME}" \
  --reference_arm P-fixed \
  --kv_bytes_per_token "${KV_BYTES_PER_TOKEN}"

# --- 5. gist declaration verdict (判据1) ------------------------------------
# A VOID arm is a WARNING here, never a fatal exit: the artefacts must still be
# retrievable from the NPU box even when an arm has to be re-run under per-row
# budget allocation.
echo "=== gist declaration (判据1: >5% off P-fixed = VOID) ==="
python - "${OUTPUT_DIR}/${NAME}.analysis.json" <<'PY' || echo "gist verdict unavailable (analysis json unreadable)"
import json, sys
report = json.load(open(sys.argv[1], encoding="utf-8"))
table = report["gist_declaration"]
for entry in table["arms"]:
    print(f"  {entry['arm']:<10} mean_gist={entry['mean_gist_tokens']:>9.2f} "
          f"raw_recent={entry['mean_raw_recent_tokens']:>9.2f} "
          f"dev={entry['deviation_vs_reference'] * 100:+7.2f}%  {entry['verdict']}")
if table["any_void"]:
    print("WARNING: at least one arm is VOID under 判据1 — its numbers do not enter any "
          "ranking until per-row gist budget allocation is implemented and it is re-run.")
print(report["footnote"])
PY

echo "=== results ==="
echo "rows:     ${OUTPUT_DIR}/<arm>.jsonl"
echo "analysis: ${OUTPUT_DIR}/${NAME}.analysis.json"
echo "          ${OUTPUT_DIR}/${NAME}.analysis.md"
echo "Per-arm summaries:"
for arm in "${_arms[@]}"; do
  arm="${arm// /}"
  summary="${OUTPUT_DIR}/${arm}.summary.json"
  echo "==== ${summary} ===="
  if [[ -f "${summary}" ]]; then
    cat "${summary}"
  else
    echo "MISSING summary file. Check logs under ${TMP_DIR}"
  fi
done
