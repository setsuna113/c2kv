#!/usr/bin/env bash
# F pilot launcher (agent/f_timing_fork.py + agent/analyze_f_fork.py) on Ascend NPU.
# Mirrors agent/eval_joint_next_action_c2kv_npu.sh conventions: Ascend env,
# HF_HUB_OFFLINE=1, budget knobs identical to the joint eval so both fork
# branches stay in the joint training distribution.
#
# First version is deliberately SINGLE-CARD and SERIAL: <=200 examples x 2-3
# generations is not worth sharding, and one append-only jsonl keeps the
# resume key (qid, arm_pass, branch, rollout_index) trivially correct.  Rerun
# the same command after an interruption; only NON-skipped rollouts count as
# done, so eligibility-skips and OOM rows are retried.
#
# Env knobs (all optional).  Paths / model:
#   MODEL_PATH             joint C2KV checkpoint   (./checkpoints/qwen3-4b-joint-c2kv-npu)
#   BASE_MODEL             base model dir          (./models/Qwen3-4B-Instruct-2507)
#   TOKENIZER_PATH         tokenizer dir           (BASE_MODEL)
#   DATASET_PATH           traces parquet dir      (./datasets/agent-llm-traces)
#   OUTPUT_FILE            per-rollout jsonl       (./outputs/f_pilot/f_timing_fork_npu.jsonl)
#   PREREG_FILE            frozen prereg, sha256 stamped into every row
#                          (./configs/bdf_pilot/f_prereg.md)
#
# Example selection (must match the joint eval to keep both branches in the
# joint training distribution):
#   SPLIT                  train | eval            (eval)
#   SPLIT_MANIFEST_FILE    frozen split manifest   (empty -> ratio/seed split)
#   SPLIT_NAME             manifest name           (subset_disjoint)
#   SPLIT_SEED             ratio-split seed, ignored with a manifest (42)
#   EVAL_RATIO             ratio-split eval share, ignored with a manifest (0.1)
#   MAX_SAMPLES_PER_SESSION  rows kept per session (4)
#   REQUIRE_TOOL_CALL      True|False, E4          (True)
#   QID_MANIFEST           frozen qid list json    (empty -> loader order)
#   MAX_EXAMPLES           cap on examples, <=0 = all; ignored for the load
#                          when QID_MANIFEST is set (200)
#
# F arms:
#   ARM_SET                greedy_core|sampled|both (greedy_core)
#   TEMPERATURE            sampled-pass T          (0.7)
#   TOP_P                  sampled-pass top_p      (0.95)
#   GEN_SEED               per-rollout seed base + F4 coin seed (0)
#   L_MIN                  E3 lower bound on last-chunk tokens (64)
#   ASSERT_GREEDY_REPEAT   rerun branch-A greedy for the first N examples and
#                          require byte-identical text (2)
#
# Budget knobs (grid geometry -- changing any of these changes what the fork
# segment IS, so a run with non-default values is NOT comparable to one with
# defaults):
#   RATIO                  compression ratio, --override_ratio (8)
#   MAX_DOC_LENGTH         tokens per chunk; also the E3 upper bound (1024)
#   MAX_DOC_NUM            max chunks in the grid  (24)
#   MAX_TOOL_CHUNKS        cap on tool-side chunks (empty -> derived from
#                          MAX_DOC_NUM by the builder)
#   MIN_DOC_NUM            minimum chunks or the example is skipped (2)
#   MAX_TOOL_DEFINITION_TOKENS  E1 builder skip threshold (32000)
#   MAX_SYSTEM_LENGTH      system prompt token cap (512)
#   MAX_PROMPT_TOKENS      current-turn token cap  (1920)
#   MAX_NEW_TOKENS         decode budget per rollout (128)
#   HISTORY_SELECTION      head | tail             (tail)
#   SPLIT_OVERSIZED_HISTORY_DOCS  True|False       (True)
#
# Runtime:
#   DEVICE_TYPE            auto|cuda|npu|cpu       (npu)
#   NPU_ATTN_IMPL          attn impl for system / gist / generate (eager)
#   ASCEND_RT_VISIBLE_DEVICES  single card, this launcher is serial (0)
#   RESUME                 True|False              (True)
#   SKIP_ANALYZE           True -> do not run analyze_f_fork.py at the end (False)
#   HF_HUB_OFFLINE (1) / TOKENIZERS_PARALLELISM (false) /
#   PYTORCH_NPU_ALLOC_CONF (max_split_size_mb:128) are the house boilerplate
#   exports; they are honoured if already set in the environment.
#
# ARM_SET=sampled|both requires the do_sample/temperature/top_p parameters on
# _generate_from_input_ids; the driver probes for them and exits with a clear
# message if they are missing.  greedy_core never passes those keywords.
set -euo pipefail

# This launcher is driven ENTIRELY by the env knobs above and forwards no CLI
# arguments to the driver, so anything typed after the script name would be
# dropped on the floor.  Silently ignoring a flag someone typed is the same
# failure mode as silently overriding it -- refuse instead.
if (( $# > 0 )); then
  echo "run_f_pilot_npu.sh takes no positional arguments (got: $*)." >&2
  echo "Use the env knobs listed at the top of this file, e.g." >&2
  echo "  MAX_EXAMPLES=20 ARM_SET=greedy_core bash agent/run_f_pilot_npu.sh" >&2
  echo "For a one-off flag the knobs do not cover, call the driver directly:" >&2
  echo "  python agent/f_timing_fork.py --model ... --output_file ..." >&2
  exit 1
fi

# Ascend toolkit env (NPU server): provides torch_npu runtime libs (libhccl...).
if [[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]]; then
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi

export PYTHONPATH="$(pwd)/python:$(pwd)/python/inference:$(pwd)/agent:${PYTHONPATH:-}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-max_split_size_mb:128}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"

MODEL_PATH="${MODEL_PATH:-./checkpoints/qwen3-4b-joint-c2kv-npu}"
BASE_MODEL="${BASE_MODEL:-./models/Qwen3-4B-Instruct-2507}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${BASE_MODEL}}"
DATASET_PATH="${DATASET_PATH:-./datasets/agent-llm-traces}"
SPLIT="${SPLIT:-eval}"
SPLIT_MANIFEST_FILE="${SPLIT_MANIFEST_FILE:-}"
SPLIT_NAME="${SPLIT_NAME:-subset_disjoint}"
SPLIT_SEED="${SPLIT_SEED:-42}"
EVAL_RATIO="${EVAL_RATIO:-0.1}"
MAX_SAMPLES_PER_SESSION="${MAX_SAMPLES_PER_SESSION:-4}"
REQUIRE_TOOL_CALL="${REQUIRE_TOOL_CALL:-True}"
QID_MANIFEST="${QID_MANIFEST:-}"
MAX_EXAMPLES="${MAX_EXAMPLES:-200}"

ARM_SET="${ARM_SET:-greedy_core}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.95}"
GEN_SEED="${GEN_SEED:-0}"
L_MIN="${L_MIN:-64}"
RATIO="${RATIO:-8}"
ASSERT_GREEDY_REPEAT="${ASSERT_GREEDY_REPEAT:-2}"

OUTPUT_FILE="${OUTPUT_FILE:-./outputs/f_pilot/f_timing_fork_npu.jsonl}"
PREREG_FILE="${PREREG_FILE:-./configs/bdf_pilot/f_prereg.md}"
DEVICE_TYPE="${DEVICE_TYPE:-npu}"

MAX_DOC_LENGTH="${MAX_DOC_LENGTH:-1024}"
MAX_DOC_NUM="${MAX_DOC_NUM:-24}"
MAX_TOOL_CHUNKS="${MAX_TOOL_CHUNKS:-}"
MIN_DOC_NUM="${MIN_DOC_NUM:-2}"
MAX_TOOL_DEFINITION_TOKENS="${MAX_TOOL_DEFINITION_TOKENS:-32000}"
MAX_SYSTEM_LENGTH="${MAX_SYSTEM_LENGTH:-512}"
MAX_PROMPT_TOKENS="${MAX_PROMPT_TOKENS:-1920}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
HISTORY_SELECTION="${HISTORY_SELECTION:-tail}"
SPLIT_OVERSIZED_HISTORY_DOCS="${SPLIT_OVERSIZED_HISTORY_DOCS:-True}"
NPU_ATTN_IMPL="${NPU_ATTN_IMPL:-eager}"

# Value semantics (not mere presence) for the boolean knobs.
case "${RESUME:-True}" in
  1|true|True|yes) RESUME_VALUE="True" ;;
  0|false|False|no) RESUME_VALUE="False" ;;
  *) echo "Unrecognized RESUME=${RESUME} (use true/false)" >&2; exit 1 ;;
esac
case "${SKIP_ANALYZE:-False}" in
  1|true|True|yes) SKIP_ANALYZE_VALUE="True" ;;
  ""|0|false|False|no) SKIP_ANALYZE_VALUE="False" ;;
  *) echo "Unrecognized SKIP_ANALYZE=${SKIP_ANALYZE} (use true/false)" >&2; exit 1 ;;
esac

SPLIT_ARGS=(--split_manifest_name "${SPLIT_NAME}")
if [[ -n "${SPLIT_MANIFEST_FILE}" ]]; then
  SPLIT_ARGS=(--split_manifest_file "${SPLIT_MANIFEST_FILE}" --split_manifest_name "${SPLIT_NAME}")
fi

OPTIONAL_ARGS=()
if [[ -n "${QID_MANIFEST}" ]]; then
  OPTIONAL_ARGS+=(--qid_manifest "${QID_MANIFEST}")
fi
if [[ -n "${MAX_TOOL_CHUNKS}" ]]; then
  OPTIONAL_ARGS+=(--max_tool_chunks "${MAX_TOOL_CHUNKS}")
fi

COMMON_ARGS=(
  --model "${MODEL_PATH}"
  --base_model "${BASE_MODEL}"
  --tokenizer "${TOKENIZER_PATH}"
  --dataset_path "${DATASET_PATH}"
  --output_file "${OUTPUT_FILE}"
  --prereg_file "${PREREG_FILE}"
  --split "${SPLIT}"
  "${SPLIT_ARGS[@]}"
  --arm_set "${ARM_SET}"
  --temperature "${TEMPERATURE}"
  --top_p "${TOP_P}"
  --gen_seed "${GEN_SEED}"
  --l_min "${L_MIN}"
  --override_ratio "${RATIO}"
  --assert_greedy_repeat "${ASSERT_GREEDY_REPEAT}"
  --resume "${RESUME_VALUE}"
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
  --max_new_tokens "${MAX_NEW_TOKENS}"
  --history_selection "${HISTORY_SELECTION}"
  --split_oversized_history_docs "${SPLIT_OVERSIZED_HISTORY_DOCS}"
  --device_type "${DEVICE_TYPE}"
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
echo "SPLIT=${SPLIT}"
echo "SPLIT_MANIFEST_FILE=${SPLIT_MANIFEST_FILE}"
echo "SPLIT_NAME=${SPLIT_NAME}"
echo "SPLIT_SEED=${SPLIT_SEED}"
echo "EVAL_RATIO=${EVAL_RATIO}"
echo "MAX_SAMPLES_PER_SESSION=${MAX_SAMPLES_PER_SESSION}"
echo "REQUIRE_TOOL_CALL=${REQUIRE_TOOL_CALL}"
echo "QID_MANIFEST=${QID_MANIFEST}"
echo "MAX_EXAMPLES=${MAX_EXAMPLES}"
echo "ARM_SET=${ARM_SET}"
echo "TEMPERATURE=${TEMPERATURE}"
echo "TOP_P=${TOP_P}"
echo "GEN_SEED=${GEN_SEED}"
echo "L_MIN=${L_MIN}"
echo "RATIO=${RATIO}"
echo "MAX_DOC_LENGTH=${MAX_DOC_LENGTH}"
echo "MAX_DOC_NUM=${MAX_DOC_NUM}"
echo "MAX_TOOL_CHUNKS=${MAX_TOOL_CHUNKS}"
echo "MIN_DOC_NUM=${MIN_DOC_NUM}"
echo "MAX_TOOL_DEFINITION_TOKENS=${MAX_TOOL_DEFINITION_TOKENS}"
echo "MAX_SYSTEM_LENGTH=${MAX_SYSTEM_LENGTH}"
echo "MAX_PROMPT_TOKENS=${MAX_PROMPT_TOKENS}"
echo "MAX_NEW_TOKENS=${MAX_NEW_TOKENS}"
echo "HISTORY_SELECTION=${HISTORY_SELECTION}"
echo "SPLIT_OVERSIZED_HISTORY_DOCS=${SPLIT_OVERSIZED_HISTORY_DOCS}"
echo "NPU_ATTN_IMPL=${NPU_ATTN_IMPL}"
echo "OUTPUT_FILE=${OUTPUT_FILE}"
echo "PREREG_FILE=${PREREG_FILE}"
echo "DEVICE_TYPE=${DEVICE_TYPE}"
echo "RESUME=${RESUME_VALUE}"
echo "ASSERT_GREEDY_REPEAT=${ASSERT_GREEDY_REPEAT}"
echo "SKIP_ANALYZE=${SKIP_ANALYZE_VALUE}"

mkdir -p "$(dirname "${OUTPUT_FILE}")"

python agent/f_timing_fork.py "${COMMON_ARGS[@]}"

if [[ "${SKIP_ANALYZE_VALUE}" == "True" ]]; then
  echo "SKIP_ANALYZE=True -- run agent/analyze_f_fork.py --input_file ${OUTPUT_FILE} yourself."
  exit 0
fi

python agent/analyze_f_fork.py \
  --input_file "${OUTPUT_FILE}" \
  --coin_seed "${GEN_SEED}"

OUTPUT_STEM="${OUTPUT_FILE%.jsonl}"
echo "Run summary:"
if [[ -f "${OUTPUT_STEM}.run.json" ]]; then
  cat "${OUTPUT_STEM}.run.json"
else
  echo "MISSING ${OUTPUT_STEM}.run.json"
fi
echo "Analysis:"
if [[ -f "${OUTPUT_STEM}.analysis.md" ]]; then
  cat "${OUTPUT_STEM}.analysis.md"
else
  echo "MISSING ${OUTPUT_STEM}.analysis.md"
fi
