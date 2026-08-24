#!/usr/bin/env bash
# True-joint C2KV training on Ascend NPU: tool schemas AND history turns are
# compressed into the gist-KV context grid (python/train/train_data_joint.py);
# the system prefix carries neither (the de-leak vs. the tooldef/history paths).
#
# Env knobs (all optional):
#   MODEL_PATH             HF model dir            (./models/Qwen3-4B-Instruct-2507)
#   DATASET_PATH           traces parquet dir      (./datasets/agent-llm-traces)
#   OUTPUT_DIR             checkpoint dir          (./checkpoints/qwen3-4b-joint-c2kv-npu)
#   DOC_MODE               joint | tool_only | history_only | alternate (joint)
#   LR / NUM_TRAIN_EPOCHS / WARMUP_STEPS / PER_DEVICE_BS / GRAD_ACCUM
#   SAVE_STEPS / EVAL_STEPS / LOGGING_STEPS
#   USE_DEEPSPEED          1 -> torchrun + configs/ds_config_npu.json (1);
#                          0 -> single-process single-card python, chip chosen
#                          via ASCEND_RT_VISIBLE_DEVICES passthrough
#   SPLIT_MANIFEST_FILE / SPLIT_NAME / EXAMPLE_ORDER_FILE / MAX_SOURCE_TOKENS
#   MAX_TRAIN_EXAMPLES / MAX_EVAL_EXAMPLES / MAX_TOOL_CHUNKS (empty = omit flag)
#
# Example (single card acquired by scripts/joint_gated_run.sh):
#   USE_DEEPSPEED=0 bash agent/train_joint_next_action_c2kv_npu.sh
set -euo pipefail

# Ascend toolkit env (NPU server): provides torch_npu runtime libs (libhccl...).
if [[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]]; then
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi

export PYTHONPATH="$(pwd)/python:$(pwd)/agent:${PYTHONPATH:-}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-1800}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-max_split_size_mb:128}"
# Gist compression ratio(s) the extractor is trained at. EXPORTED on purpose:
# agent/train_unified_next_action_c2kv_npu.sh forgot to export this variable.
export C2KV_GIST_TRAIN_RATIOS="${C2KV_GIST_TRAIN_RATIOS:-8}"

MODEL_PATH="${MODEL_PATH:-./models/Qwen3-4B-Instruct-2507}"
DATASET_PATH="${DATASET_PATH:-./datasets/agent-llm-traces}"
OUTPUT_DIR="${OUTPUT_DIR:-./checkpoints/qwen3-4b-joint-c2kv-npu}"
NPU_ATTN_IMPL="${NPU_ATTN_IMPL:-npu_fusion_attention}"
USE_DEEPSPEED="${USE_DEEPSPEED:-1}"

SPLIT_MANIFEST_FILE="${SPLIT_MANIFEST_FILE:-}"
SPLIT_NAME="${SPLIT_NAME:-subset_disjoint}"
SPLIT_SEED="${SPLIT_SEED:-42}"
EVAL_RATIO="${EVAL_RATIO:-0.1}"
MAX_SAMPLES_PER_SESSION="${MAX_SAMPLES_PER_SESSION:-4}"
EXAMPLE_ORDER_FILE="${EXAMPLE_ORDER_FILE:-}"
MAX_SOURCE_TOKENS="${MAX_SOURCE_TOKENS:-}"
MAX_TRAIN_EXAMPLES="${MAX_TRAIN_EXAMPLES:-}"
MAX_EVAL_EXAMPLES="${MAX_EVAL_EXAMPLES:-}"

DOC_MODE="${DOC_MODE:-joint}"
MAX_DOC_LENGTH="${MAX_DOC_LENGTH:-1024}"
MAX_DOC_NUM="${MAX_DOC_NUM:-24}"
MAX_TOOL_CHUNKS="${MAX_TOOL_CHUNKS:-}"
LEGACY_MODE_CAPS="${LEGACY_MODE_CAPS:-}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
MAX_SYSTEM_LENGTH="${MAX_SYSTEM_LENGTH:-512}"
MAX_TOOL_DEFINITION_TOKENS="${MAX_TOOL_DEFINITION_TOKENS:-32000}"
MIN_TARGET_TOKENS="${MIN_TARGET_TOKENS:-32}"
REQUIRE_TOOL_CALL="${REQUIRE_TOOL_CALL:-True}"
HISTORY_SELECTION="${HISTORY_SELECTION:-tail}"

LR="${LR:-5e-7}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
WARMUP_STEPS="${WARMUP_STEPS:-20}"
PER_DEVICE_BS="${PER_DEVICE_BS:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
EVAL_STEPS="${EVAL_STEPS:-100}"
SAVE_STEPS="${SAVE_STEPS:-500}"
LOGGING_STEPS="${LOGGING_STEPS:-1}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
DATALOADER_PREFETCH_FACTOR="${DATALOADER_PREFETCH_FACTOR:-4}"
DATASET_SHUFFLE_SEED="${DATASET_SHUFFLE_SEED:-2948}"

ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"
export ASCEND_RT_VISIBLE_DEVICES

if [[ -z "${C2KV_GIST_CHECKPOINT_USE_REENTRANT+x}" ]]; then
  if [[ "${USE_DEEPSPEED}" == "1" ]]; then
    export C2KV_GIST_CHECKPOINT_USE_REENTRANT=True
  else
    export C2KV_GIST_CHECKPOINT_USE_REENTRANT=False
  fi
fi

if [[ -z "${NPROC_PER_NODE:-}" ]]; then
  IFS=',' read -ra _visible_npus <<< "${ASCEND_RT_VISIBLE_DEVICES}"
  NPROC_PER_NODE="${#_visible_npus[@]}"
fi

if ! find "${DATASET_PATH}" -name '*.parquet' -type f -print -quit 2>/dev/null | grep -q .; then
  echo "ERROR: no parquet files found under DATASET_PATH=${DATASET_PATH}" >&2
  echo "Expected files like: ${DATASET_PATH}/data/train-00000-of-00039.parquet" >&2
  exit 1
fi

if [[ -n "${SPLIT_MANIFEST_FILE}" && -f "${SPLIT_MANIFEST_FILE}" ]]; then
  python - "${SPLIT_MANIFEST_FILE}" "${SPLIT_NAME}" <<'PY'
import json
import sys

path, split_name = sys.argv[1], sys.argv[2]
with open(path, "r", encoding="utf-8") as f:
    manifest = json.load(f)
if "train_session_ids" in manifest and "eval_session_ids" in manifest:
    sys.exit(0)
if split_name not in manifest:
    available = sorted(key for key in manifest if key != "metadata")
    raise SystemExit(
        f"ERROR: split {split_name!r} not found in {path}. "
        f"Available splits: {available}"
    )
PY
fi

OPTIONAL_ARGS=()
if [[ -n "${SPLIT_MANIFEST_FILE}" ]]; then
  OPTIONAL_ARGS+=(--split_manifest_file "${SPLIT_MANIFEST_FILE}")
fi
if [[ -n "${EXAMPLE_ORDER_FILE}" ]]; then
  OPTIONAL_ARGS+=(--example_order_file "${EXAMPLE_ORDER_FILE}")
fi
if [[ -n "${MAX_SOURCE_TOKENS}" ]]; then
  OPTIONAL_ARGS+=(--max_source_tokens "${MAX_SOURCE_TOKENS}")
fi
if [[ -n "${MAX_TRAIN_EXAMPLES}" ]]; then
  OPTIONAL_ARGS+=(--max_train_examples "${MAX_TRAIN_EXAMPLES}")
fi
if [[ -n "${MAX_EVAL_EXAMPLES}" ]]; then
  OPTIONAL_ARGS+=(--max_eval_examples "${MAX_EVAL_EXAMPLES}")
fi
if [[ -n "${MAX_TOOL_CHUNKS}" ]]; then
  OPTIONAL_ARGS+=(--max_tool_chunks "${MAX_TOOL_CHUNKS}")
fi
# Pre-fix doc budgets, for diffing against the pre-fix small arms only.
# Value semantics (not mere presence): false/0/no keep the fixed budgets.
case "${LEGACY_MODE_CAPS}" in
  1|true|True|yes) OPTIONAL_ARGS+=(--legacy_mode_caps true) ;;
  ""|0|false|False|no) ;;
  *) echo "Unrecognized LEGACY_MODE_CAPS=${LEGACY_MODE_CAPS} (use true/false)" >&2; exit 1 ;;
esac
if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
  OPTIONAL_ARGS+=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")
fi

DATALOADER_ARGS=(--dataloader_num_workers "${DATALOADER_NUM_WORKERS}")
if (( DATALOADER_NUM_WORKERS > 0 )); then
  DATALOADER_ARGS+=(--dataloader_prefetch_factor "${DATALOADER_PREFETCH_FACTOR}")
fi

if [[ "${USE_DEEPSPEED}" == "1" ]]; then
  LAUNCHER=(torchrun --nproc_per_node "${NPROC_PER_NODE}")
  DEEPSPEED_ARGS=(--deepspeed ./configs/ds_config_npu.json)
else
  # Single-process single-card: the chip is chosen by the caller through
  # ASCEND_RT_VISIBLE_DEVICES (e.g. scripts/joint_gated_run.sh).
  LAUNCHER=(python)
  DEEPSPEED_ARGS=()
fi

echo "ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES}"
echo "NPROC_PER_NODE=${NPROC_PER_NODE}"
echo "USE_DEEPSPEED=${USE_DEEPSPEED}"
echo "MODEL_PATH=${MODEL_PATH}"
echo "DATASET_PATH=${DATASET_PATH}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "SPLIT_MANIFEST_FILE=${SPLIT_MANIFEST_FILE}"
echo "SPLIT_NAME=${SPLIT_NAME}"
echo "EXAMPLE_ORDER_FILE=${EXAMPLE_ORDER_FILE}"
echo "MAX_SOURCE_TOKENS=${MAX_SOURCE_TOKENS}"
echo "MAX_TRAIN_EXAMPLES=${MAX_TRAIN_EXAMPLES}"
echo "MAX_EVAL_EXAMPLES=${MAX_EVAL_EXAMPLES}"
echo "DOC_MODE=${DOC_MODE}"
echo "MAX_DOC_LENGTH=${MAX_DOC_LENGTH}"
echo "MAX_DOC_NUM=${MAX_DOC_NUM}"
echo "MAX_TOOL_CHUNKS=${MAX_TOOL_CHUNKS}"
echo "MAX_LENGTH=${MAX_LENGTH}"
echo "MAX_SYSTEM_LENGTH=${MAX_SYSTEM_LENGTH}"
echo "MAX_TOOL_DEFINITION_TOKENS=${MAX_TOOL_DEFINITION_TOKENS}"
echo "MIN_TARGET_TOKENS=${MIN_TARGET_TOKENS}"
echo "LR=${LR}"
echo "NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS}"
echo "WARMUP_STEPS=${WARMUP_STEPS}"
echo "PER_DEVICE_BS=${PER_DEVICE_BS}"
echo "GRAD_ACCUM=${GRAD_ACCUM}"
echo "C2KV_GIST_TRAIN_RATIOS=${C2KV_GIST_TRAIN_RATIOS}"
echo "C2KV_GIST_CHECKPOINT_USE_REENTRANT=${C2KV_GIST_CHECKPOINT_USE_REENTRANT}"
echo "SAVE_STEPS=${SAVE_STEPS}"
echo "LOGGING_STEPS=${LOGGING_STEPS}"

"${LAUNCHER[@]}" \
  agent/train_joint_next_action_c2kv.py \
  --device_type npu \
  --npu_attn_impl "${NPU_ATTN_IMPL}" \
  --attn_impl "${NPU_ATTN_IMPL}" \
  --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
  --warmup_steps "${WARMUP_STEPS}" \
  --model_name_or_path "${MODEL_PATH}" \
  --padding_side right \
  --per_device_train_batch_size "${PER_DEVICE_BS}" \
  --per_device_eval_batch_size "${PER_DEVICE_BS}" \
  --gradient_accumulation_steps "${GRAD_ACCUM}" \
  --lr_scheduler_type cosine \
  --learning_rate "${LR}" \
  --weight_decay 0.1 \
  --enable_gist True \
  --gist_param qkv \
  --gist_type dynamic-interleave \
  --gist_overlap 64 \
  --gist_residual_type embed-mean \
  --gist_gradient_checkpointing True \
  --only_train_gist True \
  --dataset_path "${DATASET_PATH}" \
  --split_seed "${SPLIT_SEED}" \
  --eval_ratio "${EVAL_RATIO}" \
  --split_manifest_name "${SPLIT_NAME}" \
  --max_samples_per_session "${MAX_SAMPLES_PER_SESSION}" \
  --doc_mode "${DOC_MODE}" \
  --max_doc_length "${MAX_DOC_LENGTH}" \
  --max_doc_num "${MAX_DOC_NUM}" \
  --max_length "${MAX_LENGTH}" \
  --max_system_length "${MAX_SYSTEM_LENGTH}" \
  --max_tool_definition_tokens "${MAX_TOOL_DEFINITION_TOKENS}" \
  --min_target_tokens "${MIN_TARGET_TOKENS}" \
  --require_tool_call "${REQUIRE_TOOL_CALL}" \
  --history_selection "${HISTORY_SELECTION}" \
  "${OPTIONAL_ARGS[@]}" \
  --output_dir "${OUTPUT_DIR}" \
  --logging_steps "${LOGGING_STEPS}" \
  --logging_nan_inf_filter False \
  --remove_unused_columns False \
  "${DEEPSPEED_ARGS[@]}" \
  --do_train True \
  --eval_strategy steps \
  --eval_steps "${EVAL_STEPS}" \
  --save_strategy steps \
  --save_steps "${SAVE_STEPS}" \
  "${DATALOADER_ARGS[@]}" \
  --bf16 True \
  --dataset_shuffle_seed "${DATASET_SHUFFLE_SEED}"
