#!/usr/bin/env bash
# True-joint C2KV training on 2x H200 (141GB, CUDA): tool schemas AND history
# turns are compressed into the gist-KV context grid
# (python/train/train_data_joint.py); the system prefix carries neither (the
# de-leak vs. the tooldef/history paths). Ported from
# agent/train_joint_next_action_c2kv_npu.sh (Ascend 8-card) — device flags and
# launcher env only; data/planner semantics unchanged.
#
# Env knobs (all optional unless marked REQUIRED):
#   MODEL_PATH             HF model dir (./models/Qwen3-4B-Instruct-2507).
#                          2026-08-25 裁定：init gate 取消（G8-small-v2 拿不到），
#                          主臂用 base dir 新鲜 gist init 直接跑。
#   DATASET_PATH           traces parquet dir      (./datasets/agent-llm-traces)
#   TOUCAN_PATH            toucan source dir       (empty = disabled)
#   EXAMPLE_ORDER_FILE     planner output file     (REQUIRED for this arm)
#   MAX_SOURCE_TOKENS      ESTIMATED-source-token budget (empty = unbounded).
#                          Convert from the presented-token target (64M-96M)
#                          via the mixture-specific rho measured after
#                          planning: if rho≈0.392, 64M presented ≈ 163M
#                          estimated, 96M ≈ 245M.
#   OUTPUT_DIR             checkpoint dir (./checkpoints/qwen3-4b-joint-c2kv-h200)
#   DOC_MODE               joint | tool_only | history_only | alternate (joint)
#   LR / NUM_TRAIN_EPOCHS / PER_DEVICE_BS / GRAD_ACCUM
#                          defaults 5e-5 / 1 / 1 / 4: 2 GPUs x 1 x 4 = eff
#                          batch 8; if microbatch 2 fits, set PER_DEVICE_BS=2
#                          GRAD_ACCUM=2 to keep eff-8.
#   WARMUP_RATIO           default 0.04 (passed with --warmup_steps 0).
#                          Escape hatch: set WARMUP_STEPS explicitly to pass a
#                          step count instead (warmup_ratio then omitted).
#   SAVE_STEPS             default 2000 PLACEHOLDER. The trainer only supports
#                          step-granularity saves: after a 100-200-step
#                          calibration run, compute presented-tokens/step and
#                          set SAVE_STEPS so a checkpoint lands every ≈16M
#                          presented source tokens.
#   EVAL_STEPS / LOGGING_STEPS / RESUME_FROM_CHECKPOINT / SEED
#   ATTN_IMPL              default flex_attention (H200 sm_90 228KB smem 放得下;
#                          若 inductor kernel 超 smem 上限, 如 sm_89 的 4090,
#                          设 ATTN_IMPL=eager 兜底). 另: system 前缀 pass 的
#                          attn impl 由 C2KV_SYSTEM_ATTN_IMPL 控制(默认 sdpa,
#                          因为 flash-attn 预编译 wheel 要 glibc>=2.32)
#   REQUIRE_TOOL_CALL      default False; ACTION_TOOL_CALL_FRAC default 0.75
#   USE_DEEPSPEED          1 -> torchrun + configs/ds_config_h200.json (1);
#                          0 -> plain torchrun DDP fallback
#   CUDA_VISIBLE_DEVICES   default 0,1; NPROC_PER_NODE derives from its entry
#                          count (single-card short gate runs:
#                          CUDA_VISIBLE_DEVICES=0 -> nproc=1)
#   SPLIT_MANIFEST_FILE / SPLIT_NAME / MAX_TRAIN_EXAMPLES / MAX_EVAL_EXAMPLES /
#   MAX_TOOL_CHUNKS / MULTISOURCE_MAX_RECORDS
#
# Intentionally NOT wired vs the NPU script: OPENSWE_PATH / QA_HOTPOTQA_PATH /
# QA_2WIKI_PATH / QA_LONGMAGPIE_PATH — this arm's mixture is toucan+traces only.
#
# Example (single-card short gate run):
#   CUDA_VISIBLE_DEVICES=0 EXAMPLE_ORDER_FILE=<planner.jsonl> \
#     bash agent/train_joint_next_action_c2kv_h200.sh
set -euo pipefail

export PYTHONPATH="$(pwd)/python:$(pwd)/agent:${PYTHONPATH:-}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
# Gist compression ratio(s) the extractor is trained at. EXPORTED on purpose:
# agent/train_unified_next_action_c2kv_npu.sh forgot to export this variable.
# Fixed 8x for this arm.
export C2KV_GIST_TRAIN_RATIOS="${C2KV_GIST_TRAIN_RATIOS:-8}"

MODEL_PATH="${MODEL_PATH:-./models/Qwen3-4B-Instruct-2507}"
DATASET_PATH="${DATASET_PATH:-./datasets/agent-llm-traces}"
OUTPUT_DIR="${OUTPUT_DIR:-./checkpoints/qwen3-4b-joint-c2kv-h200}"
ATTN_IMPL="${ATTN_IMPL:-flex_attention}"
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

TOUCAN_PATH="${TOUCAN_PATH:-}"
MULTISOURCE_MAX_RECORDS="${MULTISOURCE_MAX_RECORDS:-}"

DOC_MODE="${DOC_MODE:-joint}"
MAX_DOC_LENGTH="${MAX_DOC_LENGTH:-1024}"
MAX_DOC_NUM="${MAX_DOC_NUM:-24}"
MAX_TOOL_CHUNKS="${MAX_TOOL_CHUNKS:-}"
LEGACY_MODE_CAPS="${LEGACY_MODE_CAPS:-}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
MAX_SYSTEM_LENGTH="${MAX_SYSTEM_LENGTH:-512}"
MAX_TOOL_DEFINITION_TOKENS="${MAX_TOOL_DEFINITION_TOKENS:-32000}"
MIN_TARGET_TOKENS="${MIN_TARGET_TOKENS:-32}"
REQUIRE_TOOL_CALL="${REQUIRE_TOOL_CALL:-False}"
ACTION_TOOL_CALL_FRAC="${ACTION_TOOL_CALL_FRAC:-0.75}"
HISTORY_SELECTION="${HISTORY_SELECTION:-tail}"

LR="${LR:-5e-5}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
WARMUP_RATIO="${WARMUP_RATIO:-0.04}"
WARMUP_STEPS="${WARMUP_STEPS:-}"
PER_DEVICE_BS="${PER_DEVICE_BS:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
EVAL_STEPS="${EVAL_STEPS:-100}"
SAVE_STEPS="${SAVE_STEPS:-2000}"
LOGGING_STEPS="${LOGGING_STEPS:-1}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
DATALOADER_PREFETCH_FACTOR="${DATALOADER_PREFETCH_FACTOR:-4}"
DATASET_SHUFFLE_SEED="${DATASET_SHUFFLE_SEED:-2948}"
# HF TrainingArguments seed (sampler/训练随机流): default 42 matches every arm
# to date; set SEED=43 for the G8-small-v2 seed-2 replication (nothing else
# in the launch env may change for that arm).
SEED="${SEED:-42}"
# ddp_timeout(秒): HF 默认 1800 小于大池静默建样本窗口(实测 ~22-25min)与
# start_h200.sh 的 STALL_MIN(35min)看门狗; 对齐 NPU 各脚本(如
# scripts/train_qwen3-4b-mixed_mdoc_npu_high_ratio.sh)给 7200, env 可覆盖。
DDP_TIMEOUT="${DDP_TIMEOUT:-7200}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export CUDA_VISIBLE_DEVICES

if [[ -z "${C2KV_GIST_CHECKPOINT_USE_REENTRANT+x}" ]]; then
  if [[ "${USE_DEEPSPEED}" == "1" ]]; then
    export C2KV_GIST_CHECKPOINT_USE_REENTRANT=True
  else
    export C2KV_GIST_CHECKPOINT_USE_REENTRANT=False
  fi
fi

if [[ -z "${NPROC_PER_NODE:-}" ]]; then
  IFS=',' read -ra _visible_gpus <<< "${CUDA_VISIBLE_DEVICES}"
  NPROC_PER_NODE="${#_visible_gpus[@]}"
fi

if ! find "${DATASET_PATH}" -name '*.parquet' -type f -print -quit 2>/dev/null | grep -q .; then
  echo "ERROR: no parquet files found under DATASET_PATH=${DATASET_PATH}" >&2
  echo "Expected files like: ${DATASET_PATH}/data/train-00000-of-00039.parquet" >&2
  exit 1
fi

if [[ -z "${EXAMPLE_ORDER_FILE}" || ! -f "${EXAMPLE_ORDER_FILE}" ]]; then
  echo "ERROR: EXAMPLE_ORDER_FILE (planner output) is required for this arm." >&2
  echo "Set EXAMPLE_ORDER_FILE to the planner-produced order file." >&2
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
if [[ -n "${TOUCAN_PATH}" ]]; then
  OPTIONAL_ARGS+=(--toucan_path "${TOUCAN_PATH}")
fi
if [[ -n "${MULTISOURCE_MAX_RECORDS}" ]]; then
  OPTIONAL_ARGS+=(--multisource_max_records "${MULTISOURCE_MAX_RECORDS}")
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

# Warmup: ratio-based by default (0.04 of total steps); an explicit
# WARMUP_STEPS wins and warmup_ratio is then omitted (HF default 0.0).
WARMUP_ARGS=()
if [[ -n "${WARMUP_STEPS}" ]]; then
  WARMUP_ARGS+=(--warmup_steps "${WARMUP_STEPS}")
else
  WARMUP_ARGS+=(--warmup_steps 0 --warmup_ratio "${WARMUP_RATIO}")
fi

if [[ "${USE_DEEPSPEED}" == "1" ]]; then
  LAUNCHER=(torchrun --nproc_per_node "${NPROC_PER_NODE}" --master_port "${MASTER_PORT:-29600}")
  DEEPSPEED_ARGS=(--deepspeed ./configs/ds_config_h200.json)
else
  # Plain torchrun DDP fallback (no deepspeed); GPU set still chosen via
  # CUDA_VISIBLE_DEVICES.
  LAUNCHER=(torchrun --nproc_per_node "${NPROC_PER_NODE}" --master_port "${MASTER_PORT:-29600}")
  DEEPSPEED_ARGS=()
fi

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
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
echo "TOUCAN_PATH=${TOUCAN_PATH}"
echo "MULTISOURCE_MAX_RECORDS=${MULTISOURCE_MAX_RECORDS}"
echo "DOC_MODE=${DOC_MODE}"
echo "MAX_DOC_LENGTH=${MAX_DOC_LENGTH}"
echo "MAX_DOC_NUM=${MAX_DOC_NUM}"
echo "MAX_TOOL_CHUNKS=${MAX_TOOL_CHUNKS}"
echo "MAX_LENGTH=${MAX_LENGTH}"
echo "MAX_SYSTEM_LENGTH=${MAX_SYSTEM_LENGTH}"
echo "MAX_TOOL_DEFINITION_TOKENS=${MAX_TOOL_DEFINITION_TOKENS}"
echo "MIN_TARGET_TOKENS=${MIN_TARGET_TOKENS}"
echo "REQUIRE_TOOL_CALL=${REQUIRE_TOOL_CALL}"
echo "ACTION_TOOL_CALL_FRAC=${ACTION_TOOL_CALL_FRAC}"
echo "LR=${LR}"
echo "SEED=${SEED} (HF TrainingArguments seed; 42 = every arm to date)"
echo "DDP_TIMEOUT=${DDP_TIMEOUT}"
echo "NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS}"
echo "WARMUP_RATIO=${WARMUP_RATIO}"
echo "WARMUP_STEPS=${WARMUP_STEPS}"
echo "PER_DEVICE_BS=${PER_DEVICE_BS}"
echo "GRAD_ACCUM=${GRAD_ACCUM}"
echo "C2KV_GIST_TRAIN_RATIOS=${C2KV_GIST_TRAIN_RATIOS}"
echo "C2KV_GIST_CHECKPOINT_USE_REENTRANT=${C2KV_GIST_CHECKPOINT_USE_REENTRANT}"
echo "SAVE_STEPS=${SAVE_STEPS}"
echo "LOGGING_STEPS=${LOGGING_STEPS}"

"${LAUNCHER[@]}" \
  agent/train_joint_next_action_c2kv.py \
  --device_type cuda \
  --attn_impl "${ATTN_IMPL}" \
  --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
  "${WARMUP_ARGS[@]}" \
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
  --example_order_file "${EXAMPLE_ORDER_FILE}" \
  --max_samples_per_session "${MAX_SAMPLES_PER_SESSION}" \
  --doc_mode "${DOC_MODE}" \
  --max_doc_length "${MAX_DOC_LENGTH}" \
  --max_doc_num "${MAX_DOC_NUM}" \
  --max_length "${MAX_LENGTH}" \
  --max_system_length "${MAX_SYSTEM_LENGTH}" \
  --max_tool_definition_tokens "${MAX_TOOL_DEFINITION_TOKENS}" \
  --min_target_tokens "${MIN_TARGET_TOKENS}" \
  --require_tool_call "${REQUIRE_TOOL_CALL}" \
  --action_tool_call_frac "${ACTION_TOOL_CALL_FRAC}" \
  --history_selection "${HISTORY_SELECTION}" \
  "${OPTIONAL_ARGS[@]}" \
  --output_dir "${OUTPUT_DIR}" \
  --logging_steps "${LOGGING_STEPS}" \
  --logging_nan_inf_filter False \
  --remove_unused_columns False \
  "${DEEPSPEED_ARGS[@]}" \
  --ddp_timeout "${DDP_TIMEOUT}" \
  --do_train True \
  --eval_strategy steps \
  --eval_steps "${EVAL_STEPS}" \
  --save_strategy steps \
  --save_steps "${SAVE_STEPS}" \
  "${DATALOADER_ARGS[@]}" \
  --bf16 True \
  --dataset_shuffle_seed "${DATASET_SHUFFLE_SEED}" \
  --seed "${SEED}"
