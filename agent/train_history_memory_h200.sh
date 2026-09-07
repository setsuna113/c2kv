#!/usr/bin/env bash
# Launch the B-line history-memory trainer on one 2x H200 node.
#
# The default run order is the matched seed-42 pair: arm C, then arm B.
# Set SEEDS="42 43 44" for the formal three-seed run.  This launcher does not
# restart, prune, kill, evaluate, or upload anything.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TRAIN_SCRIPT="${REPO_ROOT}/agent/train_history_memory.py"

usage() {
  cat <<'EOF'
Usage: bash agent/train_history_memory_h200.sh [options]

Options:
  --env-file PATH     Source PATH before applying command-line selections.
                      Default: configs/b_history_h200.env when it exists.
  --arm C|B|both      Run one arm or the matched C-then-B pair (default: both).
  --seed N            Run one seed. Overrides SEEDS.
  --seeds "N ..."     Run a whitespace- or comma-separated seed list.
  --resume-from-checkpoint PATH
                      Resume PATH for one explicitly selected arm and seed.
  --dry-run           Print the resolved training commands without preflight.
  --preflight-only    Validate the environment and inputs, then exit.
  -h, --help          Show this help.

Configuration is supplied through the environment. Start from
configs/b_history_h200.env.example. The default pair is C then B at seed 42;
the formal seed set is 42, 43, 44.
EOF
}

ENV_FILE=""
args=("$@")
for arg in "${args[@]}"; do
  if [[ "${arg}" == "-h" || "${arg}" == "--help" ]]; then
    usage
    exit 0
  fi
done
for ((i = 0; i < ${#args[@]}; i++)); do
  case "${args[$i]}" in
    --env-file)
      ((i + 1 < ${#args[@]})) || { echo "--env-file requires a path" >&2; exit 2; }
      ENV_FILE="${args[$((i + 1))]}"
      ((i += 1))
      ;;
    --env-file=*) ENV_FILE="${args[$i]#*=}" ;;
  esac
done

if [[ -z "${ENV_FILE}" && -f "${REPO_ROOT}/configs/b_history_h200.env" ]]; then
  ENV_FILE="${REPO_ROOT}/configs/b_history_h200.env"
fi
if [[ -n "${ENV_FILE}" ]]; then
  [[ -f "${ENV_FILE}" ]] || { echo "env file not found: ${ENV_FILE}" >&2; exit 2; }
  # shellcheck source=/dev/null
  source "${ENV_FILE}"
fi

G_REPO="${G_REPO:-/inspire/hdd/project/wuliqifa/yanjunchi-24040/yancheng/c2kv}"
GU_BASE="${GU_BASE:-/inspire/hdd/global_user/yanjunchi-24040/yancheng_c2kv_h200}"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv-b-history/bin/python}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-${G_REPO}/models/Qwen3-4B-Instruct-2507}"
DATA_PATH="${DATA_PATH:-${GU_BASE}/data/b_history}"
OUTPUT_BASE="${OUTPUT_BASE:-${GU_BASE}/checkpoints/b_history}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
REQUIRE_H200="${REQUIRE_H200:-1}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-2}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-2}"
LEARNING_RATE="${LEARNING_RATE:-5e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
WARMUP_RATIO="${WARMUP_RATIO:-0.06}"
RATIOS="${RATIOS:-4,8}"
MAX_CHUNK_TOKENS="${MAX_CHUNK_TOKENS:-768}"
CHUNK_OVERLAP="${CHUNK_OVERLAP:-64}"
MAX_SYSTEM_TOKENS="${MAX_SYSTEM_TOKENS:-8192}"
MAX_WORKSPACE_TOKENS="${MAX_WORKSPACE_TOKENS:-4096}"
MAX_TARGET_TOKENS="${MAX_TARGET_TOKENS:-4096}"
MAX_CHUNKS="${MAX_CHUNKS:-48}"
SAVE_STEPS="${SAVE_STEPS:-1000}"
LOGGING_STEPS="${LOGGING_STEPS:-1}"
MAX_STEPS="${MAX_STEPS:--1}"
ATTN_IMPL="${ATTN_IMPL:-sdpa}"
BF16="${BF16:-true}"
WANDB_MODE="${WANDB_MODE:-offline}"
WANDB_ENTITY="${WANDB_ENTITY:-liuyc1025-university-of-cambridge}"
WANDB_PROJECT="${WANDB_PROJECT:-c2kv-history-b}"
WANDB_DIR="${WANDB_DIR:-${OUTPUT_BASE}/wandb}"
SEEDS="${SEEDS:-42}"

SELECTED_ARM="both"
CLI_RESUME=""
DRY_RUN=0
PREFLIGHT_ONLY=0
for ((i = 0; i < ${#args[@]}; i++)); do
  case "${args[$i]}" in
    --env-file) ((i += 1)) ;;
    --env-file=*) ;;
    --arm)
      ((i + 1 < ${#args[@]})) || { echo "--arm requires C, B, or both" >&2; exit 2; }
      SELECTED_ARM="${args[$((i + 1))]}"
      ((i += 1))
      ;;
    --arm=*) SELECTED_ARM="${args[$i]#*=}" ;;
    --seed)
      ((i + 1 < ${#args[@]})) || { echo "--seed requires an integer" >&2; exit 2; }
      SEEDS="${args[$((i + 1))]}"
      ((i += 1))
      ;;
    --seed=*) SEEDS="${args[$i]#*=}" ;;
    --seeds)
      ((i + 1 < ${#args[@]})) || { echo "--seeds requires a list" >&2; exit 2; }
      SEEDS="${args[$((i + 1))]}"
      ((i += 1))
      ;;
    --seeds=*) SEEDS="${args[$i]#*=}" ;;
    --resume-from-checkpoint)
      ((i + 1 < ${#args[@]})) || { echo "--resume-from-checkpoint requires a path" >&2; exit 2; }
      CLI_RESUME="${args[$((i + 1))]}"
      ((i += 1))
      ;;
    --resume-from-checkpoint=*) CLI_RESUME="${args[$i]#*=}" ;;
    --dry-run) DRY_RUN=1 ;;
    --preflight-only) PREFLIGHT_ONLY=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: ${args[$i]}" >&2; usage >&2; exit 2 ;;
  esac
done

case "${SELECTED_ARM}" in
  C|B) arms=("${SELECTED_ARM}") ;;
  both) arms=(C B) ;;
  *) echo "--arm must be C, B, or both; got ${SELECTED_ARM}" >&2; exit 2 ;;
esac

if (( DRY_RUN && PREFLIGHT_ONLY )); then
  echo "--dry-run and --preflight-only are mutually exclusive" >&2
  exit 2
fi

SEEDS="${SEEDS//,/ }"
read -r -a seeds <<< "${SEEDS}"
(( ${#seeds[@]} > 0 )) || { echo "SEEDS resolved to an empty list" >&2; exit 2; }
for seed in "${seeds[@]}"; do
  [[ "${seed}" =~ ^[0-9]+$ ]] || { echo "invalid seed: ${seed}" >&2; exit 2; }
done
if [[ -n "${CLI_RESUME}" && ( "${SELECTED_ARM}" == "both" || ${#seeds[@]} -ne 1 ) ]]; then
  echo "--resume-from-checkpoint requires one --arm (C or B) and one --seed" >&2
  exit 2
fi

[[ "${NPROC_PER_NODE}" =~ ^[1-9][0-9]*$ ]] || { echo "NPROC_PER_NODE must be positive" >&2; exit 2; }
case "${ATTN_IMPL}" in
  sdpa|eager|flex_attention) ;;
  *) echo "ATTN_IMPL must be sdpa, eager, or flex_attention" >&2; exit 2 ;;
esac
case "${BF16}" in
  true|false) ;;
  *) echo "BF16 must be true or false" >&2; exit 2 ;;
esac
case "${WANDB_MODE}" in
  disabled|offline|online) ;;
  *) echo "WANDB_MODE must be disabled, offline, or online" >&2; exit 2 ;;
esac

export CUDA_VISIBLE_DEVICES
export PYTHONPATH="${REPO_ROOT}/python:${REPO_ROOT}/agent:${PYTHONPATH:-}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export WANDB_DIR

preflight() {
  [[ -x "${PYTHON_BIN}" ]] || { echo "Python is not executable: ${PYTHON_BIN}" >&2; return 1; }
  [[ -f "${TRAIN_SCRIPT}" ]] || { echo "training entry point missing: ${TRAIN_SCRIPT}" >&2; return 1; }
  "${PYTHON_BIN}" "${TRAIN_SCRIPT}" --help >/dev/null
  local manifest_path="${DATA_PATH}"
  if [[ -d "${DATA_PATH}" ]]; then
    manifest_path="${DATA_PATH}/manifest.json"
  fi
  [[ -f "${manifest_path}" ]] || { echo "data manifest missing: ${manifest_path}" >&2; return 1; }
  [[ -s "${manifest_path}" ]] || { echo "data manifest is empty: ${manifest_path}" >&2; return 1; }
  [[ -d "${MODEL_NAME_OR_PATH}" ]] || { echo "model directory missing: ${MODEL_NAME_OR_PATH}" >&2; return 1; }
  [[ -f "${MODEL_NAME_OR_PATH}/config.json" ]] || { echo "model config missing" >&2; return 1; }
  [[ -f "${MODEL_NAME_OR_PATH}/tokenizer.json" ]] || { echo "model tokenizer.json missing" >&2; return 1; }
  if [[ ! -f "${MODEL_NAME_OR_PATH}/model.safetensors" && ! -f "${MODEL_NAME_OR_PATH}/model.safetensors.index.json" ]]; then
    echo "model weights missing under ${MODEL_NAME_OR_PATH}" >&2
    return 1
  fi

  "${PYTHON_BIN}" - "${NPROC_PER_NODE}" "${REQUIRE_H200}" <<'PY'
import sys

import accelerate
import numpy
import pyarrow
import torch
import transformers

expected_gpus = int(sys.argv[1])
require_h200 = sys.argv[2] == "1"
versions = {
    "torch": torch.__version__.split("+")[0],
    "transformers": transformers.__version__,
    "numpy": numpy.__version__,
    "pyarrow": pyarrow.__version__,
    "accelerate": accelerate.__version__,
}
expected = {
    "torch": "2.9.0",
    "transformers": "5.8.0",
    "numpy": "2.3.4",
    "pyarrow": "25.0.1",
    "accelerate": "1.14.0",
}
if versions != expected:
    raise SystemExit(f"dependency version mismatch: got {versions}, expected {expected}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available")
if torch.cuda.device_count() != expected_gpus:
    raise SystemExit(
        f"visible GPU count {torch.cuda.device_count()} != NPROC_PER_NODE {expected_gpus}"
    )
names = [torch.cuda.get_device_name(i) for i in range(expected_gpus)]
if require_h200 and any("H200" not in name.upper() for name in names):
    raise SystemExit(f"expected H200 devices, got {names}")
print({"python": sys.version.split()[0], **versions, "cuda": torch.version.cuda, "gpus": names})
PY

  if [[ "${WANDB_MODE}" != "disabled" ]]; then
    "${PYTHON_BIN}" -c "import wandb; assert wandb.__version__ == '0.29.0', wandb.__version__; print('wandb', wandb.__version__)"
  fi

  mkdir -p "${OUTPUT_BASE}" "${WANDB_DIR}"
  [[ -w "${OUTPUT_BASE}" ]] || { echo "output directory is not writable: ${OUTPUT_BASE}" >&2; return 1; }
  [[ -w "${WANDB_DIR}" ]] || { echo "W&B directory is not writable: ${WANDB_DIR}" >&2; return 1; }
}

print_command() {
  printf '%q ' "$@"
  printf '\n'
}

run_one() {
  local arm="$1"
  local seed="$2"
  local output_dir="${OUTPUT_BASE}/arm-${arm}/seed-${seed}"
  local resume=""
  if [[ "${arm}" == "C" ]]; then
    resume="${RESUME_FROM_CHECKPOINT_C:-}"
  else
    resume="${RESUME_FROM_CHECKPOINT_B:-}"
  fi
  if [[ -n "${CLI_RESUME}" ]]; then
    resume="${CLI_RESUME}"
  fi

  local cmd=(
    "${PYTHON_BIN}" -m torch.distributed.run
    --standalone
    --nproc_per_node "${NPROC_PER_NODE}"
    "${TRAIN_SCRIPT}"
    --data_path "${DATA_PATH}"
    --model_name_or_path "${MODEL_NAME_OR_PATH}"
    --output_dir "${output_dir}"
    --arm "${arm}"
    --seed "${seed}"
    --per_device_batch_size "${PER_DEVICE_BATCH_SIZE}"
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
    --num_train_epochs "${NUM_TRAIN_EPOCHS}"
    --learning_rate "${LEARNING_RATE}"
    --weight_decay "${WEIGHT_DECAY}"
    --warmup_ratio "${WARMUP_RATIO}"
    --ratios "${RATIOS}"
    --max_chunk_tokens "${MAX_CHUNK_TOKENS}"
    --chunk_overlap "${CHUNK_OVERLAP}"
    --max_system_tokens "${MAX_SYSTEM_TOKENS}"
    --max_workspace_tokens "${MAX_WORKSPACE_TOKENS}"
    --max_target_tokens "${MAX_TARGET_TOKENS}"
    --max_chunks "${MAX_CHUNKS}"
    --save_steps "${SAVE_STEPS}"
    --logging_steps "${LOGGING_STEPS}"
    --max_steps "${MAX_STEPS}"
    --attn_impl "${ATTN_IMPL}"
    --wandb_mode "${WANDB_MODE}"
    --wandb_entity "${WANDB_ENTITY}"
    --wandb_project "${WANDB_PROJECT}"
  )
  if [[ "${BF16}" == "true" ]]; then
    cmd+=(--bf16)
  else
    cmd+=(--no-bf16)
  fi
  local wandb_run_id=""
  if [[ "${arm}" == "C" ]]; then
    wandb_run_id="${WANDB_RUN_ID_C:-}"
  else
    wandb_run_id="${WANDB_RUN_ID_B:-}"
  fi
  if [[ -n "${wandb_run_id}" ]]; then
    cmd+=(--wandb_run_id "${wandb_run_id}")
  fi
  if [[ -n "${resume}" ]]; then
    cmd+=(--resume_from_checkpoint "${resume}")
  fi

  echo "arm=${arm} seed=${seed} output=${output_dir}"
  if (( DRY_RUN )); then
    print_command "${cmd[@]}"
  else
    "${cmd[@]}"
  fi
}

if (( ! DRY_RUN )); then
  preflight
fi
if (( PREFLIGHT_ONLY )); then
  echo "preflight OK"
  exit 0
fi

for seed in "${seeds[@]}"; do
  for arm in "${arms[@]}"; do
    run_one "${arm}" "${seed}"
  done
done
