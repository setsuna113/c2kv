#!/usr/bin/env bash
# D1' step-2 four-condition readout — window-gated NPU queue runner.
#
# Round-1 queue style: only fire a pass when some visible NPU chip has at least
# MIN_FREE_MIB free HBM (shared-box etiquette: the 910B3 cards are contested);
# each pass runs agent/eval_condition_variants.py with --resume, and the loop
# exits once the output jsonl holds one row per manifest qid (skipped rows are
# final and count toward completion). Crashes/OOM passes are retried at the
# next window — rows are appended + flushed per batch, so nothing is lost.
#
# Usage:
#   CHECKPOINT=./checkpoints/<conditioned-run>/checkpoint-XXX \
#     bash agent/eval_condition_variants_npu.sh
set -u

export PYTHONPATH="$(pwd)/python:$(pwd)/python/inference:$(pwd)/agent:${PYTHONPATH:-}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"
export C2KV_GIST_DOC_MICROBATCH="${C2KV_GIST_DOC_MICROBATCH:-1}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-max_split_size_mb:128}"

PYTHON_BIN="${PYTHON_BIN:-python}"
CHECKPOINT="${CHECKPOINT:-}"
TOKENIZER_PATH="${TOKENIZER_PATH:-}"
VAL_MANIFEST="${VAL_MANIFEST:-configs/d1prime_frozen_val.json}"
OUT="${OUT:-./outputs/d1prime_condition_readout.jsonl}"
DEVICE="${DEVICE:-npu:0}"
ATTN_IMPL="${NPU_ATTN_IMPL:-eager}"
RATIO="${RATIO:-4}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_EXAMPLES="${MAX_EXAMPLES:-0}"
VARIANT_SEED="${VARIANT_SEED:-0}"
DTYPE="${DTYPE:-bf16}"
DATASET_PATH="${DATASET_PATH:-}"
# Must match the conditioned training run's window (0/off would strip the
# condition_text the readout pairs on).
export C2KV_CONDITION_WINDOW_TOKENS="${C2KV_CONDITION_WINDOW_TOKENS:-256}"

MIN_FREE_MIB="${MIN_FREE_MIB:-20480}"
SLEEP_SEC="${SLEEP_SEC:-300}"
MAX_ROUNDS="${MAX_ROUNDS:-0}"  # 0 = unlimited

ts() { date '+%Y-%m-%d %H:%M:%S'; }

if [[ -z "${CHECKPOINT}" ]]; then
  echo "[$(ts)] ERROR: set CHECKPOINT to the conditioned checkpoint dir" >&2
  exit 1
fi
if [[ ! -f "${VAL_MANIFEST}" ]]; then
  echo "[$(ts)] ERROR: manifest not found: ${VAL_MANIFEST} (run agent/build_frozen_val_manifest.py first)" >&2
  exit 1
fi

# Max free HBM (MiB) across the visible chips, parsed from `npu-smi info`
# "HBM-Usage" used/total pairs. Prints 0 when npu-smi is unavailable.
free_hbm_mib() {
  npu-smi info 2>/dev/null \
    | sed -nE 's/.*[^0-9]([0-9]+)[[:space:]]*\/[[:space:]]*([0-9]+)[^0-9]*$/\1 \2/p' \
    | awk '{ f = $2 - $1; if (f > m) m = f } END { print (m == "" ? 0 : m) }'
}

MANIFEST_N="$("${PYTHON_BIN}" -c 'import json, sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["n"])' "${VAL_MANIFEST}")"
if [[ -z "${MANIFEST_N}" || "${MANIFEST_N}" -le 0 ]]; then
  echo "[$(ts)] ERROR: could not read manifest n from ${VAL_MANIFEST}" >&2
  exit 1
fi

EXTRA_ARGS=()
if [[ -n "${TOKENIZER_PATH}" ]]; then
  EXTRA_ARGS+=(--tokenizer "${TOKENIZER_PATH}")
fi
if [[ -n "${DATASET_PATH}" ]]; then
  EXTRA_ARGS+=(--dataset_path "${DATASET_PATH}")
fi

echo "[$(ts)] D1' condition readout queue: CHECKPOINT=${CHECKPOINT} OUT=${OUT}"
echo "[$(ts)] manifest=${VAL_MANIFEST} n=${MANIFEST_N} gate>=${MIN_FREE_MIB} MiB device=${DEVICE} ratio=${RATIO} batch=${BATCH_SIZE} window=${C2KV_CONDITION_WINDOW_TOKENS}"

round=0
while true; do
  rows=0
  if [[ -f "${OUT}" ]]; then
    rows="$(wc -l < "${OUT}")"
  fi
  if (( rows >= MANIFEST_N )); then
    echo "[$(ts)] complete: ${rows}/${MANIFEST_N} rows in ${OUT}"
    break
  fi
  free_mib="$(free_hbm_mib)"
  if (( free_mib < MIN_FREE_MIB )); then
    echo "[$(ts)] wait: free HBM ${free_mib} MiB < ${MIN_FREE_MIB} MiB; rows=${rows}/${MANIFEST_N}; sleep ${SLEEP_SEC}s"
    sleep "${SLEEP_SEC}"
    continue
  fi
  round=$((round + 1))
  echo "[$(ts)] round ${round}: free HBM ${free_mib} MiB; rows=${rows}/${MANIFEST_N}; launching pass"
  "${PYTHON_BIN}" agent/eval_condition_variants.py \
    --checkpoint "${CHECKPOINT}" \
    --val_manifest "${VAL_MANIFEST}" \
    --out "${OUT}" \
    --resume \
    --batch_size "${BATCH_SIZE}" \
    --device "${DEVICE}" \
    --attn_impl "${ATTN_IMPL}" \
    --ratio "${RATIO}" \
    --max_examples "${MAX_EXAMPLES}" \
    --variant_seed "${VARIANT_SEED}" \
    --dtype "${DTYPE}" \
    --condition_window_tokens "${C2KV_CONDITION_WINDOW_TOKENS}" \
    "${EXTRA_ARGS[@]}"
  pass_status=$?
  if (( pass_status != 0 )); then
    echo "[$(ts)] pass exited with status ${pass_status}; will retry at next window"
  fi
  if (( MAX_ROUNDS > 0 && round >= MAX_ROUNDS )); then
    rows=0
    if [[ -f "${OUT}" ]]; then
      rows="$(wc -l < "${OUT}")"
    fi
    echo "[$(ts)] MAX_ROUNDS=${MAX_ROUNDS} reached with rows=${rows}/${MANIFEST_N}; stopping"
    exit 2
  fi
  sleep 30
done
