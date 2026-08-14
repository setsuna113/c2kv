#!/usr/bin/env bash
# Window-gated runner for agent/recompute_s4_logp.py on the shared NPU box.
# Loops until every frozen qid is done in OUT (clean row or permanent skip):
# before each launch it requires >= FREE_MEM_MIN_MB free HBM on the target
# device (parsed from `npu-smi info`), then runs the recompute with --resume so
# each launch only scores what is left. OOM rows are retried by later launches;
# non-OOM skips are treated as permanent (mirrors the --resume semantics).
#
# Usage:  bash agent/recompute_s4_logp_npu.sh
# Env overrides: QIDS_FILE OUT MODEL_PATH TOKENIZER_PATH DATASET_PATH DEVICE
#                ATTN_IMPL RATIO MAX_EXAMPLES FREE_MEM_MIN_MB RETRY_SEC
#                NPU_PHYSICAL_ID PYTHON_BIN ASCEND_RT_VISIBLE_DEVICES
set -u  # NOT -e: the retry loop must survive npu-smi/launch failures.

export PYTHONPATH="$(pwd)/python:$(pwd)/python/inference:$(pwd)/agent:${PYTHONPATH:-}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"

QIDS_FILE="${QIDS_FILE:-./configs/s4_frozen_qids.json}"
OUT="${OUT:-./outputs/s4_logp_recompute.jsonl}"
MODEL_PATH="${MODEL_PATH:-./checkpoints/qwen3-4b-agent-history-c2kv-npu}"
TOKENIZER_PATH="${TOKENIZER_PATH:-./models/Qwen3-4B-Instruct-2507}"
DATASET_PATH="${DATASET_PATH:-./datasets/agent-llm-traces}"
DEVICE="${DEVICE:-npu:0}"
ATTN_IMPL="${ATTN_IMPL:-eager}"
RATIO="${RATIO:-4}"
MAX_EXAMPLES="${MAX_EXAMPLES:-0}"
FREE_MEM_MIN_MB="${FREE_MEM_MIN_MB:-20480}"
RETRY_SEC="${RETRY_SEC:-120}"
# npu-smi reports PHYSICAL device ids; ASCEND_RT_VISIBLE_DEVICES remaps them for
# the process. Gate the physical card backing the visible one (first of the list).
NPU_PHYSICAL_ID="${NPU_PHYSICAL_ID:-${ASCEND_RT_VISIBLE_DEVICES%%,*}}"
PYTHON_BIN="${PYTHON_BIN:-python}"

mkdir -p "$(dirname "${OUT}")"

log() {
  echo "[$(date '+%F %T')] $*"
}

total_qids() {
  "${PYTHON_BIN}" - "${QIDS_FILE}" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as fh:
    print(len(json.load(fh)["qids"]))
PY
}

# Count frozen qids already done in OUT: clean rows (both logps non-null) or
# permanent skips (skipped and reason != "oom"). Mirrors --resume semantics.
count_done() {
  "${PYTHON_BIN}" - "${QIDS_FILE}" "${OUT}" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as fh:
    qids = set(json.load(fh)["qids"])
done = 0
try:
    with open(sys.argv[2], encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("qid") not in qids:
                continue
            clean = row.get("logp_prefix_c2kv") is not None and row.get("logp_prefix_full") is not None
            permanent_skip = bool(row.get("skipped")) and row.get("skip_reason") != "oom"
            if clean or permanent_skip:
                done += 1
except FileNotFoundError:
    pass
print(done)
PY
}

# Free HBM (MiB) on physical device $1, parsed from the `npu-smi info` table.
# Each device has two rows (card row with the name, chip row with the usages);
# the chip row carries "used / total" MB pairs. On 910B3 the rightmost/largest
# pair is HBM-Usage (total 65536), so we take the pair with the largest total.
# Prints the free MiB, or nothing (and returns 1) if parsing fails.
npu_hbm_free_mb() {
  local dev="$1" rows pairs pair used total best_total=0 best_used=0
  rows="$(npu-smi info 2>/dev/null | grep -E "^\|[[:space:]]+${dev}[[:space:]]")" || return 1
  [[ -n "${rows}" ]] || return 1
  pairs="$(printf '%s\n' "${rows}" | grep -oE '[0-9]+[[:space:]]*/[[:space:]]*[0-9]+')" || return 1
  [[ -n "${pairs}" ]] || return 1
  while IFS= read -r pair; do
    used="${pair%%/*}"; total="${pair##*/}"
    used="${used//[[:space:]]/}"; total="${total//[[:space:]]/}"
    [[ "${used}" =~ ^[0-9]+$ && "${total}" =~ ^[0-9]+$ ]] || continue
    if (( total > best_total )); then
      best_total="${total}"; best_used="${used}"
    fi
  done <<< "${pairs}"
  (( best_total > 0 )) || return 1
  echo $(( best_total - best_used ))
}

total="$(total_qids)"
if [[ -z "${total}" || ! "${total}" =~ ^[0-9]+$ ]]; then
  log "fatal: cannot read qids from ${QIDS_FILE}"
  exit 2
fi
log "start: ${total} qids, qids_file=${QIDS_FILE} out=${OUT} device=${DEVICE} physical_id=${NPU_PHYSICAL_ID} gate=${FREE_MEM_MIN_MB}MB retry=${RETRY_SEC}s"

while true; do
  done_count="$(count_done)"
  if [[ -z "${done_count}" || ! "${done_count}" =~ ^[0-9]+$ ]]; then
    log "warn: done-count failed; retry in ${RETRY_SEC}s"
    sleep "${RETRY_SEC}"
    continue
  fi
  if (( done_count >= total )); then
    log "complete: ${done_count}/${total} qids done; exit"
    exit 0
  fi
  free_mb="$(npu_hbm_free_mb "${NPU_PHYSICAL_ID}")"
  if [[ -z "${free_mb}" || ! "${free_mb}" =~ ^[0-9]+$ ]]; then
    log "warn: npu-smi parse failed for device ${NPU_PHYSICAL_ID}; retry in ${RETRY_SEC}s"
    sleep "${RETRY_SEC}"
    continue
  fi
  if (( free_mb < FREE_MEM_MIN_MB )); then
    log "wait: free=${free_mb}MB < ${FREE_MEM_MIN_MB}MB (done ${done_count}/${total}); retry in ${RETRY_SEC}s"
    sleep "${RETRY_SEC}"
    continue
  fi
  log "launch: free=${free_mb}MB >= ${FREE_MEM_MIN_MB}MB (done ${done_count}/${total})"
  "${PYTHON_BIN}" agent/recompute_s4_logp.py \
    --qids_file "${QIDS_FILE}" \
    --out "${OUT}" \
    --resume \
    --max_examples "${MAX_EXAMPLES}" \
    --model_path "${MODEL_PATH}" \
    --tokenizer_path "${TOKENIZER_PATH}" \
    --dataset_path "${DATASET_PATH}" \
    --device "${DEVICE}" \
    --attn_impl "${ATTN_IMPL}" \
    --ratio "${RATIO}"
  rc=$?
  log "exit rc=${rc} (done was ${done_count}/${total}); re-checking"
  if (( rc != 0 )); then
    sleep "${RETRY_SEC}"
  fi
done
