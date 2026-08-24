#!/usr/bin/env bash
# joint_gated_run.sh — free-HBM-window gating runner for the shared 8x 910B3
# NPU server (each chip has 65536 MB HBM).
#
# Parses `npu-smi info` per chip: the used MB appears in the two-line per-chip
# block (a chip line `| N   910B3 ...` followed by a line whose HBM-Usage
# column shows `<used> / 65536`); free = 65536 - used.  Chips 0-7 are scanned
# and the FIRST with free >= MIN_FREE_MB is taken; if none qualifies, sleeps
# POLL_SEC and retries until MAX_WAIT_HOURS expires (then exits 3).
#
# On acquiring a chip, runs "$@" with ASCEND_RT_VISIBLE_DEVICES=<chip> and
# tees stdout/stderr to LOG_FILE.  Acquisition/launch/exit lines are printed
# with timestamps (and appended to LOG_FILE); the exit status is the wrapped
# command's status.
#
# Usage:
#   bash scripts/joint_gated_run.sh <command> [args...]
#   MIN_FREE_MB=40960 POLL_SEC=90 MAX_WAIT_HOURS=72 LOG_FILE=./queue.log \
#     bash scripts/joint_gated_run.sh bash agent/train_joint_next_action_c2kv_npu.sh
#
# Knobs (env): MIN_FREE_MB (40960) POLL_SEC (90) MAX_WAIT_HOURS (72)
#              LOG_FILE (./queue_$(date +%s).log) CHIP_IDS ("0 1 2 3 4 5 6 7")
set -euo pipefail

MIN_FREE_MB="${MIN_FREE_MB:-40960}"
POLL_SEC="${POLL_SEC:-90}"
MAX_WAIT_HOURS="${MAX_WAIT_HOURS:-72}"
LOG_FILE="${LOG_FILE:-./queue_$(date +%s).log}"
CHIP_TOTAL_MB=65536
CHIP_IDS="${CHIP_IDS:-0 1 2 3 4 5 6 7}"

ts() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(ts)] $*" | tee -a "${LOG_FILE}"; }

if [[ $# -lt 1 ]]; then
  echo "usage: MIN_FREE_MB=.. POLL_SEC=.. MAX_WAIT_HOURS=.. LOG_FILE=.. bash $0 <command> [args...]" >&2
  exit 2
fi

# Print the used HBM MB of chip $2 from the `npu-smi info` text on stdin
# (empty output when the block cannot be parsed).
chip_used_mb() {
  awk -v chip="$1" '
    $2 == chip && $3 ~ /^910B3/ { pending = 1; next }
    pending {
      line = $0; used = ""
      while (match(line, /[0-9]+[ \t]*\/[ \t]*65536/)) {
        tok = substr(line, RSTART, RLENGTH)
        sub(/[ \t]*\/[ \t]*65536/, "", tok)
        used = tok
        line = substr(line, RSTART + RLENGTH)
      }
      print used
      exit
    }
  '
}

deadline=$(( $(date +%s) + MAX_WAIT_HOURS * 3600 ))
acquired_chip=""
acquired_free=0
while true; do
  smi="$(npu-smi info 2>/dev/null || true)"
  free_report=()
  for chip in ${CHIP_IDS}; do
    used="$(printf '%s\n' "${smi}" | chip_used_mb "${chip}")"
    if [[ "${used}" =~ ^[0-9]+$ ]]; then
      free=$(( CHIP_TOTAL_MB - used ))
    else
      free=-1  # unparseable: treat as unavailable
    fi
    free_report+=("${chip}:${free}")
    if [[ -z "${acquired_chip}" ]] && (( free >= MIN_FREE_MB )); then
      acquired_chip="${chip}"
      acquired_free=${free}
    fi
  done
  if [[ -n "${acquired_chip}" ]]; then
    break
  fi
  if (( $(date +%s) >= deadline )); then
    log "TIMEOUT: no chip with free >= ${MIN_FREE_MB} MB within ${MAX_WAIT_HOURS}h (last free MB: ${free_report[*]})"
    exit 3
  fi
  log "waiting: no chip with free >= ${MIN_FREE_MB} MB (free MB: ${free_report[*]}); sleeping ${POLL_SEC}s"
  sleep "${POLL_SEC}"
done

log "acquired chip ${acquired_chip} (free=${acquired_free} MB >= ${MIN_FREE_MB} MB)"
log "launch: ASCEND_RT_VISIBLE_DEVICES=${acquired_chip} $*"
set +e
ASCEND_RT_VISIBLE_DEVICES="${acquired_chip}" "$@" 2>&1 | tee -a "${LOG_FILE}"
rc=$?
set -e
log "exit: status=${rc} chip=${acquired_chip}"
exit "${rc}"
