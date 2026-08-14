#!/usr/bin/env bash
# R3 window gate: print the id of the first 910B3 with >= MIN_FREE_MB free HBM.
# Polls until one frees up. Chip 6 is skipped (persistent Health=Warning).
# Usage: DEV=$(MIN_FREE_MB=28672 bash agent/r3_gate.sh)
set -euo pipefail
MIN_FREE_MB="${MIN_FREE_MB:-20480}"
SLEEP_SEC="${SLEEP_SEC:-90}"
free_mb() {
  local used
  used=$(npu-smi info | sed -n "/^| $1 .*910B3/,+1p" | tail -1 | grep -oE "[0-9]+ */ *65536" | cut -d/ -f1 | tr -d " ")
  [ -n "$used" ] && echo $((65536 - used)) || echo 0
}
while true; do
  for dev in 0 1 2 3 4 5 7; do
    fm=$(free_mb "$dev")
    if [ "$fm" -ge "$MIN_FREE_MB" ]; then
      echo "[r3_gate] dev=$dev free=${fm}MB >= ${MIN_FREE_MB}MB" >&2
      echo "$dev"
      exit 0
    fi
  done
  echo "[r3_gate] no chip with >= ${MIN_FREE_MB}MB free; sleep ${SLEEP_SEC}s" >&2
  sleep "$SLEEP_SEC"
done
