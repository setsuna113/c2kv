#!/usr/bin/env bash
set -o pipefail
T=$(cat /tmp/c2kv-toolsnorm52.T)
export ROOT="$T/client"
export SGLANG_ROOT="$T/sglang"
export PYTHONPATH="$ROOT:$SGLANG_ROOT/python"
export BFCL_PROJECT_ROOT="$T/bfcl_state"
export PYTHONDONTWRITEBYTECODE=1
export BFCL_PYTHON=/home/zhuyuhan/miniconda3/envs/bfcl/bin/python
export SGLANG_PYTHON=/home/zhuyuhan/miniconda3/envs/sglang/bin/python
export IDS_PATH="$T/inputs/correct_ids.txt"
export REFERENCE_DETAILS_PATH="$T/inputs/details.jsonl"
export MAX_EXAMPLES=52 RATIO=4 CHECKPOINT_INTERVAL=4
export REPAIR_TRIGGER=oracle
export REPAIR_EXTRACT_SOURCE=auto
export C2KV_APPEND_POSITION_FRAME=wrapper
export MAX_COMPLETION_TOKENS=4096
export CLEAN_OUTPUT=0 RUN_COMPARE=0 USE_REPAIR_PLAN=0
export DEVICE=6 PORT=34770
export DEVICES="$DEVICE" PORTS="$PORT"
for arm in c2kv d_corr_w2 d_corr_replace_w2; do
  echo "[$(date '+%F %T')] ARM $arm starting"
  RUN_ROOT="$T/runs/$arm" ARMS="$arm" \
    bash "$ROOT/c2kv_eval/scripts/run_bfcl_kv_repair_sweep.sh" \
    >"$T/${arm}.launcher.log" 2>&1
  rc=$?
  echo "[$(date '+%F %T')] ARM $arm rc=$rc"
  if [ $rc -ne 0 ]; then echo "STOP on rc=$rc"; break; fi
done
echo "[$(date '+%F %T')] ALL DONE"
