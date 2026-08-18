#!/usr/bin/env bash
# g_joint_autorun.sh — session-independent Gate-1 -> Phase-3-small pipeline.
#
# Runs entirely on the NPU server under nohup; survives ssh/session drops.
# Stages:
#   0  wait for lrcal2_{5e-7,5e-6,5e-5}/model.safetensors          (max 9h)
#   1  Gate-1 evals (joint c2kv,full @8, 128 ex) on 3 chips        (max 3h)
#   2  rule-based LR pick (scripts/g_joint_gate1_pick.py, guarded)
#   3  v2 frozen example-order file for the arms                   (max 2h)
#   4  launch the four small-budget arms on 4 chips and exit
#
# Guards: any stage failure writes $G/autorun_FAILED with the reason and
# exits non-zero — the pipeline prefers idleness over burning 4x33h on a
# broken recipe.  Progress goes to $G/autorun_status.log.
set -uo pipefail

G="$HOME/c2kv/outputs_lyc/g_joint"
WT="$HOME/c2kv-gjoint"
PY="$HOME/envs/c2kv/bin/python"
STATUS="$G/autorun_status.log"

export PYTHONPATH="$WT/python:$WT/agent:${PYTHONPATH:-}"
export PATH="$HOME/envs/c2kv/bin:/usr/local/bin:/usr/bin:/bin"
source /usr/local/Ascend/ascend-toolkit/set_env.sh >/dev/null 2>&1 || true
cd "$WT" || exit 1

log() { echo "[$(date '+%F %T')] $*" | tee -a "$STATUS"; }
fail() { echo "[$(date '+%F %T')] FAIL: $*" | tee -a "$STATUS"; echo "$*" > "$G/autorun_FAILED"; exit 1; }

free_mb() {  # $1 = chip id -> free HBM in MB
  npu-smi info 2>/dev/null | sed -n "/^| $1 .*910B3/,+1p" | tail -1 | awk -F'/' '{gsub(/[^0-9]/,"",$1); print 65536 - $1}' | tail -1
}

wait_chip() {  # $1 = preferred chip, $2 = min free MB, $3 = max seconds -> echoes chip
  local pref="$1" min="$2" max="$3" waited=0 c
  while (( waited < max )); do
    for c in "$pref" 0 1 2 3 4 5 6 7; do
      local fm; fm=$(free_mb "$c")
      if [[ -n "$fm" ]] && (( fm >= min )); then echo "$c"; return 0; fi
    done
    sleep 90; waited=$((waited + 90))
  done
  return 1
}

# ---------------------------------------------------------------- stage 0
log "stage 0: waiting for lrcal2 checkpoints"
WAITED=0
for T in 5e-7 5e-6 5e-5; do
  while [[ ! -f "$G/lrcal2_$T/model.safetensors" ]]; do
    if (( WAITED > 32400 )); then fail "stage 0 timeout waiting lrcal2_$T"; fi
    if grep -qE "Traceback|RuntimeError" "$G/lrcal2_$T.log" 2>/dev/null && ! pgrep -f train_joint_next_action >/dev/null; then
      fail "lrcal2_$T died (see lrcal2_$T.log)"
    fi
    sleep 300; WAITED=$((WAITED + 300))
  done
done
log "stage 0 done: all lrcal2 checkpoints saved"

# ---------------------------------------------------------------- stage 1
log "stage 1: gate-1 evals"
declare -A EVALPID=()
IDX=0
for T in 5e-7 5e-6 5e-5; do
  CHIP=$(wait_chip "$IDX" 40960 7200) || fail "stage 1: no free chip for eval $T"
  log "gate1 eval $T on chip $CHIP"
  MODEL_PATH="$G/lrcal2_$T" \
  BASE_MODEL="$HOME/c2kv/models/Qwen3-4B-Instruct-2507" \
  DATASET_PATH="$HOME/c2kv/datasets/agent-llm-traces" \
  OUTPUT_FILE="$G/gate1v2_$T.jsonl" \
  SPLIT=eval \
  SPLIT_MANIFEST_FILE="$G/taskproxy_disjoint_v1.json" \
  SPLIT_NAME=taskproxy_disjoint \
  CONDITIONS=joint COMPARE_MODES=c2kv,full RATIOS=8 MAX_EXAMPLES=128 \
  ASCEND_RT_VISIBLE_DEVICES="$CHIP" \
  nohup bash agent/eval_joint_next_action_c2kv_npu.sh > "$G/gate1v2_$T.log" 2>&1 &
  EVALPID[$T]=$!
  IDX=$((IDX + 1))
  sleep 10
done
WAITED=0
for T in 5e-7 5e-6 5e-5; do
  while [[ ! -f "$G/gate1v2_$T.summary.json" ]]; do
    if (( WAITED > 10800 )); then fail "stage 1 timeout waiting gate1v2_$T.summary.json"; fi
    if ! kill -0 "${EVALPID[$T]}" 2>/dev/null && [[ ! -f "$G/gate1v2_$T.summary.json" ]]; then
      fail "gate1 eval $T died (see gate1v2_$T.log)"
    fi
    sleep 120; WAITED=$((WAITED + 120))
  done
done
log "stage 1 done: gate-1 summaries present"

# ---------------------------------------------------------------- stage 2
log "stage 2: LR pick"
if ! "$PY" scripts/g_joint_gate1_pick.py --gate_dir "$G" --stem gate1v2 --out "$G/gate1v2_picked_lr.txt" >> "$STATUS" 2>&1; then
  fail "stage 2: LR pick guard tripped (see gate1v2_picked_lr.json)"
fi
PICKED_LR=$(tr -d '[:space:]' < "$G/gate1v2_picked_lr.txt")
log "stage 2 done: picked LR=$PICKED_LR"

# ---------------------------------------------------------------- stage 3
log "stage 3: v2 frozen order file"
ORDER="$G/train_order_v2.json"
nohup "$PY" - > "$G/orderfile_v2.log" 2>&1 <<EOF &
from train.train_data_joint import AgentLLMTracesJointSource
import json
src = AgentLLMTracesJointSource(
    path="$HOME/c2kv/datasets/agent-llm-traces-v2",
    split="train",
    split_manifest_file="$G/taskproxy_disjoint_v2.json",
    split_manifest_name="taskproxy_disjoint",
    max_samples_per_session=4,
    require_tool_call=True,
)
qids = [ex.qid for ex in src]
with open("$ORDER", "w", encoding="utf-8") as f:
    json.dump(qids, f)
print("wrote", len(qids), "qids")
EOF
ORDERPID=$!
WAITED=0
while [[ ! -f "$ORDER" ]]; do
  if (( WAITED > 7200 )); then fail "stage 3 timeout waiting order file"; fi
  if ! kill -0 $ORDERPID 2>/dev/null && [[ ! -f "$ORDER" ]]; then
    fail "stage 3 order build died (see orderfile_v2.log)"
  fi
  sleep 60; WAITED=$((WAITED + 60))
done
NQ=$("$PY" -c "import json;print(len(json.load(open('$ORDER'))))")
log "stage 3 done: $NQ qids in order file"
(( NQ >= 4000 )) || fail "stage 3: only $NQ examples, small budget infeasible"

# ---------------------------------------------------------------- stage 4
BUDGET=32000000
if [[ -f "$G/official_tokens.json" ]]; then
  P_OFF=$("$PY" -c "import json;d=json.load(open('$G/official_tokens.json'));print(int(d.get('total',{}).get('P_src',0)))" 2>/dev/null || echo 0)
  if [[ "$P_OFF" -gt 0 ]]; then
    BUDGET=$(( P_OFF / 16 ))
    log "P_official=$P_OFF -> small budget=$BUDGET"
  fi
else
  log "official_tokens.json absent -> default small budget=32000000"
fi
log "stage 4: launching small arms (LR=$PICKED_LR, budget=$BUDGET)"

launch_arm() {  # $1 name, $2 doc_mode, $3 preferred chip
  local name="$1" mode="$2" pref="$3" chip
  chip=$(wait_chip "$pref" 46000 10800) || fail "stage 4: no chip for arm $name"
  log "arm $name ($mode) on chip $chip"
  MODEL_PATH="$HOME/c2kv/models/Qwen3-4B-Instruct-2507" \
  DATASET_PATH="$HOME/c2kv/datasets/agent-llm-traces-v2" \
  SPLIT_MANIFEST_FILE="$G/taskproxy_disjoint_v2.json" \
  SPLIT_NAME=taskproxy_disjoint \
  EXAMPLE_ORDER_FILE="$ORDER" \
  USE_DEEPSPEED=0 MAX_SOURCE_TOKENS="$BUDGET" MAX_EVAL_EXAMPLES=64 NUM_TRAIN_EPOCHS=1 \
  LR="$PICKED_LR" WARMUP_STEPS=20 GRAD_ACCUM=4 SAVE_STEPS=100000 LOGGING_STEPS=20 \
  DOC_MODE="$mode" \
  OUTPUT_DIR="$G/small_$name" \
  ASCEND_RT_VISIBLE_DEVICES="$chip" \
  nohup bash agent/train_joint_next_action_c2kv_npu.sh > "$G/small_$name.log" 2>&1 &
  echo "small_$name pid $! chip $chip mode $mode" >> "$STATUS"
  sleep 20
}

launch_arm joint joint 0
launch_arm alternate alternate 1
launch_arm sep_tool tool_only 2
launch_arm sep_hist history_only 3

log "autorun complete: 4 small arms launched (LR=$PICKED_LR, budget=$BUDGET)"
