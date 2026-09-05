#!/usr/bin/env bash
set -euo pipefail

# t33 capture rerun (survey item 4.0-2): the frozen d_r2 battery, both arms,
# with capture instrumentation.  Env block is VERBATIM the r2 battery log
# (results/bdf_pilot/logs/battery_full.log); the only additions are the
# --capture_out flags via EXTRA_ARGS.
#
# Usage:
#   SMOKE=1 bash agent/t33_capture_npu.sh          # 3 rows/arm, smoke dirs
#   bash agent/t33_capture_npu.sh                  # full 900 rows/arm
# Chips come from T33_CHIP_FULL / T33_CHIP_C2KV (defaults 1 / 2).  NEVER 5/6/7.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# Bare setsid/cron envs: set_env.sh references vars that are unbound under
# set -u, and `python` must resolve to the c2kv env (same guards as
# run_d_pilot_npu.sh).
export PATH="$HOME/envs/c2kv/bin:$PATH"
if [[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]]; then
  set +u
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  set -u
fi

SMOKE="${SMOKE:-0}"
CHIP_FULL="${T33_CHIP_FULL:-1}"
CHIP_C2KV="${T33_CHIP_C2KV:-2}"
OUT_ROOT="${T33_OUT_ROOT:-/home/liuyancheng/c2kv/outputs_lyc/t33}"
MAX_EXAMPLES=900
SUFFIX=""

if [[ "${SMOKE}" == "1" ]]; then
  OUT_ROOT="${OUT_ROOT}/smoke"
  MAX_EXAMPLES=3
  SUFFIX=".smoke"
fi

export MODEL_PATH=/home/liuyancheng/c2kv/outputs_lyc/g_joint/fixed_joint
export BASE_MODEL=/home/liuyancheng/c2kv/models/Qwen3-4B-Instruct-2507
export TOKENIZER_PATH=/home/liuyancheng/c2kv/models/Qwen3-4B-Instruct-2507
export DATASET_PATH=/home/liuyancheng/c2kv/datasets/agent-llm-traces-v2
export SPLIT=eval
export MAX_EXAMPLES="${MAX_EXAMPLES}"
export MAX_DOC_LENGTH=768
export MAX_DOC_NUM=16
export MAX_HISTORY_TOKENS=12288
export MAX_PROMPT_TOKENS=1536
export MAX_BASELINE_INPUT_TOKENS=16000
export HISTORY_SELECTION=tail
export TRUNCATE_SELECTION=tail
export SPLIT_OVERSIZED_HISTORY_DOCS=True
export INCLUDE_TOOLS=True
export PARALLEL_EVAL=True

CAPTURE_OUT="${OUT_ROOT}/capture"
mkdir -p "${OUT_ROOT}" "${CAPTURE_OUT}"

run_arm() {
  local mode="$1" ratio="$2" chip="$3" part="$4"
  local out="${OUT_ROOT}/battery_${mode}${SUFFIX}.jsonl"
  echo "[t33] arm=${mode} chip=${chip} out=${out} capture=${CAPTURE_OUT}/${mode}"
  (
    export ASCEND_RT_VISIBLE_DEVICES="${chip}"
    export COMPARE_MODES="${mode}"
    export RATIOS="${ratio}"
    export OUTPUT_FILE="${out}"
    export TMP_DIR="${out%.jsonl}.parts"
    export EXTRA_ARGS="--capture_out ${CAPTURE_OUT} --capture_part ${part}"
    bash agent/eval_agent_history_c2kv_npu.sh
  ) > "${OUT_ROOT}/battery_${mode}${SUFFIX}.log" 2>&1
}

run_arm full 1 "${CHIP_FULL}" p0 &
PID_FULL=$!
run_arm c2kv 8 "${CHIP_C2KV}" p0 &
PID_C2KV=$!

FAIL=0
wait "${PID_FULL}" || FAIL=1
wait "${PID_C2KV}" || FAIL=1

echo "[t33] arms done (fail=${FAIL}); capture errors summary:"
for mode in full c2kv; do
  python - "$OUT_ROOT" "$mode" "$SUFFIX" <<'PYEOF'
import json, sys
out_root, mode, suffix = sys.argv[1], sys.argv[2], sys.argv[3]
try:
    summary = json.load(open(f"{out_root}/battery_{mode}{suffix}.summary.json"))
    print(mode, "rows:", summary.get("num_rows"), "capture_errors:", summary.get("t33_capture_errors"))
except FileNotFoundError:
    print(mode, "summary missing")
PYEOF
done
exit "${FAIL}"
