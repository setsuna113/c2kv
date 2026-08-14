#!/usr/bin/env bash
# R3 T-C: D1' readout re-analysis — 5 bootstrap seeds x 20000 reps (pure CPU),
# plus the offline condition-length replay + length-confound regression.
# Run from the task/r3-discrimination worktree root.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-$HOME/envs/c2kv/bin/python}"
LYC="${LYC:-$HOME/c2kv}"
READOUT="${READOUT:-$LYC/outputs_lyc/r2_d1prime/d1prime_condition_readout.jsonl}"
OUT_DIR="${OUT_DIR:-$LYC/outputs_lyc/r3_discrimination/t_c}"
TOKENIZER="${TOKENIZER:-$LYC/checkpoints/qwen3-4b-agent-history-c2kv-npu/checkpoint-2678}"
mkdir -p "$OUT_DIR"

for seed in 0 1 2 3 4; do
  "$PY" agent/analyze_condition_readout.py --readout "$READOUT" --reps 20000 --seed "$seed" \
    --out "$OUT_DIR/readout_seed${seed}.json" > "$OUT_DIR/readout_seed${seed}.log" 2>&1
  echo "[t_c] seed=$seed done"
done

"$PY" agent/r3_d1prime_length_replay.py --readout "$READOUT" \
  --tokenizer "$TOKENIZER" --out_prefix "$OUT_DIR/t_c_length" \
  > "$OUT_DIR/length_replay.log" 2>&1
echo "[t_c] length replay done"
