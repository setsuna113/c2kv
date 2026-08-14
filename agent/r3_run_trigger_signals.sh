#!/usr/bin/env bash
# R3 T-D: trigger-signal pipeline rerun on the same frozen inputs as round 2
# (outputs_lyc/r2_logp/s4_logp_clean.jsonl + merged_{A,B,C,D}.jsonl), with the
# explicit positive-orientation logp_prefix_full diagnostic added.
# Pure CPU. Run from the task/r3-discrimination worktree root.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-$HOME/envs/c2kv/bin/python}"
LYC="${LYC:-$HOME/c2kv}"
OUT_DIR="${OUT_DIR:-$LYC/outputs_lyc/r3_discrimination/t_d}"
mkdir -p "$OUT_DIR"

"$PY" agent/analyze_trigger_signals.py \
  --logp_jsonl "$LYC/outputs_lyc/r2_logp/s4_logp_clean.jsonl" \
  --arm_a "$LYC/outputs_lyc/merged_A.jsonl" \
  --arm_b "$LYC/outputs_lyc/merged_B.jsonl" \
  --arm_c "$LYC/outputs_lyc/merged_C.jsonl" \
  --arm_d "$LYC/outputs_lyc/merged_D.jsonl" \
  --out_prefix "$OUT_DIR/trigger_signals" \
  --bootstrap_reps 20000 --bootstrap_seed 0 \
  > "$OUT_DIR/run.log" 2>&1
echo "[t_d] done -> $OUT_DIR/trigger_signals.{json,md}"
