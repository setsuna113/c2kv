#!/usr/bin/env bash
# Priority eval queue for one arm, sharing the training GPU.
#   job = (step, ratio). Candidate steps fixed 2026-09-11 before any score: 500 750 1000 1098.
#   priority: smoke(250, r8, 8 tasks) once -> any arrived step lacking ratio-8 (lowest step first)
#             -> any arrived step lacking ratio-4 -> wait. Ratio 8 is the deliverable; 4 is secondary.
#   A job whose output dir exists is never re-run automatically (in progress or failed: inspect by hand).
set -uo pipefail
ARM="$1"
CK="/workspace/checkpoints/b_history/arm-${ARM}/seed-42"
EV=/workspace/bc-eval
PY=/workspace/.venv-bc-eval/bin/python
# Shares the GPU with training (training holds ~46 GB reserved). Make the eval server's
# caching allocator return freed blocks earlier; allocator settings do not change numerics.
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,garbage_collection_threshold:0.7"
STEPS=(500 750 1000)  # 1098 shipped unevaluated (user decision 2026-09-11)
stamp() { date -u +%FT%TZ; }
arrived() { [[ -d "${CK}/checkpoint-$1" && ! -d "${CK}/checkpoint-$1.pending" ]]; }
run_one() {
  local step="$1" ratio="$2" tasks="$3" tag="$4" wall="$5"
  local out="${EV}/${tag}-arm${ARM}-step${step}-r${ratio}"
  local manifest="${EV}/manifest-${tag}-arm${ARM}-step${step}-r${ratio}.json"
  echo "$(stamp) building ${manifest}"
  $PY - "$ARM" "$step" "$ratio" "$tasks" "$out" "$manifest" "$wall" <<'PYEOF'
import json, sys
arm, step, ratio, tasks, out, manifest, wall = sys.argv[1:8]
wall = int(wall)
cfg = {
  "schema": "history-memory-native-bfcl-eval-v1", "split": "dev",
  "runtime_root": "/workspace/c2kv-a-runtime",
  "bfcl_benchmark_dir": "/workspace/gorilla/berkeley-function-call-leaderboard",
  "task_manifest": tasks,
  "eval_policy": "/workspace/c2kv-a-runtime/benchmarks/memory_runtime/configs/a_always_compress_v1.eval-policy.json",
  "output_dir": out, "view_mode": "ac_exact_persistent", "device": "cuda:0", "dtype": "bfloat16",
  "ratio": int(ratio), "max_new_tokens": 4096, "max_decisions": 6000, "max_generation_calls": 12000,
  "server_max_wall_seconds": wall, "bfcl_max_wall_seconds": wall - 1800, "ready_timeout_seconds": 1200,
  "torch_threads": 4,
  "candidates": [{"arm": arm, "checkpoint": f"/workspace/checkpoints/b_history/arm-{arm}/seed-42/checkpoint-{step}"}],
}
json.dump(cfg, open(manifest, "w"), indent=2)
PYEOF
  cd /workspace/c2kv-b-history
  if ! $PY agent/eval_history_memory.py --manifest "$manifest" --dry-run > "${manifest%.json}.dryrun.json" 2> "${manifest%.json}.dryrun.err"; then
    echo "$(stamp) DRY-RUN FAILED ${tag} step ${step} r${ratio}; see ${manifest%.json}.dryrun.err"
    mkdir -p "$out"; echo "dry-run failed" > "$out/QUEUE_FAILED"; return 1
  fi
  echo "$(stamp) START ${tag} arm ${ARM} step ${step} r${ratio} -> ${out}"
  local t0=$(date +%s)
  $PY agent/eval_history_memory.py --manifest "$manifest" > "${manifest%.json}.run.log" 2>&1
  local rc=$?
  echo "$(stamp) END ${tag} arm ${ARM} step ${step} r${ratio} rc=${rc} wall=$(( $(date +%s) - t0 ))s"
  [[ -f "$out/selection.json" ]] && $PY -c "import json,sys; s=json.load(open(sys.argv[1])); w=s.get('winners') or s; print('  result:', json.dumps(w)[:300])" "$out/selection.json"
  return $rc
}
# smoke: pipeline + per-task timing on checkpoint-250 (8 tasks, ratio 8). Not a candidate.
until arrived 250; do sleep 60; done
[[ -d "${EV}/smoke-arm${ARM}-step250-r8" ]] || run_one 250 8 "${EV}/smoke-tasks.json" smoke 7200
while true; do
  job=""
  for ratio in 8 4; do
    for step in "${STEPS[@]}"; do
      if arrived "$step" && [[ ! -d "${EV}/dev-arm${ARM}-step${step}-r${ratio}" ]]; then job="$step $ratio"; break 2; fi
    done
  done
  if [[ -z "$job" ]]; then
    all_done=1
    for ratio in 8 4; do for step in "${STEPS[@]}"; do [[ -d "${EV}/dev-arm${ARM}-step${step}-r${ratio}" ]] || all_done=0; done; done
    (( all_done )) && break
    sleep 120; continue
  fi
  set -- $job
  run_one "$1" "$2" "${EV}/dev-tasks.json" dev 43200
done
echo "$(stamp) QUEUE DONE arm ${ARM}"
