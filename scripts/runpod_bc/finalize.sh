#!/usr/bin/env bash
# End-of-run finalizer for one arm (user decision 2026-09-11 21:05 UTC):
#   wait for training completed + dev evals {500,750,1000} x {r8,r4} complete
#   -> select_bc.py -> upload winner + checkpoint-1098 (model files only) + provenance to HF
#   -> verify listing -> write FINALIZED.json (the patrol then terminates the pod).
set -uo pipefail
ARM="$1"
CK="/workspace/checkpoints/b_history/arm-${ARM}/seed-42"
EV=/workspace/bc-eval
PY=/workspace/.venv-bc-eval/bin/python
HF=/workspace/c2kv-b-history/.venv-b-history/bin/hf
REPO="Jasonning/c2kv"; PREFIX="b_history/arm-${ARM}/seed-42"
stamp() { date -u +%FT%TZ; }
training_done() { [[ -f "$CK/latest.json" ]] && python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); sys.exit(0 if d["completed"] and d["checkpoint"]=="checkpoint-1098" else 1)' "$CK/latest.json" && [[ -d "$CK/checkpoint-1098" && ! -d "$CK/checkpoint-1098.pending" ]]; }
evals_done() { for s in 500 750 1000; do for r in 8 4; do [[ -f "$EV/dev-arm${ARM}-step${s}-r${r}/arm-${ARM}-step-${s}/bfcl/official_summary.json" ]] || return 1; done; done; }
echo "$(stamp) finalize arm ${ARM}: waiting for training completed + 6 dev evals"
until training_done && evals_done; do sleep 120; done
echo "$(stamp) training completed and 6 dev evals present; selecting"
$PY "$EV/select_bc.py" "$ARM" | tee "$EV/select_arm${ARM}.out"
WINNER=$(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); w=d["combined_winner"]; assert d["complete"], "selection incomplete"; print(w["step"])' "$EV/selection_combined_arm${ARM}.json") || { echo "$(stamp) SELECTION INCOMPLETE; not uploading"; exit 1; }
echo "$(stamp) winner step ${WINNER}; uploading winner + 1098 (model files only)"
for step in "$WINNER" 1098; do
  "$HF" upload "$REPO" "$CK/checkpoint-${step}" "$PREFIX/checkpoint-${step}" --repo-type model \
     --exclude "optimizer.pt" "rng-rank-*.pt" --commit-message "B/C history training arm ${ARM} seed 42: checkpoint-${step} ($( [[ $step == 1098 ]] && echo final-unevaluated || echo dev-selected-winner )), model files only" \
     || { echo "$(stamp) UPLOAD FAILED checkpoint-${step}"; exit 1; }
done
mkdir -p "$EV/provenance-arm${ARM}"
cp "$CK/train_history.jsonl" "$CK/latest.json" "$EV/selection_combined_arm${ARM}.json" "$EV/dev-tasks.json" "$EV/provenance-arm${ARM}/"
for s in 500 750 1000; do for r in 8 4; do d="$EV/dev-arm${ARM}-step${s}-r${r}"; mkdir -p "$EV/provenance-arm${ARM}/dev-step${s}-r${r}"; cp "$d/run.json" "$d/selection.json" "$d/arm-${ARM}-step-${s}/candidate.json" "$d/arm-${ARM}-step-${s}/bfcl/official_summary.json" "$d/arm-${ARM}-step-${s}/bfcl/final.json" "$EV/provenance-arm${ARM}/dev-step${s}-r${r}/" 2>/dev/null; cp -r "$d/arm-${ARM}-step-${s}/bfcl/bfcl/score" "$EV/provenance-arm${ARM}/dev-step${s}-r${r}/" 2>/dev/null; done; done
cp "$EV/manifest-dev-arm${ARM}-"*.json "$EV/eval_queue_arm${ARM}.log" "$EV/provenance-arm${ARM}/" 2>/dev/null
"$HF" upload "$REPO" "$EV/provenance-arm${ARM}" "$PREFIX/provenance" --repo-type model --commit-message "B/C history training arm ${ARM} seed 42: selection provenance (train_history, dev-128 official summaries, selection)" || echo "$(stamp) provenance upload failed (non-fatal)"
echo "$(stamp) verifying HF listing"
$PY - "$ARM" "$WINNER" <<'PYEOF' | tee "$EV/hf_verify_arm${ARM}.json"
import json, os, sys
from huggingface_hub import HfApi
arm, winner = sys.argv[1], sys.argv[2]
api = HfApi(); repo = "Jasonning/c2kv"; prefix = f"b_history/arm-{arm}/seed-42"
ok, report = True, {}
for step in (winner, "1098"):
    local = {f: os.path.getsize(f"/workspace/checkpoints/b_history/arm-{arm}/seed-42/checkpoint-{step}/{f}")
             for f in os.listdir(f"/workspace/checkpoints/b_history/arm-{arm}/seed-42/checkpoint-{step}")
             if f not in ("optimizer.pt",) and not f.startswith("rng-rank-") and not f.startswith(".")}
    remote = {t.path.split("/")[-1]: t.size for t in api.list_repo_tree(repo, path_in_repo=f"{prefix}/checkpoint-{step}", recursive=True) if hasattr(t, "size")}
    same = local == remote
    ok &= same
    report[f"checkpoint-{step}"] = {"match": same, "local_files": len(local), "remote_files": len(remote),
                                    "local_bytes": sum(local.values()), "remote_bytes": sum(remote.values()),
                                    "missing_remote": sorted(set(local) - set(remote)), "size_mismatch": sorted(k for k in local if k in remote and local[k] != remote[k])}
print(json.dumps({"arm": arm, "winner": int(winner), "final": 1098, "verified": ok, "checkpoints": report}, indent=2))
sys.exit(0 if ok else 2)
PYEOF
rc=${PIPESTATUS[0]}
if [[ $rc -eq 0 ]]; then
  python3 -c 'import json,sys,datetime; json.dump({"arm": sys.argv[1], "winner_step": int(sys.argv[2]), "final_step": 1098, "hf_verified": True, "finalized_utc": datetime.datetime.utcnow().isoformat()+"Z", "ok_to_terminate": True}, open(sys.argv[3],"w"), indent=2)' "$ARM" "$WINNER" "$EV/FINALIZED.json"
  echo "$(stamp) FINALIZED arm ${ARM}: winner ${WINNER} + 1098 on HF, verified. ok_to_terminate=true"
else
  echo "$(stamp) HF VERIFY FAILED (rc=$rc); NOT marking finalized"
fi
