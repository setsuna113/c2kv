#!/usr/bin/env python3
"""Combined dev selection for one arm across ratio 8 and ratio 4 (pre-registered 2026-09-11).

Rule (fixed before any score existed): per arm, score = (correct_r8 + correct_r4) / (2 * 128)
over the frozen dev-128 set; exact ties -> earlier step. Per-ratio winners are reported as
secondary evidence. Uses the runner's own read_official_summary / select_best; a candidate
missing either ratio, or with an incomplete official run, disqualifies that step (no imputation).
"""
import json, sys
from pathlib import Path

sys.path.insert(0, "/workspace/c2kv-b-history/python")
from history_memory.checkpoint_eval import read_official_summary, select_best  # noqa: E402

ARM = sys.argv[1]
STEPS = (500, 750, 1000)  # 1098 excluded from on-pod selection (user 2026-09-11 21:05 UTC: ship best-of-evaluated + final, no post-training eval tail); no 1098 score existed at that time
RATIOS = (8, 4)
EV = Path("/workspace/bc-eval")
N = 128

per_ratio = {r: [] for r in RATIOS}
combined, disqualified = [], {}
for step in STEPS:
    cells = {}
    for r in RATIOS:
        cdir = EV / f"dev-arm{ARM}-step{step}-r{r}" / f"arm-{ARM}-step-{step}"
        try:
            official = read_official_summary(cdir / "bfcl" / "official_summary.json", cdir / "bfcl" / "final.json", N)
        except Exception as error:  # incomplete / failed / missing
            disqualified.setdefault(step, {})[f"r{r}"] = f"{type(error).__name__}: {error}"
            continue
        cells[r] = official
        per_ratio[r].append({"arm": ARM, "step": step, "ratio": r, **official})
    if len(cells) == len(RATIOS):
        correct = sum(cells[r]["correct_count"] for r in RATIOS)
        combined.append({"arm": ARM, "step": step, "correct_count": correct, "n_scored": len(RATIOS) * N,
                         "selection_score": correct / (len(RATIOS) * N),
                         "per_ratio": {f"r{r}": cells[r] for r in RATIOS}})

result = {
    "schema": "history-memory-bc-combined-selection-v1",
    "arm": ARM, "steps": list(STEPS), "ratios": list(RATIOS), "dev_tasks": str(EV / "dev-tasks.json"), "n_per_ratio": N,
    "rule": "argmax (correct_r8 + correct_r4) / 256 over dev-128 among {500,750,1000}; ties -> earlier step; a step lacking a complete official run for either ratio is disqualified; checkpoint-1098 is shipped alongside unevaluated",
    "loss_used_for_selection": False,
    "combined_candidates": combined,
    "combined_winner": select_best(combined)[ARM] if combined else None,
    "per_ratio_winner": {f"r{r}": (select_best(per_ratio[r])[ARM] if per_ratio[r] else None) for r in RATIOS},
    "disqualified": disqualified,
    "complete": not disqualified and len(combined) == len(STEPS),
}
out = EV / f"selection_combined_arm{ARM}.json"
out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
print(json.dumps({k: result[k] for k in ("combined_winner", "per_ratio_winner", "disqualified", "complete")}, indent=2)[:2000])
print("written", out)
