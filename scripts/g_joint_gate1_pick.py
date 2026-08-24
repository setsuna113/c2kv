"""Gate-1 LR selection for the g-joint experiment (rule-based, auditable).

Reads the three Gate-1 eval summaries (lrcal2_{5e-7,5e-6,5e-5}) and picks the
LR for the small-budget arms:

- primary metric: c2kv joint @ ratio 8 ``tool_name_accuracy`` (fallback key
  ``tool_name_match``); tie-break: ``exact_match``; final tie: prefer 5e-6.
- GUARDS (any failure -> exit 3, no arms may launch):
  * every candidate's c2kv tool_name_accuracy < --min_tool_name_accuracy
    (default 0.05): the recipe is broken at every LR; do not burn 4x33h;
  * any candidate's ``gist_init_fractions`` mean > 0.5: gist weights silently
    re-initialised at load — investigate before training more.

Writes the picked LR to --out (plain text, e.g. "5e-6") plus a JSON rationale.
Usage:
  python g_joint_gate1_pick.py --gate_dir ~/c2kv/outputs_lyc/g_joint \
      --out ~/c2kv/outputs_lyc/g_joint/gate1v2_picked_lr.txt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

LRS = ["5e-7", "5e-6", "5e-5"]
PREFER_ON_TIE = "5e-6"


def _metric(group: Dict[str, Any], *names: str) -> Optional[float]:
    for name in names:
        value = group.get(name)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def load_c2kv_result(summary_path: Path) -> Dict[str, Any]:
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    results = data.get("results") or []
    for group in results:
        if (
            group.get("condition") == "joint"
            and group.get("mode") == "c2kv"
            and str(group.get("ratio")) == "8"
        ):
            return {"summary": data, "group": group}
    raise RuntimeError(f"no joint/c2kv/ratio8 result group in {summary_path}")


def pick(gate_dir: Path, stem: str, min_tool_name_accuracy: float) -> Dict[str, Any]:
    table: List[Dict[str, Any]] = []
    for lr in LRS:
        summary_path = gate_dir / f"{stem}_{lr}.summary.json"
        if not summary_path.exists():
            raise RuntimeError(f"missing gate-1 summary: {summary_path}")
        loaded = load_c2kv_result(summary_path)
        group = loaded["group"]
        tool_name = _metric(group, "tool_name_accuracy", "tool_name_match") or 0.0
        exact = _metric(group, "exact_match") or 0.0
        init_fracs = (loaded["summary"].get("gist_init_fractions") or {})
        init_mean = None
        if isinstance(init_fracs, dict) and init_fracs:
            values = [float(v) for v in init_fracs.values() if isinstance(v, (int, float))]
            init_mean = sum(values) / len(values) if values else None
        table.append(
            {
                "lr": lr,
                "tool_name_accuracy": tool_name,
                "exact_match": exact,
                "gist_init_fraction_mean": init_mean,
                "num_valid": group.get("num_valid"),
            }
        )

    failures = []
    for row in table:
        if row["gist_init_fraction_mean"] is not None and row["gist_init_fraction_mean"] > 0.5:
            failures.append(f"{row['lr']}: gist_init_fraction_mean={row['gist_init_fraction_mean']:.3f} > 0.5")
    if all(row["tool_name_accuracy"] < min_tool_name_accuracy for row in table):
        failures.append(
            f"all c2kv tool_name_accuracy < {min_tool_name_accuracy}: "
            + ", ".join(f"{r['lr']}={r['tool_name_accuracy']:.4f}" for r in table)
        )

    best = sorted(
        table,
        key=lambda row: (
            row["tool_name_accuracy"],
            row["exact_match"],
            row["lr"] == PREFER_ON_TIE,
        ),
        reverse=True,
    )[0]
    return {
        "picked_lr": None if failures else best["lr"],
        "table": table,
        "guard_failures": failures,
        "rule": "max(tool_name_accuracy), tie max(exact_match), tie prefer 5e-6",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate_dir", required=True)
    parser.add_argument("--stem", default="gate1v2")
    parser.add_argument("--min_tool_name_accuracy", type=float, default=0.05)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    result = pick(Path(args.gate_dir), args.stem, args.min_tool_name_accuracy)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text((result["picked_lr"] or "NONE") + "\n", encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if result["picked_lr"] is None:
        print("GUARD FAILURE: no LR picked", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
