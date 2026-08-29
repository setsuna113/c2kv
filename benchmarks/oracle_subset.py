"""Task-level oracle: the tasks a reference arm solved and a target arm did not.

Block 2's repair arm is triggered by an oracle, not by a detector: run the
`full` arm, run the compressed arm, and re-run only the tasks where full
succeeded and the compressed arm failed.  That join needs per-task rows from
both arms, which `run.py` now writes as `rows_<arm>.jsonl`.

    python benchmarks/oracle_subset.py \
        --reference results/bench/tau2_full/rows_full.jsonl \
        --target    results/bench/tau2_hybrid/rows_hybrid.jsonl \
        --out       results/bench/tau2_oracle_hybrid.txt

The id list it writes is what `run.py --task-ids @<file>` consumes.

Success is `semantic_score >= --success-threshold` (default 1.0, i.e. the
benchmark's own pass mark for a 0/1 scorer; tau2's reward is already 0/1 per
task, BFCL's per-entry score is 0/1, ToolSandbox's similarity is continuous so
a threshold must be chosen deliberately and is recorded in the manifest).

Tasks missing from either arm are never silently dropped: they are counted and
listed under `unpaired`, because an arm that crashed on a task is not the same
as an arm that failed it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional


def load_rows(path: Path) -> Dict[str, dict]:
    if not path.exists():
        raise SystemExit(f"FATAL: rows file not found: {path}")
    rows: Dict[str, dict] = {}
    duplicates: List[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        task_id = str(row.get("task_id"))
        if task_id in rows:
            duplicates.append(task_id)
        rows[task_id] = row
    if not rows:
        raise SystemExit(f"FATAL: {path} has no rows")
    if duplicates:
        raise SystemExit(
            f"FATAL: {path} has {len(duplicates)} duplicate task_id(s), e.g. "
            f"{duplicates[:5]} — the oracle join would be ambiguous."
        )
    return rows


def succeeded(row: dict, threshold: float) -> Optional[bool]:
    """True/False, or None when the task has no usable score."""
    score = row.get("semantic_score")
    if score is None:
        return None
    try:
        return float(score) >= threshold
    except (TypeError, ValueError):
        return None


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference", type=Path, required=True,
                        help="rows_<arm>.jsonl of the arm that should succeed (normally full)")
    parser.add_argument("--target", type=Path, required=True,
                        help="rows_<arm>.jsonl of the compressed arm being repaired")
    parser.add_argument("--out", type=Path, required=True,
                        help="destination for the task-id list (one per line)")
    parser.add_argument("--manifest", type=Path,
                        help="optional JSON manifest describing the join (defaults to <out>.json)")
    parser.add_argument("--success-threshold", type=float, default=1.0)
    args = parser.parse_args(argv)

    reference = load_rows(args.reference)
    target = load_rows(args.target)

    trigger, unpaired, unscored = [], [], []
    for task_id in sorted(set(reference) | set(target)):
        if task_id not in reference or task_id not in target:
            unpaired.append(task_id)
            continue
        ref_ok = succeeded(reference[task_id], args.success_threshold)
        tgt_ok = succeeded(target[task_id], args.success_threshold)
        if ref_ok is None or tgt_ok is None:
            unscored.append(task_id)
            continue
        if ref_ok and not tgt_ok:
            trigger.append(task_id)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(trigger) + ("\n" if trigger else ""), encoding="utf-8")

    n_paired = len(set(reference) & set(target))
    manifest = {
        "reference_file": str(args.reference),
        "target_file": str(args.target),
        "success_threshold": args.success_threshold,
        "n_reference": len(reference),
        "n_target": len(target),
        "n_paired": n_paired,
        "n_trigger": len(trigger),
        # L1 in the battery's language: how often the trigger fires at all.
        # It is a property of the arm pair, not of any repair arm.
        "trigger_rate_L1": (len(trigger) / n_paired) if n_paired else None,
        "n_unpaired": len(unpaired),
        "unpaired": unpaired[:50],
        "n_unscored": len(unscored),
        "unscored": unscored[:50],
        "trigger_task_ids": trigger,
    }
    manifest_path = args.manifest or args.out.with_suffix(args.out.suffix + ".json")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False),
                             encoding="utf-8")

    print(json.dumps({k: v for k, v in manifest.items()
                      if k not in ("trigger_task_ids", "unpaired", "unscored")},
                     indent=2))
    if unpaired:
        print(f"WARNING: {len(unpaired)} task(s) present in only one arm — see {manifest_path}")
    if unscored:
        print(f"WARNING: {len(unscored)} task(s) had no usable semantic_score — see {manifest_path}")


if __name__ == "__main__":
    main()
