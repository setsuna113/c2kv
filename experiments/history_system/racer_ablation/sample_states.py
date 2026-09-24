"""Freeze one task-balanced, prefix-valid decision state per held-out task from M2.

Rule (fixed before any probe run): for each held-out task, the sampling frame is
every M2 decision with a completed draft whose turn has previous_turn_valid is True
(turn 0 always; later turns need an officially valid prefix). One state is drawn
uniformly with a seed fixed per task. Nothing about the current draft, the current
turn's outcome, the risk score or recoverability is read. Tasks without a frame are
kept in the coverage table; nothing is redrawn later.

usage: python -m racer_ablation.sample_states BFCL_DIR OUT_DIR M2_ROOT [M2_ROOT ...]
"""
from __future__ import annotations

import hashlib
import sys

from .common import CELLS, draft_tokens, raw_result, steps, task_rows, turn_of, write_json
from .heldout import heldout_tasks
from .turn_scorer import TurnScorer

SEED = "racer-abl-20260924"


def terminated_turn(records):
    failed = [record for record in records if record.get("status") != "ok"]
    return turn_of(failed[-1]["decision_key"]) if failed else None


def sample(scorer, rows, tasks):
    states, targets = {}, {}
    for task in tasks:
        row = rows.get(task)
        if row is None:
            states[task] = {"status": "no_m2_record"}
            continue
        records = steps(row)
        result = raw_result(row)
        stop = terminated_turn(records)
        labels = scorer.turn_outcomes(task, result or [], terminated_turn=stop)
        frame = [record["decision_key"] for record in records
                 if draft_tokens(record) is not None
                 and scorer.previous_turn_valid(labels, turn_of(record["decision_key"])) is True]
        entry = {"decisions": len(records), "frame_size": len(frame), "turn_labels": labels,
                 "terminated_turn": stop}
        if not frame:
            entry["status"] = "no_prefix_valid_draft_state"
        else:
            index = int(hashlib.sha256(f"{SEED}|{task}".encode()).hexdigest(), 16) % len(frame)
            entry.update(status="sampled", decision_key=frame[index], frame_index=index)
            targets[task] = frame[index]
        states[task] = entry
    return states, targets


def main(argv):
    bfcl_dir, out_dir, roots = argv[0], argv[1], argv[2:]
    rows, failures = task_rows(roots, CELLS["M2"])
    tasks = [task for task in heldout_tasks() if task in rows]
    states, targets = sample(TurnScorer(bfcl_dir), rows, tasks)
    write_json(f"{out_dir}/probe_states.json", {
        "schema": "racer-ablation-probe-states-v1", "seed": SEED, "m2_roots": roots,
        "heldout_tasks_available": len(tasks), "states": states,
        "m2_method_failures": sorted(failures["method"]),
        "m2_harness_failures": sorted(failures["harness"])})
    write_json(f"{out_dir}/probe_targets.json", {
        "schema": "racer-probe-targets-v1", "seed": SEED, "targets": targets})
    print(f"held-out tasks with M2 records {len(tasks)}; sampled {len(targets)}")


if __name__ == "__main__":
    main(sys.argv[1:])
