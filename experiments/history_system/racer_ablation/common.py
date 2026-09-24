"""Locate shard cells and read their per-task and per-decision records."""
from __future__ import annotations

import glob
import json
import os
from pathlib import Path

BENCHMARK = "bfcl_long_context"
CATEGORY = "multi_turn_long_context"
CELLS = {
    "M0": "racer_v2_c2kv_c1_v2_verified_b256",
    "M1": "racer_v2_c2kv_c1_v2_core_b256",
    "M2": "racer_v2_c2kv_c1_v2_verified_protected_off_b256",
    "M3": "racer_v2_c2kv_c1_v2_selfrev_b256",
    "M4": "racer_v2_c2kv_c1_v2_nodraftq_b256",
    "P": "racer_v2_c2kv_c1_v2_probe_b256",
}
ARCHIVED = (".failed.", ".stopped.", ".superseded.", ".infra.", ".portclash.", ".killed.",
            ".dropped.", ".paused.", ".void.")


def cell_dirs(roots, cell):
    """Live (non-archived) directories of one cell below any of the roots."""
    found = set()
    for root in roots:
        found.update(glob.glob(os.path.join(root, "**", "closed_loop", f"{BENCHMARK}__{cell}"),
                               recursive=True))
    return sorted(Path(path) for path in found if not any(tag in path for tag in ARCHIVED))


def task_rows(roots, cell):
    """task_id -> summary row; every task may appear in exactly one shard."""
    rows, failures = {}, {"harness": set(), "method": set()}
    for directory in cell_dirs(roots, cell):
        summary = directory / f"summary_{cell}.json"
        if not summary.is_file():
            continue
        document = json.loads(summary.read_text(encoding="utf-8"))
        failures["harness"].update(document.get("harness_failure_task_ids") or ())
        failures["method"].update(document.get("method_failure_task_ids") or ())
        for row in document["task_rows"]:
            if row["task_id"] in rows:
                raise ValueError(f"{cell}: task {row['task_id']} appears in more than one shard")
            rows[row["task_id"]] = dict(row, _cell_dir=str(directory))
    return rows, failures


def shard_dir(row):
    return Path(row["_cell_dir"]) / "native" / "task_shards" / row["task_id"]


def steps(row):
    path = shard_dir(row) / "server" / "steps.jsonl"
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def raw_result(row):
    """The official BFCL raw model result (turn -> step -> response) for one task."""
    pattern = str(shard_dir(row) / "bfcl" / "bfcl" / "result" / "*" / "multi_turn"
                  / f"BFCL_v4_{CATEGORY}_result.json")
    for path in glob.glob(pattern):
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                entry = json.loads(line)
                if entry.get("id") == row["task_id"]:
                    return entry["result"]
    return None


def turn_of(decision_key):
    turn, _ = decision_key.split("/", 1)
    return int(turn.removeprefix("turn-"))


def draft_trace(record):
    trace = record.get("generation_trace") or []
    return trace[0] if trace and trace[0].get("phase") == "draft" else None


def draft_tokens(record):
    trace = draft_trace(record)
    if trace is None or trace.get("status") != "completed":
        return None
    return list(trace["generation"]["token_ids"])


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1, sort_keys=True), encoding="utf-8")
