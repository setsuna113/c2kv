"""Explicit repair cohorts; their scores are never whole-cell scores."""
from __future__ import annotations

import json
from pathlib import Path

from .artifact_io import atomic_json


def validate_subsets(subsets):
    if not isinstance(subsets, dict) or not subsets:
        raise ValueError("task_subsets must be a non-empty CELL -> task IDs mapping")
    for cell_id, ids in subsets.items():
        if not isinstance(cell_id, str) or not cell_id.strip():
            raise ValueError("Task subset cell IDs must be non-empty strings")
        if (not isinstance(ids, list) or not ids
                or any(not isinstance(task, str) or not task.strip()
                       or task != task.strip() or "," in task for task in ids)):
            raise ValueError(f"{cell_id}: task IDs must be a non-empty list of non-empty strings")
        if len(ids) != len(set(ids)):
            raise ValueError(f"{cell_id}: duplicate task IDs")
    return subsets


def with_task_subsets(config, values=(), path=None):
    if not values and path is None:
        return config
    if "task_subsets" in config:
        raise ValueError("Task subsets are already frozen in this config")
    subsets = validate_subsets(json.loads(path.read_text(encoding="utf-8"))) if path else {}
    for value in values:
        cell_id, separator, ids = value.partition("=")
        if not separator:
            raise ValueError("--task-subset requires CELL=id,...")
        if cell_id in subsets:
            raise ValueError(f"Duplicate task subset cell: {cell_id}")
        subsets[cell_id] = ids.split(",")
    return dict(config, task_subsets=validate_subsets(subsets))


def select_subset_cells(rows, subsets):
    from benchmarks.arms import get_arm

    validate_subsets(subsets)
    unknown = set(subsets) - {row["cell_id"] for row in rows}
    if unknown:
        raise ValueError(f"Unknown task subset cells: {sorted(unknown)}")
    selected = []
    for row in rows:
        if row["cell_id"] not in subsets:
            continue
        if row["adapter"] not in {"bfcl", "acon_appworld"} or get_arm(row["arm"]).native_controller:
            raise ValueError("Task subsets support non-native BFCL/AppWorld cells only")
        selected.append(dict(row, **subset_metadata(row["cell_id"], subsets[row["cell_id"]])))
    return selected


def subset_metadata(cell_id, ids):
    return {"result_scope": "repair_subset", "cell_id": cell_id,
            "task_ids": list(ids), "expected_subset_n": len(ids), "whole_cell_score": False}


def is_subset(cell, directory=None):
    return (cell.get("result_scope") == "repair_subset" or "task_ids" in cell
            or (directory is not None and (directory / "task_subset.json").exists()))


def finish_subset(cell, directory: Path):
    """Keep the official subset score, but label it before marking completion."""
    metadata = subset_metadata(cell["cell_id"], cell["task_ids"])
    path = directory / f"summary_{cell['arm']}.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    scored = summary.get("n_scored", summary.get("n"))
    if isinstance(scored, bool) or scored != metadata["expected_subset_n"]:
        raise RuntimeError(f"Repair subset scored {scored} tasks; expected {metadata['expected_subset_n']}")
    atomic_json(path, dict(summary, **metadata))
    return metadata
