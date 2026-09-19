"""Canonical BFCL result collection for resumable generality cells.

Raw batch directories are immutable attempts.  Retried tasks can therefore
have more than one official row; completion and scoring must operate on one
canonical row per task rather than on marker files or raw row counts.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

try:
    from .bfcl_completion import completion_kind, terminal_failure_kind
except ImportError:
    from bfcl_completion import completion_kind, terminal_failure_kind


RESULT_GLOB = "batches/*/bfcl_worker/bfcl/result/**/*.json"


def ordered_unique(values: Iterable[str]) -> list[str]:
    """Return unique nonempty task IDs while preserving manifest order."""
    return list(dict.fromkeys(value for value in values if value))


def bfcl_row_is_valid(row: dict[str, Any]) -> bool:
    """Whether a row is terminal for the official scorer.

    An incorrect or empty model result, context overflow, or invalid HiAgent
    retrieval is scored as-is.  Other tracebacks and rows without ``result``
    are retryable execution failures.
    """
    return completion_kind(row) != "incomplete"


def collect_bfcl_results(
    cell_dir: Path, expected_task_ids: Iterable[str]
) -> dict[str, Any]:
    """Collect immutable attempts and resolve one canonical row per task.

    A valid row always beats an invalid row.  Historical cells may contain
    several valid attempts; the latest one wins to match the offline rescore
    convention, with path and line number making ties deterministic.
    """
    cell_dir = Path(cell_dir)
    expected = ordered_unique(expected_task_ids)
    expected_set = set(expected)
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    total_rows = 0
    malformed_rows = 0

    for result_path in sorted(cell_dir.glob(RESULT_GLOB)):
        try:
            mtime_ns = result_path.stat().st_mtime_ns
            lines = result_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            total_rows += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                malformed_rows += 1
                continue
            if not isinstance(row, dict):
                malformed_rows += 1
                continue
            task_id = row.get("id")
            if not isinstance(task_id, str) or not task_id:
                malformed_rows += 1
                continue
            by_task[task_id].append({
                "task_id": task_id,
                "row": row,
                "path": str(result_path),
                "category": result_path.parent.name,
                "mtime_ns": mtime_ns,
                "line_number": line_number,
                "valid": bfcl_row_is_valid(row),
            })

    canonical: dict[str, dict[str, Any]] = {}
    for task_id, attempts in by_task.items():
        if task_id not in expected_set:
            continue
        canonical[task_id] = max(
            attempts,
            key=lambda item: (
                bool(item["valid"]),
                int(item["mtime_ns"]),
                str(item["path"]),
                int(item["line_number"]),
            ),
        )

    valid_task_ids = [
        task_id for task_id in expected
        if task_id in canonical and canonical[task_id]["valid"]
    ]
    invalid_task_ids = [
        task_id for task_id in expected
        if task_id in canonical and not canonical[task_id]["valid"]
    ]
    missing_task_ids = [task_id for task_id in expected if task_id not in canonical]
    terminal_failures = {
        task_id: kind
        for task_id in expected
        if task_id in canonical
        if (kind := terminal_failure_kind(canonical[task_id]["row"])) is not None
    }
    refill_set = set(invalid_task_ids) | set(missing_task_ids)
    refill_task_ids = [task_id for task_id in expected if task_id in refill_set]
    unexpected_task_ids = sorted(set(by_task) - expected_set)
    expected_rows = sum(len(by_task.get(task_id, ())) for task_id in expected)

    return {
        "schema": "generality-bfcl-canonical-results-v1",
        "expected_task_ids": expected,
        "canonical": canonical,
        "valid_task_ids": valid_task_ids,
        "invalid_task_ids": invalid_task_ids,
        "missing_task_ids": missing_task_ids,
        "terminal_failures": terminal_failures,
        "refill_task_ids": refill_task_ids,
        "unexpected_task_ids": unexpected_task_ids,
        "expected_count": len(expected),
        "valid_count": len(valid_task_ids),
        "invalid_count": len(invalid_task_ids),
        "missing_count": len(missing_task_ids),
        "unique_ids": len(canonical),
        "all_unique_ids": len(by_task),
        "expected_rows": expected_rows,
        "total_rows": total_rows,
        "duplicate_rows": max(0, expected_rows - len(canonical)),
        "malformed_rows": malformed_rows,
    }


def completion_receipt(completion: dict[str, Any]) -> dict[str, Any]:
    """Strip full result rows for the small on-disk progress receipt."""
    return {
        key: value
        for key, value in completion.items()
        if key != "canonical"
    }
