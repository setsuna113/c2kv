"""Declared ACEBench task failures for exact typed text-history budget codes.

Upstream ``generate.py`` re-raises any task exception and ends the run. When
the adapter sets ``ENV`` (text-history budget arms only), ``acebench_cli``
lets one agent task end on an exact typed HTTP 422 budget code instead: the
task writes no official result row and appends one receipt to ``ENV``.

The adapter accepts a receipt only when it names a requested task without a
result row, its episode ended as failed, and the proxy's last row for that
episode's measurement session carries the same exact code. Every other
exception, and every run without such a receipt, is unchanged.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

from .text_budget_failures import (
    TEXT_HISTORY_BUDGET_FAILURE_CODES, proxy_text_budget_failure_code,
)

ENV = "C2KV_ACEBENCH_TASK_FAILURES"
VERSION = "acebench-declared-task-failure-v1"
RECEIPTS = Path("measurement") / "acebench_task_failures.jsonl"


def receipt(task_id: str, code: str, error: Any, episode: Mapping[str, Any]) -> Dict[str, Any]:
    """One receipt row, written by the instrumented generator."""
    return {
        "version": VERSION,
        "task_id": str(task_id),
        "task_failure_kind": code,
        "exception_type": type(error).__name__,
        "status_code": getattr(error, "status_code", None),
        "episode_id": episode.get("task"),
        "episode_instance_id": episode.get("session"),
        "unix_ns": time.time_ns(),
    }


def _session_key(instance: str) -> str:
    """The proxy's conversation id for an explicit measurement session."""
    return hashlib.sha256(json.dumps(
        ["measurement_session", instance], ensure_ascii=False,
        sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def verified_task_failures(out_dir: Path, expected_ids: Iterable[str],
                           result_ids: Iterable[str]) -> Dict[str, str]:
    """Return ``{official task id: code}`` for receipts bound to their evidence."""
    from measurement.telemetry import read_jsonl

    path = Path(out_dir) / RECEIPTS
    if not path.is_file():
        return {}
    expected, results = set(expected_ids), set(result_ids)
    events = list(read_jsonl(Path(out_dir) / "measurement" / "harness_events.jsonl"))
    declared: Dict[str, str] = {}
    sessions: Dict[str, str] = {}
    for row in read_jsonl(path):
        if not isinstance(row, dict):
            raise SystemExit(f"FATAL: invalid ACEBench task failure receipt: {row!r}")
        task, code = row.get("task_id"), row.get("task_failure_kind")
        session = row.get("episode_instance_id")
        if (row.get("version") != VERSION or not isinstance(task, str) or not task
                or code not in TEXT_HISTORY_BUDGET_FAILURE_CODES
                or row.get("exception_type") != "UnprocessableEntityError"
                or row.get("status_code") != 422
                or not isinstance(session, str) or not session):
            raise SystemExit(f"FATAL: invalid ACEBench task failure receipt: {row!r}")
        if task in declared:
            raise SystemExit(f"FATAL: duplicate ACEBench task failure receipt: {task}")
        if task not in expected or task in results:
            raise SystemExit(
                f"FATAL: ACEBench task failure {task} is not a requested task without a result")
        starts = [event for event in events if event.get("event_type") == "episode_start"
                  and event.get("episode_instance_id") == session]
        ends = [event for event in events if event.get("event_type") == "episode_end"
                and event.get("episode_instance_id") == session]
        if (len(starts) != 1 or starts[0].get("episode_id") != row.get("episode_id")
                or len(ends) != 1 or ends[0].get("status") != "failed"):
            raise SystemExit(
                f"FATAL: ACEBench task failure {task} has no single failed episode")
        declared[task] = code
        sessions[_session_key(session)] = task
    latest: Dict[str, Dict[str, Any]] = {}
    for log in sorted((Path(out_dir) / "logs").glob("proxy_*.jsonl")):
        for row in read_jsonl(log):
            task = sessions.get(row.get("conv_id"))
            if task is not None:
                latest[task] = row
    for task, code in declared.items():
        if proxy_text_budget_failure_code(latest.get(task)) != code:
            raise SystemExit(
                f"FATAL: ACEBench task failure {task} lacks the matching proxy declaration")
    return declared
