"""Run a fixed C2 task list as isolated one-task runner invocations.

The actor engine is owned by the surrounding evaluation launcher.  This
dispatcher deliberately reuses ``runner.py`` for each task so its BFCL and
runtime-failure semantics stay authoritative.  A task-local, structured
extraction-budget terminal is the only failure that permits the next fixed
task to start; all other failures stop dispatch.  No task is retried.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


TYPED_BUDGET_ERROR = "SGLangExtractionBudgetExhausted"
TYPED_BUDGET_MESSAGE_PREFIX = "C2KV_EXTRACTION_BUDGET_EXHAUSTED:"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def _save(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _last_jsonl_record(path: Path) -> dict[str, Any]:
    """Read only the final nonempty JSONL record, even for a large trace."""

    with path.open("rb") as stream:
        stream.seek(0, os.SEEK_END)
        position = stream.tell()
        buffer = bytearray()
        while position > 0:
            position -= 1
            stream.seek(position)
            byte = stream.read(1)
            if byte == b"\n" and buffer:
                line = bytes(reversed(buffer)).strip()
                if line:
                    value = json.loads(line.decode("utf-8"))
                    if not isinstance(value, dict):
                        raise TypeError(f"JSONL record must be an object: {path}")
                    return value
                buffer.clear()
            else:
                buffer.extend(byte)
        line = bytes(reversed(buffer)).strip()
    if not line:
        raise ValueError(f"JSONL file has no records: {path}")
    value = json.loads(line.decode("utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSONL record must be an object: {path}")
    return value


def typed_budget_error_path(record: Mapping[str, Any]) -> str | None:
    """Require the typed code and fixed prefix; never classify a message alone."""

    error = record.get("error")
    if not isinstance(error, Mapping):
        return None
    if (
        error.get("type") == TYPED_BUDGET_ERROR
        and isinstance(error.get("message"), str)
        and error["message"].startswith(TYPED_BUDGET_MESSAGE_PREFIX)
    ):
        return "error.type"
    cause = error.get("cause")
    if isinstance(cause, Mapping):
        cause_message = cause.get("message")
        if (
            cause.get("type") == TYPED_BUDGET_ERROR
            and isinstance(cause_message, str)
            and cause_message.startswith(TYPED_BUDGET_MESSAGE_PREFIX)
        ):
            return "error.cause.type"
        if (
            cause.get("code") == TYPED_BUDGET_ERROR
            and isinstance(cause_message, str)
            and cause_message.startswith(TYPED_BUDGET_MESSAGE_PREFIX)
        ):
            return "error.cause.code"
    return None


def classify_attempt(
    *,
    task_id: str,
    returncode: int,
    manifest: Mapping[str, Any],
    final_step: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Validate one runner attempt and decide whether dispatch may continue."""

    if manifest.get("task_ids") != [task_id]:
        raise RuntimeError(f"Single-task runner manifest changed for {task_id}")
    outcomes = manifest.get("task_outcomes")
    if not isinstance(outcomes, list) or len(outcomes) != 1:
        raise RuntimeError(f"Single-task runner did not emit exactly one outcome: {task_id}")
    outcome = outcomes[0]
    if not isinstance(outcome, dict) or outcome.get("task_id") != task_id:
        raise RuntimeError(f"Single-task runner outcome changed for {task_id}")

    if (
        returncode == 0
        and manifest.get("status") == "completed_fixed_manifest"
        and outcome.get("outcome") == "official_completed"
        and outcome.get("runtime_completed") is True
    ):
        return {
            "task_id": task_id,
            "classification": "official_completed",
            "continue_dispatch": True,
            "runner_returncode": returncode,
            "typed_error_path": None,
            "outcome": copy.deepcopy(outcome),
        }

    typed_path = typed_budget_error_path(final_step or {})
    if (
        returncode == 6
        and manifest.get("status") == "stopped_on_actor_runtime_failure"
        and outcome.get("outcome") == "runtime_failure_in_denominator"
        and outcome.get("runtime_completed") is False
        and typed_path is not None
    ):
        typed_outcome = copy.deepcopy(outcome)
        typed_outcome.update(
            {
                "outcome": "runtime_failure_in_denominator",
                "runtime_completed": False,
                "official_zero_accepted_for_quality": False,
                "typed_terminal": TYPED_BUDGET_ERROR,
                "typed_error_path": typed_path,
                "dispatch_stop_reason": None,
            }
        )
        return {
            "task_id": task_id,
            "classification": "typed_extraction_budget_runtime_failure",
            "continue_dispatch": True,
            "runner_returncode": returncode,
            "typed_error_path": typed_path,
            "outcome": typed_outcome,
        }

    failed_outcome = copy.deepcopy(outcome)
    failed_outcome["dispatch_stop_reason"] = "unhandled_single_task_failure"
    return {
        "task_id": task_id,
        "classification": "unhandled_failure",
        "continue_dispatch": False,
        "runner_returncode": returncode,
        "typed_error_path": typed_path,
        "outcome": failed_outcome,
    }


def aggregate_attempts(
    task_ids: Sequence[str], attempts: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("Fixed task list contains duplicates")
    attempted_ids = [row.get("task_id") for row in attempts]
    if attempted_ids != list(task_ids[: len(attempts)]):
        raise ValueError("Attempts must be the exact fixed-manifest prefix")
    stop_rows = [row for row in attempts if not row.get("continue_dispatch")]
    if len(stop_rows) > 1 or (stop_rows and attempts[-1] is not stop_rows[0]):
        raise ValueError("Dispatch must stop immediately after the first unhandled failure")

    outcomes = [copy.deepcopy(row["outcome"]) for row in attempts]
    outcomes.extend(
        {
            "task_id": task_id,
            "outcome": "not_started",
            "in_fixed_denominator": True,
            "official_summary": None,
            "runtime_completed": False,
        }
        for task_id in task_ids[len(attempts) :]
    )
    terminal = not stop_rows and len(attempts) == len(task_ids)
    return {
        "schema": "a-history-system-c2-remaining-run-v1",
        "status": (
            "completed_fixed_manifest_with_typed_runtime_failures"
            if terminal
            else "stopped_on_unhandled_runtime_failure"
            if stop_rows
            else "running_fixed_manifest"
        ),
        "state": "completed" if terminal else "failed" if stop_rows else "running",
        "task_ids": list(task_ids),
        "whole_task_denominator": len(task_ids),
        "denominator_observed": len(task_ids),
        "task_cells_started": len(attempts),
        "completed_task_cells": len(attempts),
        "automatic_retries": 0,
        "automatic_reruns": 0,
        "task_outcomes": outcomes,
        "attempt_receipts": [copy.deepcopy(dict(row)) for row in attempts],
        "counts": {
            "official_completed": sum(
                row.get("classification") == "official_completed" for row in attempts
            ),
            "typed_extraction_budget_runtime_failure": sum(
                row.get("classification")
                == "typed_extraction_budget_runtime_failure"
                for row in attempts
            ),
            "unhandled_failure": len(stop_rows),
            "not_started": len(task_ids) - len(attempts),
        },
        "quality_contract": {
            "sample_label": "preliminary, n=1",
            "typed_budget_official_zero_accepted": False,
            "full_d20_quality_available": False,
        },
    }


def _one_task_design(
    design: Mapping[str, Any], task_id: str, *, task_manifest_sha256: str
) -> dict[str, Any]:
    value = copy.deepcopy(dict(design))
    value["task_ids"] = [task_id]
    value["task_manifest_sha256"] = task_manifest_sha256
    value["limits"] = {**value["limits"], "tasks": 1}
    value["run_id_template"] = f"{value['run_id_template']}_{task_id}"
    value["search_contract"] = {
        **value["search_contract"],
        "outer_fixed_manifest": "C2-remaining8-single-attempt-v1",
        "outer_task_id": task_id,
    }
    value["automatic_reruns"] = 0
    return value


def run_fixed_manifest(args: argparse.Namespace) -> int:
    design_path = Path(args.design).resolve()
    design = _read(design_path)
    task_ids = design.get("task_ids")
    if not isinstance(task_ids, list) or len(task_ids) != 8 or len(set(task_ids)) != 8:
        raise RuntimeError("C2 remaining dispatcher requires exactly eight unique tasks")
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    output.mkdir(parents=True)
    aggregate_path = output / "stage_manifest.json"
    attempts: list[dict[str, Any]] = []
    started = time.monotonic()
    started_at = _now()
    package_root = Path(args.runtime_root).resolve().parents[2]
    for index, task_id in enumerate(task_ids):
        task_root = output / "task_attempts" / task_id
        task_root.mkdir(parents=True)
        single_manifest = task_root / "task_manifest.json"
        _save(
            single_manifest,
            {
                "schema": "a-history-system-task-manifest-v1",
                "manifest_id": f"D20-C2-remaining8-{task_id}",
                "stage": "development_search_single_attempt",
                "task_ids": [task_id],
                "fixed_denominator": 1,
                "outer_fixed_manifest_sha256": design.get("task_manifest_sha256"),
                "automatic_retries": 0,
                "automatic_reruns": 0,
            },
        )
        single_design = task_root / "design.json"
        _save(
            single_design,
            _one_task_design(
                design, task_id, task_manifest_sha256=_sha(single_manifest)
            ),
        )
        result_root = task_root / "results"
        log_path = task_root / "runner.log"
        command = [
            args.python,
            str(package_root / "history_system" / "runner.py"),
            "run",
            "--design",
            str(single_design),
            "--runtime-root",
            str(Path(args.runtime_root).resolve()),
            "--checkpoint",
            args.checkpoint,
            "--output",
            str(result_root),
            "--benchmark-dir",
            args.benchmark_dir,
            "--python",
            args.python,
            "--bfcl-python",
            args.bfcl_python,
            "--port-base",
            str(args.port_base + index),
        ]
        with log_path.open("x", encoding="utf-8", newline="\n") as log:
            completed = subprocess.run(
                command,
                cwd=package_root,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        child_manifest_path = result_root / "stage_manifest.json"
        if not child_manifest_path.is_file():
            raise RuntimeError(f"Runner emitted no manifest for {task_id}")
        child_manifest = _read(child_manifest_path)
        steps_path = result_root / "task_shards" / task_id / "server" / "steps.jsonl"
        final_step = _last_jsonl_record(steps_path) if steps_path.is_file() else None
        receipt = classify_attempt(
            task_id=task_id,
            returncode=completed.returncode,
            manifest=child_manifest,
            final_step=final_step,
        )
        receipt.update(
            {
                "started_once": True,
                "attempt_index": index + 1,
                "runner_manifest": str(child_manifest_path),
                "runner_manifest_sha256": _sha(child_manifest_path),
                "runner_log": str(log_path),
                "final_step_path": str(steps_path) if steps_path.is_file() else None,
                "final_step_sha256": _sha(steps_path) if steps_path.is_file() else None,
            }
        )
        attempts.append(receipt)
        aggregate = aggregate_attempts(task_ids, attempts)
        aggregate.update(
            {
                "started_at": started_at,
                "wall_seconds": time.monotonic() - started,
                "source_design": str(design_path),
                "source_design_sha256": _sha(design_path),
            }
        )
        _save(aggregate_path, aggregate)
        if not receipt["continue_dispatch"]:
            return 6
    aggregate = aggregate_attempts(task_ids, attempts)
    aggregate.update(
        {
            "started_at": started_at,
            "finished_at": _now(),
            "wall_seconds": time.monotonic() - started,
            "wall_seconds_final": True,
            "source_design": str(design_path),
            "source_design_sha256": _sha(design_path),
        }
    )
    _save(aggregate_path, aggregate)
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", choices=("run",))
    parser.add_argument("--design", required=True)
    parser.add_argument("--runtime-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--benchmark-dir", required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--bfcl-python", required=True)
    parser.add_argument("--port-base", required=True, type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    return run_fixed_manifest(_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
