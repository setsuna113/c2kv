"""Replay the stalled-operation selector on archived observable prefixes.

The report contains only aggregate counts, task/decision positions, tool names,
and controller statuses.  It never emits call arguments or observed results.
"""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path
from typing import Any

from history_memory.events import EventStore

from .failed_operation import records as failed_operation_records
from .stalled_operation import select_stalled_operation


def _task_number(path: Path) -> int:
    return int(path.parent.parent.name.rsplit("_", 1)[-1])


def _result_path(steps_path: Path) -> Path:
    paths = list(
        steps_path.parent.parent.glob(
            "bfcl/bfcl/result/*/multi_turn/*_result.json"
        )
    )
    if len(paths) != 1:
        raise RuntimeError(
            f"Expected one archived result beside {steps_path}; found {len(paths)}"
        )
    return paths[0]


def _read_rows(steps_path: Path) -> list[dict[str, Any]]:
    with steps_path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _append_observed_result(
    messages: list[dict[str, Any]],
    row: dict[str, Any],
    archived_step: list[dict[str, Any]],
    *,
    task_number: int,
) -> None:
    response = row.get("response") or {}
    calls = response.get("tool_calls") or []
    if response:
        messages.append(copy.deepcopy(response))
    if not calls:
        return
    tool_rows = [item for item in archived_step if item.get("role") == "tool"]
    if len(tool_rows) != len(calls):
        raise RuntimeError(
            "Archived call/result count mismatch at "
            f"task {task_number}, {row['decision_key']}"
        )
    for call, tool_row in zip(calls, tool_rows):
        messages.append(
            {
                "role": "tool",
                "tool_call_id": call["id"],
                "content": tool_row["content"],
            }
        )


def _failure_receipt_parity(
    diagnostic: dict[str, Any], receipt: dict[str, Any]
) -> bool:
    if len(diagnostic["failures"]) != receipt["failure_record_count"]:
        return False
    selected = receipt.get("selected_record")
    if not diagnostic["failures"]:
        return selected is None
    if not isinstance(selected, dict):
        return False
    return (
        diagnostic["failures"][0]["result_source_index"]
        == selected.get("result_source_index")
    )


def _structural_displacement(controller: dict[str, Any]) -> str:
    raw = set(controller["raw_event_ids"])
    gist = set(controller["gist_event_ids"])
    mandatory = set(controller["mandatory_raw_event_ids"])
    protected = set(controller["protected_event_ids"])
    reserve = controller["raw_reserve"]
    reserve_event = (
        reserve.get("admitted_event_id")
        if reserve.get("status") == "extra_event_admitted"
        else None
    )
    reserve_ok = (
        reserve_event is not None
        and reserve_event in raw
        and reserve_event in gist
        and reserve_event not in mandatory
        and reserve_event not in protected
    )
    lexical = controller["source_needs"]["admitted_event_ids"]
    lexical_ok = any(
        event_id in raw
        and event_id in gist
        and event_id not in mandatory
        and event_id not in protected
        for event_id in reversed(lexical)
    )
    if reserve_ok and lexical_ok:
        return "raw_reserve_then_one_lexical_available"
    if reserve_ok:
        return "raw_reserve_available"
    if lexical_ok:
        return "one_lexical_available"
    return "no_allowed_raw_demotion"


def analyze(returned_root: Path, *, expected_b0: int) -> dict[str, Any]:
    steps_paths = sorted(returned_root.glob("task_shards/*/server/steps.jsonl"))
    total_rows = 0
    controller_rows = 0
    exact_b0_rows = 0
    rows_without_controller_trace: list[dict[str, Any]] = []
    replayed_rows = 0
    failure_receipt_parity_rows = 0
    unreplayed_tasks: list[int] = []
    reconstruction_mismatches: list[dict[str, Any]] = []
    opportunities: list[dict[str, Any]] = []

    for steps_path in steps_paths:
        task_number = _task_number(steps_path)
        row_list = _read_rows(steps_path)
        total_rows += len(row_list)
        rows = {row["decision_key"]: row for row in row_list}
        for row in row_list:
            if not row.get("generation_trace"):
                rows_without_controller_trace.append(
                    {"task": task_number, "decision_key": row["decision_key"]}
                )
                continue
            controller_rows += 1
            controller = row["generation_trace"][0]["controller"]
            ratio = str(row["ratio"])
            actual = controller["actual_history_bytes"]
            budget = min(
                controller["history_budget_bytes"],
                controller["workspace_budget_bytes"],
            )
            if (
                actual == controller["per_ratio"][ratio]["history_bytes"]
                and actual <= budget
                and actual <= expected_b0
            ):
                exact_b0_rows += 1

        archived_result = json.loads(_result_path(steps_path).read_text(encoding="utf-8"))
        if "inference_log" not in archived_result:
            unreplayed_tasks.append(task_number)
            continue
        turns = [
            item
            for item in archived_result["inference_log"]
            if isinstance(item, dict)
        ]
        messages: list[dict[str, Any]] = []
        for turn_index, turn in enumerate(turns):
            messages.extend(copy.deepcopy(turn["begin_of_turn_query"]))
            step_keys = sorted(
                (key for key in turn if key.startswith("step_")),
                key=lambda key: int(key.split("_", 1)[1]),
            )
            for step_key in step_keys:
                decision_key = f"turn-{turn_index}/{step_key.replace('_', '-')}"
                row = rows[decision_key]
                controller = row["generation_trace"][0]["controller"]
                store = EventStore.from_messages("archived-prefix-replay", messages)
                diagnostic = failed_operation_records(store)
                parity = _failure_receipt_parity(
                    diagnostic, controller["failed_operation_cue"]
                )
                replayed_rows += 1
                if parity:
                    failure_receipt_parity_rows += 1
                else:
                    reconstruction_mismatches.append(
                        {
                            "task": task_number,
                            "decision_key": decision_key,
                            "replayed_failure_count": len(diagnostic["failures"]),
                            "archived_failure_count": controller[
                                "failed_operation_cue"
                            ]["failure_record_count"],
                        }
                    )
                selected = select_stalled_operation(store)
                if parity and selected.status == "triggered":
                    opportunities.append(
                        {
                            "task": task_number,
                            "decision_key": decision_key,
                            "selected_tool": selected.selected_tool,
                            "trigger_reasons": list(selected.trigger_reasons),
                            "unresolved_failed_signatures": (
                                selected.unresolved_failed_signatures
                            ),
                            "max_failed_observations_for_one_signature": (
                                selected.max_failed_observations_for_one_signature
                            ),
                            "max_later_observed_calls": (
                                selected.max_later_observed_calls
                            ),
                            "source_receipt_count": len(selected.sources),
                            "archived_failure_cue_status": controller[
                                "failed_operation_cue"
                            ]["status"],
                            "archived_raw_reserve_status": controller[
                                "raw_reserve"
                            ]["status"],
                            "structural_displacement": _structural_displacement(
                                controller
                            ),
                        }
                    )
                _append_observed_result(
                    messages,
                    row,
                    turn[step_key],
                    task_number=task_number,
                )

    trigger_reasons = Counter(
        reason for row in opportunities for reason in row["trigger_reasons"]
    )
    tasks = Counter(row["task"] for row in opportunities)
    cue_statuses = Counter(
        row["archived_failure_cue_status"] for row in opportunities
    )
    displacement = Counter(row["structural_displacement"] for row in opportunities)
    return {
        "version": "stalled-operation-archived-prefix-audit-v1",
        "input_scope": "observable archived prefixes and controller receipts only",
        "uses_gold_future_or_hidden_state": False,
        "arguments_or_result_values_emitted": False,
        "task_step_files": len(steps_paths),
        "server_rows": total_rows,
        "auditable_controller_rows": controller_rows,
        "exact_b0_rows": exact_b0_rows,
        "rows_without_controller_trace": rows_without_controller_trace,
        "replayed_rows": replayed_rows,
        "failure_receipt_parity_rows": failure_receipt_parity_rows,
        "unreplayed_tasks": sorted(unreplayed_tasks),
        "reconstruction_mismatches": reconstruction_mismatches,
        "triggered_rows": len(opportunities),
        "triggered_by_task": {
            str(task): tasks[task] for task in sorted(tasks)
        },
        "trigger_reason_counts": dict(sorted(trigger_reasons.items())),
        "archived_failure_cue_status_on_triggered_rows": dict(
            sorted(cue_statuses.items())
        ),
        "structural_displacement_on_triggered_rows": dict(
            sorted(displacement.items())
        ),
        "task120_triggered_rows": tasks[120],
        "triggered_opportunities": opportunities,
        "expected_b0": expected_b0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--returned-root", type=Path, required=True)
    parser.add_argument("--expected-b0", type=int, default=113_246_208)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = analyze(args.returned_root, expected_b0=args.expected_b0)
    payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.write_text(payload, encoding="utf-8")


if __name__ == "__main__":
    main()
