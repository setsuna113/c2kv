"""Build and validate an exact-state T02 recovery contract.

The recovery keeps completed state triplets under their original plan
provenance.  Every incomplete slot is collected again from a live actor and
receives a new state identity.  A recovery plan therefore contains only the
new states; label merging retains both plan hashes instead of rewriting the
old results as if they came from the new plan.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import t02


CONTRACT_SCHEMA = "t02-exact-recovery-contract-v1"
COMBINED_LABEL_SCHEMA = "t02-multi-plan-labeled-set-v1"


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _slot(state: Mapping[str, Any]) -> tuple[str, str]:
    task_id = state.get("task_id")
    draft_kind = state.get("draft_kind")
    if not isinstance(task_id, str) or not task_id:
        raise ValueError("state task_id must be a nonempty string")
    if draft_kind not in {"call", "stop"}:
        raise ValueError("state draft_kind must be call or stop")
    return task_id, draft_kind


def _slot_record(state: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "task_id": state["task_id"],
        "draft_kind": state["draft_kind"],
        "task_group_id": state["task_group_id"],
        "split": state["split"],
        "source_plan_state_id": state["state_id"],
        "source_decision_key": state["decision_key"],
    }


def _validate_source_plan(plan: Mapping[str, Any]) -> tuple[list[Mapping[str, Any]], dict[str, str]]:
    t02.check_plan(plan)
    states = plan["states"]
    slots = [_slot(state) for state in states]
    if len(slots) != len(set(slots)):
        raise ValueError("source plan does not uniquely identify states by task_id/draft_kind")
    group_splits: dict[str, str] = {}
    for state in states:
        group = state["task_group_id"]
        split = state["split"]
        if group in group_splits and group_splits[group] != split:
            raise ValueError("source plan puts a task group in both splits")
        group_splits[group] = split
    return states, group_splits


def derive_recovery_contract(
    plan: Mapping[str, Any],
    ledger: Mapping[str, Any],
    partials: Mapping[str, Mapping[str, Any]],
    *,
    complete_branch_cap: int,
    documented_prior_complete_branches: int,
    additional_v6_complete_branches: int,
    incomplete_started_branch_attempts: int,
    prior_source_starts: int,
) -> dict[str, Any]:
    """Derive the smallest source subset for the frozen 119-state plan.

    The deterministic one-state reduction is independent of labels: among
    unexecuted calibration states whose removal also removes one source task,
    drop the last state in the already-frozen plan order.
    """

    states, group_splits = _validate_source_plan(plan)
    if type(complete_branch_cap) is not int or complete_branch_cap <= 0:
        raise ValueError("complete branch cap must be positive")
    for value, name in (
        (documented_prior_complete_branches, "documented prior branches"),
        (additional_v6_complete_branches, "additional v6 complete branches"),
        (incomplete_started_branch_attempts, "incomplete started branch attempts"),
        (prior_source_starts, "prior source starts"),
    ):
        if type(value) is not int or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")

    by_id = {state["state_id"]: state for state in states}
    branch_rows: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    result_sources: dict[tuple[str, str], str] = {}
    partial_receipts = []
    for worker_id, artifact in sorted(partials.items()):
        rows = artifact.get("results")
        if not isinstance(rows, list):
            raise ValueError(f"{worker_id} partial results must contain a result list")
        if artifact.get("complete_branch_executions") != len(rows):
            raise ValueError(f"{worker_id} complete branch count differs from its results")
        for row in rows:
            if not isinstance(row, Mapping) or row.get("execution_status") != "complete":
                raise ValueError("only completed branch results may consume the complete-branch budget")
            state_id, branch_id = row.get("state_id"), row.get("branch_id")
            if state_id not in by_id or branch_id not in t02.BRANCH_IDS:
                raise ValueError("partial result is outside the source plan")
            if branch_id in branch_rows[state_id]:
                raise ValueError("a source state repeats a completed branch")
            branch_rows[state_id][branch_id] = row
            result_sources[(state_id, branch_id)] = worker_id
        partial_receipts.append(
            {
                "worker_id": worker_id,
                "complete_branch_executions": len(rows),
                "plan_sha256": artifact.get("plan_sha256"),
            }
        )

    complete = {
        state_id for state_id, rows in branch_rows.items()
        if set(rows) == set(t02.BRANCH_IDS)
    }
    partial = set(branch_rows) - complete
    noncomplete = [state for state in states if state["state_id"] not in complete]
    remaining_by_task = Counter(state["task_id"] for state in noncomplete)
    group_counts = Counter(state["task_group_id"] for state in states)
    eligible_drop = [
        state for state in noncomplete
        if state["split"] == "calibration"
        and state["state_id"] not in branch_rows
        and remaining_by_task[state["task_id"]] == 1
        and group_counts[state["task_group_id"]] > 1
    ]
    if not eligible_drop:
        raise ValueError("no outcome-independent calibration state can reduce one source task")
    plan_index = {state["state_id"]: index for index, state in enumerate(states)}
    dropped = max(eligible_drop, key=lambda state: plan_index[state["state_id"]])

    v6_complete_branches = sum(len(rows) for rows in branch_rows.values())
    conservative_spent = (
        documented_prior_complete_branches
        + additional_v6_complete_branches
        + v6_complete_branches
    )
    available = complete_branch_cap - conservative_spent
    new_full_state_capacity = available // len(t02.BRANCH_IDS)
    desired_recollection = len(states) - len(complete) - 1
    if desired_recollection > new_full_state_capacity:
        raise ValueError("frozen plan cannot recover all but one state inside the branch cap")
    recapture = [
        state for state in noncomplete if state["state_id"] != dropped["state_id"]
    ]
    if len(recapture) != desired_recollection:
        raise AssertionError("recovery state arithmetic changed")
    recapture_slots = {_slot(state) for state in recapture}
    preserved_slots = {_slot(by_id[state_id]) for state_id in complete}
    if recapture_slots & preserved_slots:
        raise ValueError("recovery would recollect a preserved complete slot")

    ledger_tasks = ledger.get("tasks")
    if not isinstance(ledger_tasks, list):
        raise ValueError("source ledger tasks must be a list")
    tasks_by_id = {row.get("task_id"): row for row in ledger_tasks if isinstance(row, Mapping)}
    source_task_ids = {state["task_id"] for state in recapture}
    if set(tasks_by_id).issuperset(source_task_ids) is False:
        raise ValueError("source ledger omits a required recovery task")
    source_tasks = []
    states_by_task: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for state in recapture:
        states_by_task[state["task_id"]].append(state)
    for task_id in sorted(source_task_ids, key=lambda item: tasks_by_id[item]["task_index"]):
        source = tasks_by_id[task_id]
        if source.get("status") != "completed":
            raise ValueError("recovery task was not completed in the source ledger")
        source_tasks.append(
            {
                "task_id": task_id,
                "source_task_index": source["task_index"],
                "expected_slots": [
                    _slot_record(state)
                    for state in sorted(states_by_task[task_id], key=_slot)
                ],
            }
        )

    final_states = [by_id[state_id] for state_id in complete] + recapture
    final_split_counts = Counter(state["split"] for state in final_states)
    final_groups = {state["task_group_id"] for state in final_states}
    if final_groups != set(group_splits):
        raise ValueError("recovery would remove an original task group")
    if any(Counter(state["task_group_id"] for state in final_states)[group] > plan["max_states_per_task_group"] for group in final_groups):
        raise ValueError("recovery exceeds the original group cap")

    preserved_states = []
    for state_id in sorted(complete, key=plan_index.__getitem__):
        state = by_id[state_id]
        preserved_states.append(
            {
                **_slot_record(state),
                "state_id": state_id,
                "branches": [
                    {
                        "branch_id": branch_id,
                        "worker_id": result_sources[(state_id, branch_id)],
                    }
                    for branch_id in t02.BRANCH_IDS
                ],
            }
        )
    partial_states = []
    for state_id in sorted(partial, key=plan_index.__getitem__):
        state = by_id[state_id]
        partial_states.append(
            {
                **_slot_record(state),
                "state_id": state_id,
                "completed_branches": sorted(branch_rows[state_id]),
                "reuse_for_recollected_state": False,
                "accounting": "complete_branch_cost_only",
            }
        )

    new_branches = len(recapture) * len(t02.BRANCH_IDS)
    cumulative = conservative_spent + new_branches
    contract = {
        "schema": CONTRACT_SCHEMA,
        "selection_policy": {
            "source_plan_is_frozen": True,
            "uses_future_success_or_labels": False,
            "drop_rule": (
                "last source-plan state that is unexecuted, calibration, keeps its "
                "task group, and removes one whole source task"
            ),
            "slot_identity": ["task_id", "draft_kind"],
            "missing_or_ambiguous_slot_policy": "fail_without_substitution",
        },
        "source_plan": {
            "schema": plan["schema"],
            "sha256": _digest(plan),
            "target_states": len(states),
            "split_counts": dict(Counter(state["split"] for state in states)),
            "task_group_count": len(group_splits),
        },
        "source_ledger": {
            "schema": ledger.get("schema"),
            "sha256": _digest(ledger),
            "task_count": len(ledger_tasks),
            "candidate_count": len(ledger.get("candidates", [])),
        },
        "group_split_bindings": dict(sorted(group_splits.items())),
        "budget": {
            "complete_branch_cap": complete_branch_cap,
            "documented_complete_branches_before_v6": documented_prior_complete_branches,
            "v6_completed_branch_executions": v6_complete_branches,
            "v6_additional_complete_branches_outside_partial_results": additional_v6_complete_branches,
            "v6_completed_triplet_branches_reused": len(complete) * 3,
            "v6_partial_completed_branches_cost_only": sum(len(branch_rows[state_id]) for state_id in partial),
            "v6_incomplete_started_branch_attempts_audited_separately": incomplete_started_branch_attempts,
            "new_complete_branch_executions": new_branches,
            "complete_branches_before_recovery": conservative_spent,
            "cumulative_complete_branches": cumulative,
            "remaining_complete_branch_slack": complete_branch_cap - cumulative,
            "incomplete_started_attempts_consume_complete_branch_cap": False,
        },
        "target": {
            "combined_exact_state_count": len(final_states),
            "preserved_exact_state_count": len(complete),
            "new_exact_state_count": len(recapture),
            "split_counts": dict(final_split_counts),
            "task_group_count": len(final_groups),
        },
        "source_budget": {
            "prior_source_starts": prior_source_starts,
            "new_source_task_starts": len(source_tasks),
            "cumulative_source_starts": prior_source_starts + len(source_tasks),
            "original_variant_universe": len(ledger_tasks),
            "strict_subset_only": True,
            "automatic_reruns": 0,
        },
        "snapshot_contract": {
            "required_components": list(plan["required_snapshot_components"]),
            "new_live_exact_snapshot_required_for_every_recollected_slot": True,
            "prefix_replay_is_exact_snapshot": False,
            "old_partial_and_new_branches_may_form_one_state": False,
        },
        "preserved_states": preserved_states,
        "cost_only_partial_states": partial_states,
        "dropped_state": {
            **_slot_record(dropped),
            "source_plan_index": plan_index[dropped["state_id"]],
            "reason": "one-state branch-cap reduction with one-source-task reduction",
        },
        "recapture_slots": [_slot_record(state) for state in recapture],
        "source_tasks": source_tasks,
        "partial_result_receipts": partial_receipts,
        "label_merge_contract": {
            "preserved_and_recovery_plans_keep_distinct_sha256": True,
            "combined_rows_are_for_training_only": True,
            "do_not_rewrite_old_result_plan_sha256": True,
        },
    }
    validate_recovery_contract(contract)
    return contract


def validate_recovery_contract(contract: Mapping[str, Any]) -> None:
    if contract.get("schema") != CONTRACT_SCHEMA:
        raise ValueError(f"recovery contract schema must be {CONTRACT_SCHEMA}")
    preserved = contract.get("preserved_states")
    recapture = contract.get("recapture_slots")
    source_tasks = contract.get("source_tasks")
    if not all(isinstance(value, list) for value in (preserved, recapture, source_tasks)):
        raise ValueError("recovery state and source task manifests must be lists")
    preserved_slots = {(row["task_id"], row["draft_kind"]) for row in preserved}
    recapture_slots = {(row["task_id"], row["draft_kind"]) for row in recapture}
    if len(preserved_slots) != len(preserved) or len(recapture_slots) != len(recapture):
        raise ValueError("recovery manifest repeats a slot")
    if preserved_slots & recapture_slots:
        raise ValueError("preserved and recaptured slots overlap")
    task_slots = {
        (slot["task_id"], slot["draft_kind"])
        for task in source_tasks for slot in task["expected_slots"]
    }
    if task_slots != recapture_slots:
        raise ValueError("source task manifest does not cover exactly the recovery slots")
    target = contract["target"]
    if target["preserved_exact_state_count"] != len(preserved):
        raise ValueError("preserved state count differs")
    if target["new_exact_state_count"] != len(recapture):
        raise ValueError("recovery state count differs")
    if target["combined_exact_state_count"] != len(preserved) + len(recapture):
        raise ValueError("combined state count differs")
    budget = contract["budget"]
    if budget["cumulative_complete_branches"] > budget["complete_branch_cap"]:
        raise ValueError("recovery exceeds the complete-branch cap")


def build_recovery_plan(
    contract: Mapping[str, Any],
    candidate_states: Sequence[Mapping[str, Any]],
    *,
    source_plan: Mapping[str, Any],
    frozen_subsequent_policy: Mapping[str, Any],
) -> dict[str, Any]:
    """Select newly captured states by frozen task/draft slots.

    New state IDs are expected.  Old state IDs are lineage only and are never
    used as a substitute for an unavailable live snapshot.
    """

    validate_recovery_contract(contract)
    _validate_source_plan(source_plan)
    if _digest(source_plan) != contract["source_plan"]["sha256"]:
        raise ValueError("source plan digest differs from the frozen recovery contract")
    by_slot: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for state in candidate_states:
        by_slot[_slot(state)].append(state)
    expected = {
        (row["task_id"], row["draft_kind"]): row
        for row in contract["recapture_slots"]
    }
    missing = [slot for slot in expected if len(by_slot.get(slot, [])) != 1]
    if missing:
        raise ValueError(f"missing or ambiguous recovery slots: {missing}")
    planned = []
    for slot, lineage in expected.items():
        state = t02._normalize_candidate_state(by_slot[slot][0])
        if state["task_group_id"] != lineage["task_group_id"]:
            raise ValueError("recollected slot changed task group")
        state["split"] = lineage["split"]
        row = dict(state)
        row["schema"] = t02.PLANNED_STATE_SCHEMA
        row["branches"] = list(t02._branch_actions(state, seed=source_plan["seed"]))
        row["branch_selection_uses_future_outcome"] = False
        planned.append(row)
    planned.sort(key=lambda row: (row["split"] != "train", row["task_group_id"], row["state_id"]))
    split_counts = Counter(row["split"] for row in planned)
    groups = {row["task_group_id"] for row in planned}
    policy = json.loads(_canonical(frozen_subsequent_policy).decode("utf-8"))
    plan = {
        "schema": t02.PLAN_SCHEMA,
        "seed": source_plan["seed"],
        "target_states": len(planned),
        "target_train_states": split_counts["train"],
        "minimum_task_groups": len(groups),
        "max_states_per_task_group": source_plan["max_states_per_task_group"],
        "max_states_per_task": source_plan["max_states_per_task"],
        "branches_per_state": len(t02.BRANCH_IDS),
        "max_complete_branch_executions": len(planned) * len(t02.BRANCH_IDS),
        "authorized_complete_branch_cap": len(planned) * len(t02.BRANCH_IDS),
        "frozen_subsequent_policy": policy,
        "frozen_subsequent_policy_sha256": _digest(policy),
        "required_snapshot_components": list(t02.REQUIRED_COMPONENTS),
        "prefix_replay_is_exact_snapshot": False,
        "forbidden_manifests": source_plan["forbidden_manifests"],
        "rejected_forbidden_state_ids": [],
        "sampling": {
            "selection_unit": "frozen_source_plan_slot",
            "slot_fields": ["task_id", "draft_kind"],
            "branch_is_independent_sample": False,
            "uses_future_success": False,
            "source_plan_sha256": contract["source_plan"]["sha256"],
        },
        "states": planned,
    }
    t02.check_plan(plan)
    return plan


def merge_label_partitions(
    contract: Mapping[str, Any],
    preserved_labels: Mapping[str, Any],
    recovery_labels: Mapping[str, Any],
) -> dict[str, Any]:
    """Create a training row set while retaining two independent plan hashes."""

    validate_recovery_contract(contract)
    old_rows = preserved_labels.get("rows")
    new_rows = recovery_labels.get("rows")
    if not isinstance(old_rows, list) or not isinstance(new_rows, list):
        raise ValueError("both label partitions must contain rows")
    old_by_id = {row.get("state_id"): row for row in old_rows if isinstance(row, Mapping)}
    old_ids = {row["state_id"] for row in contract["preserved_states"]}
    if not old_ids.issubset(old_by_id):
        raise ValueError("preserved label partition omits a completed state")
    selected_old = [old_by_id[row["state_id"]] for row in contract["preserved_states"]]
    expected_new_slots = {
        (row["task_id"], row["draft_kind"]) for row in contract["recapture_slots"]
    }
    actual_new_slots = {_slot(row) for row in new_rows}
    if actual_new_slots != expected_new_slots or len(new_rows) != len(expected_new_slots):
        raise ValueError("recovery labels do not cover exactly the recaptured slots")
    bindings = contract["group_split_bindings"]
    combined = [*selected_old, *new_rows]
    if len({row["state_id"] for row in combined}) != len(combined):
        raise ValueError("label partitions repeat a state ID")
    for row in combined:
        if bindings.get(row.get("task_group_id")) != row.get("split"):
            raise ValueError("label row changed the frozen task-group split")
    if len(combined) != contract["target"]["combined_exact_state_count"]:
        raise ValueError("combined training rows differ from recovery target")
    return {
        "schema": COMBINED_LABEL_SCHEMA,
        "state_count": len(combined),
        "rows": combined,
        "partitions": [
            {
                "role": "preserved_v6_completed_triplets",
                "plan_sha256": preserved_labels.get("plan_sha256"),
                "source_artifact_sha256": _digest(preserved_labels),
                "selected_state_count": len(selected_old),
            },
            {
                "role": "new_exact_recollection",
                "plan_sha256": recovery_labels.get("plan_sha256"),
                "source_artifact_sha256": _digest(recovery_labels),
                "selected_state_count": len(new_rows),
            },
        ],
        "recovery_contract_sha256": _digest(contract),
        "rows_sha256": _digest(combined),
        "old_result_plan_sha256_rewritten": False,
    }


def extract_prefill_contract(labels: Mapping[str, Any]) -> dict[str, Any]:
    """Extract the shared C1 prefill contract from a multi-plan label set."""

    if labels.get("schema") != COMBINED_LABEL_SCHEMA:
        raise ValueError(f"labels schema must be {COMBINED_LABEL_SCHEMA}")
    rows = labels.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("combined labels must contain rows")
    contracts = []
    for row in rows:
        question = row.get("q") if isinstance(row, Mapping) else None
        contract = question.get("prefill_contract") if isinstance(question, Mapping) else None
        if not isinstance(contract, Mapping) or not contract:
            raise ValueError("every combined label row must bind q.prefill_contract")
        contracts.append(dict(contract))
    encoded = {_canonical(contract) for contract in contracts}
    if len(encoded) != 1:
        raise ValueError("combined label rows mix prefill contracts")
    return contracts[0]


def training_lane_receipt(
    labels_path: Path, artifact_path: Path, manifest_path: Path, lane: str
) -> dict[str, Any]:
    labels = _read(labels_path)
    if labels.get("schema") != COMBINED_LABEL_SCHEMA or labels.get("state_count") != 118:
        raise ValueError("training receipts require the frozen 118-state multi-plan labels")
    if lane not in {"C1", "C4_turn", "C4_task"}:
        raise ValueError("unknown T02 training lane")
    if not artifact_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("training artifact or frozen manifest is missing")
    return {
        "schema": "t02-training-lane-receipt-v1",
        "status": "completed",
        "lane": lane,
        "artifact": str(artifact_path),
        "artifact_sha256": _file_sha256(artifact_path),
        "labels_sha256": _file_sha256(labels_path),
        "state_count": labels["state_count"],
        "frozen_manifest_sha256": _file_sha256(manifest_path),
    }


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _cli() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    derive = sub.add_parser("derive")
    derive.add_argument("--plan", type=Path, required=True)
    derive.add_argument("--ledger", type=Path, required=True)
    derive.add_argument("--partial", type=Path, action="append", required=True)
    derive.add_argument("--complete-branch-cap", type=int, default=360)
    derive.add_argument("--documented-prior-complete-branches", type=int, default=1)
    derive.add_argument("--additional-v6-complete-branches", type=int, default=1)
    derive.add_argument("--incomplete-started-branch-attempts", type=int, default=5)
    derive.add_argument("--prior-source-starts", type=int, default=158)
    derive.add_argument("--additional-branch-receipt", type=Path, required=True)
    derive.add_argument("--output", type=Path, required=True)
    extract = sub.add_parser("extract-prefill-contract")
    extract.add_argument("--labels", type=Path, required=True)
    extract.add_argument("--output", type=Path, required=True)
    receipt = sub.add_parser("write-training-receipt")
    receipt.add_argument("--labels", type=Path, required=True)
    receipt.add_argument("--artifact", type=Path, required=True)
    receipt.add_argument("--manifest", type=Path, required=True)
    receipt.add_argument("--lane", required=True)
    receipt.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "derive":
        partials = {path.stem.removesuffix(".partial_results"): _read(path) for path in args.partial}
        contract = derive_recovery_contract(
            _read(args.plan), _read(args.ledger), partials,
            complete_branch_cap=args.complete_branch_cap,
            documented_prior_complete_branches=args.documented_prior_complete_branches,
            additional_v6_complete_branches=args.additional_v6_complete_branches,
            incomplete_started_branch_attempts=args.incomplete_started_branch_attempts,
            prior_source_starts=args.prior_source_starts,
        )
        contract["input_artifacts"] = {
            "plan": {"path": str(args.plan), "file_sha256": _file_sha256(args.plan)},
            "ledger": {"path": str(args.ledger), "file_sha256": _file_sha256(args.ledger)},
            "partials": [
                {"path": str(path), "file_sha256": _file_sha256(path)} for path in args.partial
            ],
            "additional_complete_branch_receipt": {
                "path": str(args.additional_branch_receipt),
                "file_sha256": _file_sha256(args.additional_branch_receipt),
            },
        }
        _write(args.output, contract)
        print(json.dumps({
            "output": str(args.output),
            "sha256": _file_sha256(args.output),
            "target": contract["target"],
            "source_budget": contract["source_budget"],
            "budget": contract["budget"],
        }, ensure_ascii=False, sort_keys=True))
        return 0
    if args.command == "extract-prefill-contract":
        value = extract_prefill_contract(_read(args.labels))
        _write(args.output, value)
        print(json.dumps(value, ensure_ascii=False, sort_keys=True))
        return 0
    if args.command == "write-training-receipt":
        value = training_lane_receipt(
            args.labels, args.artifact, args.manifest, args.lane
        )
        _write(args.output, value)
        print(json.dumps(value, ensure_ascii=False, sort_keys=True))
        return 0
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(_cli())
