"""Plan, execute, reuse, merge, and label H0 proposal-matched T02 data.

This is a versioned companion to :mod:`t02`.  It deliberately leaves the old
three-branch A0/A1/A2 schemas unchanged.  The only scored interventions here
are the frozen ``Slex`` and ``Ssrc`` proposals from the runtime protocol; empty
or duplicate proposals never create duplicate continuation executions.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

import t02
from benchmarks.memory_runtime.recovery.set_protocol import (
    PROPOSAL_PROTOCOL,
    build_proposal_catalog,
)


PLAN_SCHEMA = "proposal-t02-plan-h0-v1"
RESULT_SET_SCHEMA = "proposal-t02-results-h0-v1"
LABELED_SET_SCHEMA = "proposal-t02-labeled-dataset-h0-v1"
LABELED_STATE_SCHEMA = "proposal-t02-labeled-state-h0-v1"
SOURCE_BINDING_SCHEMA = "proposal-t02-source-binding-h0-v1"
MAX_NEW_STATES = 60
MAX_NEW_CONTINUATIONS = 180
BRANCH_BY_ORIGIN = {"empty": "A0", "Slex": "A1lex", "Ssrc": "A2src"}


class ProposalT02Error(ValueError):
    """A proposal-matched T02 artifact violates its frozen contract."""


class ProposalBranchAdapter(Protocol):
    def capabilities(self) -> Mapping[str, Any]: ...
    def capture_state(self, state: Mapping[str, Any], *, frozen_policy: Mapping[str, Any]) -> Mapping[str, Any]: ...
    def restore_state(self, snapshot: Mapping[str, Any], *, frozen_policy: Mapping[str, Any]) -> Mapping[str, Any]: ...
    def run_branch(self, state: Mapping[str, Any], branch: Mapping[str, Any], *, frozen_policy: Mapping[str, Any], restore_receipt: Mapping[str, Any]) -> Mapping[str, Any]: ...


def _json_copy(value: Any, label: str = "value") -> Any:
    try:
        return json.loads(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise ProposalT02Error(f"{label} must contain finite JSON-native values") from error


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()


def _file_digest(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load_policy(path: str | Path) -> dict[str, Any]:
    """Read either a policy object or a source plan containing that object."""
    value = _read(path)
    if isinstance(value, Mapping) and isinstance(value.get("frozen_subsequent_policy"), Mapping):
        value = value["frozen_subsequent_policy"]
    if not isinstance(value, Mapping) or not value:
        raise ProposalT02Error("continuation source must be a policy or a plan with frozen_subsequent_policy")
    return _json_copy(value, "continuation policy")


def proposal_context(state: Mapping[str, Any]) -> dict[str, Any]:
    """Reconstruct exactly the observable q/d context consumed by selectors."""
    q = state.get("q")
    draft = state.get("draft")
    if not isinstance(q, Mapping) or not isinstance(draft, Mapping):
        raise ProposalT02Error("proposal state requires q and draft mappings")
    context = _json_copy(q, "state.q")
    aliases = {
        "task": "goal",
        "text": "draft_text",
        "tool_calls": "draft_tool_calls",
        "token_logprobs": "draft_logprobs",
    }
    for source, target in aliases.items():
        if target not in context:
            if source in draft:
                context[target] = copy.deepcopy(draft[source])
            elif source in context:
                context[target] = copy.deepcopy(context[source])
    if "parse_ok" not in context:
        context["parse_ok"] = bool(draft.get("parse_ok", True))
    if "is_stop" not in context:
        context["is_stop"] = not bool(context.get("draft_tool_calls"))
    context.setdefault("goal", "")
    context.setdefault("raw_visible", [])
    context.setdefault("draft_text", "")
    context.setdefault("draft_tool_calls", [])
    return context


def _runtime_candidates(state: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for candidate in state["candidates"]:
        row = copy.deepcopy(dict(candidate))
        row["unit_id"] = row["candidate_id"]
        row["event_id"] = row["source_id"]
        rows.append(row)
    return rows


def _runtime_actions(state: Mapping[str, Any]) -> list[tuple[str, ...]]:
    return [tuple(action["candidate_ids"]) for action in state["allowed_actions"]]


def _origin_aliases(receipt: Mapping[str, Any], action: Sequence[str]) -> list[str]:
    wanted = list(action)
    for row in receipt["origin_aliases"]:
        if row.get("selected_ids") == wanted:
            return list(row["origins"])
    return []


def proposal_branches(state: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return A0 plus each distinct, nonempty frozen proposal exactly once."""
    actions, receipt = build_proposal_catalog(
        proposal_context(state), _runtime_candidates(state), _runtime_actions(state)
    )
    by_ids = {tuple(action["candidate_ids"]): action for action in state["allowed_actions"]}
    branches = []
    for selected in actions:
        aliases = _origin_aliases(receipt, selected)
        if not aliases:
            raise ProposalT02Error("proposal action lacks an origin alias")
        primary = aliases[0]
        branch_id = BRANCH_BY_ORIGIN[primary]
        action = by_ids.get(tuple(selected))
        if action is None:
            raise ProposalT02Error("proposal helper returned an action outside the original legal catalog")
        branches.append({
            "branch_id": branch_id,
            "action_id": action["action_id"],
            "candidate_ids": list(selected),
            "proposal_origins": aliases,
            "submit_original_draft": not selected,
            "regenerate": bool(selected),
            "selection_source": "proposal_catalog_" + "_".join(aliases),
        })
    if not branches or branches[0]["branch_id"] != "A0" or branches[0]["candidate_ids"]:
        raise ProposalT02Error("proposal catalog must begin with the empty A0 action")
    action_sets = []
    for branch in branches:
        ids = branch["candidate_ids"]
        if len(ids) != len(set(ids)):
            raise ProposalT02Error("proposal branch repeats a candidate ID")
        action_sets.append(frozenset(ids))
    if len(action_sets) != len(set(action_sets)):
        raise ProposalT02Error("proposal branches repeat an action")
    return branches, receipt


def candidate_source_contract(state: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Bind exact source spans, text/token identity, and legal action identity."""
    bound = []
    for candidate in state["candidates"]:
        bound.append({
            key: copy.deepcopy(candidate.get(key))
            for key in (
                "candidate_id", "unit_id", "source_id", "event_id", "rank", "feasible",
                "source_type", "source_indices", "provenance", "text_sha256", "text", "token_count",
            )
        })
    return bound


def observation_digest(state: Mapping[str, Any]) -> str:
    return _digest({
        "state_id": state.get("state_id"),
        "task_id": state.get("task_id"),
        "task_group_id": t02.canonical_task_group_id(state.get("task_id")),
        "decision_key": state.get("decision_key"),
        "q": state.get("q"),
        "draft": state.get("draft"),
        "candidate_sources": candidate_source_contract(state),
        "allowed_actions": state.get("allowed_actions", state.get("actions")),
    })


def _group_splits(rows: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    result: dict[str, str] = {}
    for row in rows:
        group = row.get("task_group_id")
        split = row.get("split")
        if not isinstance(group, str) or split not in {"train", "calibration"}:
            raise ProposalT02Error("every source row needs a canonical task_group_id and train/calibration split")
        previous = result.setdefault(group, split)
        if previous != split:
            raise ProposalT02Error(f"task group {group!r} crosses train/calibration splits")
    return result


def _assign_splits(states: list[dict[str, Any]], *, train_states: int, seed: int,
                   locked: Mapping[str, str]) -> None:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for state in states:
        groups[state["task_group_id"]].append(state)
    for group, rows in groups.items():
        explicit = {row.get("split") for row in rows if row.get("split") is not None}
        if len(explicit) > 1:
            raise ProposalT02Error(f"task group {group!r} has conflicting explicit splits")
        frozen = locked.get(group)
        if frozen is not None and explicit and explicit != {frozen}:
            raise ProposalT02Error(f"task group {group!r} changes its prior split")
        if frozen is not None:
            for row in rows:
                row["split"] = frozen
        elif explicit:
            for row in rows:
                row["split"] = next(iter(explicit))
    assigned_train = sum(len(rows) for rows in groups.values() if rows[0].get("split") == "train")
    unlocked = [group for group, rows in groups.items() if rows[0].get("split") is None]
    ordered = sorted(unlocked, key=lambda group: _digest([seed, group, "proposal-split"]))
    possible: dict[int, tuple[str, ...]] = {0: ()}
    for group in ordered:
        size = len(groups[group])
        for count, chosen in list(possible.items()):
            possible.setdefault(count + size, (*chosen, group))
    needed = max(0, train_states - assigned_train)
    chosen_count = min(possible, key=lambda count: (abs(count - needed), -count))
    chosen = set(possible[chosen_count])
    for group in unlocked:
        split = "train" if group in chosen else "calibration"
        for row in groups[group]:
            row["split"] = split
    _group_splits(states)


def build_plan(candidate_states: Sequence[Mapping[str, Any]], *,
               forbidden_manifest_paths: Mapping[str, str | Path],
               frozen_subsequent_policy: Mapping[str, Any], seed: int = 0,
               target_states: int = MAX_NEW_STATES, train_states: int = 40,
               split_bindings: Mapping[str, str] | None = None,
               exclude_state_ids: Sequence[str] = (),
               max_new_continuations: int = MAX_NEW_CONTINUATIONS,
               allow_partial: bool = False,
               incomplete_collection_reason: str | None = None) -> dict[str, Any]:
    """Build one H0 plan with at most 60 new states and 180 unique branches."""
    if type(target_states) is not int or not 0 < target_states <= MAX_NEW_STATES:
        raise ProposalT02Error(f"target_states must be in [1, {MAX_NEW_STATES}]")
    if type(max_new_continuations) is not int or not 0 < max_new_continuations <= MAX_NEW_CONTINUATIONS:
        raise ProposalT02Error(f"max_new_continuations must be in [1, {MAX_NEW_CONTINUATIONS}]")
    if type(train_states) is not int or not 0 <= train_states <= target_states:
        raise ProposalT02Error("train_states must be between zero and target_states")
    policy = _json_copy(frozen_subsequent_policy, "frozen subsequent policy")
    if not isinstance(policy, dict) or not policy:
        raise ProposalT02Error("frozen_subsequent_policy must be a nonempty object")
    if policy.get("history_profile", "H0") != "H0":
        raise ProposalT02Error("proposal T02 v1 is H0-only")
    forbidden, manifest_receipts = t02.load_forbidden_manifests(forbidden_manifest_paths)
    excluded = set(exclude_state_ids)
    eligible = []
    rejected = []
    seen = set()
    catalogs = {}
    for raw in candidate_states:
        state = t02._normalize_candidate_state(raw)
        state_id = state["state_id"]
        if state_id in seen:
            raise ProposalT02Error(f"duplicate state_id: {state_id}")
        seen.add(state_id)
        if state_id in excluded:
            rejected.append({"state_id": state_id, "reason": "already_reused"})
            continue
        if state["task_id"] in forbidden or state["task_group_id"] in forbidden:
            rejected.append({"state_id": state_id, "reason": "D128_F128_union_group"})
            continue
        branches, receipt = proposal_branches(state)
        if len(branches) == 1:
            rejected.append({"state_id": state_id, "reason": "no_nonempty_proposal"})
            continue
        planned = copy.deepcopy(state)
        planned["schema"] = t02.PLANNED_STATE_SCHEMA
        planned["branches"] = branches
        planned["branch_selection_uses_future_outcome"] = False
        eligible.append(planned)
        catalogs[state_id] = receipt
    eligible.sort(key=lambda row: (_digest([seed, row["task_group_id"]]), _digest([seed, row["state_id"]])))
    if len(eligible) < target_states and not allow_partial:
        raise ProposalT02Error(f"need {target_states} eligible proposal states, found {len(eligible)}")
    if not eligible:
        raise ProposalT02Error("no eligible proposal states were captured")
    selected = eligible[:target_states]
    incomplete = len(selected) < target_states
    if incomplete and (not isinstance(incomplete_collection_reason, str)
                       or not incomplete_collection_reason):
        raise ProposalT02Error("a partial plan requires an incomplete_collection_reason")
    if not incomplete and incomplete_collection_reason is not None:
        raise ProposalT02Error("a complete plan cannot carry an incomplete collection reason")
    continuation_count = sum(len(state["branches"]) for state in selected)
    if continuation_count > max_new_continuations:
        raise ProposalT02Error("selected proposal states exceed the new continuation ceiling")
    locks = dict(split_bindings or {})
    if any(value not in {"train", "calibration"} for value in locks.values()):
        raise ProposalT02Error("split_bindings values must be train or calibration")
    _assign_splits(
        selected, train_states=min(train_states, len(selected)), seed=seed, locked=locks
    )
    selected.sort(key=lambda row: (row["split"] != "train", row["task_group_id"], row["state_id"]))
    selected_ids = {row["state_id"] for row in selected}
    plan = {
        "schema": PLAN_SCHEMA,
        "proposal_protocol": PROPOSAL_PROTOCOL,
        "history_profile": "H0",
        "seed": seed,
        "new_state_count": len(selected),
        "new_state_ceiling": MAX_NEW_STATES,
        "collection": {
            "requested_state_target": target_states,
            "requested_train_state_target": train_states,
            "captured_candidate_state_count": len(candidate_states),
            "eligible_state_count": len(eligible),
            "actual_state_count": len(selected),
            "complete": not incomplete,
            "incomplete_reason": incomplete_collection_reason,
        },
        "planned_continuation_executions": continuation_count,
        "new_continuation_ceiling": max_new_continuations,
        "old_359_360_budget_affected": False,
        "source_snapshot_contract": "live_process_exact_state_only",
        "frozen_subsequent_policy": policy,
        "frozen_subsequent_policy_sha256": t02._digest(policy),
        "required_snapshot_components": list(t02.REQUIRED_COMPONENTS),
        "forbidden_manifests": manifest_receipts,
        "rejected_states": rejected,
        "split_bindings": {**locks, **_group_splits(selected)},
        "proposal_catalogs": {state_id: catalogs[state_id] for state_id in selected_ids},
        "states": selected,
    }
    check_plan(plan)
    return plan


def check_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(plan, Mapping) or plan.get("schema") != PLAN_SCHEMA:
        raise ProposalT02Error(f"plan schema must be {PLAN_SCHEMA}")
    if plan.get("proposal_protocol") != PROPOSAL_PROTOCOL or plan.get("history_profile") != "H0":
        raise ProposalT02Error("plan proposal protocol or H0 identity changed")
    states = plan.get("states")
    if not isinstance(states, list) or not states or len(states) > MAX_NEW_STATES:
        raise ProposalT02Error("plan states must be a nonempty list within the 60-state ceiling")
    if plan.get("new_state_count") != len(states):
        raise ProposalT02Error("new_state_count differs from plan states")
    collection = plan.get("collection")
    if not isinstance(collection, Mapping):
        raise ProposalT02Error("plan lacks collection target/capture accounting")
    requested = collection.get("requested_state_target")
    captured = collection.get("captured_candidate_state_count")
    eligible = collection.get("eligible_state_count")
    complete = collection.get("complete")
    reason = collection.get("incomplete_reason")
    if (type(requested) is not int or not 0 < requested <= MAX_NEW_STATES
            or type(captured) is not int or captured < len(states)
            or type(eligible) is not int or not len(states) <= eligible <= captured
            or collection.get("actual_state_count") != len(states)
            or complete is not (len(states) == requested)
            or (complete and reason is not None)
            or (not complete and (not isinstance(reason, str) or not reason))):
        raise ProposalT02Error("plan collection target/capture accounting is inconsistent")
    groups = _group_splits(states)
    state_ids = set()
    total = 0
    for state in states:
        normalized = dict(state)
        branches = normalized.pop("branches", None)
        normalized.pop("split", None)
        normalized.pop("branch_selection_uses_future_outcome", None)
        normalized["schema"] = t02.CANDIDATE_SCHEMA
        t02._normalize_candidate_state(normalized)
        state_id = state["state_id"]
        if state_id in state_ids:
            raise ProposalT02Error("plan repeats a state_id")
        state_ids.add(state_id)
        expected, receipt = proposal_branches(normalized)
        if branches != expected or plan.get("proposal_catalogs", {}).get(state_id) != receipt:
            raise ProposalT02Error("planned branches differ from the frozen proposal helper")
        sets = []
        for branch in branches:
            ids = branch["candidate_ids"]
            if len(ids) != len(set(ids)):
                raise ProposalT02Error("plan branch repeats a candidate ID")
            sets.append(frozenset(ids))
        if len(sets) != len(set(sets)) or not sets or sets[0]:
            raise ProposalT02Error("plan repeats a branch action or lacks A0")
        total += len(branches)
    if total != plan.get("planned_continuation_executions"):
        raise ProposalT02Error("planned continuation count is inconsistent")
    ceiling = plan.get("new_continuation_ceiling")
    if type(ceiling) is not int or total > ceiling or ceiling > MAX_NEW_CONTINUATIONS:
        raise ProposalT02Error("planned continuations exceed the 180-execution ceiling")
    if plan.get("old_359_360_budget_affected") is not False:
        raise ProposalT02Error("proposal T02 must not mutate the old 359/360 budget")
    policy = plan.get("frozen_subsequent_policy")
    if t02._digest(policy) != plan.get("frozen_subsequent_policy_sha256"):
        raise ProposalT02Error("frozen continuation policy digest mismatch")
    return {
        "valid": True,
        "state_count": len(states),
        "continuation_count": total,
        "split_counts": dict(Counter(state["split"] for state in states)),
        "task_group_count": len(groups),
    }


def _check_capabilities(adapter: ProposalBranchAdapter, policy: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(adapter.capabilities())
    required = {
        "schema": t02.CAPABILITY_SCHEMA,
        "exact_same_state_restore": True,
        "frozen_subsequent_policy": True,
        "official_turn_outcome": True,
        "official_task_outcome": True,
    }
    if any(value.get(key) != expected for key, expected in required.items()):
        raise t02.T02CapabilityError("adapter cannot prove exact H0 same-state continuation")
    if set(value.get("components", ())) != set(t02.REQUIRED_COMPONENTS):
        raise t02.T02CapabilityError("adapter lacks required exact snapshot components")
    return value


def execute_plan(plan: Mapping[str, Any], adapter: ProposalBranchAdapter, *,
                 state_ids: Sequence[str] | None = None,
                 on_progress: Callable[[Mapping[str, Any]], None] | None = None,
                 on_checkpoint: Callable[[Mapping[str, Any]], None] | None = None) -> dict[str, Any]:
    """Execute each distinct proposal once from an exact live H0 snapshot."""
    check_plan(plan)
    policy = plan["frozen_subsequent_policy"]
    capabilities = _check_capabilities(adapter, policy)
    requested = None if state_ids is None else set(state_ids)
    planned_ids = {state["state_id"] for state in plan["states"]}
    if requested is not None and not requested.issubset(planned_ids):
        raise ProposalT02Error("execution requested a state outside the plan")
    results = []
    snapshots = []
    executed = []

    def checkpoint(status: str) -> dict[str, Any]:
        value = {
            "schema": RESULT_SET_SCHEMA,
            "status": status,
            "proposal_protocol": PROPOSAL_PROTOCOL,
            "plan_sha256": _digest(plan),
            "adapter_capabilities": capabilities,
            "expected_continuation_executions": sum(
                len(state["branches"]) for state in plan["states"]
                if requested is None or state["state_id"] in requested
            ),
            "complete_continuation_executions": len(results),
            "continuation_ceiling": plan["new_continuation_ceiling"],
            "snapshots": copy.deepcopy(snapshots),
            "results": copy.deepcopy(results),
            "executed_state_ids": list(executed),
        }
        if on_checkpoint is not None:
            on_checkpoint(value)
        return value

    checkpoint("in_progress")
    for state in plan["states"]:
        if requested is not None and state["state_id"] not in requested:
            continue
        snapshot = t02._check_snapshot(
            adapter.capture_state(state, frozen_policy=policy),
            state_id=state["state_id"], policy_sha256=plan["frozen_subsequent_policy_sha256"],
        )
        snapshots.append(snapshot)
        checkpoint("in_progress")
        for branch in state["branches"]:
            restore = t02._check_restore(
                adapter.restore_state(snapshot, frozen_policy=policy), snapshot=snapshot
            )
            result = dict(adapter.run_branch(
                state, branch, frozen_policy=policy, restore_receipt=restore,
            ))
            if result.get("schema") != t02.RESULT_SCHEMA or result.get("state_id") != state["state_id"]:
                raise ProposalT02Error("adapter returned an invalid proposal result row")
            if result.get("branch_id") != branch["branch_id"] or result.get("candidate_ids") != branch["candidate_ids"]:
                raise ProposalT02Error("adapter result changed branch action identity")
            if result.get("execution_status") != "complete":
                raise ProposalT02Error("proposal branch did not complete")
            execution = result.get("execution_receipt")
            expected_execution = {
                "submitted_original_draft": branch["submit_original_draft"],
                "recovery_candidate_ids": branch["candidate_ids"],
                "regeneration_count": 1 if branch["regenerate"] else 0,
                "continued_with_frozen_policy": True,
                "observations_replayed_from_other_branch": False,
            }
            if execution != expected_execution:
                raise ProposalT02Error("proposal result execution receipt differs from branch contract")
            t02._check_official_outcome(result.get("official_outcome"))
            results.append(_json_copy(result, "proposal result"))
            checkpoint("in_progress")
            if on_progress is not None:
                on_progress({"state_id": state["state_id"], "branch_id": branch["branch_id"],
                             "complete_continuations": len(results)})
        finish = getattr(adapter, "finish_state", None)
        if callable(finish):
            finish(state["state_id"], completed_branches=len(state["branches"]))
        elif len(state["branches"]) != len(t02.BRANCH_IDS):
            raise t02.T02CapabilityError("variable-branch proposal execution requires adapter.finish_state")
        executed.append(state["state_id"])
        checkpoint("in_progress")
    return checkpoint("completed")


def _outcome(indexed: Mapping[tuple[str, str], Mapping[str, Any]], state_id: str,
             branch_id: str) -> dict[str, Any]:
    row = indexed.get((state_id, branch_id))
    if row is None:
        return {"execution_status": "missing", "turn_success": None,
                "task_success": None, "official_outcome": None}
    official = t02._check_official_outcome(row["official_outcome"])
    return {"execution_status": row["execution_status"],
            "turn_success": official["turn_success"], "task_success": official["task_success"],
            "official_outcome": official}


def _index_results(plan: Mapping[str, Any], results: Mapping[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    """Validate result/snapshot/restore/action bindings before deriving labels."""
    if not isinstance(results, Mapping) or results.get("schema") != RESULT_SET_SCHEMA:
        raise ProposalT02Error(f"results schema must be {RESULT_SET_SCHEMA}")
    if results.get("proposal_protocol") != PROPOSAL_PROTOCOL or results.get("plan_sha256") != _digest(plan):
        raise ProposalT02Error("results do not bind the proposal plan and protocol")
    rows = results.get("results")
    snapshots = results.get("snapshots")
    if (not isinstance(rows, list) or not isinstance(snapshots, list)
            or results.get("complete_continuation_executions") != len(rows)):
        raise ProposalT02Error("proposal result or snapshot count is inconsistent")
    expected = {
        (state["state_id"], branch["branch_id"]): (state, branch)
        for state in plan["states"] for branch in state["branches"]
    }
    if len(rows) != len(expected):
        raise ProposalT02Error("proposal results are partial or contain extra executions")
    snapshots_by_state = {}
    for snapshot in snapshots:
        state_id = snapshot.get("state_id") if isinstance(snapshot, Mapping) else None
        state = next((row for row in plan["states"] if row["state_id"] == state_id), None)
        if state is None or state_id in snapshots_by_state:
            raise ProposalT02Error("proposal snapshots repeat or target an out-of-plan state")
        snapshots_by_state[state_id] = t02._check_snapshot(
            snapshot, state_id=state_id,
            policy_sha256=plan["frozen_subsequent_policy_sha256"],
        )
    if set(snapshots_by_state) != {state["state_id"] for state in plan["states"]}:
        raise ProposalT02Error("proposal results lack one exact snapshot per planned state")
    indexed = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ProposalT02Error("proposal result row must be an object")
        key = (row.get("state_id"), row.get("branch_id"))
        if key not in expected or key in indexed:
            raise ProposalT02Error("proposal result repeats or targets an out-of-plan branch")
        _, branch = expected[key]
        snapshot = snapshots_by_state[key[0]]
        if (row.get("schema") != t02.RESULT_SCHEMA
                or row.get("candidate_ids") != branch["candidate_ids"]
                or row.get("snapshot_id") != snapshot["snapshot_id"]
                or row.get("frozen_policy_sha256") != plan["frozen_subsequent_policy_sha256"]
                or not t02._valid_sha256(row.get("restore_receipt_sha256"))
                or row.get("execution_status") != "complete"):
            raise ProposalT02Error("proposal result changed snapshot, policy, action, or completion identity")
        expected_execution = {
            "submitted_original_draft": branch["submit_original_draft"],
            "recovery_candidate_ids": branch["candidate_ids"],
            "regeneration_count": 1 if branch["regenerate"] else 0,
            "continued_with_frozen_policy": True,
            "observations_replayed_from_other_branch": False,
        }
        if row.get("execution_receipt") != expected_execution:
            raise ProposalT02Error("proposal result execution receipt differs from the plan")
        t02._check_official_outcome(row.get("official_outcome"))
        indexed[key] = _json_copy(row, "proposal result")
    return indexed


def _labeled_row(state: Mapping[str, Any], outcomes: Mapping[str, Mapping[str, Any]], *,
                 source: Mapping[str, Any], policy_sha256: str,
                 catalogs: Mapping[str, Any]) -> dict[str, Any]:
    baseline = outcomes["A0"]
    actions = []
    for branch in state["branches"]:
        if not branch["candidate_ids"]:
            continue
        outcome = outcomes[branch["branch_id"]]
        delta_turn, turn_status = t02._delta(outcome["turn_success"], baseline["turn_success"])
        delta_task, task_status = t02._delta(outcome["task_success"], baseline["task_success"])
        actions.append({
            "branch_id": branch["branch_id"],
            "candidate_ids": copy.deepcopy(branch["candidate_ids"]),
            "proposal_origins": copy.deepcopy(branch["proposal_origins"]),
            "delta_turn": delta_turn,
            "delta_task": delta_task,
            "turn_label_status": turn_status,
            "task_label_status": task_status,
            "execution_status": outcome["execution_status"],
        })
    return {
        "schema": LABELED_STATE_SCHEMA,
        "proposal_protocol": PROPOSAL_PROTOCOL,
        "history_profile": "H0",
        "state_id": state["state_id"],
        "benchmark": state["benchmark"],
        "task_id": state["task_id"],
        "task_group_id": state["task_group_id"],
        "decision_key": state["decision_key"],
        "split": state["split"],
        "q": copy.deepcopy(state["q"]),
        "draft": copy.deepcopy(state["draft"]),
        "draft_kind": state["draft_kind"],
        "candidates": copy.deepcopy(state["candidates"]),
        "allowed_actions": copy.deepcopy(state["allowed_actions"]),
        "proposal_catalog": copy.deepcopy(catalogs),
        "proposal_actions": actions,
        "outcomes": copy.deepcopy(dict(outcomes)),
        "continuation_policy_sha256": policy_sha256,
        "observation_sha256": observation_digest(state),
        "statistical_unit": "state",
        "branches_are_independent_states": False,
        "provenance": copy.deepcopy(dict(source)),
    }


def label_results(plan: Mapping[str, Any], results: Mapping[str, Any]) -> dict[str, Any]:
    check_plan(plan)
    indexed = _index_results(plan, results)
    labeled = []
    for state in plan["states"]:
        outcomes = {branch["branch_id"]: _outcome(indexed, state["state_id"], branch["branch_id"])
                    for branch in state["branches"]}
        labeled.append(_labeled_row(
            state, outcomes,
            source={"mode": "new_exact_execution", "plan_sha256": _digest(plan)},
            policy_sha256=plan["frozen_subsequent_policy_sha256"],
            catalogs=plan["proposal_catalogs"][state["state_id"]],
        ))
    artifact = {
        "schema": LABELED_SET_SCHEMA,
        "proposal_protocol": PROPOSAL_PROTOCOL,
        "history_profile": "H0",
        "continuation_policy_sha256": plan["frozen_subsequent_policy_sha256"],
        "state_count": len(labeled),
        "action_example_count": sum(len(row["proposal_actions"]) for row in labeled),
        "rows": labeled,
        "partitions": [{"mode": "new_exact_execution", "plan_sha256": _digest(plan),
                        "state_count": len(labeled)}],
    }
    check_labeled(artifact)
    return artifact


def create_source_binding(training_package: str | Path) -> dict[str, Any]:
    """Bind the fully audited prepared_v8 label/plan/ledger lineage.

    ``evidence_eval_trained`` already verifies every combined row against its
    preserved or per-task plan/result/label files.  Reusing that verifier avoids
    treating an arbitrary labels/policy pair as evidence of common origin.
    """
    import evidence_eval_trained

    training = Path(training_package).resolve()
    labels = evidence_eval_trained._verify_combined_labels(training)
    paths = {
        name: training / relative for name, relative in {
            "labels": "run/labels.json",
            "summary": "run/summary.json",
            "source_plan": "recovery/source_plan.json",
            "recovery_contract": "recovery/recovery_contract.json",
            "ledger": "ledger.json",
        }.items()
    }
    source_plan = _read(paths["source_plan"])
    policy = load_policy(paths["source_plan"])
    return {
        "schema": SOURCE_BINDING_SCHEMA,
        "training_package": str(training),
        "files": {name: {"path": str(path), "sha256": _file_digest(path)}
                  for name, path in paths.items()},
        "labels_canonical_sha256": _digest(labels),
        "source_plan_canonical_sha256": _digest(source_plan),
        "continuation_policy_sha256": _digest(policy),
        "snapshot_reuse": "labels_only_live_actor_backend_state_not_persisted",
        "rows_are_reusable_only_for_actually_tested_exact_action": True,
    }


def _verify_source_binding(binding: Mapping[str, Any], training_package: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(binding, Mapping) or binding.get("schema") != SOURCE_BINDING_SCHEMA:
        raise ProposalT02Error(f"source binding schema must be {SOURCE_BINDING_SCHEMA}")
    expected = create_source_binding(training_package)
    if binding != expected:
        raise ProposalT02Error("source binding differs from the audited prepared_v8 lineage")
    labels = _read(training_package / "run/labels.json")
    policy = load_policy(training_package / "recovery/source_plan.json")
    return labels, policy


def _candidate_state_from_row(row: Mapping[str, Any]) -> dict[str, Any]:
    value = {
        key: copy.deepcopy(row[key]) for key in (
            "state_id", "benchmark", "task_id", "task_group_id", "decision_key",
            "q", "draft", "draft_kind", "candidates", "allowed_actions",
        ) if key in row
    }
    value.update(schema=t02.CANDIDATE_SCHEMA,
                 risk_bucket=row.get("risk_bucket", "unknown"),
                 previous_turn_valid=row.get("previous_turn_valid"))
    return t02._normalize_candidate_state(value)


def _old_branch_index(row: Mapping[str, Any]) -> dict[tuple[str, ...], tuple[Mapping[str, Any], Mapping[str, Any]]]:
    branches, outcomes = row.get("branches"), row.get("outcomes")
    if not isinstance(branches, list) or not isinstance(outcomes, Mapping):
        raise ProposalT02Error("old row lacks branches or outcomes")
    index = {}
    for branch in branches:
        if not isinstance(branch, Mapping) or not isinstance(branch.get("candidate_ids"), list):
            raise ProposalT02Error("old row contains an invalid branch")
        outcome = outcomes.get(branch.get("branch_id"))
        if not isinstance(outcome, Mapping):
            raise ProposalT02Error("old row lacks a branch outcome")
        key = tuple(branch["candidate_ids"])
        if key in index:
            raise ProposalT02Error("old row repeats an executed action")
        index[key] = (branch, outcome)
    return index


def reuse_old_labels(labels: Mapping[str, Any], *, binding: Mapping[str, Any],
                     source_policy: Mapping[str, Any], target_policy: Mapping[str, Any],
                     forbidden_manifest_paths: Mapping[str, str | Path],
                     target_states: Sequence[Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """Reuse only exact H0 rows whose complete proposal actions were really tested."""
    if _digest(source_policy) != _digest(target_policy):
        raise ProposalT02Error("source and target regeneration/continuation contracts differ")
    if target_policy.get("history_profile", "H0") != "H0":
        raise ProposalT02Error("proposal T02 reuse is H0-only")
    rows = labels.get("rows") if isinstance(labels, Mapping) else None
    if not isinstance(rows, list):
        raise ProposalT02Error("old labels must contain rows")
    _group_splits(rows)
    forbidden, receipts = t02.load_forbidden_manifests(forbidden_manifest_paths)
    targets = None
    if target_states is not None:
        targets = {}
        for raw in target_states:
            state = t02._normalize_candidate_state(raw)
            if state["state_id"] in targets:
                raise ProposalT02Error("target states repeat a state_id")
            targets[state["state_id"]] = state
    reused, rejected = [], []
    for old in rows:
        try:
            source_state = _candidate_state_from_row(old)
            target = source_state if targets is None else targets.get(source_state["state_id"])
            if target is None:
                raise ProposalT02Error("no matching target state")
            if observation_digest(source_state) != observation_digest(target):
                raise ProposalT02Error("q/d snapshot or exact source span/token identity changed")
            if target["task_id"] in forbidden or target["task_group_id"] in forbidden:
                raise ProposalT02Error("state belongs to the D128/F128 union group")
            branches, catalog = proposal_branches(target)
            old_index = _old_branch_index(old)
            matched_outcomes = {}
            old_branch_ids = {}
            for branch in branches:
                pair = old_index.get(tuple(branch["candidate_ids"]))
                if pair is None:
                    raise ProposalT02Error("proposal action was not actually tested in the old snapshot")
                old_branch, outcome = pair
                if outcome.get("execution_status") != "complete" or not isinstance(outcome.get("official_outcome"), Mapping):
                    raise ProposalT02Error("matched old action lacks a complete official outcome")
                official = t02._check_official_outcome(outcome["official_outcome"])
                if (outcome.get("turn_success") != official["turn_success"]
                        or outcome.get("task_success") != official["task_success"]):
                    raise ProposalT02Error("old projected outcome differs from its official artifact record")
                if bool(branch["candidate_ids"]) != bool(old_branch.get("regenerate")):
                    raise ProposalT02Error("matched old action has a different regeneration contract")
                matched_outcomes[branch["branch_id"]] = copy.deepcopy(outcome)
                old_branch_ids[branch["branch_id"]] = old_branch["branch_id"]
            planned = copy.deepcopy(target)
            planned["schema"] = t02.PLANNED_STATE_SCHEMA
            planned["branches"] = branches
            planned["split"] = old["split"]
            planned["branch_selection_uses_future_outcome"] = False
            reused.append(_labeled_row(
                planned, matched_outcomes,
                source={
                    "mode": "strict_old_action_reuse",
                    "source_labels_sha256": binding["files"]["labels"]["sha256"],
                    "source_labels_canonical_sha256": binding["labels_canonical_sha256"],
                    "source_state_id": old["state_id"],
                    "matched_old_branch_ids": old_branch_ids,
                    "reuse_checks": [
                        "q_d_observation_digest", "exact_candidate_source_spans_and_tokens",
                        "exact_ordered_action_identity", "complete_official_outcome",
                        "regeneration_contract", "frozen_H0_continuation_contract",
                    ],
                },
                policy_sha256=_digest(target_policy), catalogs=catalog,
            ))
        except (ProposalT02Error, t02.T02Error, KeyError, TypeError) as error:
            rejected.append({"state_id": old.get("state_id"), "reason": str(error)})
    artifact = {
        "schema": LABELED_SET_SCHEMA,
        "proposal_protocol": PROPOSAL_PROTOCOL,
        "history_profile": "H0",
        "continuation_policy_sha256": _digest(target_policy),
        "state_count": len(reused),
        "action_example_count": sum(len(row["proposal_actions"]) for row in reused),
        "rows": reused,
        "partitions": [{"mode": "strict_old_action_reuse", "state_count": len(reused),
                        "source_labels_sha256": binding["files"]["labels"]["sha256"]}],
        "rejected": rejected,
        "forbidden_manifests": receipts,
        "new_state_count": 0,
        "new_continuation_executions": 0,
        "old_359_360_budget_affected": False,
    }
    check_labeled(artifact)
    return artifact


def check_labeled(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or value.get("schema") != LABELED_SET_SCHEMA:
        raise ProposalT02Error(f"labeled schema must be {LABELED_SET_SCHEMA}")
    if value.get("proposal_protocol") != PROPOSAL_PROTOCOL or value.get("history_profile") != "H0":
        raise ProposalT02Error("labeled proposal protocol or H0 identity changed")
    rows = value.get("rows")
    if not isinstance(rows, list) or value.get("state_count") != len(rows):
        raise ProposalT02Error("labeled state_count is inconsistent")
    _group_splits(rows) if rows else None
    state_ids = set()
    examples = 0
    for row in rows:
        if row.get("schema") != LABELED_STATE_SCHEMA or row.get("proposal_protocol") != PROPOSAL_PROTOCOL:
            raise ProposalT02Error("invalid proposal labeled row")
        state_id = row.get("state_id")
        if not isinstance(state_id, str) or not state_id or state_id in state_ids:
            raise ProposalT02Error("proposal labeled rows repeat or omit state_id")
        state_ids.add(state_id)
        if row.get("observation_sha256") != observation_digest(row):
            raise ProposalT02Error("labeled row q/d/source observation digest changed")
        if row.get("continuation_policy_sha256") != value.get("continuation_policy_sha256"):
            raise ProposalT02Error("labeled row continuation contract differs from its dataset")
        expected_branches, expected_catalog = proposal_branches(row)
        if row.get("proposal_catalog") != expected_catalog:
            raise ProposalT02Error("labeled row proposal catalog differs from the frozen helper")
        expected_nonempty = {
            frozenset(branch["candidate_ids"]): branch
            for branch in expected_branches if branch["candidate_ids"]
        }
        actions = row.get("proposal_actions")
        if (not isinstance(actions, list) or not actions
                or len(actions) != len(expected_nonempty)):
            raise ProposalT02Error("labeled row needs at least one tested nonempty proposal")
        outcomes = row.get("outcomes")
        if not isinstance(outcomes, Mapping):
            raise ProposalT02Error("labeled row lacks bound branch outcomes")
        if set(outcomes) != {branch["branch_id"] for branch in expected_branches}:
            raise ProposalT02Error("labeled row outcomes differ from proposal branches")
        baseline = outcomes.get("A0")
        if not isinstance(baseline, Mapping) or baseline.get("execution_status") != "complete":
            raise ProposalT02Error("labeled row lacks a complete A0 baseline")
        baseline_official = t02._check_official_outcome(baseline.get("official_outcome"))
        if (baseline.get("turn_success") != baseline_official["turn_success"]
                or baseline.get("task_success") != baseline_official["task_success"]):
            raise ProposalT02Error("A0 projected outcome differs from its official record")
        selected = set()
        for action in actions:
            ids = tuple(action.get("candidate_ids", ()))
            canonical_ids = frozenset(ids)
            if not ids or len(ids) != len(canonical_ids) or canonical_ids in selected:
                raise ProposalT02Error("proposal labels contain an empty or duplicate scored action")
            selected.add(canonical_ids)
            expected = expected_nonempty.get(canonical_ids)
            if (expected is None or list(ids) != expected["candidate_ids"]
                    or action.get("branch_id") != expected["branch_id"]
                    or action.get("proposal_origins") != expected["proposal_origins"]):
                raise ProposalT02Error("scored action differs from the frozen proposal identity")
            if action.get("execution_status") != "complete":
                raise ProposalT02Error("proposal label lacks origin or complete execution")
            outcome = outcomes.get(expected["branch_id"])
            if not isinstance(outcome, Mapping) or outcome.get("execution_status") != "complete":
                raise ProposalT02Error("proposal label lacks a complete bound outcome")
            official = t02._check_official_outcome(outcome.get("official_outcome"))
            if (outcome.get("turn_success") != official["turn_success"]
                    or outcome.get("task_success") != official["task_success"]):
                raise ProposalT02Error("proposal projected outcome differs from its official record")
            for target in ("turn", "task"):
                label = action.get(f"delta_{target}")
                status = action.get(f"{target}_label_status")
                if label not in {-1, 0, 1, None} or status != ("unknown" if label is None else "known"):
                    raise ProposalT02Error("proposal delta/status mismatch")
                expected_label, expected_status = t02._delta(
                    outcome[f"{target}_success"], baseline[f"{target}_success"]
                )
                if label != expected_label or status != expected_status:
                    raise ProposalT02Error("proposal delta differs from its bound outcome")
        examples += len(actions)
    if value.get("action_example_count") != examples:
        raise ProposalT02Error("action_example_count is inconsistent")
    return {"valid": True, "state_count": len(rows), "action_example_count": examples,
            "split_counts": dict(Counter(row["split"] for row in rows))}


def merge_labeled(parts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not parts:
        raise ProposalT02Error("merge requires at least one labeled artifact")
    rows, partitions = [], []
    policy = None
    state_ids = set()
    for part in parts:
        check_labeled(part)
        current = part["continuation_policy_sha256"]
        policy = current if policy is None else policy
        if current != policy:
            raise ProposalT02Error("merged proposal datasets use different continuation contracts")
        for row in part["rows"]:
            if row["state_id"] in state_ids:
                raise ProposalT02Error("merge would duplicate one snapshot state")
            state_ids.add(row["state_id"])
            rows.append(copy.deepcopy(row))
        partitions.extend(copy.deepcopy(part.get("partitions", [])))
    _group_splits(rows)
    artifact = {
        "schema": LABELED_SET_SCHEMA,
        "proposal_protocol": PROPOSAL_PROTOCOL,
        "history_profile": "H0",
        "continuation_policy_sha256": policy,
        "state_count": len(rows),
        "action_example_count": sum(len(row["proposal_actions"]) for row in rows),
        "rows": rows,
        "partitions": partitions,
    }
    check_labeled(artifact)
    return artifact


def _manifest_args(values: Sequence[str]) -> dict[str, Path]:
    result = {}
    for value in values:
        name, separator, path = value.partition("=")
        if not separator or name not in t02.FORBIDDEN_MANIFEST_NAMES:
            raise ProposalT02Error("forbidden manifests must be D128=<path> and F128=<path>")
        result[name] = Path(path)
    return result


def _rows(value: Any) -> list[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        value = value.get("states", value.get("rows"))
    if not isinstance(value, list):
        raise ProposalT02Error("artifact must contain a states/rows list")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    bind = commands.add_parser("bind-source")
    bind.add_argument("--training-package", type=Path, required=True)
    bind.add_argument("--output", type=Path, required=True)
    reuse = commands.add_parser("reuse")
    reuse.add_argument("--training-package", type=Path, required=True)
    reuse.add_argument("--source-binding", type=Path, required=True)
    reuse_target = reuse.add_mutually_exclusive_group(required=True)
    reuse_target.add_argument("--target-policy", type=Path)
    reuse_target.add_argument("--target-plan", type=Path)
    reuse.add_argument("--target-states", type=Path)
    reuse.add_argument("--forbidden-manifest", action="append", default=[], required=True)
    reuse.add_argument("--output", type=Path, required=True)
    plan = commands.add_parser("plan")
    plan.add_argument("--states", type=Path, required=True)
    plan_policy = plan.add_mutually_exclusive_group(required=True)
    plan_policy.add_argument("--target-policy", type=Path)
    plan_policy.add_argument("--target-plan", type=Path)
    plan.add_argument("--split-bindings", type=Path)
    plan.add_argument("--reuse-labels", type=Path)
    plan.add_argument("--forbidden-manifest", action="append", default=[], required=True)
    plan.add_argument("--target-states", type=int, default=MAX_NEW_STATES)
    plan.add_argument("--train-states", type=int, default=40)
    plan.add_argument("--max-new-continuations", type=int, default=MAX_NEW_CONTINUATIONS)
    plan.add_argument("--seed", type=int, default=0)
    plan.add_argument("--output", type=Path, required=True)
    label = commands.add_parser("label")
    label.add_argument("--plan", type=Path, required=True)
    label.add_argument("--results", type=Path, required=True)
    label.add_argument("--output", type=Path, required=True)
    merge = commands.add_parser("merge")
    merge.add_argument("--input", type=Path, action="append", required=True)
    merge.add_argument("--output", type=Path, required=True)
    check = commands.add_parser("check")
    check.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "bind-source":
        value = create_source_binding(args.training_package)
    elif args.command == "reuse":
        binding = _read(args.source_binding)
        target_source = args.target_policy or args.target_plan
        labels, source_policy = _verify_source_binding(
            binding, args.training_package.resolve()
        )
        targets = None if args.target_states is None else _rows(_read(args.target_states))
        value = reuse_old_labels(
            labels, binding=binding, source_policy=source_policy,
            target_policy=load_policy(target_source),
            forbidden_manifest_paths=_manifest_args(args.forbidden_manifest),
            target_states=targets,
        )
    elif args.command == "plan":
        locks = {} if args.split_bindings is None else _read(args.split_bindings)
        if isinstance(locks, Mapping) and "group_split_bindings" in locks:
            locks = locks["group_split_bindings"]
        reused = [] if args.reuse_labels is None else [row["state_id"] for row in _read(args.reuse_labels)["rows"]]
        value = build_plan(
            _rows(_read(args.states)),
            forbidden_manifest_paths=_manifest_args(args.forbidden_manifest),
            frozen_subsequent_policy=load_policy(args.target_policy or args.target_plan), seed=args.seed,
            target_states=args.target_states, train_states=args.train_states,
            split_bindings=locks, exclude_state_ids=reused,
            max_new_continuations=args.max_new_continuations,
        )
    elif args.command == "label":
        value = label_results(_read(args.plan), _read(args.results))
    elif args.command == "merge":
        value = merge_labeled([_read(path) for path in args.input])
    else:
        artifact = _read(args.artifact)
        value = check_plan(artifact) if artifact.get("schema") == PLAN_SCHEMA else check_labeled(artifact)
        print(json.dumps(value, ensure_ascii=False))
        return 0
    _write(args.output, value)
    print(json.dumps({"schema": value.get("schema"), **(
        check_plan(value) if value.get("schema") == PLAN_SCHEMA else
        check_labeled(value) if value.get("schema") == LABELED_SET_SCHEMA else
        {"valid": True}
    )}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
