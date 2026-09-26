"""Plan, execute, and label bounded T02 same-state branch collections.

Planning and labeling are offline.  Real execution is deliberately available
only through an adapter that can prove exact restoration of every required
state component.  BFCL frozen-prefix replay is useful provenance, but it is not
an actor/backend snapshot and is never accepted as that proof.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol


CANDIDATE_SCHEMA = "t02-state-candidate-v1"
PLAN_SCHEMA = "t02-collection-plan-v1"
PLANNED_STATE_SCHEMA = "t02-planned-state-v1"
RESULT_SET_SCHEMA = "t02-branch-results-v1"
RESULT_SCHEMA = "t02-branch-result-v1"
LABELED_STATE_SCHEMA = "t02-labeled-state-v1"
LABELED_SET_SCHEMA = "t02-labeled-dataset-v1"
CAPABILITY_SCHEMA = "t02-branch-adapter-capabilities-v1"
SNAPSHOT_SCHEMA = "t02-exact-snapshot-v1"
RESTORE_SCHEMA = "t02-exact-restore-v1"
PREFIX_RECEIPT_SCHEMA = "t02-bfcl-prefix-capability-v1"
RUNTIME_CAPABILITY_SCHEMA = "t02-runtime-capability-report-v1"

BRANCH_IDS = ("A0", "A1", "A2")
REQUIRED_COMPONENTS = (
    "environment",
    "actor_kv",
    "actor_positions",
    "backend_stats",
    "rng",
)
FORBIDDEN_MANIFEST_NAMES = ("D128", "F128")
FUTURE_LABEL_KEYS = frozenset(
    {
        "turn_success",
        "task_success",
        "official_outcome",
        "delta_turn",
        "delta_task",
        "future_success",
        "gold_success",
        "checker_result",
    }
)
SHA256_LENGTH = 64
A2_FALLBACK_POOL_LIMIT = 8
_BFCL_PAIRED_TASK = re.compile(
    r"^multi_turn_(?:base|long_context|miss_func|miss_param)_(\d+)$"
)


class T02Error(ValueError):
    """A T02 artifact violates the collection contract."""


class T02CapabilityError(RuntimeError):
    """A real branch executor cannot satisfy exact restoration."""


def canonical_task_group_id(task_id: str) -> str:
    """Use one exposure/split unit for all official variants of a BFCL item."""

    match = _BFCL_PAIRED_TASK.fullmatch(task_id) if isinstance(task_id, str) else None
    return f"bfcl_pair_{match.group(1)}" if match is not None else task_id


class BranchAdapter(Protocol):
    """Minimal seam for real same-state collection; no runtime is implied."""

    def capabilities(self) -> Mapping[str, Any]: ...

    def capture_state(
        self, state: Mapping[str, Any], *, frozen_policy: Mapping[str, Any]
    ) -> Mapping[str, Any]: ...

    def restore_state(
        self, snapshot: Mapping[str, Any], *, frozen_policy: Mapping[str, Any]
    ) -> Mapping[str, Any]: ...

    def run_branch(
        self,
        state: Mapping[str, Any],
        branch: Mapping[str, Any],
        *,
        frozen_policy: Mapping[str, Any],
        restore_receipt: Mapping[str, Any],
    ) -> Mapping[str, Any]: ...


def _json_copy(value: Any, *, label: str = "value") -> Any:
    try:
        return json.loads(
            json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)
        )
    except (TypeError, ValueError) as error:
        raise T02Error(f"{label} must contain finite JSON-native values") from error


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise T02Error(f"{label} must be a nonempty string")
    return value


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == SHA256_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _assert_no_future_labels(value: Any, *, path: str = "state") -> None:
    if isinstance(value, Mapping):
        leaked = sorted(FUTURE_LABEL_KEYS.intersection(value))
        if leaked:
            raise T02Error(
                f"{path} contains future outcome fields: {', '.join(leaked)}"
            )
        for key, item in value.items():
            _assert_no_future_labels(item, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_no_future_labels(item, path=f"{path}[{index}]")


def _normalize_candidate_state(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise T02Error("candidate state must be an object")
    state = _json_copy(value, label="candidate state")
    if state.get("schema") != CANDIDATE_SCHEMA:
        raise T02Error(f"candidate state schema must be {CANDIDATE_SCHEMA}")
    _assert_no_future_labels(state)
    for name in (
        "state_id",
        "benchmark",
        "task_id",
        "task_group_id",
        "decision_key",
    ):
        _nonempty_string(state.get(name), name)
    if state["benchmark"] != "bfcl":
        raise T02Error("initial T02 collection accepts only official BFCL states")
    state["task_group_id"] = canonical_task_group_id(state["task_id"])
    draft_kind = state.get("draft_kind")
    if isinstance(draft_kind, str):
        draft_kind = draft_kind.lower()
    if draft_kind == "calls":
        draft_kind = "call"
    state["draft_kind"] = draft_kind
    if draft_kind not in {"call", "stop"}:
        raise T02Error("draft_kind must be call or stop")
    if state.get("risk_bucket") not in {"high", "low", "unknown"}:
        raise T02Error("risk_bucket must be high, low, or unknown")
    previous = state.get("previous_turn_valid")
    if previous is not None and type(previous) is not bool:
        raise T02Error("previous_turn_valid must be true, false, or null")
    if "q" not in state or "draft" not in state:
        raise T02Error("candidate state needs full q and draft values")

    candidates = state.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise T02Error("candidates must be a nonempty list")
    by_id: dict[str, dict[str, Any]] = {}
    ranks: set[int] = set()
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, Mapping):
            raise T02Error(f"candidates[{index}] must be an object")
        candidate_id = candidate.get("candidate_id", candidate.get("unit_id"))
        source_id = candidate.get("source_id", candidate.get("event_id"))
        _nonempty_string(candidate_id, f"candidates[{index}].candidate_id")
        _nonempty_string(source_id, f"candidates[{index}].source_id")
        rank = candidate.get("rank")
        if type(rank) is not int or rank < 0:
            raise T02Error(f"candidates[{index}].rank must be a nonnegative integer")
        if candidate_id in by_id or rank in ranks:
            raise T02Error("candidate_id and candidate rank must be unique")
        if type(candidate.get("feasible")) is not bool:
            raise T02Error(f"candidates[{index}].feasible must be boolean")
        normalized = dict(candidate)
        normalized["candidate_id"] = candidate_id
        normalized["source_id"] = source_id
        by_id[candidate_id] = normalized
        ranks.add(rank)
    state["candidates"] = sorted(by_id.values(), key=lambda row: row["rank"])

    actions = state.get("allowed_actions", state.get("actions"))
    if not isinstance(actions, list) or not actions:
        raise T02Error("allowed_actions must be a nonempty list")
    normalized_actions: list[dict[str, Any]] = []
    action_ids: set[str | int] = set()
    action_sets: set[tuple[str, ...]] = set()
    for index, action in enumerate(actions):
        if not isinstance(action, Mapping):
            raise T02Error(f"allowed_actions[{index}] must be an object")
        action_id = action.get("action_id")
        if isinstance(action_id, bool) or not isinstance(action_id, (str, int)):
            raise T02Error(
                f"allowed_actions[{index}].action_id must be a string or integer"
            )
        if isinstance(action_id, str) and not action_id:
            raise T02Error(f"allowed_actions[{index}].action_id cannot be empty")
        if isinstance(action_id, int) and action_id < 0:
            raise T02Error(f"allowed_actions[{index}].action_id cannot be negative")
        candidate_ids = action.get("candidate_ids")
        if not isinstance(candidate_ids, list) or any(
            not isinstance(item, str) or not item for item in candidate_ids
        ):
            raise T02Error(
                f"allowed_actions[{index}].candidate_ids must be a string list"
            )
        if len(candidate_ids) != len(set(candidate_ids)):
            raise T02Error("one allowed action cannot repeat a candidate")
        missing = sorted(set(candidate_ids) - set(by_id))
        if missing:
            raise T02Error(f"allowed action references unknown candidates: {missing}")
        infeasible = [item for item in candidate_ids if not by_id[item]["feasible"]]
        if infeasible:
            raise T02Error(f"allowed action contains infeasible candidates: {infeasible}")
        canonical_set = tuple(sorted(candidate_ids))
        if action_id in action_ids or canonical_set in action_sets:
            raise T02Error("action_id and candidate set must be unique")
        action_ids.add(action_id)
        action_sets.add(canonical_set)
        normalized = dict(action)
        normalized["candidate_ids"] = list(candidate_ids)
        normalized_actions.append(normalized)
    if () not in action_sets:
        raise T02Error("allowed_actions must contain the empty no-recovery action")
    state["allowed_actions"] = normalized_actions
    state.pop("actions", None)

    local_ids = state.get("local_llm_selected_ids")
    if local_ids is not None:
        if state.get("set_selector") != "local_llm":
            proposal = state.get("local_llm_proposal")
            if (not isinstance(proposal, dict) or proposal.get("selector") != "local_llm"
                    or proposal.get("candidate_ids") != local_ids):
                raise T02Error("local_llm_selected_ids requires a bound local_llm proposal")
        if not isinstance(local_ids, list) or any(
            not isinstance(item, str) or not item for item in local_ids
        ):
            raise T02Error("local_llm_selected_ids must be null or a string list")
        if len(local_ids) != len(set(local_ids)):
            raise T02Error("local_llm_selected_ids cannot contain duplicates")
        if set(local_ids) - set(by_id):
            raise T02Error("local_llm_selected_ids references an unknown candidate")
    if state.get("bfcl_prefix_replay_bundle") is not None:
        state["prefix_replay_capability"] = inspect_bfcl_prefix_replay(
            state["bfcl_prefix_replay_bundle"]
        )
    return state


def _derive_task_identity(session_id: str) -> tuple[str, str]:
    parts = session_id.split("/")
    if len(parts) != 3 or parts[0] != "bfcl" or not parts[1] or not parts[2].startswith("attempt-"):
        raise T02Error(
            "server selection_state session_id must be bfcl/<task_id>/attempt-<n>"
        )
    return "bfcl", parts[1]


def lift_server_steps(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Lift root runtime ``recovery_checks[*].selection_state`` exports."""

    states: list[dict[str, Any]] = []
    for step_index, raw_step in enumerate(rows):
        if not isinstance(raw_step, Mapping):
            raise T02Error(f"server step {step_index} must be an object")
        session_id = _nonempty_string(raw_step.get("session_id"), "step.session_id")
        decision_key = _nonempty_string(
            raw_step.get("decision_key"), "step.decision_key"
        )
        benchmark, task_id = _derive_task_identity(session_id)
        checks = raw_step.get("recovery_checks")
        if not isinstance(checks, list):
            raise T02Error(f"server step {step_index} lacks recovery_checks")
        for round_index, check in enumerate(checks):
            selection = check.get("selection_state") if isinstance(check, Mapping) else None
            if selection is None:
                continue
            if not isinstance(selection, Mapping):
                raise T02Error("selection_state must be an object")
            state = _json_copy(selection, label="selection_state")
            state.update(
                schema=CANDIDATE_SCHEMA,
                state_id=state.get(
                    "state_id", f"{session_id}/{decision_key}/round-{round_index + 1}"
                ),
                benchmark=benchmark,
                task_id=state.get("task_id", task_id),
                task_group_id=canonical_task_group_id(task_id),
                session_id=session_id,
                decision_key=decision_key,
                recovery_round=state.get("recovery_round", round_index + 1),
            )
            if "allowed_actions" not in state and "actions" in state:
                state["allowed_actions"] = state.pop("actions")
            draft = state.get("draft")
            if not isinstance(draft, Mapping):
                raise T02Error("selection_state.draft must be an object")
            kind = state.get("draft_kind", draft.get("kind"))
            if isinstance(kind, str):
                kind = kind.lower()
            if kind not in {"call", "stop"}:
                calls = draft.get("tool_calls")
                kind = "call" if isinstance(calls, list) and calls else "stop"
            state["draft_kind"] = kind
            state.setdefault("risk_bucket", "unknown")
            state.setdefault("previous_turn_valid", None)
            states.append(_normalize_candidate_state(state))
    if not states:
        raise T02Error("server steps contain no recovery_checks[*].selection_state")
    return states


def inspect_bfcl_prefix_replay(bundle: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the existing request-prefix gate without overstating restore."""

    try:
        from benchmarks.memory_runtime.bfcl_prefix_replay import FrozenPrefixReplay

        gate = FrozenPrefixReplay(bundle)
    except (ImportError, TypeError, ValueError, RuntimeError) as error:
        raise T02Error(f"invalid BFCL frozen-prefix replay bundle: {error}") from error
    return {
        "schema": PREFIX_RECEIPT_SCHEMA,
        "replay_schema": gate.receipt()["schema"],
        "task_id": gate.task_id,
        "bundle_schema_validated": True,
        "request_prefix_replay": True,
        "exact_environment_restore": False,
        "exact_actor_backend_restore": False,
        "acceptable_as_t02_snapshot": False,
    }


def inspect_runtime_capabilities() -> dict[str, Any]:
    """Report the audited local execution boundary for T02.

    This is intentionally a static contract report for the checked-in runtime,
    not a probe that starts SGLang or assumes an external service's behavior.
    The in-process BFCL adapter and exact actor/backend snapshot path are
    implemented; a real NPU receipt is still required before calling the path
    validated on the production backend.
    """

    return {
        "schema": RUNTIME_CAPABILITY_SCHEMA,
        "offline_planning": True,
        "offline_labeling": True,
        "branch_execution_api": "execute_plan_requires_explicit_adapter",
        "exact_branch_adapter_available": True,
        "status": "implemented_requires_live_backend_validation",
        "implementation": "t02_bfcl.ExactBFCLBranchAdapter",
        "real_backend_validated": False,
        "audited_runtime": {
            "bfcl_frozen_prefix_replay": {
                "available": True,
                "scope": "request_and_action_prefix_validation_then_live_execution",
                "acceptable_as_t02_snapshot": False,
            },
            "sglang_session_cache": {
                "external_cache_state": "owned_bounded_exact_snapshot_registry",
                "device_raw_snapshot_present": True,
                "acceptable_as_t02_snapshot": True,
            },
            "native_t02_snapshot_restore_endpoint": True,
            "official_bfcl_continuation_adapter": True,
        },
        "missing_exact_restore_components": [],
        "missing_integration_hooks": [],
        "execution_behavior_without_adapter": "fail_closed",
    }


def _extract_manifest_ids(value: Any, *, name: str) -> set[str]:
    if isinstance(value, Mapping):
        if "task_ids" in value:
            value = value["task_ids"]
        elif "tasks" in value:
            value = value["tasks"]
        elif "task_id" in value:
            value = [value]
        else:
            raise T02Error(f"{name} manifest needs task_ids, tasks, or task_id rows")
    if not isinstance(value, list):
        raise T02Error(f"{name} manifest must resolve to a list")
    ids: set[str] = set()
    for index, item in enumerate(value):
        if isinstance(item, str):
            task_id = item
        elif isinstance(item, Mapping):
            task_id = item.get("task_id", item.get("id"))
        else:
            task_id = None
        ids.add(_nonempty_string(task_id, f"{name}[{index}].task_id"))
    if not ids:
        raise T02Error(f"{name} manifest cannot be empty")
    return ids


def load_forbidden_manifests(paths: Mapping[str, str | Path]) -> tuple[set[str], list[dict[str, Any]]]:
    missing = sorted(set(FORBIDDEN_MANIFEST_NAMES) - set(paths))
    if missing:
        raise T02Error(f"explicit forbidden manifests are required: {', '.join(missing)}")
    all_ids: set[str] = set()
    receipts = []
    for name in FORBIDDEN_MANIFEST_NAMES:
        path = Path(paths[name]).resolve()
        if not path.is_file():
            raise T02Error(f"{name} manifest does not exist: {path}")
        try:
            if path.suffix.lower() == ".jsonl":
                value = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            else:
                value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise T02Error(f"{name} manifest is unreadable JSON") from error
        ids = _extract_manifest_ids(value, name=name)
        all_ids.update(ids)
        all_ids.update(canonical_task_group_id(task_id) for task_id in ids)
        receipts.append(
            {
                "name": name,
                "path": str(path),
                "sha256": _file_digest(path),
                "task_count": len(ids),
                "task_ids": sorted(ids),
                "task_ids_sha256": _digest(sorted(ids)),
            }
        )
    return all_ids, receipts


def _action_for_ids(state: Mapping[str, Any], candidate_ids: Sequence[str]) -> dict[str, Any] | None:
    wanted = set(candidate_ids)
    for action in state["allowed_actions"]:
        if set(action["candidate_ids"]) == wanted and len(action["candidate_ids"]) == len(candidate_ids):
            return action
    return None


def _branch_actions(state: Mapping[str, Any], *, seed: int) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    candidates = {row["candidate_id"]: row for row in state["candidates"]}
    feasible = [row for row in state["candidates"] if row["feasible"]]
    singleton_actions = {
        action["candidate_ids"][0]: action
        for action in state["allowed_actions"]
        if len(action["candidate_ids"]) == 1
    }
    top = feasible[0] if feasible else None
    if top is None or top["candidate_id"] not in singleton_actions:
        raise T02Error(f"state {state['state_id']} has no feasible singleton A1 action")
    a1_action = singleton_actions[top["candidate_id"]]

    local_ids = state.get("local_llm_selected_ids")
    a2_action = None
    a2_source = None
    if local_ids:
        local = _action_for_ids(state, local_ids)
        if local is not None and set(local["candidate_ids"]) != set(a1_action["candidate_ids"]):
            a2_action, a2_source = local, "local_llm_distinct_nonempty"
    if a2_action is None:
        alternatives = []
        for action in state["allowed_actions"]:
            ids = action["candidate_ids"]
            if not ids or set(ids) == set(a1_action["candidate_ids"]):
                continue
            if len(ids) == 1:
                kind = "alternative_singleton"
            elif len(ids) == 2 and len({candidates[item]["source_id"] for item in ids}) == 2:
                kind = "distinct_source_pair"
            else:
                continue
            rank_key = (sum(candidates[item]["rank"] for item in ids), len(ids), tuple(ids))
            alternatives.append((rank_key, kind, action))
        alternatives.sort(key=lambda row: row[0])
        pool = alternatives[:A2_FALLBACK_POOL_LIMIT]
        if not pool:
            raise T02Error(
                f"state {state['state_id']} has no distinct nonempty A2 action"
            )
        choice = int(
            hashlib.sha256(f"{seed}:{state['state_id']}:A2".encode("utf-8")).hexdigest(),
            16,
        ) % len(pool)
        _, kind, a2_action = pool[choice]
        a2_source = f"seeded_{kind}"

    a0 = {
        "branch_id": "A0",
        "action_id": _action_for_ids(state, [])["action_id"],
        "candidate_ids": [],
        "submit_original_draft": True,
        "regenerate": False,
        "selection_source": "held_original_draft_no_recovery",
    }
    a1 = {
        "branch_id": "A1",
        "action_id": a1_action["action_id"],
        "candidate_ids": list(a1_action["candidate_ids"]),
        "submit_original_draft": False,
        "regenerate": True,
        "selection_source": "top_ranked_feasible_singleton",
    }
    a2 = {
        "branch_id": "A2",
        "action_id": a2_action["action_id"],
        "candidate_ids": list(a2_action["candidate_ids"]),
        "submit_original_draft": False,
        "regenerate": True,
        "selection_source": a2_source,
    }
    return a0, a1, a2


def _sample_states(
    states: Sequence[dict[str, Any]], *, target_states: int, seed: int,
    max_states_per_task_group: int = 2, max_states_per_task: int = 2,
) -> list[dict[str, Any]]:
    if len(states) < target_states:
        raise T02Error(f"need {target_states} eligible states, found {len(states)}")
    ordered = sorted(
        states,
        key=lambda state: (
            0 if state.get("previous_turn_valid") is True else 1 if state.get("previous_turn_valid") is None else 2,
            _digest([seed, state["state_id"]]),
        ),
    )
    required_tags = {
        ("draft_kind", "call"),
        ("draft_kind", "stop"),
        ("risk_bucket", "high"),
        ("risk_bucket", "low"),
        ("a2_cardinality", "single"),
        ("a2_cardinality", "multiple"),
    }
    chosen: list[dict[str, Any]] = []
    group_counts: Counter[str] = Counter()
    task_counts: Counter[str] = Counter()
    group_draft_kinds: dict[str, set[str]] = defaultdict(set)
    covered: set[tuple[str, str]] = set()

    def tags(state: Mapping[str, Any]) -> set[tuple[str, str]]:
        return {
            ("draft_kind", state["draft_kind"]),
            ("risk_bucket", state["risk_bucket"]),
            (
                "a2_cardinality",
                "multiple" if len(state["branches"][2]["candidate_ids"]) > 1 else "single",
            ),
        }

    while len(chosen) < target_states:
        eligible = [
            state
            for state in ordered
            if state not in chosen
            and group_counts[state["task_group_id"]] < max_states_per_task_group
            and task_counts[state["task_id"]] < max_states_per_task
        ]
        if not eligible:
            break
        best = min(
            eligible,
            key=lambda state: (
                -len(tags(state).intersection(required_tags - covered)),
                group_counts[state["task_group_id"]],
                0 if state["draft_kind"] not in group_draft_kinds[state["task_group_id"]] else 1,
                task_counts[state["task_id"]],
                0 if state.get("previous_turn_valid") is True else 1 if state.get("previous_turn_valid") is None else 2,
                _digest([seed, state["state_id"]]),
            ),
        )
        chosen.append(best)
        group_counts[best["task_group_id"]] += 1
        task_counts[best["task_id"]] += 1
        group_draft_kinds[best["task_group_id"]].add(best["draft_kind"])
        covered.update(tags(best))
    if len(chosen) != target_states:
        raise T02Error(
            "cannot meet target_states within per-family and per-variant state caps"
        )
    missing = sorted(required_tags - covered)
    if missing:
        raise T02Error(f"sample does not cover required strata: {missing}")
    return chosen


def _assign_splits(states: Sequence[dict[str, Any]], *, train_states: int, seed: int) -> None:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for state in states:
        groups[state["task_group_id"]].append(state)
    ordered = sorted(groups, key=lambda group: _digest([seed, group, "split"]))
    possible: dict[int, tuple[str, ...]] = {0: ()}
    for group in ordered:
        size = len(groups[group])
        additions = {
            count + size: (*selected, group)
            for count, selected in possible.items()
            if count + size <= len(states)
        }
        for count, selected in additions.items():
            possible.setdefault(count, selected)
    chosen_count = min(possible, key=lambda count: (abs(count - train_states), -count))
    train_groups = set(possible[chosen_count])
    if not train_groups or len(train_groups) == len(groups):
        raise T02Error("task-disjoint train/calibration split is impossible")
    for state in states:
        state["split"] = "train" if state["task_group_id"] in train_groups else "calibration"


def build_plan(
    candidate_states: Sequence[Mapping[str, Any]],
    *,
    forbidden_manifest_paths: Mapping[str, str | Path],
    frozen_subsequent_policy: Mapping[str, Any],
    seed: int = 0,
    target_states: int = 60,
    train_states: int = 40,
    min_task_groups: int = 30,
    max_states_per_task_group: int = 2,
    max_states_per_task: int = 2,
    max_complete_branch_executions: int = 180,
) -> dict[str, Any]:
    """Build the bounded T02 plan without consulting any future outcome."""

    if type(seed) is not int:
        raise T02Error("seed must be an integer")
    for value, name in (
        (target_states, "target_states"),
        (train_states, "train_states"),
        (min_task_groups, "min_task_groups"),
        (max_states_per_task_group, "max_states_per_task_group"),
        (max_states_per_task, "max_states_per_task"),
        (max_complete_branch_executions, "max_complete_branch_executions"),
    ):
        if type(value) is not int or value <= 0:
            raise T02Error(f"{name} must be a positive integer")
    if not 0 < train_states < target_states:
        raise T02Error("train_states must be between zero and target_states")
    if target_states * len(BRANCH_IDS) > max_complete_branch_executions:
        raise T02Error("target states exceed the explicitly authorized complete branch budget")
    if min_task_groups > target_states:
        raise T02Error("min_task_groups cannot exceed target_states")
    policy = _json_copy(frozen_subsequent_policy, label="frozen subsequent policy")
    if not isinstance(policy, dict) or not policy:
        raise T02Error("frozen_subsequent_policy must be a nonempty object")
    forbidden_ids, manifest_receipts = load_forbidden_manifests(forbidden_manifest_paths)

    normalized = []
    seen_states: set[str] = set()
    rejected_forbidden = []
    for raw in candidate_states:
        state = _normalize_candidate_state(raw)
        if state["state_id"] in seen_states:
            raise T02Error(f"duplicate state_id: {state['state_id']}")
        seen_states.add(state["state_id"])
        if state["task_id"] in forbidden_ids or state["task_group_id"] in forbidden_ids:
            rejected_forbidden.append(state["state_id"])
            continue
        branches = _branch_actions(state, seed=seed)
        planned = dict(state)
        planned["schema"] = PLANNED_STATE_SCHEMA
        planned["branches"] = list(branches)
        planned["branch_selection_uses_future_outcome"] = False
        normalized.append(planned)
    selected = _sample_states(
        normalized, target_states=target_states, seed=seed,
        max_states_per_task_group=max_states_per_task_group,
        max_states_per_task=max_states_per_task,
    )
    groups = {state["task_group_id"] for state in selected}
    if len(groups) < min_task_groups:
        raise T02Error(
            f"sample needs at least {min_task_groups} task groups, found {len(groups)}"
        )
    _assign_splits(selected, train_states=train_states, seed=seed)
    selected.sort(key=lambda state: (state["split"] != "train", state["task_group_id"], state["state_id"]))
    plan = {
        "schema": PLAN_SCHEMA,
        "seed": seed,
        "target_states": target_states,
        "target_train_states": train_states,
        "minimum_task_groups": min_task_groups,
        "max_states_per_task_group": max_states_per_task_group,
        "max_states_per_task": max_states_per_task,
        "branches_per_state": 3,
        "max_complete_branch_executions": target_states * 3,
        "authorized_complete_branch_cap": max_complete_branch_executions,
        "frozen_subsequent_policy": policy,
        "frozen_subsequent_policy_sha256": _digest(policy),
        "required_snapshot_components": list(REQUIRED_COMPONENTS),
        "prefix_replay_is_exact_snapshot": False,
        "forbidden_manifests": manifest_receipts,
        "rejected_forbidden_state_ids": rejected_forbidden,
        "sampling": {
            "selection_unit": "state",
            "branch_is_independent_sample": False,
            "uses_future_success": False,
            "previous_turn_valid_priority": [True, None, False],
            "strata": [
                "draft_call",
                "draft_stop",
                "risk_high",
                "risk_low",
                "a2_singleton",
                "a2_multiple_candidates",
            ],
        },
        "states": selected,
    }
    check_plan(plan)
    return plan


def build_smoke_plan(
    candidate_state: Mapping[str, Any],
    *,
    forbidden_manifest_paths: Mapping[str, str | Path],
    frozen_subsequent_policy: Mapping[str, Any],
    seed: int = 0,
) -> dict[str, Any]:
    """Build one real A0/A1/A2 smoke without weakening production sampling."""

    if type(seed) is not int:
        raise T02Error("seed must be an integer")
    policy = _json_copy(frozen_subsequent_policy, label="frozen subsequent policy")
    if not isinstance(policy, dict) or not policy:
        raise T02Error("frozen_subsequent_policy must be a nonempty object")
    forbidden_ids, receipts = load_forbidden_manifests(forbidden_manifest_paths)
    state = _normalize_candidate_state(candidate_state)
    if state["task_id"] in forbidden_ids or state["task_group_id"] in forbidden_ids:
        raise T02Error("smoke state belongs to D128/F128")
    planned = dict(state)
    planned.update(
        schema=PLANNED_STATE_SCHEMA,
        split="smoke",
        branches=list(_branch_actions(state, seed=seed)),
        branch_selection_uses_future_outcome=False,
    )
    plan = {
        "schema": PLAN_SCHEMA,
        "collection_mode": "smoke",
        "seed": seed,
        "target_states": 1,
        "target_train_states": 0,
        "minimum_task_groups": 1,
        "max_states_per_task_group": 1,
        "max_states_per_task": 1,
        "branches_per_state": 3,
        "max_complete_branch_executions": 3,
        "authorized_complete_branch_cap": 3,
        "frozen_subsequent_policy": policy,
        "frozen_subsequent_policy_sha256": _digest(policy),
        "required_snapshot_components": list(REQUIRED_COMPONENTS),
        "prefix_replay_is_exact_snapshot": False,
        "forbidden_manifests": receipts,
        "rejected_forbidden_state_ids": [],
        "sampling": {
            "selection_unit": "state",
            "branch_is_independent_sample": False,
            "uses_future_success": False,
            "strata": [],
            "scope": "one-state integration smoke only",
        },
        "states": [planned],
    }
    check_plan(plan)
    return plan


def check_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(plan, Mapping) or plan.get("schema") != PLAN_SCHEMA:
        raise T02Error(f"plan schema must be {PLAN_SCHEMA}")
    smoke = plan.get("collection_mode", "production") == "smoke"
    if plan.get("collection_mode", "production") not in {"production", "smoke"}:
        raise T02Error("plan collection_mode must be production or smoke")
    states = plan.get("states")
    if not isinstance(states, list) or len(states) != plan.get("target_states"):
        raise T02Error("plan state count differs from target_states")
    if plan.get("max_complete_branch_executions") != len(states) * 3:
        raise T02Error("plan branch cap must be exactly three per state")
    authorized_cap = plan.get("authorized_complete_branch_cap", 180)
    if type(authorized_cap) is not int or authorized_cap <= 0:
        raise T02Error("plan authorized complete branch cap must be a positive integer")
    if plan["max_complete_branch_executions"] > authorized_cap:
        raise T02Error("plan exceeds its explicitly authorized complete branch cap")
    if tuple(plan.get("required_snapshot_components", ())) != REQUIRED_COMPONENTS:
        raise T02Error("plan snapshot components differ from the T02 restore contract")
    if plan.get("prefix_replay_is_exact_snapshot") is not False:
        raise T02Error("prefix replay cannot be declared an exact T02 snapshot")
    if _digest(plan.get("frozen_subsequent_policy")) != plan.get(
        "frozen_subsequent_policy_sha256"
    ):
        raise T02Error("frozen subsequent policy digest mismatch")
    manifest_rows = plan.get("forbidden_manifests")
    if not isinstance(manifest_rows, list) or len(manifest_rows) != len(FORBIDDEN_MANIFEST_NAMES) or {
        row.get("name") for row in manifest_rows if isinstance(row, Mapping)
    } != set(FORBIDDEN_MANIFEST_NAMES):
        raise T02Error("plan must bind explicit D128 and F128 manifests")
    forbidden_ids: set[str] = set()
    for row in manifest_rows:
        task_ids = row.get("task_ids")
        if (
            not isinstance(task_ids, list)
            or any(not isinstance(task_id, str) or not task_id for task_id in task_ids)
            or len(task_ids) != len(set(task_ids))
            or row.get("task_count") != len(task_ids)
            or row.get("task_ids_sha256") != _digest(sorted(task_ids))
            or not _valid_sha256(row.get("sha256"))
        ):
            raise T02Error(f"{row['name']} forbidden manifest receipt is invalid")
        forbidden_ids.update(task_ids)
        forbidden_ids.update(canonical_task_group_id(task_id) for task_id in task_ids)
    ids: set[str] = set()
    task_splits: dict[str, str] = {}
    group_counts: Counter[str] = Counter()
    task_counts: Counter[str] = Counter()
    max_group = plan.get("max_states_per_task_group")
    max_task = plan.get("max_states_per_task", max_group)
    if type(max_group) is not int or max_group <= 0 or type(max_task) is not int or max_task <= 0:
        raise T02Error("plan state caps must be positive integers")
    coverage: set[tuple[str, str]] = set()
    for raw in states:
        if not isinstance(raw, Mapping) or raw.get("schema") != PLANNED_STATE_SCHEMA:
            raise T02Error("plan contains an invalid planned state")
        state_id = _nonempty_string(raw.get("state_id"), "state_id")
        if state_id in ids:
            raise T02Error("plan repeats a state_id")
        ids.add(state_id)
        group = _nonempty_string(raw.get("task_group_id"), "task_group_id")
        if raw.get("task_id") in forbidden_ids or group in forbidden_ids:
            raise T02Error("plan contains a D128/F128 task")
        split = raw.get("split")
        expected_splits = {"smoke"} if smoke else {"train", "calibration"}
        if split not in expected_splits:
            raise T02Error(
                "planned state split must be smoke" if smoke
                else "planned state split must be train or calibration"
            )
        if group in task_splits and task_splits[group] != split:
            raise T02Error("one task group appears in both train and calibration")
        task_splits[group] = split
        group_counts[group] += 1
        task_counts[raw["task_id"]] += 1
        if group_counts[group] > max_group:
            raise T02Error("a task group exceeds max_states_per_task_group")
        if task_counts[raw["task_id"]] > max_task:
            raise T02Error("a task variant exceeds max_states_per_task")
        branches = raw.get("branches")
        if not isinstance(branches, list) or [row.get("branch_id") for row in branches] != list(BRANCH_IDS):
            raise T02Error("each state must contain ordered A0, A1, A2 branches")
        if branches[0]["candidate_ids"] or branches[0]["regenerate"] or not branches[0]["submit_original_draft"]:
            raise T02Error("A0 must submit the held original draft without recovery")
        if len(branches[1]["candidate_ids"]) != 1 or not branches[1]["regenerate"]:
            raise T02Error("A1 must regenerate with one feasible candidate")
        if not branches[2]["candidate_ids"] or not branches[2]["regenerate"]:
            raise T02Error("A2 must regenerate with a nonempty action")
        if set(branches[1]["candidate_ids"]) == set(branches[2]["candidate_ids"]):
            raise T02Error("A2 must differ from A1")
        coverage.update(
            {
                ("draft_kind", raw["draft_kind"]),
                ("risk_bucket", raw["risk_bucket"]),
                (
                    "a2_cardinality",
                    "multiple" if len(branches[2]["candidate_ids"]) > 1 else "single",
                ),
            }
        )
    if len(task_splits) < plan.get("minimum_task_groups"):
        raise T02Error("plan contains too few task groups")
    required = {
        ("draft_kind", "call"),
        ("draft_kind", "stop"),
        ("risk_bucket", "high"),
        ("risk_bucket", "low"),
        ("a2_cardinality", "single"),
        ("a2_cardinality", "multiple"),
    }
    if not smoke and not required.issubset(coverage):
        raise T02Error("plan is missing a required sampling stratum")
    split_counts = Counter(state["split"] for state in states)
    return {
        "schema": "t02-artifact-check-v1",
        "artifact_schema": PLAN_SCHEMA,
        "valid": True,
        "state_count": len(states),
        "task_group_count": len(task_splits),
        "split_counts": dict(split_counts),
        "complete_branch_cap": plan["max_complete_branch_executions"],
    }


def _check_capabilities(value: Mapping[str, Any]) -> dict[str, Any]:
    capabilities = _json_copy(value, label="adapter capabilities")
    if not isinstance(capabilities, dict) or capabilities.get("schema") != CAPABILITY_SCHEMA:
        raise T02CapabilityError(f"adapter capabilities schema must be {CAPABILITY_SCHEMA}")
    if capabilities.get("exact_same_state_restore") is not True:
        raise T02CapabilityError("adapter lacks exact_same_state_restore")
    components = capabilities.get("components")
    if (
        not isinstance(components, list)
        or len(components) != len(REQUIRED_COMPONENTS)
        or set(components) != set(REQUIRED_COMPONENTS)
    ):
        raise T02CapabilityError(
            "adapter must restore environment, actor KV/positions, backend stats, and RNG"
        )
    for name in (
        "frozen_subsequent_policy",
        "official_turn_outcome",
        "official_task_outcome",
    ):
        if capabilities.get(name) is not True:
            raise T02CapabilityError(f"adapter capability {name} is required")
    return capabilities


def _check_snapshot(
    snapshot: Mapping[str, Any], *, state_id: str, policy_sha256: str
) -> dict[str, Any]:
    value = _json_copy(snapshot, label="snapshot")
    if value.get("schema") != SNAPSHOT_SCHEMA:
        raise T02CapabilityError(f"snapshot schema must be {SNAPSHOT_SCHEMA}")
    if value.get("state_id") != state_id:
        raise T02CapabilityError("snapshot state_id mismatch")
    _nonempty_string(value.get("snapshot_id"), "snapshot_id")
    if value.get("frozen_policy_sha256") != policy_sha256:
        raise T02CapabilityError("snapshot frozen policy digest mismatch")
    digests = value.get("component_digests")
    if not isinstance(digests, Mapping) or set(digests) != set(REQUIRED_COMPONENTS):
        raise T02CapabilityError("snapshot component digest set is incomplete")
    if not all(_valid_sha256(digest) for digest in digests.values()):
        raise T02CapabilityError("snapshot component digests must be lowercase sha256")
    return value


def _check_restore(
    receipt: Mapping[str, Any], *, snapshot: Mapping[str, Any]
) -> dict[str, Any]:
    value = _json_copy(receipt, label="restore receipt")
    if value.get("schema") != RESTORE_SCHEMA:
        raise T02CapabilityError(f"restore receipt schema must be {RESTORE_SCHEMA}")
    for name in ("state_id", "snapshot_id", "frozen_policy_sha256"):
        if value.get(name) != snapshot.get(name):
            raise T02CapabilityError(f"restore receipt {name} mismatch")
    if value.get("component_digests") != snapshot.get("component_digests"):
        raise T02CapabilityError("restore receipt component digests differ from snapshot")
    if value.get("restored") is not True:
        raise T02CapabilityError("adapter did not affirm exact restoration")
    return value


def _check_official_outcome(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise T02CapabilityError("completed branch lacks official_outcome")
    outcome = _json_copy(value, label="official outcome")
    if outcome.get("source") != "official":
        raise T02CapabilityError("branch outcome source must be official")
    _nonempty_string(outcome.get("scorer"), "official_outcome.scorer")
    _nonempty_string(outcome.get("artifact"), "official_outcome.artifact")
    if not _valid_sha256(outcome.get("artifact_sha256")):
        raise T02CapabilityError("official outcome artifact_sha256 is invalid")
    for name in ("turn_success", "task_success"):
        value = outcome.get(name)
        if value is not None and type(value) is not bool:
            raise T02CapabilityError(f"official_outcome.{name} must be true, false, or null")
    return outcome


def execute_plan(plan: Mapping[str, Any], adapter: BranchAdapter, *,
                 state_ids: Sequence[str] | None = None,
                 on_progress: Callable[[Mapping[str, Any]], None] | None = None) -> dict[str, Any]:
    """Execute exactly three independently restored branches per planned state."""

    check_plan(plan)
    capabilities = _check_capabilities(adapter.capabilities())
    policy = plan["frozen_subsequent_policy"]
    policy_sha256 = plan["frozen_subsequent_policy_sha256"]
    snapshots = []
    results = []
    owned = None if state_ids is None else set(state_ids)
    planned_ids = {state["state_id"] for state in plan["states"]}
    if owned is not None and (len(owned) != len(state_ids) or not owned <= planned_ids):
        raise T02CapabilityError("worker state IDs are repeated or absent from the global plan")
    selected_states = [state for state in plan["states"]
                       if owned is None or state["state_id"] in owned]

    def result_set():
        return {
            "schema": RESULT_SET_SCHEMA,
            "plan_sha256": _digest(plan),
            "adapter_capabilities": capabilities,
            "complete_branch_executions": len(results),
            "branch_cap": plan["max_complete_branch_executions"],
            "authorized_complete_branch_cap": plan.get(
                "authorized_complete_branch_cap", plan["max_complete_branch_executions"]
            ),
            "snapshots": list(snapshots),
            "results": list(results),
            "executed_state_ids": [snapshot["state_id"] for snapshot in snapshots],
            "assigned_state_ids": [state["state_id"] for state in selected_states],
            "partial_worker_result": owned is not None,
        }

    for state in selected_states:
        snapshot = _check_snapshot(
            adapter.capture_state(state, frozen_policy=policy),
            state_id=state["state_id"],
            policy_sha256=policy_sha256,
        )
        snapshots.append(snapshot)
        for branch in state["branches"]:
            if len(results) >= plan["max_complete_branch_executions"]:
                raise T02CapabilityError("complete branch cap would be exceeded")
            restore = _check_restore(
                adapter.restore_state(snapshot, frozen_policy=policy), snapshot=snapshot
            )
            restore_sha256 = _digest(restore)
            raw_result = adapter.run_branch(
                state,
                branch,
                frozen_policy=policy,
                restore_receipt=restore,
            )
            result = _json_copy(raw_result, label="branch result")
            if result.get("schema") != RESULT_SCHEMA:
                raise T02CapabilityError(f"branch result schema must be {RESULT_SCHEMA}")
            expected = {
                "state_id": state["state_id"],
                "branch_id": branch["branch_id"],
                "candidate_ids": branch["candidate_ids"],
                "snapshot_id": snapshot["snapshot_id"],
                "restore_receipt_sha256": restore_sha256,
                "frozen_policy_sha256": policy_sha256,
            }
            for name, expected_value in expected.items():
                if result.get(name) != expected_value:
                    raise T02CapabilityError(f"branch result {name} is not bound to this run")
            if result.get("execution_status") != "complete":
                raise T02CapabilityError("branch execution is incomplete; refusing stale labels")
            result["official_outcome"] = _check_official_outcome(
                result.get("official_outcome")
            )
            execution = result.get("execution_receipt")
            expected_execution = {
                "submitted_original_draft": branch["submit_original_draft"],
                "recovery_candidate_ids": branch["candidate_ids"],
                "regeneration_count": 1 if branch["regenerate"] else 0,
                "continued_with_frozen_policy": True,
                "observations_replayed_from_other_branch": False,
            }
            if execution != expected_execution:
                raise T02CapabilityError(
                    "branch execution receipt does not match the A0/A1/A2 contract"
                )
            results.append(result)
            if on_progress is not None:
                on_progress(result_set())
    return result_set()


def _index_results(value: Any, plan: Mapping[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    if isinstance(value, Mapping):
        if value.get("schema") != RESULT_SET_SCHEMA:
            raise T02Error(f"result artifact schema must be {RESULT_SET_SCHEMA}")
        if value.get("plan_sha256") != _digest(plan):
            raise T02Error("result artifact belongs to a different plan")
        rows = value.get("results")
    else:
        rows = value
    if not isinstance(rows, list):
        raise T02Error("results must be a list or a branch result artifact")
    planned = {
        (state["state_id"], branch["branch_id"]): branch
        for state in plan["states"]
        for branch in state["branches"]
    }
    indexed = {}
    for row in rows:
        if not isinstance(row, Mapping) or row.get("schema") != RESULT_SCHEMA:
            raise T02Error("result rows must use the branch result schema")
        key = (row.get("state_id"), row.get("branch_id"))
        if key not in planned:
            raise T02Error(f"result row is not in the plan: {key}")
        if key in indexed:
            raise T02Error(f"duplicate result row: {key}")
        if row.get("candidate_ids") != planned[key]["candidate_ids"]:
            raise T02Error("result candidate_ids differ from the planned intervention")
        if row.get("execution_status") != "complete":
            raise T02Error("only complete branch results can produce labels")
        _nonempty_string(row.get("snapshot_id"), "result.snapshot_id")
        if not _valid_sha256(row.get("restore_receipt_sha256")):
            raise T02Error("result restore_receipt_sha256 is invalid")
        if row.get("frozen_policy_sha256") != plan["frozen_subsequent_policy_sha256"]:
            raise T02Error("result frozen policy digest differs from the plan")
        branch = planned[key]
        if row.get("execution_receipt") != {
            "submitted_original_draft": branch["submit_original_draft"],
            "recovery_candidate_ids": branch["candidate_ids"],
            "regeneration_count": 1 if branch["regenerate"] else 0,
            "continued_with_frozen_policy": True,
            "observations_replayed_from_other_branch": False,
        }:
            raise T02Error("result execution receipt differs from the branch contract")
        try:
            outcome = _check_official_outcome(row.get("official_outcome"))
        except T02CapabilityError as error:
            raise T02Error(str(error)) from error
        normalized = _json_copy(row, label="result row")
        normalized["official_outcome"] = outcome
        indexed[key] = normalized
    return indexed


def _outcome_or_unknown(
    indexed: Mapping[tuple[str, str], Mapping[str, Any]], state_id: str, branch_id: str
) -> dict[str, Any]:
    row = indexed.get((state_id, branch_id))
    if row is None:
        return {
            "execution_status": "missing",
            "turn_success": None,
            "task_success": None,
            "official_outcome": None,
        }
    outcome = row["official_outcome"]
    return {
        "execution_status": row.get("execution_status", "complete"),
        "turn_success": outcome["turn_success"],
        "task_success": outcome["task_success"],
        "official_outcome": outcome,
    }


def _delta(intervention: Any, baseline: Any) -> tuple[int | None, str]:
    if intervention is None or baseline is None:
        return None, "unknown"
    value = int(intervention) - int(baseline)
    if value not in {-1, 0, 1}:  # pragma: no cover - bool contract makes this defensive
        raise T02Error("T02 delta must be -1, 0, or 1")
    return value, "known"


def label_results(plan: Mapping[str, Any], results: Any) -> dict[str, Any]:
    """Produce one row per state; missing or unavailable outcomes remain null."""

    check_plan(plan)
    indexed = _index_results(results, plan)
    labeled = []
    for state in plan["states"]:
        outcomes = {
            branch_id: _outcome_or_unknown(indexed, state["state_id"], branch_id)
            for branch_id in BRANCH_IDS
        }
        baseline = outcomes["A0"]
        labels = {}
        for branch_id in ("A1", "A2"):
            turn, turn_status = _delta(
                outcomes[branch_id]["turn_success"], baseline["turn_success"]
            )
            task, task_status = _delta(
                outcomes[branch_id]["task_success"], baseline["task_success"]
            )
            labels[branch_id] = {
                "candidate_ids": next(
                    branch["candidate_ids"]
                    for branch in state["branches"]
                    if branch["branch_id"] == branch_id
                ),
                "delta_turn": turn,
                "delta_task": task,
                "turn_label_status": turn_status,
                "task_label_status": task_status,
            }
        a0_turn = baseline["turn_success"]
        tested_action_ids = {branch["action_id"] for branch in state["branches"]}
        untested_actions = [
            {
                "action_id": action["action_id"],
                "candidate_ids": copy.deepcopy(action["candidate_ids"]),
                "delta_turn": None,
                "delta_task": None,
                "turn_label_status": "unknown",
                "task_label_status": "unknown",
            }
            for action in state["allowed_actions"]
            if action["action_id"] not in tested_action_ids
        ]
        labeled.append(
            {
                "schema": LABELED_STATE_SCHEMA,
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
                "branches": copy.deepcopy(state["branches"]),
                "outcomes": outcomes,
                "labels": labels,
                "untested_actions": untested_actions,
                "c1_risk_label": None if a0_turn is None else 1 - int(a0_turn),
                "c1_label_status": "unknown" if a0_turn is None else "known",
                "c1_label_source": "A0.official_outcome.turn_success",
                "statistical_unit": "state",
                "branches_are_independent_states": False,
            }
        )
    artifact = {
        "schema": LABELED_SET_SCHEMA,
        "plan_sha256": _digest(plan),
        "state_count": len(labeled),
        "observed_branch_count": len(indexed),
        "rows": labeled,
    }
    check_labeled(artifact)
    return artifact


def check_labeled(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or value.get("schema") != LABELED_SET_SCHEMA:
        raise T02Error(f"labeled artifact schema must be {LABELED_SET_SCHEMA}")
    rows = value.get("rows")
    if not isinstance(rows, list) or value.get("state_count") != len(rows):
        raise T02Error("labeled state_count is inconsistent")
    ids = set()
    unknown = 0
    for row in rows:
        if not isinstance(row, Mapping) or row.get("schema") != LABELED_STATE_SCHEMA:
            raise T02Error("invalid labeled state row")
        state_id = _nonempty_string(row.get("state_id"), "state_id")
        if state_id in ids:
            raise T02Error("labeled dataset repeats a state_id")
        ids.add(state_id)
        if row.get("statistical_unit") != "state" or row.get("branches_are_independent_states") is not False:
            raise T02Error("labeled dataset must keep state as the statistical unit")
        outcomes = row.get("outcomes")
        if not isinstance(outcomes, Mapping) or set(outcomes) != set(BRANCH_IDS):
            raise T02Error("labeled row needs A0, A1, and A2 outcomes")
        baseline_turn = outcomes["A0"].get("turn_success")
        expected_risk = None if baseline_turn is None else 1 - int(baseline_turn)
        if row.get("c1_risk_label") != expected_risk:
            raise T02Error("C1 risk label must derive only from A0 turn_success")
        for branch_id in ("A1", "A2"):
            label = row.get("labels", {}).get(branch_id)
            if not isinstance(label, Mapping):
                raise T02Error("labeled row lacks an intervention label")
            for name in ("delta_turn", "delta_task"):
                if label.get(name) not in {-1, 0, 1, None}:
                    raise T02Error(f"{name} must be -1, 0, 1, or null")
                unknown += label.get(name) is None
        untested = row.get("untested_actions")
        if not isinstance(untested, list):
            raise T02Error("labeled row needs an explicit untested_actions list")
        for action in untested:
            if not isinstance(action, Mapping) or any(
                action.get(name) is not None for name in ("delta_turn", "delta_task")
            ):
                raise T02Error("untested action labels must remain null")
            if action.get("turn_label_status") != "unknown" or action.get(
                "task_label_status"
            ) != "unknown":
                raise T02Error("untested action status must remain unknown")
    return {
        "schema": "t02-artifact-check-v1",
        "artifact_schema": LABELED_SET_SCHEMA,
        "valid": True,
        "state_count": len(rows),
        "unknown_delta_count": unknown,
    }


def check_artifact(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise T02Error("check expects one JSON object artifact")
    if value.get("schema") == PLAN_SCHEMA:
        return check_plan(value)
    if value.get("schema") == LABELED_SET_SCHEMA:
        return check_labeled(value)
    if value.get("schema") == RESULT_SET_SCHEMA:
        results = value.get("results")
        if not isinstance(results, list):
            raise T02Error("branch result artifact needs a results list")
        if value.get("complete_branch_executions") != len(results):
            raise T02Error("complete_branch_executions differs from result row count")
        branch_cap = value.get("branch_cap")
        authorized_cap = value.get("authorized_complete_branch_cap", 180)
        if (
            type(branch_cap) is not int
            or type(authorized_cap) is not int
            or not 0 <= len(results) <= branch_cap <= authorized_cap
        ):
            raise T02Error("branch result count or cap is invalid")
        keys = set()
        for row in results:
            if not isinstance(row, Mapping) or row.get("schema") != RESULT_SCHEMA:
                raise T02Error("branch result artifact contains an invalid row")
            key = (row.get("state_id"), row.get("branch_id"))
            if key in keys or row.get("branch_id") not in BRANCH_IDS:
                raise T02Error("branch result keys must be unique A0/A1/A2 pairs")
            keys.add(key)
            if row.get("execution_status") != "complete":
                raise T02Error("branch result artifact contains an incomplete row")
            _nonempty_string(row.get("snapshot_id"), "result.snapshot_id")
            if not _valid_sha256(row.get("restore_receipt_sha256")):
                raise T02Error("branch result restore receipt digest is invalid")
            execution = row.get("execution_receipt")
            if (
                not isinstance(execution, Mapping)
                or execution.get("continued_with_frozen_policy") is not True
                or execution.get("observations_replayed_from_other_branch") is not False
                or execution.get("regeneration_count") not in {0, 1}
            ):
                raise T02Error("branch result execution receipt is invalid")
            try:
                _check_official_outcome(row.get("official_outcome"))
            except T02CapabilityError as error:
                raise T02Error(str(error)) from error
        return {
            "schema": "t02-artifact-check-v1",
            "artifact_schema": RESULT_SET_SCHEMA,
            "valid": True,
            "complete_branch_executions": len(results),
        }
    raise T02Error(f"unsupported artifact schema: {value.get('schema')}")


def _read_rows(path: Path) -> list[Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    value = json.loads(text)
    if isinstance(value, list):
        return value
    if isinstance(value, Mapping) and isinstance(value.get("rows"), list):
        return value["rows"]
    return [value]


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _manifest_args(values: Sequence[str]) -> dict[str, Path]:
    result = {}
    for value in values:
        if "=" not in value:
            raise T02Error("--forbidden-manifest must be NAME=PATH")
        name, path = value.split("=", 1)
        if name in result:
            raise T02Error(f"duplicate forbidden manifest: {name}")
        result[name] = Path(path)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan = subparsers.add_parser("plan", help="sample states and freeze A0/A1/A2")
    plan.add_argument("--states", type=Path, required=True)
    plan.add_argument("--input-kind", choices=("states", "server-steps"), default="states")
    plan.add_argument("--forbidden-manifest", action="append", default=[], metavar="NAME=PATH")
    plan.add_argument("--frozen-policy", type=Path, required=True)
    plan.add_argument("--output", type=Path, required=True)
    plan.add_argument("--seed", type=int, default=0)
    plan.add_argument("--target-states", type=int, default=60)
    plan.add_argument("--train-states", type=int, default=40)
    plan.add_argument("--min-task-groups", type=int, default=30)
    plan.add_argument("--max-states-per-task-group", type=int, default=2)
    plan.add_argument("--max-states-per-task", type=int, default=2)
    plan.add_argument("--max-complete-branch-executions", type=int, default=180)

    label = subparsers.add_parser("label", help="derive official A0-relative labels")
    label.add_argument("--plan", type=Path, required=True)
    label.add_argument("--results", type=Path, required=True)
    label.add_argument("--output", type=Path, required=True)

    check = subparsers.add_parser("check", help="validate a plan, result, or label artifact")
    check.add_argument("--artifact", type=Path, required=True)
    subparsers.add_parser(
        "capabilities",
        help="report audited offline support and missing exact runtime hooks",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "capabilities":
        print(json.dumps(inspect_runtime_capabilities(), sort_keys=True))
        return 0
    if args.command == "plan":
        rows = _read_rows(args.states)
        states = lift_server_steps(rows) if args.input_kind == "server-steps" else rows
        artifact = build_plan(
            states,
            forbidden_manifest_paths=_manifest_args(args.forbidden_manifest),
            frozen_subsequent_policy=_read_json(args.frozen_policy),
            seed=args.seed,
            target_states=args.target_states,
            train_states=args.train_states,
            min_task_groups=args.min_task_groups,
            max_states_per_task_group=args.max_states_per_task_group,
            max_states_per_task=args.max_states_per_task,
            max_complete_branch_executions=args.max_complete_branch_executions,
        )
        _write_json(args.output, artifact)
        print(json.dumps(check_plan(artifact), sort_keys=True))
        return 0
    if args.command == "label":
        plan = _read_json(args.plan)
        raw_results = _read_json(args.results) if args.results.suffix.lower() != ".jsonl" else _read_rows(args.results)
        artifact = label_results(plan, raw_results)
        _write_json(args.output, artifact)
        print(json.dumps(check_labeled(artifact), sort_keys=True))
        return 0
    artifact = _read_json(args.artifact)
    print(json.dumps(check_artifact(artifact), sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
