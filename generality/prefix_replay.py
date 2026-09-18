"""Recorded-prefix replay primitives for cross-backend calibration.

The calibration unit is a state from the frozen T02 trace.  A target backend
must see that recorded prefix directly; it must never discover the state by
running the task from turn zero.  This module contains the source-row
validation and BFCL environment reconstruction used by ``calibrate.py``.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


class PrefixReplayIntegrityError(ValueError):
    """The source row cannot be replayed without changing its semantics."""


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _canonical(value: Any) -> str:
    return json.dumps(_jsonable(value), ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def _raw_message(message: Mapping[str, Any]) -> dict[str, Any]:
    """Copy a visible message without changing serialized tool arguments."""
    return copy.deepcopy(dict(message))


def _semantic_message(message: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize only for semantic equality checks, never for model input."""
    value = _raw_message(message)
    calls = value.get("tool_calls")
    if isinstance(calls, list):
        normalized = []
        for call in calls:
            call = dict(call)
            function = dict(call.get("function") or {})
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    pass
            function["arguments"] = arguments
            call["function"] = function
            normalized.append(call)
        value["tool_calls"] = normalized
    return value


def prefix_payload(row: Mapping[str, Any]) -> dict[str, Any]:
    q = row.get("q")
    if not isinstance(q, Mapping):
        raise PrefixReplayIntegrityError("source row has no q object")
    raw_visible = q.get("raw_visible")
    observation = q.get("last_action_observation")
    if not isinstance(raw_visible, list) or not raw_visible:
        raise PrefixReplayIntegrityError("q.raw_visible is missing or empty")
    if not isinstance(observation, list):
        raise PrefixReplayIntegrityError("q.last_action_observation is missing")
    if not all(isinstance(m, Mapping) for m in raw_visible + observation):
        raise PrefixReplayIntegrityError("prefix messages must be JSON objects")
    visible = [_raw_message(m) for m in raw_visible]
    obs = [_raw_message(m) for m in observation]
    semantic_visible = [_semantic_message(m) for m in raw_visible]
    semantic_obs = [_semantic_message(m) for m in observation]
    # The observation must be the contiguous assistant/tool suffix immediately
    # before the current user message.  This rejects a row whose state binding
    # was silently assembled from a different trajectory.
    expected_suffix = [m for m in semantic_visible if m.get("role") in ("assistant", "tool")]
    if (semantic_obs and expected_suffix[-len(semantic_obs):] != semantic_obs) or (
            not semantic_obs and expected_suffix):
        raise PrefixReplayIntegrityError(
            "q.last_action_observation is not the suffix of q.raw_visible")
    payload = {
        "schema": "t02-recorded-prefix-v1",
        "state_id": row.get("state_id"),
        "task_id": row.get("task_id"),
        "task_group_id": row.get("task_group_id"),
        "decision_key": row.get("decision_key"),
        "session_id": q.get("session_id"),
        "raw_source_ids": q.get("raw_source_ids"),
        "raw_visible": visible,
        "last_action_observation": obs,
    }
    source_ids = payload.get("raw_source_ids")
    if (not isinstance(source_ids, list) or not source_ids or
            any(not isinstance(value, str) or not value for value in source_ids)):
        raise PrefixReplayIntegrityError("source raw_source_ids binding is incomplete")
    if not all(isinstance(payload[k], str) and payload[k] for k in
               ("state_id", "task_id", "decision_key", "session_id")):
        raise PrefixReplayIntegrityError("source state identity is incomplete")
    payload["prefix_payload_sha256"] = hashlib.sha256(
        _canonical(payload).encode("utf-8")).hexdigest()
    return payload


def decision_position(decision_key: str) -> tuple[int, int]:
    match = re.fullmatch(r"turn-(\d+)/step-(\d+)", decision_key)
    if match is None:
        raise PrefixReplayIntegrityError(f"invalid decision_key: {decision_key!r}")
    return int(match.group(1)), int(match.group(2))


def bind_source_trace(row: Mapping[str, Any], source_root: str | Path) -> dict[str, Any]:
    """Bind a state to the source-collection prefix before its first decision.

    Branch runs occur after source collection in the frozen actor journal.
    Stop at the first target occurrence; never consume its draft or later
    counterfactual branch responses when reconstructing the preceding state.
    """
    payload = prefix_payload(row)
    paths = list(Path(source_root).glob(f"workers/*/tasks/*-{payload['task_id']}/actor/steps.jsonl"))
    if len(paths) != 1:
        raise PrefixReplayIntegrityError(f"expected one source actor trace, found {len(paths)}")
    path = paths[0]
    records = []
    digest = hashlib.sha256()
    found = False
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            if record.get("session_id") != payload["session_id"]:
                raise PrefixReplayIntegrityError("source actor trace contains a different session")
            if record.get("decision_key") == payload["decision_key"]:
                found = True
                break
            if record.get("status") != "ok" or not isinstance(record.get("response"), Mapping):
                raise PrefixReplayIntegrityError("source actor trace has an uncommitted decision")
            compact = {key: record[key] for key in ("session_id", "decision_key", "response")}
            records.append(compact)
            digest.update((_canonical(compact) + "\n").encode("utf-8"))
    if not found:
        raise PrefixReplayIntegrityError("source decision is absent from actor trace")
    result = copy.deepcopy(dict(row))
    result["_recorded_prefix_steps"] = records
    result["_source_trace"] = {"path": str(path), "prefix_sha256": digest.hexdigest(),
                               "prior_decisions": len(records)}
    return result


def response_message(message: Mapping[str, Any]) -> dict[str, Any]:
    """Convert a recorded assistant message to BFCLTaskEnvironment input."""
    value = _raw_message(message)
    if value.get("role") != "assistant":
        raise PrefixReplayIntegrityError("replayed response is not assistant")
    calls = []
    for call in value.get("tool_calls") or []:
        call = copy.deepcopy(dict(call))
        function = dict(call.get("function") or {})
        name = function.get("name")
        arguments = function.get("arguments")
        if not isinstance(name, str) or not name:
            raise PrefixReplayIntegrityError("recorded tool call has no name")
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments or {}, ensure_ascii=False,
                                   separators=(",", ":"))
        function["arguments"] = arguments
        call["function"] = function
        calls.append(call)
    return {"role": "assistant", "content": value.get("content"),
            **({"tool_calls": calls} if calls else {})}


def _tool_call_signature(call: Mapping[str, Any]) -> tuple[str, Any]:
    function = call.get("function") or {}
    args = function.get("arguments", {})
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            pass
    return str(function.get("name")), _jsonable(args)


def _same_tool_response(expected: Mapping[str, Any], actual: Mapping[str, Any]) -> bool:
    if expected.get("role") != "tool" or actual.get("role") != "tool":
        return False
    if expected.get("tool_call_id") != actual.get("tool_call_id"):
        return False
    return str(expected.get("content")) == str(actual.get("content"))


@dataclass
class ReplayedEnvironment:
    """A live BFCL environment restored by executing the recorded prefix."""

    env: Any
    bindings: Any
    task: Mapping[str, Any]
    ground_truth: Sequence[Sequence[str]]
    turn_index: int
    prefix_payload_sha256: str
    replayed_assistant_messages: int
    replayed_tool_messages: int
    previous_turn_valid: bool | None
    pending_payload: dict[str, Any]

    def current_payload(self) -> dict[str, Any]:
        """Return the decision already opened and validated during restore."""
        return copy.deepcopy(self.pending_payload)


def restore_bfcl_prefix(row: Mapping[str, Any], bindings: Any | None = None) -> ReplayedEnvironment:
    """Restore the source task state by replaying recorded assistant actions.

    The target model is not involved here.  BFCL executes every recorded
    action against a fresh official environment, and each resulting tool
    observation is checked against the source row.  A mismatch makes the row
    unknown instead of silently assigning a label from a different state.
    """
    payload = prefix_payload(row)
    if bindings is None:
        from t02_bfcl import OfficialBFCLBindings, BFCLTaskEnvironment
        bindings = OfficialBFCLBindings()
    else:
        BFCLTaskEnvironment = getattr(bindings, "environment_class", None)
        if BFCLTaskEnvironment is None:
            from t02_bfcl import BFCLTaskEnvironment
    # The NPU's pinned t02_bfcl helper predates the current BFCL package and
    # omits ``model_style`` in make_handler().  Patch that compatibility field
    # at the boundary; it does not change the recorded prefix or checker.
    if not hasattr(bindings, "model_style"):
        from bfcl_eval.constants.enums import ModelStyle
        bindings.model_style = ModelStyle.OPENAI_COMPLETIONS
    if not getattr(bindings, "_prefix_replay_make_handler_patched", False):
        make_handler = bindings.make_handler
        def _compat_make_handler(namespace: str):
            handler = make_handler(namespace)
            if not hasattr(handler, "model_style") and hasattr(bindings, "model_style"):
                handler.model_style = bindings.model_style
            return handler
        bindings.make_handler = _compat_make_handler
        bindings._prefix_replay_make_handler_patched = True
    task, ground_truth = bindings.load_task(payload["task_id"])
    env = BFCLTaskEnvironment(task, ground_truth, bindings=bindings,
                              namespace=f"prefix_replay_{payload['state_id'][:12]}")
    visible = payload["raw_visible"]
    target_turn, _ = decision_position(payload["decision_key"])
    trace = row.get("_recorded_prefix_steps")
    if not isinstance(trace, list):
        raise PrefixReplayIntegrityError(
            "recorded source trace is required; raw_visible is a compressed view, not an environment log")
    replayed_assistants = 0
    for record in trace:
        current = env.next_payload()
        if current["decision_key"] != record.get("decision_key"):
            raise PrefixReplayIntegrityError("source trace decision order differs from official environment")
        if record.get("session_id") != payload["session_id"]:
            raise PrefixReplayIntegrityError("source trace session differs from source row")
        env.commit_response(response_message(record["response"]))
        replayed_assistants += 1
    current = env.next_payload()
    if current["decision_key"] != payload["decision_key"]:
        raise PrefixReplayIntegrityError(
            f"replay decision {current['decision_key']} != source {payload['decision_key']}")
    full_messages = current["messages"]
    source_messages = [m for m in full_messages if m.get("role") != "system"]
    # Event IDs address original non-system message indices. A tool action event
    # also owns the contiguous tool observations following its assistant message.
    selected = []
    selected_indices = set()
    for source_id in payload["raw_source_ids"]:
        prefix = payload["session_id"] + ":m"
        if not source_id.startswith(prefix) or not source_id[len(prefix):].isdigit():
            raise PrefixReplayIntegrityError("raw source event identity differs from source session")
        index = int(source_id[len(prefix):])
        if index >= len(source_messages):
            raise PrefixReplayIntegrityError("raw source event index is outside reconstructed history")
        indices = [index]
        if source_messages[index].get("role") == "assistant":
            following = index + 1
            while following < len(source_messages) and source_messages[following].get("role") == "tool":
                indices.append(following)
                following += 1
        selected_indices.update(indices)
    selected = [source_messages[i] for i in sorted(selected_indices)]
    if [_semantic_message(m) for m in selected] != [_semantic_message(m) for m in visible]:
        raise PrefixReplayIntegrityError(
            "reconstructed source events and tool observations do not equal q.raw_visible")
    current_messages = [_semantic_message(m) for m in full_messages]
    goal = row.get("q", {}).get("goal")
    user_messages = [m for m in current_messages if m.get("role") == "user"]
    if not isinstance(goal, str) or not user_messages or user_messages[-1].get("content") != goal:
        raise PrefixReplayIntegrityError(
            "official environment current user does not equal recorded q.goal")
    turn = target_turn
    if env.turn_index != turn:
        raise PrefixReplayIntegrityError(
            f"environment turn {env.turn_index} != source turn {turn}")
    # The complete source log restores only the external environment/checker.
    # The target model must never receive omitted raw history, including on
    # later continuation decisions. Keep only system instructions and the
    # recorded visible view in the append-only inference history.
    projected = [copy.deepcopy(m) for m in full_messages if m.get("role") == "system"]
    projected.extend(copy.deepcopy(visible))
    env.inference_data["message"] = projected
    current["messages"] = copy.deepcopy(projected)
    return ReplayedEnvironment(
        env=env, bindings=bindings, task=task, ground_truth=ground_truth,
        turn_index=turn, prefix_payload_sha256=payload["prefix_payload_sha256"],
        replayed_assistant_messages=replayed_assistants,
        replayed_tool_messages=sum(m.get("role") == "tool" for m in source_messages),
        previous_turn_valid=env.previous_turn_valid(),
        pending_payload=current,
    )


def official_current_turn(replayed: ReplayedEnvironment) -> dict[str, Any]:
    """Score the source turn after the target A0 continuation."""
    result = replayed.bindings.score_turn_prefix(
        replayed.env.handler, replayed.env.all_model_response,
        replayed.ground_truth, replayed.task, replayed.turn_index)
    valid = result.get("valid")
    return {
        "turn_success": valid if type(valid) is bool else None,
        "checker": replayed.bindings.make_json_serializable(result),
        "status": "known" if type(valid) is bool else "unknown",
    }
