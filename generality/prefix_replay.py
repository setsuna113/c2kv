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

    def current_payload(self) -> dict[str, Any]:
        return self.env.next_payload()


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
    assistants = [m for m in visible if m.get("role") == "assistant"]
    tools = [m for m in visible if m.get("role") == "tool"]
    if not assistants and not tools and not payload["last_action_observation"]:
        # A first decision after one or more empty BFCL turns has no action or
        # observation to replay.  Advancing those empty turns is exact only in
        # this empty-prefix case; any non-empty missing state remains unknown.
        while env.turn_index < target_turn and not env.finished:
            env._finish_turn()  # noqa: SLF001 - source row explicitly has no action
    tool_index = 0
    index = 0
    # The source trace stores action-observation events, and may omit the
    # assistant's final natural-language stop for a BFCL turn.  A user message
    # therefore marks a turn boundary even when the preceding raw prefix ends
    # with a tool observation.  Advance the official environment at that
    # boundary instead of inventing a new model response.
    while index < len(visible):
        message = visible[index]
        if message.get("role") != "assistant":
            index += 1
            continue
        if env.finished:
            raise PrefixReplayIntegrityError("source prefix has actions after task end")
        before_turn = env.turn_index
        # The source trace may have captured this already-open decision slot.
        # Consume it directly; only open a new slot when the environment is
        # idle.  Calling next_payload() unconditionally is what produced
        # "Commit the outstanding BFCL response first" for turn-1/step-N.
        commit_recorded = getattr(env, "commit_recorded_response", None)
        if callable(commit_recorded):
            commit_recorded(response_message(message))
        else:
            if not getattr(env, "_awaiting_response", False):
                env.next_payload()
            env.commit_response(response_message(message))
        produced = [m for m in env.inference_data.get("message", [])
                    if m.get("role") == "tool"]
        expected_calls = len(message.get("tool_calls") or [])
        expected_slice = tools[tool_index:tool_index + expected_calls]
        actual_slice = produced[-expected_calls:] if expected_calls else []
        if len(actual_slice) != len(expected_slice) or any(
                not _same_tool_response(e, a)
                for e, a in zip(expected_slice, actual_slice)):
            raise PrefixReplayIntegrityError(
                f"environment observation differs for source state {payload['state_id']}")
        tool_index += expected_calls
        next_index = index + 1
        while next_index < len(visible) and visible[next_index].get("role") == "tool":
            next_index += 1
        if (next_index < len(visible) and visible[next_index].get("role") == "user"
                and env.turn_index == before_turn and not env.finished):
            env._finish_turn()  # noqa: SLF001 - exact BFCL source-turn boundary
        index = next_index
    if tool_index != len(tools):
        raise PrefixReplayIntegrityError("source prefix contains an orphan tool observation")
    current = env.next_payload()
    current_messages = [_semantic_message(m) for m in current["messages"]]
    visible_messages = [_semantic_message(m) for m in payload["raw_visible"]]
    found = any(
        current_messages[offset:offset + len(visible_messages)] == visible_messages
        for offset in range(len(current_messages) - len(visible_messages) + 1)
    )
    if not found:
        raise PrefixReplayIntegrityError(
            "official environment prefix does not equal q.raw_visible")
    goal = row.get("q", {}).get("goal")
    user_messages = [m for m in current_messages if m.get("role") == "user"]
    if not isinstance(goal, str) or not user_messages or user_messages[-1].get("content") != goal:
        raise PrefixReplayIntegrityError(
            "official environment current user does not equal recorded q.goal")
    turn = target_turn
    if env.turn_index != turn:
        raise PrefixReplayIntegrityError(
            f"environment turn {env.turn_index} != source turn {turn}")
    return ReplayedEnvironment(
        env=env, bindings=bindings, task=task, ground_truth=ground_truth,
        turn_index=turn, prefix_payload_sha256=payload["prefix_payload_sha256"],
        replayed_assistant_messages=len(assistants),
        replayed_tool_messages=len(tools),
        previous_turn_valid=env.previous_turn_valid(),
    )


def official_current_turn(replayed: ReplayedEnvironment) -> dict[str, Any]:
    """Score the source turn after the target A0 continuation."""
    result = replayed.bindings.score_turn_prefix(
        replayed.env.handler, replayed.env.all_model_response,
        replayed.ground_truth, replayed.task, replayed.turn_index)
    valid = result.get("valid")
    return {
        "turn_success": valid if type(valid) is bool else None,
        "checker": result,
        "status": "known" if type(valid) is bool else "unknown",
    }
