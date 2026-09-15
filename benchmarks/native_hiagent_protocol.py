"""Shared wire contract for the opt-in native HiAgent bridge."""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


REQUEST_SCHEMA = "a-native-hiagent-call-v1"
RECEIPT_SCHEMA = "a-native-hiagent-receipt-v1"
POLICY_SAMPLING_SCHEMA = "a-native-hiagent-policy-sampling-v1"
ENDPOINT_PATH = "/v1/native-hiagent/completions"
PHASES = frozenset({"compressor", "policy", "trajectory_retrieval_policy"})
OFFICIAL_CONTEXT_FIELDS = frozenset(
    {"benchmark", "task_id", "user_turn", "step", "attempt"}
)
POLICY_SAMPLING_FIELDS = frozenset(
    {"temperature", "seed", "max_completion_tokens", "top_p", "top_k", "min_p"}
)
REQUEST_CONTROL_FIELDS = frozenset({"tool_choice", "parallel_tool_calls"})
RECEIPT_FIELD = "native_hiagent_receipt"
RECEIPT_REQUIRED_FIELDS = frozenset({
    "schema",
    "parent_request_id",
    "official_eval_context",
    "proxy_attempt_uid",
    "native_attempt_uid",
    "phase",
    "call_ordinal",
    "native_session_id",
    "native_decision_key",
})


def _json_snapshot(value: Any) -> Any:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise ValueError("native HiAgent protocol values must be finite JSON") from error
    return json.loads(encoded)


def _nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a nonempty string")
    return value


def validate_official_eval_context(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != OFFICIAL_CONTEXT_FIELDS:
        raise ValueError(
            "official_eval_context must contain exactly benchmark, task_id, "
            "user_turn, step and attempt"
        )
    result = _json_snapshot(dict(value))
    _nonempty_string(result["benchmark"], "official_eval_context.benchmark")
    _nonempty_string(result["task_id"], "official_eval_context.task_id")
    for field in ("user_turn", "step", "attempt"):
        item = result[field]
        if type(item) is not int or item < 0:
            raise ValueError(f"official_eval_context.{field} must be a nonnegative integer")
    if result["attempt"] != 0:
        raise ValueError("official_eval_context.attempt must be zero")
    return result


def validate_policy_sampling(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("policy sampling must be a mapping")
    unknown = sorted(set(value) - POLICY_SAMPLING_FIELDS)
    if unknown:
        raise ValueError(f"unsupported policy sampling fields: {unknown!r}")
    missing = sorted({"temperature", "seed", "max_completion_tokens"} - set(value))
    if missing:
        raise ValueError(f"policy sampling is missing explicit fields: {missing!r}")
    result = _json_snapshot(dict(value))
    temperature = result["temperature"]
    if (
        isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or not math.isfinite(float(temperature))
        or float(temperature) < 0
    ):
        raise ValueError("policy sampling temperature must be finite and nonnegative")
    if type(result["seed"]) is not int:
        raise ValueError("policy sampling seed must be an integer")
    if type(result["max_completion_tokens"]) is not int or result["max_completion_tokens"] <= 0:
        raise ValueError("policy sampling max_completion_tokens must be a positive integer")
    for field in ("top_p", "min_p"):
        if field in result and (
            isinstance(result[field], bool)
            or not isinstance(result[field], (int, float))
            or not math.isfinite(float(result[field]))
        ):
            raise ValueError(f"policy sampling {field} must be finite numeric data")
    if "top_k" in result and type(result["top_k"]) is not int:
        raise ValueError("policy sampling top_k must be an integer")
    return result


def load_policy_sampling(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    value = json.loads(source.read_text(encoding="utf-8-sig"))
    if not isinstance(value, Mapping) or value.get("schema") != POLICY_SAMPLING_SCHEMA:
        raise ValueError(
            f"policy sampling file must use schema {POLICY_SAMPLING_SCHEMA!r}"
        )
    return validate_policy_sampling({key: item for key, item in value.items() if key != "schema"})


def _validate_messages_tools(payload: Mapping[str, Any]) -> tuple[list[dict], list[dict]]:
    messages = payload.get("messages")
    if (
        not isinstance(messages, Sequence)
        or isinstance(messages, (str, bytes, bytearray))
        or not messages
        or any(not isinstance(message, Mapping) for message in messages)
    ):
        raise ValueError("model call messages must be a nonempty sequence of mappings")
    tools = payload.get("tools", [])
    if tools is None:
        tools = []
    if (
        not isinstance(tools, Sequence)
        or isinstance(tools, (str, bytes, bytearray))
        or any(not isinstance(tool, Mapping) for tool in tools)
    ):
        raise ValueError("model call tools must be a sequence of mappings")
    return _json_snapshot(list(messages)), _json_snapshot(list(tools))


def _compressor_sampling(payload: Mapping[str, Any]) -> dict[str, Any]:
    max_tokens = payload.get("max_tokens")
    max_completion_tokens = payload.get("max_completion_tokens")
    if max_tokens is not None and max_completion_tokens is not None \
            and max_tokens != max_completion_tokens:
        raise ValueError("compressor max_tokens and max_completion_tokens disagree")
    cap = max_completion_tokens if max_completion_tokens is not None else max_tokens
    if type(cap) is not int or cap != 100:
        raise ValueError("native HiAgent compressor must preserve its 100-token cap")
    if payload.get("stop") != ["\n\n"]:
        raise ValueError("native HiAgent compressor must preserve stop=['\\n\\n']")
    sampling = {
        "temperature": payload.get("temperature"),
        "seed": payload.get("seed"),
        "max_completion_tokens": cap,
        "stop": _json_snapshot(payload["stop"]),
    }
    for field in ("top_p", "top_k", "min_p"):
        if field in payload:
            sampling[field] = _json_snapshot(payload[field])
    if sampling["temperature"] != 0.0:
        raise ValueError("native HiAgent compressor temperature must remain zero")
    if type(sampling["seed"]) is not int:
        raise ValueError("native HiAgent compressor seed must be an integer")
    return sampling


def build_model_call(
    payload: Mapping[str, Any],
    *,
    phase: str,
    policy_sampling: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError("payload must be a mapping")
    if phase not in PHASES:
        raise ValueError(f"phase must be one of {sorted(PHASES)}")
    model = _nonempty_string(payload.get("model"), "model")
    messages, tools = _validate_messages_tools(payload)
    sampling = (
        _compressor_sampling(payload)
        if phase == "compressor"
        else validate_policy_sampling(policy_sampling)
    )
    for field in REQUEST_CONTROL_FIELDS:
        if field in payload:
            sampling[field] = _json_snapshot(payload[field])
    return {
        "model": model,
        "messages": messages,
        "tools": tools,
        "sampling": sampling,
    }


def native_call_identity(
    official_eval_context: Mapping[str, Any],
    *,
    parent_request_id: str,
    proxy_attempt_uid: str,
    call_ordinal: int,
) -> tuple[str, str]:
    context = validate_official_eval_context(official_eval_context)
    _nonempty_string(parent_request_id, "parent_request_id")
    _nonempty_string(proxy_attempt_uid, "proxy_attempt_uid")
    if type(call_ordinal) is not int or call_ordinal <= 0:
        raise ValueError("call_ordinal must be a positive integer")
    digest = hashlib.sha256(
        json.dumps(
            [parent_request_id, proxy_attempt_uid, call_ordinal],
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()[:16]
    session_id = (
        f"{context['benchmark']}/{context['task_id']}/attempt-{context['attempt']}/"
        f"hiagent-call-{call_ordinal}-{digest}"
    )
    return session_id, "model-call-0"


def build_call_envelope(
    payload: Mapping[str, Any],
    *,
    parent_request_id: str,
    official_eval_context: Mapping[str, Any],
    proxy_attempt_uid: str,
    phase: str,
    call_ordinal: int,
    policy_sampling: Mapping[str, Any],
) -> dict[str, Any]:
    context = validate_official_eval_context(official_eval_context)
    parent_request_id = _nonempty_string(parent_request_id, "parent_request_id")
    proxy_attempt_uid = _nonempty_string(proxy_attempt_uid, "proxy_attempt_uid")
    session_id, decision_key = native_call_identity(
        context,
        parent_request_id=parent_request_id,
        proxy_attempt_uid=proxy_attempt_uid,
        call_ordinal=call_ordinal,
    )
    envelope = {
        "schema": REQUEST_SCHEMA,
        "parent_request_id": parent_request_id,
        "official_eval_context": context,
        "proxy_attempt_uid": proxy_attempt_uid,
        "phase": phase,
        "call_ordinal": call_ordinal,
        "native_session_id": session_id,
        "native_decision_key": decision_key,
        "model_call": build_model_call(
            payload, phase=phase, policy_sampling=policy_sampling
        ),
    }
    return validate_call_envelope(envelope)


def validate_call_envelope(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("native HiAgent call envelope must be a mapping")
    required = {
        "schema", "parent_request_id", "official_eval_context",
        "proxy_attempt_uid", "phase", "call_ordinal", "native_session_id",
        "native_decision_key", "model_call",
    }
    if set(value) != required or value.get("schema") != REQUEST_SCHEMA:
        raise ValueError("native HiAgent call envelope has an invalid schema or fields")
    result = _json_snapshot(dict(value))
    context = validate_official_eval_context(result["official_eval_context"])
    _nonempty_string(result["parent_request_id"], "parent_request_id")
    _nonempty_string(result["proxy_attempt_uid"], "proxy_attempt_uid")
    if result["phase"] not in PHASES:
        raise ValueError(f"phase must be one of {sorted(PHASES)}")
    if type(result["call_ordinal"]) is not int or result["call_ordinal"] <= 0:
        raise ValueError("call_ordinal must be a positive integer")
    expected_session, expected_decision = native_call_identity(
        context,
        parent_request_id=result["parent_request_id"],
        proxy_attempt_uid=result["proxy_attempt_uid"],
        call_ordinal=result["call_ordinal"],
    )
    if result["native_session_id"] != expected_session:
        raise ValueError("native_session_id does not match the call identity")
    if result["native_decision_key"] != expected_decision:
        raise ValueError("native_decision_key does not match the call identity")
    model_call = result["model_call"]
    if not isinstance(model_call, Mapping) or set(model_call) != {
        "model", "messages", "tools", "sampling"
    }:
        raise ValueError("model_call must contain model, messages, tools and sampling")
    _nonempty_string(model_call["model"], "model_call.model")
    _validate_messages_tools(model_call)
    if not isinstance(model_call["sampling"], Mapping):
        raise ValueError("model_call.sampling must be a mapping")
    return result


def validate_receipt(value: Any, envelope: Mapping[str, Any] | None = None) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not RECEIPT_REQUIRED_FIELDS <= set(value):
        raise ValueError("native HiAgent response lacks the required receipt fields")
    result = _json_snapshot(dict(value))
    if result["schema"] != RECEIPT_SCHEMA:
        raise ValueError("native HiAgent receipt has an invalid schema")
    for field in (
        "parent_request_id", "proxy_attempt_uid", "native_attempt_uid",
        "native_session_id", "native_decision_key",
    ):
        _nonempty_string(result[field], f"receipt.{field}")
    validate_official_eval_context(result["official_eval_context"])
    if result["phase"] not in PHASES:
        raise ValueError("native HiAgent receipt phase is invalid")
    if type(result["call_ordinal"]) is not int or result["call_ordinal"] <= 0:
        raise ValueError("native HiAgent receipt call_ordinal must be positive")
    if envelope is not None:
        request = validate_call_envelope(envelope)
        for field in (
            "parent_request_id", "official_eval_context", "proxy_attempt_uid",
            "phase", "call_ordinal", "native_session_id", "native_decision_key",
        ):
            if result[field] != request[field]:
                raise ValueError(f"native HiAgent receipt changes {field}")
    return result


def validate_response(value: Any, envelope: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("native HiAgent response must be an object")
    receipt = validate_receipt(value.get(RECEIPT_FIELD), envelope)
    choices = value.get("choices")
    if (
        not isinstance(choices, list)
        or not choices
        or not isinstance(choices[0], Mapping)
        or not isinstance(choices[0].get("message"), Mapping)
    ):
        raise ValueError("native HiAgent response must contain an OpenAI message choice")
    result = _json_snapshot(dict(value))
    result[RECEIPT_FIELD] = receipt
    return result


__all__ = [
    "ENDPOINT_PATH",
    "OFFICIAL_CONTEXT_FIELDS",
    "PHASES",
    "POLICY_SAMPLING_SCHEMA",
    "RECEIPT_FIELD",
    "RECEIPT_REQUIRED_FIELDS",
    "RECEIPT_SCHEMA",
    "REQUEST_SCHEMA",
    "build_call_envelope",
    "build_model_call",
    "load_policy_sampling",
    "native_call_identity",
    "validate_call_envelope",
    "validate_official_eval_context",
    "validate_policy_sampling",
    "validate_receipt",
    "validate_response",
]
