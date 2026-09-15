"""Typed, internal-only subgoal proposals for a held native action."""
from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

from history_memory.subgoals import _declaration_candidate


POLICY = "post-action-internal-v1"
TOOL_NAME = "record_subgoal_proposal"
DIRECTIVE = (
    "The environment action below has already been chosen. Do not change or execute it. "
    "Use only record_subgoal_proposal exactly once. Set status to \"propose\" with a concise "
    "one-line milestone when a new subgoal should be assigned to this action; otherwise set "
    "status to \"unassigned\" and subgoal to \"\". Do not call an environment tool and do not "
    "serialize a tool call in assistant content.\nHeld environment action (canonical JSON, "
    "transport IDs omitted): {held_action_json}"
)
INTERNAL_TOOL = {
    "type": "function",
    "function": {
        "description": (
            "Record one candidate subgoal for the already chosen environment action. "
            "This is controller state and is never executed."
        ),
        "name": TOOL_NAME,
        "parameters": {
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": ["propose", "unassigned"]},
                "subgoal": {
                    "type": "string",
                    "description": (
                        "A concise one-line milestone; use an empty string only when status "
                        "is unassigned."
                    ),
                },
            },
            "required": ["status", "subgoal"],
            "additionalProperties": False,
        },
        "strict": False,
    },
}
TOOL_CHOICE = {"type": "function", "function": {"name": TOOL_NAME}}


def _sorted(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _sorted(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_sorted(item) for item in value]
    return value


def canonical_held_action(message: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return ordered name/arguments only; transport ids are deliberately absent."""
    result = []
    for call in message.get("tool_calls") or []:
        function = call.get("function") or {}
        raw = function.get("arguments") or "{}"
        try:
            arguments = json.loads(raw) if isinstance(raw, str) else raw
        except json.JSONDecodeError:
            arguments = raw
        result.append({"name": function.get("name"), "arguments": _sorted(arguments)})
    return result


def action_sha256(message: Mapping[str, Any]) -> str:
    encoded = json.dumps(canonical_held_action(message), ensure_ascii=False,
                         sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def should_trigger(runtime_metadata: Mapping[str, Any], held: Mapping[str, Any]) -> bool:
    organization = runtime_metadata.get("subgoal_organization") or {}
    if organization.get("n_active") != 0 or not canonical_held_action(held):
        return False
    label, violation = _declaration_candidate(held)
    return label is None or violation is not None


def build_payload(actor_payload: Mapping[str, Any], held: Mapping[str, Any]) -> dict[str, Any]:
    """Reuse only the prepared B0 messages and the already-held native action."""
    action_json = json.dumps(canonical_held_action(held), ensure_ascii=False,
                             sort_keys=True, separators=(",", ":"))
    payload = {
        "model": actor_payload.get("model"),
        "messages": copy.deepcopy(actor_payload.get("messages") or []) + [{
            "role": "user", "content": DIRECTIVE.format(held_action_json=action_json)}],
        "tools": [copy.deepcopy(INTERNAL_TOOL)],
        "tool_choice": copy.deepcopy(TOOL_CHOICE),
        "stream": False,
        "temperature": 0.001,
        "seed": 0,
        "max_completion_tokens": 256,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    return {key: value for key, value in payload.items() if value is not None}


def parse_proposal(message: Mapping[str, Any]) -> dict[str, str]:
    if message.get("role") != "assistant":
        raise ValueError("Proposal response role must be assistant")
    calls = message.get("tool_calls")
    if not isinstance(calls, Sequence) or isinstance(calls, (str, bytes)) or len(calls) != 1:
        raise ValueError("Proposal response must contain exactly one tool call")
    function = calls[0].get("function") if isinstance(calls[0], Mapping) else None
    if not isinstance(function, Mapping) or function.get("name") != TOOL_NAME:
        raise ValueError("Proposal response used an unexpected tool")
    raw = function.get("arguments")
    try:
        arguments = json.loads(raw) if isinstance(raw, str) else raw
    except json.JSONDecodeError as error:
        raise ValueError("Proposal arguments are not valid JSON") from error
    if not isinstance(arguments, Mapping) or set(arguments) != {"status", "subgoal"}:
        raise ValueError("Proposal arguments do not match the frozen schema")
    status, subgoal = arguments["status"], arguments["subgoal"]
    if status not in {"propose", "unassigned"} or not isinstance(subgoal, str):
        raise ValueError("Proposal values do not match the frozen schema")
    if "\n" in subgoal or "\r" in subgoal:
        raise ValueError("Proposal subgoal must be one line")
    if status == "propose" and not subgoal.strip():
        raise ValueError("Proposed subgoal is empty")
    if status == "unassigned" and subgoal != "":
        raise ValueError("Unassigned proposal must use an empty subgoal")
    return {"status": status, "subgoal": subgoal.strip()}


def strict_candidate(held: Mapping[str, Any], proposal: Mapping[str, str]) -> dict[str, Any] | None:
    if proposal["status"] == "unassigned":
        return None
    candidate = copy.deepcopy(dict(held))
    candidate["role"] = "assistant"
    candidate["content"] = f"Subgoal: {proposal['subgoal']}"
    label, violation = _declaration_candidate(candidate)
    if violation is not None or label != proposal["subgoal"]:
        raise ValueError("Proposal was rejected by the production subgoal parser")
    return candidate


def project_returned_declaration(data: Any, normalized: dict[str, Any],
                                 declaration: str) -> bool:
    """Put an accepted declaration in the client-visible response, preserving calls."""
    choices = data.get("choices") if isinstance(data, dict) else None
    if (not isinstance(choices, list) or not choices
            or not isinstance(choices[0], dict)
            or not isinstance(choices[0].get("message"), dict)):
        return False
    before = action_sha256(normalized)
    normalized["content"] = declaration
    choices[0]["message"]["content"] = declaration
    if action_sha256(normalized) != before:
        raise ValueError("Subgoal declaration changed the environment action")
    return True
