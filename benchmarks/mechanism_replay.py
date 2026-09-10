#!/usr/bin/env python3
"""Replay exact or explicitly constructed requests through C2KV interventions.

This is a one-step mechanism diagnostic.  It calls the existing proxy
assembly functions and :class:`SglangBackend` directly; it does not start a
proxy server, run an official scorer, continue a tool trajectory, or retry a
failed HTTP request.  The JSONL output is append-only within a newly-created
file and records every extract, repair-extract, and policy-generation POST.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
import sys
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
from urllib import request as urlrequest
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit


# proxy.py intentionally supports direct script execution and imports its
# siblings as top-level modules.  Reuse that exact module rather than copying
# its assembly rules into this diagnostic runner.
BENCHMARKS_DIR = Path(__file__).resolve().parent
if str(BENCHMARKS_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS_DIR))

import proxy as proxy_mod  # noqa: E402
from arms import ARMS, Arm, get_arm, history_kv_spec  # noqa: E402
from backends.base import BackendError  # noqa: E402
from backends.sglang import SglangBackend  # noqa: E402
from checkpoint_profile import ProfileError, resolve_checkpoint_profile  # noqa: E402


SCHEMA_VERSION = "mechanism_replay_v1"
SUPPORTED_PROTECTION = "current_user_turn_raw"
GOAL_REMINDER_PREFIX = "\n\nCurrent user request (verbatim):\n"
PREVIOUS_CALL_PREFIX = "\n\nPrevious tool call (verbatim):\n"
STAGE2_ARM_ORDER = (
    "full",
    "latest_only",
    "c2kv4",
    "c2kv4_goal_reminder",
    "latest_only_goal_reminder",
    "c2kv4_current_user_turn_raw",
)
STAGE2_NEW_ARMS = frozenset({
    "latest_only",
    "c2kv4_goal_reminder",
    "latest_only_goal_reminder",
})
STAGE3_ARM_ORDER = (
    "full",
    "c2kv4_goal_reminder",
    "latest_only_goal_reminder",
    "c2kv4_goal_action_receipt",
    "latest_only_goal_action_receipt",
    "c2kv4_current_user_turn_raw",
)
STAGE3_NEW_ARMS = frozenset({
    "c2kv4_goal_action_receipt",
    "latest_only_goal_action_receipt",
})
STAGED_FEATURE_ARMS = STAGE2_NEW_ARMS | STAGE3_NEW_ARMS
STAGE3_RECEIPT_BASE = {
    "c2kv4_goal_action_receipt": "c2kv4_goal_reminder",
    "latest_only_goal_action_receipt": "latest_only_goal_reminder",
}
STAGE2_ARM_ROLES = {
    "full": "uncompressed_training_dialect_reference_before",
    "latest_only": "explicit_history_ablation_current_raw_input_only",
    "c2kv4": "gist_reference_from_stage1",
    "c2kv4_goal_reminder": (
        "gist_history_plus_verbatim_goal_in_current_raw_tool_result"),
    "latest_only_goal_reminder": (
        "history_ablation_with_identical_raw_goal_and_current_tool_result"),
    "c2kv4_current_user_turn_raw": (
        "identical_full_request_repeat_after_interventions_no_compression"),
}
INPUT_MODES = (
    "exact_saved_request_v1",
    "constructed_from_logged_transition_v1",
)
CONSTRUCTED_NOTE = (
    "Constructed from logged decoded calls and tool results; original assistant "
    "text and transport ids were not recorded, so this payload is not a "
    "historical replay."
)
EXPECTED_COMMON_CORPUS_SIZE = 35
RUNTIME_REQUIRED = (
    "kv_resident_tokens",
    "bytes_per_kv_token",
    "c2kv_query_proj",
    "c2kv_query_proj_effective",
    "c2kv_query_proj_source",
    "c2kv_query_proj_decode_verified",
    "c2kv_layout",
)


class ReplayError(RuntimeError):
    """Invalid frozen input or unsupported replay configuration."""


class BudgetError(ReplayError):
    """A frozen pilot budget would be exceeded."""


class RuntimeContractError(ReplayError):
    """The server answered, but omitted or contradicted required evidence."""


class HttpPostError(ReplayError):
    """One non-retried HTTP POST failed."""

    def __init__(self, path: str, status: int, detail: str):
        super().__init__(f"POST {path} failed ({status}): {detail[:1000]}")
        self.path = path
        self.status = status


@dataclass(frozen=True)
class PilotBudget:
    max_cases: int
    max_policy_generations: int
    max_http_requests: int
    max_wall_seconds: float


@dataclass(frozen=True)
class Regime:
    doc_packing: str
    max_doc_length: int
    max_doc_num: int
    query_projection: str


@dataclass(frozen=True)
class Intervention:
    name: str
    role: str
    arm: Arm
    protection: Optional[str] = None
    history_transform: Optional[str] = None
    goal_reminder: bool = False
    action_receipt: bool = False


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()


def _require_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ReplayError(f"{label} must be an integer >= {minimum}")
    return value


def _require_number(value: Any, label: str, *, positive: bool = False) -> float:
    if (not isinstance(value, (int, float)) or isinstance(value, bool)
            or not math.isfinite(float(value))
            or (positive and float(value) <= 0)):
        suffix = "positive " if positive else "finite "
        raise ReplayError(f"{label} must be a {suffix}number")
    return float(value)


def load_plan(path: Path) -> Dict[str, Any]:
    try:
        plan = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ReplayError(f"cannot read replay plan {path}: {error}") from error
    if not isinstance(plan, dict):
        raise ReplayError("replay plan must be a JSON object")
    if plan.get("input_mode") not in INPUT_MODES:
        raise ReplayError(f"plan.input_mode must be one of {INPUT_MODES}")
    pilot = plan.get("pilot")
    if not isinstance(pilot, dict):
        raise ReplayError("replay plan is missing object pilot")
    if pilot.get("automatic_http_retries") != 0:
        raise ReplayError("pilot.automatic_http_retries must be exactly 0")
    if pilot.get("stop_on_runtime_contract_failure") is not True:
        raise ReplayError("pilot.stop_on_runtime_contract_failure must be true")
    arms = pilot.get("arms")
    if not isinstance(arms, list) or not arms:
        raise ReplayError("pilot.arms must be a non-empty list")
    return plan


def plan_budget(plan: Mapping[str, Any]) -> PilotBudget:
    pilot = plan["pilot"]
    return PilotBudget(
        max_cases=_require_int(pilot.get("max_cases"), "pilot.max_cases", minimum=1),
        max_policy_generations=_require_int(
            pilot.get("max_policy_generations"),
            "pilot.max_policy_generations", minimum=1),
        max_http_requests=_require_int(
            pilot.get("max_http_requests"), "pilot.max_http_requests", minimum=1),
        max_wall_seconds=_require_number(
            pilot.get("max_wall_seconds"), "pilot.max_wall_seconds", positive=True),
    )


def plan_regime(plan: Mapping[str, Any]) -> Regime:
    pilot = plan["pilot"]
    packing = pilot.get("doc_packing")
    projection = pilot.get("query_projection")
    if packing not in proxy_mod.DOC_PACKINGS:
        raise ReplayError(
            f"pilot.doc_packing must be one of {proxy_mod.DOC_PACKINGS}")
    if projection not in ("base", "gist"):
        raise ReplayError("pilot.query_projection must be 'base' or 'gist'")
    return Regime(
        doc_packing=str(packing),
        max_doc_length=_require_int(
            pilot.get("max_doc_length"), "pilot.max_doc_length", minimum=1),
        max_doc_num=_require_int(
            pilot.get("max_doc_num"), "pilot.max_doc_num", minimum=1),
        query_projection=str(projection),
    )


def _arm_from_plan(item: Mapping[str, Any]) -> Intervention:
    name = item.get("name")
    role = item.get("role")
    if not isinstance(name, str) or not name:
        raise ReplayError("every pilot arm needs a non-empty name")
    if not isinstance(role, str) or not role:
        raise ReplayError(f"pilot arm {name!r} needs a non-empty role")

    if name in (
        "latest_only",
        "latest_only_goal_reminder",
        "latest_only_goal_action_receipt",
    ):
        return Intervention(
            name=name,
            role=role,
            arm=Arm(name=name, compress_history=False),
            history_transform="latest_only",
            goal_reminder=name != "latest_only",
            action_receipt=name.endswith("_goal_action_receipt"),
        )
    if name in ("c2kv4_goal_reminder", "c2kv4_goal_action_receipt"):
        return Intervention(
            name=name,
            role=role,
            arm=replace(get_arm("c2kv4"), name=name),
            goal_reminder=True,
            action_receipt=name.endswith("_goal_action_receipt"),
        )

    history = item.get("history_kv")
    protection = item.get("protection")
    base_name = item.get("base_arm")
    if history is not None:
        if not isinstance(history, dict):
            raise ReplayError(f"pilot arm {name!r}: history_kv must be an object")
        arm = Arm(name=name, compress_history=False, history_kv=dict(history))
        history_kv_spec(arm)  # validate the existing backend contract now
        if protection is not None or base_name is not None:
            raise ReplayError(
                f"pilot arm {name!r}: history_kv cannot be combined with protection")
        return Intervention(name=name, role=role, arm=arm)

    if protection is None and name.endswith("_current_user_turn_raw"):
        protection = SUPPORTED_PROTECTION
        base_name = name[:-len("_current_user_turn_raw")]
    if protection is not None:
        if protection != SUPPORTED_PROTECTION:
            raise ReplayError(
                f"pilot arm {name!r}: unsupported protection {protection!r}")
        if not isinstance(base_name, str) or not base_name:
            raise ReplayError(f"pilot arm {name!r}: protection needs base_arm")
        base = get_arm(base_name)
        if not base.compress_history or base.native_messages:
            raise ReplayError(
                f"pilot arm {name!r}: protection base must be a standard C2KV arm")
        return Intervention(
            name=name, role=role, arm=replace(base, name=name),
            protection=SUPPORTED_PROTECTION,
        )

    if name in ARMS:
        arm = get_arm(name)
    else:
        ratio_match = re.fullmatch(r"c2kv(\d+)", name)
        if ratio_match is None:
            raise ReplayError(f"pilot arm {name!r} is not registered or a c2kv<N> arm")
        arm = Arm(
            name=name, compress_history=True,
            ratio=int(ratio_match.group(1)),
            description="one-step standard C2KV mechanism replay",
        )
        arm.validate()
    if arm.text_policy or arm.repair or arm.recover or arm.gold_recovery:
        raise ReplayError(
            f"pilot arm {name!r} is outside one-step mechanism replay scope")
    return Intervention(name=name, role=role, arm=arm)


def plan_interventions(plan: Mapping[str, Any]) -> List[Intervention]:
    interventions = [_arm_from_plan(item) for item in plan["pilot"]["arms"]]
    names = [item.name for item in interventions]
    if len(names) != len(set(names)):
        raise ReplayError("pilot arm names must be unique")
    if not interventions or interventions[0].arm.name != "full":
        raise ReplayError("the first pilot arm must be full for paired comparisons")
    names_tuple = tuple(names)
    uses_staged_feature = any(name in STAGED_FEATURE_ARMS for name in names)
    if (uses_staged_feature
            and names_tuple not in (STAGE2_ARM_ORDER, STAGE3_ARM_ORDER)):
        raise ReplayError(
            "staged interventions must appear exactly as either "
            f"{STAGE2_ARM_ORDER} or {STAGE3_ARM_ORDER}")
    return interventions


def _validate_stage2_transform_metadata(plan: Mapping[str, Any]) -> None:
    transforms = plan.get("stage2_transforms")
    required = (
        "goal_source", "goal_reminder", "reminder_suffix_template",
        "latest_only", "audit", "identity_control",
    )
    if (not isinstance(transforms, dict)
            or any(not isinstance(transforms.get(key), str)
                   or not transforms.get(key) for key in required)):
        raise ReplayError(
            "stage2_transforms must contain the frozen non-empty text fields")
    if transforms["reminder_suffix_template"] != (
            GOAL_REMINDER_PREFIX + "{original_goal}"):
        raise ReplayError(
            "stage2 reminder_suffix_template differs from the implemented suffix")


def _validate_stage2_plan(plan: Mapping[str, Any]) -> None:
    arms = plan["pilot"]["arms"]
    actual_roles = {item.get("name"): item.get("role") for item in arms}
    if actual_roles != STAGE2_ARM_ROLES:
        raise ReplayError(
            "stage2 arm name/role mapping differs from the frozen protocol")
    _validate_stage2_transform_metadata(plan)


def _validate_stage3_plan(plan: Mapping[str, Any]) -> None:
    """Validate receipt metadata in addition to the retained stage2 goal gate."""
    _validate_stage2_transform_metadata(plan)
    transforms = plan.get("stage3_transforms")
    required = (
        "goal_source", "previous_tool_call_source", "action_receipt_prefix",
        "action_receipt_json", "appended_order", "latest_only",
        "pair_controls", "identity_control", "accounting",
    )
    if (not isinstance(transforms, dict)
            or any(not isinstance(transforms.get(key), str)
                   or not transforms.get(key) for key in required)):
        raise ReplayError(
            "stage3_transforms must contain the frozen non-empty text fields")
    if transforms["action_receipt_prefix"] != PREVIOUS_CALL_PREFIX:
        raise ReplayError(
            "stage3 action_receipt_prefix differs from the implemented prefix")


def _request_payload(case: Mapping[str, Any], input_mode: str) -> Dict[str, Any]:
    request_record = case.get("request")
    if not isinstance(request_record, dict):
        raise ReplayError("case request must be an object")
    expected_status = (
        "exact" if input_mode == "exact_saved_request_v1"
        else "constructed_from_logged_transition_v1")
    if request_record.get("status") != expected_status:
        raise ReplayError(
            f"case request.status must be {expected_status!r}, got "
            f"{request_record.get('status')!r}")
    payload = request_record.get("payload")
    if not isinstance(payload, dict):
        raise ReplayError(
            "exact case request.payload must be the full saved OpenAI body")
    messages = payload.get("messages")
    tools = payload.get("tools")
    if not isinstance(messages, list) or not messages:
        raise ReplayError("exact request payload needs a non-empty messages list")
    if not isinstance(tools, list):
        raise ReplayError("exact request payload needs a tools list (possibly empty)")
    if payload.get("stream") is True:
        raise ReplayError("stream=true cannot be replayed by this buffered diagnostic")
    if any(not isinstance(message, dict) for message in messages):
        raise ReplayError("every request message must be an object")
    if any(not isinstance(tool, dict) for tool in tools):
        raise ReplayError("every request tool must be an object")
    return copy.deepcopy(payload)


def _case_task_number(case: Mapping[str, Any]) -> Tuple[int, str]:
    numeric = case.get("task_numeric_id")
    task_id = case.get("task_id")
    if isinstance(numeric, int) and not isinstance(numeric, bool) and numeric >= 0:
        if not isinstance(task_id, (str, int)) or isinstance(task_id, bool):
            raise ReplayError("case task_id must be a string or integer")
        return numeric, str(task_id)
    identity = case.get("identity")
    if not isinstance(identity, dict):
        raise ReplayError("case identity must be an object")
    task_id = identity.get("task_id")
    if not isinstance(task_id, (str, int)) or isinstance(task_id, bool):
        raise ReplayError("case identity.task_id must be a string or integer")
    text = str(task_id)
    match = re.search(r"(\d+)(?!.*\d)", text)
    if match is None:
        raise ReplayError(
            f"case task_id {text!r} has no numeric official ID for pilot sorting")
    return int(match.group(1)), text


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _validate_constructed_request(
    case: Mapping[str, Any], payload: Mapping[str, Any], actual_fp: str,
) -> Dict[str, Any]:
    request_record = case["request"]
    original_fp = request_record.get("original_message_fingerprint")
    constructed_fp = request_record.get("constructed_message_fingerprint")
    if not _is_sha256(original_fp) or not _is_sha256(constructed_fp):
        raise ReplayError(
            f"case {case['case_id']!r} needs original and constructed fingerprints")
    if constructed_fp != actual_fp:
        raise ReplayError(
            f"case {case['case_id']!r} constructed fingerprint mismatch")
    if case.get("fingerprint") != original_fp:
        raise ReplayError(
            f"case {case['case_id']!r} top-level fingerprint is not the original fp")
    if constructed_fp == original_fp:
        raise ReplayError(
            f"case {case['case_id']!r} constructed input unexpectedly equals original fp")
    if request_record.get("non_equivalence") != CONSTRUCTED_NOTE:
        raise ReplayError(
            f"case {case['case_id']!r} lacks the frozen non-equivalence statement")
    construction = request_record.get("construction")
    if not isinstance(construction, dict):
        raise ReplayError(f"case {case['case_id']!r} lacks construction metadata")
    if construction.get("assistant_content") != "":
        raise ReplayError("constructed assistant content must be explicitly empty")
    expected_scheme = (
        "c2kv_constructed_<task_numeric_id>_r1_<zero_based_call_index>")
    if construction.get("tool_call_id_scheme") != expected_scheme:
        raise ReplayError("constructed tool_call_id_scheme differs from the frozen scheme")
    call_count = _require_int(
        construction.get("call_count"), "request.construction.call_count")
    result_count = _require_int(
        construction.get("tool_result_count"),
        "request.construction.tool_result_count")
    messages = payload["messages"]
    assistants = [message for message in messages
                  if message.get("role") == "assistant" and message.get("tool_calls")]
    calls = [call for message in assistants for call in message.get("tool_calls") or []]
    results = [message for message in messages if message.get("role") == "tool"]
    if len(calls) != call_count or len(results) != result_count:
        raise ReplayError("constructed call/result counts do not match payload")
    prefix = f"c2kv_constructed_{case['task_numeric_id']}_r1_"
    expected_ids = [f"{prefix}{index}" for index in range(call_count)]
    actual_ids = [call.get("id") for call in calls]
    if actual_ids != expected_ids:
        raise ReplayError("constructed tool call ids do not follow the frozen scheme")
    call_ids = set(actual_ids)
    if any(message.get("tool_call_id") not in call_ids for message in results):
        raise ReplayError("constructed tool result does not reference a constructed call")
    sources = request_record.get("sources")
    if not isinstance(sources, dict):
        raise ReplayError(f"case {case['case_id']!r} lacks request.sources")
    for key in ("full_first_proxy", "full_second_proxy", "full_result_transition"):
        source = sources.get(key)
        if not isinstance(source, dict):
            raise ReplayError(f"case {case['case_id']!r} lacks request.sources.{key}")
        for field in ("path", "line", "task_id", "user_turn", "step"):
            if source.get(field) is None:
                raise ReplayError(
                    f"case {case['case_id']!r} lacks request.sources.{key}.{field}")
        if str(source["task_id"]) != str(case["task_id"]):
            raise ReplayError(
                f"case {case['case_id']!r} request source task mismatch")
        expected_step = 1 if key == "full_second_proxy" else 0
        if source["user_turn"] != 0 or source["step"] != expected_step:
            raise ReplayError(
                f"case {case['case_id']!r} request.sources.{key} coordinate mismatch")
    return {
        "input_mode": "constructed_from_logged_transition_v1",
        "original_message_fingerprint": original_fp,
        "constructed_message_fingerprint": constructed_fp,
        "non_equivalence": CONSTRUCTED_NOTE,
    }


def validate_case(
    case: Mapping[str, Any], line_number: int, input_mode: str,
) -> Dict[str, Any]:
    if not isinstance(case, dict):
        raise ReplayError(f"case line {line_number} must be a JSON object")
    case_id = case.get("case_id")
    if not isinstance(case_id, str) or not case_id:
        raise ReplayError(f"case line {line_number} needs a non-empty case_id")
    expected_schema = (
        "c2kv.mechanism_case.v1" if input_mode == "exact_saved_request_v1"
        else "c2kv.mechanism_constructed_case.v1")
    if case.get("schema") != expected_schema:
        raise ReplayError(f"case {case_id!r} has unsupported schema")
    if case.get("label") != "preliminary, n=1":
        raise ReplayError(f"case {case_id!r} must be labelled preliminary, n=1")
    for key in ("actions", "provenance"):
        if not isinstance(case.get(key), dict):
            raise ReplayError(f"case {case_id!r} needs object {key}")
    # Current corpus schema keeps these fields at top level.  The older
    # draft nested identity/join form remains readable so an already-frozen
    # exact corpus does not need a cosmetic rewrite.
    identity = case.get("identity")
    if identity is None:
        full_source = ((case.get("actions") or {}).get("full_r1") or {}).get("source")
        full_source = full_source if isinstance(full_source, dict) else {}
        identity = {
            "task_id": case.get("task_id"),
            "task_numeric_id": case.get("task_numeric_id"),
            "request_ordinal": case.get("request_ordinal"),
            "turn_index": full_source.get("user_turn"),
            "step_index": full_source.get("step"),
        }
    if not isinstance(identity, dict):
        raise ReplayError(f"case {case_id!r} needs identity metadata")
    for key in ("task_id", "turn_index", "step_index"):
        if identity.get(key) is None:
            raise ReplayError(f"case {case_id!r} is missing identity.{key}")
    if identity["turn_index"] != 0 or identity["step_index"] != 1:
        raise ReplayError(
            f"case {case_id!r} must identify user_turn=0, step=1")
    provenance = case["provenance"]
    if not provenance:
        raise ReplayError(f"case {case_id!r} has empty provenance")
    payload = _request_payload(case, input_mode)
    if case.get("request_ordinal") != 1:
        raise ReplayError(f"case {case_id!r} must be request_ordinal=1")
    if (not isinstance(case.get("task_numeric_id"), int)
            or isinstance(case.get("task_numeric_id"), bool)
            or case["task_numeric_id"] < 0):
        raise ReplayError(f"case {case_id!r} needs non-negative task_numeric_id")
    actual_fp = proxy_mod.messages_fingerprint(payload["messages"])
    if (input_mode == "constructed_from_logged_transition_v1"
            and payload.get("c2kv_eval_context") != case.get("eval_context")):
        raise ReplayError(
            f"case {case_id!r} payload c2kv_eval_context differs from source metadata")
    join = case.get("join")
    if join is None:
        join = {"fp": case.get("fingerprint"),
                "eval_context": case.get("eval_context")}
    if not isinstance(join, dict):
        raise ReplayError(f"case {case_id!r} needs join metadata")
    expected_fp = join.get("fp")
    input_disclosure: Dict[str, Any]
    if input_mode == "constructed_from_logged_transition_v1":
        input_disclosure = _validate_constructed_request(case, payload, actual_fp)
        expected_fp = case["request"]["constructed_message_fingerprint"]
        join = {
            **join,
            "fp": expected_fp,
            "original_fp": case["request"]["original_message_fingerprint"],
        }
    else:
        input_disclosure = {
            "input_mode": "exact_saved_request_v1",
            "original_message_fingerprint": expected_fp,
            "constructed_message_fingerprint": None,
            "non_equivalence": None,
        }
    if not isinstance(expected_fp, str) or not expected_fp:
        raise ReplayError(f"case {case_id!r} is missing join.fp")
    if expected_fp != actual_fp:
        raise ReplayError(
            f"case {case_id!r} message fingerprint mismatch: "
            f"saved={expected_fp}, reconstructed={actual_fp}")
    if not isinstance(join.get("eval_context"), dict):
        raise ReplayError(f"case {case_id!r} is missing join.eval_context")
    eval_context = join["eval_context"]
    if (eval_context.get("attempt") != 0
            or eval_context.get("user_turn") != 0
            or eval_context.get("step") != 1
            or str(eval_context.get("task_id")) != str(identity["task_id"])):
        raise ReplayError(
            f"case {case_id!r} eval_context must be attempt=0, user_turn=0, step=1")
    request_record = case["request"]
    if (input_mode == "exact_saved_request_v1"
            and request_record.get("canonical_fingerprint_verified") is not True):
        raise ReplayError(f"case {case_id!r} has an unverified canonical fingerprint")
    snapshot = request_record.get("tools_snapshot")
    if not isinstance(snapshot, dict):
        raise ReplayError(f"case {case_id!r} request.tools_snapshot must be an object")
    snapshot_fields = ["path", "sha256", "task_id", "tool_count"]
    if input_mode == "constructed_from_logged_transition_v1":
        snapshot_fields.append("task_line")
    for key in snapshot_fields:
        if snapshot.get(key) is None:
            raise ReplayError(
                f"case {case_id!r} is missing request.tools_snapshot.{key}")
    if snapshot["tool_count"] != len(payload["tools"]):
        raise ReplayError(f"case {case_id!r} tool snapshot count mismatch")
    if not _is_sha256(snapshot.get("sha256")):
        raise ReplayError(f"case {case_id!r} tools snapshot sha256 is invalid")
    for arm_name in ("full_r1", "c2kv4"):
        raw_action = case["actions"].get(arm_name)
        action = _canonical_saved_action(raw_action)
        if action is None:
            raise ReplayError(f"case {case_id!r} has invalid actions.{arm_name}")
        source = raw_action.get("source") if isinstance(raw_action, dict) else None
        if not isinstance(source, dict):
            raise ReplayError(f"case {case_id!r} lacks actions.{arm_name}.source")
        for field in ("path", "line", "task_id", "user_turn", "step"):
            if source.get(field) is None:
                raise ReplayError(
                    f"case {case_id!r} lacks actions.{arm_name}.source.{field}")
        if str(source["task_id"]) != str(identity["task_id"]):
            raise ReplayError(f"case {case_id!r} action source task mismatch")
        if (source["user_turn"] != identity["turn_index"]
                or source["step"] != identity["step_index"]):
            raise ReplayError(f"case {case_id!r} action source coordinate mismatch")
    out = copy.deepcopy(case)
    out["identity"] = copy.deepcopy(identity)
    out["join"] = copy.deepcopy(join)
    out["_input_disclosure"] = input_disclosure
    out["_payload"] = payload
    out["_line"] = line_number
    out["_task_sort"] = _case_task_number(case)
    return out


def load_case_corpus(path: Path) -> List[Dict[str, Any]]:
    """Read corpus rows and only validate fields needed for frozen selection."""
    cases: List[Dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ReplayError(
                        f"invalid JSON on case line {line_number}: {error}") from error
                if not isinstance(value, dict):
                    raise ReplayError(f"case line {line_number} must be a JSON object")
                case_id = value.get("case_id")
                if not isinstance(case_id, str) or not case_id:
                    raise ReplayError(f"case line {line_number} needs a non-empty case_id")
                item = copy.deepcopy(value)
                item["_line"] = line_number
                item["_task_sort"] = _case_task_number(value)
                cases.append(item)
    except OSError as error:
        raise ReplayError(f"cannot read cases {path}: {error}") from error
    if not cases:
        raise ReplayError("case corpus is empty")
    ids = [case["case_id"] for case in cases]
    if len(ids) != len(set(ids)):
        raise ReplayError("case_id values must be unique")
    return cases


def load_cases(path: Path, input_mode: str) -> List[Dict[str, Any]]:
    """Validate every row; useful for a corpus expected to contain only runnable rows."""
    return [validate_case(case, int(case["_line"]), input_mode)
            for case in load_case_corpus(path)]


def select_pilot_cases(
    cases: Sequence[Dict[str, Any]], budget: PilotBudget,
) -> List[Dict[str, Any]]:
    return sorted(
        cases,
        key=lambda case: (
            case["_task_sort"],
            case["case_id"],
        ),
    )[:budget.max_cases]


def preflight_case_corpus(
    path: Path, input_mode: str, budget: PilotBudget, *,
    expected_corpus_size: int = EXPECTED_COMMON_CORPUS_SIZE,
) -> Tuple[List[Dict[str, Any]], int]:
    """Select first, then validate: missing selected input can never be backfilled."""
    corpus = load_case_corpus(path)
    if len(corpus) != expected_corpus_size:
        raise ReplayError(
            f"frozen common corpus must contain exactly {expected_corpus_size} rows; "
            f"found {len(corpus)}")
    task_numbers = [case["_task_sort"][0] for case in corpus]
    if len(task_numbers) != len(set(task_numbers)):
        raise ReplayError("frozen common corpus has duplicate numeric official task IDs")
    selected_raw = select_pilot_cases(corpus, budget)
    if len(selected_raw) != budget.max_cases:
        raise ReplayError(
            f"frozen pilot needs {budget.max_cases} selected cases; "
            f"found {len(selected_raw)}")
    selected = [
        validate_case(case, int(case["_line"]), input_mode)
        for case in selected_raw
    ]
    return selected, len(corpus)


def validate_cases_for_plan(
    cases: Sequence[Mapping[str, Any]], plan: Mapping[str, Any],
) -> None:
    decode = plan["pilot"].get("decode")
    if not isinstance(decode, dict):
        raise ReplayError("pilot.decode must record the recovered generation parameters")
    required = ("model", "temperature", "store", "max_completion_tokens")
    for field in required:
        if field not in decode:
            raise ReplayError(f"pilot.decode is missing {field}")
    for case in cases:
        payload = case["_payload"]
        conflicts = [field for field in required
                     if payload.get(field) != decode.get(field)]
        if conflicts:
            raise ReplayError(
                f"case {case['case_id']!r} generation parameters differ from "
                f"pilot.decode: {', '.join(conflicts)}")
        if "seed" in payload:
            raise ReplayError(
                f"case {case['case_id']!r} invents seed although the source sent none")
        forbidden = sorted(set(payload) & {
            "c2kv_oracle", "c2kv_gold", "c2kv_checker", "c2kv_state_answer",
        })
        if forbidden:
            raise ReplayError(
                f"case {case['case_id']!r} carries privileged fields {forbidden}")
    names = tuple(item.get("name") for item in plan["pilot"]["arms"])
    if names in (STAGE2_ARM_ORDER, STAGE3_ARM_ORDER):
        for case in cases:
            messages = case["_payload"]["messages"]
            target = locate_goal_and_tool_target(messages)
            tool_results = [message for message in messages
                            if message.get("role") == "tool"]
            if len(tool_results) != 1:
                raise ReplayError(
                    f"staged case {case['case_id']!r} must contain one first tool result")
            if names == STAGE3_ARM_ORDER:
                calls = [
                    call
                    for message in messages
                    if message.get("role") == "assistant"
                    for call in (message.get("tool_calls") or [])
                ]
                if len(calls) != 1:
                    raise ReplayError(
                        f"stage3 case {case['case_id']!r} must contain one tool call")
                c2_goal_index, latest_goal_index = 1, 2
            else:
                c2_goal_index, latest_goal_index = 3, 4
            c2_goal = _arm_from_plan(plan["pilot"]["arms"][c2_goal_index])
            latest_goal = _arm_from_plan(
                plan["pilot"]["arms"][latest_goal_index])
            c2_messages, _ = transform_intervention_messages(messages, c2_goal)
            latest_messages, latest_audit = transform_intervention_messages(
                messages, latest_goal)
            expected_tail = target["tool_content"] + GOAL_REMINDER_PREFIX + target["goal"]
            if c2_messages[target["tool_index"]].get("content") != expected_tail:
                raise ReplayError("c2kv4 goal reminder changed the frozen raw tail")
            latest_tool = next(
                (message for message in latest_messages if message.get("role") == "tool"),
                None)
            if not isinstance(latest_tool, dict) or latest_tool.get("content") != expected_tail:
                raise ReplayError("latest-only goal reminder differs from c2kv4 goal reminder")
            if not latest_audit["removed_original_message_indices"]:
                raise ReplayError("latest_only removed no history in a frozen staged case")

            if names == STAGE3_ARM_ORDER:
                c2_receipt = _arm_from_plan(plan["pilot"]["arms"][3])
                latest_receipt = _arm_from_plan(plan["pilot"]["arms"][4])
                c2_receipt_messages, c2_receipt_audit = (
                    transform_intervention_messages(messages, c2_receipt))
                latest_receipt_messages, latest_receipt_audit = (
                    transform_intervention_messages(messages, latest_receipt))
                receipt_json = json.dumps({
                    "name": target["previous_call_name"],
                    "arguments": target["previous_call_arguments"],
                }, ensure_ascii=False, separators=(",", ":"))
                expected_receipt_tail = (
                    target["tool_content"] + PREVIOUS_CALL_PREFIX + receipt_json
                    + GOAL_REMINDER_PREFIX + target["goal"])
                if (c2_receipt_messages[target["tool_index"]].get("content")
                        != expected_receipt_tail):
                    raise ReplayError(
                        "c2kv4 action receipt changed the frozen raw tail")
                latest_receipt_tool = next(
                    (message for message in latest_receipt_messages
                     if message.get("role") == "tool"), None)
                if (not isinstance(latest_receipt_tool, dict)
                        or latest_receipt_tool.get("content")
                        != expected_receipt_tail):
                    raise ReplayError(
                        "latest-only action receipt differs from c2kv4 action receipt")
                if not latest_receipt_audit["removed_original_message_indices"]:
                    raise ReplayError(
                        "latest_only receipt removed no history in a frozen stage3 case")
                if (c2_receipt_audit["previous_call_source"]
                        != latest_receipt_audit["previous_call_source"]):
                    raise ReplayError(
                        "stage3 receipt branches differ in previous-call provenance")

            protected, _ = protect_current_user_turn(messages, get_arm("c2kv4"))
            mapped, _ = _mapped_messages_for_cutoff(messages)
            cutoff = proxy_mod._history_cutoff(mapped)
            compressible = [
                index for index, message in enumerate(mapped)
                if (index < cutoff and message.get("role") != "system"
                    and not (protected.hybrid_top_k
                             and index >= cutoff - protected.hybrid_top_k))
            ]
            if compressible:
                raise ReplayError(
                    f"stage2 full repeat would still compress indices {compressible}")


def preflight_inputs(plan_path: Path, cases_path: Path) -> Dict[str, Any]:
    """Run the complete offline plan/corpus gate without creating an HTTP client."""
    plan = load_plan(plan_path)
    budget = plan_budget(plan)
    regime = plan_regime(plan)
    interventions = plan_interventions(plan)
    input_mode = str(plan["input_mode"])
    selected, corpus_count = preflight_case_corpus(
        cases_path, input_mode, budget)
    validate_cases_for_plan(selected, plan)
    planned_generations = len(selected) * len(interventions)
    if planned_generations > budget.max_policy_generations:
        raise BudgetError(
            f"selected matrix needs {planned_generations} policy generations, "
            f"budget is {budget.max_policy_generations}")
    if tuple(item.name for item in interventions) == STAGE2_ARM_ORDER:
        _validate_stage2_plan(plan)
        fixed = {
            "max_cases": 8,
            "max_policy_generations": 48,
            "max_http_requests": 128,
            "max_wall_seconds": 1800.0,
        }
        actual = {
            "max_cases": budget.max_cases,
            "max_policy_generations": budget.max_policy_generations,
            "max_http_requests": budget.max_http_requests,
            "max_wall_seconds": budget.max_wall_seconds,
        }
        if actual != fixed or planned_generations != 48:
            raise BudgetError(f"stage2 budget must be exactly {fixed}")
    if tuple(item.name for item in interventions) == STAGE3_ARM_ORDER:
        _validate_stage3_plan(plan)
        fixed = {
            "max_cases": 8,
            "max_policy_generations": 48,
            "max_http_requests": 128,
            "max_wall_seconds": 1800.0,
        }
        actual = {
            "max_cases": budget.max_cases,
            "max_policy_generations": budget.max_policy_generations,
            "max_http_requests": budget.max_http_requests,
            "max_wall_seconds": budget.max_wall_seconds,
        }
        if actual != fixed or planned_generations != 48:
            raise BudgetError(f"stage3 budget must be exactly {fixed}")
    return {
        "plan": plan,
        "budget": budget,
        "regime": regime,
        "interventions": interventions,
        "input_mode": input_mode,
        "selected_cases": selected,
        "case_corpus_count": corpus_count,
        "planned_policy_generations": planned_generations,
    }


def validate_profile_for_plan(
    profile: Mapping[str, Any], plan: Mapping[str, Any],
    regime: Regime, interventions: Sequence[Intervention],
) -> None:
    fingerprint = profile.get("profile_fingerprint")
    if not isinstance(fingerprint, str) or len(fingerprint) != 64:
        raise ReplayError("resolved checkpoint profile has no valid fingerprint")
    serving = profile.get("serving")
    if not isinstance(serving, dict):
        raise ReplayError("resolved checkpoint profile lacks serving metadata")
    expected = {
        "doc_packing": regime.doc_packing,
        "max_doc_length": regime.max_doc_length,
        "max_doc_num": regime.max_doc_num,
        "query_projection": regime.query_projection,
    }
    conflicts = [
        f"{key}: plan={value!r}, profile={serving.get(key)!r}"
        for key, value in expected.items() if serving.get(key) != value
    ]
    if conflicts:
        raise ReplayError(
            "pilot regime conflicts with checkpoint profile: " + "; ".join(conflicts))
    checkpoint_reference = plan["pilot"].get("checkpoint_reference")
    checkpoint_name = (profile.get("checkpoint") or {}).get("name")
    if (not isinstance(checkpoint_reference, str) or not checkpoint_reference
            or checkpoint_name != checkpoint_reference):
        raise ReplayError(
            "pilot.checkpoint_reference must exactly match resolved checkpoint name")
    supported_ratios = set(profile.get("training", {}).get("compression_ratios") or [])
    needed_ratios = {item.arm.ratio for item in interventions
                     if item.arm.compress_history}
    missing = sorted(needed_ratios - supported_ratios)
    if missing:
        raise ReplayError(
            f"checkpoint profile does not advertise compression ratios {missing}")


def _mapped_messages_for_cutoff(
    messages: Sequence[Mapping[str, Any]],
) -> Tuple[List[Dict[str, Any]], int]:
    mapped = [
        ({"role": "user", "content": proxy_mod._stringify_content(dict(message))}
         if message.get("role") == "tool" else dict(message))
        for message in messages
    ]
    inserted = 0
    if not any(message.get("role") == "system" for message in mapped):
        mapped.insert(0, {
            "role": "system", "content": proxy_mod.DEFAULT_SYSTEM_PROMPT,
        })
        inserted = 1
    return mapped, inserted


def protect_current_user_turn(
    messages: Sequence[Mapping[str, Any]], arm: Arm,
) -> Tuple[Arm, Dict[str, Any]]:
    """Protect the last original role=user turn with proxy hybrid semantics.

    The user boundary is found before proxy._assemble maps role=tool to
    role=user.  ``hybrid_top_k`` then keeps the interval from that real user
    through the proxy's ordinary current-block cutoff raw.  Existing packing,
    action rendering, and query projection remain untouched.
    """
    real_users = [index for index, message in enumerate(messages)
                  if message.get("role") == "user"]
    if not real_users:
        raise ReplayError("current-user-turn protection found no real user message")
    last_user = real_users[-1]
    mapped, inserted = _mapped_messages_for_cutoff(messages)
    mapped_user = last_user + inserted
    cutoff = proxy_mod._history_cutoff(mapped)
    hybrid_top_k = max(0, cutoff - mapped_user)
    protected = replace(arm, hybrid_top_k=hybrid_top_k)
    protected_indices = list(range(last_user, len(messages)))
    identity = hybrid_top_k == 0
    return protected, {
        "kind": SUPPORTED_PROTECTION,
        "last_real_user_index": last_user,
        "proxy_current_start_before_protection": cutoff,
        "hybrid_top_k": hybrid_top_k,
        "protected_original_message_indices": protected_indices,
        "protected_message_count": len(protected_indices),
        "matched_budget": False,
        "identity_before_assembly": identity,
    }


def locate_goal_and_tool_target(
    messages: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Find the real user goal and final tool result before role normalization."""
    user_indices = [index for index, message in enumerate(messages)
                    if message.get("role") == "user"]
    tool_indices = [index for index, message in enumerate(messages)
                    if message.get("role") == "tool"]
    if not user_indices:
        raise ReplayError("stage2 input has no original role=user goal")
    if not tool_indices:
        raise ReplayError("stage2 input has no original role=tool result")
    user_index = user_indices[-1]
    tool_index = tool_indices[-1]
    if tool_index <= user_index:
        raise ReplayError("stage2 final tool result does not follow the real user goal")
    goal = messages[user_index].get("content")
    tool_content = messages[tool_index].get("content")
    if not isinstance(goal, str) or not goal:
        raise ReplayError("stage2 real user goal must be a non-empty string")
    if not isinstance(tool_content, str):
        raise ReplayError("stage2 final tool result content must be a string")
    tool_call_id = messages[tool_index].get("tool_call_id")
    if not isinstance(tool_call_id, str) or not tool_call_id:
        raise ReplayError("stage2 final tool result lacks tool_call_id")
    targets = []
    for index, message in enumerate(messages[:tool_index]):
        if message.get("role") != "assistant":
            continue
        for call_index, call in enumerate(message.get("tool_calls") or []):
            if isinstance(call, dict) and call.get("id") == tool_call_id:
                targets.append((index, call_index))
    if len(targets) != 1:
        raise ReplayError(
            "stage2 final tool result must target exactly one prior assistant call")
    target_assistant_index, target_call_index = targets[0]
    call = messages[target_assistant_index]["tool_calls"][target_call_index]
    function = call.get("function")
    if not isinstance(function, dict):
        raise ReplayError("matched original assistant tool call lacks function object")
    previous_call_name = function.get("name")
    previous_call_arguments = function.get("arguments")
    if not isinstance(previous_call_name, str) or not previous_call_name:
        raise ReplayError("matched original assistant tool call lacks function.name")
    if not isinstance(previous_call_arguments, str):
        raise ReplayError(
            "matched original assistant tool call function.arguments must be a string")
    return {
        "goal_index": user_index,
        "goal": goal,
        "tool_index": tool_index,
        "tool_content": tool_content,
        "tool_call_id": tool_call_id,
        "target_assistant_index": target_assistant_index,
        "target_call_index": target_call_index,
        "previous_call_name": previous_call_name,
        "previous_call_arguments": previous_call_arguments,
    }


def transform_intervention_messages(
    messages: Sequence[Mapping[str, Any]], intervention: Intervention,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Apply frozen stage2/stage3 transforms without touching v1 arms."""
    original = [copy.deepcopy(dict(message)) for message in messages]
    transformed = copy.deepcopy(original)
    applies_stage2 = bool(
        intervention.history_transform or intervention.goal_reminder
        or intervention.action_receipt)
    target = locate_goal_and_tool_target(original) if applies_stage2 else None
    removed: List[int] = []
    original_to_transformed = {index: index for index in range(len(original))}

    if intervention.history_transform == "latest_only":
        mapped, inserted = _mapped_messages_for_cutoff(original)
        mapped_cutoff = proxy_mod._history_cutoff(mapped)
        original_cutoff = max(0, mapped_cutoff - inserted)
        system_head_end = 0
        while (system_head_end < len(original)
               and original[system_head_end].get("role") == "system"):
            system_head_end += 1
        kept = list(range(system_head_end)) + list(
            range(max(system_head_end, original_cutoff), len(original)))
        kept_set = set(kept)
        removed = [index for index in range(len(original)) if index not in kept_set]
        transformed = [copy.deepcopy(original[index]) for index in kept]
        original_to_transformed = {
            original_index: transformed_index
            for transformed_index, original_index in enumerate(kept)
        }
    elif intervention.history_transform is not None:
        raise ReplayError(
            f"unknown history transform {intervention.history_transform!r}")

    reminder: Dict[str, Any] = {
        "applied": False,
        "prefix": GOAL_REMINDER_PREFIX,
        "tool_message_index_original": target["tool_index"] if target else None,
        "tool_message_index_transformed": None,
        "appended_suffix": None,
    }
    previous_call_source = ({
        "assistant_message_index_original": target["target_assistant_index"],
        "tool_call_index_original": target["target_call_index"],
        "tool_result_message_index_original": target["tool_index"],
        "tool_call_id": target["tool_call_id"],
        "name": target["previous_call_name"],
        "arguments": target["previous_call_arguments"],
    } if target else None)
    receipt_json = None
    receipt_text = None
    if intervention.action_receipt:
        assert target is not None
        receipt_json = json.dumps({
            "name": target["previous_call_name"],
            "arguments": target["previous_call_arguments"],
        }, ensure_ascii=False, separators=(",", ":"))
        receipt_text = PREVIOUS_CALL_PREFIX + receipt_json
    action_receipt: Dict[str, Any] = {
        "applied": intervention.action_receipt,
        "prefix": PREVIOUS_CALL_PREFIX,
        "compact_json": receipt_json,
        "receipt_text": receipt_text,
        "appended_suffix": None,
    }
    if intervention.goal_reminder:
        assert target is not None
        transformed_tool_index = original_to_transformed.get(target["tool_index"])
        if transformed_tool_index is None:
            raise ReplayError("latest_only removed the tool result selected for reminder")
        tool_message = copy.deepcopy(transformed[transformed_tool_index])
        if tool_message.get("content") != target["tool_content"]:
            raise ReplayError("tool result changed before the goal reminder was applied")
        appended_suffix = (
            (receipt_text or "") + GOAL_REMINDER_PREFIX + target["goal"])
        tool_message["content"] = target["tool_content"] + appended_suffix
        transformed[transformed_tool_index] = tool_message
        reminder.update({
            "applied": True,
            "tool_message_index_transformed": transformed_tool_index,
            "append_offset_chars": len(target["tool_content"]),
            "appended_goal_chars": len(target["goal"]),
            "appended_suffix": appended_suffix,
            "result_content_sha256": hashlib.sha256(
                tool_message["content"].encode("utf-8")).hexdigest(),
        })
        if intervention.action_receipt:
            action_receipt["appended_suffix"] = appended_suffix

    audit = {
        "history_transform": intervention.history_transform,
        "original_messages_fingerprint": proxy_mod.messages_fingerprint(original),
        "transformed_messages_fingerprint": proxy_mod.messages_fingerprint(transformed),
        "removed_original_message_indices": removed,
        "kept_original_message_indices": [
            index for index in range(len(original)) if index not in set(removed)],
        "goal_source": ({
            "message_index_original": target["goal_index"],
            "role": "user",
            "content": target["goal"],
            "content_sha256": hashlib.sha256(
                target["goal"].encode("utf-8")).hexdigest(),
        } if target else None),
        "tool_target": ({
            "message_index_original": target["tool_index"],
            "tool_call_id": target["tool_call_id"],
            "target_assistant_index": target["target_assistant_index"],
            "target_call_index": target["target_call_index"],
            "original_content": target["tool_content"],
            "original_content_sha256": hashlib.sha256(
                target["tool_content"].encode("utf-8")).hexdigest(),
        } if target else None),
        "previous_call_source": previous_call_source,
        "action_receipt": action_receipt,
        "reminder": reminder,
        "matched_budget": False if applies_stage2 else None,
        "ablation": (
            "remove_all_history_keep_system_head_and_existing_raw_current_tail"
            if intervention.history_transform == "latest_only" else None),
    }
    return transformed, audit


@contextmanager
def _proxy_scope(regime: Regime, backend: SglangBackend):
    names = (
        "DOC_PACKING", "MAX_DOC_LENGTH", "MAX_DOC_NUM", "QUERY_PROJECTION",
        "BACKEND", "CACHE",
    )
    old = {name: getattr(proxy_mod, name) for name in names}
    proxy_mod.DOC_PACKING = regime.doc_packing
    proxy_mod.MAX_DOC_LENGTH = regime.max_doc_length
    proxy_mod.MAX_DOC_NUM = regime.max_doc_num
    proxy_mod.QUERY_PROJECTION = regime.query_projection
    proxy_mod.BACKEND = backend
    proxy_mod.CACHE = proxy_mod.ExtractCache()
    try:
        yield
    finally:
        for name, value in old.items():
            setattr(proxy_mod, name, value)


def _slim_counts(counts: Mapping[str, Any]) -> Dict[str, Any]:
    out = {key: value for key, value in counts.items()
           if key != "compressed_records"}
    out["compressed_records"] = [
        {
            "message_index": record.get("message_index"),
            "source_indices": record.get("source_indices"),
            "out_index": record.get("out_index"),
            "role": record.get("role"),
            "content_sha256": hashlib.sha256(
                str(record.get("content") or "").encode("utf-8")
            ).hexdigest(),
            "key_hash": (record.get("record") or {}).get("key_hash"),
            "gist_len": (record.get("record") or {}).get("gist_len"),
            "original_seq_len": (
                record.get("record") or {}).get("original_seq_len"),
        }
        for record in counts.get("compressed_records") or []
    ]
    return out


def prepare_trial(
    payload: Mapping[str, Any], intervention: Intervention,
    regime: Regime, post_json,
) -> Dict[str, Any]:
    """Prepare one request through the existing proxy and SGLang backend."""
    original_messages = copy.deepcopy(list(payload.get("messages") or []))
    arm = intervention.arm
    protection: Optional[Dict[str, Any]] = None
    messages, transform = transform_intervention_messages(
        original_messages, intervention)
    if intervention.protection == SUPPORTED_PROTECTION:
        arm, protection = protect_current_user_turn(messages, arm)
    backend = SglangBackend(post_json)
    started = time.perf_counter()
    with _proxy_scope(regime, backend):
        assembled, counts = proxy_mod._assemble(messages, arm)
        history_context = proxy_mod._history_kv_context(assembled, counts, arm)
        staged = copy.deepcopy(dict(payload))
        staged.pop("c2kv_eval_context", None)
        if staged.pop("c2kv_oracle", None) is not None:
            raise ReplayError("saved c2kv_oracle is outside mechanism replay scope")
        staged["messages"] = assembled
        staged["c2kv_use_gist_projection"] = regime.query_projection == "gist"
        prepared = backend.prepare_chat(
            staged, arm, None,
            context={
                "conversation_id": proxy_mod.conversation_id(messages),
                "history_kv": history_context,
                "kv_reuse": None,
            },
        )
    if protection is not None:
        no_compression = int(counts.get("n_gist_messages") or 0) == 0
        protection["result"] = (
            "identity_raw_no_compression_control" if no_compression
            else "older_history_compressed_current_user_turn_raw"
        )
        protection["no_compression"] = no_compression
    current_start = int(counts.get("current_start_out_index") or 0)
    raw_tail = copy.deepcopy(assembled[current_start:])
    return {
        "arm": arm,
        "prepared_request": prepared,
        "assembled_messages": assembled,
        "counts": counts,
        "counts_record": _slim_counts(counts),
        "history_context": history_context,
        "protection": protection,
        "input_transform": transform,
        "raw_tail_messages": raw_tail,
        "raw_tail_fingerprint": _json_sha256(raw_tail),
        "assemble_sec": time.perf_counter() - started,
        "backend": backend,
    }


class HttpRecorder:
    """Single-attempt JSON POST client with hard run-level budgets."""

    def __init__(self, upstream: str, budget: PilotBudget,
                 run_started: float, timeout: float):
        split = urlsplit(upstream)
        if (split.scheme not in ("http", "https") or not split.netloc
                or split.username or split.password or split.query or split.fragment):
            raise ReplayError("--upstream must be an http(s) base URL without credentials")
        self.upstream = upstream.rstrip("/")
        self.budget = budget
        self.run_started = run_started
        self.timeout = timeout
        self.events: List[Dict[str, Any]] = []
        self.policy_generations = 0
        self._opener = urlrequest.build_opener(urlrequest.ProxyHandler({}))

    def _remaining(self) -> float:
        return self.budget.max_wall_seconds - (time.perf_counter() - self.run_started)

    def post(self, path: str, payload: Dict[str, Any], timeout: int) -> Dict[str, Any]:
        if len(self.events) >= self.budget.max_http_requests:
            raise BudgetError("pilot.max_http_requests would be exceeded")
        is_policy = path.rstrip("/").endswith("/v1/chat/completions")
        if is_policy and self.policy_generations >= self.budget.max_policy_generations:
            raise BudgetError("pilot.max_policy_generations would be exceeded")
        remaining = self._remaining()
        if remaining <= 0:
            raise BudgetError("pilot.max_wall_seconds was reached before HTTP POST")
        effective_timeout = max(0.1, min(float(timeout), self.timeout, remaining))
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        event: Dict[str, Any] = {
            "ordinal": len(self.events) + 1,
            "path": path,
            "kind": "policy_generation" if is_policy else "auxiliary",
            "attempt": 1,
            "automatic_retries": 0,
            "request_sha256": hashlib.sha256(body).hexdigest(),
            "request_bytes": len(body),
            "timeout_sec": effective_timeout,
            "status": "started",
        }
        self.events.append(event)
        if is_policy:
            self.policy_generations += 1
        started = time.perf_counter()
        request = urlrequest.Request(
            f"{self.upstream}{path}", data=body,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        try:
            with self._opener.open(request, timeout=effective_timeout) as response:
                raw = response.read()
                event["http_status"] = response.status
        except HTTPError as error:
            raw = error.read()
            event.update({
                "status": "failed", "http_status": error.code,
                "wall_sec": time.perf_counter() - started,
                "response_bytes": len(raw),
                "error": raw.decode("utf-8", "replace")[:2000],
            })
            raise HttpPostError(path, error.code, event["error"]) from error
        except (URLError, OSError, TimeoutError) as error:
            event.update({
                "status": "failed", "http_status": 0,
                "wall_sec": time.perf_counter() - started,
                "response_bytes": 0, "error": str(error)[:2000],
            })
            raise HttpPostError(path, 0, str(error)) from error
        event["wall_sec"] = time.perf_counter() - started
        event["response_bytes"] = len(raw)
        try:
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            event.update({"status": "failed", "error": f"invalid JSON: {error}"})
            raise HttpPostError(path, int(event["http_status"]), event["error"]) from error
        if not isinstance(value, dict):
            event.update({"status": "failed", "error": "response is not a JSON object"})
            raise HttpPostError(path, int(event["http_status"]), event["error"])
        event["status"] = "ok"
        event["response_sha256"] = _json_sha256(value)
        if path.endswith("/extract"):
            event["extract_measurement"] = {
                key: value.get(key) for key in (
                    "success", "key_hash", "gist_len", "original_seq_len",
                    "token_len", "requested_span_tokens", "selected_token_count",
                    "history_kv_method", "history_selection_metadata",
                ) if key in value
            }
        if self._remaining() < 0:
            raise BudgetError("pilot.max_wall_seconds was exceeded during HTTP POST")
        return value


def _integer_field(mapping: Mapping[str, Any], key: str) -> int:
    value = mapping.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise RuntimeContractError(f"response {key} must be a non-negative integer")
    return value


def validate_runtime_measurement(
    data: Mapping[str, Any], normalized: Mapping[str, Any],
    prepared: Mapping[str, Any], intervention: Intervention,
    regime: Regime,
) -> Dict[str, Any]:
    """Validate and return the server-owned measurement block."""
    usage = data.get("usage")
    metadata = data.get("metadata")
    if not isinstance(usage, dict):
        raise RuntimeContractError("response is missing usage")
    prompt_tokens = _integer_field(usage, "prompt_tokens")
    completion_tokens = _integer_field(usage, "completion_tokens")
    if not isinstance(metadata, dict):
        raise RuntimeContractError("response is missing metadata")
    runtime = metadata.get("sglang_runtime")
    if not isinstance(runtime, dict):
        raise RuntimeContractError("response is missing metadata.sglang_runtime")
    absent = [key for key in RUNTIME_REQUIRED if key not in runtime]
    if absent:
        raise RuntimeContractError(
            "metadata.sglang_runtime lacks required fields: " + ", ".join(absent))
    if runtime.get("c2kv_query_proj_decode_verified") is not True:
        raise RuntimeContractError("decode query projection is not verified")
    expected_projection = (
        "base" if intervention.arm.history_kv else regime.query_projection)
    if runtime.get("c2kv_query_proj_effective") != expected_projection:
        raise RuntimeContractError(
            "effective query projection differs from the frozen intervention")
    if runtime.get("c2kv_injection_error"):
        raise RuntimeContractError(
            f"server reported C2KV injection error: {runtime['c2kv_injection_error']}")
    if not isinstance(runtime.get("kv_resident_tokens"), int):
        raise RuntimeContractError("server omitted integer kv_resident_tokens")
    unit_bytes = runtime.get("bytes_per_kv_token")
    if (not isinstance(unit_bytes, int) or isinstance(unit_bytes, bool)
            or unit_bytes <= 0):
        raise RuntimeContractError("server omitted positive integer bytes_per_kv_token")
    layout = runtime.get("c2kv_layout")
    if not isinstance(layout, list) or any(not isinstance(item, dict) for item in layout):
        raise RuntimeContractError("c2kv_layout must be a list of objects")

    counts = prepared["counts"]
    gist_rows = [item for item in layout if item.get("kind") == "gist"]
    repair_rows = [item for item in layout if item.get("kind") == "repair"]
    expected_gists = int(counts.get("n_gist_messages") or 0)
    if len(gist_rows) != expected_gists:
        raise RuntimeContractError(
            f"actual gist count {len(gist_rows)} != assembled {expected_gists}")
    actual_gist_tokens = sum(_integer_field(item, "gist_len") for item in gist_rows)
    if actual_gist_tokens != int(counts.get("gist_tokens") or 0):
        raise RuntimeContractError(
            "actual injected gist tokens differ from extract ledger")

    history_context = prepared.get("history_context")
    expected_raw_injection = bool(
        intervention.arm.history_kv and history_context
        and history_context.get("history_out_indices"))
    requested_raw_tokens: Optional[int] = None
    selected_raw_tokens: Optional[int] = None
    if expected_raw_injection:
        if not repair_rows:
            raise RuntimeContractError("raw-history control injected no repair span")
        report = metadata.get("kv_memory_report")
        if not isinstance(report, dict):
            raise RuntimeContractError("raw-history control lacks kv_memory_report")
        method = intervention.arm.history_kv["method"]
        if report.get("history_kv_method") != method:
            raise RuntimeContractError("raw-history method echo mismatch")
        requested = _integer_field(report, "history_kv_requested_span_tokens")
        selected = _integer_field(report, "history_kv_selected_token_count")
        requested_raw_tokens = requested
        selected_raw_tokens = selected
        spec = history_kv_spec(intervention.arm) or {}
        target = spec.get("target_tokens")
        if target is not None and selected != min(int(target), requested):
            raise RuntimeContractError("raw-history target token budget was not applied")
        if target is None and float(spec.get("retention_ratio") or 0) == 1.0:
            if selected != requested:
                raise RuntimeContractError("retention_ratio=1.0 did not retain the full raw span")
    elif repair_rows:
        raise RuntimeContractError("unexpected raw repair injection in this intervention")

    actual_repair_tokens = sum(
        _integer_field(item, "repair_len") for item in repair_rows)
    if (selected_raw_tokens is not None
            and actual_repair_tokens != selected_raw_tokens):
        raise RuntimeContractError(
            "actual injected raw tokens differ from history selection ledger")
    logical_prompt_tokens = (
        prompt_tokens + actual_gist_tokens + actual_repair_tokens)
    original_selected = counts.get("original_tokens")
    if not isinstance(original_selected, int) or isinstance(original_selected, bool):
        original_selected = None
    if requested_raw_tokens is not None:
        full_equivalent_history_tokens: Optional[int] = requested_raw_tokens
        active_history_tokens: Optional[int] = selected_raw_tokens
        history_source = "kv_memory_report.raw_history_selection"
    elif expected_gists:
        full_equivalent_history_tokens = original_selected
        active_history_tokens = actual_gist_tokens
        history_source = "proxy_extract_ledger_and_runtime_layout"
    else:
        full_equivalent_history_tokens = None
        active_history_tokens = None
        history_source = "unavailable_for_direct_raw_messages"

    return {
        "usage_raw": copy.deepcopy(usage),
        "prompt_tokens_raw": prompt_tokens,
        "completion_tokens": completion_tokens,
        "injection": {
            "layout": copy.deepcopy(layout),
            "gist_tokens_actual": actual_gist_tokens,
            "repair_tokens_actual": actual_repair_tokens,
            "gist_seen": copy.deepcopy(runtime.get("c2kv_gist_seen")),
            "position_correction": copy.deepcopy(
                runtime.get("c2kv_position_correction")),
            "history_kv_report": copy.deepcopy(metadata.get("kv_memory_report")),
        },
        "tensor_bytes": {
            "bytes_per_kv_token": unit_bytes,
            "logical_active_prompt_tokens": logical_prompt_tokens,
            "logical_active_prompt_bytes": logical_prompt_tokens * unit_bytes,
            "full_equivalent_selected_history_tokens": full_equivalent_history_tokens,
            "full_equivalent_selected_history_bytes": (
                full_equivalent_history_tokens * unit_bytes
                if full_equivalent_history_tokens is not None else None),
            "active_history_tokens": active_history_tokens,
            "active_history_bytes": (
                active_history_tokens * unit_bytes
                if active_history_tokens is not None else None),
            "history_source": history_source,
            "scope": (
                "history_KV_tensor_payload_and_logical_active_prompt_only; "
                "excludes page rounding, pool duplication, indexes, extraction "
                "prefill and model or allocator memory"
            ),
        },
        "decode": {
            "finish_reason": normalized.get("finish_reason"),
            "content": normalized.get("content"),
            "tool_calls": copy.deepcopy(normalized.get("tool_calls")),
            "query_projection_flag": runtime.get("c2kv_query_proj"),
            "query_projection_effective": runtime.get(
                "c2kv_query_proj_effective"),
            "query_projection_source": runtime.get("c2kv_query_proj_source"),
            "query_projection_decode_verified": runtime.get(
                "c2kv_query_proj_decode_verified"),
        },
        "runtime": copy.deepcopy(runtime),
        "normalized_cost": copy.deepcopy(normalized.get("cost") or {}),
    }


def _canonical_saved_action(value: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(value, dict):
        return None
    for key in ("action", "canonical_action", "response_action"):
        if isinstance(value.get(key), dict):
            value = value[key]
            break
    if value.get("kind") in ("tool_calls", "text"):
        calls = value.get("tool_calls") or []
        if not isinstance(calls, list) or any(not isinstance(call, dict) for call in calls):
            return None
        return {
            "kind": value["kind"],
            "tool_calls": [{
                "name": call.get("name"),
                "arguments_raw": call.get("arguments_raw"),
                "arguments": copy.deepcopy(call.get("arguments")),
                "arguments_json_valid": call.get("arguments_json_valid") is True,
            } for call in calls],
            "text": value.get("text"),
        }
    if "tool_calls" in value or "content" in value or "text" in value:
        return _live_action({
            "tool_calls": value.get("tool_calls"),
            "content": value.get("content", value.get("text")),
        })
    return None


def _live_action(message: Mapping[str, Any]) -> Dict[str, Any]:
    calls = message.get("tool_calls") or []
    normalized_calls = []
    for call in calls:
        function = (call or {}).get("function") or {}
        raw = function.get("arguments")
        raw = raw if isinstance(raw, str) else json.dumps(
            raw if raw is not None else {}, ensure_ascii=False,
            separators=(",", ":"))
        try:
            arguments = json.loads(raw)
            valid = True
        except json.JSONDecodeError:
            arguments = None
            valid = False
        normalized_calls.append({
            "name": function.get("name"),
            "arguments_raw": raw,
            "arguments": arguments,
            "arguments_json_valid": valid,
        })
    return {
        "kind": "tool_calls" if normalized_calls else "text",
        "tool_calls": normalized_calls,
        "text": message.get("content"),
    }


def _agreement(action: Mapping[str, Any], reference: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    if reference is None:
        return {"status": "reference_unavailable", "action_match": None,
                "action_raw_match": None, "kind_match": None,
                "tool_calls_match": None, "text_match": None}
    kind_match = action.get("kind") == reference.get("kind")
    text_match = action.get("text") == reference.get("text")
    canonical_calls = [
        {"name": call.get("name"), "arguments": call.get("arguments")}
        for call in action.get("tool_calls") or []
    ]
    canonical_reference = [
        {"name": call.get("name"), "arguments": call.get("arguments")}
        for call in reference.get("tool_calls") or []
    ]
    raw_calls = [
        {"name": call.get("name"), "arguments_raw": call.get("arguments_raw")}
        for call in action.get("tool_calls") or []
    ]
    raw_reference = [
        {"name": call.get("name"), "arguments_raw": call.get("arguments_raw")}
        for call in reference.get("tool_calls") or []
    ]
    calls_match = canonical_calls == canonical_reference
    calls_raw_match = raw_calls == raw_reference
    action_match = kind_match and (
        calls_match if action.get("kind") == "tool_calls" else text_match)
    raw_match = kind_match and (
        calls_raw_match if action.get("kind") == "tool_calls" else text_match)
    return {
        "status": "compared",
        "action_match": action_match,
        "action_raw_match": raw_match,
        "kind_match": kind_match,
        "tool_calls_match": calls_match,
        "text_match": text_match,
    }


def _error_kind(error: BaseException) -> str:
    if isinstance(error, BackendError):
        return str(error.kind)
    if isinstance(error, RuntimeContractError):
        return "runtime_contract_failure"
    if isinstance(error, BudgetError):
        return "budget_failure"
    if isinstance(error, HttpPostError):
        return "http_failure"
    if isinstance(error, ReplayError):
        return "replay_failure"
    return type(error).__name__


def _require_identical_artifact(
    label: str, reference: Any, candidate: Any,
) -> Dict[str, Any]:
    reference_sha256 = _json_sha256(reference)
    candidate_sha256 = _json_sha256(candidate)
    matched = reference == candidate
    result = {
        "status": "compared",
        "matched": matched,
        "reference_sha256": reference_sha256,
        "candidate_sha256": candidate_sha256,
    }
    result["failure_message"] = (
        None if matched else f"{label} identity check failed")
    return result


def _validate_goal_raw_tail(prepared: Mapping[str, Any]) -> None:
    transform = prepared["input_transform"]
    reminder = transform["reminder"]
    receipt = transform["action_receipt"]
    goal_source = transform["goal_source"]
    tool_target = transform["tool_target"]
    raw_tail = prepared["raw_tail_messages"]
    expected_suffix = (
        (receipt.get("receipt_text") or "") + GOAL_REMINDER_PREFIX
        + goal_source["content"])
    expected_content = tool_target["original_content"] + expected_suffix
    if (len(raw_tail) != 1 or raw_tail[0].get("role") != "user"
            or raw_tail[0].get("content") != expected_content
            or reminder.get("append_offset_chars")
            != len(tool_target["original_content"])
            or reminder.get("appended_suffix") != expected_suffix
            or (receipt.get("applied")
                and receipt.get("appended_suffix") != expected_suffix)
            or hashlib.sha256(expected_content.encode("utf-8")).hexdigest()
            != reminder.get("result_content_sha256")):
        raise RuntimeContractError(
            "goal intervention did not preserve the exact result, cue, and raw tail")


def run_trial(
    case: Mapping[str, Any], intervention: Intervention,
    regime: Regime, http: HttpRecorder,
    full_action: Optional[Mapping[str, Any]], run_started: float,
    *, expected_full_request: Optional[Mapping[str, Any]] = None,
    expected_goal_raw_tail: Optional[Sequence[Mapping[str, Any]]] = None,
    expected_receipt_raw_tail: Optional[Sequence[Mapping[str, Any]]] = None,
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    event_start = len(http.events)
    started = time.perf_counter()
    base: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "row_type": "trial",
        "trial_id": uuid.uuid4().hex,
        "status": "started",
        "label": "preliminary, n=1",
        "scope": "one-step mechanism diagnostic; not official full-task or turn success",
        "case_id": case["case_id"],
        "identity": copy.deepcopy(case["identity"]),
        "join": copy.deepcopy(case["join"]),
        "provenance": copy.deepcopy(case["provenance"]),
        "saved_actions": copy.deepcopy(case["actions"]),
        "full_task_success": copy.deepcopy(case.get("full_task_success")),
        "full_task_success_source": copy.deepcopy(
            case.get("full_task_success_source")),
        "intervention": {
            "name": intervention.name,
            "role": intervention.role,
            "base_arm": intervention.arm.name,
            "protection": intervention.protection,
            "history_transform": intervention.history_transform,
            "goal_reminder": intervention.goal_reminder,
            "action_receipt": intervention.action_receipt,
            "history_kv": copy.deepcopy(intervention.arm.history_kv),
            "matched_budget_claim": False,
        },
        "request_sha256": _json_sha256(case["_payload"]),
        "input": copy.deepcopy(case["_input_disclosure"]),
    }
    try:
        prepared = prepare_trial(
            case["_payload"], intervention, regime, http.post)
        base.update({
            "prepared_request": prepared["prepared_request"],
            "prepared_request_sha256": _json_sha256(
                prepared["prepared_request"]),
            "assembly": prepared["counts_record"],
            "protection": prepared["protection"],
            "input_transform": prepared["input_transform"],
            "raw_tail_messages": prepared["raw_tail_messages"],
            "raw_tail_fingerprint": prepared["raw_tail_fingerprint"],
            "stage2_controls": {},
        })
        if intervention.goal_reminder:
            _validate_goal_raw_tail(prepared)
        if expected_full_request is not None:
            control = _require_identical_artifact(
                "stage2 full begin/end prepared request",
                expected_full_request, prepared["prepared_request"])
            base["stage2_controls"]["full_repeat_prepared_request"] = control
            if not control["matched"]:
                raise RuntimeContractError(str(control["failure_message"]))
        if expected_goal_raw_tail is not None:
            control = _require_identical_artifact(
                "stage2 reminder-arm raw tail",
                expected_goal_raw_tail, prepared["raw_tail_messages"])
            base["stage2_controls"]["goal_reminder_raw_tail"] = control
            if not control["matched"]:
                raise RuntimeContractError(str(control["failure_message"]))
        if expected_receipt_raw_tail is not None:
            control = _require_identical_artifact(
                "stage3 action-receipt-arm raw tail",
                expected_receipt_raw_tail, prepared["raw_tail_messages"])
            base["stage2_controls"]["goal_action_receipt_raw_tail"] = control
            if not control["matched"]:
                raise RuntimeContractError(str(control["failure_message"]))
        data = http.post(
            "/v1/chat/completions", prepared["prepared_request"], 600)
        base["raw_response"] = copy.deepcopy(data)
        normalized = prepared["backend"].normalize_response(data)
        measurement = validate_runtime_measurement(
            data, normalized, prepared, intervention, regime)
        action = _live_action({
            "content": normalized.get("content"),
            "tool_calls": normalized.get("tool_calls"),
        })
        saved_full = _canonical_saved_action(case["actions"].get("full_r1"))
        base.update({
            "status": "ok",
            "measurement": measurement,
            "action": action,
            "paired_full_agreement": _agreement(action, full_action),
            "legacy_full_descriptive_agreement": {
                "scope": (
                    "descriptive reference only; not historical replay, "
                    "gold correctness, or BFCL success"),
                **_agreement(action, saved_full),
            },
        })
        if intervention.arm.name == "full":
            full_action = action
            base["paired_full_agreement"] = _agreement(action, action)
    except Exception as error:
        base.update({
            "status": "failed", "error_kind": _error_kind(error),
            "error": str(error),
        })
    base["timing"] = {
        "assemble_sec": prepared.get("assemble_sec") if "prepared" in locals() else None,
        "trial_wall_sec": time.perf_counter() - started,
        "run_elapsed_sec": time.perf_counter() - run_started,
    }
    base["http"] = copy.deepcopy(http.events[event_start:])
    base["http_request_count"] = len(base["http"])
    base["policy_generation_count"] = sum(
        event.get("kind") == "policy_generation" for event in base["http"])
    return base, full_action


def _append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")
        handle.flush()


def _raw_prompt_cost_control(
    reference_row: Mapping[str, Any], candidate_row: Mapping[str, Any],
    reference_intervention: str, scope: str,
) -> Dict[str, Any]:
    reference_tokens = _integer_field(
        reference_row["measurement"], "prompt_tokens_raw")
    candidate_tokens = _integer_field(
        candidate_row["measurement"], "prompt_tokens_raw")
    return {
        "status": "measured",
        "reference_intervention": reference_intervention,
        "reference_prompt_tokens_raw": reference_tokens,
        "candidate_prompt_tokens_raw": candidate_tokens,
        "prompt_tokens_delta": candidate_tokens - reference_tokens,
        "scope": scope,
        "matched_final_budget": False,
    }


def execute(args: argparse.Namespace) -> int:
    preflight = preflight_inputs(args.plan, args.cases)
    plan = preflight["plan"]
    budget = preflight["budget"]
    regime = preflight["regime"]
    interventions = preflight["interventions"]
    input_mode = preflight["input_mode"]
    selected = preflight["selected_cases"]
    corpus_count = preflight["case_corpus_count"]
    planned_generations = preflight["planned_policy_generations"]
    is_stage2 = tuple(item.name for item in interventions) == STAGE2_ARM_ORDER
    is_stage3 = tuple(item.name for item in interventions) == STAGE3_ARM_ORDER
    if is_stage3 and plan.get("dispatch_authorized") is not True:
        raise ReplayError(
            "stage3 protocol is offline preparation and is not authorized for dispatch")
    try:
        profile = resolve_checkpoint_profile(
            args.checkpoint, profile_path=args.checkpoint_profile,
            require_serving_e2e=True)
    except ProfileError as error:
        raise ReplayError(str(error)) from error
    validate_profile_for_plan(profile, plan, regime, interventions)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    try:
        with args.out.open("x", encoding="utf-8"):
            pass
    except FileExistsError as error:
        raise ReplayError(f"refusing to overwrite existing output {args.out}") from error

    run_id = uuid.uuid4().hex
    run_started = time.perf_counter()
    http = HttpRecorder(args.upstream, budget, run_started, args.timeout)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "row_type": "run_manifest",
        "run_id": run_id,
        "status": "started",
        "plan_path": str(args.plan.resolve()),
        "plan_sha256": _json_sha256(plan),
        "cases_path": str(args.cases.resolve()),
        "input_mode": input_mode,
        "case_corpus_count": corpus_count,
        "selected_case_ids": [case["case_id"] for case in selected],
        "selection": "first max_cases sorted by numeric official task ID",
        "interventions": [item.name for item in interventions],
        "planned_policy_generations": planned_generations,
        "budget": budget.__dict__,
        "automatic_http_retries": 0,
        "failure_policy": "stop the pilot after the first failed trial",
        "smoke": "the first selected case is the smoke; no extra generation",
        "regime": regime.__dict__,
        "checkpoint_profile": copy.deepcopy(profile),
        "claim_limits": copy.deepcopy(plan.get("claim_limits") or []),
        "stage2_transforms": (
            copy.deepcopy(plan.get("stage2_transforms"))
            if (is_stage2 or is_stage3) else None),
        "stage3_transforms": (
            copy.deepcopy(plan.get("stage3_transforms")) if is_stage3 else None),
    }
    _append_jsonl(args.out, manifest)

    completed = 0
    failed: Optional[Dict[str, Any]] = None
    full_identity_verified = 0
    goal_tail_identity_verified = 0
    receipt_tail_identity_verified = 0
    for case in selected:
        full_action: Optional[Dict[str, Any]] = None
        first_full_request: Optional[Dict[str, Any]] = None
        c2kv_goal_raw_tail: Optional[List[Dict[str, Any]]] = None
        c2kv_receipt_raw_tail: Optional[List[Dict[str, Any]]] = None
        case_rows: Dict[str, Dict[str, Any]] = {}
        for intervention in interventions:
            expected_full_request = (
                first_full_request
                if (is_stage2 or is_stage3)
                and intervention.name == "c2kv4_current_user_turn_raw"
                else None)
            expected_goal_raw_tail = (
                c2kv_goal_raw_tail
                if (is_stage2 or is_stage3)
                and intervention.name == "latest_only_goal_reminder"
                else None)
            expected_receipt_raw_tail = (
                c2kv_receipt_raw_tail
                if is_stage3
                and intervention.name == "latest_only_goal_action_receipt"
                else None)
            row, full_action = run_trial(
                case, intervention, regime, http, full_action, run_started,
                expected_full_request=expected_full_request,
                expected_goal_raw_tail=expected_goal_raw_tail,
                expected_receipt_raw_tail=expected_receipt_raw_tail)
            if is_stage3:
                row["stage3_controls"] = row.pop("stage2_controls")
            row_controls = (
                row["stage3_controls"] if is_stage3 else row["stage2_controls"])
            if row["status"] == "ok":
                if (is_stage2 or is_stage3) and intervention.name == "full":
                    first_full_request = copy.deepcopy(row["prepared_request"])
                elif ((is_stage2 or is_stage3)
                      and intervention.name == "c2kv4_goal_reminder"):
                    c2kv_goal_raw_tail = copy.deepcopy(row["raw_tail_messages"])
                elif (is_stage3
                      and intervention.name == "c2kv4_goal_action_receipt"):
                    c2kv_receipt_raw_tail = copy.deepcopy(row["raw_tail_messages"])

                reminder_reference = {
                    "c2kv4_goal_reminder": "c2kv4",
                    "latest_only_goal_reminder": "latest_only",
                }.get(intervention.name)
                if is_stage2 and reminder_reference is not None:
                    reference_row = case_rows.get(reminder_reference)
                    if reference_row is None or reference_row.get("status") != "ok":
                        raise ReplayError(
                            f"stage2 missing prior {reminder_reference!r} row")
                    row_controls["raw_goal_prompt_cost"] = (
                        _raw_prompt_cost_control(
                            reference_row, row, reminder_reference,
                            "goal reminder added to the same history condition"))
                receipt_reference = STAGE3_RECEIPT_BASE.get(intervention.name)
                if is_stage3 and receipt_reference is not None:
                    reference_row = case_rows.get(receipt_reference)
                    if reference_row is None or reference_row.get("status") != "ok":
                        raise ReplayError(
                            f"stage3 missing prior {receipt_reference!r} row")
                    row_controls["raw_action_receipt_prompt_cost"] = (
                        _raw_prompt_cost_control(
                            reference_row, row, receipt_reference,
                            "action receipt added to the same goal-only base"))
                if ((is_stage2 or is_stage3)
                        and intervention.name == "latest_only_goal_reminder"):
                    goal_tail_identity_verified += 1
                if (is_stage3
                        and intervention.name
                        == "latest_only_goal_action_receipt"):
                    receipt_tail_identity_verified += 1
                if ((is_stage2 or is_stage3)
                        and intervention.name == "c2kv4_current_user_turn_raw"):
                    full_identity_verified += 1
                    row_controls["output_stability_against_first_full"] = (
                        copy.deepcopy(row["paired_full_agreement"]))
            row["run_id"] = run_id
            _append_jsonl(args.out, row)
            case_rows[intervention.name] = row
            completed += 1
            if row["status"] != "ok":
                failed = row
                break
        if failed is not None:
            break

    elapsed = time.perf_counter() - run_started
    final = {
        "schema_version": SCHEMA_VERSION,
        "row_type": "run_final",
        "run_id": run_id,
        "status": "failed" if failed else "complete",
        "completed_trial_count": completed,
        "planned_trial_count": planned_generations,
        "http_request_count": len(http.events),
        "policy_generation_count": http.policy_generations,
        "wall_sec": elapsed,
        "failure_trial_id": failed.get("trial_id") if failed else None,
        "failure_kind": failed.get("error_kind") if failed else None,
        "scope": "diagnostic next-action replay; not official full-task success",
        "stage2_controls": ({
            "full_repeat_prepared_identity_verified_cases": (
                full_identity_verified),
            "goal_reminder_raw_tail_identity_verified_cases": (
                goal_tail_identity_verified),
        } if is_stage2 else None),
        "stage3_controls": ({
            "full_repeat_prepared_identity_verified_cases": (
                full_identity_verified),
            "goal_only_raw_tail_identity_verified_cases": (
                goal_tail_identity_verified),
            "goal_action_receipt_raw_tail_identity_verified_cases": (
                receipt_tail_identity_verified),
        } if is_stage3 else None),
    }
    _append_jsonl(args.out, final)
    return 1 if failed else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path,
                        help="frozen protocol JSON containing pilot arms and budgets")
    parser.add_argument("--cases", required=True, type=Path,
                        help="exact or explicitly constructed request cases JSONL")
    parser.add_argument("--out", required=True, type=Path,
                        help="new append-only replay JSONL; existing files are refused")
    parser.add_argument("--upstream", required=True,
                        help="SGLang base URL, e.g. http://127.0.0.1:34000")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--checkpoint-profile", required=True, type=Path,
                        help="explicit checkpoint profile; implicit/unprofiled runs are refused")
    parser.add_argument("--timeout", type=float, default=600.0,
                        help="per-POST timeout, also capped by pilot.max_wall_seconds")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.timeout <= 0 or not math.isfinite(args.timeout):
            raise ReplayError("--timeout must be a positive finite number")
        return execute(args)
    except (ReplayError, ProfileError) as error:
        parser.error(str(error))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
