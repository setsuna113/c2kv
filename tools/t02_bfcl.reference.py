"""Exact in-process BFCL continuation adapter for bounded T02 collection.

The adapter keeps BFCL's executable environment and the actor/backend snapshot
at the same unsubmitted decision.  Branches restore both sides before applying
one legal intervention, then continue with the frozen C0 policy and use BFCL's
official multi-turn checkers for turn-prefix and complete-task outcomes.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import pickle
import re
import sys
import uuid
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "runtime" / "python"))
sys.path.insert(0, str(HERE / "runtime"))

import t02
from benchmarks.memory_runtime.always_compress import CapacityInfeasible


ENV_SNAPSHOT_SCHEMA = "t02-bfcl-environment-snapshot-v1"
OUTCOME_ARTIFACT_SCHEMA = "t02-bfcl-official-outcome-v1"
RUN_SUMMARY_SCHEMA = "t02-bfcl-collection-v1"
CAPACITY_TERMINATION_SCHEMA = "t02-bfcl-capacity-termination-v1"
BFCL_CATEGORIES = (
    "multi_turn_base",
    "multi_turn_long_context",
    "multi_turn_miss_func",
    "multi_turn_miss_param",
)


def _normalize_json_object_keys(value: Any, *, path: str = "$") -> Any:
    """Project JSON object keys to strings before canonical sorting.

    Python's JSON encoder accepts string, integer, float, boolean, and null
    object keys, but ``sort_keys=True`` cannot compare a mixture of their
    Python representations.  Normalize each key through the encoder's own
    JSON semantics first.  Reject collisions instead of silently choosing one
    value when distinct Python keys encode to the same JSON string.
    """

    if isinstance(value, dict):
        normalized: dict[str, Any] = {}
        source_keys: dict[str, Any] = {}
        for key, item in value.items():
            encoded_key = json.dumps(
                {key: None}, ensure_ascii=False, allow_nan=False
            )
            normalized_key = next(iter(json.loads(encoded_key)))
            if normalized_key in normalized:
                raise ValueError(
                    f"JSON object key collision at {path}: "
                    f"{source_keys[normalized_key]!r} and {key!r} both encode "
                    f"as {normalized_key!r}"
                )
            source_keys[normalized_key] = key
            normalized[normalized_key] = _normalize_json_object_keys(
                item, path=f"{path}.{normalized_key}"
            )
        return normalized
    if isinstance(value, (list, tuple)):
        return [
            _normalize_json_object_keys(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    return value


def _json_copy(value: Any) -> Any:
    normalized = _normalize_json_object_keys(value)
    return json.loads(json.dumps(
        normalized, ensure_ascii=False, sort_keys=True, allow_nan=False
    ))


def _json_digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode("utf-8")).hexdigest()


def _pickle_digest(value: Any) -> str:
    return hashlib.sha256(pickle.dumps(value, protocol=5)).hexdigest()


def _write_json(path: Path, value: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    path.write_bytes(encoded)
    return hashlib.sha256(encoded).hexdigest()


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-") or "item"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_family_bindings(task_manifest_path: str | Path,
                         audit_path: str | Path) -> tuple[list[str], dict[str, str], dict[str, Any]]:
    """Bind each authorized BFCL variant to an audited source family."""

    manifest_path = Path(task_manifest_path).resolve()
    audit_path = Path(audit_path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "a-history-system-task-manifest-v1":
        raise ValueError("T02 task manifest schema is invalid")
    if audit.get("schema") != "a-history-system-data-expansion-audit-v1":
        raise ValueError("T02 source-family audit schema is invalid")
    audit_sha256 = _file_sha256(audit_path)
    if manifest.get("provenance", {}).get("audit_sha256") != audit_sha256:
        raise ValueError("T02 task manifest is not bound to the supplied family audit")
    task_ids = _manifest_ids(manifest_path)
    clean = audit.get("clean_source_families")
    if not isinstance(clean, list):
        raise ValueError("T02 family audit lacks clean_source_families")
    groups = {
        row.get("source_group"): row
        for row in clean if isinstance(row, Mapping)
    }
    if any(type(group) is not int or group < 0 for group in groups):
        raise ValueError("T02 family audit contains an invalid source_group")
    expected_categories = set(BFCL_CATEGORIES)
    variants: dict[int, set[str]] = {group: set() for group in groups}
    bindings: dict[str, str] = {}
    pattern = re.compile(
        r"^(multi_turn_(?:base|long_context|miss_func|miss_param))_(\d+)$"
    )
    for task_id in task_ids:
        match = pattern.fullmatch(task_id)
        if match is None:
            raise ValueError(f"T02 manifest contains an unsupported task: {task_id}")
        category, raw_group = match.groups()
        group = int(raw_group)
        if group not in groups:
            raise ValueError(f"T02 manifest task lacks an audited clean family: {task_id}")
        if category in variants[group]:
            raise ValueError(f"T02 manifest repeats a category in family {group}")
        variants[group].add(category)
        bindings[task_id] = f"bfcl_pair_{group}"
    if set(bindings) != set(task_ids) or any(
        categories != expected_categories for categories in variants.values()
    ):
        raise ValueError("T02 manifest must contain all four variants of every audited family")
    if manifest.get("source_family_count") != len(groups):
        raise ValueError("T02 manifest source_family_count differs from its audit")
    receipt = {
        "schema": "t02-bfcl-family-bindings-v1",
        "manifest_sha256": _file_sha256(manifest_path),
        "audit_sha256": audit_sha256,
        "task_count": len(task_ids),
        "source_family_count": len(groups),
        "categories": list(BFCL_CATEGORIES),
    }
    return task_ids, bindings, receipt


def verify_official_source_files(bfcl_root: str | Path,
                                 task_manifest_path: str | Path) -> dict[str, Any]:
    """Verify the eight official BFCL inputs bound by the expanded manifest."""

    root = Path(bfcl_root).resolve()
    manifest = json.loads(Path(task_manifest_path).read_text(encoding="utf-8"))
    provenance = manifest.get("provenance", {})
    expected_questions = provenance.get("question_source_sha256")
    expected_answers = provenance.get("possible_answer_source_sha256")
    if set(expected_questions or {}) != set(BFCL_CATEGORIES) or set(expected_answers or {}) != set(BFCL_CATEGORIES):
        raise ValueError("T02 manifest does not bind all eight official BFCL source files")
    actual_questions, actual_answers = {}, {}
    for category in BFCL_CATEGORIES:
        filename = f"BFCL_v4_{category}.json"
        question_path = root / "bfcl_eval" / "data" / filename
        answer_path = root / "bfcl_eval" / "data" / "possible_answer" / filename
        actual_questions[category] = _file_sha256(question_path)
        actual_answers[category] = _file_sha256(answer_path)
    if actual_questions != expected_questions or actual_answers != expected_answers:
        raise ValueError("pinned BFCL question/answer files differ from the expanded manifest")
    return {
        "schema": "t02-bfcl-official-source-files-v1",
        "question_source_sha256": actual_questions,
        "possible_answer_source_sha256": actual_answers,
    }


def round_robin_family_tasks(task_ids: Sequence[str],
                             family_bindings: Mapping[str, str]) -> list[str]:
    """Visit every family once per round before taking another family variant."""

    grouped: dict[str, list[str]] = {}
    family_order: list[str] = []
    for task_id in task_ids:
        group = family_bindings.get(task_id)
        if not isinstance(group, str) or not group:
            raise ValueError(f"task lacks an audited source-family binding: {task_id}")
        if group not in grouped:
            grouped[group] = []
            family_order.append(group)
        grouped[group].append(task_id)
    rotated: dict[str, list[str]] = {}
    for index, group in enumerate(family_order):
        rows = grouped[group]
        offset = index % len(rows)
        rotated[group] = rows[offset:] + rows[:offset]
    return [
        rotated[group][round_index]
        for round_index in range(max(map(len, rotated.values()), default=0))
        for group in family_order
        if round_index < len(rotated[group])
    ]


def validate_training_feature_contract(states: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Fail before branching if C1/C4 inputs are absent or shape-incompatible."""

    expected_contract = None
    hidden_dimension = None
    for state in states:
        state_id = state.get("state_id", "<unknown>")
        q = state.get("q")
        if not isinstance(q, Mapping):
            raise t02.T02Error(f"state {state_id} q must be an object for C1/C4 training")
        for name in ("draft_logprobs", "prefill_hidden"):
            values = q.get(name)
            if not isinstance(values, list) or not values:
                raise t02.T02Error(f"state {state_id} q.{name} must be a nonempty finite vector")
            if any(type(value) not in (int, float) or not math.isfinite(value) for value in values):
                raise t02.T02Error(f"state {state_id} q.{name} must be a nonempty finite vector")
        contract = q.get("prefill_contract")
        if not isinstance(contract, Mapping) or not contract:
            raise t02.T02Error(f"state {state_id} q.prefill_contract must be a nonempty object")
        canonical_contract = _json_copy(contract)
        dimension = len(q["prefill_hidden"])
        if dimension < 8:
            raise t02.T02Error(f"state {state_id} q.prefill_hidden must have at least 8 values")
        if canonical_contract.get("layer") is None or canonical_contract.get("readout") is None:
            raise t02.T02Error(f"state {state_id} q.prefill_contract lacks layer/readout")
        position = canonical_contract.get("position")
        position_kind = (
            position.get("kind") if isinstance(position, Mapping)
            else canonical_contract.get("position_kind")
        )
        if not isinstance(position_kind, str) or not position_kind:
            raise t02.T02Error(f"state {state_id} q.prefill_contract lacks position kind")
        bindings = canonical_contract.get("bindings")
        direct = canonical_contract.get("model") is not None and canonical_contract.get("tokenizer") is not None
        if not direct and (not isinstance(bindings, Mapping) or not bindings):
            raise t02.T02Error(f"state {state_id} q.prefill_contract lacks model bindings")
        if expected_contract is None:
            expected_contract, hidden_dimension = canonical_contract, dimension
        elif canonical_contract != expected_contract or dimension != hidden_dimension:
            raise t02.T02Error(
                f"state {state_id} prefill feature contract differs from earlier states"
            )
    if not states:
        raise t02.T02Error("training feature preflight requires at least one state")
    return {
        "schema": "t02-training-feature-contract-v1",
        "state_count": len(states),
        "prefill_contract": expected_contract,
        "prefill_hidden_dimension": hidden_dimension,
        "draft_logprobs_required": True,
    }


class OfficialBFCLBindings:
    """Lazy binding to the pinned BFCL package used by the official harness."""

    def __init__(self) -> None:
        from bfcl_eval.constants.default_prompts import (
            DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_FC,
            MAXIMUM_STEP_LIMIT,
        )
        from bfcl_eval.constants.enums import ModelStyle
        from bfcl_eval.eval_checker import eval_runner
        from bfcl_eval.eval_checker.multi_turn_eval import multi_turn_utils
        from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_checker import (
            multi_turn_checker,
        )
        from bfcl_eval.model_handler.api_inference.openai_completion import OpenAICompletionsHandler
        from bfcl_eval.model_handler.base_handler import BaseHandler
        from bfcl_eval.utils import (
            load_dataset_entry,
            load_ground_truth_entry,
            make_json_serializable,
        )

        self.default_holdout_prompt = DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_FC
        self.maximum_step_limit = MAXIMUM_STEP_LIMIT
        self.eval_runner = eval_runner
        self.multi_turn_checker = multi_turn_checker
        self.multi_turn_utils = multi_turn_utils
        self.handler_class = OpenAICompletionsHandler
        self.base_handler_class = BaseHandler
        self.model_style = ModelStyle.OPENAI_COMPLETIONS
        self.load_dataset_entry = load_dataset_entry
        self.load_ground_truth_entry = load_ground_truth_entry
        self.make_json_serializable = make_json_serializable
        self._task_cache: dict[str, tuple[dict[str, Any], list[list[str]]]] = {}

    @staticmethod
    def category(task_id: str) -> str:
        category = task_id.rsplit("_", 1)[0]
        if category not in {
            "multi_turn_base", "multi_turn_long_context",
            "multi_turn_miss_func", "multi_turn_miss_param",
        }:
            raise ValueError(f"T02 does not support BFCL category for task {task_id}")
        return category

    def load_task(self, task_id: str) -> tuple[dict[str, Any], list[list[str]]]:
        cached = self._task_cache.get(task_id)
        if cached is not None:
            return copy.deepcopy(cached)
        category = self.category(task_id)
        prompts = {row["id"]: row for row in self.load_dataset_entry(category)}
        answers = {row["id"]: row["ground_truth"] for row in self.load_ground_truth_entry(category)}
        if task_id not in prompts or task_id not in answers:
            raise KeyError(f"BFCL task is absent from official prompt/answer files: {task_id}")
        value = (prompts[task_id], answers[task_id])
        self._task_cache[task_id] = copy.deepcopy(value)
        return copy.deepcopy(value)

    def make_handler(self, namespace: str):
        # Avoid creating an OpenAI client: T02Actor owns all generation.  The
        # inherited official formatting and decoding methods need only the
        # BaseHandler fields initialized here.
        handler = object.__new__(self.handler_class)
        self.base_handler_class.__init__(
            handler, model_name=namespace, temperature=0,
            registry_name=namespace, is_fc_model=True,
        )
        handler.model_style = self.model_style
        handler.model_name_underline_replaced = namespace
        return handler

    @staticmethod
    def instance_key(namespace: str, task_id: str, class_name: str, *, evaluation: bool = False) -> str:
        model = namespace + ("_eval" if evaluation else "")
        return re.sub(r"[-./:]", "_", f"{model}_{task_id}_{class_name}_instance")

    def initialize_environment(self, task: Mapping[str, Any], namespace: str) -> None:
        self.multi_turn_utils.execute_multi_turn_func_call(
            [], task.get("initial_config", {}), task["involved_classes"], namespace,
            task["id"], long_context="long_context" in self.category(task["id"]),
            is_evaL_run=False,
        )

    def capture_instances(self, task: Mapping[str, Any], namespace: str) -> dict[str, Any]:
        values = {}
        for class_name in task["involved_classes"]:
            key = self.instance_key(namespace, task["id"], class_name)
            if key not in vars(self.multi_turn_utils):
                raise RuntimeError(f"BFCL live environment instance is missing: {key}")
            values[key] = copy.deepcopy(vars(self.multi_turn_utils)[key])
        return values

    def restore_instances(self, values: Mapping[str, Any]) -> None:
        for key, value in values.items():
            vars(self.multi_turn_utils)[key] = copy.deepcopy(value)

    def clear_instances(self, task: Mapping[str, Any], namespace: str) -> None:
        for class_name in task["involved_classes"]:
            vars(self.multi_turn_utils).pop(self.instance_key(namespace, task["id"], class_name), None)

    def execute(self, calls: list[str], task: Mapping[str, Any], namespace: str) -> list[str]:
        results, _ = self.multi_turn_utils.execute_multi_turn_func_call(
            calls, task.get("initial_config", {}), task["involved_classes"], namespace,
            task["id"], long_context="long_context" in self.category(task["id"]),
            is_evaL_run=False,
        )
        return results

    def _decoded(self, handler, raw_result: Sequence[Sequence[Any]]) -> list[list[list[str]]]:
        decoded = []
        for turn in raw_result:
            turn_decoded = []
            for response in turn:
                try:
                    value = handler.decode_execute(response, has_tool_call_tag=False)
                    if not self.multi_turn_utils.is_empty_execute_response(value):
                        turn_decoded.append(value)
                except Exception:
                    continue
            decoded.append(turn_decoded)
        return decoded

    def _clear_score_namespace(self, task: Mapping[str, Any], namespace: str) -> None:
        for suffix in ("", "_ground_truth"):
            for class_name in task["involved_classes"]:
                key = self.instance_key(namespace + suffix, task["id"], class_name, evaluation=True)
                vars(self.multi_turn_utils).pop(key, None)

    def score_turn_prefix(self, handler, raw_result, ground_truth, task, turn_index: int) -> dict[str, Any]:
        namespace = f"t02_turn_{uuid.uuid4().hex}"
        decoded = self._decoded(handler, raw_result[: turn_index + 1])
        try:
            return self.multi_turn_checker(
                decoded, ground_truth[: turn_index + 1], copy.deepcopy(task),
                self.category(task["id"]), namespace,
            )
        finally:
            self._clear_score_namespace(task, namespace)

    def score_task(self, handler, raw_result, ground_truth, task) -> dict[str, Any]:
        namespace = f"t02_task_{uuid.uuid4().hex}"
        try:
            return self.eval_runner._evaluate_single_multi_turn_entry(
                handler, task["id"], copy.deepcopy(raw_result), copy.deepcopy(ground_truth),
                copy.deepcopy(task), namespace, self.category(task["id"]),
            )
        finally:
            self._clear_score_namespace(task, namespace)


class BFCLTaskEnvironment:
    """One official BFCL task loop whose generation is supplied by T02Actor."""

    def __init__(self, task: Mapping[str, Any], ground_truth: Sequence[Sequence[str]], *,
                 bindings: OfficialBFCLBindings | None = None, namespace: str | None = None) -> None:
        self.bindings = bindings or OfficialBFCLBindings()
        self.task = copy.deepcopy(dict(task))
        self.ground_truth = copy.deepcopy(list(ground_truth))
        self.namespace = namespace or f"t02_live_{uuid.uuid4().hex}"
        self.handler = self.bindings.make_handler(self.namespace)
        self._snapshots: dict[str, dict[str, Any]] = {}
        self._closed = False
        self._begin()

    @property
    def task_id(self) -> str:
        return self.task["id"]

    @property
    def finished(self) -> bool:
        return self._finished

    @property
    def turn_index(self) -> int:
        return self._turn_index

    def _begin(self) -> None:
        if len(self.task.get("question", [])) != len(self.ground_truth):
            raise ValueError("BFCL prompt and ground-truth turn counts differ")
        self.bindings.clear_instances(self.task, self.namespace)
        self.bindings.initialize_environment(self.task, self.namespace)
        data: dict[str, Any] = {}
        data = self.handler._pre_query_processing_FC(data, self.task)
        self.inference_data = self.handler._compile_tools(data, self.task)
        self.all_model_response: list[list[Any]] = []
        self._turn_index = 0
        self._global_step = 0
        self._turn_step_count = 0
        self._current_turn_response: list[Any] = []
        self._awaiting_response = False
        self._finished = not bool(self.task["question"])
        self._force_quit = False
        self._capacity_termination: dict[str, Any] | None = None
        self._previous_turn_valid_cache: dict[int, bool | None] = {}
        if not self._finished:
            self._open_turn(first=True)

    def _open_turn(self, *, first: bool) -> None:
        message = copy.deepcopy(self.task["question"][self._turn_index])
        holdout = self.task.get("missed_function", {})
        if str(self._turn_index) in holdout:
            self.task["function"].extend(copy.deepcopy(holdout[str(self._turn_index)]))
            self.inference_data = self.handler._compile_tools(self.inference_data, self.task)
            if message:
                raise ValueError("BFCL holdout-function turn unexpectedly has a user message")
            message = [{"role": "user", "content": self.bindings.default_holdout_prompt}]
        if first:
            self.inference_data = self.handler.add_first_turn_message_FC(self.inference_data, message)
        else:
            self.inference_data = self.handler._add_next_turn_user_message_FC(self.inference_data, message)
        self._current_turn_response = []
        self._turn_step_count = 0

    def next_payload(self) -> dict[str, Any]:
        if self._closed or self._finished:
            raise RuntimeError("BFCL task has no next generation payload")
        if self._awaiting_response:
            raise RuntimeError("Commit the outstanding BFCL response first")
        self._awaiting_response = True
        return {
            "session_id": f"bfcl/{self.task_id}/attempt-0",
            "decision_key": f"turn-{self._turn_index}/step-{self._global_step}",
            "messages": copy.deepcopy(self.inference_data["message"]),
            "tools": copy.deepcopy(self.inference_data.get("tools", [])),
        }

    def previous_turn_valid(self) -> bool | None:
        """Official validity of the completed prefix before the current turn."""
        if self._turn_index == 0:
            return True
        if self._turn_index not in self._previous_turn_valid_cache:
            try:
                result = self.bindings.score_turn_prefix(
                    self.handler, self.all_model_response, self.ground_truth,
                    self.task, self._turn_index - 1,
                )
                value = result.get("valid")
                self._previous_turn_valid_cache[self._turn_index] = (
                    value if type(value) is bool else None
                )
            except Exception:
                self._previous_turn_valid_cache[self._turn_index] = None
        return self._previous_turn_valid_cache[self._turn_index]

    @staticmethod
    def _response_data(response: Mapping[str, Any]) -> dict[str, Any]:
        calls = copy.deepcopy(response.get("tool_calls") or [])
        if any(not isinstance(call, Mapping) for call in calls):
            raise ValueError("actor response tool_calls must be objects")
        raw = []
        call_ids = []
        normalized_calls = []
        for call in calls:
            function = call.get("function")
            if not isinstance(function, Mapping):
                raise ValueError("actor response tool call lacks function")
            name, arguments = function.get("name"), function.get("arguments")
            if not isinstance(name, str) or not name or not isinstance(arguments, str):
                raise ValueError("actor response tool function is invalid")
            call_id = call.get("id")
            if not isinstance(call_id, str) or not call_id:
                raise ValueError("actor response tool call lacks id")
            raw.append({name: arguments})
            call_ids.append(call_id)
            normalized_calls.append(copy.deepcopy(dict(call)))
        content = response.get("content")
        if content is not None and not isinstance(content, str):
            raise ValueError("actor response content must be text or null")
        message = {"role": "assistant", "content": content}
        if normalized_calls:
            message["tool_calls"] = normalized_calls
        return {
            "model_responses": raw if normalized_calls else (content or ""),
            "model_responses_message_for_chat_history": message,
            "tool_call_ids": call_ids,
            "input_token": 0,
            "output_token": 0,
        }

    def commit_response(self, response: Mapping[str, Any]) -> None:
        if not self._awaiting_response or self._finished:
            raise RuntimeError("No BFCL response is awaiting commit")
        data = self._response_data(response)
        self._awaiting_response = False
        self._global_step += 1
        self.inference_data = self.handler._add_assistant_message_FC(self.inference_data, data)
        self._current_turn_response.append(copy.deepcopy(data["model_responses"]))
        try:
            decoded = self.handler.decode_execute(data["model_responses"], has_tool_call_tag=False)
            empty = self.bindings.multi_turn_utils.is_empty_execute_response(decoded)
        except Exception:
            decoded, empty = [], True
        if empty:
            self._finish_turn()
            return
        results = self.bindings.execute(decoded, self.task, self.namespace)
        self.inference_data = self.handler._add_execution_results_FC(self.inference_data, results, data)
        self._turn_step_count += 1
        if self._turn_step_count > self.bindings.maximum_step_limit:
            self._force_quit = True
            self._finish_turn(force=True)

    def _finish_turn(self, *, force: bool = False) -> None:
        self.all_model_response.append(copy.deepcopy(self._current_turn_response))
        self._turn_index += 1
        if force or self._turn_index >= len(self.task["question"]):
            self._finished = True
            return
        self._open_turn(first=False)

    def terminate_capacity_infeasible(
        self, error: CapacityInfeasible, *, stage: str
    ) -> dict[str, Any]:
        """Finish an unscorable continuation without fabricating a response."""

        if not isinstance(error, CapacityInfeasible):
            raise TypeError("capacity termination requires CapacityInfeasible")
        if self._closed or self._finished:
            raise RuntimeError("capacity termination requires an unfinished BFCL task")
        receipt = {
            "schema": CAPACITY_TERMINATION_SCHEMA,
            "reason": "capacity_infeasible",
            "terminal_status": "budget_terminated",
            "stage": stage,
            "error_type": type(error).__name__,
            "error": str(error),
            "turn_index": self._turn_index,
            "global_step": self._global_step,
            "response_fabricated": False,
        }
        self._capacity_termination = receipt
        self._awaiting_response = False
        self._force_quit = True
        self._finish_turn(force=True)
        return copy.deepcopy(receipt)

    def _snapshot_state(self) -> dict[str, Any]:
        return {
            "task": copy.deepcopy(self.task),
            "inference_data": copy.deepcopy(self.inference_data),
            "all_model_response": copy.deepcopy(self.all_model_response),
            "turn_index": self._turn_index,
            "global_step": self._global_step,
            "turn_step_count": self._turn_step_count,
            "current_turn_response": copy.deepcopy(self._current_turn_response),
            "awaiting_response": self._awaiting_response,
            "finished": self._finished,
            "force_quit": self._force_quit,
            "capacity_termination": copy.deepcopy(self._capacity_termination),
            "previous_turn_valid_cache": copy.deepcopy(self._previous_turn_valid_cache),
            "instances": self.bindings.capture_instances(self.task, self.namespace),
        }

    def capture(self) -> dict[str, Any]:
        if not self._awaiting_response:
            raise RuntimeError("BFCL environment capture requires an outstanding actor draft")
        state = self._snapshot_state()
        snapshot_id = uuid.uuid4().hex
        receipt = {"schema": ENV_SNAPSHOT_SCHEMA, "snapshot_id": snapshot_id,
                   "environment_sha256": _pickle_digest(state),
                   "turn_index": self._turn_index, "global_step": self._global_step}
        self._snapshots[snapshot_id] = {"state": copy.deepcopy(state), "receipt": receipt}
        return copy.deepcopy(receipt)

    def restore(self, receipt: Mapping[str, Any]) -> dict[str, Any]:
        saved = self._snapshots.get(receipt.get("snapshot_id"))
        if saved is None or dict(receipt) != saved["receipt"]:
            raise ValueError("BFCL environment snapshot receipt changed or is unknown")
        state = copy.deepcopy(saved["state"])
        if _pickle_digest(state) != saved["receipt"]["environment_sha256"]:
            raise RuntimeError("BFCL environment snapshot changed before restore")
        self.task = state["task"]
        self.inference_data = state["inference_data"]
        self.all_model_response = state["all_model_response"]
        self._turn_index = state["turn_index"]
        self._global_step = state["global_step"]
        self._turn_step_count = state["turn_step_count"]
        self._current_turn_response = state["current_turn_response"]
        self._awaiting_response = state["awaiting_response"]
        self._finished = state["finished"]
        self._force_quit = state["force_quit"]
        self._capacity_termination = state["capacity_termination"]
        self._previous_turn_valid_cache = state["previous_turn_valid_cache"]
        self.bindings.restore_instances(state["instances"])
        live = self._snapshot_state()
        if _pickle_digest(live) != saved["receipt"]["environment_sha256"]:
            raise RuntimeError("live BFCL environment differs after restore")
        return copy.deepcopy(saved["receipt"])

    def release(self, receipt: Mapping[str, Any]) -> None:
        saved = self._snapshots.pop(receipt.get("snapshot_id"), None)
        if saved is None or dict(receipt) != saved["receipt"]:
            raise ValueError("BFCL environment snapshot receipt changed or is unknown")

    def official_outcomes(self, intervention_turn: int) -> dict[str, Any]:
        if not self._finished:
            raise RuntimeError("Official outcomes require a complete BFCL continuation")
        previous_valid = self._previous_turn_valid_cache.get(
            intervention_turn, True if intervention_turn == 0 else None)
        if previous_valid is True:
            turn = self.bindings.score_turn_prefix(
                self.handler, self.all_model_response, self.ground_truth,
                self.task, intervention_turn,
            )
            turn_success = turn.get("valid") is True
        else:
            turn_success = None
            turn = {
                "valid": None,
                "status": "unknown_previous_prefix_invalid" if previous_valid is False
                else "unknown_previous_prefix_unavailable",
                "previous_turn_valid": previous_valid,
            }
        task = self.bindings.score_task(
            self.handler, self.all_model_response, self.ground_truth, self.task,
        )
        task_success = task.get("valid") is True
        if self._capacity_termination is not None:
            task_success = None
            if self._capacity_termination["turn_index"] == intervention_turn:
                turn_success = None
        return {
            "turn_success": turn_success,
            "task_success": task_success,
            "turn_checker_result": _json_copy(
                self.bindings.make_json_serializable(turn)
            ),
            "task_checker_result": _json_copy(
                self.bindings.make_json_serializable(task)
            ),
            "intervention_turn": intervention_turn,
            "completed_turns": len(self.all_model_response),
            "expected_turns": len(self.ground_truth),
            "capacity_termination": copy.deepcopy(self._capacity_termination),
        }

    def close(self) -> None:
        if self._closed:
            return
        self.bindings.clear_instances(self.task, self.namespace)
        self._snapshots.clear()
        self._closed = True


class ExactBFCLBranchAdapter:
    """Bind live BFCL environments and T02Actors to :mod:`t02` BranchAdapter."""

    def __init__(self, *, frozen_policy: Mapping[str, Any], artifact_dir: str | Path,
                 family_bindings: Mapping[str, str] | None = None) -> None:
        self.frozen_policy = _json_copy(frozen_policy)
        self.policy_sha256 = t02._digest(self.frozen_policy)
        self.artifact_dir = Path(artifact_dir).resolve()
        self._states: dict[str, dict[str, Any]] = {}
        self._active: dict[str, Any] | None = None
        self._task_refcounts: dict[int, int] = {}
        self._branch_counts: dict[str, int] = {}
        self.family_bindings = dict(family_bindings or {})
        self.source_collection_tasks = 0
        self.source_collection_decisions = 0
        self.source_collection_generation_calls = 0
        self.branch_generation_calls = 0
        self._source_task_terminals: dict[str, dict[str, Any]] = {}

    def capabilities(self) -> dict[str, Any]:
        return {
            "schema": t02.CAPABILITY_SCHEMA,
            "exact_same_state_restore": True,
            "components": list(t02.REQUIRED_COMPONENTS),
            "frozen_subsequent_policy": True,
            "official_turn_outcome": True,
            "official_task_outcome": True,
            "environment": "official_bfcl_in_process_live_instances",
            "actor": "t02_runtime.T02Actor_exact_backend_snapshot",
        }

    def _invoke_counted(self, actor: Any, counter_name: str,
                        method_name: str, *args: Any) -> Any:
        before = getattr(actor, "total_generation_calls", None)
        if type(before) is not int:
            raise TypeError("T02 actor must expose cumulative total_generation_calls")
        try:
            return getattr(actor, method_name)(*args)
        finally:
            after = getattr(actor, "total_generation_calls", None)
            if type(after) is not int or after < before:
                raise RuntimeError("T02 actor cumulative generation count regressed")
            setattr(self, counter_name, getattr(self, counter_name) + after - before)

    def _enrich_state(self, state: Mapping[str, Any], environment: BFCLTaskEnvironment) -> dict[str, Any]:
        value = copy.deepcopy(dict(state))
        value.setdefault("schema", t02.CANDIDATE_SCHEMA)
        canonical_group = t02.canonical_task_group_id(environment.task_id)
        if self.family_bindings:
            bound_group = self.family_bindings.get(environment.task_id)
            if bound_group is None:
                raise ValueError(f"BFCL task lacks an audited family binding: {environment.task_id}")
            if bound_group != canonical_group:
                raise ValueError(f"BFCL audited family binding disagrees for {environment.task_id}")
        value.update(
            benchmark="bfcl", task_id=environment.task_id,
            task_group_id=canonical_group,
            previous_turn_valid=environment.previous_turn_valid(),
            bfcl_turn_index=environment.turn_index,
            state_scope="exact_live_environment_actor_checkpoint",
        )
        return value

    @staticmethod
    def _branchable(state: Mapping[str, Any], *, seed: int) -> dict[str, Any] | None:
        try:
            normalized = t02._normalize_candidate_state(state)
            t02._branch_actions(normalized, seed=seed)
            return normalized
        except t02.T02Error:
            return None

    def discover_task(self, environment: BFCLTaskEnvironment, actor: Any, *,
                      seed: int, max_states: int = 2) -> list[dict[str, Any]]:
        if max_states <= 0:
            raise ValueError("max_states must be positive")
        self.source_collection_tasks += 1
        found = []
        captured_kinds: set[str] = set()
        try:
            while not environment.finished:
                state = self._invoke_counted(
                    actor, "source_collection_generation_calls", "hold",
                    environment.next_payload(),
                )
                self.source_collection_decisions += 1
                if state.get("collectable", True) is not False:
                    enriched = self._enrich_state(state, environment)
                    normalized = self._branchable(enriched, seed=seed)
                    kind = normalized.get("draft_kind") if normalized is not None else None
                    if (normalized is not None and len(found) < max_states
                            and kind not in captured_kinds):
                        nonempty = [row for row in state.get("allowed_actions", [])
                                    if row.get("candidate_ids")]
                        if len(nonempty) >= 2 and hasattr(actor, "propose_alternative"):
                            state = actor.propose_alternative()
                            normalized = self._branchable(
                                self._enrich_state(state, environment), seed=seed)
                            if normalized is None:
                                raise RuntimeError("local A2 proposal invalidated a branchable state")
                        env_snapshot = environment.capture()
                        actor_snapshot = actor.capture()
                        self.register_state(normalized, environment=environment, actor=actor,
                                            environment_snapshot=env_snapshot,
                                            actor_snapshot=actor_snapshot)
                        found.append(normalized)
                        captured_kinds.add(kind)
                        state = normalized
                selected = state.get("selected_ids", []) if isinstance(state, Mapping) else []
                response = self._invoke_counted(
                    actor, "source_collection_generation_calls", "submit_held", selected
                )
                environment.commit_response(response)
                if len(found) >= max_states:
                    break
        except CapacityInfeasible as error:
            self._source_task_terminals[environment.task_id] = (
                environment.terminate_capacity_infeasible(
                    error, stage="source_collection"
                )
            )
        if not found:
            actor.close()
            environment.close()
        return found

    def source_task_terminal(self, task_id: str) -> dict[str, Any] | None:
        return copy.deepcopy(self._source_task_terminals.get(task_id))

    def register_state(self, state: Mapping[str, Any], *, environment: BFCLTaskEnvironment,
                       actor: Any, environment_snapshot: Mapping[str, Any],
                       actor_snapshot: Mapping[str, Any]) -> None:
        state_id = state["state_id"]
        if state_id in self._states:
            raise ValueError(f"duplicate live T02 state: {state_id}")
        actor_digests = actor_snapshot.get("component_digests")
        if not isinstance(actor_digests, Mapping) or set(actor_digests) != {
            "actor_kv", "actor_positions", "backend_stats", "rng"
        }:
            raise ValueError("actor snapshot lacks exact KV/positions/backend/RNG digests")
        components = {"environment": environment_snapshot["environment_sha256"],
                      **dict(actor_digests)}
        snapshot_id = uuid.uuid4().hex
        public = {
            "schema": t02.SNAPSHOT_SCHEMA,
            "state_id": state_id,
            "snapshot_id": snapshot_id,
            "frozen_policy_sha256": self.policy_sha256,
            "component_digests": components,
        }
        key = id(actor)
        self._task_refcounts[key] = self._task_refcounts.get(key, 0) + 1
        self._states[state_id] = {
            "state": _json_copy(state), "public": public,
            "environment": environment, "actor": actor,
            "environment_snapshot": _json_copy(environment_snapshot),
            "actor_snapshot": _json_copy(actor_snapshot),
            "actor_key": key,
        }

    def prune(self, selected_state_ids: Sequence[str]) -> None:
        selected = set(selected_state_ids)
        missing = selected - set(self._states)
        if missing:
            raise ValueError(f"selected states lack live snapshots: {sorted(missing)}")
        for state_id in list(self._states):
            if state_id not in selected:
                self._release_state(state_id)

    def capture_state(self, state: Mapping[str, Any], *, frozen_policy: Mapping[str, Any]) -> dict[str, Any]:
        if _json_copy(frozen_policy) != self.frozen_policy:
            raise ValueError("frozen continuation policy changed after live capture")
        registered = self._states.get(state["state_id"])
        if registered is None:
            raise ValueError(f"planned state has no live exact snapshot: {state['state_id']}")
        planned = dict(state)
        for key in ("branches", "split", "branch_selection_uses_future_outcome"):
            planned.pop(key, None)
        planned["schema"] = t02.CANDIDATE_SCHEMA
        if t02._normalize_candidate_state(planned) != registered["state"]:
            raise ValueError("planned state differs from its captured live candidate")
        self._active = registered
        return copy.deepcopy(registered["public"])

    def restore_state(self, snapshot: Mapping[str, Any], *, frozen_policy: Mapping[str, Any]) -> dict[str, Any]:
        if self._active is None or dict(snapshot) != self._active["public"]:
            raise ValueError("restore does not target the active live BFCL state")
        if _json_copy(frozen_policy) != self.frozen_policy:
            raise ValueError("frozen continuation policy changed before restore")
        env = self._active["environment"].restore(self._active["environment_snapshot"])
        actor = self._active["actor"].restore(self._active["actor_snapshot"])
        components = {"environment": env["environment_sha256"],
                      **actor["component_digests"]}
        if components != snapshot["component_digests"]:
            raise RuntimeError("combined BFCL/actor restore digests differ from capture")
        return {
            "schema": t02.RESTORE_SCHEMA,
            "state_id": snapshot["state_id"],
            "snapshot_id": snapshot["snapshot_id"],
            "frozen_policy_sha256": snapshot["frozen_policy_sha256"],
            "component_digests": components,
            "restored": True,
        }

    def run_branch(self, state: Mapping[str, Any], branch: Mapping[str, Any], *,
                   frozen_policy: Mapping[str, Any],
                   restore_receipt: Mapping[str, Any]) -> dict[str, Any]:
        if self._active is None or self._active["state"]["state_id"] != state["state_id"]:
            raise ValueError("branch does not target the active live BFCL state")
        environment, actor = self._active["environment"], self._active["actor"]
        intervention_turn = environment.turn_index
        capacity_termination = None
        response = self._invoke_counted(
            actor, "branch_generation_calls", "submit_held", branch["candidate_ids"]
        )
        environment.commit_response(response)
        try:
            while not environment.finished:
                response = self._invoke_counted(
                    actor, "branch_generation_calls", "generate", environment.next_payload()
                )
                environment.commit_response(response)
        except CapacityInfeasible as error:
            capacity_termination = environment.terminate_capacity_infeasible(
                error, stage="branch_continuation"
            )
        outcomes = environment.official_outcomes(intervention_turn)
        artifact = {
            "schema": OUTCOME_ARTIFACT_SCHEMA,
            "state_id": state["state_id"], "branch_id": branch["branch_id"],
            "candidate_ids": list(branch["candidate_ids"]),
            "snapshot_id": self._active["public"]["snapshot_id"],
            "frozen_policy_sha256": self.policy_sha256,
            "official_checker": {
                "turn_prefix": "bfcl_eval.eval_checker.multi_turn_eval.multi_turn_checker.multi_turn_checker",
                "task": "bfcl_eval.eval_checker.eval_runner._evaluate_single_multi_turn_entry",
            },
            "outcomes": outcomes,
            "model_result": _json_copy(environment.all_model_response),
            "capacity_termination": capacity_termination,
        }
        artifact_path = self.artifact_dir / _safe_name(state["state_id"]) / f"{branch['branch_id']}.json"
        artifact_sha256 = _write_json(artifact_path, artifact)
        result = {
            "schema": t02.RESULT_SCHEMA,
            "state_id": state["state_id"], "branch_id": branch["branch_id"],
            "candidate_ids": list(branch["candidate_ids"]),
            "snapshot_id": self._active["public"]["snapshot_id"],
            "restore_receipt_sha256": t02._digest(restore_receipt),
            "frozen_policy_sha256": self.policy_sha256,
            "execution_status": "complete",
            "execution_receipt": {
                "submitted_original_draft": branch["submit_original_draft"],
                "recovery_candidate_ids": list(branch["candidate_ids"]),
                "regeneration_count": 1 if branch["regenerate"] else 0,
                "continued_with_frozen_policy": True,
                "observations_replayed_from_other_branch": False,
            },
            "official_outcome": {
                "source": "official",
                "scorer": "BFCL_v4_multi_turn_checker",
                "artifact": str(artifact_path),
                "artifact_sha256": artifact_sha256,
                "turn_success": outcomes["turn_success"],
                "task_success": outcomes["task_success"],
                "capacity_termination": capacity_termination,
            },
        }
        count = self._branch_counts.get(state["state_id"], 0) + 1
        self._branch_counts[state["state_id"]] = count
        if count == len(t02.BRANCH_IDS):
            self._release_state(state["state_id"])
            self._active = None
        return result

    def _release_state(self, state_id: str) -> None:
        registered = self._states.pop(state_id)
        registered["actor"].release(registered["actor_snapshot"])
        registered["environment"].release(registered["environment_snapshot"])
        key = registered["actor_key"]
        self._task_refcounts[key] -= 1
        if self._task_refcounts[key] == 0:
            registered["actor"].close()
            registered["environment"].close()
            del self._task_refcounts[key]

    def close(self) -> None:
        for state_id in list(self._states):
            self._release_state(state_id)
        self._active = None

    def cost_summary(self) -> dict[str, int]:
        return {
            "source_collection_tasks": self.source_collection_tasks,
            "source_collection_decisions": self.source_collection_decisions,
            "source_collection_generation_calls": self.source_collection_generation_calls,
            "branch_generation_calls": self.branch_generation_calls,
            "total_generation_calls": (
                self.source_collection_generation_calls + self.branch_generation_calls
            ),
            "completed_branch_executions": sum(self._branch_counts.values()),
        }


def _manifest_ids(path: str | Path) -> list[str]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(value, Mapping):
        value = value.get("task_ids", value.get("tasks"))
    if not isinstance(value, list):
        raise ValueError(f"task manifest must contain a list: {path}")
    result = [row if isinstance(row, str) else row.get("task_id", row.get("id")) for row in value]
    if any(not isinstance(item, str) or not item for item in result):
        raise ValueError(f"task manifest contains an invalid task id: {path}")
    return result


def collect_live_plan(*, task_ids: Sequence[str], actor_factory: Callable[[str, int], Any],
                      bindings: OfficialBFCLBindings,
                      forbidden_manifest_paths: Mapping[str, str | Path],
                      frozen_policy: Mapping[str, Any], artifact_dir: str | Path,
                      seed: int, target_states: int, train_states: int,
                      min_task_groups: int, candidate_snapshot_cap: int = 128,
                      max_states_per_group: int = 2, max_states_per_task: int = 2,
                      max_branch_executions: int = 180,
                      family_bindings: Mapping[str, str] | None = None,
                      max_source_task_starts: int | None = None,
                      on_progress: Callable[[Mapping[str, Any]], None] | None = None,
                      smoke: bool = False):
    adapter = ExactBFCLBranchAdapter(
        frozen_policy=frozen_policy, artifact_dir=artifact_dir,
        family_bindings=family_bindings,
    )
    candidates: list[dict[str, Any]] = []
    forbidden, _ = t02.load_forbidden_manifests(forbidden_manifest_paths)
    group_counts: dict[str, int] = {}
    ordered_task_ids = (
        round_robin_family_tasks(task_ids, family_bindings)
        if family_bindings else list(task_ids)
    )
    try:
        for task_index, task_id in enumerate(ordered_task_ids):
            if len(candidates) >= candidate_snapshot_cap:
                break
            group_id = (
                family_bindings[task_id] if family_bindings
                else t02.canonical_task_group_id(task_id)
            )
            if task_id in forbidden or group_id in forbidden:
                continue
            group_remaining = max_states_per_group - group_counts.get(group_id, 0)
            if group_remaining <= 0:
                continue
            if (max_source_task_starts is not None
                    and adapter.source_collection_tasks >= max_source_task_starts):
                break
            task, ground_truth = bindings.load_task(task_id)
            environment = BFCLTaskEnvironment(task, ground_truth, bindings=bindings)
            actor = actor_factory(task_id, task_index)
            if on_progress is not None:
                on_progress({
                    "phase": "source_task_started", "task_id": task_id,
                    "source_task_starts": adapter.source_collection_tasks + 1,
                    "candidate_count": len(candidates),
                })
            remaining = candidate_snapshot_cap - len(candidates)
            discovered = adapter.discover_task(
                environment, actor, seed=seed,
                max_states=min(max_states_per_task, remaining, group_remaining),
            )
            candidates.extend(discovered)
            group_counts[group_id] = group_counts.get(group_id, 0) + len(discovered)
            if on_progress is not None:
                on_progress({
                    "phase": "source_task_collected", "task_id": task_id,
                    "source_task_starts": adapter.source_collection_tasks,
                    "candidate_count": len(candidates),
                    "families_with_candidates": sum(count > 0 for count in group_counts.values()),
                    "generation_calls": adapter.cost_summary(),
                })
            if smoke and candidates:
                plan = t02.build_smoke_plan(
                    candidates[0], forbidden_manifest_paths=forbidden_manifest_paths,
                    frozen_subsequent_policy=frozen_policy, seed=seed,
                )
                plan["training_feature_contract"] = validate_training_feature_contract(
                    plan["states"]
                )
                adapter.prune([candidates[0]["state_id"]])
                return plan, candidates, adapter
            if len(candidates) >= target_states:
                try:
                    plan = t02.build_plan(
                        candidates, forbidden_manifest_paths=forbidden_manifest_paths,
                        frozen_subsequent_policy=frozen_policy, seed=seed,
                        target_states=target_states, train_states=train_states,
                        min_task_groups=min_task_groups,
                        max_states_per_task_group=max_states_per_group,
                        max_states_per_task=max_states_per_task,
                        max_complete_branch_executions=max_branch_executions,
                    )
                except t02.T02Error:
                    plan = None
                if plan is not None:
                    plan["training_feature_contract"] = validate_training_feature_contract(
                        plan["states"]
                    )
                    selected = [row["state_id"] for row in plan["states"]]
                    adapter.prune(selected)
                    return plan, candidates, adapter
        raise t02.T02Error(
            f"could not build T02 plan from {len(candidates)} live candidates "
            f"within snapshot cap {candidate_snapshot_cap}"
        )
    except BaseException:
        adapter.close()
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bfcl-root", type=Path, required=True)
    parser.add_argument("--bfcl-dependency-path", type=Path)
    parser.add_argument("--task-manifest", type=Path, required=True)
    parser.add_argument("--family-audit", type=Path, required=True)
    parser.add_argument("--d128-manifest", type=Path, required=True)
    parser.add_argument("--f128-manifest", type=Path, required=True)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--backend-url", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target-states", type=int)
    parser.add_argument("--train-states", type=int)
    parser.add_argument("--min-task-groups", type=int)
    parser.add_argument("--max-states-per-group", type=int, default=2)
    parser.add_argument("--max-states-per-task", type=int, default=2)
    parser.add_argument("--max-branch-executions", type=int, default=180)
    parser.add_argument("--consumed-complete-branch-executions", type=int, default=0)
    parser.add_argument("--max-source-task-starts", type=int, default=104)
    parser.add_argument("--consumed-source-task-starts", type=int, default=0)
    parser.add_argument("--candidate-snapshot-cap", type=int, default=128)
    parser.add_argument("--backend-snapshot-cap", type=int, default=128)
    parser.add_argument("--resource-coordination-approved", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    args.bfcl_root = args.bfcl_root.resolve()
    if args.bfcl_dependency_path is not None:
        args.bfcl_dependency_path = args.bfcl_dependency_path.resolve()
    args.task_manifest = args.task_manifest.resolve()
    args.family_audit = args.family_audit.resolve()
    args.d128_manifest = args.d128_manifest.resolve()
    args.f128_manifest = args.f128_manifest.resolve()
    args.design = args.design.resolve()
    args.checkpoint = args.checkpoint.resolve()
    args.out = args.out.resolve()
    if not args.resource_coordination_approved:
        raise ValueError(
            "resource coordination is not approved; leave launch disabled until the user authorizes it"
        )
    if not 0 < args.candidate_snapshot_cap <= args.backend_snapshot_cap:
        raise ValueError("candidate-snapshot-cap must be positive and within backend-snapshot-cap")
    for value, name in (
        (args.max_states_per_group, "max-states-per-group"),
        (args.max_states_per_task, "max-states-per-task"),
        (args.max_branch_executions, "max-branch-executions"),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    if args.max_states_per_group > 8 or args.max_states_per_task > 2:
        raise ValueError("approved expanded collection caps are 8 states per family and 2 per variant")
    if args.max_branch_executions > 360:
        raise ValueError("approved T02 complete branch budget is at most 360")
    if args.consumed_complete_branch_executions < 0:
        raise ValueError("consumed-complete-branch-executions cannot be negative")
    if (args.consumed_source_task_starts < 0 or args.max_source_task_starts <= 0
            or args.consumed_source_task_starts + args.max_source_task_starts > 104):
        raise ValueError("source starts including previous attempts must not exceed 104")
    if not args.smoke and any(value is None for value in (
            args.target_states, args.train_states, args.min_task_groups)):
        raise ValueError(
            "production requires explicit target-states, train-states, and min-task-groups"
        )
    requested_states = 1 if args.smoke else args.target_states
    requested_branches = requested_states * len(t02.BRANCH_IDS)
    remaining_branch_budget = (
        args.max_branch_executions - args.consumed_complete_branch_executions
    )
    if requested_branches > remaining_branch_budget:
        raise ValueError(
            "requested branches plus previously consumed branches exceed max-branch-executions"
        )
    if requested_states > args.candidate_snapshot_cap:
        raise ValueError("target states exceed candidate snapshot cap")
    if args.out.exists():
        raise ValueError(f"output directory already exists: {args.out}")
    if not (args.bfcl_root / "bfcl_eval").is_dir():
        raise ValueError("bfcl-root does not contain the official bfcl_eval package")
    task_ids, family_bindings, family_receipt = load_family_bindings(
        args.task_manifest, args.family_audit
    )
    official_source_receipt = verify_official_source_files(
        args.bfcl_root, args.task_manifest
    )
    args.out.mkdir(parents=True)
    os.environ["BFCL_PROJECT_ROOT"] = str((args.out / "bfcl_state").resolve())
    sys.path.insert(0, str(args.bfcl_root.resolve()))
    if args.bfcl_dependency_path is not None:
        # Append so the pinned BFCL checkout wins while missing dependencies can
        # resolve without replacing the active Python or accelerator packages.
        sys.path.append(str(args.bfcl_dependency_path))
    os.chdir(args.bfcl_root)
    bindings = OfficialBFCLBindings()
    frozen_policy = {
        "name": "evidence_sets_v1_C0_frozen_continuation",
        "version": 1,
        "design_sha256": hashlib.sha256(args.design.read_bytes()).hexdigest(),
        "sampling": {"mode": "greedy", "temperature": 0, "seed": 0},
        "resource_coordination_approved": True,
        "family_bindings": family_receipt,
        "official_source_files": official_source_receipt,
        "complete_branch_budget": {
            "authorized_total": args.max_branch_executions,
            "consumed_before_run": args.consumed_complete_branch_executions,
            "available_for_run": remaining_branch_budget,
        },
        "source_task_budget": {
            "authorized_total": 104,
            "consumed_before_run": args.consumed_source_task_starts,
            "available_for_run": args.max_source_task_starts,
        },
    }
    from t02_runtime import build_actor

    shared_models: list[Any | None] = [None]

    def actor_factory(task_id: str, task_index: int):
        actor = build_actor(
            design_path=args.design, checkpoint=args.checkpoint,
            backend_url=args.backend_url,
            output_dir=args.out / "actors" / f"{task_index:03d}-{_safe_name(task_id)}",
        )
        controller = actor.runner.controller
        if shared_models[0] is None:
            shared_models[0] = controller.backends
        else:
            while controller is not None:
                if hasattr(controller, "backends"):
                    controller.backends = shared_models[0]
                controller = getattr(controller, "base", None)
        return actor

    forbidden = {"D128": args.d128_manifest, "F128": args.f128_manifest}
    plan = results = labels = adapter = None
    try:
        plan, candidates, adapter = collect_live_plan(
            task_ids=task_ids, actor_factory=actor_factory,
            bindings=bindings, forbidden_manifest_paths=forbidden,
            frozen_policy=frozen_policy, artifact_dir=args.out / "official_outcomes",
            seed=args.seed, target_states=1 if args.smoke else args.target_states,
            train_states=0 if args.smoke else args.train_states,
            min_task_groups=1 if args.smoke else args.min_task_groups,
            candidate_snapshot_cap=args.candidate_snapshot_cap,
            max_states_per_group=(1 if args.smoke else args.max_states_per_group),
            max_states_per_task=(1 if args.smoke else args.max_states_per_task),
            max_branch_executions=remaining_branch_budget,
            family_bindings=family_bindings,
            max_source_task_starts=args.max_source_task_starts,
            on_progress=lambda value: _write_json(args.out / "progress.json", {
                "schema": "t02-collection-progress-v1", **value,
                "consumed_source_task_starts_before_run": args.consumed_source_task_starts,
            }),
            smoke=args.smoke,
        )
        _write_json(args.out / "candidates.json", {
            "schema": "t02-live-candidates-v1", "states": candidates,
        })
        _write_json(args.out / "plan.json", plan)
        prepared_summary = {
            "schema": RUN_SUMMARY_SCHEMA, "status": "prepared",
            "mode": "smoke" if args.smoke else "production",
            "candidate_count": len(candidates), "state_count": len(plan["states"]),
            "planned_complete_branch_executions": len(plan["states"]) * len(t02.BRANCH_IDS),
            "consumed_complete_branch_executions_before_run": (
                args.consumed_complete_branch_executions
            ),
            "authorized_complete_branch_cap": args.max_branch_executions,
            "generation_calls": adapter.cost_summary(),
            "actual_split_counts": dict(Counter(
                state["split"] for state in plan["states"]
            )),
            "training_feature_contract": plan["training_feature_contract"],
            "family_bindings": family_receipt,
            "official_source_files": official_source_receipt,
            "plan_sha256": t02._digest(plan),
        }
        _write_json(args.out / "summary.json", prepared_summary)
        results = t02.execute_plan(plan, adapter)
        _write_json(args.out / "results.json", results)
        labels = t02.label_results(plan, results)
        _write_json(args.out / "labels.json", labels)
        summary = {
            "schema": RUN_SUMMARY_SCHEMA, "status": "completed",
            "mode": "smoke" if args.smoke else "production",
            "candidate_count": len(candidates), "state_count": len(plan["states"]),
            "complete_branch_executions": results["complete_branch_executions"],
            "cumulative_complete_branch_executions": (
                args.consumed_complete_branch_executions
                + results["complete_branch_executions"]
            ),
            "authorized_complete_branch_cap": args.max_branch_executions,
            "planned_complete_branch_executions": len(plan["states"]) * len(t02.BRANCH_IDS),
            "generation_calls": adapter.cost_summary(),
            "training_feature_contract": plan["training_feature_contract"],
            "family_bindings": family_receipt,
            "official_source_files": official_source_receipt,
            "actual_split_counts": dict(Counter(
                state["split"] for state in plan["states"]
            )),
            "plan_sha256": t02._digest(plan),
        }
        _write_json(args.out / "summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False), flush=True)
        return 0
    except BaseException as error:
        failure = {
            "schema": RUN_SUMMARY_SCHEMA,
            "status": "failed",
            "mode": "smoke" if args.smoke else "production",
            "error_type": type(error).__name__,
            "error": str(error),
            "authorized_complete_branch_cap": args.max_branch_executions,
            "consumed_complete_branch_executions_before_run": (
                args.consumed_complete_branch_executions
            ),
        }
        if adapter is not None:
            failure["generation_calls"] = adapter.cost_summary()
        if plan is not None:
            failure.update(
                state_count=len(plan["states"]),
                planned_complete_branch_executions=len(plan["states"]) * len(t02.BRANCH_IDS),
                actual_split_counts=dict(Counter(
                    state["split"] for state in plan["states"]
                )),
                plan_sha256=t02._digest(plan),
            )
        _write_json(args.out / "summary.json", failure)
        raise
    finally:
        if adapter is not None:
            adapter.close()


if __name__ == "__main__":
    raise SystemExit(main())
