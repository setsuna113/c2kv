"""Opt-in, fail-closed fixtures for a controlled native serving workload."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping


MANIFEST_SCHEMA = "c2kv.controlled_workload_fixture.v1"
TASK_SCHEMA = "c2kv.controlled_task_fixture.v1"


class ControlledWorkloadError(RuntimeError):
    """A fixture, request, or generated response differs from the control."""


def _canonical(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise ControlledWorkloadError("Controlled workload value is not finite JSON") from error


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ControlledWorkloadError(message)


class ControlledWorkload:
    """Own one task fixture and the next expected decision/generation ordinal."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        manifest = self._read_json(self.directory / "manifest.json")
        _require(manifest.get("schema") == MANIFEST_SCHEMA,
                 "Controlled workload manifest schema mismatch")
        tasks = manifest.get("tasks")
        _require(isinstance(tasks, dict) and tasks,
                 "Controlled workload manifest has no tasks")
        self._task_entries = tasks
        self._task_id: str | None = None
        self._task: dict[str, Any] | None = None
        self.fixture_sha256: str | None = None
        self._decision_index = 0
        self._generation_index = 0
        self._active: dict[str, Any] | None = None
        self._actual_outer_request_id: str | None = None

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, ValueError) as error:
            raise ControlledWorkloadError(f"Cannot read controlled fixture: {path}") from error
        _require(isinstance(value, dict), f"Controlled fixture is not an object: {path}")
        return value

    def _load_task(self, task_id: str) -> None:
        _require(self._task is None, "Controlled workload changed task in one server")
        entry = self._task_entries.get(task_id)
        _require(isinstance(entry, dict), f"Controlled workload has no task {task_id!r}")
        name, expected_sha = entry.get("file"), entry.get("sha256")
        _require(isinstance(name, str) and name == f"{task_id}.json"
                 and Path(name).name == name,
                 "Controlled workload task filename mismatch")
        _require(isinstance(expected_sha, str)
                 and re.fullmatch(r"[0-9a-f]{64}", expected_sha) is not None,
                 "Controlled workload task digest is invalid")
        path = self.directory / name
        try:
            data = path.read_bytes()
        except OSError as error:
            raise ControlledWorkloadError(f"Cannot read controlled task fixture: {path}") from error
        _require(hashlib.sha256(data).hexdigest() == expected_sha,
                 "Controlled workload task digest mismatch")
        try:
            task = json.loads(data.decode("utf-8"))
        except (UnicodeError, ValueError) as error:
            raise ControlledWorkloadError("Controlled workload task is not UTF-8 JSON") from error
        _require(isinstance(task, dict) and task.get("schema") == TASK_SCHEMA
                 and task.get("task_id") == task_id
                 and isinstance(task.get("session_id"), str)
                 and isinstance(task.get("decisions"), list),
                 "Controlled workload task schema or identity mismatch")
        for index, decision in enumerate(task["decisions"]):
            _require(isinstance(decision, dict)
                     and isinstance(decision.get("runner_payload"), dict)
                     and isinstance(decision.get("response"), dict)
                     and isinstance(decision.get("generations"), list),
                     f"Controlled workload decision {index} is incomplete")
            for generation in decision["generations"]:
                _require(isinstance(generation, dict)
                         and isinstance(generation.get("request"), dict)
                         and isinstance(generation.get("response"), dict)
                         and isinstance(generation.get("prepared_input"), dict)
                         and isinstance(generation.get("phase"), str)
                         and isinstance(generation.get("attempt_uid"), str),
                         f"Controlled workload decision {index} has an incomplete generation")
                response = generation["response"]
                ids, logprobs = response.get("output_ids"), response.get("token_logprobs")
                _require(isinstance(ids, list) and ids
                         and all(type(token) is int and token >= 0 for token in ids)
                         and isinstance(logprobs, list) and len(logprobs) == len(ids)
                         and all(isinstance(value, (int, float)) and not isinstance(value, bool)
                                 for value in logprobs)
                         and isinstance(response.get("finish_reason"), str)
                         and isinstance(response.get("shadow_features"), (dict, type(None))),
                         f"Controlled workload decision {index} has an invalid generation response")
        self._task_id, self._task = task_id, task
        self.fixture_sha256 = expected_sha

    def begin_decision(self, task_id: str, runner_payload: Mapping[str, Any]) -> None:
        if self._task is None:
            self._load_task(task_id)
        _require(task_id == self._task_id and self._active is None,
                 "Controlled workload task or decision overlap")
        decisions = self._task["decisions"]
        _require(self._decision_index < len(decisions),
                 "Controlled workload has an extra decision")
        expected = decisions[self._decision_index]
        actual = _canonical(runner_payload)
        source = expected["runner_payload"]
        _require(isinstance(actual.get("outer_request_id"), str)
                 and actual["outer_request_id"]
                 and isinstance(source.get("outer_request_id"), str)
                 and actual.get("session_id") == self._task["session_id"]
                 and {key: value for key, value in actual.items()
                      if key != "outer_request_id"}
                 == {key: value for key, value in source.items()
                     if key != "outer_request_id"},
                 f"Controlled workload decision {self._decision_index} input mismatch")
        self._active = expected
        self._actual_outer_request_id = actual["outer_request_id"]
        self._generation_index = 0

    def next_generation(self, request: Mapping[str, Any], *, phase: str,
                        memory: Any) -> dict[str, Any]:
        _require(self._active is not None, "Controlled workload has no active decision")
        generations = self._active["generations"]
        index = self._generation_index
        _require(index < len(generations), "Controlled workload has an extra generation")
        expected = generations[index]
        _require(phase == expected["phase"],
                 f"Controlled workload generation {index} phase mismatch")

        from benchmarks.memory_runtime.event_native import memory_to_dict

        prepared = _canonical(memory_to_dict(memory))
        _require(prepared == expected["prepared_input"],
                 f"Controlled workload generation {index} prepared input mismatch")

        source = expected["request"]
        current = _canonical(request)
        stable_fields = (
            "schema", "session_id", "packing_version", "raw_layout_profile",
            "encoding_scope", "system_input_ids", "workspace_input_ids",
            "encoder_chunks", "compression_ratio",
            "shadow_features", "raw_tool_segments", "tool_gist_segments",
            "paper_whole_full_kv_tokens", "sampling_profile",
        )
        _require(all(current.get(name) == source.get(name) for name in stable_fields),
                 f"Controlled workload generation {index} native input mismatch")
        _require(current.get("sampling_params") == source.get("sampling_params"),
                 f"Controlled workload generation {index} sampling mismatch")
        _require(current.get("outer_request_id") == self._actual_outer_request_id,
                 f"Controlled workload generation {index} decision identity mismatch")
        length = len(expected["response"]["output_ids"])
        _require(length <= current["sampling_params"]["max_new_tokens"],
                 "Controlled workload output length exceeds the original cap")
        self._generation_index += 1
        return expected

    def complete_decision(self, record: Mapping[str, Any]) -> None:
        _require(self._active is not None, "Controlled workload has no active decision")
        expected = self._active
        count = len(expected["generations"])
        _require(self._generation_index == count
                 and isinstance(record, Mapping)
                 and len(record.get("generation_trace", ())) == count,
                 f"Controlled workload decision {self._decision_index} generation count mismatch")
        _require(record.get("response") == expected["response"],
                 f"Controlled workload decision {self._decision_index} response mismatch")
        self._decision_index += 1
        self._active = None
        self._actual_outer_request_id = None


__all__ = ["ControlledWorkload", "ControlledWorkloadError", "MANIFEST_SCHEMA", "TASK_SCHEMA"]
