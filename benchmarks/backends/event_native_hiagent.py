"""Opt-in client side of the native HiAgent phase bridge."""
from __future__ import annotations

import json
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict, Optional

from .base import Backend, BackendError

try:  # ``python benchmarks/proxy.py`` loads ``backends`` as a top-level package.
    from native_hiagent_protocol import (
        ENDPOINT_PATH,
        RECEIPT_FIELD,
        build_call_envelope,
        load_policy_sampling,
        validate_official_eval_context,
        validate_policy_sampling,
        validate_response,
    )
except ModuleNotFoundError:  # Package import used by tests and library callers.
    from ..native_hiagent_protocol import (
        ENDPOINT_PATH,
        RECEIPT_FIELD,
        build_call_envelope,
        load_policy_sampling,
        validate_official_eval_context,
        validate_policy_sampling,
        validate_response,
    )


def _json_snapshot(value: Any) -> Any:
    return json.loads(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ))


class NativeHiAgentBridgeClient:
    """Build one isolated native request for each reserved proxy model call."""

    def __init__(
        self,
        policy_sampling: Mapping[str, Any],
        *,
        max_calls_per_task: int = 96,
    ) -> None:
        if type(max_calls_per_task) is not int or max_calls_per_task <= 0:
            raise ValueError("max_calls_per_task must be a positive integer")
        self.policy_sampling = validate_policy_sampling(policy_sampling)
        self.max_calls_per_task = max_calls_per_task
        self._call_counts: dict[tuple[str, str], int] = {}
        self._lock = threading.Lock()

    @classmethod
    def from_policy_sampling_file(
        cls,
        path: str | Path,
        *,
        max_calls_per_task: int = 96,
    ) -> "NativeHiAgentBridgeClient":
        return cls(
            load_policy_sampling(path),
            max_calls_per_task=max_calls_per_task,
        )

    @staticmethod
    def _task_key(context: Mapping[str, Any]) -> tuple[str, str]:
        normalized = validate_official_eval_context(context)
        return normalized["benchmark"], normalized["task_id"]

    def prepare_call(
        self,
        payload: Mapping[str, Any],
        *,
        parent_request_id: str,
        official_eval_context: Mapping[str, Any],
        proxy_attempt_uid: str,
        phase: str,
    ) -> tuple[str, dict[str, Any]]:
        """Return ``(endpoint_path, normalized_envelope)`` for one call.

        The proxy invokes this only after its shared generation budget and
        durable attempt journal have accepted the call.  The lock makes the
        per-task ordinal and the unique native session identity atomic.
        """
        context = validate_official_eval_context(official_eval_context)
        task_key = self._task_key(context)
        with self._lock:
            ordinal = self._call_counts.get(task_key, 0) + 1
            if ordinal > self.max_calls_per_task:
                raise RuntimeError(
                    f"native HiAgent call limit exhausted for task "
                    f"{context['task_id']!r} ({self.max_calls_per_task}/"
                    f"{self.max_calls_per_task})"
                )
            envelope = build_call_envelope(
                payload,
                parent_request_id=parent_request_id,
                official_eval_context=context,
                proxy_attempt_uid=proxy_attempt_uid,
                phase=phase,
                call_ordinal=ordinal,
                policy_sampling=self.policy_sampling,
            )
            self._call_counts[task_key] = ordinal
        return ENDPOINT_PATH, envelope

    def validate_response(
        self,
        value: Any,
        envelope: Mapping[str, Any],
    ) -> dict[str, Any]:
        return validate_response(value, envelope)

    def call_count(self, official_eval_context: Mapping[str, Any]) -> int:
        task_key = self._task_key(official_eval_context)
        with self._lock:
            return self._call_counts.get(task_key, 0)


class EventNativeHiAgentBackend(Backend):
    """Chat shaping for the explicitly supported native HiAgent arms."""

    name = "event_native_hiagent"

    def __init__(self, post_json) -> None:
        self._post_json = post_json

    def prepare_chat(
        self,
        payload: Dict[str, Any],
        arm,
        repair_plan: Optional[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        arm_shape = (
            getattr(arm, "name", None),
            getattr(arm, "text_policy", None),
            getattr(arm, "compress_history", None),
            getattr(arm, "native_messages", None),
        )
        if arm_shape not in {
            ("hiagent_full_native", "hiagent_full", False, True),
            (
                "hiagent_envelope_only_native",
                "hiagent_envelope_only",
                False,
                True,
            ),
        }:
            raise BackendError(
                "upstream",
                "event_native_hiagent requires the exact hiagent_full_native or "
                "hiagent_envelope_only_native arm",
            )
        if repair_plan is not None:
            raise BackendError(
                "upstream", "event_native_hiagent does not accept a repair plan"
            )
        return _json_snapshot(payload)

    def normalize_response(self, data: Dict[str, Any]) -> Dict[str, Any]:
        choices = data.get("choices")
        if (
            not isinstance(choices, list)
            or not choices
            or not isinstance(choices[0], Mapping)
            or not isinstance(choices[0].get("message"), Mapping)
        ):
            raise BackendError("upstream", "native HiAgent response lacks a message")
        receipt = data.get(RECEIPT_FIELD)
        if not isinstance(receipt, Mapping):
            raise BackendError("upstream", "native HiAgent response lacks its receipt")
        choice = choices[0]
        message = choice["message"]
        return {
            "content": message.get("content"),
            "tool_calls": message.get("tool_calls"),
            "finish_reason": choice.get("finish_reason"),
            "usage": data.get("usage"),
            "cost": {RECEIPT_FIELD: _json_snapshot(receipt)},
        }


__all__ = ["EventNativeHiAgentBackend", "NativeHiAgentBridgeClient"]
