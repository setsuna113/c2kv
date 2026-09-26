"""SGLang transport for event-native packed history generation.

The adapter owns no model or KV tensors.  It sends the exact ``PackedMemory``
layout to the C2KV SGLang engine, validates the returned layout/accounting, and
exposes the small generator interface consumed by ``EventNativeDecisionRunner``.
There is deliberately no alternate generation path when the upstream request
fails.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterator, Mapping, Sequence
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, Request, build_opener

from .encoding_scope import validate_encoding_scope
from .packing import PACKING_VERSION, EncoderChunk, PackedMemory

if TYPE_CHECKING:
    from .shadow_features import ShadowFeatureConfig


REQUEST_SCHEMA = "c2kv-native-packed-generation-v1"
RESPONSE_SCHEMA = "c2kv-native-packed-generation-response-v1"
HANDLE_SCHEMA = "c2kv-native-chunk-handle-v1"
GENERATION_SCHEMA = "event-native-sglang-generation-v1"
HTTP_JOURNAL_SCHEMA = "event-native-sglang-http-v1"
SESSION_CACHE_POLICY = "external-sglang-content-addressed-chunks-v1"
SHADOW_FEATURE_SCHEMA = "event-native-shadow-features-v1"
EXTRACTION_FAILURE_SCHEMA = "c2kv-native-extraction-failure-v1"
EXTRACTION_BUDGET_ERROR_CODE = "C2KV_EXTRACTION_BUDGET_EXHAUSTED"
PREWARM_RESPONSE_SCHEMA = "c2kv-native-prewarm-response-v1"
PREWARM_HTTP_JOURNAL_SCHEMA = "event-native-sglang-prewarm-http-v1"
ASYNC_PREWARM_POLL_INTERVAL_SECONDS = 0.05


@dataclass(frozen=True)
class SGLangEventNativeGenerationResult:
    """The observable ``EventNativeGenerationResult`` interface."""

    token_ids: tuple[int, ...]
    finish_reason: str
    token_logprobs: tuple[float, ...]
    stats: dict[str, Any]


class SGLangEventNativeError(RuntimeError):
    """One terminal upstream or response-contract failure."""


class SGLangTransportError(SGLangEventNativeError):
    """Ambiguous infrastructure failure; retry only from a fresh task boundary."""


class SGLangExtractionBudgetExhausted(SGLangEventNativeError):
    """A verified upstream stop after consuming the remaining extraction budget."""

    def __init__(self, message: str, receipt: Mapping[str, Any]):
        super().__init__(message)
        self.receipt = copy.deepcopy(dict(receipt))


@dataclass
class _Session:
    session_id: str
    generation: int = 0
    handles: set[str] = field(default_factory=set)
    materialized_history_handles: set[str] = field(default_factory=set)


@dataclass
class _DecisionScope:
    session_id: str | None
    session_handles: set[str]
    retained_handles: set[str] = field(default_factory=set)
    generate_calls: int = 0
    pending_stats: dict[str, Any] | None = None
    reset_reason: str | None = None
    generation_handles: dict[int, tuple[str, set[str]]] = field(default_factory=dict)
    generation_results: dict[int, SGLangEventNativeGenerationResult] = field(default_factory=dict)
    extracted_handles: set[str] = field(default_factory=set)


def _positive_int(value: Any, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _positive_finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a positive finite number")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return parsed


def _json_snapshot(value: Any, name: str) -> Any:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        return json.loads(encoded)
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} must contain finite JSON data") from error


def _json_object(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a JSON object")
    copied = _json_snapshot(value, name)
    if not isinstance(copied, dict):
        raise TypeError(f"{name} must be a JSON object")
    return copied


def _base_url(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("upstream must be a nonempty http:// base URL")
    parsed = urlsplit(value)
    if (
        parsed.scheme != "http"
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path.rstrip("/")
    ):
        raise ValueError(
            "upstream must be a bare http:// host:port URL without credentials, path, query, or fragment"
        )
    return value.rstrip("/")


def _token_ids(value: Any, name: str, *, nonempty: bool = False) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise TypeError(f"{name} must be a sequence of token IDs")
    result = tuple(value)
    if nonempty and not result:
        raise ValueError(f"{name} must not be empty")
    if any(type(token) is not int or token < 0 for token in result):
        raise ValueError(f"{name} must contain nonnegative integer token IDs")
    return result


def _eos_ids(value: Any) -> tuple[int, ...]:
    if isinstance(value, bool):
        raise TypeError("eos_token_ids must be an integer or sequence of integers")
    values = (value,) if isinstance(value, int) else _token_ids(value, "eos_token_ids")
    if not values:
        raise ValueError("eos_token_ids must not be empty")
    return tuple(sorted(set(values)))


def _nonnegative_int(value: Any, name: str) -> int:
    if type(value) is not int or value < 0:
        raise SGLangEventNativeError(f"{name} must be a nonnegative integer")
    return value


def _optional_nonnegative_int(value: Any, name: str) -> int | None:
    if value is None:
        return None
    return _nonnegative_int(value, name)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


class _HTTPJournal:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.name:
            raise ValueError("journal_path must include a file name")
        self._lock = threading.Lock()

    def append(self, record: Mapping[str, Any]) -> None:
        encoded = _canonical_bytes(record) + b"\n"
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            flags = os.O_APPEND | os.O_CREAT | os.O_WRONLY | getattr(os, "O_BINARY", 0)
            descriptor = os.open(self.path, flags, 0o600)
            try:
                remaining = memoryview(encoded)
                while remaining:
                    written = os.write(descriptor, remaining)
                    if written <= 0:
                        raise OSError("HTTP journal append made no progress")
                    remaining = remaining[written:]
                os.fsync(descriptor)
            finally:
                os.close(descriptor)


class SGLangEventNativeGenerator:
    """Finite, no-fallback adapter for SGLang's native packed C2KV endpoint."""

    cache_trace_schema = "event-native-cache-trace-v1"
    decode_strategy = "incremental"
    session_cache_policy = SESSION_CACHE_POLICY

    def __init__(
        self,
        upstream: str,
        *,
        expected_model_path: str | Path,
        model_context: int,
        max_new_tokens: int,
        max_generation_calls: int,
        max_extraction_calls: int,
        max_tool_extraction_calls: int | None = None,
        max_tool_repair_calls: int | None = None,
        expected_tool_checkpoint_contract: Mapping[str, Any] | None = None,
        timeout_seconds: float,
        eos_token_ids: int | Sequence[int],
        eos_source: str,
        encoding_scope: str = "current",
        journal_path: str | Path | None = None,
        sampling_params: Mapping[str, Any] | None = None,
        sampling_profile: str = "greedy-v1",
        shadow_feature_config: "ShadowFeatureConfig | None" = None,
        max_response_bytes: int = 16 * 1024 * 1024,
        opener: Any | None = None,
    ) -> None:
        self.upstream = _base_url(upstream)
        if not isinstance(expected_model_path, (str, Path)):
            raise TypeError("expected_model_path must be a path string")
        self.expected_model_path = str(expected_model_path).rstrip("/")
        if not self.expected_model_path:
            raise ValueError("expected_model_path must be nonempty")
        self.model_context = _positive_int(model_context, "model_context")
        self.max_new_tokens = _positive_int(max_new_tokens, "max_new_tokens")
        self.max_generation_calls = _positive_int(
            max_generation_calls, "max_generation_calls"
        )
        self.timeout_seconds = _positive_finite(timeout_seconds, "timeout_seconds")
        self.eos_token_ids = _eos_ids(eos_token_ids)
        if not isinstance(eos_source, str) or not eos_source:
            raise ValueError("eos_source must be a nonempty string")
        self.eos_source = eos_source
        self.encoding_scope = validate_encoding_scope(encoding_scope)
        self.max_extraction_calls = _positive_int(
            max_extraction_calls, "max_extraction_calls"
        )
        self.max_tool_extraction_calls = (
            None if max_tool_extraction_calls is None else
            _positive_int(max_tool_extraction_calls, "max_tool_extraction_calls"))
        self.max_tool_repair_calls = (
            None if max_tool_repair_calls is None else
            _positive_int(max_tool_repair_calls, "max_tool_repair_calls"))
        self.expected_tool_checkpoint_contract = (
            None if expected_tool_checkpoint_contract is None else
            _json_object(expected_tool_checkpoint_contract, "expected_tool_checkpoint_contract"))
        self.tool_repair_calls = 0
        self.max_response_bytes = _positive_int(max_response_bytes, "max_response_bytes")

        sampling = _json_object(
            {"temperature": 0.0, "seed": 0}
            if sampling_params is None
            else sampling_params,
            "sampling_params",
        )
        forbidden = {"max_new_tokens", "max_tokens", "stream", "n"} & set(sampling)
        if forbidden:
            raise ValueError(
                "sampling_params must omit per-call fields: " + ", ".join(sorted(forbidden))
            )
        temperature = sampling.get("temperature", 0)
        if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
            raise ValueError("sampling_params.temperature must be numeric")
        if sampling_profile not in {"greedy-v1", "acebench-agent-v1"}:
            raise ValueError("Unknown native sampling profile")
        if sampling_profile == "acebench-agent-v1":
            if float(temperature) != 0.001 or sampling.get("top_p") != 1:
                raise ValueError("ACEBench Agent requires temperature=0.001 and top_p=1")
            if "seed" in sampling or "sampling_seed" in sampling:
                raise ValueError("ACEBench Agent profile has no explicit seed")
        elif float(temperature) != 0.0:
            raise ValueError("event-native D3/GP serving requires greedy temperature=0")
        self.sampling_profile = sampling_profile
        self.sampling_params = sampling
        self.shadow_feature_config = shadow_feature_config

        self._http_journal = _HTTPJournal(journal_path) if journal_path is not None else None
        self._opener = opener or build_opener(ProxyHandler({}))
        if not callable(getattr(self._opener, "open", None)):
            raise TypeError("opener must expose open(request, timeout=...)")

        self._requests_submitted = 0
        self.extraction_calls_reserved = 0
        self.tool_extraction_calls_reserved = 0
        self._model_binding: dict[str, Any] | None = None
        self._tool_projection_identity: str | None = None
        self._kv_bytes_per_token: int | None = None
        self._active_decision_scope: _DecisionScope | None = None
        self._session_cache: _Session | None = None
        self.last_generation_trace: dict[str, Any] | None = None
        self.last_cache_lifecycle_trace: dict[str, Any] | None = None
        self._exact_snapshots: dict[str, dict[str, Any]] = {}
        controlled_directory = os.environ.get("C2KV_NATIVE_CONTROLLED_WORKLOAD_DIR")
        if controlled_directory:
            from .controlled_workload import ControlledWorkload

            self.controlled_workload = ControlledWorkload(controlled_directory)
        else:
            self.controlled_workload = None
        self.async_compression_enabled = os.environ.get("C2KV_NATIVE_ASYNC_COMPRESSION") == "1"
        self.background_fit_budget_enabled = (
            self.async_compression_enabled
            and os.environ.get("C2KV_NATIVE_PREWARM_FIT_BUDGET") == "1")
        self._background_fit_stats = {
            "policy": "whole-unit-history-budget-v1", "checks": 0,
            "skipped_unit_checks": 0, "skipped_chunk_checks": 0,
            "skipped_source_token_checks": 0, "history_budget_tokens": None,
        }
        self.cross_turn_prewarm_enabled = os.environ.get("C2KV_NATIVE_CROSS_TURN_PREWARM") == "1"
        self._prewarm_owner_id = str(uuid.uuid4()) if (
            self.cross_turn_prewarm_enabled or self.async_compression_enabled) else None
        self._prewarm_job: dict[str, Any] | None = None
        self._prewarm_last_receipt: dict[str, Any] | None = None
        self._prewarm_budget_unknown = False
        self._background_submitted_chunks = 0
        self._background_settled_chunks = 0
        self._background_model_calls = 0
        self._background_sync_wall = {"poll": 0.0, "submit": 0.0, "final_drain": 0.0}
        self._background_poll_rpcs = 0
        self._background_coalesced_polls = 0
        self._last_background_provider_offer: dict[str, Any] | None = None
        self._background_provider_failed = False
        self._background_provider_interrupt: BaseException | None = None
        self._last_decision_extracted_handles: set[str] = set()

    def _exact_local_state(self) -> dict[str, Any]:
        session = self._session_cache
        scope = self._active_decision_scope
        state = {
            "session": None if session is None else {
                "session_id": session.session_id, "generation": session.generation,
                "handles": sorted(session.handles),
                "materialized_history_handles": sorted(session.materialized_history_handles),
            },
            "scope": None if scope is None else {
                "session_id": scope.session_id,
                "session_handles": sorted(scope.session_handles),
                "retained_handles": sorted(scope.retained_handles),
                "generate_calls": scope.generate_calls,
                "pending_stats": copy.deepcopy(scope.pending_stats),
                "reset_reason": scope.reset_reason,
                "generation_handles": [
                    {"generation_index": index, "generation_id": generation_id,
                     "handles": sorted(handles)}
                    for index, (generation_id, handles) in sorted(scope.generation_handles.items())
                ],
            },
            "requests_submitted": self._requests_submitted,
            "extraction_calls_reserved": self.extraction_calls_reserved,
            "sampling_params": copy.deepcopy(self.sampling_params),
            "last_generation_trace": copy.deepcopy(self.last_generation_trace),
            "last_cache_lifecycle_trace": copy.deepcopy(self.last_cache_lifecycle_trace),
            "model_binding": copy.deepcopy(self._model_binding),
            "kv_bytes_per_token": self._kv_bytes_per_token,
        }
        if self._tool_projection_identity is not None:
            state["tool_projection_identity"] = self._tool_projection_identity
        if self.max_tool_extraction_calls is not None:
            state["tool_extraction_calls_reserved"] = self.tool_extraction_calls_reserved
        if self.max_tool_repair_calls is not None:
            state["tool_repair_calls"] = self.tool_repair_calls
        return state

    def _exact_request(self, operation: str, snapshot_id: str | None = None) -> dict[str, Any]:
        response, status = self._read_json(Request(
            self.upstream + "/v1/c2kv/exact_state",
            data=_canonical_bytes({"operation": operation, "snapshot_id": snapshot_id}),
            headers={"Content-Type": "application/json"}, method="POST",
        ), label="SGLang exact_state")
        if status != 200 or not isinstance(response, dict):
            error = response.get("error") if isinstance(response, dict) else response
            raise SGLangEventNativeError(f"Exact state {operation} failed: HTTP {status}: {error}")
        if response.get("schema") != "c2kv-exact-backend-state-v1" or response.get("operation") != operation:
            raise SGLangEventNativeError("Exact state response schema/operation mismatch")
        if snapshot_id is not None and response.get("snapshot_id") != snapshot_id:
            raise SGLangEventNativeError("Exact state snapshot ID mismatch")
        if operation != "release":
            digests = response.get("component_digests")
            if response.get("exact") is not True or not isinstance(digests, dict) or set(digests) != {"actor_kv", "actor_positions", "backend_stats", "rng"}:
                raise SGLangEventNativeError("Exact backend state components are incomplete")
            for value in digests.values():
                if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                    raise SGLangEventNativeError("Exact backend digest is malformed")
        return response

    @staticmethod
    def _exact_receipt(remote: Mapping[str, Any], local: Mapping[str, Any]) -> dict[str, Any]:
        digests = dict(remote["component_digests"])
        for name, payload in {
            "actor_kv": {"session": local["session"], "scope": local["scope"]},
            "actor_positions": {"last_generation_trace": local["last_generation_trace"]},
            "backend_stats": local,
            "rng": {"sampling_params": local["sampling_params"]},
        }.items():
            digests[name] = hashlib.sha256(_canonical_bytes({
                "backend": digests[name], "adapter": payload,
            })).hexdigest()
        return {"schema": "event-native-sglang-exact-state-v1",
                "snapshot_id": remote["snapshot_id"], "component_digests": digests,
                "exact": True, "backend": copy.deepcopy(dict(remote))}

    def capture_exact_state(self) -> dict[str, Any]:
        """Copy live backend tensors and current adapter decision/session state."""
        self._ensure_model_info()
        local = self._exact_local_state()
        remote = self._exact_request("capture")
        receipt = self._exact_receipt(remote, local)
        self._exact_snapshots[remote["snapshot_id"]] = {
            "local": local, "remote": remote, "receipt": receipt,
        }
        return copy.deepcopy(receipt)

    def restore_exact_state(self, snapshot: Mapping[str, Any]) -> dict[str, Any]:
        """Restore tensors plus adapter state; preserve an active context object."""
        snapshot_id = snapshot.get("snapshot_id")
        saved = self._exact_snapshots.get(snapshot_id)
        if saved is None or snapshot.get("component_digests") != saved["receipt"]["component_digests"]:
            raise SGLangEventNativeError("Unknown or changed exact adapter snapshot")
        local = copy.deepcopy(saved["local"])
        scope = self._active_decision_scope
        known_results = {} if scope is None else dict(scope.generation_results)
        if (scope is None) != (local["scope"] is None):
            raise SGLangEventNativeError("Restore must use the captured decision-scope boundary")
        remote = self._exact_request("restore", snapshot_id)
        if remote.get("verified_from_live_state") is not True or remote["component_digests"] != saved["remote"]["component_digests"]:
            raise SGLangEventNativeError("Backend failed live exact-state verification")
        session = local["session"]
        self._session_cache = None if session is None else _Session(
            session_id=session["session_id"], generation=session["generation"],
            handles=set(session["handles"]),
            materialized_history_handles=set(session.get("materialized_history_handles", ())),
        )
        if scope is not None:
            for name, value in local["scope"].items():
                if name == "generation_handles":
                    value = {row["generation_index"]: (row["generation_id"], set(row["handles"]))
                             for row in value}
                elif name in {"session_handles", "retained_handles"}:
                    value = set(value)
                setattr(scope, name, value)
            scope.generation_results = {
                index: known_results[index]
                for index, (generation_id, _) in scope.generation_handles.items()
                if index in known_results
                and known_results[index].stats.get("sglang_transport", {}).get("generation_id") == generation_id
            }
        self._requests_submitted = local["requests_submitted"]
        self.extraction_calls_reserved = local["extraction_calls_reserved"]
        if "tool_extraction_calls_reserved" in local:
            self.tool_extraction_calls_reserved = local["tool_extraction_calls_reserved"]
        if "tool_repair_calls" in local:
            self.tool_repair_calls = local["tool_repair_calls"]
        self.sampling_params = local["sampling_params"]
        self.last_generation_trace = local["last_generation_trace"]
        self.last_cache_lifecycle_trace = local["last_cache_lifecycle_trace"]
        self._model_binding = local["model_binding"]
        self._tool_projection_identity = local.get("tool_projection_identity")
        self._kv_bytes_per_token = local["kv_bytes_per_token"]
        receipt = self._exact_receipt(remote, self._exact_local_state())
        if receipt["component_digests"] != snapshot["component_digests"]:
            raise SGLangEventNativeError("Adapter state differs after exact restoration")
        receipt["verified_from_live_state"] = True
        return receipt

    def release_exact_state(self, snapshot: Mapping[str, Any]) -> None:
        snapshot_id = snapshot.get("snapshot_id")
        if snapshot_id not in self._exact_snapshots:
            raise SGLangEventNativeError("Unknown exact adapter snapshot")
        response = self._exact_request("release", snapshot_id)
        if response.get("released") is not True:
            raise SGLangEventNativeError("Backend did not release the exact snapshot")
        del self._exact_snapshots[snapshot_id]

    @property
    def shadow_feature_config(self) -> "ShadowFeatureConfig | None":
        return self._shadow_feature_config

    @shadow_feature_config.setter
    def shadow_feature_config(self, config: "ShadowFeatureConfig | None") -> None:
        if config is not None:
            if type(getattr(config, "enabled", None)) is not bool:
                raise TypeError(
                    "shadow_feature_config must expose the ShadowFeatureConfig contract"
                )
            prefill_layer = getattr(config, "prefill_layer", None)
            if prefill_layer is not None and type(prefill_layer) is not int:
                raise TypeError("shadow_feature_config.prefill_layer must be int or None")
            if getattr(config, "memgen_layer", None) is not None:
                raise ValueError(
                    "SGLang native packed generation supports prefill shadow features only"
                )
        self._shadow_feature_config = config

    def configure_shadow_features(self, config: "ShadowFeatureConfig | None") -> None:
        self.shadow_feature_config = config

    def kv_bytes_per_token(self) -> int:
        """Read the serving model's exact KV unit once, without loading weights."""

        self._ensure_model_info()
        assert self._kv_bytes_per_token is not None
        return self._kv_bytes_per_token

    @contextmanager
    def decision_scope(self, *, session_id: str | None = None) -> Iterator[None]:
        if self._active_decision_scope is not None:
            raise RuntimeError("decision_scope is not reentrant")
        if session_id is not None and (
            not isinstance(session_id, str) or not session_id
        ):
            raise ValueError("session_id must be None or a nonempty string")
        self._reconcile_for_session(session_id)
        self._last_decision_extracted_handles = set()

        reset_reason = None
        if session_id is None:
            self._session_cache = None
            session_handles: set[str] = set()
        else:
            if self._session_cache is not None and self._session_cache.session_id != session_id:
                self._session_cache = None
                reset_reason = "session_id_changed"
            session_handles = (
                set(self._session_cache.handles)
                if self._session_cache is not None
                else set()
            )
        scope = _DecisionScope(
            session_id=session_id,
            session_handles=session_handles,
            reset_reason=reset_reason,
        )
        self._active_decision_scope = scope
        try:
            yield
        except BaseException:
            if scope.pending_stats is not None:
                scope.pending_stats["session_cache_commit_status"] = "cleared_on_failure"
            try:
                if self.async_compression_enabled:
                    self.reconcile_cross_turn_prewarm(operation="cancel")
            finally:
                if session_id is not None:
                    self._session_cache = None
            raise
        else:
            self._last_decision_extracted_handles = set(scope.extracted_handles)
            if session_id is not None and scope.pending_stats is not None:
                generation = (
                    self._session_cache.generation + 1
                    if self._session_cache is not None
                    else 1
                )
                self._session_cache = _Session(
                    session_id=session_id,
                    generation=generation,
                    handles=set(scope.retained_handles),
                    materialized_history_handles=(
                        set(self._session_cache.materialized_history_handles)
                        if self._session_cache is not None else set()
                    ),
                )
                scope.pending_stats["session_cache_commit_status"] = "committed"
        finally:
            self._active_decision_scope = None

    def close_session(self) -> None:
        if (
            self._active_decision_scope is not None
            and self._active_decision_scope.session_id is not None
        ):
            raise RuntimeError("cannot close a session inside its active decision_scope")
        self.reconcile_cross_turn_prewarm(operation="cancel")
        self._session_cache = None

    def session_cache_info(self) -> dict[str, Any]:
        session = self._session_cache
        return {
            "policy": self.session_cache_policy,
            "session_id": session.session_id if session is not None else None,
            "generation": session.generation if session is not None else 0,
            "cpu_memo_present": False,
            "cpu_memo_logical_bytes": 0,
            "device_raw_snapshot_present": False,
            "device_raw_snapshot_logical_bytes": 0,
            "device_raw_snapshot_backing_bytes": 0,
            "device_raw_snapshot_prefix_tokens": 0,
            "device_raw_snapshot_raw_tokens": 0,
            "last_transfer_bytes_in": 0,
            "last_transfer_bytes_out": 0,
            "last_transfer_bytes_total": 0,
            "chunk_handles_present": bool(session and session.handles),
            "chunk_handle_count": len(session.handles) if session is not None else 0,
            "materialized_history_handle_count": (
                len(session.materialized_history_handles) if session is not None else 0
            ),
            "external_cache_state": "content_addressed_lru_unowned",
            "remote_pins_held": False,
            "last_lifecycle_trace": self.last_cache_lifecycle_trace,
            **({"cross_turn_prewarm": {
                "enabled": True,
                "outstanding_job_id": self._prewarm_job["job_id"] if self._prewarm_job else None,
                "last_receipt": copy.deepcopy(self._prewarm_last_receipt),
                "budget_known": not self._prewarm_budget_unknown,
            }} if self.cross_turn_prewarm_enabled else {}),
            **({"async_compression": self._background_stats()} if self.async_compression_enabled else {}),
        }

    def generate(
        self,
        memory: PackedMemory,
        *,
        ratio: int,
        max_new_tokens: int,
        eos_token_id: int | Sequence[int] | None = None,
        trace_context: Mapping[str, Any] | None = None,
        compression_chunks: Sequence[EncoderChunk] | None = None,
        paper_whole_full_kv_tokens: int | None = None,
        background_chunk_provider: Callable[[], Sequence[EncoderChunk] | None] | None = None,
        background_history_budget_tokens: int | None = None,
    ) -> SGLangEventNativeGenerationResult:
        """Submit one exact packed decision.  A failure is terminal and unretried."""

        self.last_generation_trace = None
        if not isinstance(memory, PackedMemory):
            raise TypeError("memory must be PackedMemory")
        ratio = _positive_int(ratio, "ratio")
        requested_tokens = _positive_int(max_new_tokens, "max_new_tokens")
        if requested_tokens > self.max_new_tokens:
            raise ValueError("max_new_tokens exceeds this adapter's finite cap")
        if trace_context is not None and not isinstance(trace_context, Mapping):
            raise TypeError("trace_context must be a mapping or None")
        if background_chunk_provider is not None and not callable(background_chunk_provider):
            raise TypeError("background_chunk_provider must be callable or None")
        if paper_whole_full_kv_tokens is not None:
            _positive_int(paper_whole_full_kv_tokens, "paper_whole_full_kv_tokens")
        if self._requests_submitted >= self.max_generation_calls:
            raise RuntimeError("SGLang generation cap exhausted; automatic retry is disabled")
        context = self._trace_context(trace_context)
        scope = self._active_decision_scope
        session_id = scope.session_id if scope is not None else context.get("session_id")
        self._reconcile_for_session(session_id)

        if self._session_cache is not None and scope is None:
            self._session_cache = None
        if scope is not None and scope.pending_stats is not None:
            scope.pending_stats["session_cache_commit_status"] = "discarded_by_regeneration"
            scope.pending_stats = None

        self._ensure_model_info()
        if self.background_fit_budget_enabled:
            _positive_int(background_history_budget_tokens, "background_history_budget_tokens")
        selected, extras, anchored = self._prepare_chunks(
            memory, ratio, compression_chunks,
            background_history_budget_tokens=background_history_budget_tokens)
        logical_tokens = (
            len(memory.system_input_ids)
            + sum(len(item["token_ids"]) for item in selected)
            + len(memory.workspace_input_ids)
            - sum(item["token_end"] - item["token_start"] for item in anchored)
        )
        if logical_tokens + requested_tokens > self.model_context:
            raise ValueError("logical packed prompt plus completion exceeds model_context")

        generation_index = (scope.generate_calls + 1) if scope is not None else 1
        generation_id = self._generation_id(context, generation_index)
        self._last_background_provider_offer = None
        self._background_provider_failed = False
        self._background_provider_interrupt = None
        remaining_extractions = (
            self.max_extraction_calls - self.extraction_calls_reserved
            - self._background_reserved_chunks()
        )
        if remaining_extractions < 0:
            raise RuntimeError("finite extraction-call cap is already exhausted")

        shadow_request = None
        config = self.shadow_feature_config
        if config is not None and config.enabled:
            shadow_request = {"prefill_layer": config.prefill_layer}
        effective_eos_ids = (
            self.eos_token_ids if eos_token_id is None else _eos_ids(eos_token_id)
        )
        engine_sampling = copy.deepcopy(self.sampling_params)
        if "seed" in engine_sampling:
            seed = engine_sampling.pop("seed")
            if "sampling_seed" in engine_sampling and engine_sampling["sampling_seed"] != seed:
                raise ValueError("seed and sampling_seed must agree")
            engine_sampling["sampling_seed"] = seed
        payload = {
            "schema": REQUEST_SCHEMA,
            "rid": context.get("attempt_uid") or generation_id,
            "session_id": session_id,
            "generation_id": generation_id,
            "packing_version": PACKING_VERSION,
            "raw_layout_profile": memory.raw_layout_profile,
            "encoding_scope": self.encoding_scope,
            "system_input_ids": list(memory.system_input_ids),
            "workspace_input_ids": list(memory.workspace_input_ids),
            "encoder_chunks": selected[:len(memory.chunks)],
            "compression_chunks": extras,
            "compression_ratio": ratio,
            "max_extraction_calls": remaining_extractions,
            "sampling_params": {
                **engine_sampling,
                "max_new_tokens": requested_tokens,
                "stop_token_ids": list(effective_eos_ids),
            },
            "shadow_features": shadow_request,
        }
        if paper_whole_full_kv_tokens is not None:
            payload["paper_whole_full_kv_tokens"] = paper_whole_full_kv_tokens
        if self.max_tool_extraction_calls is not None:
            payload["max_tool_extraction_calls"] = (
                self.max_tool_extraction_calls - getattr(self, "tool_extraction_calls_reserved", 0))
        if memory.raw_tool_segments:
            payload["raw_tool_segments"] = [dict(item) for item in memory.raw_tool_segments]
        if anchored:
            payload["tool_gist_segments"] = anchored
        if self.sampling_profile != "greedy-v1":
            payload["sampling_profile"] = self.sampling_profile
        outer_request_id = context.get("outer_request_id")
        if outer_request_id is not None:
            if not isinstance(outer_request_id, str) or not outer_request_id:
                raise ValueError("trace_context.outer_request_id must be a nonempty string")
            payload["outer_request_id"] = outer_request_id

        if self.async_compression_enabled and extras:
            try:
                self._submit_background_rows(
                    extras, ratio=ratio, session_id=session_id or generation_id,
                    outer_request_id=outer_request_id or generation_id,
                    selected_rows=selected,
                    after_native_rid=payload["rid"],
                )
            except BaseException:
                self.reconcile_cross_turn_prewarm(operation="cancel")
                raise
            extras = []
            payload["compression_chunks"] = []
            payload["max_extraction_calls"] = (
                self.max_extraction_calls - self.extraction_calls_reserved
                - self._background_reserved_chunks()
            )
        if self.async_compression_enabled and background_chunk_provider is not None:
            selected_history = {row["handle"] for row in selected
                                if row.get("projection_set") != "tool"}
            selected_tool = {row["handle"] for row in selected
                             if row.get("projection_set") == "tool"}
            selected_reserve = len(selected_history)
            if self.max_tool_extraction_calls is None:
                selected_reserve += len(selected_tool)
            payload["max_extraction_calls"] = min(
                payload["max_extraction_calls"], selected_reserve)

        controlled_generation = None
        if self.controlled_workload is not None:
            controlled_generation = self.controlled_workload.next_generation(
                payload, phase=context.get("phase"), memory=memory)
            requested_tokens = len(controlled_generation["response"]["output_ids"])
            payload["sampling_params"].update(
                max_new_tokens=requested_tokens, min_new_tokens=requested_tokens,
                ignore_eos=True, stop_token_ids=[])

        request_index = self._requests_submitted + 1
        self._requests_submitted += 1
        if scope is not None:
            scope.generate_calls += 1
        started = time.perf_counter()
        try:
            if self.async_compression_enabled and background_chunk_provider is not None:
                response, http_status = self._post_with_background_provider(
                    payload, request_index,
                    measurement_phase=context.get("phase"),
                    provider=background_chunk_provider,
                    ratio=ratio,
                    session_id=session_id or generation_id,
                    outer_request_id=outer_request_id or generation_id,
                    selected_chunks=tuple(memory.chunks) + tuple(
                        chunk for segment in memory.tool_gist_segments
                        for chunk in segment["chunks"]),
                    background_history_budget_tokens=background_history_budget_tokens,
                )
            else:
                response, http_status = self._post_native_generate(
                    payload, request_index,
                    measurement_phase=context.get("phase"),
                )
            self._charge_extractions(response)
            if self._background_provider_failed and self._prewarm_job is not None:
                self.reconcile_cross_turn_prewarm(operation="cancel")
            if self._background_provider_interrupt is not None:
                raise self._background_provider_interrupt
            if (self.async_compression_enabled and self.extraction_calls_reserved
                    + self._background_reserved_chunks() > self.max_extraction_calls):
                raise SGLangEventNativeError("foreground extraction exceeded reserved background budget")
            result = self._result_from_response(
                response,
                payload=payload,
                memory=memory,
                selected=selected,
                extras=extras,
                ratio=ratio,
                requested_tokens=requested_tokens,
                request_index=request_index,
                http_status=http_status,
                wall_seconds=time.perf_counter() - started,
                scope=scope,
                effective_eos_ids=effective_eos_ids,
            )
            if controlled_generation is not None:
                source_response = controlled_generation["response"]
                if len(result.token_ids) != requested_tokens:
                    raise SGLangEventNativeError(
                        "Controlled workload actual decode length mismatch")
                real_shadow = result.stats.get("shadow_features")
                source_shadow = source_response["shadow_features"]
                stats = dict(result.stats)
                if source_shadow is None:
                    stats.pop("shadow_features", None)
                else:
                    stats["shadow_features"] = copy.deepcopy(source_shadow)
                stats["controlled_workload"] = {
                    "schema": "c2kv.controlled_generation_receipt.v1",
                    "fixture_sha256": self.controlled_workload.fixture_sha256,
                    "source_attempt_uid": controlled_generation["attempt_uid"],
                    "actual_rid": payload["rid"],
                    "source_outer_request_id": controlled_generation["request"].get(
                        "outer_request_id"),
                    "actual_outer_request_id": payload.get("outer_request_id"),
                    "actual_output_ids_sha256": hashlib.sha256(
                        _canonical_bytes(list(result.token_ids))).hexdigest(),
                    "actual_shadow_sha256": (hashlib.sha256(
                        _canonical_bytes(real_shadow)).hexdigest()
                        if real_shadow is not None else None),
                    "actual_output_tokens": len(result.token_ids),
                    "controlled_fields": ["output_ids", "token_logprobs",
                                          "finish_reason", "shadow_features"],
                }
                result = replace(
                    result,
                    token_ids=tuple(source_response["output_ids"]),
                    token_logprobs=tuple(source_response["token_logprobs"]),
                    finish_reason=source_response["finish_reason"],
                    stats=stats,
                )
            self._record_materialized_history(
                session_id, (row["handle"] for row in (*selected, *extras)
                             if row.get("projection_set") != "tool")
            )
        except BaseException:
            try:
                if self.async_compression_enabled:
                    self.reconcile_cross_turn_prewarm(operation="cancel")
            finally:
                if scope is not None and scope.session_id is not None:
                    self._session_cache = None
            raise

        if scope is not None:
            handles = {item["handle"] for item in selected}
            if generation_index in scope.generation_handles:
                raise SGLangEventNativeError("Decision generation index was reused")
            scope.generation_handles[generation_index] = (generation_id, handles)
            scope.generation_results[generation_index] = result
            scope.retained_handles = set(handles)
            scope.extracted_handles.update(item["handle"] for item in (*selected, *extras))
            scope.pending_stats = result.stats
        if self.async_compression_enabled:
            result.stats["async_compression"] = self._background_stats()
        return result

    def resolve_decision(self, response, *, result, record):
        """Commit the selected generation's logical chunk view for the next turn."""
        scope = self._active_decision_scope
        if scope is None:
            raise SGLangEventNativeError("Native commit requires an active decision scope")
        if not isinstance(result, SGLangEventNativeGenerationResult):
            raise TypeError("Native commit requires a generation result")
        stats = result.stats
        index = stats.get("decision_scope_generation_index")
        transport = stats.get("sglang_transport") or {}
        view = scope.generation_handles.get(index) if type(index) is int else None
        if (view is None or view[0] != transport.get("generation_id")
                or scope.generation_results.get(index) is not result):
            raise SGLangEventNativeError("Selected native generation is not in this decision scope")
        expected = (record.get("commit_validation") or {}).get("selected_generation_index")
        if expected is not None and expected != index - 1:
            raise SGLangEventNativeError("Native commit result differs from selected generation")
        if scope.pending_stats is not None and scope.pending_stats is not stats:
            scope.pending_stats["session_cache_commit_status"] = "discarded_by_resolution"
        scope.retained_handles = set(view[1])
        scope.pending_stats = stats
        if scope.session_id is not None:
            stats["session_cache_commit_status"] = "pending"
        return {"schema": "event-native-sglang-commit-v1",
                "selected_generation_index": index - 1,
                "selected_generation_id": view[0],
                "retained_chunk_handles": sorted(view[1])}
    @property
    def last_decision_extracted_handles(self) -> frozenset[str]:
        return frozenset(self._last_decision_extracted_handles)

    def _background_reserved_chunks(self) -> int:
        if not self.async_compression_enabled or self._prewarm_job is None:
            return 0
        return len(self._prewarm_job["submitted_handles"])

    def _known_materialized_history(self, session_id: str | None) -> set[str]:
        session = self._session_cache
        if session is None or session.session_id != session_id:
            return set()
        return session.materialized_history_handles

    def _record_materialized_history(
        self, session_id: str | None, handles: Iterator[str],
    ) -> None:
        if not self.async_compression_enabled or session_id is None:
            return
        handles = set(handles)
        if not handles:
            return
        if self._session_cache is None or self._session_cache.session_id != session_id:
            self._session_cache = _Session(session_id=session_id)
        self._session_cache.materialized_history_handles.update(handles)

    def _background_stats(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "outstanding_job_id": self._prewarm_job["job_id"] if self._prewarm_job else None,
            "submitted_chunks": self._background_submitted_chunks,
            "reserved_chunks": self._background_reserved_chunks(),
            "settled_chunks": self._background_settled_chunks,
            "settled_model_calls": self._background_model_calls,
            "sync_poll_wall_seconds": self._background_sync_wall["poll"],
            "poll_rpc_count": self._background_poll_rpcs,
            "coalesced_poll_count": self._background_coalesced_polls,
            "sync_submit_wall_seconds": self._background_sync_wall["submit"],
            "sync_final_drain_wall_seconds": self._background_sync_wall["final_drain"],
            "response_waited_for_extras": False,
            "provider_offer": copy.deepcopy(self._last_background_provider_offer),
            "last_receipt": copy.deepcopy(self._prewarm_last_receipt),
            "budget_known": not self._prewarm_budget_unknown,
            **({"budget_fit_admission": dict(self._background_fit_stats)}
               if self.background_fit_budget_enabled else {}),
        }

    def _post_with_background_provider(
        self, payload: Mapping[str, Any], request_index: int, *,
        measurement_phase: str | None,
        provider: Callable[[], Sequence[EncoderChunk] | None],
        ratio: int, session_id: str, outer_request_id: str,
        selected_chunks: Sequence[EncoderChunk],
        background_history_budget_tokens: int | None = None,
    ) -> tuple[Any, int]:
        # Only the HTTP native POST runs on the worker. All budget and job state
        # stays on the caller thread while CPU history preparation may finish.
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="c2kv-native-http") as pool:
            pending = pool.submit(
                self._post_native_generate, payload, request_index,
                measurement_phase=measurement_phase)
            candidate: tuple[EncoderChunk, ...] | None = None
            provider_pending = True
            next_offer_at = 0.0
            while not pending.done():
                try:
                    if provider_pending:
                        ready = provider()
                        if ready is not None:
                            candidate = tuple(ready)
                            provider_pending = False
                    if (candidate is not None and not pending.done()
                            and time.perf_counter() >= next_offer_at):
                        offer = self.offer_background_chunks(
                            candidate, ratio=ratio, session_id=session_id,
                            outer_request_id=outer_request_id,
                            selected_chunks=selected_chunks,
                            history_budget_tokens=background_history_budget_tokens,
                            after_native_rid=payload["rid"])
                        self._last_background_provider_offer = offer
                        if (isinstance(offer, Mapping) and offer.get("status") == "deferred"
                                and offer.get("reason") == "background_job_running"):
                            next_offer_at = time.perf_counter() + 0.05
                        else:
                            candidate = None
                except BaseException as error:
                    self._background_provider_failed = True
                    self._last_background_provider_offer = {
                        "status": "failed", "error_type": type(error).__name__,
                        "message": str(error),
                    }
                    if not isinstance(error, Exception):
                        self._background_provider_interrupt = error
                    candidate = None
                    provider_pending = False
                try:
                    return pending.result(timeout=0.01)
                except FutureTimeoutError:
                    continue
                except BaseException as error:
                    if pending.done():
                        result = pending.result()
                        self._background_provider_interrupt = error
                        return result
                    self._background_provider_interrupt = error
                    provider_pending = False
                    candidate = None
            return pending.result()

    def _reconcile_for_session(self, session_id: str | None) -> None:
        if not self.async_compression_enabled:
            self.reconcile_cross_turn_prewarm()
            return
        if self._prewarm_job is None:
            if self._prewarm_budget_unknown:
                raise SGLangEventNativeError("async compression extraction budget is unknown")
            return
        operation = "poll" if self._prewarm_job["session_id"] == session_id else "cancel"
        self.reconcile_cross_turn_prewarm(operation=operation)

    def offer_background_chunks(
        self, chunks: Sequence[EncoderChunk], *, ratio: int,
        session_id: str, outer_request_id: str,
        selected_chunks: Sequence[EncoderChunk] | None = None,
        after_native_rid: str | None = None,
        history_budget_tokens: int | None = None,
    ) -> dict[str, Any] | None:
        """Offer known history while retaining capacity for current selected chunks."""
        if not self.async_compression_enabled:
            return None
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("session_id must be a nonempty string")
        if not isinstance(outer_request_id, str) or not outer_request_id:
            raise ValueError("outer_request_id must be a nonempty string")
        if after_native_rid is not None and (
            not isinstance(after_native_rid, str) or not after_native_rid
        ):
            raise ValueError("after_native_rid must be None or a nonempty string")
        scope = self._active_decision_scope
        if scope is not None and scope.session_id != session_id:
            raise ValueError("background offer session_id differs from decision_scope")
        self._reconcile_for_session(session_id)
        if self._prewarm_job is not None:
            return {"status": "deferred", "reason": "background_job_running",
                    "job_id": self._prewarm_job["job_id"]}
        if selected_chunks is None:
            return {"status": "deferred", "reason": "selected_reservation_unknown"}
        self._ensure_model_info()
        ratio = _positive_int(ratio, "ratio")
        selected = [self._chunk_payload(chunk, ratio) for chunk in selected_chunks]
        selected_handles = {row["handle"] for row in selected}
        rows = []
        seen = (set(selected_handles) | self._last_decision_extracted_handles
                | self._known_materialized_history(session_id))
        candidates = [self._chunk_payload(chunk, ratio) for chunk in chunks]
        # Test full units before removing already materialized parts. A cached
        # prefix does not make an oversized whole event fit the history budget.
        for row in self._budget_fitting_background_rows(
                candidates, ratio=ratio, history_budget_tokens=history_budget_tokens):
            if row.get("projection_set") == "tool" or row["handle"] in seen:
                continue
            seen.add(row["handle"])
            rows.append(row)
        return self._submit_background_rows(
            rows, ratio=ratio, session_id=session_id,
            outer_request_id=outer_request_id, selected_rows=selected,
            after_native_rid=after_native_rid,
        )

    def _budget_fitting_background_rows(
        self, rows: Sequence[Mapping[str, Any]], *, ratio: int,
        history_budget_tokens: int | None,
    ) -> list[Mapping[str, Any]]:
        """Skip optional whole units that cannot fit any legal foreground view."""
        if not self.background_fit_budget_enabled:
            return list(rows)
        budget = _positive_int(history_budget_tokens, "background_history_budget_tokens")
        if self.encoding_scope not in {"current", "event"}:
            raise ValueError("budget-fit prewarm requires whole-event current/event encoding")
        units: dict[tuple[Any, ...], dict[str, Mapping[str, Any]]] = {}
        for row in rows:
            if row.get("projection_set") == "tool":
                continue
            key = (row["event_id"], tuple(row["source_indices"]))
            units.setdefault(key, {})[row["handle"]] = row
        rejected = set()
        skipped_units = skipped_tokens = 0
        for chunks in units.values():
            gist_tokens = sum(
                (len(row["token_ids"]) + row.get("compression_ratio", ratio) - 1)
                // row.get("compression_ratio", ratio) for row in chunks.values())
            if gist_tokens > budget:
                skipped_units += 1
                rejected.update(chunks)
                skipped_tokens += sum(len(row["token_ids"]) for row in chunks.values())
        stats = self._background_fit_stats
        stats["checks"] += 1
        stats["history_budget_tokens"] = budget
        stats["skipped_unit_checks"] += skipped_units
        stats["skipped_chunk_checks"] += len(rejected)
        stats["skipped_source_token_checks"] += skipped_tokens
        return [row for row in rows if row["handle"] not in rejected]

    def _submit_background_rows(
        self, rows: Sequence[Mapping[str, Any]], *, ratio: int,
        session_id: str | None, outer_request_id: str,
        selected_rows: Sequence[Mapping[str, Any]],
        after_native_rid: str | None = None,
    ) -> dict[str, Any] | None:
        if not self.async_compression_enabled or not rows or session_id is None:
            return None
        if self._prewarm_budget_unknown:
            raise SGLangEventNativeError("async compression extraction budget is unknown")
        if self._prewarm_job is not None:
            return {"status": "deferred", "reason": "background_job_running",
                    "job_id": self._prewarm_job["job_id"]}
        selected_history = {row["handle"] for row in selected_rows
                            if row.get("projection_set") != "tool"}
        selected_tool = {row["handle"] for row in selected_rows
                         if row.get("projection_set") == "tool"}
        selected_reserve = len(selected_history)
        if self.max_tool_extraction_calls is None:
            selected_reserve += len(selected_tool)
        # At most one chunk per prompt token can become selected per finite
        # generation. Stop optional prewarming at that same session-scale bound.
        known_capacity = (self.model_context * self.max_generation_calls
                          - len(self._known_materialized_history(session_id)))
        if known_capacity <= 0:
            return None
        allowance = (self.max_extraction_calls - self.extraction_calls_reserved
                      - selected_reserve)
        if allowance <= 0:
            return None
        admitted = []
        tokens = 0
        seen = (set(selected_history) | selected_tool
                | self._known_materialized_history(session_id))
        for row in rows:
            if row.get("projection_set") == "tool" or row["handle"] in seen:
                continue
            length = len(row["token_ids"])
            if length > 8192 or tokens + length > 65536:
                break
            admitted.append({"handle": row["handle"], "token_ids": row["token_ids"],
                             "compression_ratio": row.get("compression_ratio", ratio)})
            seen.add(row["handle"])
            tokens += length
            if len(admitted) >= min(32, allowance, known_capacity):
                break
        if not admitted:
            return None
        job = {"owner_id": self._prewarm_owner_id, "job_id": str(uuid.uuid4()),
               "session_id": session_id, "outer_request_id": outer_request_id,
               "submitted_handles": [row["handle"] for row in admitted],
               "acknowledged": False}
        payload = {"operation": "submit", "owner_id": job["owner_id"],
                   "job_id": job["job_id"], "session_id": session_id,
                   "chunks": admitted, "max_extraction_calls": len(admitted),
                   "outer_request_id": outer_request_id,
                   "scheduling": "overlap-native-generation-v1"}
        if after_native_rid is not None:
            payload["after_native_rid"] = after_native_rid
        self._prewarm_job = job
        started = time.perf_counter()
        try:
            ack = self._prewarm_request(payload)
            if ack.get("status") not in {"queued", "running", "completed", "cancelled", "failed"}:
                raise SGLangEventNativeError("SGLang async prewarm submit was not acknowledged")
            if ack.get("submitted_chunks") != len(admitted):
                raise SGLangEventNativeError("SGLang async prewarm submit chunk count mismatch")
            job["acknowledged"] = True
            self._background_submitted_chunks += len(admitted)
        except BaseException:
            self._prewarm_budget_unknown = True
            raise
        finally:
            self._background_sync_wall["submit"] += time.perf_counter() - started
        return {"status": ack["status"], "job_id": job["job_id"],
                "submitted_chunks": len(admitted),
                "submitted_handles": list(job["submitted_handles"])}

    def _prewarm_request(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        operation = payload["operation"]
        request = Request(
            self.upstream + "/c2kv_native_prewarm",
            data=_canonical_bytes(payload),
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        if self._http_journal is not None:
            self._http_journal.append({
                "schema": PREWARM_HTTP_JOURNAL_SCHEMA,
                "event": "request", "operation": operation,
                "timestamp_unix_ns": time.time_ns(), "request": payload,
            })
        response, status = self._read_json(request, label=f"SGLang prewarm {operation}")
        if self._http_journal is not None:
            self._http_journal.append({
                "schema": PREWARM_HTTP_JOURNAL_SCHEMA,
                "event": "response", "operation": operation,
                "timestamp_unix_ns": time.time_ns(), "http_status": status,
                "response": response,
            })
        if status != 200 or not isinstance(response, Mapping):
            raise SGLangEventNativeError(f"SGLang prewarm {operation} returned HTTP {status}")
        if response.get("schema") != PREWARM_RESPONSE_SCHEMA:
            raise SGLangEventNativeError("SGLang prewarm response schema mismatch")
        if (response.get("owner_id"), response.get("job_id"), response.get("session_id")) != (
            payload["owner_id"], payload["job_id"], payload["session_id"]
        ):
            raise SGLangEventNativeError("SGLang prewarm job identity mismatch")
        return dict(response)

    def submit_cross_turn_prewarm(
        self, chunks: Sequence[EncoderChunk], *, ratio: int,
        session_id: str, outer_request_id: str,
    ) -> dict[str, Any] | None:
        """Queue bounded, already observed history chunks after a committed decision."""
        if not self.cross_turn_prewarm_enabled:
            return None
        if self._active_decision_scope is not None:
            raise RuntimeError("cross-turn prewarm must be submitted outside decision_scope")
        if self._prewarm_job is not None:
            raise RuntimeError("an earlier cross-turn prewarm job is still outstanding")
        if self._prewarm_budget_unknown:
            raise SGLangEventNativeError("cross-turn prewarm extraction budget is unknown")
        remaining = self.max_extraction_calls - self.extraction_calls_reserved
        if remaining <= 0 or not chunks:
            return None
        self._ensure_model_info()
        rows = []
        seen = set(self._last_decision_extracted_handles)
        tokens = 0
        for chunk in chunks:
            row = self._chunk_payload(chunk, ratio)
            if row["handle"] in seen:
                continue
            length = len(row["token_ids"])
            if length > 8192 or tokens + length > 65536:
                break
            rows.append({
                "handle": row["handle"], "token_ids": row["token_ids"],
                "compression_ratio": chunk.compression_ratio or ratio,
            })
            seen.add(row["handle"])
            tokens += length
            if len(rows) >= min(32, remaining):
                break
        if not rows:
            return None
        job = {"owner_id": self._prewarm_owner_id, "job_id": str(uuid.uuid4()),
               "session_id": session_id, "outer_request_id": outer_request_id,
               "submitted_handles": [row["handle"] for row in rows]}
        payload = {
            "operation": "submit", "owner_id": job["owner_id"],
            "job_id": job["job_id"], "session_id": session_id,
            "chunks": rows, "max_extraction_calls": remaining,
            "outer_request_id": outer_request_id,
        }
        # Keep the identity before the network call: a lost ACK can still be
        # cancelled/drained on session cleanup without submitting another job.
        self._prewarm_job = job
        try:
            ack = self._prewarm_request(payload)
            if ack.get("status") not in {"queued", "completed", "cancelled"}:
                raise SGLangEventNativeError("SGLang prewarm submit was not acknowledged")
            if ack.get("submitted_chunks") != len(rows):
                raise SGLangEventNativeError("SGLang prewarm submit chunk count mismatch")
        except BaseException:
            self._prewarm_budget_unknown = True
            raise
        return {"status": ack["status"], "job_id": job["job_id"],
                "submitted_chunks": len(rows), "submitted_handles": job["submitted_handles"]}

    def reconcile_cross_turn_prewarm(self, *, operation: str = "drain") -> dict[str, Any] | None:
        """Stop the queued tail and account for at most one in-flight extraction."""
        if self._prewarm_budget_unknown and operation != "cancel":
            raise SGLangEventNativeError("cross-turn prewarm extraction budget is unknown")
        if not (self.cross_turn_prewarm_enabled or self.async_compression_enabled) or self._prewarm_job is None:
            return None
        if operation not in ({"drain", "cancel", "poll"} if self.async_compression_enabled
                             else {"drain", "cancel"}):
            raise ValueError("unsupported prewarm reconciliation operation")
        job = self._prewarm_job
        started = time.perf_counter()
        if operation == "poll":
            previous = job.get("last_poll_receipt")
            if (previous is not None
                    and started - job["last_poll_at"] < ASYNC_PREWARM_POLL_INTERVAL_SECONDS):
                self._background_coalesced_polls += 1
                return copy.deepcopy(previous)
            self._background_poll_rpcs += 1
        try:
            receipt = self._prewarm_request({
                "operation": operation, "owner_id": job["owner_id"],
                "job_id": job["job_id"], "session_id": job["session_id"],
                "outer_request_id": job["outer_request_id"],
            })
            self._prewarm_last_receipt = copy.deepcopy(receipt)
            if self.async_compression_enabled and operation == "poll" and receipt.get("status") in {"queued", "running"}:
                job["last_poll_at"] = started
                job["last_poll_receipt"] = copy.deepcopy(receipt)
                return copy.deepcopy(receipt)
            if receipt.get("status") not in {"completed", "cancelled", "failed"}:
                raise SGLangEventNativeError("SGLang prewarm drain did not finish the job")
            if receipt.get("budget_known") is not True:
                self._prewarm_job = None
                raise SGLangEventNativeError("SGLang prewarm final extraction budget is unknown")
            submitted = _nonnegative_int(receipt.get("submitted_chunks"), "prewarm.submitted_chunks")
            completed = _nonnegative_int(receipt.get("completed_chunks"), "prewarm.completed_chunks")
            hits = _nonnegative_int(receipt.get("cache_hits"), "prewarm.cache_hits")
            calls = _nonnegative_int(receipt.get("model_calls"), "prewarm.model_calls")
            cancelled = _nonnegative_int(receipt.get("cancelled_chunks"), "prewarm.cancelled_chunks")
            extraction = receipt.get("extraction")
            if not isinstance(extraction, Mapping):
                raise SGLangEventNativeError("SGLang prewarm extraction receipt is missing")
            if (submitted != len(job["submitted_handles"]) or
                    completed != hits + calls or completed + cancelled != submitted or
                    extraction.get("model_calls") != calls or
                    extraction.get("history_model_calls") != calls or
                    extraction.get("tool_model_calls") != 0):
                raise SGLangEventNativeError("SGLang prewarm extraction accounting mismatch")
            if (calls > self.max_extraction_calls - self.extraction_calls_reserved or
                    (self.async_compression_enabled and calls > len(job["submitted_handles"]))):
                raise SGLangEventNativeError("SGLang prewarm exceeded the finite extraction budget")
            results = receipt.get("results")
            if not isinstance(results, list) or len(results) != completed:
                raise SGLangEventNativeError("SGLang prewarm result count mismatch")
            if (len({row.get("handle") for row in results if isinstance(row, Mapping)}) != completed
                    or any(not isinstance(row, Mapping) or
                           row.get("handle") not in job["submitted_handles"] or
                           type(row.get("cache_hit")) is not bool for row in results)):
                raise SGLangEventNativeError("SGLang prewarm result handles are invalid")
        except BaseException:
            self._prewarm_budget_unknown = True
            raise
        finally:
            if self.async_compression_enabled:
                key = "poll" if operation == "poll" else "final_drain"
                self._background_sync_wall[key] += time.perf_counter() - started
        self.extraction_calls_reserved += calls
        if self.async_compression_enabled:
            if not job.get("acknowledged", True):
                self._background_submitted_chunks += submitted
            self._background_settled_chunks += submitted
            self._background_model_calls += calls
            self._record_materialized_history(job["session_id"], (
                row["handle"] for row in results
            ))
        self._prewarm_budget_unknown = False
        self._prewarm_last_receipt = copy.deepcopy(receipt)
        self._prewarm_job = None
        return copy.deepcopy(receipt)

    def _ensure_model_info(self) -> None:
        if self._model_binding is not None:
            return
        request = Request(
            self.upstream + "/model_info",
            headers={"Accept": "application/json"},
            method="GET",
        )
        response, status = self._read_json(request, label="SGLang /model_info")
        if status != 200:
            raise SGLangEventNativeError(f"SGLang /model_info returned HTTP {status}")
        if not isinstance(response, Mapping):
            raise SGLangEventNativeError("SGLang /model_info must return a JSON object")
        native = response.get("c2kv_native_packed")
        if not isinstance(native, Mapping):
            raise SGLangEventNativeError(
                "SGLang /model_info lacks c2kv_native_packed admission data"
            )
        if self.async_compression_enabled:
            features = native.get("serving_features")
            if (not isinstance(features, Mapping) or
                    features.get("async_compression") != "nonblocking-history-v1"):
                raise SGLangEventNativeError("Engine lacks nonblocking-history-v1 admission")
        elif self.cross_turn_prewarm_enabled:
            features = native.get("serving_features")
            if (not isinstance(features, Mapping)
                    or features.get("cross_turn_prewarm") != "cross-turn-prewarm-v1"):
                raise SGLangEventNativeError("Engine lacks cross-turn-prewarm-v1 admission")
        if self.sampling_profile != "greedy-v1" and self.sampling_profile not in native.get("sampling_profiles", []):
            raise SGLangEventNativeError("Engine does not support the requested native sampling profile")
        binding = _json_object(native.get("model_binding"), "model_binding")
        if not binding:
            raise SGLangEventNativeError("model_binding must not be empty")
        if binding.get("model_path") != self.expected_model_path:
            raise SGLangEventNativeError(
                "SGLang model_path does not match expected_model_path"
            )
        if response.get("model_path") != self.expected_model_path:
            raise SGLangEventNativeError(
                "SGLang top-level model_path does not match expected_model_path"
            )
        if self.expected_tool_checkpoint_contract is not None:
            tool = native.get("tool_gist")
            expected = self.expected_tool_checkpoint_contract
            if not isinstance(tool, Mapping) or tool.get("enabled") is not True:
                raise SGLangEventNativeError("SGLang tool gist projection is not enabled")
            if tool.get("extract_projection_set") != "tool":
                raise SGLangEventNativeError("SGLang tool gist projection set differs")
            if tool.get("config_sha256") != expected.get("config_sha256"):
                raise SGLangEventNativeError("SGLang tool checkpoint config hash differs")
            source = tool.get("source")
            if not isinstance(source, str) or Path(source).resolve() != Path(expected["checkpoint"]).resolve():
                raise SGLangEventNativeError("SGLang tool checkpoint source differs")
            identity = tool.get("identity")
            if not isinstance(identity, str) or not identity:
                raise SGLangEventNativeError("SGLang tool gist projection identity is missing")
            self._tool_projection_identity = identity
        kv_bytes = _nonnegative_int(
            native.get("kv_bytes_per_token"), "kv_bytes_per_token"
        )
        if kv_bytes <= 0:
            raise SGLangEventNativeError("kv_bytes_per_token must be positive")
        self._model_binding = binding
        self._kv_bytes_per_token = kv_bytes

    def repair_tool_span(self, input_ids: Sequence[int], *, span_start: int,
                         span_end: int, method: str, target_tokens: int,
                         selectable_relative_indices: Sequence[int] | None = None) -> dict[str, Any]:
        """Extract one query-conditioned raw tool region with a separate cap."""
        allowed = ({"h2o", "snapkv"} if selectable_relative_indices is None else
                   {"streamingllm", "h2o", "snapkv", "pyramidkv"})
        if method not in allowed:
            raise ValueError("Unsupported raw tool repair method")
        ids = _token_ids(input_ids, "input_ids", nonempty=True)
        if (type(span_start) is not int or type(span_end) is not int
                or not 0 <= span_start < span_end <= len(ids)):
            raise ValueError("Tool span must be inside the exact logical prompt")
        _positive_int(target_tokens, "target_tokens")
        if target_tokens > span_end - span_start:
            raise ValueError("Tool target cannot exceed its raw source")
        if self.max_tool_repair_calls is not None and self.tool_repair_calls >= self.max_tool_repair_calls:
            raise SGLangEventNativeError("Finite tool repair-call cap is exhausted")
        self._ensure_model_info()
        body = {
            "input_ids": list(ids), "span_start": span_start, "span_end": span_end,
            "position_offset": 0, "raw_kv_position_mode": "rotated",
            "repair_mode": "history_kv_" + method,
            "history_kv_method": method,
            "history_kv_target_tokens": target_tokens,
            "history_kv_recent_window": 64,
            "history_kv_kernel_size": 5,
            "history_kv_pooling": "avgpool",
            "history_kv_h2o_recent_fraction": 0.5,
        }
        if selectable_relative_indices is not None:
            indices = list(selectable_relative_indices)
            if (any(type(index) is not int or not 0 <= index < span_end - span_start
                    for index in indices) or indices != sorted(set(indices))):
                raise ValueError("Tool selectable indices must be unique sorted offsets")
            if target_tokens < span_end - span_start - len(indices):
                raise ValueError("Tool target cannot evict protected interface tokens")
            body["history_kv_selectable_relative_indices"] = indices
            body["history_kv_pooling"] = "maxpool" if method == "snapkv" else "avgpool"
            body["history_kv_recent_window"] = 64 if method == "pyramidkv" else 16
            body["history_kv_kernel_size"] = 5 if method == "pyramidkv" else 7
        self.tool_repair_calls += 1
        response, status = self._read_json(Request(
            self.upstream + "/v1/c2kv/repair_extract",
            data=_canonical_bytes(body),
            headers={"Content-Type": "application/json"}, method="POST"),
            label="SGLang tool repair_extract")
        if status != 200 or not isinstance(response, Mapping) or response.get("success") is not True:
            raise SGLangEventNativeError(f"Tool repair_extract failed: HTTP {status}")
        key_hash = response.get("key_hash")
        token_len = response.get("token_len")
        if not isinstance(key_hash, str) or not key_hash or type(token_len) is not int:
            raise SGLangEventNativeError("Tool repair_extract lacks a retained KV handle")
        physical_limit = span_end - span_start if method == "pyramidkv" else target_tokens
        if not 0 < token_len <= physical_limit:
            raise SGLangEventNativeError("Tool repair_extract exceeded the retained target")
        if response.get("original_seq_len") != len(ids):
            raise SGLangEventNativeError("Tool repair_extract source length mismatch")
        if (response.get("span_start"), response.get("span_end")) != (span_start, span_end):
            raise SGLangEventNativeError("Tool repair_extract span mismatch")
        if response.get("history_kv_method") not in {method, "snapkv_persistent" if method == "snapkv" else method}:
            raise SGLangEventNativeError("Tool repair_extract method mismatch")
        return dict(response)

    def _prepare_chunks(
        self,
        memory: PackedMemory,
        ratio: int,
        compression_chunks: Sequence[EncoderChunk] | None,
        *, background_history_budget_tokens: int | None = None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        placements = memory.gist_layout(ratio)
        selected = [
            self._chunk_payload(
                placement.chunk,
                ratio,
                source_position_start=placement.source_position_start,
                gist_position_ids=placement.position_ids,
            )
            for placement in placements
        ]
        anchored = []
        for segment in memory.tool_gist_segments:
            start = segment["token_start"]
            end = segment["token_end"]
            cursor = start
            rows = []
            for chunk in segment["chunks"]:
                chunk_ratio = chunk.compression_ratio or ratio
                length = len(chunk.token_ids)
                positions = tuple(cursor + min(offset + chunk_ratio, length) - 1
                                  for offset in range(0, length, chunk_ratio))
                rows.append(self._chunk_payload(
                    chunk, ratio, source_position_start=cursor,
                    gist_position_ids=positions))
                cursor += length
            anchored.append({"token_start": start, "token_end": end,
                             "chunks": rows})
            selected.extend(rows)
        selected_handles = {item["handle"] for item in selected}
        if len(selected_handles) != len(selected):
            raise ValueError("memory contains duplicate selected encoder chunks")

        extras: list[dict[str, Any]] = []
        if compression_chunks is not None:
            if not isinstance(compression_chunks, Sequence) or isinstance(
                compression_chunks, (str, bytes, bytearray)
            ):
                raise TypeError("compression_chunks must be a sequence or None")
            all_eligible: dict[str, dict[str, Any]] = {}
            for chunk in compression_chunks:
                item = self._chunk_payload(chunk, ratio)
                previous = all_eligible.setdefault(item["handle"], item)
                if previous != item:
                    raise ValueError("stable chunk handle collision")
            if not selected_handles <= set(all_eligible):
                raise ValueError(
                    "eligible compression chunks must include every retained gist chunk"
                )
            admitted = self._budget_fitting_background_rows(
                list(all_eligible.values()), ratio=ratio,
                history_budget_tokens=background_history_budget_tokens)
            extras = [
                item
                for item in admitted
                if item["handle"] not in selected_handles
            ]
        return selected, extras, anchored

    def _chunk_payload(
        self,
        chunk: EncoderChunk,
        ratio: int,
        *,
        source_position_start: int | None = None,
        gist_position_ids: Sequence[int] | None = None,
    ) -> dict[str, Any]:
        if not isinstance(chunk, EncoderChunk):
            raise TypeError("encoder chunks must be EncoderChunk instances")
        token_ids = _token_ids(chunk.token_ids, "chunk.token_ids", nonempty=True)
        descriptor = {
            "event_id": chunk.event_id,
            "part_index": chunk.part_index,
            "source_indices": list(chunk.source_indices),
            "source_token_start": chunk.source_token_start,
            "source_token_end": chunk.source_token_end,
            "token_ids": list(token_ids),
        }
        chunk_ratio = chunk.compression_ratio or ratio
        if type(chunk_ratio) is not int or chunk_ratio <= 0:
            raise ValueError("chunk.compression_ratio must be positive")
        if chunk.projection_set is not None:
            if chunk.projection_set not in {"tool", "history"}:
                raise ValueError("chunk.projection_set is unsupported")
            descriptor["projection_set"] = chunk.projection_set
        if chunk.compression_ratio is not None:
            descriptor["compression_ratio"] = chunk_ratio
        if not isinstance(chunk.event_id, str) or not chunk.event_id:
            raise ValueError("chunk.event_id must be a nonempty string")
        for name in ("part_index", "source_token_start", "source_token_end"):
            if type(descriptor[name]) is not int or descriptor[name] < 0:
                raise ValueError(f"chunk.{name} must be a nonnegative integer")
        _token_ids(chunk.source_indices, "chunk.source_indices", nonempty=True)
        if chunk.source_token_end <= chunk.source_token_start:
            raise ValueError("chunk source token range must be nonempty")

        chunk_id = hashlib.sha256(_canonical_bytes(descriptor)).hexdigest()
        canonical_chunk = {"chunk_id": chunk_id, **descriptor}
        assert self._model_binding is not None
        handle_input = {
            "schema": HANDLE_SCHEMA,
            "model_binding": self._model_binding,
            "packing_version": PACKING_VERSION,
            "encoding_scope": self.encoding_scope,
            "compression_ratio": chunk_ratio,
            "chunk": canonical_chunk,
        }
        if chunk.projection_set == "tool":
            if self._tool_projection_identity is None:
                raise SGLangEventNativeError("SGLang tool gist projection identity is missing")
            handle_input["projection_identity"] = self._tool_projection_identity
        handle = hashlib.sha256(_canonical_bytes(handle_input)).hexdigest()
        result = {**canonical_chunk, "handle": handle}
        if source_position_start is not None:
            if type(source_position_start) is not int or source_position_start < 0:
                raise ValueError("source_position_start must be nonnegative")
            positions = _token_ids(
                gist_position_ids, "gist_position_ids", nonempty=True
            )
            result.update(
                source_position_start=source_position_start,
                gist_position_ids=list(positions),
            )
        return result

    def _post_native_generate(
        self,
        payload: Mapping[str, Any],
        request_index: int,
        *,
        measurement_phase: str | None = None,
    ) -> tuple[Any, int]:
        body = _canonical_bytes(payload)
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        outer_request_id = payload.get("outer_request_id")
        if isinstance(outer_request_id, str) and outer_request_id:
            headers["X-C2KV-Measurement-Request-Id"] = outer_request_id
        if isinstance(measurement_phase, str) and measurement_phase:
            headers["X-C2KV-Measurement-Phase"] = measurement_phase
        request = Request(
            self.upstream + "/v1/c2kv/native_generate",
            data=body,
            headers=headers,
            method="POST",
        )
        if self._http_journal is not None:
            self._http_journal.append(
                {
                    "schema": HTTP_JOURNAL_SCHEMA,
                    "event": "request",
                    "status": "started",
                    "request_index": request_index,
                    "timestamp_unix_ns": time.time_ns(),
                    "request": payload,
                    "retries": 0,
                }
            )
        response: Any = None
        status: int | None = None
        try:
            response, status = self._read_json(
                request, label="SGLang /v1/c2kv/native_generate"
            )
            if status != 200:
                error = response.get("error") if isinstance(response, Mapping) else None
                if (
                    status == 400
                    and isinstance(error, Mapping)
                    and error.get("code") == EXTRACTION_BUDGET_ERROR_CODE
                ):
                    receipt = self._validate_extraction_budget_failure(response, payload)
                    model_calls = receipt["model_calls"]
                    if self.extraction_calls_reserved + model_calls > self.max_extraction_calls:
                        raise SGLangEventNativeError(
                            "extraction-budget failure receipt exceeds the task cap"
                        )
                    self.extraction_calls_reserved += model_calls
                    raise SGLangExtractionBudgetExhausted(
                        error["message"], receipt
                    )
                message = (
                    error.get("message") if isinstance(error, Mapping) else error
                )
                suffix = f": {message}" if isinstance(message, str) and message else ""
                error_type = SGLangTransportError if status in {502, 503, 504} else SGLangEventNativeError
                raise error_type(
                    f"SGLang native_generate returned HTTP {status}{suffix}"
                )
            return response, status
        finally:
            if self._http_journal is not None:
                self._http_journal.append(
                    {
                        "schema": HTTP_JOURNAL_SCHEMA,
                        "event": "response",
                        "status": "completed" if status == 200 else "failed",
                        "request_index": request_index,
                        "timestamp_unix_ns": time.time_ns(),
                        "http_status": status,
                        "response": response,
                        "retries": 0,
                    }
                )

    def _read_json(self, request: Request, *, label: str,
                   allow_empty: bool = False) -> tuple[Any, int]:
        try:
            with self._opener.open(request, timeout=self.timeout_seconds) as opened:
                status_value = getattr(opened, "status", None)
                status = int(opened.getcode() if status_value is None else status_value)
                raw = opened.read(self.max_response_bytes + 1)
        except HTTPError as error:
            status = int(error.code)
            try:
                raw = error.read(self.max_response_bytes + 1)
            except OSError:
                raw = b""
        except (OSError, TimeoutError) as error:
            raise SGLangTransportError(f"{label} transport failed without retry") from error
        if len(raw) > self.max_response_bytes:
            raise SGLangEventNativeError(f"{label} response exceeds the byte cap")
        if allow_empty and not raw.strip():
            return None, status
        try:
            return json.loads(raw.decode("utf-8")), status
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise SGLangEventNativeError(f"{label} did not return valid UTF-8 JSON") from error

    def _charge_extractions(self, response: Any) -> None:
        if not isinstance(response, Mapping):
            return
        extraction = response.get("extraction")
        if not isinstance(extraction, Mapping):
            return
        if self.max_tool_extraction_calls is None:
            value = extraction.get("model_calls")
            if type(value) is int and value >= 0:
                self.extraction_calls_reserved += value
            return
        history = extraction.get("history_model_calls")
        tool = extraction.get("tool_model_calls")
        if type(history) is int and history >= 0 and type(tool) is int and tool >= 0:
            self.extraction_calls_reserved += history
            self.tool_extraction_calls_reserved += tool

    def _validate_extraction_budget_failure(
        self, response: Any, payload: Mapping[str, Any]
    ) -> dict[str, Any]:
        if not isinstance(response, Mapping):
            raise SGLangEventNativeError(
                "extraction-budget failure response must be an object"
            )
        error = response.get("error")
        if not isinstance(error, Mapping):
            raise SGLangEventNativeError(
                "extraction-budget failure error must be an object"
            )
        if error.get("code") != EXTRACTION_BUDGET_ERROR_CODE:
            raise SGLangEventNativeError("extraction-budget failure code mismatch")
        message = error.get("message")
        if not isinstance(message, str) or not message.startswith(
            EXTRACTION_BUDGET_ERROR_CODE + ":"
        ):
            raise SGLangEventNativeError(
                "extraction-budget failure message mismatch"
            )
        value = response.get("extraction")
        if not isinstance(value, Mapping):
            raise SGLangEventNativeError(
                "extraction-budget failure receipt must be an object"
            )
        if value.get("schema") != EXTRACTION_FAILURE_SCHEMA:
            raise SGLangEventNativeError(
                "extraction-budget failure receipt schema mismatch"
            )
        names = (
            "requested_chunks",
            "unique_chunks",
            "processed_chunks",
            "cache_hits",
            "cache_misses",
            "model_calls",
            "max_extraction_calls",
            "failed_chunk_index",
        )
        result = {
            name: _nonnegative_int(
                value.get(name), f"response.extraction.{name}"
            )
            for name in names
        }
        rows = [*payload["encoder_chunks"],
                *(chunk for segment in payload.get("tool_gist_segments", ())
                  for chunk in segment["chunks"]),
                *payload["compression_chunks"]]
        handles = {row["handle"] for row in rows}
        if result["requested_chunks"] != len(rows):
            raise SGLangEventNativeError(
                "extraction-budget failure requested_chunks mismatch"
            )
        if result["unique_chunks"] != len(handles):
            raise SGLangEventNativeError(
                "extraction-budget failure unique_chunks mismatch"
            )
        if result["processed_chunks"] != (
            result["cache_hits"] + result["cache_misses"]
        ):
            raise SGLangEventNativeError(
                "extraction-budget failure processed count mismatch"
            )
        if result["failed_chunk_index"] != result["processed_chunks"]:
            raise SGLangEventNativeError(
                "extraction-budget failure index mismatch"
            )
        if result["processed_chunks"] >= result["unique_chunks"]:
            raise SGLangEventNativeError(
                "extraction-budget failure does not identify an unprocessed miss"
            )
        if result["model_calls"] != result["cache_misses"]:
            raise SGLangEventNativeError(
                "extraction-budget failure model_calls/cache_misses disagree"
            )
        if result["max_extraction_calls"] != payload["max_extraction_calls"]:
            raise SGLangEventNativeError(
                "extraction-budget failure max_extraction_calls mismatch"
            )
        if result["model_calls"] != result["max_extraction_calls"]:
            raise SGLangEventNativeError(
                "extraction-budget failure did not consume the request budget"
            )
        return {"schema": EXTRACTION_FAILURE_SCHEMA, **result}

    def _result_from_response(
        self,
        response: Any,
        *,
        payload: Mapping[str, Any],
        memory: PackedMemory,
        selected: Sequence[Mapping[str, Any]],
        extras: Sequence[Mapping[str, Any]],
        ratio: int,
        requested_tokens: int,
        request_index: int,
        http_status: int,
        wall_seconds: float,
        scope: _DecisionScope | None,
        effective_eos_ids: tuple[int, ...],
    ) -> SGLangEventNativeGenerationResult:
        if not isinstance(response, Mapping):
            raise SGLangEventNativeError("native_generate response must be a JSON object")
        for name, expected in (
            ("schema", RESPONSE_SCHEMA),
            ("rid", payload["rid"]),
            ("session_id", payload["session_id"]),
            ("generation_id", payload["generation_id"]),
        ):
            if response.get(name) != expected:
                raise SGLangEventNativeError(f"native_generate response {name} mismatch")
        if ("outer_request_id" in response
                and response.get("outer_request_id") != payload.get("outer_request_id")):
            raise SGLangEventNativeError(
                "native_generate response outer_request_id mismatch"
            )
        if ("native_request_id" in response
                and response.get("native_request_id") != payload["rid"]):
            raise SGLangEventNativeError(
                "native_generate response native_request_id mismatch"
            )
        if not isinstance(response.get("text"), str):
            raise SGLangEventNativeError("native_generate response lacks text")
        output_ids = _token_ids(response.get("output_ids"), "response.output_ids")
        raw_logprobs = response.get("token_logprobs")
        if not isinstance(raw_logprobs, Sequence) or isinstance(
            raw_logprobs, (str, bytes, bytearray)
        ):
            raise SGLangEventNativeError("response.token_logprobs must be a sequence")
        if len(raw_logprobs) != len(output_ids):
            raise SGLangEventNativeError("response token/logprob counts disagree")
        logprobs = []
        for value in raw_logprobs:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise SGLangEventNativeError("response token_logprobs must be numeric")
            value = float(value)
            if not math.isfinite(value):
                raise SGLangEventNativeError("response token_logprobs must be finite")
            logprobs.append(value)
        finish_reason = response.get("finish_reason")
        if isinstance(finish_reason, Mapping):
            finish_reason = finish_reason.get("type")
        if not isinstance(finish_reason, str) or not finish_reason:
            raise SGLangEventNativeError("response.finish_reason must be nonempty")
        if len(output_ids) > requested_tokens:
            raise SGLangEventNativeError("response exceeds requested max_new_tokens")

        selected_rows = self._validate_chunk_results(
            response.get("encoder_chunks"), selected, "encoder_chunks", ratio
        )
        extra_rows = self._validate_chunk_results(
            response.get("compression_chunks"), extras, "compression_chunks", ratio
        )
        extraction = self._validate_extraction(
            response.get("extraction"),
            selected_rows + extra_rows,
            expected_max_extraction_calls=payload["max_extraction_calls"],
            expected_max_tool_extraction_calls=payload.get("max_tool_extraction_calls"),
        )
        costs = self._validate_costs(response.get("costs"), memory, ratio)
        shadow = self._validate_shadow_features(
            response.get("shadow_features"), memory
        )
        allocator = response.get("allocator")
        if allocator is not None:
            allocator = _json_object(allocator, "response.allocator")
        paper_measurement = response.get("paper_measurement")
        if paper_measurement is not None:
            paper_measurement = _json_object(
                paper_measurement, "response.paper_measurement"
            )
        native_telemetry = response.get("telemetry")
        if native_telemetry is not None:
            native_telemetry = _json_object(native_telemetry, "response.telemetry")
        native_request_ids = response.get("request_ids")
        if native_request_ids is not None:
            native_request_ids = _json_object(
                native_request_ids, "response.request_ids"
            )

        before_scope = set(scope.retained_handles) if scope is not None else set()
        before_session = set(scope.session_handles) if scope is not None else set()
        selected_by_handle = {row["handle"]: row for row in selected_rows}
        extras_by_handle = {row["handle"]: row for row in extra_rows}
        scope_reused = {
            handle for handle, row in selected_by_handle.items()
            if row["cache_hit"] and handle in before_scope
        }
        session_reused = {
            handle for handle, row in selected_by_handle.items()
            if row["cache_hit"] and handle in before_session and handle not in before_scope
        }
        unretained_reused = sum(row["cache_hit"] for row in extras_by_handle.values())
        kv_bytes = self.kv_bytes_per_token()
        prefix_tokens = costs["system_prefix_kv_tokens"] + costs["gist_prefix_kv_tokens"]
        prompt_resident_tokens = costs["resident_kv_tokens"]
        final_resident_tokens = prompt_resident_tokens + max(0, len(output_ids) - 1)
        generation_index = scope.generate_calls if scope is not None else 1

        stats = {
            "schema": GENERATION_SCHEMA,
            "backend": "sglang_c2kv_native_packed",
            "raw_layout_profile": memory.raw_layout_profile,
            "packing_version": PACKING_VERSION,
            "encoding_scope": self.encoding_scope,
            "ratio": ratio,
            "requested_max_new_tokens": requested_tokens,
            "generated_tokens": len(output_ids),
            "workspace_tokens": len(memory.workspace_input_ids),
            "source_tokens": (
                len(memory.system_input_ids)
                + sum(len(chunk.token_ids) for chunk in memory.chunks)
                + len(memory.workspace_input_ids)
                + sum(sum(len(chunk.token_ids) for chunk in segment["chunks"])
                      - (segment["token_end"] - segment["token_start"])
                      for segment in memory.tool_gist_segments)
            ),
            "packed_encoder_tokens": costs["presented_encoder_tokens"],
            "materialized_encoder_tokens": extraction["materialized_encoder_tokens"],
            "scope_reused_encoder_tokens": sum(
                len(selected_by_handle[handle]["token_ids"])
                for handle in scope_reused
            ),
            "gist_tokens": costs["gist_tokens"],
            "system_prefix_kv_tokens": costs["system_prefix_kv_tokens"],
            "gist_prefix_kv_tokens": costs["gist_prefix_kv_tokens"],
            "system_prefix_kv_logical_bytes": costs["system_prefix_kv_logical_bytes"],
            "gist_prefix_kv_logical_bytes": costs["gist_prefix_kv_logical_bytes"],
            "resident_prefix_kv_tokens": prefix_tokens,
            "resident_prefix_kv_bytes": prefix_tokens * kv_bytes,
            "kv_bytes_per_token": kv_bytes,
            "unique_chunks": len(selected_by_handle),
            "chunk_placements": len(selected_rows),
            "max_extraction_calls": self.max_extraction_calls,
            "extraction_calls_reserved": self.extraction_calls_reserved,
            "extracted_chunks": extraction["model_calls"],
            "unretained_extracted_chunks": sum(
                not row["cache_hit"] for row in extras_by_handle.values()
            ),
            "unretained_reused_chunks": unretained_reused,
            "scope_reused_chunks": len(scope_reused),
            "reused_chunk_placements": len(selected_rows) - len(selected_by_handle),
            "scope_reused_chunk_placements": len(scope_reused),
            # The engine contract does not expose exact radix-forward work.
            "system_prefill_calls": None,
            "system_prefill_tokens": None,
            "scope_reused_system_prefill_calls": None,
            "scope_reused_system_prefill_tokens": None,
            "decision_scope_active": scope is not None,
            "decision_scope_generation_index": generation_index,
            "scope_system_kv_logical_bytes_before": 0,
            "scope_system_kv_logical_bytes_after": 0,
            "scope_gist_kv_logical_bytes_before": 0,
            "scope_gist_kv_logical_bytes_after": 0,
            "decode_strategy": "external_sglang_incremental",
            "prefill_chunk_size": None,
            "target_execution_strategy": "external_sglang_incremental",
            "target_forward_calls": None,
            "target_input_tokens": None,
            "recomputed_raw_tokens": None,
            "raw_prefill_forward_calls": None,
            "raw_prefill_input_tokens": None,
            "one_token_decode_forward_calls": None,
            "one_token_decode_input_tokens": None,
            "raw_recompute_forward_calls": 0,
            "raw_recompute_input_tokens": 0,
            "suffix_recompute_tokens": 0,
            "resident_kv_tokens_after_raw_prefill": prompt_resident_tokens,
            "resident_kv_logical_bytes_after_raw_prefill": costs[
                "resident_kv_logical_bytes"
            ],
            "resident_kv_tokens_final": final_resident_tokens,
            "resident_kv_logical_bytes_final": final_resident_tokens * kv_bytes,
            "torch_allocator_peak_allocated_bytes": None,
            "server_kv_memory_report": allocator,
            "session_cache_policy": self.session_cache_policy,
            "session_scope_active": scope is not None and scope.session_id is not None,
            "session_reused_raw_tokens": 0,
            "session_reused_gist_chunks": len(session_reused),
            "session_reused_encoder_tokens": sum(
                len(selected_by_handle[handle]["token_ids"])
                for handle in session_reused
            ),
            "session_reused_system_tokens": 0,
            "session_cpu_memo_gist_hits": len(session_reused),
            "session_cpu_memo_system_hit": False,
            "session_cache_cpu_logical_bytes_before": 0,
            "session_pending_device_resident_logical_bytes": 0,
            "session_pending_device_backing_bytes": 0,
            "session_cache_transfer_bytes_in": 0,
            "raw_cache_reuse_eligibility_reason": "external_server_session_disabled",
            "session_cache_reset_reason": scope.reset_reason if scope is not None else None,
            "session_cache_eviction_reason": (
                "same_decision_regeneration"
                if scope is not None and generation_index > 1
                else None
            ),
            "session_cache_commit_status": (
                "pending"
                if scope is not None and scope.session_id is not None
                else "not_applicable"
            ),
            "raw_recompute_strategy": "unknown_external_sglang_radix_execution",
            "parameter_version": copy.deepcopy(self._model_binding),
            "eos_token_ids": list(
                effective_eos_ids
            ),
            "eos_source": self.eos_source,
            "elapsed_sec": wall_seconds,
            "sglang_transport": {
                "schema": HTTP_JOURNAL_SCHEMA,
                "url": self.upstream + "/v1/c2kv/native_generate",
                "http_status": http_status,
                "request_index": request_index,
                "wall_seconds": wall_seconds,
                "timeout_seconds": self.timeout_seconds,
                "retries": 0,
                "rid": payload["rid"],
                "generation_id": payload["generation_id"],
                "outer_request_id": payload.get("outer_request_id"),
                "native_request_id": response.get("native_request_id", payload["rid"]),
            },
            "outer_request_id": payload.get("outer_request_id"),
            "native_request_id": response.get("native_request_id", payload["rid"]),
            "paper_measurement": paper_measurement,
            "native_telemetry": native_telemetry,
            "native_request_ids": native_request_ids,
            "sglang_extraction": extraction,
            "sglang_chunk_handles": {
                "selected": [row["handle"] for row in selected_rows],
                "compression_only": [row["handle"] for row in extra_rows],
                "server_cache_keys": [
                    row["cache_key"] for row in selected_rows + extra_rows
                ],
            },
        }
        if shadow is not None:
            stats["shadow_features"] = shadow
        execution = response.get("serving_execution")
        if execution is not None:
            stats["native_serving_execution"] = _json_object(execution, "response.serving_execution")
        runtime = response.get("sglang_runtime")
        if isinstance(runtime, Mapping) and runtime.get("c2kv_raw_prefix_cache") is not None:
            stats["native_raw_prefix_cache"] = _json_object(
                runtime["c2kv_raw_prefix_cache"], "response.sglang_runtime.c2kv_raw_prefix_cache")
        return SGLangEventNativeGenerationResult(
            token_ids=output_ids,
            finish_reason=finish_reason,
            token_logprobs=tuple(logprobs),
            stats=stats,
        )

    def _validate_chunk_results(
        self,
        value: Any,
        requested: Sequence[Mapping[str, Any]],
        name: str,
        ratio: int,
    ) -> list[dict[str, Any]]:
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
            raise SGLangEventNativeError(f"response.{name} must be a sequence")
        rows = list(value)
        if len(rows) != len(requested):
            raise SGLangEventNativeError(f"response.{name} count mismatch")
        validated = []
        for index, (raw, expected) in enumerate(zip(rows, requested, strict=True)):
            if not isinstance(raw, Mapping):
                raise SGLangEventNativeError(f"response.{name}[{index}] must be an object")
            for field_name in ("chunk_id", "handle"):
                if raw.get(field_name) != expected[field_name]:
                    raise SGLangEventNativeError(
                        f"response.{name}[{index}].{field_name} mismatch"
                    )
            cache_key = raw.get("cache_key")
            if not isinstance(cache_key, str) or not cache_key:
                raise SGLangEventNativeError(
                    f"response.{name}[{index}].cache_key must be nonempty"
                )
            if type(raw.get("cache_hit")) is not bool:
                raise SGLangEventNativeError(
                    f"response.{name}[{index}].cache_hit must be bool"
                )
            original = _nonnegative_int(
                raw.get("original_seq_len"),
                f"response.{name}[{index}].original_seq_len",
            )
            gist_len = _nonnegative_int(
                raw.get("gist_len"), f"response.{name}[{index}].gist_len"
            )
            if original != len(expected["token_ids"]):
                raise SGLangEventNativeError(
                    f"response.{name}[{index}] original_seq_len mismatch"
                )
            chunk_ratio = expected.get("compression_ratio", ratio)
            if gist_len != (original + chunk_ratio - 1) // chunk_ratio:
                raise SGLangEventNativeError(
                    f"response.{name}[{index}] gist_len mismatch"
                )
            validated.append(
                {
                    **dict(raw),
                    "token_ids": tuple(expected["token_ids"]),
                    "original_seq_len": original,
                    "gist_len": gist_len,
                }
            )
        return validated

    def _validate_extraction(
        self,
        value: Any,
        rows: Sequence[Mapping[str, Any]],
        *,
        expected_max_extraction_calls: int,
        expected_max_tool_extraction_calls: int | None = None,
    ) -> dict[str, Any]:
        if not isinstance(value, Mapping):
            raise SGLangEventNativeError("response.extraction must be an object")
        result = {
            name: _nonnegative_int(value.get(name), f"response.extraction.{name}")
            for name in (
                "requested_chunks",
                "unique_chunks",
                "cache_hits",
                "cache_misses",
                "model_calls",
            )
        }
        result["max_extraction_calls"] = _optional_nonnegative_int(
            value.get("max_extraction_calls"),
            "response.extraction.max_extraction_calls",
        )
        if result["max_extraction_calls"] != expected_max_extraction_calls:
            raise SGLangEventNativeError(
                "response.extraction.max_extraction_calls mismatch"
            )
        if expected_max_tool_extraction_calls is not None:
            tool_limit = _optional_nonnegative_int(
                value.get("max_tool_extraction_calls"),
                "response.extraction.max_tool_extraction_calls")
            if tool_limit != expected_max_tool_extraction_calls:
                raise SGLangEventNativeError("Tool extraction cap echo mismatch")
            history_calls = _nonnegative_int(
                value.get("history_model_calls"),
                "response.extraction.history_model_calls")
            tool_calls = _nonnegative_int(
                value.get("tool_model_calls"),
                "response.extraction.tool_model_calls")
            if history_calls + tool_calls != result["model_calls"]:
                raise SGLangEventNativeError("Tool/history extraction accounting mismatch")
            if (self.extraction_calls_reserved > self.max_extraction_calls or
                    self.tool_extraction_calls_reserved > self.max_tool_extraction_calls):
                raise SGLangEventNativeError("Separate extraction budget exceeded")
            result.update(history_model_calls=history_calls,
                          tool_model_calls=tool_calls,
                          max_tool_extraction_calls=tool_limit)
        handles = {row["handle"] for row in rows}
        if result["requested_chunks"] != len(rows):
            raise SGLangEventNativeError("response.extraction.requested_chunks mismatch")
        if result["unique_chunks"] != len(handles):
            raise SGLangEventNativeError("response.extraction.unique_chunks mismatch")
        if result["cache_hits"] + result["cache_misses"] != len(handles):
            raise SGLangEventNativeError("response extraction hit/miss counts disagree")
        if result["model_calls"] != result["cache_misses"]:
            raise SGLangEventNativeError("response extraction model_calls/cache_misses disagree")
        row_hits: dict[str, bool] = {}
        for row in rows:
            previous = row_hits.setdefault(row["handle"], row["cache_hit"])
            if previous != row["cache_hit"]:
                raise SGLangEventNativeError("duplicate handle has inconsistent cache_hit")
        if sum(row_hits.values()) != result["cache_hits"]:
            raise SGLangEventNativeError("response extraction cache_hits mismatch")
        if self.extraction_calls_reserved > self.max_extraction_calls:
            raise SGLangEventNativeError("server exceeded finite extraction-call budget")
        result["materialized_encoder_tokens"] = sum(
            len(row["token_ids"])
            for row in {row["handle"]: row for row in rows}.values()
            if not row["cache_hit"]
        )
        return result

    def _validate_costs(
        self, value: Any, memory: PackedMemory, ratio: int
    ) -> dict[str, int]:
        if not isinstance(value, Mapping):
            raise SGLangEventNativeError("response.costs must be an object")
        names = (
            "system_tokens",
            "raw_tokens",
            "presented_encoder_tokens",
            "gist_tokens",
            "resident_kv_tokens",
            "system_prefix_kv_logical_bytes",
            "gist_prefix_kv_logical_bytes",
            "raw_workspace_kv_logical_bytes",
            "resident_kv_logical_bytes",
        )
        result = {
            name: _nonnegative_int(value.get(name), f"response.costs.{name}")
            for name in names
        }
        expected = memory.costs(ratio)
        for name in (
            "system_tokens",
            "raw_tokens",
            "presented_encoder_tokens",
            "gist_tokens",
            "resident_kv_tokens",
        ):
            if result[name] != expected[name]:
                raise SGLangEventNativeError(f"response.costs.{name} mismatch")
        kv_bytes = self.kv_bytes_per_token()
        byte_expectations = {
            "system_prefix_kv_logical_bytes": expected["system_tokens"] * kv_bytes,
            "gist_prefix_kv_logical_bytes": expected["gist_tokens"] * kv_bytes,
            "raw_workspace_kv_logical_bytes": expected["raw_tokens"] * kv_bytes,
            "resident_kv_logical_bytes": expected["resident_kv_tokens"] * kv_bytes,
        }
        if memory.raw_tool_segments or memory.tool_gist_segments:
            for name in ("system_prefix_kv_tokens", "gist_prefix_kv_tokens",
                         "workspace_resident_kv_tokens"):
                result[name] = _nonnegative_int(
                    value.get(name), f"response.costs.{name}")
            if result["gist_prefix_kv_tokens"] != expected["gist_tokens"]:
                raise SGLangEventNativeError(
                    "response.costs.gist_prefix_kv_tokens mismatch")
            if sum(result[name] for name in (
                    "system_prefix_kv_tokens", "gist_prefix_kv_tokens",
                    "workspace_resident_kv_tokens")) != expected["resident_kv_tokens"]:
                raise SGLangEventNativeError(
                    "response.costs physical KV components do not equal resident_kv_tokens")
            byte_expectations = {
                "system_prefix_kv_logical_bytes": result["system_prefix_kv_tokens"] * kv_bytes,
                "gist_prefix_kv_logical_bytes": result["gist_prefix_kv_tokens"] * kv_bytes,
                "raw_workspace_kv_logical_bytes": result["workspace_resident_kv_tokens"] * kv_bytes,
                "resident_kv_logical_bytes": expected["resident_kv_tokens"] * kv_bytes,
            }
            for name in ("raw_tool_source_tokens", "raw_tool_resident_tokens",
                         "anchored_tool_source_tokens", "anchored_tool_gist_tokens"):
                if name in expected and _nonnegative_int(value.get(name),
                        f"response.costs.{name}") != expected[name]:
                    raise SGLangEventNativeError(f"response.costs.{name} mismatch")
        for name, expected_value in byte_expectations.items():
            if result[name] != expected_value:
                raise SGLangEventNativeError(f"response.costs.{name} mismatch")
        if not (memory.raw_tool_segments or memory.tool_gist_segments):
            result["system_prefix_kv_tokens"] = expected["system_tokens"]
            result["gist_prefix_kv_tokens"] = expected["gist_tokens"]
            result["workspace_resident_kv_tokens"] = expected["raw_tokens"]
        return result

    def _validate_shadow_features(
        self, value: Any, memory: PackedMemory
    ) -> dict[str, Any] | None:
        config = self.shadow_feature_config
        enabled = config is not None and config.enabled
        if not enabled:
            if value is not None:
                raise SGLangEventNativeError(
                    "server returned shadow_features when they were disabled"
                )
            return None
        if not isinstance(value, Mapping):
            raise SGLangEventNativeError("response.shadow_features must be an object")
        result = _json_object(value, "response.shadow_features")
        if result.get("schema") != SHADOW_FEATURE_SCHEMA:
            raise SGLangEventNativeError("response shadow feature schema mismatch")
        prefill = result.get("prefill")
        if not isinstance(prefill, Mapping):
            raise SGLangEventNativeError("response shadow_features.prefill is missing")
        if prefill.get("status") == "captured":
            if type(prefill.get("layer")) is not int or prefill["layer"] < 0:
                raise SGLangEventNativeError("captured prefill layer must be normalized")
            position = prefill.get("position")
            expected_position = (
                len(memory.system_input_ids)
                + sum(len(chunk.token_ids) for chunk in memory.chunks)
                + len(memory.workspace_input_ids)
                + sum(sum(len(chunk.token_ids) for chunk in segment["chunks"])
                      - (segment["token_end"] - segment["token_start"])
                      for segment in memory.tool_gist_segments)
                - 1
            )
            if (
                not isinstance(position, Mapping)
                or position.get("kind") != "prompt_last"
                or position.get("logical_position") != expected_position
            ):
                raise SGLangEventNativeError("captured prompt_last position mismatch")
            if prefill.get("readout") != "decoder_layer_output":
                raise SGLangEventNativeError("captured prefill readout mismatch")
            hidden = prefill.get("hidden")
            if not isinstance(hidden, list) or not hidden:
                raise SGLangEventNativeError("captured prefill hidden vector is empty")
            if any(
                isinstance(item, bool)
                or not isinstance(item, (int, float))
                or not math.isfinite(float(item))
                for item in hidden
            ):
                raise SGLangEventNativeError("captured prefill hidden vector is invalid")
        elif prefill.get("status") != "unavailable":
            raise SGLangEventNativeError("prefill status must be captured or unavailable")
        return result

    @staticmethod
    def _trace_context(value: Mapping[str, Any] | None) -> dict[str, Any]:
        if value is None:
            return {}
        allowed = {
            "attempt_uid", "session_id", "decision_key", "phase",
            "outer_request_id",
        }
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(
                "trace_context has unsupported fields: " + ", ".join(sorted(unknown))
            )
        result = dict(value)
        for name, item in result.items():
            if not isinstance(item, str) or not item:
                raise ValueError(f"trace_context.{name} must be a nonempty string")
        return result

    @staticmethod
    def _generation_id(context: Mapping[str, Any], generation_index: int) -> str:
        attempt_uid = context.get("attempt_uid")
        if isinstance(attempt_uid, str):
            return attempt_uid
        material = {
            "session_id": context.get("session_id"),
            "decision_key": context.get("decision_key"),
            "phase": context.get("phase"),
            "generation_index": generation_index,
            "nonce": time.time_ns(),
        }
        return hashlib.sha256(_canonical_bytes(material)).hexdigest()


__all__ = [
    "EXTRACTION_BUDGET_ERROR_CODE",
    "EXTRACTION_FAILURE_SCHEMA",
    "GENERATION_SCHEMA",
    "HANDLE_SCHEMA",
    "HTTP_JOURNAL_SCHEMA",
    "REQUEST_SCHEMA",
    "RESPONSE_SCHEMA",
    "SESSION_CACHE_POLICY",
    "SGLangEventNativeError",
    "SGLangTransportError",
    "SGLangExtractionBudgetExhausted",
    "SGLangEventNativeGenerationResult",
    "SGLangEventNativeGenerator",
]
