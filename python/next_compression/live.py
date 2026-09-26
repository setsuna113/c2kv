"""Bounded OpenAI chat serving for one next-compression checkpoint.

The live path rebuilds the same target-independent ``Decision`` and
``PackedMemory`` used by training.  It deliberately supports one checkpoint
and one ratio per process; history and tool checkpoints are never composed.
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
import time
import uuid
from dataclasses import dataclass, fields
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Mapping, Sequence

from history_memory.dataset import (
    Decision,
    event_store_session_id,
    iter_decisions,
    snapshot_and_validate_rows,
)
from history_memory.packing import (
    EncoderChunk,
    MemoryView,
    PackedMemory,
    PackingBudgetError,
    pack_memory,
    select_view,
)
from history_memory.events import EventRecord, EventStore, Message
from history_memory.preparation import tokenizer_identity
from history_memory.sources import (
    SourceRowError,
    _parse_tools,
    normalize_openai_messages,
)

from .common import SCHEMA, TRAINING_PROFILE, sha256_file
from . import history as history_module
from .history import HistoryPreparationConfig, _prepare_a_view
from .tools import (
    ToolPackingError,
    ToolPreparationConfig,
    pack_tool_memory,
    tool_variant_material,
)
from .vendor.a_runtime.history_memory.events import (
    EventRecord as VendorEventRecord,
    EventStore as VendorEventStore,
    Message as VendorMessage,
)


LIVE_PROTOCOL = "next-compression-openai-live-v1"
DEFAULT_MAX_REQUEST_BYTES = 16 * 1024 * 1024
_DUAL_CHECKPOINT_FIELDS = frozenset(
    {
        "checkpoint",
        "checkpoints",
        "history_checkpoint",
        "tool_checkpoint",
        "c2kv_checkpoint",
        "c2kv_checkpoints",
    }
)
_TOOL_CALL_OPEN = "<tool_call>"
_TOOL_CALL_CLOSE = "</tool_call>"


class LiveRequestError(ValueError):
    """A stable HTTP error that callers can classify without parsing prose."""

    def __init__(
        self,
        status: int,
        code: str,
        message: str,
        *,
        error_type: str = "invalid_request_error",
        param: str | None = None,
    ) -> None:
        self.status = status
        self.code = code
        self.error_type = error_type
        self.param = param
        super().__init__(message)

    def to_dict(self) -> dict[str, Any]:
        return {
            "message": str(self),
            "type": self.error_type,
            "param": self.param,
            "code": self.code,
        }


@dataclass(frozen=True)
class TrainingBinding:
    path: Path
    sha256: str
    manifest: Mapping[str, Any]
    preparation_config: HistoryPreparationConfig | ToolPreparationConfig


@dataclass(frozen=True)
class ParsedNativeResponse:
    content: str | None
    tool_calls: tuple[dict[str, Any], ...]
    status: str
    error: str | None = None


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read JSON object {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _manifest_profile(manifest: Mapping[str, Any], field: str) -> str:
    value = manifest.get(field)
    if value is None and isinstance(manifest.get("preparation"), Mapping):
        value = manifest["preparation"].get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Training manifest must declare {field}")
    return value


def _history_config(raw: Mapping[str, Any], variant: str) -> HistoryPreparationConfig:
    field_names = {item.name for item in fields(HistoryPreparationConfig)}
    selected_fields = {"h0_decision_ids", "h1_decision_ids"}
    required = field_names - selected_fields
    missing = sorted(required - set(raw))
    if missing:
        raise ValueError(f"Training manifest history config is incomplete: {missing}")
    extras = set(raw) - required
    allowed_extras = {"h0_decision_ids_count", "h1_decision_ids_count"}
    if extras - allowed_extras:
        raise ValueError(
            "Training manifest history config has unknown fields: "
            f"{sorted(extras - allowed_extras)}"
        )
    for name in allowed_extras & set(raw):
        if raw[name] is not None and (type(raw[name]) is not int or raw[name] < 0):
            raise ValueError(f"Training manifest {name} must be a nonnegative integer or null")
    values = {name: raw[name] for name in required}
    values["variants"] = tuple(values["variants"])
    values["ratios"] = tuple(values["ratios"])
    values["h0_decision_ids"] = None
    values["h1_decision_ids"] = None
    config = HistoryPreparationConfig(**values)
    if variant not in config.variants:
        raise ValueError("Checkpoint variant is absent from the frozen history config")
    return config


def _tool_config(raw: Mapping[str, Any]) -> ToolPreparationConfig:
    field_names = {item.name for item in fields(ToolPreparationConfig)}
    if set(raw) != field_names:
        missing = sorted(field_names - set(raw))
        extra = sorted(set(raw) - field_names)
        raise ValueError(
            "Training manifest tool config differs from the frozen schema: "
            f"missing={missing}, extra={extra}"
        )
    values = dict(raw)
    values["ratios"] = tuple(values["ratios"])
    return ToolPreparationConfig(**values)


def load_training_binding(
    path: str | Path,
    *,
    tokenizer: Any,
    checkpoint_profile: Mapping[str, Any],
) -> TrainingBinding:
    """Validate the exact prepared-corpus manifest bound into ``config.json``."""

    manifest_path = Path(path).resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Training manifest does not exist: {manifest_path}")
    manifest_sha256 = sha256_file(manifest_path)
    if manifest_sha256 != checkpoint_profile.get("corpus_identity"):
        raise ValueError(
            "Training manifest SHA256 differs from checkpoint corpus identity"
        )
    manifest = _read_json_object(manifest_path)
    expected = {
        "schema": SCHEMA,
        "training_profile": TRAINING_PROFILE,
        "variant": checkpoint_profile["variant"],
        "compression_domain": checkpoint_profile["compression_domain"],
        "ratios": [8, 12],
    }
    for field, value in expected.items():
        if manifest.get(field) != value:
            raise ValueError(f"Training manifest differs from checkpoint for {field}")
    for field in ("render_profile", "loss_profile"):
        if _manifest_profile(manifest, field) != checkpoint_profile[field]:
            raise ValueError(f"Training manifest differs from checkpoint for {field}")
    tokenizer_contract = manifest.get("tokenizer")
    if not isinstance(tokenizer_contract, Mapping):
        raise ValueError("Training manifest must declare its tokenizer")
    if tokenizer_contract.get("sha256") != tokenizer_identity(tokenizer)["sha256"]:
        raise ValueError("Checkpoint tokenizer differs from the training manifest")
    preparation = manifest.get("preparation")
    if not isinstance(preparation, Mapping) or not isinstance(
        preparation.get("config"), Mapping
    ):
        raise ValueError("Training manifest must contain preparation.config")
    variant = str(checkpoint_profile["variant"])
    if variant.startswith("H"):
        config: HistoryPreparationConfig | ToolPreparationConfig = _history_config(
            preparation["config"], variant
        )
    else:
        config = _tool_config(preparation["config"])
    return TrainingBinding(manifest_path, manifest_sha256, manifest, config)


def _canonical_digest(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _require_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LiveRequestError(400, "invalid_generation_control", f"{field} must be numeric", param=field)
    result = float(value)
    if not math.isfinite(result):
        raise LiveRequestError(400, "invalid_generation_control", f"{field} must be finite", param=field)
    return result


def _validate_generation_controls(
    payload: Mapping[str, Any], *, accepted_models: Sequence[str], server_max_new_tokens: int
) -> tuple[int, tuple[str, ...], str, dict[str, Any]]:
    requested_model = payload.get("model")
    if requested_model not in accepted_models:
        raise LiveRequestError(
            400,
            "model_alias_mismatch",
            f"model must be one of the served aliases {list(accepted_models)!r}",
            param="model",
        )
    for field in _DUAL_CHECKPOINT_FIELDS & set(payload):
        raise LiveRequestError(
            400,
            "dual_checkpoint_composition_unsupported",
            "Requests cannot select or compose checkpoints; start one server per checkpoint",
            param=field,
        )
    if payload.get("stream", False) is not False:
        raise LiveRequestError(400, "unsupported_streaming", "stream=true is unsupported", param="stream")
    if payload.get("n", 1) != 1:
        raise LiveRequestError(400, "unsupported_n", "Only n=1 is supported", param="n")
    temperature = _require_number(payload.get("temperature", 0.0), "temperature")
    if temperature != 0.0:
        raise LiveRequestError(400, "greedy_only", "Only temperature=0 greedy generation is supported", param="temperature")
    top_p = _require_number(payload.get("top_p", 1.0), "top_p")
    if top_p != 1.0:
        raise LiveRequestError(400, "greedy_only", "Only top_p=1 is supported", param="top_p")
    presence_penalty = _require_number(payload.get("presence_penalty", 0.0), "presence_penalty")
    if presence_penalty not in (0.0, 0.5):
        raise LiveRequestError(
            400,
            "unsupported_penalty",
            "presence_penalty must be 0 or the AppWorld wire default 0.5",
            param="presence_penalty",
        )
    frequency_penalty = _require_number(payload.get("frequency_penalty", 0.0), "frequency_penalty")
    if frequency_penalty != 0.0:
        raise LiveRequestError(400, "unsupported_penalty", "frequency_penalty must be 0", param="frequency_penalty")
    if payload.get("logprobs") not in (None, False):
        raise LiveRequestError(400, "unsupported_logprobs", "logprobs are unsupported", param="logprobs")
    if payload.get("top_logprobs") is not None or payload.get("logit_bias") not in (None, {}):
        raise LiveRequestError(400, "unsupported_logits_control", "Per-token logit controls are unsupported")
    seed = payload.get("seed", 0)
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise LiveRequestError(400, "invalid_seed", "seed must be a nonnegative integer", param="seed")
    template_kwargs = payload.get("chat_template_kwargs")
    if template_kwargs not in (None, {}, {"enable_thinking": False}):
        raise LiveRequestError(
            400,
            "unsupported_chat_template_kwargs",
            "Only enable_thinking=false is supported",
            param="chat_template_kwargs",
        )
    if payload.get("tool_choice") not in (None, "auto"):
        raise LiveRequestError(
            400,
            "unsupported_tool_choice",
            "Only automatic tool choice is supported",
            param="tool_choice",
        )
    if payload.get("parallel_tool_calls") not in (None, True):
        raise LiveRequestError(
            400,
            "unsupported_parallel_tool_calls",
            "parallel_tool_calls=false cannot be enforced by this runtime",
            param="parallel_tool_calls",
        )
    if payload.get("response_format") is not None:
        raise LiveRequestError(400, "unsupported_response_format", "response_format is unsupported", param="response_format")
    if payload.get("store") not in (None, False):
        raise LiveRequestError(400, "unsupported_store", "store=true is unsupported", param="store")

    values = [payload[name] for name in ("max_tokens", "max_completion_tokens") if payload.get(name) is not None]
    if len(values) == 2 and values[0] != values[1]:
        raise LiveRequestError(400, "conflicting_token_caps", "max_tokens and max_completion_tokens differ")
    max_new_tokens = values[0] if values else server_max_new_tokens
    if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or max_new_tokens <= 0:
        raise LiveRequestError(400, "invalid_max_tokens", "max_tokens must be a positive integer", param="max_tokens")
    if max_new_tokens > server_max_new_tokens:
        raise LiveRequestError(
            400,
            "max_tokens_exceeds_server_cap",
            f"Requested {max_new_tokens} tokens; server cap is {server_max_new_tokens}",
            param="max_tokens",
        )

    raw_stop = payload.get("stop")
    if raw_stop is None:
        stops: tuple[str, ...] = ()
    elif isinstance(raw_stop, str):
        stops = (raw_stop,)
    elif isinstance(raw_stop, Sequence) and not isinstance(raw_stop, (bytes, bytearray)):
        stops = tuple(raw_stop)
    else:
        raise LiveRequestError(400, "invalid_stop", "stop must be a string or a sequence of strings", param="stop")
    if len(stops) > 8 or any(not isinstance(item, str) or not item or len(item) > 1024 for item in stops):
        raise LiveRequestError(400, "invalid_stop", "stop allows 1-8 nonempty strings of at most 1024 characters", param="stop")
    controls = {
        "profile": "wire-defaults-normalized-to-frozen-greedy-v1",
        "client": {
            "temperature": temperature,
            "top_p": top_p,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
            "seed": seed,
            "chat_template_kwargs": template_kwargs,
        },
        "normalized": {
            "temperature": 0.0,
            "top_p": 1.0,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "seed": 0,
        },
    }
    return max_new_tokens, stops, str(requested_model), controls


def _has_ace_text_observation(messages: Sequence[Mapping[str, Any]]) -> bool:
    return any(
        isinstance(message, Mapping)
        and message.get("role") == "tool"
        and isinstance(message.get("tool_call_id"), str)
        and message["tool_call_id"].startswith("acebench-execution-")
        for message in messages
    )


def _normalize_ace_text_messages(
    raw_messages: Sequence[Any],
) -> list[dict[str, Any]]:
    """Validate the ACE wire rows without inventing native call bindings."""

    messages: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_messages):
        if not isinstance(raw, Mapping):
            raise SourceRowError("invalid_message", f"index={index}")
        role = raw.get("role") or raw.get("type")
        if role not in {"system", "developer", "user", "assistant", "tool"}:
            raise SourceRowError("unsupported_role", f"index={index} role={role!r}")
        content = raw.get("content")
        if content is not None and not isinstance(content, str):
            raise SourceRowError("non_text_content", f"index={index}")
        if raw.get("function_call") is not None or raw.get("tool_calls"):
            raise SourceRowError(
                "mixed_ace_native_tool_history",
                f"index={index}",
            )
        message: dict[str, Any] = {"role": role, "content": content}
        if isinstance(raw.get("name"), str):
            message["name"] = raw["name"]
        if role == "tool":
            call_id = raw.get("tool_call_id") or raw.get("toolCallId")
            if not isinstance(call_id, str) or not call_id:
                raise SourceRowError("unmatched_tool_result", f"index={index}")
            message["tool_call_id"] = call_id
        messages.append(message)
    return messages


def _ace_text_observation_decision(
    messages: Sequence[Mapping[str, Any]],
    tools: Sequence[Mapping[str, Any]],
    request_digest: str,
) -> Decision:
    """Preserve ACE's assistant-text/execution-observation pairs as opaque events."""

    session_id = f"live-{request_digest}"
    store_session_id = event_store_session_id("live-openai-chat", session_id)
    snapshots = tuple(Message.from_dict(message) for message in messages)
    events: list[EventRecord] = []
    index = 0
    while index < len(messages):
        message = messages[index]
        if message.get("role") == "tool":
            raise ValueError(
                f"Unmatched ACEBench execution observation at source index {index}"
            )
        if index + 1 < len(messages):
            following = messages[index + 1]
            call_id = following.get("tool_call_id")
            if (
                message.get("role") == "assistant"
                and following.get("role") == "tool"
                and isinstance(call_id, str)
                and call_id.startswith("acebench-execution-")
            ):
                events.append(
                    EventRecord(
                        event_id=f"{store_session_id}:m{index}",
                        kind="acebench_execution_opaque",
                        source_indices=(index, index + 1),
                        complete=False,
                    )
                )
                index += 2
                continue
        role = message["role"]
        events.append(
            EventRecord(
                event_id=f"{store_session_id}:m{index}",
                kind="instruction" if role in {"system", "developer"} else role,
                source_indices=(index,),
                complete=True,
            )
        )
        index += 1
    decision_index = sum(
        1
        for message in messages
        if message.get("role") == "assistant"
        and (message.get("content") or message.get("tool_calls"))
    )
    return Decision(
        decision_id=f"decision-{request_digest}",
        session_id=session_id,
        source="live-openai-chat",
        split="train",
        task_id=session_id,
        template_id=LIVE_PROTOCOL,
        decision_index=decision_index,
        source_message_index=len(messages),
        store=EventStore(store_session_id, snapshots, tuple(events)),
        target=Message.from_dict(
            {"role": "assistant", "content": "__c2kv_live_target_placeholder__"}
        ),
        tools_json=json.dumps(list(tools), ensure_ascii=False, allow_nan=False),
    )


def _live_decision(
    payload: Mapping[str, Any],
) -> tuple[Decision, tuple[dict[str, Any], ...], str, str]:
    raw_messages = payload.get("messages")
    if isinstance(raw_messages, (str, bytes, bytearray)) or not isinstance(raw_messages, Sequence) or not raw_messages:
        raise LiveRequestError(400, "invalid_messages", "messages must be a nonempty sequence", param="messages")
    try:
        tools = tuple(_parse_tools(payload.get("tools") or []))
        request_digest = _canonical_digest([raw_messages, tools])
        messages = (
            _normalize_ace_text_messages(raw_messages)
            if _has_ace_text_observation(raw_messages)
            else normalize_openai_messages(
                raw_messages,
                namespace=f"live-{request_digest[:16]}",
            )
        )
        if not messages:
            raise ValueError("normalization removed every message")
        if _has_ace_text_observation(messages):
            decision = _ace_text_observation_decision(
                messages, tools, request_digest
            )
            return (
                decision,
                tools,
                request_digest,
                "acebench-text-observation-opaque-v1",
            )
        row = {
            "session_id": f"live-{request_digest}",
            "source": "live-openai-chat",
            "split": "train",
            "task_id": f"live-{request_digest}",
            "template_id": LIVE_PROTOCOL,
            "messages": [
                *messages,
                {"role": "assistant", "content": "__c2kv_live_target_placeholder__"},
            ],
            "tools": list(tools),
        }
        snapshot = snapshot_and_validate_rows((row,))[0]
        decisions = tuple(iter_decisions(snapshot))
        if not decisions:
            raise ValueError("request did not produce a live decision")
        decision = decisions[-1]
        if decision.source_message_index != len(snapshot["messages"]) - 1:
            raise AssertionError("live placeholder was not the final decision")
    except LiveRequestError:
        raise
    except (SourceRowError, TypeError, ValueError) as error:
        raise LiveRequestError(
            400,
            "invalid_message_history",
            str(error),
            param="messages",
        ) from error
    return decision, tools, request_digest, "history-memory-normalized-openai-v1"


def _all_raw_memory(
    decision: Decision,
    tokenizer: Any,
    config: HistoryPreparationConfig | ToolPreparationConfig,
) -> PackedMemory:
    view = MemoryView(
        gist_event_ids=(),
        raw_event_ids=tuple(event.event_id for event in decision.store.events),
        evidence_event_ids=(),
    )
    return pack_memory(
        decision.store,
        view,
        tokenizer,
        tools=decision.tools or None,
        max_chunk_tokens=config.max_chunk_tokens,
        chunk_overlap=config.chunk_overlap,
        max_chunks=config.max_chunks,
        max_raw_tokens=None,
    )


def _memory_limits(
    memory: PackedMemory,
    *,
    ratio: int,
    max_new_tokens: int,
    config: HistoryPreparationConfig | ToolPreparationConfig,
) -> None:
    costs = memory.costs(ratio)
    if isinstance(config, HistoryPreparationConfig):
        if costs["system_tokens"] > config.max_system_tokens:
            raise PackingBudgetError("system_tokens_over_limit")
        if costs["raw_tokens"] > config.max_workspace_tokens:
            raise PackingBudgetError("workspace_tokens_over_limit")
        if costs["presented_encoder_tokens"] > config.max_encoder_tokens:
            raise PackingBudgetError("encoder_tokens_over_limit")
        target_cap = config.max_target_tokens
        sequence_cap = config.max_sequence_tokens
    else:
        target_cap = config.max_target_tokens
        sequence_cap = config.max_sequence_tokens
    if target_cap is not None and max_new_tokens > target_cap:
        raise PackingBudgetError("generation_tokens_over_training_target_limit")
    if costs["resident_kv_tokens"] + max_new_tokens > sequence_cap:
        raise PackingBudgetError("resident_context_over_limit")


def _prepare_adapted_a_view(
    decision: Decision,
    tokenizer: Any,
    config: HistoryPreparationConfig,
) -> tuple[PackedMemory, dict[str, Any]]:
    """Run the frozen A selector on an already-normalized non-native source store."""

    packing = {
        "ratios": list(config.ratios),
        "recent_tool_events": config.recent_tool_events,
        "max_chunk_tokens": config.max_chunk_tokens,
        "chunk_overlap": config.chunk_overlap,
        "max_chunks": config.max_chunks,
        "max_encoder_tokens": config.max_encoder_tokens,
        "max_system_tokens": config.max_system_tokens,
        "max_workspace_tokens": config.max_workspace_tokens,
        "max_target_tokens": config.max_target_tokens,
        "max_sequence_tokens": config.max_sequence_tokens,
    }
    policy = {
        "mode": "persistent",
        "history_budget_bytes": config.history_budget_bytes,
        "workspace_budget_bytes": config.workspace_budget_bytes,
        "lease_decisions": 0,
        "max_retrieved_events": config.max_retrieved_events,
        "kv_bytes_per_token": config.kv_bytes_per_token,
        "source_commit": history_module.A_POLICY_SOURCE_COMMIT,
        "history_budget_definition": history_module.A_HISTORY_BUDGET_DEFINITION,
        "workspace_budget_definition": history_module.A_WORKSPACE_BUDGET_DEFINITION,
        "current_input_baseline": history_module.A_CURRENT_INPUT_BASELINE,
    }
    controller = history_module.EventNativeS0Controller(
        tokenizer,
        packing=packing,
        policy=policy,
        model_context=config.max_sequence_tokens,
        s0_config={
            "source_index_max_events": config.source_index_max_events,
            "predictor_prompt_token_cap": config.predictor_prompt_token_cap,
            "predictor_completion_token_cap": 256,
            "latest_complete_tool_protection": "budgeted",
        },
    )
    vendor_store = VendorEventStore(
        session_id=decision.store.session_id,
        messages=tuple(
            VendorMessage.from_dict(message.to_dict())
            for message in decision.store.messages
        ),
        events=tuple(
            VendorEventRecord(
                event_id=event.event_id,
                kind=event.kind,
                source_indices=event.source_indices,
                complete=event.complete,
                tool_call_ids=event.tool_call_ids,
                missing_tool_call_ids=event.missing_tool_call_ids,
            )
            for event in decision.store.events
        ),
    )
    prepared = controller._prepare_view(
        vendor_store,
        tuple(decision.tools),
        ratio=config.ratios[0],
        max_new_tokens=config.a_max_new_tokens,
        decision_key=decision.decision_id,
        decision_index=decision.decision_index + 1,
    )
    vendor_memory = prepared.memory
    memory = PackedMemory(
        view=MemoryView(
            tuple(vendor_memory.view.gist_event_ids),
            tuple(vendor_memory.view.raw_event_ids),
            tuple(vendor_memory.view.evidence_event_ids),
        ),
        system_input_ids=tuple(vendor_memory.system_input_ids),
        workspace_input_ids=tuple(vendor_memory.workspace_input_ids),
        raw_source_indices=tuple(vendor_memory.raw_source_indices),
        chunks=tuple(
            EncoderChunk(
                chunk.event_id,
                chunk.part_index,
                tuple(chunk.source_indices),
                chunk.source_token_start,
                chunk.source_token_end,
                tuple(chunk.token_ids),
            )
            for chunk in vendor_memory.chunks
        ),
        raw_layout_profile=vendor_memory.raw_layout_profile,
    )
    actual = prepared.metadata
    coverage = actual["source_coverage"]
    eligible_sources = coverage["eligible_source_indices"]
    unrepresented_sources = coverage["unrepresented_source_indices"] or []
    return memory, {
        "version": history_module.A_VIEW_VERSION,
        "source": history_module.A_VIEW_SOURCE,
        "source_sha256": history_module.A_VIEW_SOURCE_SHA256,
        "vendored_source_sha256": history_module.A_VIEW_VENDOR_SHA256,
        "vendor_manifest_sha256": history_module.A_VIEW_VENDOR_MANIFEST_SHA256,
        "mode": history_module.A_VIEW_MODE,
        "actual_controller_mode": actual["mode"],
        "controller_invocation_ratio": config.ratios[0],
        "eligible_event_ids": actual["eligible_extraction"]["eligible_event_ids"],
        "raw_event_ids": actual["raw_event_ids"],
        "gist_event_ids": actual["gist_event_ids"],
        "eligible_source_count": len(eligible_sources),
        "covered_eligible_source_count": len(eligible_sources)
        - len(unrepresented_sources),
        "unrepresented_eligible_source_count": len(unrepresented_sources),
        "source_adapter": "acebench-text-observation-opaque-v1",
    }


def _hybrid_tool_material(
    decision: Decision,
    tools: Sequence[Mapping[str, Any]],
    top_k: int,
) -> tuple[Any, dict[str, Any]]:
    """Section 3.2 hybrid allocation: lexical top-k native, remainder as T0 blocks."""
    from .exp1_tools import (
        RANKER,
        decision_messages,
        layout_material,
        lexical_rank,
        query_text,
    )

    if type(top_k) is not int or top_k <= 0:
        raise ValueError("tool_top_k must be a positive integer")
    ranked = lexical_rank(tools, query_text(decision_messages(decision)))
    native = tuple(sorted(ranked[:top_k]))
    material = layout_material(tools, "hybrid", native_indices=native)
    return material, {
        "layout": "hybrid",
        "k": top_k,
        "ranker": RANKER,
        "native_tool_indices": list(native),
        "lexical_rank": list(ranked),
    }


def _prepare_memory(
    decision: Decision,
    tools: Sequence[Mapping[str, Any]],
    *,
    tokenizer: Any,
    variant: str,
    mode: str,
    ratio: int,
    max_new_tokens: int,
    config: HistoryPreparationConfig | ToolPreparationConfig,
    source_profile: str,
    tool_layout: str = "variant",
    tool_top_k: int = 3,
) -> tuple[PackedMemory, dict[str, Any]]:
    metadata: dict[str, Any] = {}
    reason: str | None = None
    tool_documents = 0
    compressed_tool_definitions = 0
    if tool_layout not in ("variant", "hybrid"):
        raise ValueError("tool_layout must be 'variant' or 'hybrid'")
    if mode == "full":
        memory = _all_raw_memory(decision, tokenizer, config)
        reason = "mode_full"
    elif variant in {"H0", "H1"}:
        assert isinstance(config, HistoryPreparationConfig)
        view = select_view(decision.store, recent_tool_events=config.recent_tool_events)
        memory = pack_memory(
            decision.store,
            view,
            tokenizer,
            tools=decision.tools or None,
            max_chunk_tokens=config.max_chunk_tokens,
            chunk_overlap=config.chunk_overlap,
            max_chunks=config.max_chunks,
        )
        if not memory.chunks:
            reason = "no_compressible_history_events"
    elif variant in {"H2", "H3"}:
        assert isinstance(config, HistoryPreparationConfig)
        if source_profile == "acebench-text-observation-opaque-v1":
            memory, a_view = _prepare_adapted_a_view(decision, tokenizer, config)
            metadata["a_view"] = a_view
        else:
            prepared = _prepare_a_view(decision, tokenizer, config)
            memory = prepared.memory
            metadata["a_view"] = dict(prepared.metadata)
        if not memory.chunks:
            reason = "no_admitted_history_events"
    else:
        assert isinstance(config, ToolPreparationConfig)
        if not tools:
            memory = _all_raw_memory(decision, tokenizer, config)
            reason = "no_native_tool_definitions"
        else:
            layout_info: dict[str, Any] | None = None
            try:
                if tool_layout == "hybrid":
                    if variant != "T0":
                        raise ValueError("The hybrid tool layout requires a T0 checkpoint")
                    material, layout_info = _hybrid_tool_material(decision, tools, tool_top_k)
                else:
                    material = tool_variant_material(tools, variant)
            except ToolPackingError as error:
                if variant == "T1" and getattr(error, "reason", str(error)) == "no_compressible_tool_information":
                    memory = _all_raw_memory(decision, tokenizer, config)
                    reason = "no_descriptive_tool_fields"
                else:
                    raise
            else:
                tool_documents = len(material.documents)
                compressed_tool_definitions = len(
                    {int(item["tool_index"]) for item in material.documents}
                )
                memory = pack_tool_memory(
                    decision.store,
                    tools,
                    tokenizer,
                    variant=variant,
                    config=config,
                    material=material if layout_info is not None else None,
                )
                if layout_info is not None:
                    metadata["tool_layout"] = layout_info
    base_context_only = mode == "full" or reason in {
        "no_native_tool_definitions",
        "no_descriptive_tool_fields",
    }
    if not base_context_only:
        _memory_limits(
            memory,
            ratio=ratio,
            max_new_tokens=max_new_tokens,
            config=config,
        )
    costs = memory.costs(ratio)
    event_ids = {chunk.event_id for chunk in memory.chunks}
    source_indices = {
        source_index for chunk in memory.chunks for source_index in chunk.source_indices
    }
    total_tools = len(tools)
    coverage = {
        "native_tool_definitions": total_tools,
        "compressed_tool_documents": tool_documents,
        "compressed_tool_definitions": compressed_tool_definitions,
        "tool_definition_coverage": (
            compressed_tool_definitions / total_tools if total_tools else 0.0
        ),
        "compressed_history_events": len(event_ids) if variant.startswith("H") else 0,
        "compressed_history_source_messages": len(source_indices) if variant.startswith("H") else 0,
    }
    if "a_view" in metadata:
        a_view = metadata["a_view"]
        coverage.update(
            {
                "eligible_history_sources": a_view["eligible_source_count"],
                "covered_history_sources": a_view["covered_eligible_source_count"],
                "unrepresented_history_sources": a_view[
                    "unrepresented_eligible_source_count"
                ],
            }
        )
    return memory, {
        "requested_mode": mode,
        "effective_mode": "compressed" if memory.chunks else "raw",
        "applied": bool(memory.chunks),
        "reason": reason,
        "budget_profile": (
            "base-model-context-only"
            if base_context_only
            else "frozen-training-preparation"
        ),
        "chunk_count": len(memory.chunks),
        "compressed_document_count": len(event_ids),
        "coverage": coverage,
        "token_counts": costs,
        **metadata,
    }


def _decode(tokenizer: Any, token_ids: Sequence[int]) -> str:
    return tokenizer.decode(
        list(token_ids),
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )


def _stop_callback(tokenizer: Any, stops: Sequence[str]):
    if not stops:
        return None

    def stopped(token_ids: tuple[int, ...]) -> bool:
        text = _decode(tokenizer, token_ids)
        return any(text.endswith(stop) for stop in stops)

    return stopped


def _strip_stop(text: str, stops: Sequence[str]) -> str:
    positions = [position for stop in stops if (position := text.find(stop)) >= 0]
    return text[: min(positions)] if positions else text


def parse_native_response(text: str) -> ParsedNativeResponse:
    """Parse complete Qwen ``<tool_call>`` blocks without partial submission."""

    if _TOOL_CALL_OPEN not in text and _TOOL_CALL_CLOSE not in text:
        return ParsedNativeResponse(text, (), "text")
    cursor = 0
    pieces: list[str] = []
    calls: list[dict[str, Any]] = []
    while cursor < len(text):
        start = text.find(_TOOL_CALL_OPEN, cursor)
        close_without_open = text.find(_TOOL_CALL_CLOSE, cursor)
        if start < 0:
            if close_without_open >= 0:
                return ParsedNativeResponse(text, (), "invalid_tool_call", "closing_tag_without_open")
            pieces.append(text[cursor:])
            break
        if 0 <= close_without_open < start:
            return ParsedNativeResponse(text, (), "invalid_tool_call", "closing_tag_without_open")
        pieces.append(text[cursor:start])
        body_start = start + len(_TOOL_CALL_OPEN)
        end = text.find(_TOOL_CALL_CLOSE, body_start)
        if end < 0:
            return ParsedNativeResponse(text, (), "invalid_tool_call", "unclosed_tool_call")
        raw = text[body_start:end].strip()
        try:
            value = json.loads(raw)
            if not isinstance(value, Mapping):
                raise ValueError("tool call must be a JSON object")
            name = value.get("name")
            arguments = value.get("arguments", {})
            if isinstance(arguments, str):
                arguments = json.loads(arguments) if arguments.strip() else {}
            if not isinstance(name, str) or not name or not isinstance(arguments, Mapping):
                raise ValueError("tool call requires a name and arguments object")
        except (json.JSONDecodeError, ValueError) as error:
            return ParsedNativeResponse(text, (), "invalid_tool_call", str(error))
        identity = _canonical_digest([len(calls), name, arguments])[:24]
        calls.append(
            {
                "id": f"call_{identity}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(
                        dict(arguments),
                        ensure_ascii=False,
                        separators=(",", ":"),
                        allow_nan=False,
                    ),
                },
            }
        )
        cursor = end + len(_TOOL_CALL_CLOSE)
    content = "".join(pieces).strip()
    return ParsedNativeResponse(content or None, tuple(calls), "tool_calls")


class LiveNextCompressionService:
    """One loaded checkpoint, serialized generation, and finite work budget."""

    def __init__(
        self,
        generator: Any,
        tokenizer: Any,
        checkpoint_profile: Mapping[str, Any],
        training_binding: TrainingBinding,
        *,
        ratio: int,
        mode: str,
        model: str,
        model_aliases: Sequence[str] = (),
        max_new_tokens: int,
        max_requests: int,
        max_request_bytes: int = DEFAULT_MAX_REQUEST_BYTES,
        ledger_path: str | Path | None = None,
        tool_layout: str = "variant",
        tool_top_k: int = 3,
        max_raw_tokens: int | None = None,
    ) -> None:
        if ratio not in (8, 12):
            raise ValueError("ratio must be 8 or 12")
        if mode not in ("compressed", "full"):
            raise ValueError("mode must be compressed or full")
        if tool_layout not in ("variant", "hybrid"):
            raise ValueError("tool_layout must be 'variant' or 'hybrid'")
        if type(tool_top_k) is not int or tool_top_k <= 0:
            raise ValueError("tool_top_k must be a positive integer")
        if tool_layout == "hybrid" and checkpoint_profile.get("variant") != "T0":
            raise ValueError("The hybrid tool layout requires a T0 checkpoint")
        if max_raw_tokens is not None and (type(max_raw_tokens) is not int or max_raw_tokens <= 0):
            raise ValueError("max_raw_tokens must be a positive integer or None")
        if max_raw_tokens is not None:
            import dataclasses

            if not isinstance(training_binding.preparation_config, ToolPreparationConfig):
                raise ValueError("max_raw_tokens applies to tool checkpoints only")
            training_binding = dataclasses.replace(
                training_binding,
                preparation_config=dataclasses.replace(
                    training_binding.preparation_config, max_raw_tokens=max_raw_tokens
                ),
            )
        if not isinstance(model, str) or not model:
            raise ValueError("model alias must be nonempty")
        for name, value in (
            ("max_new_tokens", max_new_tokens),
            ("max_requests", max_requests),
            ("max_request_bytes", max_request_bytes),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        config = training_binding.preparation_config
        target_cap = config.max_target_tokens
        if mode == "compressed" and target_cap is not None and max_new_tokens > target_cap:
            raise ValueError(
                "Server max_new_tokens exceeds the frozen training target limit"
            )
        self.generator = generator
        self.tokenizer = tokenizer
        self.profile = dict(checkpoint_profile)
        self.binding = training_binding
        self.ratio = ratio
        self.mode = mode
        self.model = model
        self.tool_layout = tool_layout
        self.tool_top_k = tool_top_k
        self.max_raw_tokens_override = max_raw_tokens
        if any(not isinstance(alias, str) or not alias for alias in model_aliases):
            raise ValueError("model aliases must be nonempty strings")
        self.accepted_models = tuple(dict.fromkeys((model, *model_aliases)))
        self.max_new_tokens = max_new_tokens
        self.max_requests = max_requests
        self.max_request_bytes = max_request_bytes
        self._lock = threading.Lock()
        self._requests_started = 0
        self._requests_completed = 0
        self._requests_failed = 0
        self._completion_tokens = 0
        self._ledger_path = Path(ledger_path).resolve() if ledger_path is not None else None

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str | Path,
        training_manifest: str | Path,
        **kwargs: Any,
    ) -> "LiveNextCompressionService":
        from .inference import load_next_checkpoint

        generator, tokenizer, profile = load_next_checkpoint(
            checkpoint,
            device=kwargs.pop("device"),
            dtype=kwargs.pop("dtype"),
            decode_strategy="incremental",
        )
        binding = load_training_binding(
            training_manifest,
            tokenizer=tokenizer,
            checkpoint_profile=profile,
        )
        return cls(generator, tokenizer, profile, binding, **kwargs)

    def _binding_metadata(self) -> dict[str, Any]:
        return {
            "protocol": LIVE_PROTOCOL,
            "model": self.model,
            "accepted_model_aliases": list(self.accepted_models),
            "checkpoint": self.profile["checkpoint"],
            "config_sha256": self.profile["config_sha256"],
            "training_manifest": str(self.binding.path),
            "training_manifest_sha256": self.binding.sha256,
            "checkpoint_corpus_identity": self.profile["corpus_identity"],
            "training_profile": self.profile["training_profile"],
            "variant": self.profile["variant"],
            "compression_domain": self.profile["compression_domain"],
            "render_profile": self.profile["render_profile"],
            "loss_profile": self.profile["loss_profile"],
            "initialization_id": self.profile["initialization_id"],
            "ratio": self.ratio,
            "mode": self.mode,
            "tool_layout": self.tool_layout,
            "tool_top_k": self.tool_top_k if self.tool_layout == "hybrid" else None,
            "max_raw_tokens_override": self.max_raw_tokens_override,
            "device": self.profile["device"],
            "dtype": self.profile["dtype"],
        }

    def health(self) -> dict[str, Any]:
        with self._lock:
            counters = {
                "requests_started": self._requests_started,
                "requests_completed": self._requests_completed,
                "requests_failed": self._requests_failed,
                "requests_remaining": self.max_requests - self._requests_started,
                "completion_tokens_emitted": self._completion_tokens,
            }
        return {
            "status": "ok",
            **self._binding_metadata(),
            "limits": {
                "max_new_tokens_per_request": self.max_new_tokens,
                "max_requests": self.max_requests,
                "max_total_completion_tokens": self.max_requests
                * self.max_new_tokens,
                "max_request_bytes": self.max_request_bytes,
            },
            "capabilities": {
                "stream": False,
                "sampling": "greedy",
                "native_tool_calls": True,
                "multiple_tool_calls": True,
                "dual_checkpoint_composition": False,
                "serialized_generation": True,
                "request_local_cache_cleanup": True,
            },
            "counters": counters,
        }

    def error_body(self, error: LiveRequestError) -> dict[str, Any]:
        return {"error": error.to_dict(), "x_c2kv": self._binding_metadata()}

    def _write_ledger(self, record: Mapping[str, Any]) -> None:
        if self._ledger_path is None:
            return
        with self._ledger_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(record, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
                + "\n"
            )
            handle.flush()

    def complete(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(payload, Mapping):
            raise LiveRequestError(400, "invalid_json_body", "Request body must be a JSON object")
        max_new_tokens, stops, requested_model, generation_controls = _validate_generation_controls(
            payload,
            accepted_models=self.accepted_models,
            server_max_new_tokens=self.max_new_tokens,
        )
        decision, tools, request_digest, source_profile = _live_decision(payload)
        started = time.time()
        with self._lock:
            if self._requests_started >= self.max_requests:
                raise LiveRequestError(
                    429,
                    "request_cap_exhausted",
                    "Finite server request cap is exhausted",
                    error_type="c2kv_limit_error",
                )
            self._requests_started += 1
            ordinal = self._requests_started
            request_id = f"chatcmpl-c2kv-{ordinal:08d}-{uuid.uuid4().hex[:12]}"
            try:
                try:
                    memory, compression = _prepare_memory(
                        decision,
                        tools,
                        tokenizer=self.tokenizer,
                        variant=self.profile["variant"],
                        mode=self.mode,
                        ratio=self.ratio,
                        max_new_tokens=max_new_tokens,
                        config=self.binding.preparation_config,
                        source_profile=source_profile,
                        tool_layout=self.tool_layout,
                        tool_top_k=self.tool_top_k,
                    )
                except (PackingBudgetError, ToolPackingError, ValueError) as error:
                    reason = getattr(error, "reason", None) or str(error)
                    raise LiveRequestError(
                        422,
                        "packing_budget_exceeded",
                        reason,
                        error_type="c2kv_model_failure",
                    ) from error
                callback = _stop_callback(self.tokenizer, stops)
                try:
                    with self.generator.decision_scope(session_id=None):
                        generation = self.generator.generate(
                            memory,
                            ratio=self.ratio,
                            max_new_tokens=max_new_tokens,
                            token_prefix_stop=callback,
                            trace_context={
                                "attempt_uid": request_id,
                                "session_id": decision.store.session_id,
                                "decision_key": decision.decision_id,
                                "phase": LIVE_PROTOCOL,
                            },
                        )
                except ValueError as error:
                    raise LiveRequestError(
                        422,
                        "generation_context_exceeded",
                        str(error),
                        error_type="c2kv_model_failure",
                    ) from error
                text = _strip_stop(_decode(self.tokenizer, generation.token_ids), stops)
                parsed = (
                    parse_native_response(text)
                    if tools
                    else ParsedNativeResponse(text, (), "text_no_native_tools")
                )
                message: dict[str, Any] = {
                    "role": "assistant",
                    "content": parsed.content,
                }
                if parsed.tool_calls:
                    message["tool_calls"] = list(parsed.tool_calls)
                finish_reason = (
                    "tool_calls"
                    if parsed.tool_calls
                    else "length"
                    if generation.finish_reason == "length"
                    else "stop"
                )
                completion_tokens = len(generation.token_ids)
                token_counts = {
                    **compression["token_counts"],
                    "completion_tokens": completion_tokens,
                    "physical_sequence_tokens": compression["token_counts"][
                        "resident_kv_tokens"
                    ]
                    + completion_tokens,
                }
                x_c2kv = {
                    **self._binding_metadata(),
                    "request_ordinal": ordinal,
                    "request_sha256": request_digest,
                    "source_profile": source_profile,
                    "generation_controls": generation_controls,
                    "compression": compression,
                    "token_counts": token_counts,
                    "native_parse_status": parsed.status,
                    "native_parse_error": parsed.error,
                    "native_finish_reason": generation.finish_reason,
                    "runtime": {
                        key: generation.stats.get(key)
                        for key in (
                            "elapsed_sec",
                            "extraction_calls",
                            "target_forward_calls",
                            "target_input_tokens",
                            "resident_kv_tokens_final",
                        )
                        if key in generation.stats
                    },
                }
                response = {
                    "id": request_id,
                    "object": "chat.completion",
                    "created": int(started),
                    "model": requested_model,
                    "system_fingerprint": f"c2kv-{self.profile['config_sha256'][:16]}",
                    "choices": [
                        {
                            "index": 0,
                            "message": message,
                            "finish_reason": finish_reason,
                            "logprobs": None,
                        }
                    ],
                    "usage": {
                        "prompt_tokens": compression["token_counts"][
                            "resident_kv_tokens"
                        ],
                        "completion_tokens": completion_tokens,
                        "total_tokens": compression["token_counts"][
                            "resident_kv_tokens"
                        ]
                        + completion_tokens,
                    },
                    "x_c2kv": x_c2kv,
                }
                self._requests_completed += 1
                self._completion_tokens += completion_tokens
                self._write_ledger(
                    {
                        "request_ordinal": ordinal,
                        "request_sha256": request_digest,
                        "status": "completed",
                        "native_parse_status": parsed.status,
                        "finish_reason": finish_reason,
                        "token_counts": token_counts,
                        "compression": {
                            key: compression[key]
                            for key in (
                                "effective_mode",
                                "applied",
                                "reason",
                                "chunk_count",
                                "compressed_document_count",
                                "coverage",
                            )
                        },
                    }
                )
                return response
            except BaseException as error:
                self._requests_failed += 1
                self._write_ledger(
                    {
                        "request_ordinal": ordinal,
                        "request_sha256": request_digest,
                        "status": "failed",
                        "error_type": type(error).__name__,
                        "error_code": getattr(error, "code", None),
                    }
                )
                raise
            finally:
                self.generator.close_session()


def make_http_server(
    service: LiveNextCompressionService, host: str, port: int
) -> ThreadingHTTPServer:
    """Construct a local HTTP server; model work remains serialized by service."""

    if host != "127.0.0.1":
        raise ValueError("This server is intentionally bound to 127.0.0.1")
    if type(port) is not int or not 1 <= port <= 65535:
        raise ValueError("port must be between 1 and 65535")

    class Handler(BaseHTTPRequestHandler):
        server_version = "C2KVNextCompression/1"

        def log_message(self, format: str, *args: Any) -> None:
            return

        def _send(self, status: int, value: Mapping[str, Any]) -> None:
            body = json.dumps(
                value,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            if self.path.rstrip("/") == "/health":
                self._send(200, service.health())
            else:
                self._send(
                    404,
                    service.error_body(
                        LiveRequestError(404, "not_found", "Endpoint not found")
                    ),
                )

        def do_POST(self) -> None:  # noqa: N802
            if self.path.rstrip("/") != "/v1/chat/completions":
                self._send(
                    404,
                    service.error_body(
                        LiveRequestError(404, "not_found", "Endpoint not found")
                    ),
                )
                return
            try:
                length = int(self.headers.get("Content-Length", ""))
                if length <= 0:
                    raise LiveRequestError(400, "invalid_content_length", "Content-Length must be positive")
                if length > service.max_request_bytes:
                    raise LiveRequestError(
                        413,
                        "request_too_large",
                        f"Request exceeds {service.max_request_bytes} bytes",
                        error_type="c2kv_limit_error",
                    )
                raw = self.rfile.read(length)
                value = json.loads(raw.decode("utf-8"))
                response = service.complete(value)
            except LiveRequestError as error:
                self._send(error.status, service.error_body(error))
                return
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
                request_error = LiveRequestError(400, "invalid_json_body", str(error))
                self._send(400, service.error_body(request_error))
                return
            except Exception as error:
                server_error = LiveRequestError(
                    500,
                    "server_runtime_error",
                    f"{type(error).__name__}: {error}",
                    error_type="c2kv_server_error",
                )
                self._send(500, service.error_body(server_error))
                return
            self._send(200, response)

    return ThreadingHTTPServer((host, port), Handler)


__all__ = [
    "DEFAULT_MAX_REQUEST_BYTES",
    "LIVE_PROTOCOL",
    "LiveNextCompressionService",
    "LiveRequestError",
    "ParsedNativeResponse",
    "TrainingBinding",
    "load_training_binding",
    "make_http_server",
    "parse_native_response",
]
