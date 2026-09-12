"""Bounded CPU preparation for the H0/H1/H2/H3 history variants.

H0 and H1 use the static recent-tool-one C2KV view.  H2/H3 use the
deployed A-line ``a-event-native-s0-v1`` allocation order: mandatory current
input, budgeted latest tool event, one whole-event gist reservation, lexical
raw selection, whole-event gist refill, and at most one spare raw copy.  H3
keeps H2's exact input and target while assigning positive token weights.

The module consumes normalized rows only.  It never reads a target while
selecting a memory view, never truncates a target, and rejects H1/H2/H3 as one
group whenever either representation cannot satisfy the frozen limits.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

from history_memory.dataset import Decision, iter_decisions, snapshot_and_validate_rows
from history_memory.packing import (
    EncoderChunk,
    MemoryView,
    PackedMemory,
    PackingBudgetError,
    pack_memory,
    pack_target,
    select_view,
)
from history_memory.policy import _direct_source_candidates

from .common import PreparedDecision, make_record
from .vendor.a_runtime.event_native_policy import (
    CURRENT_INPUT_BASELINE as A_CURRENT_INPUT_BASELINE,
    HISTORY_BUDGET_DEFINITION as A_HISTORY_BUDGET_DEFINITION,
    POLICY_SOURCE_COMMIT as A_POLICY_SOURCE_COMMIT,
    WORKSPACE_BUDGET_DEFINITION as A_WORKSPACE_BUDGET_DEFINITION,
)
from .vendor.a_runtime.event_native_s0_policy import EventNativeS0Controller


A_VIEW_VERSION = "a-event-native-s0-v1"
A_VIEW_SOURCE = "c2kv-a-runtime/benchmarks/memory_runtime/event_native_s0_policy.py"
A_VIEW_SOURCE_SHA256 = "fbbc086f99453184ab8630912cc453edb66a2601272d7b2ba374fa2b4b94fab7"
A_VIEW_VENDOR_SHA256 = "b58f1210d5099918e6eb89d2979178e79dd09c07393cd5949e9ced4f884e08cf"
A_VIEW_VENDOR_MANIFEST_SHA256 = "7ef3986df7f5c080d309861b0b387af3548425a879109457c7e663522e80f088"
A_VIEW_MODE = "ac_native_needs_lexical_raw_reserve_failed_operation"
HISTORY_PREPARATION_VERSION = "next-history-preparation-v1"
_VARIANTS = ("H0", "H1", "H2", "H3")
_REVISION_RE = re.compile(
    r"(?:\b(?:actually|instead|revise|revision|correction|replace|"
    r"change\s+(?:it|this|that|to|from|the\s+\w+)|correct\s+(?:it|this|that)|"
    r"ignore\s+(?:the\s+)?previous|new\s+requirement)\b|"
    r"改成|改为|更正|修订|换成|忽略之前|不要.*了)",
    re.IGNORECASE,
)
@dataclass(frozen=True)
class HistoryPreparationConfig:
    """Frozen CPU preparation and A-view limits for the 8/12 mix."""

    variants: tuple[str, ...] = _VARIANTS
    ratios: tuple[int, ...] = (8, 12)
    recent_tool_events: int = 1
    max_chunk_tokens: int = 768
    chunk_overlap: int = 64
    max_chunks: int = 48
    max_encoder_tokens: int = 36_864
    max_system_tokens: int = 8_192
    max_workspace_tokens: int = 4_096
    max_target_tokens: int = 4_096
    max_sequence_tokens: int = 16_384
    a_max_new_tokens: int = 4_096
    kv_bytes_per_token: int = 147_456
    history_budget_bytes: int = 113_246_208
    workspace_budget_bytes: int = 113_246_208
    source_index_max_events: int = 12
    predictor_prompt_token_cap: int = 2_048
    max_retrieved_events: int = 2
    workers: int = 1
    h1_require_gist: bool = True
    argument_token_weight: float = 3.0
    eos_token_weight: float = 2.0
    h0_decision_ids: frozenset[str] | None = None
    h1_decision_ids: frozenset[str] | None = None

    def __post_init__(self) -> None:
        if not self.variants or any(item not in _VARIANTS for item in self.variants):
            raise ValueError(f"variants must be drawn from {_VARIANTS!r}")
        if len(set(self.variants)) != len(self.variants):
            raise ValueError("variants must not contain duplicates")
        trio = {"H1", "H2", "H3"}
        if trio & set(self.variants) and not trio <= set(self.variants):
            raise ValueError("H1, H2, and H3 must be prepared together")
        if self.ratios != (8, 12):
            raise ValueError("This delivery is frozen to ratios (8, 12)")
        positive_ints = (
            "max_chunk_tokens",
            "max_chunks",
            "max_encoder_tokens",
            "max_system_tokens",
            "max_workspace_tokens",
            "max_target_tokens",
            "max_sequence_tokens",
            "a_max_new_tokens",
            "kv_bytes_per_token",
            "history_budget_bytes",
            "workspace_budget_bytes",
            "predictor_prompt_token_cap",
            "workers",
        )
        for name in positive_ints:
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.recent_tool_events) is not int or self.recent_tool_events < 0:
            raise ValueError("recent_tool_events must be a nonnegative integer")
        if not 0 <= self.chunk_overlap < self.max_chunk_tokens:
            raise ValueError("Require max_chunk_tokens > chunk_overlap >= 0")
        if not 0 < self.source_index_max_events <= 12:
            raise ValueError("source_index_max_events must be between one and twelve")
        if self.max_retrieved_events != 2:
            raise ValueError("The deployed A S0 controller requires max_retrieved_events=2")
        for name in ("argument_token_weight", "eos_token_weight"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class _PreparedAView:
    memory: PackedMemory
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class _Candidate:
    decision: Decision
    static_view: MemoryView
    decision_type: str
    post_tool: bool
    state_revision: bool
    history_dependency: bool


def _prepare_a_view(
    decision: Decision, tokenizer: Any, config: HistoryPreparationConfig
) -> _PreparedAView:
    """Invoke the frozen deployed A S0 controller dependency closure directly."""

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
        "source_commit": A_POLICY_SOURCE_COMMIT,
        "history_budget_definition": A_HISTORY_BUDGET_DEFINITION,
        "workspace_budget_definition": A_WORKSPACE_BUDGET_DEFINITION,
        "current_input_baseline": A_CURRENT_INPUT_BASELINE,
    }
    s0_config = {
        "source_index_max_events": config.source_index_max_events,
        "predictor_prompt_token_cap": config.predictor_prompt_token_cap,
        "predictor_completion_token_cap": 256,
        "latest_complete_tool_protection": "budgeted",
    }
    controller = EventNativeS0Controller(
        tokenizer,
        packing=packing,
        policy=policy,
        model_context=config.max_sequence_tokens,
        s0_config=s0_config,
    )
    payload = {
        "session_id": decision.store.session_id,
        "decision_key": decision.decision_id,
        "messages": [message.to_dict() for message in decision.store.messages],
        "tools": list(decision.tools),
    }
    prepared = controller.prepare(
        payload,
        ratio=config.ratios[0],
        max_new_tokens=config.a_max_new_tokens,
    )
    actual = prepared.metadata
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
    coverage = actual["source_coverage"]
    eligible_sources = coverage["eligible_source_indices"]
    unrepresented_sources = coverage["unrepresented_source_indices"] or []
    metadata = {
        "version": A_VIEW_VERSION,
        "source": A_VIEW_SOURCE,
        "source_sha256": A_VIEW_SOURCE_SHA256,
        "vendored_source_sha256": A_VIEW_VENDOR_SHA256,
        "vendor_manifest_sha256": A_VIEW_VENDOR_MANIFEST_SHA256,
        "mode": A_VIEW_MODE,
        "actual_controller_mode": actual["mode"],
        "compression_policy": actual["compression_policy"],
        "history_view_protocol": actual["history_view_protocol"],
        "allocation_enforced_ratios": list(config.ratios),
        "controller_invocation_ratio": config.ratios[0],
        "raw_source_cutoff": actual["raw_source_cutoff"],
        "common_input_source_indices": actual["common_input_source_indices"],
        "eligible_event_ids": actual["eligible_extraction"]["eligible_event_ids"],
        "eligible_chunk_count": actual["eligible_extraction"]["eligible_chunk_count"],
        "protected_event_ids": actual["protected_event_ids"],
        "requested_event_ids": actual["source_needs"]["requested_event_ids"],
        "retrieved_event_ids": actual["retrieved_event_ids"],
        "latest_complete_tool_status": actual["latest_complete_tool_protection"]["status"],
        "gist_reservation_skips": actual["gist_reservation"]["skipped_events"],
        "gist_refill_skips": actual["skipped_gist_refill_events"],
        "raw_retrieval_skips": actual["source_needs"]["skipped_for_budget"],
        "raw_reserve_status": actual["raw_reserve"]["status"],
        "raw_reserve_event_id": actual["raw_reserve"]["admitted_event_id"],
        "failed_operation_cue_status": actual["failed_operation_cue"]["status"],
        "raw_event_ids": actual["raw_event_ids"],
        "gist_event_ids": actual["gist_event_ids"],
        "raw_gist_overlap_event_ids": actual["eligible_extraction"][
            "raw_gist_overlap_event_ids"
        ],
        "omitted_event_ids": actual["omitted_event_ids"],
        "eligible_source_count": len(eligible_sources),
        "raw_eligible_source_count": len(coverage["raw_source_indices"]),
        "gist_touched_eligible_source_count": len(
            coverage["gist_touched_source_indices"]
        ),
        "covered_eligible_source_count": len(eligible_sources)
        - len(unrepresented_sources),
        "unrepresented_eligible_source_count": len(unrepresented_sources),
        "per_ratio": actual["per_ratio"],
        "actual_controller": actual,
    }
    return _PreparedAView(memory, metadata)


def _check_static(
    memory: PackedMemory,
    target_ids: Sequence[int],
    config: HistoryPreparationConfig,
) -> None:
    if len(memory.system_input_ids) > config.max_system_tokens:
        raise PackingBudgetError("Static system input exceeds its complete cap")
    if len(memory.workspace_input_ids) > config.max_workspace_tokens:
        raise PackingBudgetError("Static workspace exceeds its complete cap")
    if sum(len(chunk.token_ids) for chunk in memory.chunks) > config.max_encoder_tokens:
        raise PackingBudgetError("Static encoder input exceeds its complete cap")
    for ratio in config.ratios:
        costs = memory.costs(ratio)
        if costs["resident_kv_tokens"] + len(target_ids) > config.max_sequence_tokens:
            raise PackingBudgetError(f"Static ratio {ratio} exceeds the complete sequence cap")


def _target_type(decision: Decision, *, source_final: bool) -> str:
    target = decision.target.to_dict()
    if target.get("tool_calls"):
        return "tool_call"
    return "terminal_stop" if source_final else "non_tool_response"


def _latest_user_text(decision: Decision) -> str:
    for event in reversed(decision.store.events):
        if event.kind == "user":
            content = decision.store.event_messages(event.event_id)[0].to_dict().get("content")
            return content if isinstance(content, str) else ""
    return ""


def _candidate(
    decision: Decision,
    config: HistoryPreparationConfig,
    *,
    source_final: bool,
) -> _Candidate:
    static = select_view(decision.store, recent_tool_events=config.recent_tool_events)
    users = [event for event in decision.store.events if event.kind == "user"]
    dependency = bool(
        _direct_source_candidates(
            decision.store,
            current_user=users[-1] if users else None,
            excluded=set(static.raw_event_ids),
        )
    )
    return _Candidate(
        decision,
        static,
        _target_type(decision, source_final=source_final),
        bool(decision.prefix and decision.prefix[-1].role == "tool"),
        bool(_REVISION_RE.search(_latest_user_text(decision))),
        dependency,
    )


def _balance_h1_candidates(
    candidates: Sequence[_Candidate],
    *,
    explicit_selection: bool,
    audit: Counter[str],
) -> list[_Candidate]:
    """Downsample action types while keeping high-value history cases first."""

    counts = Counter(candidate.decision_type for candidate in candidates)
    for kind, count in counts.items():
        audit[f"trio.before_balance.type.{kind}"] = count
    if explicit_selection or len(counts) < 2:
        audit["trio.balance_bypassed_explicit_ids"] += int(explicit_selection)
        return list(candidates)
    quota = min(counts.values())

    def rank(candidate: _Candidate) -> tuple[int, int, int, str]:
        stable = hashlib.sha256(candidate.decision.decision_id.encode("utf-8")).hexdigest()
        return (
            -int(candidate.state_revision),
            -int(candidate.post_tool),
            -int(candidate.history_dependency),
            stable,
        )

    selected_ids: set[str] = set()
    for kind in sorted(counts):
        pool = sorted(
            (candidate for candidate in candidates if candidate.decision_type == kind),
            key=rank,
        )
        selected_ids.update(candidate.decision.decision_id for candidate in pool[:quota])
        audit[f"trio.balance_dropped.type.{kind}"] += len(pool) - quota
    balanced = [
        candidate
        for candidate in candidates
        if candidate.decision.decision_id in selected_ids
    ]
    audit["trio.balance_quota_per_type"] = quota
    return balanced


def _row_allows(row: Mapping[str, Any], variant: str) -> bool:
    declared = row.get("history_variants", row.get("eligible_variants"))
    if declared is not None:
        return variant in declared
    flag = row.get("h0_eligible" if variant == "H0" else "h1_eligible")
    return True if flag is None else bool(flag)


def _selected(decision_id: str, selected_ids: frozenset[str] | None) -> bool:
    return selected_ids is None or decision_id in selected_ids


def _argument_values(value: Any) -> Iterator[str]:
    if isinstance(value, Mapping):
        for child in value.values():
            yield from _argument_values(child)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for child in value:
            yield from _argument_values(child)
    elif value is not None:
        yield str(value)


def _encode_plain(tokenizer: Any, text: str) -> tuple[int, ...]:
    encode = getattr(tokenizer, "encode", None)
    if not callable(encode):
        return ()
    ids = encode(text, add_special_tokens=False)
    if hasattr(ids, "input_ids"):
        ids = ids.input_ids
    return tuple(int(token) for token in ids)


def _find_subsequence(sequence: Sequence[int], pattern: Sequence[int]) -> Iterator[int]:
    if not pattern or len(pattern) > len(sequence):
        return
    for start in range(len(sequence) - len(pattern) + 1):
        if tuple(sequence[start : start + len(pattern)]) == tuple(pattern):
            yield start


def _h3_weights(
    decision: Decision,
    target_ids: tuple[int, ...],
    tokenizer: Any,
    config: HistoryPreparationConfig,
) -> tuple[tuple[float, ...], Mapping[str, int]]:
    weights = [1.0] * len(target_ids)
    values: set[str] = set()
    for call in decision.target.to_dict().get("tool_calls") or ():
        arguments = call.get("function", {}).get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except ValueError:
                pass
        values.update(_argument_values(arguments))
    weighted_positions: set[int] = set()
    matched_values = 0
    for value in sorted(values):
        patterns = {
            _encode_plain(tokenizer, value),
            _encode_plain(tokenizer, json.dumps(value, ensure_ascii=False)),
        }
        value_matched = False
        for pattern in patterns:
            for start in _find_subsequence(target_ids, pattern):
                value_matched = True
                for position in range(start, start + len(pattern)):
                    weights[position] = max(weights[position], config.argument_token_weight)
                    weighted_positions.add(position)
        matched_values += int(value_matched)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    eos_positions = (
        [position for position, token_id in enumerate(target_ids) if token_id == eos_token_id]
        if type(eos_token_id) is int
        else []
    )
    for position in eos_positions:
        weights[position] = max(weights[position], config.eos_token_weight)
    if any(weight <= 0 or not math.isfinite(weight) for weight in weights):
        raise AssertionError("H3 must preserve positive full-token CE")
    return tuple(weights), {
        "argument_value_count": len(values),
        "matched_argument_value_count": matched_values,
        "argument_weighted_token_count": len(weighted_positions),
        "eos_token_id": eos_token_id,
        "eos_weighted_token_count": len(eos_positions),
        "positive_full_ce_token_count": len(weights),
    }


def _emit_record(
    variant: str,
    candidate: _Candidate,
    memory: PackedMemory,
    target_ids: tuple[int, ...],
    ratio: int,
    weight: float,
    metadata: Mapping[str, Any],
    tokenizer: Any,
    config: HistoryPreparationConfig,
    make_record_fn: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    target_weights = None
    final_metadata = dict(metadata)
    if variant == "H3":
        target_weights, weight_metadata = _h3_weights(
            candidate.decision, target_ids, tokenizer, config
        )
        final_metadata["target_weighting"] = dict(weight_metadata)
    prepared = PreparedDecision(
        memory=memory,
        target_ids=target_ids,
        ratio=ratio,
        weight=weight,
        decision_id=candidate.decision.decision_id,
        target_weights=target_weights,
    )
    return make_record_fn(
        prepared,
        session_key=candidate.decision.store.session_id,
        source=candidate.decision.source,
        split=candidate.decision.split,
        metadata=final_metadata,
        target_weights=target_weights,
    )


def _attempt_h0(
    item: _Candidate,
    tokenizer: Any,
    config: HistoryPreparationConfig,
) -> tuple[_Candidate, tuple[int, ...] | None, PackedMemory | None, str | None]:
    try:
        target_ids = pack_target(
            tokenizer, item.decision.target, max_target_tokens=config.max_target_tokens
        )
        memory = pack_memory(
            item.decision.store,
            item.static_view,
            tokenizer,
            tools=item.decision.tools or None,
            max_chunk_tokens=config.max_chunk_tokens,
            chunk_overlap=config.chunk_overlap,
            max_chunks=config.max_chunks,
        )
        _check_static(memory, target_ids, config)
        return item, target_ids, memory, None
    except (PackingBudgetError, ValueError) as error:
        return item, None, None, type(error).__name__


def _attempt_trio(
    item: _Candidate,
    tokenizer: Any,
    config: HistoryPreparationConfig,
) -> tuple[
    _Candidate,
    tuple[int, ...] | None,
    PackedMemory | None,
    _PreparedAView | None,
    str | None,
]:
    try:
        target_ids = pack_target(
            tokenizer, item.decision.target, max_target_tokens=config.max_target_tokens
        )
        h1_memory = pack_memory(
            item.decision.store,
            item.static_view,
            tokenizer,
            tools=item.decision.tools or None,
            max_chunk_tokens=config.max_chunk_tokens,
            chunk_overlap=config.chunk_overlap,
            max_chunks=config.max_chunks,
        )
        _check_static(h1_memory, target_ids, config)
        h2 = _prepare_a_view(item.decision, tokenizer, config)
        return item, target_ids, h1_memory, h2, None
    except (PackingBudgetError, ValueError) as error:
        return item, None, None, None, type(error).__name__


def _ordered_attempts(
    items: Sequence[_Candidate],
    attempt: Callable[[_Candidate, Any, HistoryPreparationConfig], Any],
    tokenizer: Any,
    config: HistoryPreparationConfig,
) -> Iterator[Any]:
    if config.workers == 1:
        for item in items:
            yield attempt(item, tokenizer, config)
        return
    with ThreadPoolExecutor(max_workers=config.workers) as executor:
        yield from executor.map(
            lambda item: attempt(item, tokenizer, config),
            items,
        )


def iter_history_records(
    rows: Iterable[Mapping[str, Any]],
    tokenizer: Any,
    config: HistoryPreparationConfig | None = None,
    *,
    make_record_fn: Callable[..., dict[str, Any]] | None = None,
) -> tuple[Iterator[tuple[str, dict[str, Any]]], Counter[str]]:
    """Return a streaming record iterator and an audit filled on exhaustion."""

    config = config or HistoryPreparationConfig()
    if not isinstance(config, HistoryPreparationConfig):
        raise TypeError("config must be HistoryPreparationConfig")
    if not callable(getattr(tokenizer, "apply_chat_template", None)):
        raise TypeError("tokenizer must expose apply_chat_template")
    writer = make_record if make_record_fn is None else make_record_fn
    if not callable(writer):
        raise TypeError("make_record_fn must be callable")
    snapshots = snapshot_and_validate_rows(rows)
    audit: Counter[str] = Counter()

    def generate() -> Iterator[tuple[str, dict[str, Any]]]:
        candidates: list[_Candidate] = []
        h0_candidates: list[_Candidate] = []
        for row in snapshots:
            audit["sessions.visited"] += 1
            if row["split"] != "train":
                audit["sessions.skipped_nontrain"] += 1
                continue
            row_decisions = tuple(iter_decisions(row))
            for decision_index, decision in enumerate(row_decisions):
                audit["decisions.visited"] += 1
                item = _candidate(
                    decision,
                    config,
                    source_final=decision_index == len(row_decisions) - 1,
                )
                audit[f"decisions.type.{item.decision_type}"] += 1
                audit[f"decisions.source.{decision.source}"] += 1
                audit["decisions.post_tool"] += int(item.post_tool)
                audit["decisions.state_revision"] += int(item.state_revision)
                audit["decisions.history_dependency"] += int(item.history_dependency)
                if (
                    "H0" in config.variants
                    and _row_allows(row, "H0")
                    and _selected(decision.decision_id, config.h0_decision_ids)
                ):
                    h0_candidates.append(item)
                if (
                    "H1" in config.variants
                    and _row_allows(row, "H1")
                    and _selected(decision.decision_id, config.h1_decision_ids)
                ):
                    if config.h1_require_gist and not item.static_view.gist_event_ids:
                        audit["trio.skipped_no_gist"] += 1
                    else:
                        candidates.append(item)

        candidates = _balance_h1_candidates(
            candidates,
            explicit_selection=config.h1_decision_ids is not None,
            audit=audit,
        )

        audit["workers"] = config.workers
        for item, target_ids, memory, error_type in _ordered_attempts(
            h0_candidates, _attempt_h0, tokenizer, config
        ):
            decision = item.decision
            if error_type is not None:
                audit[f"H0.skipped.{error_type}"] += 1
                continue
            assert target_ids is not None and memory is not None
            metadata = {
                "preparation_version": HISTORY_PREPARATION_VERSION,
                "variant": "H0",
                "view_policy": "static-recent-tool-one",
                "decision_type": item.decision_type,
                "post_tool": item.post_tool,
                "state_revision": item.state_revision,
                "history_dependency": item.history_dependency,
            }
            for ratio in config.ratios:
                record = _emit_record(
                    "H0",
                    item,
                    memory,
                    target_ids,
                    ratio,
                    1.0,
                    metadata,
                    tokenizer,
                    config,
                    writer,
                )
                audit["H0.records"] += 1
                audit[f"H0.records.ratio.{ratio}"] += 1
                audit[f"H0.records.source.{decision.source}"] += 1
                audit["H0.gist_bearing_records"] += int(bool(memory.chunks))
                audit["H0.gist_tokens"] += memory.costs(ratio)["gist_tokens"]
                yield "H0", record

        for item, target_ids, h1_memory, h2, error_type in _ordered_attempts(
            candidates, _attempt_trio, tokenizer, config
        ):
            decision = item.decision
            if error_type is not None:
                audit[f"trio.skipped.{error_type}"] += 1
                audit[f"trio.skipped.source.{decision.source}"] += 1
                audit[f"trio.skipped.type.{item.decision_type}"] += 1
                continue
            assert target_ids is not None and h1_memory is not None and h2 is not None
            common_metadata = {
                "preparation_version": HISTORY_PREPARATION_VERSION,
                "decision_type": item.decision_type,
                "post_tool": item.post_tool,
                "state_revision": item.state_revision,
                "history_dependency": item.history_dependency,
                "selection_profile": "gist-bearing-action-balanced-history-v1",
            }
            changed = (
                item.static_view.gist_event_ids != h2.memory.view.gist_event_ids
                or item.static_view.raw_event_ids != h2.memory.view.raw_event_ids
            )
            audit["trio.decisions"] += 1
            audit["trio.view_changed"] += int(changed)
            audit[f"trio.type.{item.decision_type}"] += 1
            audit[f"trio.source.{decision.source}"] += 1
            audit["trio.post_tool"] += int(item.post_tool)
            audit["trio.state_revision"] += int(item.state_revision)
            audit["trio.history_dependency"] += int(item.history_dependency)
            audit["trio.a_omitted_events"] += len(h2.metadata["omitted_event_ids"])
            audit["trio.a_raw_gist_overlap_events"] += len(
                h2.metadata["raw_gist_overlap_event_ids"]
            )
            audit["trio.a_eligible_sources"] += h2.metadata["eligible_source_count"]
            audit["trio.a_covered_sources"] += h2.metadata[
                "covered_eligible_source_count"
            ]
            audit["trio.a_unrepresented_sources"] += h2.metadata[
                "unrepresented_eligible_source_count"
            ]
            audit["trio.a_complete_coverage_decisions"] += int(
                h2.metadata["unrepresented_eligible_source_count"] == 0
            )
            for ratio in config.ratios:
                variants = (
                    (
                        "H1",
                        h1_memory,
                        {
                            **common_metadata,
                            "variant": "H1",
                            "view_policy": "static-recent-tool-one",
                            "view_changed_from_h1_to_h2": changed,
                        },
                    ),
                    (
                        "H2",
                        h2.memory,
                        {
                            **common_metadata,
                            "variant": "H2",
                            "view_policy": A_VIEW_VERSION,
                            "view_changed_from_h1_to_h2": changed,
                            "a_view": dict(h2.metadata),
                        },
                    ),
                    (
                        "H3",
                        h2.memory,
                        {
                            **common_metadata,
                            "variant": "H3",
                            "view_policy": A_VIEW_VERSION,
                            "view_changed_from_h1_to_h2": changed,
                            "a_view": dict(h2.metadata),
                        },
                    ),
                )
                for variant, memory, metadata in variants:
                    record = _emit_record(
                        variant,
                        item,
                        memory,
                        target_ids,
                        ratio,
                        1.0,
                        metadata,
                        tokenizer,
                        config,
                        writer,
                    )
                    audit[f"{variant}.records"] += 1
                    audit[f"{variant}.records.ratio.{ratio}"] += 1
                    audit[f"{variant}.records.source.{decision.source}"] += 1
                    audit[f"{variant}.records.type.{item.decision_type}"] += 1
                    audit[f"{variant}.gist_bearing_records"] += int(bool(memory.chunks))
                    audit[f"{variant}.gist_tokens"] += memory.costs(ratio)["gist_tokens"]
                    yield variant, record
        audit["complete"] = 1

    return generate(), audit


def prepare_history(
    rows: Iterable[Mapping[str, Any]],
    tokenizer: Any,
    config: HistoryPreparationConfig | None = None,
    *,
    make_record_fn: Callable[..., dict[str, Any]] | None = None,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, int]]:
    """Materialize the bounded iterator for tests and small CPU preparations."""

    resolved = config or HistoryPreparationConfig()
    records = {variant: [] for variant in resolved.variants}
    iterator, audit = iter_history_records(
        rows, tokenizer, resolved, make_record_fn=make_record_fn
    )
    for variant, record in iterator:
        records[variant].append(record)
    return records, dict(audit)


__all__ = [
    "A_VIEW_VERSION",
    "HISTORY_PREPARATION_VERSION",
    "HistoryPreparationConfig",
    "iter_history_records",
    "prepare_history",
]
