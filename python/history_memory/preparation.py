"""Deterministic CPU preparation and loading of matched C/B decisions.

The prepared corpus stores canonical sessions once and keeps one compact row
per ratio/repetition pair.  Loading reconstructs the observable prefix and
packs the selected arm with the tokenizer whose behavior was fingerprinted in
the manifest.  Neither preparation nor loading requires model weights.
"""

from __future__ import annotations

import hashlib
import json
import math
import multiprocessing
import os
import shutil
from collections import Counter, OrderedDict, defaultdict, deque
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

from .dataset import (
    Decision,
    event_store_session_id,
    iter_decision_metadata,
    iter_decisions,
    snapshot_and_validate_rows,
)
from .evidence import EVIDENCE_VERSION
from .events import EventStore
from .packing import (
    PACKING_VERSION,
    RAW_LAYOUT_PROFILE,
    MemoryView,
    PackedMemory,
    PackingCache,
    PackingBudgetError,
    encode_event_chunks,
    native_ids,
    pack_memory,
    pack_target,
    raw_workspace_messages,
    select_view,
    visible_message,
)
from .policy import BudgetExceeded, ConversationMemory, RuntimeConfig


PREPARED_SCHEMA_VERSION = "history-memory-paired-v1"
POLICY_SOURCE_COMMIT = "affe0e3bd29cce06beadd5a67b1e629f8ca77022"
SESSIONS_FILE = "sessions.jsonl"
DECISIONS_FILE = "paired_decisions.jsonl"
MANIFEST_FILE = "manifest.json"


@dataclass(frozen=True)
class PackingConfig:
    """Finite all-or-skip limits shared by C and B."""

    ratios: tuple[int, ...] = (4, 8)
    recent_tool_events: int = 1
    max_chunk_tokens: int = 768
    chunk_overlap: int = 64
    max_chunks: int = 48
    max_encoder_tokens: int = 36_864
    max_system_tokens: int = 8_192
    max_workspace_tokens: int = 4_096
    max_target_tokens: int = 4_096
    max_sequence_tokens: int = 16_384

    def __post_init__(self) -> None:
        if not self.ratios or any(
            isinstance(ratio, bool) or not isinstance(ratio, int) or ratio < 1
            for ratio in self.ratios
        ):
            raise ValueError("ratios must contain positive integers")
        if len(set(self.ratios)) != len(self.ratios):
            raise ValueError("ratios must not contain duplicates")
        for field in (
            "max_chunk_tokens",
            "max_chunks",
            "max_encoder_tokens",
            "max_system_tokens",
            "max_workspace_tokens",
            "max_target_tokens",
            "max_sequence_tokens",
        ):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{field} must be a positive integer")
        if not 0 <= self.chunk_overlap < self.max_chunk_tokens:
            raise ValueError("Require max_chunk_tokens > chunk_overlap >= 0")
        if isinstance(self.recent_tool_events, bool) or self.recent_tool_events < 0:
            raise ValueError("recent_tool_events must be a nonnegative integer")


@dataclass(frozen=True)
class SamplingConfig:
    """Finite bounds applied before ratio/repetition expansion."""

    max_rows_per_source: int = 50_000
    max_sessions: int = 50_000
    max_decisions_per_session: int = 64
    max_total_decisions: int = 100_000
    max_presented_tokens_per_arm: int = 48_000_000
    qa_target_fraction: float = 0.15
    repetitions: int = 1
    seed: int = 42

    def __post_init__(self) -> None:
        for field in (
            "max_rows_per_source",
            "max_sessions",
            "max_decisions_per_session",
            "max_total_decisions",
            "max_presented_tokens_per_arm",
            "repetitions",
        ):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{field} must be a positive integer")
        if not 0.0 <= self.qa_target_fraction < 1.0:
            raise ValueError("qa_target_fraction must be in [0, 1)")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")


@dataclass(frozen=True)
class PreparedRecord:
    memory: PackedMemory
    target_ids: tuple[int, ...]
    ratio: int
    weight: float
    repetition_index: int
    decision_id: str
    session_key: str
    decision_index: int
    source: str
    split: str


@dataclass(frozen=True)
class _DecisionCandidate:
    decision_id: str
    session_key: str
    source: str
    source_message_index: int


@dataclass(frozen=True)
class _PlannedDecision:
    decision_ordinal: int
    session_key: str
    decision_id: str
    decision_index: int
    source_message_index: int
    source: str
    split: str
    target_json_text: str
    c_view: MemoryView
    b_view: MemoryView
    c_costs: Mapping[str, Mapping[str, Any]]
    b_costs: Mapping[str, Mapping[str, Any]]
    policy_metadata: Mapping[str, Any]


@dataclass(frozen=True)
class _SessionPlan:
    decisions_visited: int
    candidates: tuple[_PlannedDecision, ...]
    audit: Mapping[str, int]


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def _json_safe(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))
    except (TypeError, ValueError):
        return repr(value)


def tokenizer_identity(tokenizer: Any) -> dict[str, Any]:
    """Fingerprint tokenizer behavior without binding preparation to a path."""
    attributes = {
        "class": f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}",
        "vocab_size": getattr(tokenizer, "vocab_size", None),
        "special_tokens_map": _json_safe(getattr(tokenizer, "special_tokens_map", {})),
        "chat_template": getattr(tokenizer, "chat_template", None),
    }
    probes: list[list[int]] = []
    for messages, tools, generation in (
        ([{"role": "user", "content": "history-memory tokenizer probe"}], None, False),
        ([{"role": "system", "content": "probe"}, {"role": "user", "content": "go"}], None, True),
        (
            [{"role": "user", "content": "call"}],
            [{"type": "function", "function": {"name": "probe", "description": "probe", "parameters": {"type": "object", "properties": {}}}}],
            True,
        ),
    ):
        probes.append(list(native_ids(tokenizer, messages, tools=tools, generation=generation)))
    payload = {"attributes": attributes, "probe_ids": probes}
    return {
        "sha256": hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest(),
        **attributes,
    }


def _view_dict(view: MemoryView) -> dict[str, Any]:
    return {
        "gist_event_ids": list(view.gist_event_ids),
        "raw_event_ids": list(view.raw_event_ids),
        "evidence_event_ids": list(view.evidence_event_ids),
    }


def _view_from_dict(value: Mapping[str, Any]) -> MemoryView:
    if not isinstance(value, Mapping):
        raise ValueError("Prepared memory view must be a mapping")
    return MemoryView(
        tuple(value.get("gist_event_ids", ())),
        tuple(value.get("raw_event_ids", ())),
        tuple(value.get("evidence_event_ids", ())),
    )


def _stable_rank(text: str, seed: int) -> tuple[str, str]:
    digest = hashlib.sha256(f"{seed}\0{text}".encode("utf-8")).hexdigest()
    return digest, text


def _bounded_rows(
    rows: Sequence[Mapping[str, Any]], sampling: SamplingConfig, audit: Counter[str]
) -> tuple[Mapping[str, Any], ...]:
    by_source: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
    seen_sessions: set[str] = set()
    for row in rows:
        if row["split"] != "train":
            audit["sessions_skipped_nontrain_before_expansion"] += 1
            continue
        source = row["source"]
        session_key = event_store_session_id(source, row["session_id"])
        if session_key in seen_sessions:
            raise ValueError(f"Duplicate canonical session identity: {session_key}")
        seen_sessions.add(session_key)
        by_source[source].append(row)
    bounded_by_source: list[Mapping[str, Any]] = []
    for source in sorted(by_source):
        source_rows = sorted(
            by_source[source],
            key=lambda row: _stable_rank(
                event_store_session_id(row["source"], row["session_id"]), sampling.seed
            ),
        )
        chosen = source_rows[: sampling.max_rows_per_source]
        audit["sessions_skipped_max_rows_per_source"] += len(source_rows) - len(chosen)
        bounded_by_source.extend(chosen)
    ranked = sorted(
        bounded_by_source,
        key=lambda row: _stable_rank(
            event_store_session_id(row["source"], row["session_id"]), sampling.seed
        ),
    )
    result = ranked[: sampling.max_sessions]
    audit["sessions_skipped_max_sessions"] += len(ranked) - len(result)
    audit["sessions_after_bounds"] = len(result)
    for row in result:
        audit[f"sessions_after_bounds.source.{row['source']}"] += 1
    return tuple(result)


def _select_decision_ids(
    decisions_by_session: Sequence[Sequence[_DecisionCandidate]],
    sampling: SamplingConfig,
    audit: Counter[str],
) -> frozenset[str]:
    candidates: list[_DecisionCandidate] = []
    for decisions in decisions_by_session:
        ranked = sorted(
            decisions,
            key=lambda decision: _stable_rank(decision.decision_id, sampling.seed),
        )
        chosen = ranked[: sampling.max_decisions_per_session]
        audit["decisions_skipped_max_per_session"] += len(ranked) - len(chosen)
        candidates.extend(chosen)

    nonqa = [decision for decision in candidates if not decision.source.startswith("qa:")]
    qa = [decision for decision in candidates if decision.source.startswith("qa:")]
    nonqa.sort(key=lambda decision: _stable_rank(decision.decision_id, sampling.seed))
    qa.sort(key=lambda decision: _stable_rank(decision.decision_id, sampling.seed))

    if nonqa:
        qa_for_all_nonqa = round(
            len(nonqa) * sampling.qa_target_fraction / (1.0 - sampling.qa_target_fraction)
        )
        qa_limit = min(len(qa), qa_for_all_nonqa)
        nonqa_limit = len(nonqa)
    else:
        # A QA-only source build remains useful, while the manifest records
        # that the requested mixture could not be achieved.
        qa_limit = len(qa)
        nonqa_limit = 0
        if qa:
            audit["qa_fraction_unachievable_without_nonqa"] += 1

    total_limit = sampling.max_total_decisions
    if qa_limit + nonqa_limit > total_limit:
        desired_qa = round(total_limit * sampling.qa_target_fraction)
        qa_limit = min(qa_limit, desired_qa)
        nonqa_limit = min(nonqa_limit, total_limit - qa_limit)
        remaining = total_limit - qa_limit - nonqa_limit
        if remaining:
            add_nonqa = min(remaining, len(nonqa) - nonqa_limit)
            nonqa_limit += add_nonqa
            remaining -= add_nonqa
        if remaining:
            qa_limit += min(remaining, len(qa) - qa_limit)

    selected = nonqa[:nonqa_limit] + qa[:qa_limit]
    audit["candidate_nonqa_decisions"] = len(nonqa)
    audit["candidate_qa_decisions"] = len(qa)
    audit["decisions_selected_before_packing"] = len(selected)
    audit["qa_decisions_selected_before_packing"] = qa_limit
    audit["qa_decisions_dropped_for_mix_or_total_cap"] = len(qa) - qa_limit
    audit["nonqa_decisions_dropped_for_total_cap"] = len(nonqa) - nonqa_limit
    return frozenset(decision.decision_id for decision in selected)


def _history_components(
    store: EventStore,
    view: MemoryView,
    tokenizer: Any,
    *,
    ratio: int,
    max_chunk_tokens: int,
    chunk_overlap: int,
    tools: Sequence[Mapping[str, Any]],
    cache: PackingCache | None = None,
) -> tuple[int, int]:
    """Return gist and charged raw tokens under the fixed current-input baseline.

    The free baseline contains source system messages, the latest user source
    message, and the last visible source message, deduplicated in source order.
    Subtracting that exact native rendering from the full raw view keeps every
    other raw/evidence token inside H, including extra raw events restored
    during a multi-tool turn.
    """
    full_messages = raw_workspace_messages(store, view)
    full_tokens = len(
        native_ids(tokenizer, full_messages, tools=tools, generation=True, cache=cache)
    )
    baseline_indices = {
        index
        for event in store.events
        if event.kind == "instruction"
        for index in event.source_indices
    }
    users = [event for event in store.events if event.kind == "user"]
    if users:
        baseline_indices.add(max(users[-1].source_indices))
    if store.messages:
        baseline_indices.add(len(store.messages) - 1)
    baseline_messages = [
        visible_message(store.messages[index]) for index in sorted(baseline_indices)
    ]
    baseline_tokens = (
        len(native_ids(tokenizer, baseline_messages, tools=tools, generation=True, cache=cache))
        if baseline_messages
        else 0
    )
    raw_tokens = max(0, full_tokens - baseline_tokens)
    gist_tokens = 0
    for event in store.events:
        if event.event_id not in view.gist_event_ids:
            continue
        for chunk in encode_event_chunks(
            store,
            event.event_id,
            tokenizer,
            max_chunk_tokens=max_chunk_tokens,
            chunk_overlap=chunk_overlap,
            cache=cache,
        ):
            gist_tokens += math.ceil(len(chunk.token_ids) / ratio)
    return gist_tokens, raw_tokens


class _DecisionMeasurements:
    """Reuse exact measurements within a decision and tokenization within a session.

    Native prompt lengths are measured as complete rendered sequences. They
    cannot be obtained by summing per-message lengths because evidence packets
    and chat-template tokenization are not additive.
    """

    def __init__(self, decision: Decision, tokenizer: Any, packing: PackingConfig,
                 cache: PackingCache) -> None:
        self.decision = decision
        self.tokenizer = tokenizer
        self.packing = packing
        self.cache = cache
        self.tools = decision.tools
        self._raw_lengths: dict[MemoryView, int] = {}
        self._history: dict[MemoryView, tuple[int, dict[int, int]]] = {}
        self._baseline_tokens: int | None = None
        self._packed: dict[MemoryView, tuple[PackedMemory, tuple[int, ...], dict[str, Any]]] = {}

    def raw_tokens(self, view: MemoryView) -> int:
        if view not in self._raw_lengths:
            self._raw_lengths[view] = len(native_ids(
                self.tokenizer, raw_workspace_messages(self.decision.store, view),
                tools=self.tools, generation=True, cache=self.cache,
            ))
        return self._raw_lengths[view]

    def history_components(self, view: MemoryView, ratio: int) -> tuple[int, int]:
        if view not in self._history:
            store = self.decision.store
            full_tokens = self.raw_tokens(view)
            if self._baseline_tokens is None:
                indices = {
                    index for event in store.events if event.kind == "instruction"
                    for index in event.source_indices
                }
                users = [event for event in store.events if event.kind == "user"]
                if users:
                    indices.add(max(users[-1].source_indices))
                if store.messages:
                    indices.add(len(store.messages) - 1)
                messages = [visible_message(store.messages[index]) for index in sorted(indices)]
                self._baseline_tokens = len(native_ids(
                    self.tokenizer, messages, tools=self.tools, generation=True, cache=self.cache,
                )) if messages else 0
            raw_tokens = max(0, full_tokens - self._baseline_tokens)
            gist_ids = set(view.gist_event_ids)
            lengths = tuple(
                len(chunk.token_ids)
                for event in store.events if event.event_id in gist_ids
                for chunk in encode_event_chunks(
                    store, event.event_id, self.tokenizer,
                    max_chunk_tokens=self.packing.max_chunk_tokens,
                    chunk_overlap=self.packing.chunk_overlap, cache=self.cache,
                )
            )
            self._history[view] = (raw_tokens, {
                value: sum(math.ceil(length / value) for length in lengths)
                for value in self.packing.ratios
            })
        raw_tokens, gist_by_ratio = self._history[view]
        return gist_by_ratio[ratio], raw_tokens


def _evidence_increment_tokens(
    store: EventStore,
    base_view: MemoryView,
    candidate_view: MemoryView,
    tokenizer: Any,
    tools: Sequence[Mapping[str, Any]],
    measurements: _DecisionMeasurements | None = None,
) -> int:
    if measurements is not None:
        return max(0, measurements.raw_tokens(candidate_view) - measurements.raw_tokens(base_view))
    base = native_ids(
        tokenizer,
        raw_workspace_messages(store, base_view),
        tools=tools,
        generation=True,
    )
    candidate = native_ids(
        tokenizer,
        raw_workspace_messages(store, candidate_view),
        tools=tools,
        generation=True,
    )
    return max(0, len(candidate) - len(base))


def _pack_checked(
    decision: Decision,
    view: MemoryView,
    tokenizer: Any,
    config: PackingConfig,
    measurements: _DecisionMeasurements | None = None,
) -> tuple[PackedMemory, tuple[int, ...], dict[str, Any]]:
    if measurements is not None and view in measurements._packed:
        memory, target_ids, costs = measurements._packed[view]
        return memory, target_ids, {key: dict(value) for key, value in costs.items()}
    cache = measurements.cache if measurements is not None else None
    memory = pack_memory(
        decision.store,
        view,
        tokenizer,
        tools=measurements.tools if measurements is not None else decision.tools,
        max_chunk_tokens=config.max_chunk_tokens,
        chunk_overlap=config.chunk_overlap,
        max_chunks=config.max_chunks,
        max_raw_tokens=config.max_system_tokens + config.max_workspace_tokens,
        cache=cache,
    )
    if len(memory.system_input_ids) > config.max_system_tokens:
        raise PackingBudgetError(
            f"System/tools need {len(memory.system_input_ids)} tokens; budget is {config.max_system_tokens}"
        )
    if len(memory.workspace_input_ids) > config.max_workspace_tokens:
        raise PackingBudgetError(
            f"Workspace needs {len(memory.workspace_input_ids)} tokens; budget is {config.max_workspace_tokens}"
        )
    encoder_tokens = sum(len(chunk.token_ids) for chunk in memory.chunks)
    if encoder_tokens > config.max_encoder_tokens:
        raise PackingBudgetError(
            f"Encoder chunks need {encoder_tokens} tokens; budget is {config.max_encoder_tokens}"
        )
    target_ids = pack_target(
        tokenizer, decision.target, max_target_tokens=config.max_target_tokens, cache=cache,
    )
    costs_by_ratio: dict[str, Any] = {}
    for ratio in config.ratios:
        costs = memory.costs(ratio)
        sequence_tokens = costs["resident_kv_tokens"] + len(target_ids)
        if sequence_tokens > config.max_sequence_tokens:
            raise PackingBudgetError(
                f"Resident memory plus target need {sequence_tokens} tokens at ratio {ratio}; "
                f"budget is {config.max_sequence_tokens}"
            )
        costs_by_ratio[str(ratio)] = {
            **costs,
            "target_tokens": len(target_ids),
            "sequence_tokens": sequence_tokens,
        }
    if measurements is not None:
        measurements._packed[view] = (memory, target_ids, {
            key: dict(value) for key, value in costs_by_ratio.items()
        })
    return memory, target_ids, costs_by_ratio


def _enforce_history_budget(
    decision: Decision,
    view: MemoryView,
    tokenizer: Any,
    packing: PackingConfig,
    costs_by_ratio: Mapping[str, dict[str, Any]],
    *,
    kv_bytes_per_token: int,
    history_budget_bytes: int,
    measurements: _DecisionMeasurements | None = None,
) -> None:
    for ratio in packing.ratios:
        if measurements is not None:
            gist_tokens, historical_raw_tokens = measurements.history_components(view, ratio)
        else:
            gist_tokens, historical_raw_tokens = _history_components(
                decision.store,
                view,
                tokenizer,
                ratio=ratio,
                max_chunk_tokens=packing.max_chunk_tokens,
                chunk_overlap=packing.chunk_overlap,
                tools=decision.tools,
            )
        history_tokens = gist_tokens + historical_raw_tokens
        history_bytes = history_tokens * kv_bytes_per_token
        costs_by_ratio[str(ratio)].update(
            history_gist_tokens=gist_tokens,
            history_raw_tokens=historical_raw_tokens,
            history_total_tokens=history_tokens,
            history_bytes=history_bytes,
        )
        if history_bytes > history_budget_bytes:
            raise PackingBudgetError(
                f"Historical gist plus raw need {history_bytes} bytes at ratio {ratio}; "
                f"budget is {history_budget_bytes}"
            )


class PrefixLifecyclePlanner:
    """Adapt the fixed A policy to token-exact B/W preparation budgets."""

    def __init__(
        self,
        tokenizer: Any,
        packing: PackingConfig,
        policy_config: RuntimeConfig,
        *,
        kv_bytes_per_token: int,
    ) -> None:
        if isinstance(kv_bytes_per_token, bool) or kv_bytes_per_token < 1:
            raise ValueError("kv_bytes_per_token must be a positive integer")
        self.tokenizer = tokenizer
        self.packing = packing
        self.policy_config = policy_config
        self.kv_bytes_per_token = kv_bytes_per_token
        self._memories: dict[str, ConversationMemory] = {}
        self._packing_cache: PackingCache | None = None

    def measurements(self, decision: Decision) -> _DecisionMeasurements:
        """Keep event tokenization between prefixes, without caching policy state globally."""
        if self._packing_cache is None or self._packing_cache.session_id != decision.store.session_id:
            self._packing_cache = PackingCache(decision.store.session_id)
        return _DecisionMeasurements(decision, self.tokenizer, self.packing, self._packing_cache)

    def __call__(
        self, decision: Decision, static_view: MemoryView,
        *, measurements: _DecisionMeasurements | None = None,
    ) -> tuple[MemoryView, Mapping[str, Any]]:
        store = decision.store
        measurements = measurements or self.measurements(decision)
        memory = self._memories.setdefault(
            store.session_id, ConversationMemory(store.session_id, self.policy_config)
        )
        planning_ratio = min(self.packing.ratios)
        subcap = min(
            self.policy_config.history_budget_bytes,
            self.policy_config.workspace_budget_bytes,
        )
        cache: dict[tuple[str, ...], tuple[int, int, int, int]] = {}

        def measured(event_ids: tuple[str, ...]) -> tuple[int, int, int, int]:
            requested_ids = set(event_ids)
            ordered = tuple(
                event.event_id for event in store.events if event.event_id in requested_ids
            )
            if ordered not in cache:
                view = select_view(
                    store,
                    recent_tool_events=self.packing.recent_tool_events,
                    restored_event_ids=ordered,
                )
                evidence_tokens = _evidence_increment_tokens(
                    store, static_view, view, self.tokenizer, measurements.tools,
                    measurements=measurements,
                )
                gist_tokens, historical_raw_tokens = measurements.history_components(view, planning_ratio)
                evidence_bytes = evidence_tokens * self.kv_bytes_per_token
                history_bytes = (
                    gist_tokens + historical_raw_tokens
                ) * self.kv_bytes_per_token
                # A single scalar threshold enforces both independent limits:
                # evidence <= W and complete historical memory <= B.
                dual_cost = max(
                    evidence_bytes,
                    max(
                        0,
                        history_bytes
                        - self.policy_config.history_budget_bytes
                        + subcap,
                    ),
                )
                cache[ordered] = (
                    dual_cost,
                    evidence_bytes,
                    history_bytes,
                    evidence_tokens,
                )
            return cache[ordered]

        selection = memory.prepare(
            store,
            lambda event_ids: measured(event_ids)[0],
            set(static_view.raw_event_ids),
            decision.decision_id,
        )
        view = select_view(
            store,
            recent_tool_events=self.packing.recent_tool_events,
            restored_event_ids=selection.selected_event_ids,
        )
        dual_cost, evidence_bytes, history_bytes, evidence_tokens = measured(
            selection.selected_event_ids
        )
        if evidence_bytes > self.policy_config.workspace_budget_bytes:
            raise BudgetExceeded(
                required_event_ids=selection.selected_event_ids,
                required_cost=evidence_bytes,
                budget=self.policy_config.workspace_budget_bytes,
            )
        if history_bytes > self.policy_config.history_budget_bytes:
            raise BudgetExceeded(
                required_event_ids=selection.selected_event_ids,
                required_cost=history_bytes,
                budget=self.policy_config.history_budget_bytes,
            )
        metadata = {
            **dict(selection.metadata),
            "planning_ratio": planning_ratio,
            "kv_bytes_per_token": self.kv_bytes_per_token,
            "evidence_tokens": evidence_tokens,
            "evidence_bytes": evidence_bytes,
            "history_bytes": history_bytes,
            "dual_constraint_cost_bytes": dual_cost,
            "protected_event_ids": list(selection.protected_event_ids),
            "retrieved_event_ids": list(selection.retrieved_event_ids),
            "retained_event_ids": list(selection.retained_event_ids),
            "selected_event_ids": list(selection.selected_event_ids),
        }
        return view, metadata


def _packing_manifest(config: PackingConfig) -> dict[str, Any]:
    value = asdict(config)
    value["ratios"] = list(config.ratios)
    return value


def _plan_decision(
    decision: Decision,
    decision_ordinal: int,
    planner: PrefixLifecyclePlanner,
    tokenizer: Any,
    packing: PackingConfig,
    policy_config: RuntimeConfig,
    selected_ids: frozenset[str],
    audit: Counter[str],
    *,
    kv_bytes_per_token: int,
) -> _PlannedDecision | None:
    """Advance one session-local policy decision and pack selected candidates."""
    static_view = select_view(
        decision.store, recent_tool_events=packing.recent_tool_events
    )
    measurements = planner.measurements(decision)
    try:
        lifecycle_view, policy_metadata = planner(
            decision, static_view, measurements=measurements
        )
    except (BudgetExceeded, PackingBudgetError) as exc:
        audit[f"decisions_policy_skipped.{type(exc).__name__}"] += 1
        return None
    if decision.decision_id not in selected_ids:
        audit["decisions_advanced_but_not_selected"] += 1
        return None
    try:
        c_memory, c_target, c_costs = _pack_checked(
            decision, static_view, tokenizer, packing, measurements
        )
        b_memory, b_target, b_costs = _pack_checked(
            decision, lifecycle_view, tokenizer, packing, measurements
        )
        _enforce_history_budget(
            decision,
            static_view,
            tokenizer,
            packing,
            c_costs,
            kv_bytes_per_token=kv_bytes_per_token,
            history_budget_bytes=policy_config.history_budget_bytes,
            measurements=measurements,
        )
        _enforce_history_budget(
            decision,
            lifecycle_view,
            tokenizer,
            packing,
            b_costs,
            kv_bytes_per_token=kv_bytes_per_token,
            history_budget_bytes=policy_config.history_budget_bytes,
            measurements=measurements,
        )
    except (PackingBudgetError, ValueError) as exc:
        audit[f"decision_pairs_skipped.{type(exc).__name__}"] += 1
        return None
    if c_target != b_target:
        raise AssertionError("Matched arms produced different targets")
    return _PlannedDecision(
        decision_ordinal=decision_ordinal,
        session_key=decision.store.session_id,
        decision_id=decision.decision_id,
        decision_index=decision.decision_index,
        source_message_index=decision.source_message_index,
        source=decision.source,
        split=decision.split,
        target_json_text=decision.target.json_text,
        c_view=c_memory.view,
        b_view=b_memory.view,
        c_costs=c_costs,
        b_costs=b_costs,
        policy_metadata=dict(policy_metadata),
    )


def _plan_session(
    row: Mapping[str, Any],
    tokenizer: Any,
    packing: PackingConfig,
    policy_config: RuntimeConfig,
    selected_ids: frozenset[str],
    *,
    kv_bytes_per_token: int,
) -> _SessionPlan:
    # Policy state and tokenization caches intentionally live for one session.
    planner = PrefixLifecyclePlanner(
        tokenizer, packing, policy_config, kv_bytes_per_token=kv_bytes_per_token
    )
    audit: Counter[str] = Counter()
    candidates: list[_PlannedDecision] = []
    decisions_visited = 0
    for decision_ordinal, decision in enumerate(iter_decisions(row)):
        decisions_visited += 1
        candidate = _plan_decision(
            decision,
            decision_ordinal,
            planner,
            tokenizer,
            packing,
            policy_config,
            selected_ids,
            audit,
            kv_bytes_per_token=kv_bytes_per_token,
        )
        if candidate is not None:
            candidates.append(candidate)
    return _SessionPlan(
        decisions_visited=decisions_visited,
        candidates=tuple(candidates),
        audit=dict(audit),
    )


_SESSION_WORKER_TOKENIZER: Any = None
_SESSION_WORKER_PACKING: PackingConfig | None = None
_SESSION_WORKER_POLICY_CONFIG: RuntimeConfig | None = None
_SESSION_WORKER_SELECTED_IDS: frozenset[str] | None = None
_SESSION_WORKER_KV_BYTES_PER_TOKEN: int | None = None


def _initialize_session_worker(
    tokenizer: Any,
    packing: PackingConfig,
    policy_config: RuntimeConfig,
    selected_ids: frozenset[str],
    kv_bytes_per_token: int,
) -> None:
    global _SESSION_WORKER_TOKENIZER
    global _SESSION_WORKER_PACKING
    global _SESSION_WORKER_POLICY_CONFIG
    global _SESSION_WORKER_SELECTED_IDS
    global _SESSION_WORKER_KV_BYTES_PER_TOKEN
    _SESSION_WORKER_TOKENIZER = tokenizer
    _SESSION_WORKER_PACKING = packing
    _SESSION_WORKER_POLICY_CONFIG = policy_config
    _SESSION_WORKER_SELECTED_IDS = selected_ids
    _SESSION_WORKER_KV_BYTES_PER_TOKEN = kv_bytes_per_token


def _plan_session_worker(row: Mapping[str, Any]) -> _SessionPlan:
    if (
        _SESSION_WORKER_PACKING is None
        or _SESSION_WORKER_POLICY_CONFIG is None
        or _SESSION_WORKER_SELECTED_IDS is None
        or _SESSION_WORKER_KV_BYTES_PER_TOKEN is None
    ):
        raise RuntimeError("Session preparation worker was not initialized")
    return _plan_session(
        row,
        _SESSION_WORKER_TOKENIZER,
        _SESSION_WORKER_PACKING,
        _SESSION_WORKER_POLICY_CONFIG,
        _SESSION_WORKER_SELECTED_IDS,
        kv_bytes_per_token=_SESSION_WORKER_KV_BYTES_PER_TOKEN,
    )


@contextmanager
def _parallel_session_plans(
    rows: Iterable[Mapping[str, Any]],
    tokenizer: Any,
    packing: PackingConfig,
    policy_config: RuntimeConfig,
    selected_ids: frozenset[str],
    *,
    kv_bytes_per_token: int,
    workers: int,
) -> Iterator[Iterator[_SessionPlan]]:
    """Yield ordered session plans while keeping at most two tasks per worker."""
    context = multiprocessing.get_context("spawn")
    pool = context.Pool(
        processes=workers,
        initializer=_initialize_session_worker,
        initargs=(
            tokenizer,
            packing,
            policy_config,
            selected_ids,
            kv_bytes_per_token,
        ),
    )
    pending: deque[Any] = deque()
    row_iterator = iter(rows)
    exhausted = False

    def fill_pending() -> None:
        nonlocal exhausted
        while not exhausted and len(pending) < 2 * workers:
            try:
                row = next(row_iterator)
            except StopIteration:
                exhausted = True
                break
            pending.append(pool.apply_async(_plan_session_worker, (row,)))

    def ordered_results() -> Iterator[_SessionPlan]:
        while pending:
            result = pending.popleft().get()
            fill_pending()
            yield result

    completed = False
    try:
        fill_pending()
        yield ordered_results()
        completed = True
    finally:
        try:
            if completed:
                pool.close()
            else:
                pool.terminate()
        finally:
            pool.join()


def _reduce_planned_decision(
    candidate: _PlannedDecision,
    session_index: int,
    packing: PackingConfig,
    sampling: SamplingConfig,
    audit: Counter[str],
    prepared: list[dict[str, Any]],
    used_session_indices: set[int],
    presented_tokens: dict[str, int],
    resident_tokens: dict[str, int],
) -> None:
    # Keep this cap before target canonicalization: cap-skipped targets need not
    # be hashable under the prepared-corpus canonical JSON contract.
    if all(
        presented_tokens["C"]
        + candidate.c_costs[str(ratio)]["presented_encoder_tokens"]
        > sampling.max_presented_tokens_per_arm
        or presented_tokens["B"]
        + candidate.b_costs[str(ratio)]["presented_encoder_tokens"]
        > sampling.max_presented_tokens_per_arm
        for ratio in packing.ratios
    ):
        audit["decision_pairs_skipped_presented_token_cap"] += (
            sampling.repetitions * len(packing.ratios)
        )
        return
    changed = candidate.c_view != candidate.b_view
    target_sha256 = hashlib.sha256(
        _canonical_json(json.loads(candidate.target_json_text)).encode("utf-8")
    ).hexdigest()
    emitted_for_decision = False
    for repetition_index in range(sampling.repetitions):
        for ratio in packing.ratios:
            c_costs = candidate.c_costs[str(ratio)]
            b_costs = candidate.b_costs[str(ratio)]
            c_presented = c_costs["presented_encoder_tokens"]
            b_presented = b_costs["presented_encoder_tokens"]
            if (
                presented_tokens["C"] + c_presented
                > sampling.max_presented_tokens_per_arm
                or presented_tokens["B"] + b_presented
                > sampling.max_presented_tokens_per_arm
            ):
                audit["decision_pairs_skipped_presented_token_cap"] += 1
                continue
            presented_tokens["C"] += c_presented
            presented_tokens["B"] += b_presented
            resident_tokens["C"] += c_costs["resident_kv_tokens"]
            resident_tokens["B"] += b_costs["resident_kv_tokens"]
            if c_costs["gist_tokens"]:
                audit["gist_bearing_pairs.arm.C"] += 1
            if b_costs["gist_tokens"]:
                audit["gist_bearing_pairs.arm.B"] += 1
            emitted_for_decision = True
            if changed:
                audit["changed_view_pairs"] += 1
            prepared.append(
                {
                    "schema_version": PREPARED_SCHEMA_VERSION,
                    "session_index": session_index,
                    "session_key": candidate.session_key,
                    "decision_id": candidate.decision_id,
                    "decision_index": candidate.decision_index,
                    "source_message_index": candidate.source_message_index,
                    "source": candidate.source,
                    "split": candidate.split,
                    "ratio": ratio,
                    "weight": 1.0,
                    "repetition_index": repetition_index,
                    "target_sha256": target_sha256,
                    "arms": {
                        "C": {
                            "view": _view_dict(candidate.c_view),
                            "costs": c_costs,
                        },
                        "B": {
                            "view": _view_dict(candidate.b_view),
                            "costs": b_costs,
                            "policy": candidate.policy_metadata,
                        },
                    },
                }
            )
    if emitted_for_decision:
        used_session_indices.add(session_index)
        audit[
            "changed_view_base_decisions"
            if changed
            else "identical_view_base_decisions"
        ] += 1
        audit["retrieved_event_selections"] += len(
            candidate.policy_metadata.get("retrieved_event_ids", ())
        )
        audit["retained_event_selections"] += len(
            candidate.policy_metadata.get("retained_event_ids", ())
        )
        audit["released_event_selections"] += len(
            candidate.policy_metadata.get("expired_lease_event_ids", ())
        ) + len(candidate.policy_metadata.get("revision_cancelled_event_ids", ()))
        audit["protected_extra_event_selections"] += len(
            set(candidate.policy_metadata.get("protected_event_ids", ()))
            - set(candidate.c_view.raw_event_ids)
        )


def prepare_paired_corpus(
    rows: Iterable[Mapping[str, Any]],
    tokenizer: Any,
    output_dir: str | Path,
    *,
    packing: PackingConfig | None = None,
    sampling: SamplingConfig | None = None,
    policy_config: RuntimeConfig | None = None,
    kv_bytes_per_token: int,
    source_audit: Mapping[str, int] | None = None,
    allow_unchanged_b: bool = False,
    progress: Callable[[Mapping[str, Any]], None] | None = None,
    workers: int = 1,
) -> dict[str, Any]:
    """Prepare matched arm rows and atomically publish a compact corpus."""
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")
    packing = packing or PackingConfig()
    sampling = sampling or SamplingConfig()
    policy_config = policy_config or RuntimeConfig(mode="persistent")
    if progress is not None:
        progress({"phase": "validate_sources"})
    snapshots = snapshot_and_validate_rows(rows)
    audit: Counter[str] = Counter()
    bounded = _bounded_rows(snapshots, sampling, audit)
    del snapshots
    if progress is not None:
        progress({"phase": "candidates", "sessions_total": len(bounded), "sessions_done": 0})
    decisions_by_session: list[tuple[_DecisionCandidate, ...]] = []
    for row in bounded:
        lightweight: list[_DecisionCandidate] = []
        for decision in iter_decision_metadata(row):
            lightweight.append(
                _DecisionCandidate(
                    decision_id=decision.decision_id,
                    session_key=decision.session_key,
                    source=decision.source,
                    source_message_index=decision.source_message_index,
                )
            )
        decisions_by_session.append(tuple(lightweight))
        if progress is not None:
            progress({"phase": "candidates", "sessions_total": len(bounded),
                      "sessions_done": len(decisions_by_session)})
    audit["observable_decisions_before_selection"] = sum(
        len(decisions) for decisions in decisions_by_session
    )
    selected_ids = _select_decision_ids(decisions_by_session, sampling, audit)
    del decisions_by_session
    prepared: list[dict[str, Any]] = []
    used_session_indices: set[int] = set()
    presented_tokens = {"C": 0, "B": 0}
    resident_tokens = {"C": 0, "B": 0}
    decisions_visited = 0
    if workers == 1:
        for session_index, row in enumerate(bounded):
            # ConversationMemory caches prefix signatures for idempotency. Keep
            # that state through one session, then release it before the next.
            planner = PrefixLifecyclePlanner(
                tokenizer, packing, policy_config, kv_bytes_per_token=kv_bytes_per_token
            )
            for decision_ordinal, decision in enumerate(iter_decisions(row)):
                decisions_visited += 1
                if progress is not None:
                    progress({"phase": "planning", "sessions_total": len(bounded),
                              "sessions_done": session_index,
                              "decisions_visited": decisions_visited,
                              "decisions_total": audit["observable_decisions_before_selection"],
                              "paired_exposures_written": len(prepared),
                              "presented_tokens": dict(presented_tokens)})
                candidate = _plan_decision(
                    decision,
                    decision_ordinal,
                    planner,
                    tokenizer,
                    packing,
                    policy_config,
                    selected_ids,
                    audit,
                    kv_bytes_per_token=kv_bytes_per_token,
                )
                if candidate is not None:
                    _reduce_planned_decision(
                        candidate,
                        session_index,
                        packing,
                        sampling,
                        audit,
                        prepared,
                        used_session_indices,
                        presented_tokens,
                        resident_tokens,
                    )
    else:
        with _parallel_session_plans(
            bounded,
            tokenizer,
            packing,
            policy_config,
            selected_ids,
            kv_bytes_per_token=kv_bytes_per_token,
            workers=workers,
        ) as session_plans:
            for session_index, session_plan in enumerate(session_plans):
                audit.update(session_plan.audit)
                candidate_iterator = iter(session_plan.candidates)
                candidate = next(candidate_iterator, None)
                for decision_ordinal in range(session_plan.decisions_visited):
                    decisions_visited += 1
                    if progress is not None:
                        progress({"phase": "planning", "sessions_total": len(bounded),
                                  "sessions_done": session_index,
                                  "decisions_visited": decisions_visited,
                                  "decisions_total": audit["observable_decisions_before_selection"],
                                  "paired_exposures_written": len(prepared),
                                  "presented_tokens": dict(presented_tokens)})
                    if candidate is not None and candidate.decision_ordinal == decision_ordinal:
                        _reduce_planned_decision(
                            candidate,
                            session_index,
                            packing,
                            sampling,
                            audit,
                            prepared,
                            used_session_indices,
                            presented_tokens,
                            resident_tokens,
                        )
                        candidate = next(candidate_iterator, None)
                if candidate is not None:
                    raise AssertionError("Session plan contains an invalid decision ordinal")
    if not prepared:
        raise ValueError("No paired decisions fit the requested sources and budgets")
    audit["gist_bearing_pairs.arm.C"] += 0
    audit["gist_bearing_pairs.arm.B"] += 0
    if not audit["changed_view_pairs"] and not allow_unchanged_b:
        raise ValueError(
            "Every prepared B view equals C; increase policy budgets or check that "
            "the selected sources contain observable recovery/protection triggers. "
            "Use allow_unchanged_b only for a bounded smoke fixture."
        )
    if not audit["gist_bearing_pairs.arm.B"] and not allow_unchanged_b:
        raise ValueError(
            "Every prepared B record has zero gist tokens, so the compressor "
            "would receive no B-arm gradient. Tighten recovery/workspace budgets "
            "or add longer sessions. Use allow_unchanged_b only for a bounded "
            "smoke fixture."
        )

    session_remap = {
        old_index: new_index for new_index, old_index in enumerate(sorted(used_session_indices))
    }
    session_rows: list[dict[str, Any]] = []
    for old_index in sorted(used_session_indices):
        session = dict(bounded[old_index])
        session["schema_version"] = PREPARED_SCHEMA_VERSION
        session["session_index"] = session_remap[old_index]
        session_rows.append(session)
    for pair in prepared:
        pair["session_index"] = session_remap[pair["session_index"]]

    audit["sessions_written"] = len(session_rows)
    audit["base_decisions_written"] = len(
        {pair["decision_id"] for pair in prepared}
    )
    audit["paired_exposures_written"] = len(prepared)
    audit["qa_paired_exposures_written"] = sum(
        pair["source"].startswith("qa:") for pair in prepared
    )
    audit["presented_tokens.arm.C"] = presented_tokens["C"]
    audit["presented_tokens.arm.B"] = presented_tokens["B"]
    audit["resident_kv_tokens.arm.C"] = resident_tokens["C"]
    audit["resident_kv_tokens.arm.B"] = resident_tokens["B"]
    for pair in prepared:
        audit[f"paired_exposures.source.{pair['source']}"] += 1
    first_pair_by_decision: dict[str, dict[str, Any]] = {}
    for pair in prepared:
        first_pair_by_decision.setdefault(pair["decision_id"], pair)
    for pair in first_pair_by_decision.values():
        audit[f"base_decisions.source.{pair['source']}"] += 1
    qa_base = sum(
        pair["source"].startswith("qa:") for pair in first_pair_by_decision.values()
    )
    statistics = {
        "actual_qa_base_decision_fraction": qa_base / len(first_pair_by_decision),
        "actual_qa_exposure_fraction": audit["qa_paired_exposures_written"] / len(prepared),
        "actual_source_base_decisions": {
            source: count
            for source, count in sorted(
                (key.removeprefix("base_decisions.source."), value)
                for key, value in audit.items()
                if key.startswith("base_decisions.source.")
            )
        },
        "actual_source_exposures": {
            source: count
            for source, count in sorted(
                (key.removeprefix("paired_exposures.source."), value)
                for key, value in audit.items()
                if key.startswith("paired_exposures.source.")
            )
        },
        "actual_source_sessions": {
            source: sum(session["source"] == source for session in session_rows)
            for source in sorted({session["source"] for session in session_rows})
        },
    }
    manifest = {
        "schema_version": PREPARED_SCHEMA_VERSION,
        "packing_version": PACKING_VERSION,
        "raw_layout_profile": RAW_LAYOUT_PROFILE,
        "evidence_version": EVIDENCE_VERSION,
        "files": {"sessions": SESSIONS_FILE, "paired_decisions": DECISIONS_FILE},
        "arms": ["C", "B"],
        "tokenizer": tokenizer_identity(tokenizer),
        "packing": _packing_manifest(packing),
        "sampling": asdict(sampling),
        "policy": {
            **asdict(policy_config),
            "source_commit": POLICY_SOURCE_COMMIT,
            "kv_bytes_per_token": kv_bytes_per_token,
            "history_budget_definition": "gist plus charged raw after subtracting the fixed current-input baseline",
            "workspace_budget_definition": "incremental native evidence packet",
            "current_input_baseline": "source system messages plus latest user source message plus last visible source message, deduplicated; same tools and generation prompt",
        },
        "counts": dict(sorted(audit.items())),
        "statistics": statistics,
        "source_audit": dict(sorted((source_audit or {}).items())),
        "allow_unchanged_b": bool(allow_unchanged_b),
    }

    destination = Path(output_dir)
    if destination.exists():
        if not destination.is_dir() or any(destination.iterdir()):
            raise FileExistsError(f"Prepared output directory is not empty: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    pending = destination.with_name(destination.name + f".pending-{os.getpid()}")
    if pending.exists():
        raise FileExistsError(f"Stale prepared-output staging directory exists: {pending}")
    pending.mkdir(parents=True)
    try:
        for filename, records in (
            (SESSIONS_FILE, session_rows),
            (DECISIONS_FILE, prepared),
        ):
            with (pending / filename).open("w", encoding="utf-8", newline="\n") as handle:
                for record in records:
                    handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        manifest["file_integrity"] = {
            filename: _file_integrity(pending / filename)
            for filename in (SESSIONS_FILE, DECISIONS_FILE)
        }
        (pending / MANIFEST_FILE).write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        if destination.exists():
            destination.rmdir()
        pending.rename(destination)
    except Exception:
        if pending.exists():
            shutil.rmtree(pending)
        raise
    if progress is not None:
        progress({"phase": "written", "sessions_done": len(bounded),
                  "sessions_total": len(bounded), "decisions_visited": decisions_visited,
                  "paired_exposures_written": len(prepared),
                  "presented_tokens": dict(presented_tokens)})
    return manifest


def _read_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            records.append(value)
    return tuple(records)


def _file_integrity(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return {"sha256": digest.hexdigest(), "bytes": size}


class PreparedCorpus:
    """Sequence-like loader for one arm of a paired prepared corpus."""

    def __init__(self, data_path: str | Path, tokenizer: Any, arm: str) -> None:
        if arm not in {"C", "B"}:
            raise ValueError("arm must be C or B")
        supplied = Path(data_path)
        manifest_path = supplied / MANIFEST_FILE if supplied.is_dir() else supplied
        self.root = manifest_path.parent
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("schema_version") != PREPARED_SCHEMA_VERSION:
            raise ValueError("Unsupported prepared corpus schema")
        if self.manifest.get("packing_version") != PACKING_VERSION:
            raise ValueError("Prepared corpus packing version differs from this checkout")
        if self.manifest.get("raw_layout_profile") != RAW_LAYOUT_PROFILE:
            raise ValueError("Prepared corpus raw layout differs from this checkout")
        if self.manifest.get("evidence_version") != EVIDENCE_VERSION:
            raise ValueError("Prepared corpus evidence version differs from this checkout")
        if self.manifest.get("tokenizer", {}).get("sha256") != tokenizer_identity(tokenizer)["sha256"]:
            raise ValueError("Tokenizer behavior differs from the prepared corpus")
        self.tokenizer = tokenizer
        self.arm = arm
        files = self.manifest["files"]
        integrity = self.manifest.get("file_integrity")
        if not isinstance(integrity, Mapping):
            raise ValueError("Prepared manifest has no file integrity contract")
        for filename in (files["sessions"], files["paired_decisions"]):
            expected = integrity.get(filename)
            actual = _file_integrity(self.root / filename)
            if expected != actual:
                raise ValueError(f"Prepared file integrity check failed: {filename}")
        self.sessions = _read_jsonl(self.root / files["sessions"])
        self.records = _read_jsonl(self.root / files["paired_decisions"])
        self.packing = PackingConfig(
            **{
                **self.manifest["packing"],
                "ratios": tuple(self.manifest["packing"]["ratios"]),
            }
        )
        self._decision_cache: OrderedDict[tuple[int, int], Decision] = OrderedDict()
        self._max_cached_decisions = 8
        self.session_keys = tuple(record["session_key"] for record in self.records)
        self.decision_ids = tuple(record["decision_id"] for record in self.records)
        self.group_ids = tuple(
            (record["session_key"], record["repetition_index"], record["ratio"])
            for record in self.records
        )
        self._validate_index()

    def _validate_index(self) -> None:
        for expected, session in enumerate(self.sessions):
            if session.get("schema_version") != PREPARED_SCHEMA_VERSION:
                raise ValueError("Session row has an incompatible schema")
            if session.get("session_index") != expected:
                raise ValueError("Session indices must be dense and ordered")
        valid_ratios = set(self.packing.ratios)
        for record in self.records:
            if record.get("schema_version") != PREPARED_SCHEMA_VERSION:
                raise ValueError("Decision row has an incompatible schema")
            if record.get("ratio") not in valid_ratios:
                raise ValueError("Decision ratio is outside the manifest contract")
            if not isinstance(record.get("session_index"), int) or not 0 <= record["session_index"] < len(self.sessions):
                raise ValueError("Decision references an unknown session")
            if set(record.get("arms", {})) != {"C", "B"}:
                raise ValueError("Every decision row must contain exactly the C/B pair")

    def __len__(self) -> int:
        return len(self.records)

    def _decision(self, session_index: int, source_message_index: int) -> Decision:
        key = (session_index, source_message_index)
        cached = self._decision_cache.get(key)
        if cached is not None:
            self._decision_cache.move_to_end(key)
            return cached
        found = next(
            (
                decision
                for decision in iter_decisions(self.sessions[session_index])
                if decision.source_message_index == source_message_index
            ),
            None,
        )
        if found is None:
            raise ValueError("Prepared row references a missing assistant decision")
        self._decision_cache[key] = found
        self._decision_cache.move_to_end(key)
        while len(self._decision_cache) > self._max_cached_decisions:
            self._decision_cache.popitem(last=False)
        return found

    def __getitem__(self, index: int) -> PreparedRecord:
        record = self.records[index]
        decision = self._decision(record["session_index"], record["source_message_index"])
        if decision.decision_id != record["decision_id"]:
            raise ValueError("Prepared decision identity no longer matches its session")
        target_hash = hashlib.sha256(
            _canonical_json(decision.target_dict()).encode("utf-8")
        ).hexdigest()
        if target_hash != record["target_sha256"]:
            raise ValueError("Prepared target hash no longer matches its session")
        view = _view_from_dict(record["arms"][self.arm]["view"])
        memory, target_ids, costs = _pack_checked(
            decision, view, self.tokenizer, self.packing
        )
        _enforce_history_budget(
            decision,
            view,
            self.tokenizer,
            self.packing,
            costs,
            kv_bytes_per_token=int(self.manifest["policy"]["kv_bytes_per_token"]),
            history_budget_bytes=int(self.manifest["policy"]["history_budget_bytes"]),
        )
        expected_costs = record["arms"][self.arm]["costs"]
        if costs[str(record["ratio"])] != expected_costs:
            raise ValueError("Repacked token costs differ from the prepared record")
        return PreparedRecord(
            memory=memory,
            target_ids=target_ids,
            ratio=record["ratio"],
            weight=float(record["weight"]),
            repetition_index=record["repetition_index"],
            decision_id=record["decision_id"],
            session_key=record["session_key"],
            decision_index=record["decision_index"],
            source=record["source"],
            split=record["split"],
        )


__all__ = [
    "DECISIONS_FILE",
    "MANIFEST_FILE",
    "POLICY_SOURCE_COMMIT",
    "PREPARED_SCHEMA_VERSION",
    "SESSIONS_FILE",
    "PackingConfig",
    "PrefixLifecyclePlanner",
    "PreparedCorpus",
    "PreparedRecord",
    "SamplingConfig",
    "prepare_paired_corpus",
    "tokenizer_identity",
]
