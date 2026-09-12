"""Native event-packed S0 with lexical raw retrieval and no draft recovery."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .history_memory.events import EventStore
from .history_memory.packing import (
    EncoderChunk,
    MemoryView,
    PackedMemory,
    PackingBudgetError,
    encode_event_chunks,
    native_ids,
    pack_memory,
    visible_message,
)

from .adapter import raw_source_cutoff
from .always_compress import (
    ALWAYS_COMPRESSION_POLICY,
    CapacityInfeasible,
    coverage_accounting,
)
from .event_native_always import (
    NATIVE_ALWAYS_IMPLEMENTATION_PROFILE,
    NATIVE_RAW_S0_MODE,
    NATIVE_S0_MODE,
)
from .event_native_policy import (
    CURRENT_INPUT_BASELINE,
    EVENT_NATIVE_POLICY_VERSION,
    HISTORY_BUDGET_DEFINITION,
    POLICY_SOURCE_COMMIT,
    WORKSPACE_BUDGET_DEFINITION,
    EventNativeController,
    _PackingConfig,
    _canonical_json,
    _json_snapshot,
)
from .event_native_raw import NO_GIST_RAW_LAYOUT, RuntimeMemoryView
from .failed_operation import PROMPT_CAP, VERSION as FAILED_OPERATION_VERSION
from .failed_operation import cue_for, records as failed_operation_records
from .policy import PolicyInputError
from .raw_reserve import RESERVE_VERSION
from .source_needs import (
    NEEDS_VERSION,
    SourceRequest,
    build_needs_input,
    fit_prediction_input,
    lexical_source_ids,
)


EVENT_NATIVE_S0_VERSION = "a-event-native-s0-v1"
EVENT_NATIVE_RAW_S0_VERSION = "a-event-native-raw-s0-v1"
NATIVE_RAW_IMPLEMENTATION_PROFILE = "event-native-raw-representation-v1"
ALWAYS_COMPRESS_GIST_LAYOUT = "event-native-always-compress-gist-v1"
S0_CONFIG_DEFAULTS = {
    "source_index_max_events": 12,
    "predictor_prompt_token_cap": 2048,
    "predictor_completion_token_cap": 256,
    "latest_complete_tool_protection": "budgeted",
}
_S0_CONFIG_FIELDS = frozenset(S0_CONFIG_DEFAULTS)
_FIXED_GEOMETRY = {
    "max_chunk_tokens": 768,
    "chunk_overlap": 64,
    "max_chunks": 48,
}


@dataclass
class PreparedEventNativeS0:
    """One native S0/Raw decision and its pre-generation extraction plan."""

    memory: PackedMemory
    metadata: dict[str, Any]
    eligible_chunks: tuple[EncoderChunk, ...]
    _owner: object = field(repr=False, compare=False)
    _session_id: str = field(repr=False, compare=False)
    _decision_key: str = field(repr=False, compare=False)
    _checked_signature: str | None = field(default=None, repr=False, compare=False)
    _checked_result: dict[str, Any] | None = field(default=None, repr=False, compare=False)


@dataclass
class _SessionState:
    message_json: tuple[str, ...]
    tools_json: str
    decision_index: int
    decisions: dict[str, tuple[tuple[Any, ...], PreparedEventNativeS0]]
    active_decision_key: str


@dataclass(frozen=True)
class _Measurement:
    memory: PackedMemory
    raw_prompt_tokens: int
    common_raw_prompt_tokens: int
    raw_history_tokens: int
    per_ratio: dict[str, dict[str, int]]
    logical_sequence_tokens: int
    reasons: tuple[str, ...]


class EventNativeS0Controller:
    """Preserve the frozen S0 controller on the native event packing interface.

    In the C2KV representation, every complete historical event is offered to
    pre-generation extraction.  The matched Raw representation keeps the same
    cutoff, protection, lexical selection and B0 checks with no extraction.
    """

    def __init__(
        self,
        tokenizer: Any,
        *,
        packing: Mapping[str, Any],
        policy: Mapping[str, Any],
        model_context: int | None = None,
        s0_config: Mapping[str, Any] | None = None,
        history_representation: str = "c2kv",
    ) -> None:
        if not callable(getattr(tokenizer, "apply_chat_template", None)):
            raise TypeError("tokenizer must expose apply_chat_template")
        if model_context is not None and not _positive_int(model_context):
            raise ValueError("model_context must be a positive integer or None")
        if history_representation not in {"c2kv", "raw"}:
            raise ValueError("history_representation must be 'c2kv' or 'raw'")
        self.tokenizer = tokenizer
        self.packing = _PackingConfig.from_mapping(packing)
        for name, expected in _FIXED_GEOMETRY.items():
            if getattr(self.packing, name) != expected:
                raise ValueError(
                    f"Native S0 requires packing.{name}={expected}; got "
                    f"{getattr(self.packing, name)}"
                )
        self.policy_config, self.kv_bytes_per_token = EventNativeController._parse_policy(
            policy
        )
        if self.policy_config.lease_decisions != 0:
            raise ValueError("Native S0 requires lease_decisions=0")
        if self.policy_config.max_retrieved_events != 2:
            raise ValueError("Native S0 requires max_retrieved_events=2")
        self.policy = _json_snapshot(policy)
        self.model_context = model_context
        self.s0_config = self._parse_s0_config(s0_config)
        self.history_representation = history_representation
        self._owner = object()
        self._sessions: dict[str, _SessionState] = {}

    def prepare(
        self,
        payload: Mapping[str, Any],
        *,
        ratio: int,
        max_new_tokens: int,
    ) -> PreparedEventNativeS0:
        session_id, decision_key, store, tools, tools_json, message_json = (
            self._validate_request(payload, ratio, max_new_tokens)
        )
        signature = (message_json, tools_json, ratio, max_new_tokens)
        state = self._sessions.get(session_id)
        if state is not None:
            if state.tools_json != tools_json:
                raise PolicyInputError(
                    "Tools changed within a session; use a new explicit session_id"
                )
            EventNativeController._validate_monotone_prefix(
                state.message_json, message_json
            )
            cached = state.decisions.get(decision_key)
            if cached is not None:
                old_signature, prepared = cached
                if old_signature != signature:
                    raise PolicyInputError(
                        f"decision_key {decision_key!r} was reused with different input"
                    )
                return prepared

        decision_index = (state.decision_index if state is not None else 0) + 1
        prepared = self._prepare_view(
            store,
            tools,
            ratio=ratio,
            max_new_tokens=max_new_tokens,
            decision_key=decision_key,
            decision_index=decision_index,
        )
        decisions = dict(state.decisions) if state is not None else {}
        decisions[decision_key] = (signature, prepared)
        self._sessions[session_id] = _SessionState(
            message_json=message_json,
            tools_json=tools_json,
            decision_index=decision_index,
            decisions=decisions,
            active_decision_key=decision_key,
        )
        return prepared

    def reconsider(
        self,
        prepared: PreparedEventNativeS0,
        draft_tool_calls: Any,
        *,
        draft_text: str,
        parse_error: str | None = None,
    ) -> dict[str, Any]:
        if (
            not isinstance(prepared, PreparedEventNativeS0)
            or prepared._owner is not self._owner
        ):
            raise PolicyInputError("Prepared decision belongs to another native S0 controller")
        if not isinstance(draft_text, str):
            raise TypeError("draft_text must be a string")
        if parse_error is not None and not isinstance(parse_error, str):
            raise TypeError("parse_error must be a string or None")
        signature = _canonical_json(
            {
                "draft_text": draft_text,
                "draft_tool_calls": draft_tool_calls,
                "parse_error": parse_error,
            }
        )
        if prepared._checked_result is not None:
            if signature != prepared._checked_signature:
                raise PolicyInputError(
                    "A prepared decision cannot inspect a second different draft"
                )
            return _copy_reconsideration(prepared._checked_result)

        state = self._sessions.get(prepared._session_id)
        if state is None or state.active_decision_key != prepared._decision_key:
            raise PolicyInputError("Prepared decision is stale for the active prefix")
        decision = {
            "version": (
                EVENT_NATIVE_RAW_S0_VERSION
                if self.history_representation == "raw"
                else EVENT_NATIVE_S0_VERSION
            ),
            "status": "no_op",
            "reason": (
                "native_raw_s0_single_generation"
                if self.history_representation == "raw"
                else "native_s0_single_generation"
            ),
            "gap_type": None,
            "candidate_event_id": None,
            "bindings": [],
            "judges_action_correctness": False,
            "decision_index": prepared.metadata["decision_index"],
            "upgrade_count": 0,
            "regeneration_allowed": False,
            "upgraded_event_id": None,
        }
        metadata = copy.deepcopy(prepared.metadata)
        metadata["exact_recovery"] = copy.deepcopy(decision)
        metadata["post_draft_exact_recovery_applied"] = False
        result = {
            "regenerate": False,
            "memory": prepared.memory,
            "metadata": metadata,
            "decision": copy.deepcopy(decision),
        }
        prepared._checked_signature = signature
        prepared._checked_result = _copy_reconsideration(result)
        return result

    def _prepare_view(
        self,
        store: EventStore,
        tools: tuple[dict[str, Any], ...],
        *,
        ratio: int,
        max_new_tokens: int,
        decision_key: str,
        decision_index: int,
    ) -> PreparedEventNativeS0:
        if self.history_representation == "raw":
            return self._prepare_raw_view(
                store,
                tools,
                ratio=ratio,
                max_new_tokens=max_new_tokens,
                decision_key=decision_key,
                decision_index=decision_index,
            )
        cutoff = raw_source_cutoff([message.to_dict() for message in store.messages])
        common_source_indices = {
            index
            for event in store.events
            if event.kind == "instruction"
            for index in event.source_indices
        } | set(range(cutoff, len(store.messages)))
        common_messages = [
            visible_message(store.messages[index])
            for index in sorted(common_source_indices)
        ]
        common_tokens = self._count(common_messages, tools) if common_messages else 0

        mandatory_ids = {
            event.event_id
            for event in store.events
            if not event.complete
            or event.kind == "instruction"
            or bool(set(event.source_indices) & set(range(cutoff, len(store.messages))))
        }
        users = [event for event in store.events if event.kind == "user"]
        if users:
            mandatory_ids.add(users[-1].event_id)

        eligible_event_ids = tuple(
            event.event_id
            for event in store.events
            if event.complete
            and event.kind != "instruction"
            and any(index < cutoff for index in event.source_indices)
        )
        eligible_source_indices = frozenset(
            index
            for event_id in eligible_event_ids
            for index in store.event(event_id).source_indices
            if index < cutoff
        )
        eligible_set = set(eligible_event_ids)
        eligible_chunks = tuple(
            chunk
            for event_id in eligible_event_ids
            for chunk in encode_event_chunks(
                store,
                event_id,
                self.tokenizer,
                max_chunk_tokens=self.packing.max_chunk_tokens,
                chunk_overlap=self.packing.chunk_overlap,
            )
        )

        complete_tools = [
            event
            for event in store.events
            if event.kind == "tool_event" and event.complete
        ]
        latest_tool = complete_tools[-1] if complete_tools else None
        latest_optional = bool(
            latest_tool
            and latest_tool.event_id not in mandatory_ids
            and latest_tool.event_id in eligible_set
        )
        protected_ids = set(mandatory_ids)
        if latest_tool is not None:
            protected_ids.add(latest_tool.event_id)
        raw_ids = set(protected_ids)
        budget = min(
            self.policy_config.history_budget_bytes,
            self.policy_config.workspace_budget_bytes,
        )
        latest_receipt = {
            "version": "budgeted-latest-complete-tool-v1",
            "policy": self.s0_config["latest_complete_tool_protection"],
            "status": (
                "none"
                if latest_tool is None
                else "admitted"
                if latest_optional
                else "mandatory-common"
            ),
            "event_id": latest_tool.event_id if latest_tool else None,
            "source_indices": list(latest_tool.source_indices) if latest_tool else [],
            "initially_protected": latest_tool is not None,
            "selector_recomputed_after_skip": False,
            "indexed_after_skip": False,
            "budget_bytes": budget,
            "candidate_required_native_bytes": None,
            "candidate_active_history_bytes": None,
            "reason": (
                "no_complete_tool_event"
                if latest_tool is None
                else "latest_complete_tool_is_optional_historical_raw"
                if latest_optional
                else "latest_complete_tool_is_in_the_mandatory_common_suffix"
            ),
        }

        requested, fitted, needs_receipt = self._lexical_request(
            store, tools, raw_ids, recent_tool_event_visible=True
        )
        reservation = self._reserve_minimum_gist(
            store,
            tools,
            raw_ids,
            mandatory_ids,
            eligible_event_ids,
            common_tokens,
            max_new_tokens,
        )
        if reservation is None and latest_optional:
            required = self._try_measure(
                store,
                tools,
                raw_ids,
                mandatory_ids,
                (),
                eligible_event_ids,
                common_tokens,
                max_new_tokens,
            )
            representation_candidates = [
                candidate
                for event_id in self._gist_priority(store, eligible_event_ids)
                if (
                    candidate := self._try_measure(
                        store,
                        tools,
                        raw_ids,
                        mandatory_ids,
                        (event_id,),
                        eligible_event_ids,
                        common_tokens,
                        max_new_tokens,
                    )
                )
                is not None
            ]
            latest_receipt.update(
                status="skipped",
                candidate_required_native_bytes=(
                    self._raw_history_bytes(required) if required is not None else budget + 1
                ),
                candidate_active_history_bytes=(
                    min(
                        (self._max_history_bytes(candidate) for candidate in representation_candidates),
                        default=None,
                    )
                ),
                reason=(
                    "protected_native_over_budget"
                    if required is None or self._raw_history_bytes(required) > budget
                    else "required_representation_over_budget"
                ),
            )
            raw_ids.remove(latest_tool.event_id)
            protected_ids.remove(latest_tool.event_id)
            requested, fitted, needs_receipt = self._lexical_request(
                store, tools, raw_ids, recent_tool_event_visible=False
            )
            latest_receipt.update(
                selector_recomputed_after_skip=True,
                indexed_after_skip=bool(
                    fitted
                    and latest_tool.event_id
                    in {entry["source_id"] for entry in fitted["index"]}
                ),
            )
            reservation = self._reserve_minimum_gist(
                store,
                tools,
                raw_ids,
                mandatory_ids,
                eligible_event_ids,
                common_tokens,
                max_new_tokens,
            )
        if reservation is None:
            failure_detail = self._reservation_failure_detail(
                store,
                tools,
                raw_ids,
                mandatory_ids,
                eligible_event_ids,
                common_tokens,
                max_new_tokens,
            )
            raise CapacityInfeasible(
                "Native S0 mandatory raw input and minimum whole-event gist "
                f"cannot fit the declared limits: {failure_detail!r}"
            )

        view, measure, reservation_receipt = reservation
        if latest_optional and latest_receipt["status"] == "admitted":
            latest_receipt.update(
                candidate_required_native_bytes=self._raw_history_bytes(measure),
                candidate_active_history_bytes=self._max_history_bytes(measure),
                reason="protected_native_and_required_representation_fit",
            )

        admitted: list[str] = []
        skipped: list[dict[str, Any]] = []
        for event_id in requested.source_ids:
            if event_id not in eligible_set:
                raise PolicyInputError("Lexical selector returned non-eligible history")
            candidate_raw = set(view.raw_event_ids) | {event_id}
            candidate = self._try_measure(
                store,
                tools,
                candidate_raw,
                mandatory_ids,
                view.gist_event_ids,
                eligible_event_ids,
                common_tokens,
                max_new_tokens,
            )
            if candidate is None or candidate.reasons:
                skipped.append(
                    {
                        "event_id": event_id,
                        "reason": "requested_source_budget",
                        "candidate_history_bytes": (
                            None if candidate is None else self._max_history_bytes(candidate)
                        ),
                        "admission_failures": (
                            ["packing_budget_exceeded"]
                            if candidate is None
                            else list(candidate.reasons)
                        ),
                    }
                )
                continue
            view, measure = candidate.memory.view, candidate
            admitted.append(event_id)

        gist_priority = self._gist_priority(store, eligible_event_ids)
        retained_gist = list(view.gist_event_ids)
        skipped_gist: list[dict[str, Any]] = []
        for event_id in gist_priority:
            if event_id in retained_gist:
                continue
            candidate = self._try_measure(
                store,
                tools,
                view.raw_event_ids,
                mandatory_ids,
                (*retained_gist, event_id),
                eligible_event_ids,
                common_tokens,
                max_new_tokens,
            )
            if candidate is None or candidate.reasons:
                skipped_gist.append(
                    {
                        "event_id": event_id,
                        "reasons": (
                            ["packing_budget_exceeded"]
                            if candidate is None
                            else list(candidate.reasons)
                        ),
                    }
                )
                continue
            view, measure = candidate.memory.view, candidate
            retained_gist.append(event_id)

        reserve_candidates = [
            event
            for event in complete_tools
            if event.event_id in set(view.gist_event_ids)
            and event.event_id not in set(view.raw_event_ids)
            and event.event_id not in protected_ids
        ]
        reserve_event = reserve_candidates[-1] if reserve_candidates else None
        raw_reserve_receipt = {
            "version": RESERVE_VERSION,
            "condition": "recent_gist_backed_event",
            "maximum_extra_events": 1,
            "candidate_event_id": reserve_event.event_id if reserve_event else None,
            "admitted_event_id": None,
            "added_source_indices": [],
            "extra_raw_tokens": 0,
            "extra_kv_equivalent_bytes": 0,
            "unused_capacity_before_bytes": budget - self._max_history_bytes(measure),
            "status": "no_eligible_extra_event",
            "existing_raw_messages_unchanged": True,
            "gist_messages_unchanged": True,
            "protected_events_unchanged": True,
            "no_fallback_search": True,
        }
        spare_ids: list[str] = []
        if reserve_event is not None:
            candidate = self._try_measure(
                store,
                tools,
                set(view.raw_event_ids) | {reserve_event.event_id},
                mandatory_ids,
                view.gist_event_ids,
                eligible_event_ids,
                common_tokens,
                max_new_tokens,
            )
            candidate_bytes = None if candidate is None else self._max_history_bytes(candidate)
            raw_reserve_receipt["candidate_active_history_bytes"] = candidate_bytes
            if candidate is None or candidate.reasons:
                raw_reserve_receipt.update(
                    status="extra_event_over_budget",
                    admission_failures=(
                        ["packing_budget_exceeded"]
                        if candidate is None
                        else list(candidate.reasons)
                    ),
                )
            else:
                extra_tokens = candidate.raw_prompt_tokens - measure.raw_prompt_tokens
                if extra_tokens <= 0:
                    raise PolicyInputError("A complete spare raw event added no raw tokens")
                view, measure = candidate.memory.view, candidate
                spare_ids.append(reserve_event.event_id)
                raw_reserve_receipt.update(
                    status="extra_event_admitted",
                    admitted_event_id=reserve_event.event_id,
                    added_source_indices=list(reserve_event.source_indices),
                    extra_raw_tokens=extra_tokens,
                    extra_kv_equivalent_bytes=extra_tokens * self.kv_bytes_per_token,
                )

        derived_messages: tuple[dict[str, str], ...] = ()
        failure_diagnostic = failed_operation_records(store)
        failed_record = (
            failure_diagnostic["failures"][0]
            if failure_diagnostic["failures"]
            else None
        )
        failed_receipt = {
            "version": FAILED_OPERATION_VERSION,
            "status": "no_current_goal_failure",
            "selected_record": copy.deepcopy(failed_record),
            "failure_record_count": len(failure_diagnostic["failures"]),
            "incremental_raw_tokens": 0,
            "extra_bytes": 0,
            "prompt_cap": PROMPT_CAP,
            "goal_completion": "unknown",
            "input_source": "preceding observed prefix only",
            "derived_workspace_message_count": 0,
            "source_indices_are_not_raw_copies": True,
        }
        if failed_record is not None:
            cue = cue_for(failed_record)
            standalone = self._count([cue], ())
            failed_receipt["standalone_tokens"] = standalone
            if standalone > PROMPT_CAP:
                failed_receipt["status"] = "cue_prompt_cap"
            else:
                candidate = self._try_measure(
                    store,
                    tools,
                    view.raw_event_ids,
                    mandatory_ids,
                    view.gist_event_ids,
                    eligible_event_ids,
                    common_tokens,
                    max_new_tokens,
                    derived_messages=(cue,),
                )
                candidate_bytes = None if candidate is None else self._max_history_bytes(candidate)
                failed_receipt["candidate_active_history_bytes"] = candidate_bytes
                if candidate is None or candidate.reasons:
                    failures = (
                        ["packing_budget_exceeded"]
                        if candidate is None
                        else list(candidate.reasons)
                    )
                    failed_receipt.update(
                        status=(
                            "workspace_cap"
                            if any(
                                reason == "workspace_token_budget"
                                or reason.startswith("workspace_byte_budget:")
                                or reason.startswith("history_byte_budget:")
                                for reason in failures
                            )
                            else "declared_limit"
                        ),
                        admission_failures=failures,
                    )
                else:
                    delta = candidate.raw_prompt_tokens - measure.raw_prompt_tokens
                    if delta <= 0:
                        raise PolicyInputError("A nonempty failed-operation cue added no tokens")
                    derived_messages = (cue,)
                    view, measure = candidate.memory.view, candidate
                    failed_receipt.update(
                        status="admitted",
                        incremental_raw_tokens=delta,
                        extra_bytes=delta * self.kv_bytes_per_token,
                        derived_workspace_message_count=1,
                        source_already_fully_raw_visible=set(
                            failed_record["source_indices"]
                        )
                        <= set(measure.memory.raw_source_indices),
                        source_already_fully_gist_backed=failed_record["event_id"]
                        in set(view.gist_event_ids),
                    )

        coverage, packing_fragments, retained_fragments = self._coverage(
            store,
            eligible_event_ids,
            eligible_source_indices,
            eligible_chunks,
            measure.memory,
        )
        full = self._full_reference(
            store, tools, common_tokens, max_new_tokens
        )
        compression_ratio = self._compression_ratio(full, measure, coverage, ratio)
        selected_event_ids = [
            event.event_id
            for event in store.events
            if event.event_id in set(view.raw_event_ids)
        ]
        omitted_eligible = [
            event_id
            for event_id in eligible_event_ids
            if event_id not in set(view.raw_event_ids) | set(view.gist_event_ids)
        ]
        metadata = {
            "event_native_s0_version": EVENT_NATIVE_S0_VERSION,
            "event_native_policy_version": EVENT_NATIVE_POLICY_VERSION,
            "policy_source_commit": POLICY_SOURCE_COMMIT,
            "session_id": store.session_id,
            "decision_key": decision_key,
            "decision_index": decision_index,
            "view_mode": NATIVE_S0_MODE,
            "mode": NATIVE_S0_MODE,
            "route_mode": NATIVE_S0_MODE,
            "route": {
                "view_mode": NATIVE_S0_MODE,
                "baseline_identity": "S0-native-event-lexical-raw-reserve-failed-operation",
                "recovery_enabled": False,
                "max_generations_per_decision": 1,
                "legacy_1088_equivalent": False,
            },
            "compression_policy": ALWAYS_COMPRESSION_POLICY,
            "implementation_profile": NATIVE_ALWAYS_IMPLEMENTATION_PROFILE,
            "history_view_protocol": "fixed-budget-main",
            "requested_ratio": ratio,
            "max_new_tokens": max_new_tokens,
            "model_context": self.model_context,
            "configured_max_sequence_tokens": self.packing.max_sequence_tokens,
            "chunk_geometry": copy.deepcopy(_FIXED_GEOMETRY),
            "kv_bytes_per_token": self.kv_bytes_per_token,
            "history_budget_bytes": self.policy_config.history_budget_bytes,
            "workspace_budget_bytes": self.policy_config.workspace_budget_bytes,
            "shared_allocation_budget_bytes": budget,
            "history_budget_definition": HISTORY_BUDGET_DEFINITION,
            "workspace_budget_definition": WORKSPACE_BUDGET_DEFINITION,
            "current_input_baseline": CURRENT_INPUT_BASELINE,
            "s0_common_input_definition": "source instructions plus raw_source_cutoff suffix",
            "raw_source_cutoff": cutoff,
            "common_input_source_indices": sorted(common_source_indices),
            "common_raw_prompt_tokens": common_tokens,
            "raw_source_indices": list(measure.memory.raw_source_indices),
            "derived_workspace_prefix_messages": copy.deepcopy(list(derived_messages)),
            "derived_workspace_source_indices": [],
            "raw_event_ids": list(view.raw_event_ids),
            "gist_event_ids": list(view.gist_event_ids),
            "omitted_event_ids": list(view.omitted_event_ids),
            "mandatory_raw_event_ids": list(view.mandatory_raw_event_ids),
            "evidence_event_ids": [],
            "raw_evidence_event_ids": list(dict.fromkeys([*admitted, *spare_ids])),
            "selected_event_ids": selected_event_ids,
            "selected_source_indices": list(measure.memory.raw_source_indices),
            "protected_event_ids": [
                event.event_id for event in store.events if event.event_id in protected_ids
            ],
            "retrieved_event_ids": admitted,
            "retained_event_ids": [],
            "recency_selected_event_ids": spare_ids,
            "lease_decisions": 0,
            "max_retrieved_events": 2,
            "pre_draft_retrieval": True,
            "recovery_stage": "pre_generation_s0_lexical",
            "post_draft_exact_recovery_applied": False,
            "source_needs": {
                "version": NEEDS_VERSION,
                "strategy": "lexical",
                "history_representation": "c2kv-native-event",
                "input_receipt": needs_receipt,
                "candidate_event_ids": [
                    entry["source_id"] for entry in fitted["index"]
                ]
                if fitted
                else [],
                "prediction_status": requested.status,
                "requested_event_ids": list(requested.source_ids),
                "typed_needs": [],
                "admitted_event_ids": admitted,
                "skipped_for_budget": skipped,
                "predictor_prompt_token_cap": self.s0_config[
                    "predictor_prompt_token_cap"
                ],
                "predictor_completion_token_cap": self.s0_config[
                    "predictor_completion_token_cap"
                ],
                "predictor_calls": 0,
                "admission_rule": (
                    "mandatory raw, budgeted latest, minimum gist, lexical whole raw, "
                    "whole-event gist refill"
                ),
            },
            "latest_complete_tool_protection": latest_receipt,
            "gist_reservation": reservation_receipt,
            "gist_refill_priority_event_ids": gist_priority,
            "gist_refilled_event_ids": list(view.gist_event_ids),
            "skipped_gist_refill_events": skipped_gist,
            "raw_reserve": raw_reserve_receipt,
            "failed_operation_cue": failed_receipt,
            "source_coverage": coverage,
            "eligible_extraction": {
                "status": "planned_for_pre_generation_extraction",
                "source": "preceding observable EventStore prefix",
                "eligible_event_ids": list(eligible_event_ids),
                "eligible_source_indices": sorted(eligible_source_indices),
                "whole_event_encoded_source_indices": sorted(
                    {
                        index
                        for chunk in eligible_chunks
                        for index in chunk.source_indices
                    }
                ),
                "eligible_chunk_count": len(eligible_chunks),
                "eligible_presented_encoder_tokens": sum(
                    len(chunk.token_ids) for chunk in eligible_chunks
                ),
                "eligible_unique_encoder_tokens": sum(
                    max(chunk.source_token_end for chunk in eligible_chunks if chunk.event_id == event_id)
                    for event_id in eligible_event_ids
                )
                if eligible_event_ids
                else 0,
                "retained_event_ids": list(view.gist_event_ids),
                "retained_chunk_count": len(measure.memory.chunks),
                "retained_presented_encoder_tokens": sum(
                    len(chunk.token_ids) for chunk in measure.memory.chunks
                ),
                "omitted_active_gist_event_ids": omitted_eligible,
                "raw_gist_overlap_event_ids": [
                    event_id
                    for event_id in eligible_event_ids
                    if event_id in set(view.raw_event_ids) & set(view.gist_event_ids)
                ],
                "backend_execution_required": bool(eligible_chunks),
                "cache_reuse_known_after_generation": True,
                "charged_separately_from_active_history_bytes": True,
            },
            "history_packing_fragments": packing_fragments,
            "retained_history_packing_fragments": retained_fragments,
            "per_ratio": copy.deepcopy(measure.per_ratio),
            "raw_prompt_tokens": measure.raw_prompt_tokens,
            "actual_raw_history_tokens": measure.raw_history_tokens,
            "actual_gist_tokens": measure.per_ratio[str(ratio)]["history_gist_tokens"],
            "actual_history_bytes": measure.per_ratio[str(ratio)]["history_bytes"],
            "logical_sequence_tokens": measure.logical_sequence_tokens,
            "same_prefix_full_reference": full,
            "compression_ratio": compression_ratio,
            "no_eligible_history": not eligible_event_ids,
            "full_source_coverage": coverage["complete_history_coverage"],
            "atomic_packing_unit": "whole_event_all_encoder_chunks",
            "min_gist_reservation_required": bool(eligible_event_ids),
            "min_gist_reservation_met": (
                bool(view.gist_event_ids) if eligible_event_ids else None
            ),
            "legacy_1088_block_parity": False,
        }
        return PreparedEventNativeS0(
            memory=measure.memory,
            metadata=metadata,
            eligible_chunks=eligible_chunks,
            _owner=self._owner,
            _session_id=store.session_id,
            _decision_key=decision_key,
        )

    def _prepare_raw_view(
        self,
        store: EventStore,
        tools: tuple[dict[str, Any], ...],
        *,
        ratio: int,
        max_new_tokens: int,
        decision_key: str,
        decision_index: int,
    ) -> PreparedEventNativeS0:
        """Adapt the frozen legacy Raw allocation order to native packing."""

        cutoff = raw_source_cutoff([message.to_dict() for message in store.messages])
        common_source_indices = {
            index
            for event in store.events
            if event.kind == "instruction"
            for index in event.source_indices
        } | set(range(cutoff, len(store.messages)))
        common_messages = [
            visible_message(store.messages[index])
            for index in sorted(common_source_indices)
        ]
        common_tokens = self._count(common_messages, tools) if common_messages else 0

        mandatory_ids = {
            event.event_id
            for event in store.events
            if not event.complete
            or event.kind == "instruction"
            or bool(set(event.source_indices) & set(range(cutoff, len(store.messages))))
        }
        users = [event for event in store.events if event.kind == "user"]
        if users:
            mandatory_ids.add(users[-1].event_id)
        eligible_event_ids = tuple(
            event.event_id
            for event in store.events
            if event.complete
            and event.kind != "instruction"
            and any(index < cutoff for index in event.source_indices)
        )
        eligible_source_indices = frozenset(
            index
            for event_id in eligible_event_ids
            for index in store.event(event_id).source_indices
            if index < cutoff
        )
        eligible_set = set(eligible_event_ids)

        complete_tools = [
            event
            for event in store.events
            if event.kind == "tool_event" and event.complete
        ]
        latest_tool = complete_tools[-1] if complete_tools else None
        latest_optional = bool(
            latest_tool
            and latest_tool.event_id not in mandatory_ids
            and latest_tool.event_id in eligible_set
        )
        protected_ids = set(mandatory_ids)
        if latest_tool is not None:
            protected_ids.add(latest_tool.event_id)
        raw_ids = set(protected_ids)
        budget = min(
            self.policy_config.history_budget_bytes,
            self.policy_config.workspace_budget_bytes,
        )
        latest_receipt = {
            "version": "budgeted-latest-complete-tool-v1",
            "policy": self.s0_config["latest_complete_tool_protection"],
            "status": (
                "none"
                if latest_tool is None
                else "admitted"
                if latest_optional
                else "mandatory-common"
            ),
            "event_id": latest_tool.event_id if latest_tool else None,
            "source_indices": list(latest_tool.source_indices) if latest_tool else [],
            "initially_protected": latest_tool is not None,
            "selector_recomputed_after_skip": False,
            "indexed_after_skip": False,
            "budget_bytes": budget,
            "candidate_required_native_bytes": None,
            "candidate_active_history_bytes": None,
            "reason": (
                "no_complete_tool_event"
                if latest_tool is None
                else "latest_complete_tool_is_optional_historical_raw"
                if latest_optional
                else "latest_complete_tool_is_in_the_mandatory_common_suffix"
            ),
        }

        requested, fitted, needs_receipt = self._lexical_request(
            store, tools, raw_ids, recent_tool_event_visible=True
        )
        measure = self._try_measure(
            store,
            tools,
            raw_ids,
            mandatory_ids,
            (),
            eligible_event_ids,
            common_tokens,
            max_new_tokens,
        )
        if (measure is None or measure.reasons) and latest_optional:
            latest_receipt.update(
                status="skipped",
                candidate_required_native_bytes=(
                    self._raw_history_bytes(measure) if measure is not None else budget + 1
                ),
                candidate_active_history_bytes=(
                    self._max_history_bytes(measure) if measure is not None else None
                ),
                reason=(
                    "protected_native_over_budget"
                    if measure is None or self._raw_history_bytes(measure) > budget
                    else "protected_native_over_declared_limit"
                ),
            )
            raw_ids.remove(latest_tool.event_id)
            protected_ids.remove(latest_tool.event_id)
            requested, fitted, needs_receipt = self._lexical_request(
                store, tools, raw_ids, recent_tool_event_visible=False
            )
            latest_receipt.update(
                selector_recomputed_after_skip=True,
                indexed_after_skip=bool(
                    fitted
                    and latest_tool.event_id
                    in {entry["source_id"] for entry in fitted["index"]}
                ),
            )
            measure = self._try_measure(
                store,
                tools,
                raw_ids,
                mandatory_ids,
                (),
                eligible_event_ids,
                common_tokens,
                max_new_tokens,
            )
        if measure is None or measure.reasons:
            raise CapacityInfeasible(
                "Native Raw mandatory input cannot fit the declared limits: "
                f"{['packing_budget_exceeded'] if measure is None else list(measure.reasons)!r}"
            )
        if latest_optional and latest_receipt["status"] == "admitted":
            latest_receipt.update(
                candidate_required_native_bytes=self._raw_history_bytes(measure),
                candidate_active_history_bytes=self._max_history_bytes(measure),
                reason="protected_native_fits",
            )

        view = measure.memory.view
        admitted: list[str] = []
        skipped: list[dict[str, Any]] = []
        for event_id in requested.source_ids:
            if event_id not in eligible_set:
                raise PolicyInputError("Lexical selector returned non-eligible history")
            candidate = self._try_measure(
                store,
                tools,
                set(view.raw_event_ids) | {event_id},
                mandatory_ids,
                (),
                eligible_event_ids,
                common_tokens,
                max_new_tokens,
            )
            if candidate is None or candidate.reasons:
                skipped.append(
                    {
                        "event_id": event_id,
                        "reason": "requested_source_budget",
                        "candidate_history_bytes": (
                            None if candidate is None else self._max_history_bytes(candidate)
                        ),
                        "admission_failures": (
                            ["packing_budget_exceeded"]
                            if candidate is None
                            else list(candidate.reasons)
                        ),
                    }
                )
                continue
            view, measure = candidate.memory.view, candidate
            admitted.append(event_id)

        recency_ids: list[str] = []
        skipped_recency: list[dict[str, Any]] = []
        for event in reversed(store.events):
            if not event.complete or event.event_id in set(view.raw_event_ids):
                continue
            candidate = self._try_measure(
                store,
                tools,
                set(view.raw_event_ids) | {event.event_id},
                mandatory_ids,
                (),
                eligible_event_ids,
                common_tokens,
                max_new_tokens,
            )
            if candidate is None or candidate.reasons:
                skipped_recency.append(
                    {
                        "event_id": event.event_id,
                        "source_indices": list(event.source_indices),
                        "reasons": (
                            ["packing_budget_exceeded"]
                            if candidate is None
                            else list(candidate.reasons)
                        ),
                    }
                )
                continue
            view, measure = candidate.memory.view, candidate
            recency_ids.append(event.event_id)

        raw_reserve_receipt = {
            "version": RESERVE_VERSION,
            "condition": "recent_gist_backed_event",
            "maximum_extra_events": 1,
            "candidate_event_id": None,
            "admitted_event_id": None,
            "added_source_indices": [],
            "extra_raw_tokens": 0,
            "extra_kv_equivalent_bytes": 0,
            "unused_capacity_before_bytes": budget - self._max_history_bytes(measure),
            "status": "no_eligible_extra_event",
            "existing_raw_messages_unchanged": True,
            "gist_messages_unchanged": True,
            "protected_events_unchanged": True,
            "no_fallback_search": True,
            "reason": "raw_representation_has_no_gist_backed_event",
        }

        derived_messages: tuple[dict[str, str], ...] = ()
        failure_diagnostic = failed_operation_records(store)
        failed_record = (
            failure_diagnostic["failures"][0]
            if failure_diagnostic["failures"]
            else None
        )
        failed_receipt = {
            "version": FAILED_OPERATION_VERSION,
            "status": "no_current_goal_failure",
            "selected_record": copy.deepcopy(failed_record),
            "failure_record_count": len(failure_diagnostic["failures"]),
            "incremental_raw_tokens": 0,
            "extra_bytes": 0,
            "prompt_cap": PROMPT_CAP,
            "goal_completion": "unknown",
            "input_source": "preceding observed prefix only",
            "derived_workspace_message_count": 0,
            "source_indices_are_not_raw_copies": True,
        }
        if failed_record is not None:
            cue = cue_for(failed_record)
            standalone = self._count([cue], ())
            failed_receipt["standalone_tokens"] = standalone
            if standalone > PROMPT_CAP:
                failed_receipt["status"] = "cue_prompt_cap"
            else:
                candidate = self._try_measure(
                    store,
                    tools,
                    view.raw_event_ids,
                    mandatory_ids,
                    (),
                    eligible_event_ids,
                    common_tokens,
                    max_new_tokens,
                    derived_messages=(cue,),
                )
                candidate_bytes = (
                    None if candidate is None else self._max_history_bytes(candidate)
                )
                failed_receipt["candidate_active_history_bytes"] = candidate_bytes
                if candidate is None or candidate.reasons:
                    failures = (
                        ["packing_budget_exceeded"]
                        if candidate is None
                        else list(candidate.reasons)
                    )
                    failed_receipt.update(
                        status=(
                            "workspace_cap"
                            if any(
                                reason == "workspace_token_budget"
                                or reason.startswith("workspace_byte_budget:")
                                or reason.startswith("history_byte_budget:")
                                for reason in failures
                            )
                            else "declared_limit"
                        ),
                        admission_failures=failures,
                    )
                else:
                    delta = candidate.raw_prompt_tokens - measure.raw_prompt_tokens
                    if delta <= 0:
                        raise PolicyInputError(
                            "A nonempty failed-operation cue added no tokens"
                        )
                    derived_messages = (cue,)
                    view, measure = candidate.memory.view, candidate
                    failed_receipt.update(
                        status="admitted",
                        incremental_raw_tokens=delta,
                        extra_bytes=delta * self.kv_bytes_per_token,
                        derived_workspace_message_count=1,
                        source_already_fully_raw_visible=set(
                            failed_record["source_indices"]
                        )
                        <= set(measure.memory.raw_source_indices),
                        source_already_fully_gist_backed=False,
                    )

        coverage, _, _ = self._coverage(
            store,
            eligible_event_ids,
            eligible_source_indices,
            (),
            measure.memory,
        )
        full = self._full_reference(store, tools, common_tokens, max_new_tokens)
        compression_ratio = self._compression_ratio(full, measure, coverage, ratio)
        selected_event_ids = [
            event.event_id
            for event in store.events
            if event.event_id in set(view.raw_event_ids)
        ]
        omitted_eligible = [
            event_id
            for event_id in eligible_event_ids
            if event_id not in set(view.raw_event_ids)
        ]
        metadata = {
            "event_native_raw_s0_version": EVENT_NATIVE_RAW_S0_VERSION,
            "event_native_policy_version": EVENT_NATIVE_POLICY_VERSION,
            "policy_source_commit": POLICY_SOURCE_COMMIT,
            "session_id": store.session_id,
            "decision_key": decision_key,
            "decision_index": decision_index,
            "view_mode": NATIVE_RAW_S0_MODE,
            "mode": NATIVE_RAW_S0_MODE,
            "route_mode": NATIVE_RAW_S0_MODE,
            "route": {
                "view_mode": NATIVE_RAW_S0_MODE,
                "baseline_identity": (
                    "Raw-native-event-lexical-raw-reserve-failed-operation"
                ),
                "recovery_enabled": False,
                "max_generations_per_decision": 1,
                "legacy_1088_equivalent": False,
            },
            "compression_policy": None,
            "implementation_profile": NATIVE_RAW_IMPLEMENTATION_PROFILE,
            "history_representation": "raw",
            "history_view_protocol": "fixed-budget-main",
            "requested_ratio": ratio,
            "max_new_tokens": max_new_tokens,
            "model_context": self.model_context,
            "configured_max_sequence_tokens": self.packing.max_sequence_tokens,
            "chunk_geometry": copy.deepcopy(_FIXED_GEOMETRY),
            "kv_bytes_per_token": self.kv_bytes_per_token,
            "history_budget_bytes": self.policy_config.history_budget_bytes,
            "workspace_budget_bytes": self.policy_config.workspace_budget_bytes,
            "shared_allocation_budget_bytes": budget,
            "history_budget_definition": HISTORY_BUDGET_DEFINITION,
            "workspace_budget_definition": WORKSPACE_BUDGET_DEFINITION,
            "current_input_baseline": CURRENT_INPUT_BASELINE,
            "s0_common_input_definition": (
                "source instructions plus raw_source_cutoff suffix"
            ),
            "raw_source_cutoff": cutoff,
            "common_input_source_indices": sorted(common_source_indices),
            "common_raw_prompt_tokens": common_tokens,
            "raw_source_indices": list(measure.memory.raw_source_indices),
            "derived_workspace_prefix_messages": copy.deepcopy(list(derived_messages)),
            "derived_workspace_source_indices": [],
            "raw_event_ids": list(view.raw_event_ids),
            "gist_event_ids": [],
            "omitted_event_ids": list(view.omitted_event_ids),
            "mandatory_raw_event_ids": list(view.mandatory_raw_event_ids),
            "evidence_event_ids": [],
            "raw_evidence_event_ids": list(admitted),
            "selected_event_ids": selected_event_ids,
            "selected_source_indices": list(measure.memory.raw_source_indices),
            "protected_event_ids": [
                event.event_id
                for event in store.events
                if event.event_id in protected_ids
            ],
            "retrieved_event_ids": admitted,
            "retained_event_ids": [],
            "recency_selected_event_ids": recency_ids,
            "lease_decisions": 0,
            "max_retrieved_events": 2,
            "pre_draft_retrieval": True,
            "recovery_stage": "pre_generation_raw_lexical",
            "post_draft_exact_recovery_applied": False,
            "source_needs": {
                "version": NEEDS_VERSION,
                "strategy": "lexical",
                "history_representation": "raw-native-event",
                "input_receipt": needs_receipt,
                "candidate_event_ids": (
                    [entry["source_id"] for entry in fitted["index"]]
                    if fitted
                    else []
                ),
                "prediction_status": requested.status,
                "requested_event_ids": list(requested.source_ids),
                "typed_needs": [],
                "admitted_event_ids": admitted,
                "skipped_for_budget": skipped,
                "predictor_prompt_token_cap": self.s0_config[
                    "predictor_prompt_token_cap"
                ],
                "predictor_completion_token_cap": self.s0_config[
                    "predictor_completion_token_cap"
                ],
                "predictor_calls": 0,
                "admission_rule": (
                    "mandatory raw, budgeted latest, lexical whole raw, "
                    "whole-event reverse-recency refill"
                ),
            },
            "latest_complete_tool_protection": latest_receipt,
            "gist_reservation": {
                "required": False,
                "satisfied": True,
                "reserved_event_id": None,
                "reserved_gist_bytes": 0,
                "selection_budget_bytes": budget,
            },
            "gist_refill_priority_event_ids": [],
            "gist_refilled_event_ids": [],
            "skipped_gist_refill_events": [],
            "raw_refill": {
                "order": "reverse_event_recency_after_lexical",
                "refilled_event_ids": recency_ids,
                "skipped_events": skipped_recency,
            },
            "raw_reserve": raw_reserve_receipt,
            "failed_operation_cue": failed_receipt,
            "source_coverage": coverage,
            "eligible_extraction": {
                "status": "disabled_for_raw_representation",
                "source": "not_applicable",
                "candidate_history_event_ids": list(eligible_event_ids),
                "eligible_event_ids": [],
                "eligible_source_indices": [],
                "whole_event_encoded_source_indices": [],
                "eligible_chunk_count": 0,
                "eligible_presented_encoder_tokens": 0,
                "eligible_unique_encoder_tokens": 0,
                "retained_event_ids": [],
                "retained_chunk_count": 0,
                "retained_presented_encoder_tokens": 0,
                "omitted_active_gist_event_ids": omitted_eligible,
                "raw_gist_overlap_event_ids": [],
                "backend_execution_required": False,
                "cache_reuse_known_after_generation": False,
                "charged_separately_from_active_history_bytes": False,
            },
            "history_packing_fragments": [],
            "retained_history_packing_fragments": [],
            "per_ratio": copy.deepcopy(measure.per_ratio),
            "raw_prompt_tokens": measure.raw_prompt_tokens,
            "actual_raw_history_tokens": measure.raw_history_tokens,
            "actual_gist_tokens": 0,
            "actual_history_bytes": measure.per_ratio[str(ratio)]["history_bytes"],
            "logical_sequence_tokens": measure.logical_sequence_tokens,
            "same_prefix_full_reference": full,
            "compression_ratio": compression_ratio,
            "no_eligible_history": not eligible_event_ids,
            "full_source_coverage": coverage["complete_history_coverage"],
            "atomic_packing_unit": "whole_event_raw",
            "min_gist_reservation_required": False,
            "min_gist_reservation_met": None,
            "legacy_1088_block_parity": False,
        }
        return PreparedEventNativeS0(
            memory=measure.memory,
            metadata=metadata,
            eligible_chunks=(),
            _owner=self._owner,
            _session_id=store.session_id,
            _decision_key=decision_key,
        )

    def _lexical_request(self, store, tools, raw_ids, *, recent_tool_event_visible):
        context = build_needs_input(
            store,
            tools,
            excluded_event_ids=tuple(raw_ids),
            max_candidates=self.s0_config["source_index_max_events"],
            recent_tool_event_visible=recent_tool_event_visible,
        )
        fitted, receipt = fit_prediction_input(
            context,
            self._count,
            max_prompt_tokens=self.s0_config["predictor_prompt_token_cap"],
        )
        requested = SourceRequest((), (), receipt["status"])
        if fitted is not None and receipt["status"] == "ready":
            requested = SourceRequest(
                lexical_source_ids(store, fitted, max_sources=2),
                (),
                "lexical_sources_ranked",
            )
        return requested, fitted, receipt

    def _reserve_minimum_gist(
        self,
        store,
        tools,
        raw_ids,
        mandatory_ids,
        eligible_event_ids,
        common_tokens,
        max_new_tokens,
    ):
        priority = self._gist_priority(store, eligible_event_ids)
        candidates = priority if priority else [None]
        skipped = []
        for event_id in candidates:
            gist_ids = () if event_id is None else (event_id,)
            measure = self._try_measure(
                store,
                tools,
                raw_ids,
                mandatory_ids,
                gist_ids,
                eligible_event_ids,
                common_tokens,
                max_new_tokens,
            )
            if measure is None or measure.reasons:
                skipped.append(
                    {
                        "event_id": event_id,
                        "reasons": (
                            ["packing_budget_exceeded"]
                            if measure is None
                            else list(measure.reasons)
                        ),
                    }
                )
                continue
            receipt = {
                "required": bool(eligible_event_ids),
                "satisfied": bool(measure.memory.view.gist_event_ids)
                if eligible_event_ids
                else True,
                "reserved_event_id": event_id,
                "reserved_gist_bytes": (
                    0
                    if event_id is None
                    else max(
                        row["history_gist_tokens"] * self.kv_bytes_per_token
                        for row in measure.per_ratio.values()
                    )
                ),
                "minimum_active_history_bytes": self._max_history_bytes(measure),
                "skipped_events": skipped,
                "selection_budget_bytes": min(
                    self.policy_config.history_budget_bytes,
                    self.policy_config.workspace_budget_bytes,
                ),
            }
            return measure.memory.view, measure, receipt
        return None

    def _reservation_failure_detail(
        self,
        store,
        tools,
        raw_ids,
        mandatory_ids,
        eligible_event_ids,
        common_tokens,
        max_new_tokens,
    ):
        priority = self._gist_priority(store, eligible_event_ids)
        candidates = priority if priority else [None]
        rows = []
        for event_id in candidates:
            measure = self._try_measure(
                store,
                tools,
                raw_ids,
                mandatory_ids,
                () if event_id is None else (event_id,),
                eligible_event_ids,
                common_tokens,
                max_new_tokens,
            )
            rows.append(
                {
                    "event_id": event_id,
                    "reasons": ["packing_budget_exceeded"]
                    if measure is None
                    else list(measure.reasons),
                    "raw_history_tokens": None
                    if measure is None
                    else measure.raw_history_tokens,
                    "per_ratio": None
                    if measure is None
                    else copy.deepcopy(measure.per_ratio),
                    "logical_sequence_tokens": None
                    if measure is None
                    else measure.logical_sequence_tokens,
                }
            )
        return {
            "history_budget_bytes": self.policy_config.history_budget_bytes,
            "workspace_budget_bytes": self.policy_config.workspace_budget_bytes,
            "model_context": self.model_context,
            "candidates": rows,
        }

    def _try_measure(
        self,
        store,
        tools,
        raw_ids,
        mandatory_ids,
        gist_ids,
        eligible_event_ids,
        common_tokens,
        max_new_tokens,
        *,
        derived_messages=(),
    ) -> _Measurement | None:
        known = {event.event_id for event in store.events}
        raw = set(raw_ids)
        gist = set(gist_ids)
        omitted = known - raw - gist
        view = RuntimeMemoryView(
            gist_event_ids=_ordered_ids(store, gist),
            raw_event_ids=_ordered_ids(store, raw),
            evidence_event_ids=(),
            omitted_event_ids=_ordered_ids(store, omitted),
            mandatory_raw_event_ids=_ordered_ids(store, mandatory_ids),
            raw_control_layout=(
                NO_GIST_RAW_LAYOUT
                if self.history_representation == "raw"
                else ALWAYS_COMPRESS_GIST_LAYOUT
            ),
        )
        try:
            memory = pack_memory(
                store,
                view,
                self.tokenizer,
                tools=tools,
                max_chunk_tokens=self.packing.max_chunk_tokens,
                chunk_overlap=self.packing.chunk_overlap,
                max_chunks=self.packing.max_chunks,
                derived_workspace_prefix_messages=derived_messages,
            )
        except PackingBudgetError:
            return None
        raw_prompt_tokens = len(memory.system_input_ids) + len(memory.workspace_input_ids)
        raw_history_tokens = raw_prompt_tokens - common_tokens
        if raw_history_tokens < 0:
            raise PolicyInputError("Native S0 common input exceeds the prepared raw request")
        reasons = []
        if len(memory.system_input_ids) > self.packing.max_system_tokens:
            reasons.append("system_budget")
        if len(memory.workspace_input_ids) > self.packing.max_workspace_tokens:
            reasons.append("workspace_token_budget")
        encoder_tokens = sum(len(chunk.token_ids) for chunk in memory.chunks)
        if encoder_tokens > self.packing.max_encoder_tokens:
            reasons.append("encoder_budget")
        logical = (
            memory.workspace_position_start
            + len(memory.workspace_input_ids)
            + max_new_tokens
        )
        if self.model_context is not None and logical > self.model_context:
            reasons.append("model_logical_context")
        per_ratio = {}
        for candidate_ratio in self.packing.ratios:
            costs = memory.costs(candidate_ratio)
            gist_tokens = costs["gist_tokens"]
            history_tokens = raw_history_tokens + gist_tokens
            history_bytes = history_tokens * self.kv_bytes_per_token
            sequence_tokens = costs["resident_kv_tokens"] + max_new_tokens
            if history_bytes > self.policy_config.history_budget_bytes:
                reasons.append(f"history_byte_budget:{candidate_ratio}")
            if history_bytes > self.policy_config.workspace_budget_bytes:
                reasons.append(f"workspace_byte_budget:{candidate_ratio}")
            if sequence_tokens > self.packing.max_sequence_tokens:
                reasons.append(f"physical_sequence_budget:{candidate_ratio}")
            if self.model_context is not None and sequence_tokens > self.model_context:
                reasons.append(f"model_physical_context:{candidate_ratio}")
            per_ratio[str(candidate_ratio)] = {
                **costs,
                "max_new_tokens": max_new_tokens,
                "sequence_tokens": sequence_tokens,
                "logical_sequence_tokens": logical,
                "history_gist_tokens": gist_tokens,
                "history_raw_tokens": raw_history_tokens,
                "history_total_tokens": history_tokens,
                "history_bytes": history_bytes,
                "history_budget_bytes": self.policy_config.history_budget_bytes,
                "workspace_budget_bytes": self.policy_config.workspace_budget_bytes,
            }
        return _Measurement(
            memory=memory,
            raw_prompt_tokens=raw_prompt_tokens,
            common_raw_prompt_tokens=common_tokens,
            raw_history_tokens=raw_history_tokens,
            per_ratio=per_ratio,
            logical_sequence_tokens=logical,
            reasons=tuple(dict.fromkeys(reasons)),
        )

    def _coverage(
        self,
        store,
        eligible_event_ids,
        eligible_source_indices,
        eligible_chunks,
        memory,
    ):
        def fragment(chunk):
            fragment_id = (
                f"{chunk.event_id}:{chunk.part_index}:"
                f"{chunk.source_token_start}:{chunk.source_token_end}"
            )
            return {
                "fragment_id": fragment_id,
                "packing_fragment_id": fragment_id,
                "event_id": chunk.event_id,
                "source_indices": list(chunk.source_indices),
                "encoder_input_tokens": len(chunk.token_ids),
            }

        packing_fragments = [fragment(chunk) for chunk in eligible_chunks]
        retained_fragments = [fragment(chunk) for chunk in memory.chunks]
        coverage = coverage_accounting(
            eligible_sources=eligible_source_indices,
            raw_sources=set(memory.raw_source_indices),
            retained_blocks=retained_fragments,
            packing_fragments=packing_fragments,
        )
        coverage.update(
            eligible_event_ids=list(eligible_event_ids),
            raw_event_ids=[
                event_id
                for event_id in eligible_event_ids
                if event_id in set(memory.view.raw_event_ids)
            ],
            gist_event_ids=[
                event_id
                for event_id in eligible_event_ids
                if event_id in set(memory.view.gist_event_ids)
            ],
            raw_gist_overlap_event_ids=[
                event_id
                for event_id in eligible_event_ids
                if event_id in set(memory.view.raw_event_ids)
                and event_id in set(memory.view.gist_event_ids)
            ],
            eligibility_stage="observable EventStore before B/W admission",
            whole_event_encoded_source_indices=sorted(
                {
                    index
                    for chunk in eligible_chunks
                    for index in chunk.source_indices
                }
            ),
            whole_event_extra_source_indices=sorted(
                {
                    index
                    for chunk in eligible_chunks
                    for index in chunk.source_indices
                }
                - set(eligible_source_indices)
            ),
        )
        return coverage, packing_fragments, retained_fragments

    def _full_reference(self, store, tools, common_tokens, max_new_tokens):
        full_view = MemoryView(
            gist_event_ids=(),
            raw_event_ids=tuple(event.event_id for event in store.events),
        )
        full_memory = pack_memory(store, full_view, self.tokenizer, tools=tools)
        full_prompt_tokens = len(full_memory.system_input_ids) + len(
            full_memory.workspace_input_ids
        )
        full_history_tokens = max(0, full_prompt_tokens - common_tokens)
        return {
            "render_only": True,
            "generation_performed": False,
            "full_prompt_tokens": full_prompt_tokens,
            "full_history_tokens": full_history_tokens,
            "full_history_bytes": full_history_tokens * self.kv_bytes_per_token,
            "common_live_tokens": common_tokens,
            "common_live_bytes": common_tokens * self.kv_bytes_per_token,
            "max_new_tokens": max_new_tokens,
        }

    def _compression_ratio(self, full, measure, coverage, ratio):
        row = measure.per_ratio[str(ratio)]
        active = row["history_bytes"]
        history = full["full_history_bytes"]
        common = full["common_live_bytes"]
        return {
            "schema": "a-same-prefix-compression-ratio-v1",
            "denominator_source": "Full renderer on this observable prefix; no Full generation",
            "full_history_bytes": history,
            "common_live_bytes": common,
            "active_history_bytes": active,
            "active_gist_bytes": row["history_gist_tokens"] * self.kv_bytes_per_token,
            "active_raw_history_bytes": row["history_raw_tokens"] * self.kv_bytes_per_token,
            "n_history": history / active if history and active else None,
            "n_total": (common + history) / (common + active)
            if common + active
            else None,
            "configured_gist_ratio": ratio,
            "actual_source_to_gist_ratio": (
                coverage["retained_encoder_input_tokens"]
                / row["history_gist_tokens"]
                if row["history_gist_tokens"]
                and coverage["retained_encoder_input_tokens"] is not None
                else None
            ),
            "includes_coverage_loss": not coverage["complete_history_coverage"],
        }

    def _count(self, messages, tools):
        if not messages:
            return 0
        return len(
            native_ids(
                self.tokenizer,
                list(messages),
                tools=tools or None,
                generation=True,
            )
        )

    @staticmethod
    def _gist_priority(store, event_ids):
        ordered = list(event_ids)
        if not ordered:
            return []
        return [
            ordered[0],
            *(
                event.event_id
                for event in sorted(
                    (store.event(event_id) for event_id in ordered[1:]),
                    key=lambda event: max(event.source_indices),
                    reverse=True,
                )
            ),
        ]

    @staticmethod
    def _max_history_bytes(measure):
        return max(row["history_bytes"] for row in measure.per_ratio.values())

    def _raw_history_bytes(self, measure):
        return measure.raw_history_tokens * self.kv_bytes_per_token

    @staticmethod
    def _parse_s0_config(value):
        if value is None:
            result = copy.deepcopy(S0_CONFIG_DEFAULTS)
        elif not isinstance(value, Mapping):
            raise TypeError("s0_config must be a mapping or None")
        else:
            if set(value) != _S0_CONFIG_FIELDS:
                raise ValueError(
                    "s0_config must contain exactly " + ", ".join(sorted(_S0_CONFIG_FIELDS))
                )
            result = copy.deepcopy(dict(value))
        if (
            type(result["source_index_max_events"]) is not int
            or not 0 < result["source_index_max_events"] <= 12
        ):
            raise ValueError("source_index_max_events must be an integer from one to twelve")
        for name in ("predictor_prompt_token_cap", "predictor_completion_token_cap"):
            if not _positive_int(result[name]):
                raise ValueError(f"{name} must be a positive integer")
        if result["latest_complete_tool_protection"] != "budgeted":
            raise ValueError("Native S0 requires budgeted latest-complete-tool protection")
        return result

    def _validate_request(self, payload, ratio, max_new_tokens):
        if not isinstance(payload, Mapping):
            raise TypeError("payload must be a mapping")
        allowed = {"session_id", "decision_key", "messages", "tools"}
        unknown = sorted(set(payload) - allowed)
        if unknown:
            raise PolicyInputError(
                f"Targets and privileged request fields are forbidden: {unknown!r}"
            )
        if not _positive_int(ratio) or ratio not in self.packing.ratios:
            raise ValueError(
                f"ratio must be one of checkpoint packing ratios {self.packing.ratios!r}"
            )
        if not _positive_int(max_new_tokens):
            raise ValueError("max_new_tokens must be a positive integer")
        if max_new_tokens > self.packing.max_target_tokens:
            raise PackingBudgetError(
                f"Generation needs up to {max_new_tokens} tokens; budget is "
                f"{self.packing.max_target_tokens}"
            )
        session_id = payload.get("session_id")
        decision_key = payload.get("decision_key")
        if not isinstance(session_id, str) or not session_id:
            raise PolicyInputError("An explicit nonempty session_id is required")
        if not isinstance(decision_key, str) or not decision_key:
            raise PolicyInputError("An explicit nonempty decision_key is required")
        messages = payload.get("messages")
        if (
            not isinstance(messages, Sequence)
            or isinstance(messages, (str, bytes, bytearray))
            or not messages
            or any(not isinstance(message, Mapping) for message in messages)
        ):
            raise PolicyInputError("messages must be a nonempty sequence of mappings")
        raw_tools = payload.get("tools", ())
        if raw_tools is None:
            raw_tools = ()
        if (
            not isinstance(raw_tools, Sequence)
            or isinstance(raw_tools, (str, bytes, bytearray))
            or any(not isinstance(tool, Mapping) for tool in raw_tools)
        ):
            raise PolicyInputError("tools must be a sequence of mappings")
        tools = tuple(_json_snapshot(tool) for tool in raw_tools)
        names = [
            tool.get("function", {}).get("name")
            for tool in tools
            if isinstance(tool.get("function"), Mapping)
        ]
        if len(names) != len(set(names)):
            raise PolicyInputError("Failed-operation observations require unique tool names")
        tools_json = _canonical_json(tools)
        store = EventStore.from_messages(session_id, messages)
        message_json = tuple(message.json_text for message in store.messages)
        return session_id, decision_key, store, tools, tools_json, message_json


def _ordered_ids(store, ids):
    selected = set(ids)
    return tuple(event.event_id for event in store.events if event.event_id in selected)


def _copy_reconsideration(value):
    return {
        "regenerate": bool(value["regenerate"]),
        "memory": value["memory"],
        "metadata": copy.deepcopy(value["metadata"]),
        "decision": copy.deepcopy(value["decision"]),
    }


def _positive_int(value):
    return not isinstance(value, bool) and isinstance(value, int) and value > 0


__all__ = [
    "EVENT_NATIVE_S0_VERSION",
    "EventNativeS0Controller",
    "PreparedEventNativeS0",
    "S0_CONFIG_DEFAULTS",
]
