"""Native event-packed Text summary control backed by the legacy S0 allocator."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from history_memory.events import EventStore
from history_memory.packing import (
    EncoderChunk,
    PackedMemory,
    PackingBudgetError,
    native_ids,
    pack_memory,
    raw_workspace_messages,
    visible_message,
)

from .adapter import raw_source_cutoff
from .always_compress import ALWAYS_COMPRESSION_POLICY
from .event_native_always import NATIVE_TEXT_S0_MODE
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
from .event_native_s0_policy import EventNativeS0Controller
from .policy import PolicyInputError
from .source_needs_runtime import SourceNeedsRuntime


EVENT_NATIVE_TEXT_SUMMARY_VERSION = "a-event-native-text-summary-v1"
EVENT_NATIVE_TEXT_SUMMARY_MODE = NATIVE_TEXT_S0_MODE
SOURCE_NEEDS_TEXT_SUMMARY_ROUTE = (
    "text_summary_native_needs_lexical_raw_reserve_failed_operation"
)
TEXT_SUMMARY_IMPLEMENTATION_PROFILE = "event-native-text-summary-representation-v1"
SUMMARY_RESOURCE_CONFIG = {
    "summary_model": "c2kv-agent",
    "summary_prompt_token_cap": 1024,
    "summary_attempts_per_task": 1152,
}


@dataclass
class PreparedEventNativeTextSummary:
    """One allocator-owned native Text summary decision."""

    memory: PackedMemory
    metadata: dict[str, Any]
    eligible_chunks: tuple[EncoderChunk, ...] = ()
    _owner: object = field(repr=False, compare=False, default=None)
    _session_id: str = field(repr=False, compare=False, default="")
    _decision_key: str = field(repr=False, compare=False, default="")
    _checked_signature: str | None = field(default=None, repr=False, compare=False)
    _checked_result: dict[str, Any] | None = field(
        default=None, repr=False, compare=False
    )


@dataclass
class _SessionState:
    message_json: tuple[str, ...]
    tools_json: str
    decision_index: int
    decisions: dict[
        str, tuple[tuple[Any, ...], PreparedEventNativeTextSummary]
    ]
    active_decision_key: str


class EventNativeTextSummaryController:
    """Adapt the frozen Text summary allocator to native ``PackedMemory``.

    ``SourceNeedsRuntime`` remains the only owner of lexical selection,
    summary admission, reverse refill, raw reserve, and the failed-operation
    cue.  This adapter only supplies native token accounting and proves that
    the allocator's final visible request has an identical native packing.
    """

    def __init__(
        self,
        tokenizer: Any,
        *,
        packing: Mapping[str, Any],
        policy: Mapping[str, Any],
        model_context: int | None = None,
        s0_config: Mapping[str, Any] | None = None,
        run_id: str,
        summary_renderer: Any,
    ) -> None:
        if not callable(getattr(tokenizer, "apply_chat_template", None)):
            raise TypeError("tokenizer must expose apply_chat_template")
        if model_context is not None and not _positive_int(model_context):
            raise ValueError("model_context must be a positive integer or None")
        if not isinstance(run_id, str) or not run_id:
            raise ValueError("run_id must be a nonempty string")
        renderer = getattr(summary_renderer, "render", None)
        if renderer is None:
            renderer = summary_renderer
        if not callable(renderer):
            raise TypeError("summary_renderer must be callable or expose render")

        self.tokenizer = tokenizer
        self.packing = _PackingConfig.from_mapping(packing)
        self.policy_config, self.kv_bytes_per_token = (
            EventNativeController._parse_policy(policy)
        )
        if self.policy_config.lease_decisions != 0:
            raise ValueError("Native Text summary requires lease_decisions=0")
        if self.policy_config.max_retrieved_events != 2:
            raise ValueError("Native Text summary requires max_retrieved_events=2")
        self.policy = _json_snapshot(policy)
        self.model_context = model_context
        self.s0_config = EventNativeS0Controller._parse_s0_config(s0_config)
        self.run_id = run_id
        self.summary_renderer = summary_renderer
        self.summary_config = copy.deepcopy(SUMMARY_RESOURCE_CONFIG)
        self._render_summary = renderer
        self._owner = object()
        self._sessions: dict[str, _SessionState] = {}

        allocator_config = {
            "mode": SOURCE_NEEDS_TEXT_SUMMARY_ROUTE,
            "run_id": run_id,
            "bytes_per_kv_token": self.kv_bytes_per_token,
            "history_budget_bytes": self.policy_config.history_budget_bytes,
            "workspace_budget_bytes": self.policy_config.workspace_budget_bytes,
            "lease_decisions": self.policy_config.lease_decisions,
            "max_retrieved_events": self.policy_config.max_retrieved_events,
            **copy.deepcopy(self.s0_config),
            "compression_policy": ALWAYS_COMPRESSION_POLICY,
            "history_view_protocol": "fixed-budget-main",
            **copy.deepcopy(self.summary_config),
        }
        self._allocator = SourceNeedsRuntime(allocator_config, self._count)
        # ``SummaryTransport`` reads this name from legacy runtimes.  Keeping
        # it here lets wiring supply the same accounted transport without a
        # proxy renderer or a second token-count implementation.
        self._token_counter = self._count

    def prepare(
        self,
        payload: Mapping[str, Any],
        *,
        ratio: int,
        max_new_tokens: int,
    ) -> PreparedEventNativeTextSummary:
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
                previous_signature, prepared = cached
                if previous_signature != signature:
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
        prepared: PreparedEventNativeTextSummary,
        draft_tool_calls: Any,
        *,
        draft_text: str,
        parse_error: str | None = None,
    ) -> dict[str, Any]:
        if (
            not isinstance(prepared, PreparedEventNativeTextSummary)
            or prepared._owner is not self._owner
        ):
            raise PolicyInputError(
                "Prepared decision belongs to another native Text summary controller"
            )
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
            "version": EVENT_NATIVE_TEXT_SUMMARY_VERSION,
            "status": "no_op",
            "reason": "native_text_summary_single_generation",
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
    ) -> PreparedEventNativeTextSummary:
        source = [message.to_dict() for message in store.messages]
        full_messages = [visible_message(message) for message in store.messages]
        cutoff = raw_source_cutoff(source)
        task_id, attempt_id = _session_identity(store.session_id)
        parent_request_id = json.dumps(
            [store.session_id, decision_key], separators=(",", ":")
        )
        full_counts = {
            "current_start_out_index": cutoff,
            "compressed_records": [],
            "gist_tokens": 0,
            "original_tokens": 0,
            "n_gist_messages": 0,
            "compressed": 0,
            "n_docs": 0,
            "history_raw": 0,
            "dropped_docs": 0,
        }
        allocator_context = {
            # SourceNeedsRuntime owns EventStore IDs using context.task_id.
            # Keep the API-owned complete session identity at that seam.
            "task_id": store.session_id,
            "attempt_id": attempt_id,
            "decision_id": decision_key,
            "run_id": self.run_id,
        }
        renderer_context = {
            "session_id": store.session_id,
            "task_id": task_id,
            "attempt_id": attempt_id,
            "decision_id": decision_key,
            "run_id": self.run_id,
            "parent_request_id": parent_request_id,
        }

        def render_summary(messages):
            return self._render_summary(messages, copy.deepcopy(renderer_context))

        allocator_messages, allocator_counts = self._allocator.apply(
            source,
            full_messages,
            full_counts,
            allocator_context,
            tools,
            render_summary=render_summary,
        )
        allocator_meta = allocator_counts["memory_runtime"]
        selected_sources = tuple(allocator_meta["selected_source_indices"])
        selected_source_set = set(selected_sources)
        if len(selected_sources) != len(selected_source_set):
            raise PolicyInputError("Text-summary allocator repeated a source index")

        raw_ids = tuple(
            event.event_id
            for event in store.events
            if set(event.source_indices) <= selected_source_set
        )
        covered_sources = {
            index for event_id in raw_ids for index in store.event(event_id).source_indices
        }
        if covered_sources != selected_source_set:
            raise PolicyInputError(
                "Text-summary allocator selected a partial native event; native packing "
                "requires whole-event source ownership"
            )
        omitted_ids = tuple(
            event.event_id for event in store.events if event.event_id not in set(raw_ids)
        )
        protected = set(allocator_meta["protected_event_ids"])
        unknown_protected = protected - {event.event_id for event in store.events}
        if unknown_protected:
            raise PolicyInputError(
                f"Text-summary allocator returned unknown protected events: "
                f"{sorted(unknown_protected)!r}"
            )
        mandatory_ids = tuple(
            event.event_id for event in store.events if event.event_id in protected
        )
        if not set(mandatory_ids) <= set(raw_ids):
            raise PolicyInputError("A protected allocator event was not raw-visible")
        view = RuntimeMemoryView(
            gist_event_ids=(),
            raw_event_ids=raw_ids,
            evidence_event_ids=(),
            omitted_event_ids=omitted_ids,
            mandatory_raw_event_ids=mandatory_ids,
            raw_control_layout=NO_GIST_RAW_LAYOUT,
        )

        derived_indices = list(allocator_meta["representation_out_indices"])
        failed_receipt = allocator_meta["failed_operation_cue"]
        if failed_receipt["status"] == "admitted":
            derived_indices.append(int(failed_receipt["out_index"]))
        if len(derived_indices) != len(set(derived_indices)):
            raise PolicyInputError("Allocator summary and cue positions overlap")
        derived_indices.sort()
        derived_messages = []
        for index in derived_indices:
            if not 0 <= index < len(allocator_messages):
                raise PolicyInputError("Allocator derived message index is out of range")
            message = dict(allocator_messages[index])
            if (
                set(message) != {"role", "content"}
                or message["role"] != "user"
                or not isinstance(message["content"], str)
                or not message["content"]
            ):
                raise PolicyInputError(
                    "Allocator summaries and cues must be explicit derived user text"
                )
            derived_messages.append(message)

        memory = pack_memory(
            store,
            view,
            self.tokenizer,
            tools=tools or None,
            max_chunk_tokens=self.packing.max_chunk_tokens,
            chunk_overlap=self.packing.chunk_overlap,
            max_chunks=self.packing.max_chunks,
            derived_workspace_prefix_messages=derived_messages,
        )
        if memory.chunks:
            raise PolicyInputError("Native Text summary must not create gist chunks")
        if memory.raw_source_indices != selected_sources:
            raise PolicyInputError(
                "Packed native source indices disagree with the allocator ledger"
            )

        rendered = list(raw_workspace_messages(store, view))
        prefix_length = 0
        while prefix_length < len(rendered) and rendered[prefix_length]["role"] == "system":
            prefix_length += 1
        rendered[prefix_length:prefix_length] = copy.deepcopy(derived_messages)
        normalized_allocator = [visible_message(message) for message in allocator_messages]
        if rendered != normalized_allocator:
            raise PolicyInputError(
                "Native packing cannot reproduce the allocator's final message order"
            )
        final_ids = native_ids(
            self.tokenizer,
            normalized_allocator,
            tools=tools or None,
            generation=True,
        )
        packed_ids = memory.system_input_ids + memory.workspace_input_ids
        if packed_ids != final_ids:
            raise PolicyInputError(
                "Native PackedMemory tokens disagree with the allocator-visible request"
            )
        raw_prompt_tokens = len(packed_ids)
        if (
            self._count(normalized_allocator, tools)
            != allocator_meta["total_raw_prompt_tokens"]
            or raw_prompt_tokens != allocator_meta["total_raw_prompt_tokens"]
        ):
            raise PolicyInputError(
                "Native PackedMemory token count disagrees with allocator accounting"
            )
        common_tokens = int(allocator_meta["common_raw_prompt_tokens"])
        raw_history_tokens = raw_prompt_tokens - common_tokens
        active_history_bytes = raw_history_tokens * self.kv_bytes_per_token
        if raw_history_tokens < 0 or (
            raw_history_tokens != allocator_meta["raw_history_tokens"]
            or active_history_bytes != allocator_meta["active_history_bytes"]
        ):
            raise PolicyInputError(
                "Native PackedMemory history cost disagrees with the allocator ledger"
            )
        if (
            active_history_bytes > self.policy_config.history_budget_bytes
            or active_history_bytes > self.policy_config.workspace_budget_bytes
        ):
            raise PackingBudgetError(
                "Native Text summary final input exceeds the shared B/W budget"
            )

        limit_receipt = self._require_final_limits(memory, max_new_tokens)
        summary_records = copy.deepcopy(allocator_counts.get("summary_records") or [])
        representation_refs = copy.deepcopy(allocator_meta["representation_refs"])
        summary_receipt = copy.deepcopy(allocator_meta["text_summary"])
        auxiliary_receipt = {
            **summary_receipt,
            "actor_generation": False,
            "submitted_to_executor": False,
            "retained_output_messages_charged_as_raw_workspace": True,
            "producer_calls_charged_separately": True,
            "representation_ref_count": len(representation_refs),
        }
        source_coverage = copy.deepcopy(allocator_meta["source_coverage"])
        if (
            source_coverage.get("semantic_fidelity") != "unknown"
            or source_coverage.get("complete_history_coverage") is not None
        ):
            raise PolicyInputError(
                "Text summary source coverage must leave semantic fidelity unknown"
            )

        # Keep the externally reported route identity owned by the route catalog.
        # This import stays local because the catalog constructs this controller.
        from .event_native_controls import describe_event_native_route

        metadata = copy.deepcopy(allocator_meta)
        metadata.update(
            {
                "event_native_text_summary_version": EVENT_NATIVE_TEXT_SUMMARY_VERSION,
                "event_native_policy_version": EVENT_NATIVE_POLICY_VERSION,
                "policy_source_commit": POLICY_SOURCE_COMMIT,
                "session_id": store.session_id,
                "task_id": task_id,
                "attempt_id": attempt_id,
                "parent_request_id": parent_request_id,
                "decision_key": decision_key,
                "decision_index": decision_index,
                "view_mode": EVENT_NATIVE_TEXT_SUMMARY_MODE,
                "mode": EVENT_NATIVE_TEXT_SUMMARY_MODE,
                "route_mode": EVENT_NATIVE_TEXT_SUMMARY_MODE,
                "route": describe_event_native_route(NATIVE_TEXT_S0_MODE),
                # ``SourceNeedsRuntime`` needs the legacy always-compress flag
                # to select the Text-summary allocator.  It is an internal
                # allocator control, not the identity of this no-gist native
                # route, so do not expose it as actor compression work.
                "compression_policy": None,
                "implementation_profile": TEXT_SUMMARY_IMPLEMENTATION_PROFILE,
                "history_representation": "text_summary",
                "requested_ratio": ratio,
                "max_new_tokens": max_new_tokens,
                "model_context": self.model_context,
                "configured_max_sequence_tokens": self.packing.max_sequence_tokens,
                "history_budget_definition": HISTORY_BUDGET_DEFINITION,
                "workspace_budget_definition": WORKSPACE_BUDGET_DEFINITION,
                "current_input_baseline": CURRENT_INPUT_BASELINE,
                "shared_allocation_budget_bytes": min(
                    self.policy_config.history_budget_bytes,
                    self.policy_config.workspace_budget_bytes,
                ),
                "raw_source_cutoff": cutoff,
                "common_input_source_indices": copy.deepcopy(
                    allocator_meta["common_source_indices"]
                ),
                "raw_source_indices": list(memory.raw_source_indices),
                "raw_event_ids": list(view.raw_event_ids),
                "gist_event_ids": [],
                "omitted_event_ids": list(view.omitted_event_ids),
                "mandatory_raw_event_ids": list(view.mandatory_raw_event_ids),
                "evidence_event_ids": [],
                "derived_workspace_prefix_messages": copy.deepcopy(derived_messages),
                "derived_workspace_source_indices": [],
                "derived_summary_message_count": len(
                    allocator_meta["representation_out_indices"]
                ),
                "derived_failed_operation_cue_count": int(
                    failed_receipt["status"] == "admitted"
                ),
                "summary_records": summary_records,
                "representation_refs": representation_refs,
                "source_coverage": source_coverage,
                "auxiliary_summary_receipt": auxiliary_receipt,
                "raw_prompt_tokens": raw_prompt_tokens,
                "actual_raw_history_tokens": raw_history_tokens,
                "actual_gist_tokens": 0,
                "actual_history_bytes": active_history_bytes,
                "logical_sequence_tokens": limit_receipt[
                    "logical_sequence_tokens"
                ],
                "physical_sequence_tokens": limit_receipt[
                    "physical_sequence_tokens"
                ],
                "per_ratio": {
                    str(candidate_ratio): {
                        **memory.costs(candidate_ratio),
                        "max_new_tokens": max_new_tokens,
                        "sequence_tokens": limit_receipt[
                            "physical_sequence_tokens"
                        ],
                        "logical_sequence_tokens": limit_receipt[
                            "logical_sequence_tokens"
                        ],
                        "history_gist_tokens": 0,
                        "history_raw_tokens": raw_history_tokens,
                        "history_total_tokens": raw_history_tokens,
                        "history_bytes": active_history_bytes,
                        "history_budget_bytes": self.policy_config.history_budget_bytes,
                        "workspace_budget_bytes": self.policy_config.workspace_budget_bytes,
                    }
                    for candidate_ratio in self.packing.ratios
                },
                "eligible_extraction": {
                    "status": "disabled_for_text_summary_representation",
                    "eligible_event_ids": [],
                    "eligible_source_indices": [],
                    "eligible_chunk_count": 0,
                    "retained_event_ids": [],
                    "retained_chunk_count": 0,
                    "backend_execution_required": False,
                    "charged_separately_from_active_history_bytes": False,
                },
                "native_allocator_receipt": {
                    "allocator": "SourceNeedsRuntime.apply",
                    "route_mode": SOURCE_NEEDS_TEXT_SUMMARY_ROUTE,
                    "compression_policy": allocator_meta["compression_policy"],
                    "allocator_task_id": store.session_id,
                    "message_order_reproduced": True,
                    "token_ids_reproduced": True,
                    "source_indices_reproduced": True,
                    "allocator_raw_prompt_tokens": allocator_meta[
                        "total_raw_prompt_tokens"
                    ],
                    "packed_raw_prompt_tokens": raw_prompt_tokens,
                    "final_message_sha256": _digest(normalized_allocator),
                    "zero_gist_chunks": True,
                },
                "pre_draft_retrieval": True,
                "recovery_stage": "pre_generation_text_summary_lexical",
                "post_draft_exact_recovery_applied": False,
                "lease_decisions": 0,
                "max_retrieved_events": 2,
            }
        )
        metadata["source_needs"] = copy.deepcopy(allocator_meta["source_needs"])
        metadata["source_needs"]["predictor_prompt_token_cap"] = self.s0_config[
            "predictor_prompt_token_cap"
        ]
        metadata["source_needs"]["predictor_completion_token_cap"] = self.s0_config[
            "predictor_completion_token_cap"
        ]
        metadata["source_needs"]["predictor_calls"] = 0
        metadata["text_summary"] = summary_receipt
        metadata["derived_workspace_provenance"] = {
            "summary_records": copy.deepcopy(summary_records),
            "failed_operation_cue": copy.deepcopy(failed_receipt),
            "source_event_ids_are_from_original_store": True,
            "derived_messages_are_not_original_source_messages": True,
            "derived_messages_claim_exact_raw_source_coverage": False,
            "semantic_fidelity": "unknown",
        }
        workspace = metadata["pre_generation_workspace"]
        workspace.update(
            allocator_renderer=workspace.get("renderer"),
            renderer="history_memory.packing.pack_memory",
            raw_source_indices=list(memory.raw_source_indices),
            raw_event_ids=list(view.raw_event_ids),
            derived_workspace_prefix_messages=copy.deepcopy(derived_messages),
            source_messages_immutable=True,
        )

        return PreparedEventNativeTextSummary(
            memory=memory,
            metadata=metadata,
            eligible_chunks=(),
            _owner=self._owner,
            _session_id=store.session_id,
            _decision_key=decision_key,
        )

    def _require_final_limits(
        self, memory: PackedMemory, max_new_tokens: int
    ) -> dict[str, int]:
        system_tokens = len(memory.system_input_ids)
        workspace_tokens = len(memory.workspace_input_ids)
        encoder_tokens = sum(len(chunk.token_ids) for chunk in memory.chunks)
        if system_tokens > self.packing.max_system_tokens:
            raise PackingBudgetError(
                f"System/tools need {system_tokens} tokens; budget is "
                f"{self.packing.max_system_tokens}"
            )
        if workspace_tokens > self.packing.max_workspace_tokens:
            raise PackingBudgetError(
                f"Native Text summary workspace needs {workspace_tokens} tokens; "
                f"budget is {self.packing.max_workspace_tokens}"
            )
        if encoder_tokens > self.packing.max_encoder_tokens:
            raise PackingBudgetError(
                f"Text summary unexpectedly needs {encoder_tokens} encoder tokens; "
                f"budget is {self.packing.max_encoder_tokens}"
            )
        logical = (
            memory.workspace_position_start
            + len(memory.workspace_input_ids)
            + max_new_tokens
        )
        physical = memory.costs(self.packing.ratios[0])[
            "resident_kv_tokens"
        ] + max_new_tokens
        if physical > self.packing.max_sequence_tokens:
            raise PackingBudgetError(
                f"Physical sequence needs {physical} tokens; budget is "
                f"{self.packing.max_sequence_tokens}"
            )
        if self.model_context is not None and logical > self.model_context:
            raise PackingBudgetError(
                f"Logical sequence needs {logical} positions; model context is "
                f"{self.model_context}"
            )
        if self.model_context is not None and physical > self.model_context:
            raise PackingBudgetError(
                f"Physical sequence needs {physical} tokens; model context is "
                f"{self.model_context}"
            )
        return {
            "system_tokens": system_tokens,
            "workspace_tokens": workspace_tokens,
            "encoder_tokens": encoder_tokens,
            "logical_sequence_tokens": logical,
            "physical_sequence_tokens": physical,
        }

    def _validate_request(self, payload, ratio, max_new_tokens):
        if not isinstance(payload, Mapping):
            raise TypeError("payload must be a mapping")
        allowed = {"session_id", "decision_key", "messages", "tools"}
        unknown = sorted(set(payload) - allowed)
        if unknown:
            raise PolicyInputError(
                f"Targets and unknown request fields are forbidden: {unknown!r}"
            )
        if not _positive_int(ratio) or ratio not in self.packing.ratios:
            raise ValueError(
                f"ratio must be one of checkpoint packing ratios "
                f"{self.packing.ratios!r}"
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
        raw_messages = payload.get("messages")
        if (
            not isinstance(raw_messages, Sequence)
            or isinstance(raw_messages, (str, bytes, bytearray))
            or not raw_messages
            or any(not isinstance(message, Mapping) for message in raw_messages)
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
            raise PolicyInputError(
                "Failed-operation observations require unique tool names"
            )
        tools_json = _canonical_json(tools)
        store = EventStore.from_messages(session_id, raw_messages)
        message_json = tuple(message.json_text for message in store.messages)
        return session_id, decision_key, store, tools, tools_json, message_json

    def _count(self, messages, tools):
        if not messages:
            return 0
        normalized = [visible_message(message) for message in messages]
        normalized_tools = (
            tuple(_json_snapshot(tool) for tool in tools) if tools else None
        )
        return len(
            native_ids(
                self.tokenizer,
                normalized,
                tools=normalized_tools,
                generation=True,
            )
        )


def _copy_reconsideration(value):
    return {
        "regenerate": bool(value["regenerate"]),
        "memory": value["memory"],
        "metadata": copy.deepcopy(value["metadata"]),
        "decision": copy.deepcopy(value["decision"]),
    }


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _session_identity(session_id: str) -> tuple[str, int]:
    """Recover the API-owned task and attempt without expanding its payload."""

    prefix, separator, attempt_text = session_id.rpartition("/attempt-")
    if (
        not separator
        or not prefix
        or "/" not in prefix
        or not attempt_text.isdigit()
        or str(int(attempt_text)) != attempt_text
    ):
        raise PolicyInputError(
            "session_id must use <benchmark>/<task_id>/attempt-<nonnegative integer>"
        )
    _, task_id = prefix.split("/", 1)
    if not task_id:
        raise PolicyInputError("session_id contains no task identity")
    return task_id, int(attempt_text)


def _positive_int(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value > 0


__all__ = [
    "EVENT_NATIVE_TEXT_SUMMARY_MODE",
    "EVENT_NATIVE_TEXT_SUMMARY_VERSION",
    "EventNativeTextSummaryController",
    "NATIVE_TEXT_S0_MODE",
    "PreparedEventNativeTextSummary",
    "SOURCE_NEEDS_TEXT_SUMMARY_ROUTE",
    "SUMMARY_RESOURCE_CONFIG",
    "TEXT_SUMMARY_IMPLEMENTATION_PROFILE",
]
