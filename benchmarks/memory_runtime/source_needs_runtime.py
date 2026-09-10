"""Shared native workspace allocation for C2KV and raw source-needs controls."""

from __future__ import annotations

import json
import time
from typing import Any

from .adapter import RuntimeAdapter, RUNTIME_VERSION, raw_source_cutoff
from .always_compress import (
    CapacityInfeasible, coverage_accounting, eligible_history_sources, ratio_accounting,
)
from .capacity import measure_full_history
from .policy import PolicyInputError
from .observed_state import STATE_VERSION, build_observed_state, fit_observed_state, state_message
from . import native_source_needs
from .source_needs import (
    NEEDS_VERSION, SourceRequest, build_needs_input, fit_prediction_input,
    lexical_source_ids, parse_source_request, prediction_messages,
)
from .tokenization import TOOL_SCHEMA_PROFILE
from history_memory.events import EventStore


SOURCE_NEEDS_ROUTES = {
    "ac_native_needs_lexical": ("c2kv", "lexical"),
    "ac_native_needs_lexical_pruned": ("c2kv", "lexical"),
    "ac_native_needs_typed": ("c2kv", "typed"),
    "raw_native_needs_lexical": ("raw", "lexical"),
    "raw_native_needs_typed": ("raw", "typed"),
    "ac_native_needs_tool": ("c2kv", "tool"),
    "raw_native_needs_tool": ("raw", "tool"),
    "ac_native_needs_none": ("c2kv", "none"),
    "raw_native_needs_none": ("raw", "none"),
    "ac_native_state_none": ("c2kv", "none"),
    "raw_native_state_none": ("raw", "none"),
}
STATE_ROUTES = frozenset({"ac_native_state_none", "raw_native_state_none"})


class SourceNeedsRuntime(RuntimeAdapter):
    """One controller; representation-specific filling of remaining history.

    Goal, latest complete action/observation, pending events and the common
    suffix are protected. Up to two requested older events precede history
    filling. C2KV reserves a complete gist block and fills with gist; raw fills
    with complete original events. All admission uses the actual native view.
    """

    def __init__(self, config, token_counter):
        route = config["mode"]
        self.history_representation, self.source_needs_strategy = SOURCE_NEEDS_ROUTES[route]
        canonical = "ac_native_workspace" if self.history_representation == "c2kv" else "raw_recency"
        super().__init__({**config, "mode": canonical}, token_counter)
        self.route_mode = route
        self.observed_state_enabled = route in STATE_ROUTES
        self.state_prompt_token_cap = config.get("state_prompt_token_cap")
        if self.observed_state_enabled and (type(self.state_prompt_token_cap) is not int
                                            or self.state_prompt_token_cap <= 0):
            raise ValueError("Observed-state routes require an explicit positive state prompt cap")
        if not self.always_compress or self.history_view_protocol != "fixed-budget-main":
            raise ValueError("Source-needs routes require always-compress-v1 and fixed-budget-main")
        if config.get("max_retrieved_events") != 2 or config.get("lease_decisions") != 0:
            raise ValueError("Source-needs v1 requires two source slots and no cross-decision lease")
        self.source_index_max_events = config["source_index_max_events"]
        self.predictor_prompt_token_cap = config["predictor_prompt_token_cap"]
        self.predictor_completion_token_cap = config["predictor_completion_token_cap"]
        if type(self.source_index_max_events) is not int or not 0 < self.source_index_max_events <= 12:
            raise ValueError("Source index must contain between one and twelve events")
        for value in (self.predictor_prompt_token_cap, self.predictor_completion_token_cap):
            if type(value) is not int or value <= 0:
                raise ValueError("Predictor caps must be positive integers")
        self._observed_prefixes = {}

    def apply(self, messages, assembled, counts, eval_context, tools=None, *,
              render_full=None, render_compressed=None, source_predictor=None):
        with self._lock:
            output, updated = self._apply_needs(messages, assembled, counts, eval_context, tools,
                                                render_compressed, source_predictor)
            if self.route_mode == "ac_native_needs_lexical_pruned":
                from .duplicate_gist import prune_duplicate_gist
                started = time.perf_counter()
                output, updated = prune_duplicate_gist(output, updated, tools, self._token_counter)
                elapsed = time.perf_counter() - started
                updated["memory_runtime"]["gist_pruning"]["controller_wall_sec"] = elapsed
                updated["memory_runtime"]["controller_wall_sec"] += elapsed
            return output, updated

    def _apply_needs(self, messages, full_messages, full_counts, context, tools,
                     render_compressed, source_predictor):
        started = time.perf_counter()
        if not isinstance(context, dict) or any(
                key in context for key in ("gold", "oracle", "target_action", "hidden_state")):
            raise PolicyInputError("Source-needs requires observable benchmark context")
        task_id = context.get("task_id")
        attempt = context.get("attempt_id", context.get("attempt"))
        if not isinstance(task_id, str) or not task_id or attempt is None:
            raise PolicyInputError("Require explicit task_id and attempt")
        if context.get("run_id", self.run_id) != self.run_id:
            raise PolicyInputError("Source-needs run identity mismatch")
        decision = context.get("decision_id")
        if decision is None:
            if not {"user_turn", "step"} <= context.keys():
                raise PolicyInputError("Require decision_id or user_turn and step")
            decision = json.dumps([context["user_turn"], context["step"]])
        state_key = (self.run_id, task_id, str(attempt))
        signature = tuple(json.dumps(m, sort_keys=True, ensure_ascii=False) for m in messages)
        previous = self._observed_prefixes.get(state_key, ())
        if signature[:len(previous)] != previous:
            raise PolicyInputError("Source-needs input must extend its observed prefix")
        store = EventStore.from_messages(task_id, messages)
        if any(message.get("role") == "developer" for message in messages):
            raise PolicyInputError("Source-needs v1 supports the legacy system/user/tool profile")
        source_cutoff = raw_source_cutoff(messages)
        full_cutoff = int(full_counts["current_start_out_index"])
        shift = len(full_messages) - len(messages)
        if shift not in (0, 1) or full_cutoff != source_cutoff + shift:
            raise PolicyInputError("Full renderer did not preserve source row boundaries")
        if any(message.get("c2kv_key_hash") for message in full_messages):
            raise PolicyInputError("Source-needs requires the same-prefix raw Full reference")
        common_prefix = [dict(m) for m in full_messages[:full_cutoff] if m.get("role") == "system"]
        suffix = [dict(m) for m in full_messages[full_cutoff:]]
        common = common_prefix + suffix
        common_sources = set(range(source_cutoff, len(messages))) | {
            i for event in store.events if event.kind == "instruction" for i in event.source_indices}
        full = measure_full_history(full_messages, full_counts, self._token_counter,
                                    tools, self.bytes_per_kv_token)
        common_tokens = self._token_counter(common, tools)
        if common_tokens != full["common_raw_prompt_tokens"]:
            raise PolicyInputError("Source-needs changed the Full common-input boundary")
        eligible = eligible_history_sources(store, source_cutoff)
        users = [event for event in store.events if event.kind == "user"]
        complete_tools = [event for event in store.events if event.kind == "tool_event" and event.complete]
        protected_ids = {event.event_id for event in store.events
                         if not event.complete or set(event.source_indices) & set(range(source_cutoff, len(messages)))}
        if users:
            protected_ids.add(users[-1].event_id)
        if complete_tools:
            protected_ids.add(complete_tools[-1].event_id)
        selected_sources = common_sources | {
            i for event in store.events if event.event_id in protected_ids for i in event.source_indices}
        visible_ids = {event.event_id for event in store.events
                       if set(event.source_indices) <= selected_sources}
        needs_input = build_needs_input(store, tools or [], excluded_event_ids=visible_ids,
                                       max_candidates=self.source_index_max_events)
        fit_input = native_source_needs.fit_prediction_input if self.source_needs_strategy == "tool" else fit_prediction_input
        fitted, input_receipt = fit_input(needs_input, self._token_counter,
                                         max_prompt_tokens=self.predictor_prompt_token_cap)
        requested = SourceRequest((), (), input_receipt["status"])
        prediction_wall_sec = 0.0
        if fitted is not None and input_receipt["status"] == "ready":
            if self.source_needs_strategy == "lexical":
                requested = SourceRequest(lexical_source_ids(store, fitted), (), "lexical_sources_ranked")
            elif self.source_needs_strategy == "typed":
                if not callable(source_predictor):
                    raise PolicyInputError("Typed source-needs requires an accounted predictor callback")
                prediction_started = time.perf_counter()
                prediction = source_predictor(prediction_messages(fitted), input_receipt["prompt_tokens"])
                prediction_wall_sec = time.perf_counter() - prediction_started
                requested = parse_source_request(
                    None if prediction.get("tool_calls") else prediction.get("content"), fitted)
            elif self.source_needs_strategy == "tool":
                if not callable(source_predictor):
                    raise PolicyInputError("Tool source-needs requires an accounted predictor callback")
                prediction_started = time.perf_counter()
                prediction = source_predictor(native_source_needs.prediction_messages(fitted),
                    input_receipt["prompt_tokens"], tools=native_source_needs.prediction_tools(fitted))
                prediction_wall_sec = time.perf_counter() - prediction_started
                requested = native_source_needs.parse_prediction(prediction, fitted)
            else:
                requested = SourceRequest((), (), "no_retrieval_control")

        compressed_counts, records = full_counts, []
        compressed_assembly_sec = 0.0
        if self.history_representation == "c2kv" and eligible:
            if not callable(render_compressed):
                raise PolicyInputError("C2KV source-needs requires a compressed renderer")
            compressed_started = time.perf_counter()
            compressed_messages, compressed_counts = render_compressed(messages)
            compressed_assembly_sec = time.perf_counter() - compressed_started
            if [dict(m) for m in compressed_messages[int(compressed_counts["current_start_out_index"]):]
                    if not m.get("c2kv_key_hash")] != suffix:
                raise PolicyInputError("Compressed renderer changed the native live suffix")
            records = list(compressed_counts.get("compressed_records") or [])
            if not records:
                raise CapacityInfeasible("Eligible history produced no complete gist block")

        state_fitted, state_receipt, state_rows, state_budget_dropped = None, None, [], []
        if self.observed_state_enabled:
            state_fitted, state_receipt = fit_observed_state(
                build_observed_state(store), self._token_counter,
                max_prompt_tokens=self.state_prompt_token_cap)
            if state_fitted and state_fitted["calls"]:
                state_rows = [state_message(state_fitted)]

        def native_rows(indices):
            return [dict(full_messages[i + shift]) for i in sorted(indices)
                    if i < source_cutoff and messages[i].get("role") != "system"]

        def raw_history_bytes(indices):
            count = self._token_counter(common_prefix + state_rows + native_rows(indices) + suffix, tools) - common_tokens
            if count < 0:
                raise PolicyInputError("Source-needs raw-history token delta became negative")
            return count * self.bytes_per_kv_token

        budget = min(self.config.history_budget_bytes, self.config.workspace_budget_bytes)
        priority = ([0] + list(range(len(records) - 1, 0, -1))) if records else []
        while True:
            required_bytes = raw_history_bytes(selected_sources)
            reserved_index, reserved_bytes = None, 0
            for index in priority:
                size = int(records[index]["record"]["gist_len"]) * self.bytes_per_kv_token
                if size > 0 and required_bytes + size <= budget:
                    reserved_index, reserved_bytes = index, size
                    break
            if required_bytes <= budget and (not records or reserved_index is not None):
                break
            if not state_rows:
                raise CapacityInfeasible("Protected native workspace and required gist exceed the history budget")
            state_budget_dropped.append(state_fitted["calls"].pop()["latest_result"]["source_id"])
            state_fitted["omitted_distinct_calls"] += 1
            state_rows = [state_message(state_fitted)] if state_fitted["calls"] else []
        admitted, skipped = [], []
        for event_id in requested.source_ids:
            trial = selected_sources | set(store.event(event_id).source_indices)
            trial_bytes = raw_history_bytes(trial)
            if trial_bytes + reserved_bytes <= budget:
                selected_sources = trial
                admitted.append(event_id)
            else:
                skipped.append({"event_id": event_id, "reason": "requested_source_budget",
                                "candidate_history_bytes": trial_bytes + reserved_bytes})
        selected_gist_indices, recency_ids = set(), []
        if records:
            remaining = budget - raw_history_bytes(selected_sources)
            order = [reserved_index] + [index for index in priority if index != reserved_index]
            for index in order:
                size = int(records[index]["record"]["gist_len"]) * self.bytes_per_kv_token
                if size <= remaining:
                    selected_gist_indices.add(index)
                    remaining -= size
        else:
            for event in reversed(store.events):
                if not event.complete or set(event.source_indices) <= selected_sources:
                    continue
                trial = selected_sources | set(event.source_indices)
                if raw_history_bytes(trial) <= budget:
                    selected_sources = trial
                    recency_ids.append(event.event_id)

        out, retained_records, blocks = list(common_prefix), [], []
        for index, record in enumerate(records):
            if index not in selected_gist_indices:
                continue
            retained_records.append({**record, "out_index": len(out)})
            out.append(dict(compressed_messages[record["out_index"]]))
            blocks.append({"key_hash": record["record"]["key_hash"],
                "source_indices": [i - shift for i in record["source_indices"]],
                "gist_tokens": int(record["record"]["gist_len"]),
                **({"packing_fragment_id": record["packing_fragment_id"]}
                   if "packing_fragment_id" in record else {})})
        state_out_index = len(out) if state_rows else None
        out.extend(state_rows)
        native_out_start = len(out)
        restored = native_rows(selected_sources)
        out.extend(restored)
        current_out_index = len(out)
        out.extend(suffix)
        raw_tokens = self._token_counter([m for m in out if not m.get("c2kv_key_hash")], tools)
        gist_tokens = sum(block["gist_tokens"] for block in blocks)
        active_bytes = (raw_tokens - common_tokens + gist_tokens) * self.bytes_per_kv_token
        if active_bytes > budget or (self.history_representation == "c2kv" and eligible and not gist_tokens):
            raise CapacityInfeasible("Final source-needs view violates native B/W or gist reservation")
        coverage = coverage_accounting(eligible_sources=eligible, raw_sources=selected_sources,
            retained_blocks=blocks, packing_fragments=(compressed_counts.get("history_packing_fragments")
                if records else []))
        selected_ids = [event.event_id for event in store.events
                        if set(event.source_indices) <= selected_sources]
        metadata = {
            "version": RUNTIME_VERSION, "mode": self.mode, "route_mode": self.route_mode,
            "run_id": self.run_id, "task_id": task_id, "attempt_id": attempt, "decision_id": str(decision),
            "compression_policy": self.compression_policy, "history_view_protocol": self.history_view_protocol,
            "tool_schema_profile": TOOL_SCHEMA_PROFILE, "c2kv_tools_dump_expected": "full",
            "bytes_per_kv_token": self.bytes_per_kv_token, "byte_geometry_verified_by_backend": False,
            "history_budget_bytes": self.config.history_budget_bytes,
            "workspace_budget_bytes": self.config.workspace_budget_bytes,
            "budget_applies": True, "workspace_budget_applies": True,
            "active_history_bytes": active_bytes, "evidence_bytes": (raw_tokens - common_tokens) * self.bytes_per_kv_token,
            "gist_tokens": gist_tokens, "raw_history_tokens": raw_tokens - common_tokens,
            "total_raw_prompt_tokens": raw_tokens, "common_raw_prompt_tokens": common_tokens,
            "selected_event_ids": selected_ids, "selected_source_indices": sorted(selected_sources),
            "common_source_indices": sorted(common_sources), "protected_event_ids": sorted(protected_ids),
            "retrieved_event_ids": admitted, "retained_event_ids": [], "recency_selected_event_ids": recency_ids,
            "block_refs": blocks, "source_coverage": coverage, "evidence_out_index": None,
            "no_eligible_history": not eligible, "eligible_history_count": len(eligible), "full_identity_bypass": False,
            "reserved_gist_bytes": reserved_bytes,
            "gist_reservation": {"required": bool(records), "reserved_gist_bytes": reserved_bytes,
                                 "satisfied": not records or gist_tokens > 0},
            "pre_generation_workspace": {"version": "source-needs-native-workspace-v1",
                "renderer": "same-prefix Full training renderer", "post_draft_regeneration": False,
                "restored_source_indices": [i for i in sorted(selected_sources) if i < source_cutoff
                                             and messages[i].get("role") != "system"],
                "native_workspace_out_indices": list(range(native_out_start, current_out_index))},
            "source_needs": {"version": native_source_needs.TOOL_NEEDS_VERSION
                if self.source_needs_strategy == "tool" else NEEDS_VERSION, "strategy": self.source_needs_strategy,
                "history_representation": self.history_representation,
                "input_receipt": input_receipt,
                "candidate_event_ids": [entry["source_id"] for entry in fitted["index"]] if fitted else [],
                "prediction_status": requested.status, "requested_event_ids": list(requested.source_ids),
                "typed_needs": [{"kind": kind, "source_ids": list(ids)} for kind, ids in requested.needs],
                "admitted_event_ids": admitted, "skipped_for_budget": skipped,
                "predictor_wall_sec": prediction_wall_sec,
                "admission_rule": "protected_native_then_requested_then_representation_fill",
                "admission_cost": "actual_native_raw_plus_retained_gist"},
            "compressed_assembly_wall_sec": compressed_assembly_sec,
        }
        metadata["compression_ratio"] = ratio_accounting(full, metadata)
        metadata["compression_ratio"]["includes_coverage_loss"] = bool(coverage["unrepresented_source_indices"])
        if self.observed_state_enabled:
            without_state = (out[:state_out_index] + out[state_out_index + 1:]
                             if state_out_index is not None else out)
            state_tokens = raw_tokens - self._token_counter(
                [message for message in without_state if not message.get("c2kv_key_hash")], tools)
            if state_tokens < 0:
                raise PolicyInputError("Observed-state incremental cost became negative")
            metadata["state_bytes"] = state_tokens * self.bytes_per_kv_token
            metadata["native_evidence_bytes"] = metadata["evidence_bytes"] - metadata["state_bytes"]
            metadata["observed_state"] = {"version": STATE_VERSION,
                "state_prompt_token_cap": self.state_prompt_token_cap,
                "fit_receipt": state_receipt, "dropped_for_workspace": state_budget_dropped,
                "admitted_calls": len(state_fitted["calls"]) if state_fitted else 0,
                "admitted_source_ids": [row["latest_result"]["source_id"] for row in state_fitted["calls"]]
                    if state_fitted else [],
                "state_out_index": state_out_index, "incremental_raw_tokens": state_tokens,
                "state_counts_as_complete_event_coverage": False,
                "admission_rule": "bounded_state_then_native_and_one_gist_reservation; drop_oldest_state_rows_if_infeasible"}
        original_tokens = sum(int(record["record"].get("original_seq_len", 0)) for record in retained_records)
        updated = {**compressed_counts, "memory_runtime": metadata,
            "compressed_records": retained_records, "gist_tokens": gist_tokens,
            "original_tokens": original_tokens, "n_gist_messages": len(blocks),
            "compressed": len(blocks), "n_docs": len(blocks),
            "current_start_out_index": current_out_index,
            "history_raw": len(restored) + len(state_rows),
            "dropped_docs": int(compressed_counts.get("dropped_docs", 0)) + len(records) - len(blocks)}
        packed = updated.get("history_packed_original_tokens")
        if packed is not None:
            updated["history_dropped_original_tokens"] = packed - original_tokens
            updated["history_retained_fraction"] = original_tokens / packed if packed else None
        self._observed_prefixes[state_key] = signature
        metadata["controller_wall_sec"] = time.perf_counter() - started - prediction_wall_sec - compressed_assembly_sec
        return out, updated
