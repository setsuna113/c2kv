"""Shared evidence recovery over persistent SGLang H2O/SnapKV sessions.

This adapter owns one internal chat ledger and one physical session. Only the
caller's observable messages enter EventStore. Rejected drafts never become
messages or tool calls: an empty assistant closes their generation header,
then source-only evidence is appended. The next external decision must advance
the baseline's history boundary past those internal notes before their lease
can expire. No session reset or full-history prefill fallback is provided.

Use ``generate(payload, decision_key=..., history_message_count=...)`` from a
benchmark adapter; the count is the ORIGINAL baseline's message boundary.
The result contains the final OpenAI response and a content-free receipt.
This is an integration API, not an experiment supervisor or model launcher.
"""
from __future__ import annotations

import copy
import hashlib
import json
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace

_RUNTIME = Path(__file__).resolve().parent / "runtime"
for _path in (_RUNTIME / "python", _RUNTIME / "benchmarks"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from history_memory.events import EventStore
from memory_runtime.recovery.evidence_units import (
    build_catalog, deduplicate_units, expand_units, render_units, unit_is_covered, _json_tree,
)
from memory_runtime.recovery.experiment_config import parse_gp_config
from memory_runtime.recovery.selection import prepare_selection_dependencies, select_candidates


class PortableRecoveryError(RuntimeError):
    """The persistent contract cannot be met; never rebuild silently."""


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _position_hash(positions):
    return hashlib.sha256(
        ",".join(map(str, positions)).encode()
    ).hexdigest()


class PortableRecoverySession:
    """One serialized task, with independent gate/selector and source provenance.

    ``backend`` is the existing SglangBackend. ``tokenizer`` must be the exact
    local serving tokenizer. G is deliberately absent from ``shared_rule``;
    an explicit non-current G is rejected instead of relabelling H2O as C2KV.
    The resident cap includes protected prefix/current tokens and appended
    evidence. It is an explicit caller contract, not an inferred B0 budget.
    """

    def __init__(self, backend, tokenizer, *, session_id, history_spec, switches,
                 max_task_generations, max_resident_prompt_tokens, backends=None,
                 timeout=600):
        self.gp = parse_gp_config(switches)
        if self.gp.get("selection_protocol", "legacy") != "legacy":
            raise ValueError("evidence_sets_v1 currently requires the C2KV GPRecoveryController")
        if self.gp["G"] != "current":
            raise ValueError("Portable S does not implement C2KV G; pass G=current")
        if self.gp["D"] != "candidate_rule":
            raise ValueError("Portable gate currently supports D=candidate_rule only")
        if self.gp.get("detector_calibration_telemetry") or self.gp.get("detector_threshold") is not None:
            raise ValueError("Portable recovery does not collect detector features or calibration telemetry")
        if self.gp["L"] != "next_decision":
            raise ValueError("Portable leases currently support L=next_decision only")
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("session_id must be nonempty")
        for name, value in (("max_task_generations", max_task_generations),
                            ("max_resident_prompt_tokens", max_resident_prompt_tokens)):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        self.spec = copy.deepcopy(dict(history_spec))
        if self.spec.get("backend") != "physical_eviction" or self.spec.get("method") not in {
            "h2o", "snapkv", "snapkv_persistent"
        }:
            raise ValueError("Portable recovery requires persistent physical H2O/SnapKV")
        if type(self.spec.get("target_tokens")) is not int or self.spec["target_tokens"] < 1:
            raise ValueError("history_spec.target_tokens must be a positive integer")
        self.backend, self.tokenizer = backend, tokenizer
        self.session_id, self.timeout = session_id, timeout
        self.max_task_generations = max_task_generations
        self.max_resident_prompt_tokens = max_resident_prompt_tokens
        self.backends = prepare_selection_dependencies(self.gp, backends)
        self._arm = SimpleNamespace(name="portable_recovery", history_kv=self.spec,
                                    kv_reuse=None, constrain_tools=False, repair=False)
        self._source = []
        self._messages = []
        self._source_positions = []
        self._canonical_ids = []
        self._resident_positions = []
        self._lease_end = 0
        self._opened = False
        self._failed = False
        self._closed = False
        self._generation_count = 0
        self._completed = {}
        self._source_span_positions = {}

    @property
    def shared_rule(self):
        return {key: copy.deepcopy(value) for key, value in self.gp.items() if key != "G"}

    def close(self):
        """Release the owned server session; never open a replacement session."""
        if self._opened and not self._closed:
            self.backend._post_json("/close_session", {"session_id": self.session_id}, self.timeout)
        self._closed = True

    def generate(self, payload, *, decision_key, history_message_count):
        """Return only the final action; duplicate completed decisions are cached.

        Consecutive retries with the same key do not consume R or expire notes.
        A different key with no newly committed assistant cannot expire evidence
        if the original baseline boundary still leaves it current; fail before
        HTTP rather than enlarging that boundary or keeping raw notes forever.
        """
        if not isinstance(decision_key, str) or not decision_key:
            raise ValueError("decision_key must be nonempty")
        fingerprint = _json([payload, history_message_count])
        if decision_key in self._completed:
            previous, result = self._completed[decision_key]
            if previous != fingerprint:
                raise PortableRecoveryError("PORTABLE_DECISION_KEY_REUSED_WITH_DIFFERENT_INPUT")
            return copy.deepcopy(result)
        if self._closed or self._failed:
            raise PortableRecoveryError("PORTABLE_SESSION_CLOSED_OR_FAILED")
        source = copy.deepcopy(payload.get("messages") or [])
        for message in source:
            if message.get("content") is not None and not isinstance(message.get("content"), str):
                raise ValueError("Portable recovery currently requires text-only source messages")
            for call in message.get("tool_calls") or []:
                if not isinstance((call.get("function") or {}).get("arguments"), str):
                    raise ValueError("Portable tool-call arguments must be JSON strings")
        if type(history_message_count) is not int or not 0 <= history_message_count <= len(source):
            raise ValueError("history_message_count must be an original source-message boundary")
        if source[:len(self._source)] != self._source or len(source) < len(self._source):
            raise PortableRecoveryError("PORTABLE_SOURCE_PREFIX_MISMATCH")
        if payload.get("stream") or payload.get("n", 1) != 1:
            raise ValueError("Portable recovery requires stream=false and n=1")
        if payload.get("session_params") or payload.get("c2kv_kv_memory_hint"):
            raise ValueError("Portable recovery owns session_params and KV hints")
        # Stage all source and boundary changes before the first transport.
        messages = copy.deepcopy(self._messages)
        positions = list(self._source_positions)
        for message in source[len(self._source):]:
            positions.append(len(messages))
            messages.append(copy.deepcopy(message))
        cutoff = positions[history_message_count - 1] + 1 if history_message_count else 0
        if self._lease_end > cutoff:
            raise PortableRecoveryError(
                "PORTABLE_NEXT_DECISION_LEASE_STILL_CURRENT: original baseline boundary "
                "has not advanced past recovery evidence; commit the previous final "
                "assistant and its observation before a new decision"
            )
        store = EventStore.from_messages(self.session_id, source)
        eligible = {event.event_id for event in store.events if event.complete
                    and event.kind != "instruction"
                    and max(event.source_indices) < history_message_count}
        catalog = build_catalog(store, self.tokenizer, self.gp["U"])
        visible = []
        calls, rounds = [], []
        goal = next((str(message.get("content") or "") for message in reversed(source)
                     if message.get("role") == "user"), "")
        if self._generation_count >= self.max_task_generations:
            raise PortableRecoveryError("PORTABLE_TASK_GENERATION_LIMIT")
        # Token-prefix validation occurs before opening or mutating the session.
        self._shape(payload, messages, cutoff)
        self._source, self._source_positions, self._messages = source, positions, messages
        previous_lease_end, self._lease_end = self._lease_end, 0
        try:
            response = self._generate_one(payload, cutoff, calls)
            for recovery_round in range(1, self.gp["R"] + 1):
                candidates = [unit for unit in catalog if unit.event_id in eligible
                              and not unit_is_covered(unit, visible)
                              and not self._unit_is_raw(unit, positions, payload)]
                receipt = {"round": recovery_round, "gate_type": "candidate_rule",
                           "candidate_count": len(candidates), "regenerate": False}
                rounds.append(receipt)
                if not candidates:
                    receipt["reason"] = "no_new_source_units"
                    break
                if self._generation_count >= self.max_task_generations:
                    receipt["reason"] = "shared_task_generation_limit"
                    break
                draft = response["choices"][0]["message"]
                selected, selection_receipt = select_candidates(
                    candidates, goal=goal, draft_text=str(draft.get("content") or ""),
                    draft_tool_calls=draft.get("tool_calls") or [], config=self.gp,
                    backends=self.backends)
                receipt["selection"] = selection_receipt
                expanded = deduplicate_units(unit for unit in expand_units(
                    selected, catalog, store, self.gp["B"])
                    if unit.event_id in eligible and not unit_is_covered(unit, visible)
                    and not self._unit_is_raw(unit, positions, payload))
                if not expanded:
                    receipt["reason"] = "no_new_source_units"
                    break
                evidence = list(render_units(expanded, store, self.gp["P"], self.gp["order"]))
                trial = [*self._messages, {"role": "assistant", "content": ""}, *evidence]
                _, ids = self._shape(payload, trial, cutoff)
                delta = len(ids) - len(self._canonical_ids)
                if len(self._resident_positions) + delta > self.max_resident_prompt_tokens:
                    receipt["reason"] = "resident_prompt_cap"
                    break
                self._messages = trial
                self._lease_end = len(trial)
                receipt.update(reason="source_units_appended", regenerate=True,
                               appended_units=[unit.to_receipt() for unit in expanded],
                               appended_prompt_tokens=delta)
                visible.extend(expanded)
                response = self._generate_one(payload, cutoff, calls)
        except BaseException:
            # A transport may have committed even if its response was lost.
            # Such a session cannot be safely retried or silently reconstructed.
            self._failed = True
            raise
        result = {"response": response, "receipt": {
            "schema": "portable-shared-recovery-v1", "session_id": self.session_id,
            "decision_key": decision_key, "backend": self.spec["method"],
            "shared_rule": self.shared_rule, "c2kv_G_applied": False,
            "source_message_count": len(source), "internal_message_count": len(self._messages),
            "source_history_message_count": history_message_count,
            "internal_history_message_count": cutoff,
            "released_previous_evidence": bool(previous_lease_end),
            "archive_contains_recovery_copies": False, "drafts_submitted": 0,
            "raw_visibility": "rendered_source_value_spans_v1",
            "resident_prompt_cap": self.max_resident_prompt_tokens,
            "resident_prompt_cap_scope": "prompt_only_excludes_decode_and_allocator_padding",
            "generation_count": len(calls), "task_generation_count": self._generation_count,
            "rounds": rounds, "calls": calls,
        }}
        self._completed[decision_key] = (fingerprint, copy.deepcopy(result))
        return result

    def _render_text(self, payload, messages, *, generation=True):
        kwargs = dict(payload.get("chat_template_kwargs") or {})
        kwargs.setdefault("enable_thinking", False)
        return self.tokenizer.apply_chat_template(
            messages, tools=payload.get("tools"), tokenize=False,
            add_generation_prompt=generation, **kwargs)

    def _render_ids(self, payload, messages, *, generation=True):
        text = self._render_text(payload, messages, generation=generation)
        return list(self.tokenizer.encode(text, add_special_tokens=False))

    def _shape(self, payload, messages, cutoff):
        indices = [i for i, message in enumerate(messages[:cutoff])
                   if message.get("role") != "system"]
        context = {"history_kv": {
            "spec": self.spec, "session_id": self.session_id,
            "history_out_indices": indices, "history_message_count": cutoff,
            # Used only to determine whether a physical history span exists.
            "history_text": _json([messages[i] for i in indices]) if indices else "",
        }}
        request = self.backend.prepare_chat(
            {**copy.deepcopy(payload), "messages": copy.deepcopy(messages)},
            self._arm, None, context=context)
        ids = self._render_ids(request, request["messages"])
        if ids[:len(self._canonical_ids)] != self._canonical_ids:
            raise PortableRecoveryError("PORTABLE_CHAT_TEMPLATE_PREFIX_MISMATCH")
        return request, ids

    def _unit_is_raw(self, unit, source_positions, payload):
        positions = self._unit_token_positions(unit, source_positions, payload)
        return positions.issubset(self._resident_positions)

    def _unit_token_positions(self, unit, source_positions, payload):
        """Map provenance to original template token positions, never text search.

        Snapshot windows use JSON offsets, including JSON string escaping. Only
        model-visible source values (content, function name, arguments) project
        into KV; OpenAI transport keys/IDs do not. Template transformations must
        preserve the source substring exactly or this method fails explicitly.
        """
        if unit.unit_id in self._source_span_positions:
            return self._source_span_positions[unit.unit_id]
        full_text = self._render_text(payload, self._messages)
        encoded = self.tokenizer(full_text, return_offsets_mapping=True, add_special_tokens=False)
        if list(encoded["input_ids"]) != self._canonical_ids:
            raise PortableRecoveryError("PORTABLE_SOURCE_TOKENIZER_FRAME_MISMATCH")
        offsets = encoded.get("offset_mapping")
        if offsets is None or len(offsets) != len(self._canonical_ids):
            raise PortableRecoveryError("PORTABLE_SOURCE_TOKEN_OFFSETS_REQUIRED")
        result = set()
        for span in unit.provenance:
            source = self._source[span.source_index]
            if span.container_path:
                parts = [(span.container_path, span.char_start, span.char_end)]
            else:
                snapshot = json.dumps(source, ensure_ascii=False, allow_nan=False)
                parts = []
                pending = [_json_tree(snapshot)]
                while pending:
                    node = pending.pop()
                    pending.extend(node.children)
                    path = node.path
                    represented = path == ("content",) or (
                        len(path) == 4 and path[0] == "tool_calls" and path[2] == "function"
                        and path[3] in {"name", "arguments"})
                    if not represented or node.value_type != "string":
                        continue
                    value = source
                    for key in path:
                        value = value[key]
                    cursor = node.start + 1
                    covered = []
                    for index, char in enumerate(value):
                        next_cursor = cursor + len(json.dumps(char, ensure_ascii=False)[1:-1])
                        if cursor < span.char_end and next_cursor > span.char_start:
                            covered.append(index)
                        cursor = next_cursor
                    if covered:
                        parts.append((path, covered[0], covered[-1] + 1))
            for path, start, end in parts:
                changed = copy.deepcopy(self._messages)
                container = changed[source_positions[span.source_index]]
                for key in path[:-1]:
                    container = container[key]
                value = container[path[-1]]
                if not isinstance(value, str) or not 0 <= start <= end <= len(value):
                    raise PortableRecoveryError("PORTABLE_SOURCE_SPAN_INVALID")
                if start == end:
                    continue
                marker = "PORTABLESPAN" + hashlib.sha256(
                    _json([unit.unit_id, path, start, end]).encode()).hexdigest()
                if marker in full_text:
                    raise PortableRecoveryError("PORTABLE_SOURCE_MARKER_COLLISION")
                container[path[-1]] = value[:start] + marker + value[end:]
                marked = self._render_text(payload, changed)
                if marked.count(marker) != 1:
                    raise PortableRecoveryError("PORTABLE_SOURCE_SPAN_NOT_RENDERED_ONCE")
                before, after = marked.split(marker)
                if before + value[start:end] + after != full_text:
                    raise PortableRecoveryError("PORTABLE_SOURCE_SPAN_TEMPLATE_TRANSFORMED")
                char_start, char_end = len(before), len(before) + end - start
                touched = {i for i, (left, right) in enumerate(offsets)
                           if left < char_end and right > char_start}
                if not touched:
                    raise PortableRecoveryError("PORTABLE_SOURCE_SPAN_HAS_NO_TOKENS")
                result.update(touched)
        self._source_span_positions[unit.unit_id] = result
        return result

    def _generate_one(self, payload, cutoff, calls):
        request, ids = self._shape(payload, self._messages, cutoff)
        if not self._opened:
            self.backend.open_history_session(self.session_id, self.timeout)
            self._opened = True
        self._generation_count += 1
        started = time.monotonic()
        response = self.backend._post_json("/v1/chat/completions", request, self.timeout)
        elapsed = time.monotonic() - started
        choices = response.get("choices") if isinstance(response, dict) else None
        if not choices or not isinstance(choices[0].get("message"), Mapping):
            raise PortableRecoveryError("PORTABLE_INVALID_GENERATION_RESPONSE")
        if choices[0].get("finish_reason") in {"abort", "error"}:
            raise PortableRecoveryError("PORTABLE_GENERATION_ABORTED")
        report = (response.get("metadata") or {}).get("kv_memory_report") or {}
        lifecycle = report.get("history_kv_lifecycle") or {}
        if (lifecycle.get("session_id") != self.session_id
                or lifecycle.get("full_history_reprefill_performed") is not False
                or lifecycle.get("persistent_session_enabled") is not True):
            raise PortableRecoveryError("PORTABLE_PERSISTENT_LIFECYCLE_ECHO_REQUIRED")
        positions = self._resident_positions + list(range(len(self._canonical_ids), len(ids)))
        physical = report.get("history_kv_physical_eviction")
        if physical is not None:
            if not physical.get("success"):
                raise PortableRecoveryError("PORTABLE_PHYSICAL_EVICTION_FAILED")
            start = int(physical["protected_prefix_tokens"])
            end = start + int(physical["history_tokens"])
            selected = physical["selected_history_indices"]
            if (not 0 <= start <= end <= len(positions) or selected != sorted(set(selected))
                    or any(type(i) is not int or not 0 <= i < end - start for i in selected)):
                raise PortableRecoveryError("PORTABLE_INVALID_EVICTION_LEDGER")
            positions = positions[:start] + [positions[start + i] for i in selected] + positions[end:]
        elif request.get("c2kv_kv_memory_hint", {}).get("history_kv_eviction"):
            raise PortableRecoveryError("PORTABLE_PHYSICAL_EVICTION_ECHO_REQUIRED")
        summary = lifecycle.get("resident_position_summary") or {}
        if summary.get("count") != len(positions) or summary.get("sha256") != _position_hash(positions):
            raise PortableRecoveryError("PORTABLE_RESIDENT_POSITION_HASH_MISMATCH")
        if len(positions) > self.max_resident_prompt_tokens:
            raise PortableRecoveryError("PORTABLE_RESIDENT_PROMPT_CAP_EXCEEDED")
        calls.append({"elapsed_seconds": elapsed, "usage": copy.deepcopy(response.get("usage")),
                      "canonical_delta_tokens": len(ids) - len(self._canonical_ids),
                      "resident_prompt_tokens": len(positions),
                      "kv_memory_report": copy.deepcopy(report)})
        self._canonical_ids, self._resident_positions = ids, positions
        return response
