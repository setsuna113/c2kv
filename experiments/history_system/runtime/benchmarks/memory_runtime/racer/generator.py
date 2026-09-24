"""Persistent CUDA history transport for the shared held-decision runner."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from urllib.request import Request

from history_memory.sglang_generator import (
    SGLangEventNativeGenerationResult, SGLangEventNativeError, SGLangTransportError)
from ..backend_capacity import BackendCapacityConstraints
from .allocator import PersistentMemory
from .capacity import HistoryCapacityInfeasible


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


class PersistentRacerGenerator:
    """Own physical session transport; the controller owns source and policy state.

    Hidden evidence remains only in the canonical text ledger after expiry. The
    engine removes its KV before the next decision. No failed session is rebuilt.
    """

    cache_trace_schema = "event-native-cache-trace-v1"
    decode_strategy = "persistent"
    session_cache_policy = "racer-persistent-transaction-v1"
    tool_transport_schema = "racer-tool-transport-v1"

    def __init__(self, native, tokenizer, config, *, backend=None, benchmark=None):
        self.native, self.tokenizer, self.config = native, tokenizer, config
        self.benchmark = benchmark
        if backend is None:
            import benchmarks
            directory = str(Path(__file__).resolve().parents[6] / "benchmarks")
            if directory not in benchmarks.__path__:
                benchmarks.__path__.append(directory)
            from benchmarks.backends.sglang import SglangBackend
            backend = SglangBackend(self._post_backend_json)
        self.backend = backend
        self._session_id = None
        self._logical_session_id = None
        self._source = []
        self._messages = []
        self._source_positions = []
        self._decision_key = None
        self._resolution = None
        self._calls = 0
        self._closed = False
        self._tool_binder = None
        self._active_tool_binding_context = None
        self._tool_attempts = []
        self._tool_decision_attempts = {}
        self._tool_binding_transport = {}
        self._pending_commit = None
        self._held_retention = None
        self._current_retention = None
        self.last_generation_trace = None

    def __getattr__(self, name):
        return getattr(self.native, name)

    def configure_tool_memory(self, spec, *, checkpoint=None, budget_tokens=None):
        """Install the backend-owned binder before the first persistent decision."""
        if self._calls or self._source or self._messages:
            raise RuntimeError("Tool memory must be configured before RACER generation")
        if self._tool_binder is not None:
            raise RuntimeError("Tool memory is already configured")
        from .tools import PersistentToolBinder
        self._tool_binder = PersistentToolBinder(
            self.backend, self.tokenizer,
            checkpoint=checkpoint, budget_tokens=budget_tokens)

    def bind_tool_plan(self, memory, plan, payload):
        """Bind the controller's frozen plan to a JSON-safe persistent memory."""
        if self._tool_binder is None:
            raise RuntimeError("Persistent RACER tool memory is not configured")
        session_id = payload.get("session_id")
        decision_key = payload.get("decision_key")
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("Persistent tool binding requires payload.session_id")
        if not isinstance(decision_key, str) or not decision_key:
            raise ValueError("Persistent tool binding requires payload.decision_key")
        if self._active_tool_binding_context is not None:
            raise RuntimeError("Persistent tool binding is not reentrant")
        context = {"session_id": session_id, "decision_key": decision_key}
        bucket = self._tool_decision_attempts.setdefault(decision_key, [])
        before = len(bucket)
        previous = copy.deepcopy(getattr(memory, "tool_plan", None))
        self._active_tool_binding_context = context
        try:
            bound = self._tool_binder.bind_tool_plan(memory, plan, payload)
        finally:
            self._active_tool_binding_context = None
        attempts = copy.deepcopy(bucket[before:])
        receipt = copy.deepcopy(bound.tool_plan)
        binding_id = receipt.get("binding_id")
        if attempts:
            receipt["transport"] = self._tool_transport_receipt(attempts)
            self._tool_binding_transport[binding_id] = copy.deepcopy(receipt["transport"])
        elif binding_id in self._tool_binding_transport:
            receipt["transport"] = copy.deepcopy(self._tool_binding_transport[binding_id])
        elif (isinstance(previous, dict)
              and previous.get("schema") == receipt.get("schema")
              and previous.get("source_messages_sha256")
                  == receipt.get("source_messages_sha256")
              and previous.get("source_tools_sha256")
                  == receipt.get("source_tools_sha256")
              and isinstance(previous.get("transport"), dict)):
            # Recovery rebuilds a PersistentMemory but reuses the exact plan.
            # Preserve the first materialization receipt without charging it again.
            receipt["transport"] = copy.deepcopy(previous["transport"])
        else:
            receipt["transport"] = self._tool_transport_receipt(())
        from dataclasses import replace
        return replace(bound, tool_plan=receipt)

    def prepare_tool_plan(self, plan, payload):
        """Freeze the tool source layout before history accounting and packing."""
        if self._tool_binder is None:
            raise RuntimeError("Persistent RACER tool memory is not configured")
        return self._tool_binder.prepare_plan(plan, payload)

    @contextmanager
    def decision_scope(self, *, session_id=None):
        if self._closed:
            raise SGLangEventNativeError("RACER session is closed after cleanup or failure")
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("RACER decision scope requires a nonempty logical session_id")
        native_session_id = "racer-" + hashlib.sha256(session_id.encode()).hexdigest()
        if self._session_id is None:
            self._logical_session_id = session_id
            self._session_id = native_session_id
            self._held_retention = None
            self._current_retention = None
            self.backend.open_history_session(self._session_id, self.native.timeout_seconds)
        elif self._logical_session_id != session_id or self._session_id != native_session_id:
            raise SGLangEventNativeError("A persistent adapter cannot switch tasks")
        try:
            yield
        except BaseException as failure:
            try:
                self.close_session()
            except Exception as cleanup:
                # Keep the decision failure as the reported cause; the cleanup error is secondary.
                failure.add_note(f"RACER session cleanup also failed: {type(cleanup).__name__}: {cleanup}")
            raise

    def _post_json(self, path, payload, timeout, *, retries=0):
        if retries != 0:
            raise ValueError("Persistent RACER transport never retries an unknown outcome")
        request = Request(self.native.upstream + path, data=_json(payload).encode(),
                          headers={"Content-Type": "application/json"}, method="POST")
        previous_timeout = self.native.timeout_seconds
        try:
            self.native.timeout_seconds = timeout
            # The engine answers a successful /close_session with an empty body.
            response, status = self.native._read_json(
                request, label="RACER " + path, allow_empty=path == "/close_session")
        finally:
            self.native.timeout_seconds = previous_timeout
        if status != 200:
            transaction = ((payload.get("c2kv_kv_memory_hint") or {}).get(
                "persistent_history_session") or {}).get("transaction") or {}
            phase = {"regenerate": "regeneration", "draft": "draft"}.get(transaction.get("phase"))
            if phase is not None:
                capacity = HistoryCapacityInfeasible.from_response(
                    response, decision_id=transaction.get("decision_id"), phase=phase)
                if capacity is not None:
                    raise capacity
            raise SGLangEventNativeError(f"RACER {path} returned HTTP {status}: {response}")
        return response

    def _post_backend_json(self, path, payload, timeout, *, retries=0):
        """Route only physical tool work through finite counters and receipts."""
        kind = None
        if path == "/v1/c2kv/extract" and payload.get("projection_set") == "tool":
            kind = "tool_extraction"
        elif (path == "/v1/c2kv/repair_extract"
              and (str(payload.get("repair_mode") or "").startswith("history_kv_")
                   or (self._active_tool_binding_context is not None
                       and payload.get("repair_mode") == "d_corr"))):
            kind = "tool_repair"
        if kind is None:
            return self._post_json(path, payload, timeout, retries=retries)
        if retries != 0:
            raise ValueError("Persistent RACER tool transport never retries an unknown outcome")
        if self._active_tool_binding_context is None:
            raise RuntimeError("Physical tool work requires an active frozen-plan binding")

        if kind == "tool_extraction":
            limit_name, counter_name = "max_tool_extraction_calls", "tool_extraction_calls_reserved"
        else:
            limit_name, counter_name = "max_tool_repair_calls", "tool_repair_calls"
        limit = getattr(self.native, limit_name, None)
        if type(limit) is not int or limit <= 0:
            raise RuntimeError(f"Persistent tool transport requires a finite {limit_name}")
        reserved = getattr(self.native, counter_name, 0)
        if type(reserved) is not int or reserved < 0:
            raise RuntimeError(f"Native {counter_name} is invalid")
        if reserved >= limit:
            raise SGLangEventNativeError(f"Finite {kind.replace('_', ' ')} cap is exhausted")
        # Reserve before submission: a lost response may still have consumed work.
        setattr(self.native, counter_name, reserved + 1)

        attempt_index = len(self._tool_attempts) + 1
        context = copy.deepcopy(self._active_tool_binding_context)
        journal = getattr(self.native, "_http_journal", None)
        request_record = {
            "schema": self.tool_transport_schema,
            "event": "request",
            "status": "started",
            "attempt_index": attempt_index,
            "kind": kind,
            "context": context,
            "request": copy.deepcopy(payload),
            "retries": 0,
        }
        if journal is not None:
            journal.append(request_record)
        response = None
        status = "failed"
        started_ns = time.perf_counter_ns()
        try:
            response = self._post_json(path, payload, timeout, retries=0)
            if isinstance(response, dict) and response.get("success", True) is True:
                status = "completed"
            return response
        finally:
            elapsed_ns = time.perf_counter_ns() - started_ns
            attempt = self._tool_attempt_receipt(
                kind, attempt_index, context, payload, response,
                status=status, elapsed_ns=elapsed_ns)
            self._tool_attempts.append(attempt)
            self._tool_decision_attempts.setdefault(context["decision_key"], []).append(attempt)
            if journal is not None:
                journal.append({
                    "schema": self.tool_transport_schema,
                    "event": "response",
                    "status": status,
                    "attempt_index": attempt_index,
                    "kind": kind,
                    "context": context,
                    "response": copy.deepcopy(response),
                    "usage_scope": ("actual server receipt" if status == "completed"
                                    else "unknown after failed or ambiguous submission"),
                    "retries": 0,
                })

    @staticmethod
    def _tool_attempt_receipt(
        kind, attempt_index, context, request, response, *, status, elapsed_ns,
    ):
        source = request.get("token_ids")
        if source is None:
            source = request.get("input_ids")
        source_tokens = len(source) if isinstance(source, list) else None
        result = {
            "kind": kind,
            "attempt_index": attempt_index,
            "status": status,
            "session_id": context["session_id"],
            "decision_key": context["decision_key"],
            "source_tokens": source_tokens,
            "usage_known": status == "completed",
            "elapsed_ns": elapsed_ns,
        }
        if status == "completed" and isinstance(response, dict):
            result.update({
                key: copy.deepcopy(response[key])
                for key in ("key_hash", "original_seq_len", "gist_len", "token_len",
                            "span_start", "span_end", "history_kv_method")
                if key in response
            })
            if "metadata" in response:
                result["response_metadata"] = copy.deepcopy(response["metadata"])
            if "telemetry" in response:
                result["response_telemetry"] = copy.deepcopy(response["telemetry"])
        return result

    def _tool_transport_receipt(self, attempts):
        rows = [copy.deepcopy(dict(row)) for row in attempts]
        return {
            "schema": self.tool_transport_schema,
            "attempted_tool_extraction_calls": sum(
                row.get("kind") == "tool_extraction" for row in rows),
            "completed_tool_extraction_calls": sum(
                row.get("kind") == "tool_extraction" and row.get("status") == "completed"
                for row in rows),
            "attempted_tool_repair_calls": sum(
                row.get("kind") == "tool_repair" for row in rows),
            "completed_tool_repair_calls": sum(
                row.get("kind") == "tool_repair" and row.get("status") == "completed"
                for row in rows),
            "unknown_usage_calls": sum(row.get("usage_known") is not True for row in rows),
            "attempts": rows,
        }

    def _consume_tool_cost(self, decision_key):
        rows = self._tool_decision_attempts.pop(decision_key, [])
        return self._tool_transport_receipt(rows)

    def _ledger(self, memory, phase):
        source = list(copy.deepcopy(memory.source_messages))
        if phase == "draft":
            if len(source) < len(self._source):
                raise SGLangEventNativeError("RACER source archive was truncated")
            for index, previous in enumerate(self._source):
                if source[index] != previous:
                    # Query-dependent tool definitions can replace only the
                    # protected prefix. The engine verifies the source frame.
                    if index >= memory.history_start_message_count:
                        raise SGLangEventNativeError("RACER executed source prefix changed")
                    self._messages[self._source_positions[index]] = copy.deepcopy(source[index])
            pending = self._pending_commit
            for row in source[len(self._source):]:
                self._source_positions.append(len(self._messages))
                rendered = copy.deepcopy(row)
                if pending is not None:
                    if row.get("role") != "assistant" or _action_signature(row) != _action_signature(pending["response"]):
                        raise SGLangEventNativeError("RACER next source does not echo the committed action")
                    if pending["resolution"] == "commit":
                        rendered = {"role": "assistant", "content": pending["raw_text"]}
                    pending = None
                self._messages.append(rendered)
            if self._pending_commit is not None and pending is not None:
                raise SGLangEventNativeError("RACER next decision lacks its committed source action")
            self._pending_commit = None
            self._source = source
            if memory.initial_s0_messages:
                self._messages.extend([{"role": "assistant", "content": ""},
                                       *copy.deepcopy(memory.initial_s0_messages)])
        elif phase == "regeneration":
            if source != self._source or not memory.recovery_messages:
                raise SGLangEventNativeError("RACER regeneration requires source-bound evidence")
            self._messages.extend([{"role": "assistant", "content": ""},
                                   *copy.deepcopy(memory.recovery_messages)])
        else:
            raise ValueError("Unknown RACER generation phase")
        count = (self._source_positions[memory.history_message_count - 1] + 1
                 if memory.history_message_count else 0)
        start = (self._source_positions[memory.history_start_message_count]
                 if memory.history_start_message_count < len(self._source_positions) else count)
        events = [{"message_index": index, "role": row["role"], "phase": "others"}
                  for index, row in enumerate(self._messages)]
        for index, row in enumerate(self._source):
            phase = (memory.source_event_phases[index] if memory.source_event_phases else
                     "tool" if row["role"] == "tool" else
                     "act" if row["role"] == "assistant" and row.get("tool_calls") else "others")
            events[self._source_positions[index]]["phase"] = phase
        return copy.deepcopy(self._messages), count, start, events

    def generate(self, memory, *, ratio, max_new_tokens, trace_context=None,
                 compression_chunks=None, paper_whole_full_kv_tokens=None):
        if not isinstance(memory, PersistentMemory):
            raise TypeError("Persistent backend requires a persistent source plan")
        context = dict(trace_context or {})
        phase = context.get("phase")
        decision_key = context.get("decision_key")
        if not self._session_id or not decision_key:
            raise SGLangEventNativeError("RACER generation needs an active decision scope")
        if self._calls >= self.native.max_generation_calls:
            raise SGLangEventNativeError("RACER generation cap reached")
        if phase == "regeneration" and decision_key != self._decision_key:
            raise SGLangEventNativeError("RACER recovery belongs to another decision")
        self.native._ensure_model_info()
        # Regeneration appends only internal evidence. A rejected admission can
        # undo that append without copying the complete archived transcript.
        ledger_length = len(self._messages)
        messages, count, start, events = self._ledger(memory, phase)
        evidence = memory.initial_s0_messages if phase == "draft" else memory.recovery_messages
        evidence_indices = list(range(len(messages) - len(evidence) - 1, len(messages))) if evidence else []
        index_map = {index: index for index in range(len(messages))}
        payload = {"model": self.native.expected_model_path, "messages": messages,
                   "tools": list(memory.source_tools), "stream": False, "logprobs": True,
                   "max_tokens": max_new_tokens, "temperature": 0,
                   "rid": context["attempt_uid"], "chat_template_kwargs": {"enable_thinking": False}}
        payload.update(copy.deepcopy(self.native.sampling_params))
        if self._tool_binder is not None and memory.tool_plan is not None:
            payload, count, start, index_map = self._tool_binder.stage_request(
                memory, payload, history_message_count=count,
                history_start_message_count=start,
                source_message_indices=self._source_positions)
            old_events = {index_map[row["message_index"]]: row for row in events}
            events = [{"message_index": index, "role": row["role"],
                       "phase": old_events.get(index, {}).get("phase", "others")}
                      for index, row in enumerate(payload["messages"])]
        spec = self.config.history_spec(max(1, self.config.history_budget_tokens - memory.recovery_tokens))
        indices = list(range(start, count))
        staged_messages = payload["messages"]
        history = {"spec": spec, "session_id": self._session_id,
                   "history_out_indices": indices, "history_message_count": count,
                   "history_start_message_count": start,
                   "history_text": _json([
                       staged_messages[i]
                       for i in range(min(count, len(staged_messages)))
                   ]) if indices else ""}
        arm = SimpleNamespace(name=self.config.receipt()["identity"], history_kv=spec,
                              kv_reuse=None, constrain_tools=False, repair=False)
        request = self.backend.prepare_chat(payload, arm, None, context={"history_kv": history})
        hint = request.setdefault("c2kv_kv_memory_hint", {})
        hint["history_kv_method"] = spec["method"]
        hint["history_kv_backend"] = spec["backend"]
        hint["history_kv_event_messages"] = events
        persistent = hint.setdefault("persistent_history_session", {"enabled": True})
        persistent["session_id"] = self._session_id
        transaction = {"decision_id": decision_key, "phase": "regenerate" if phase == "regeneration" else "draft"}
        if phase == "regeneration":
            transaction["resolution"] = "discard"
            persistent["recovery_append"] = {"enabled": True, "replace_previous_evidence": True,
                "source_message_indices": [index_map[self._source_positions[index]]
                    for index in (memory.native_evidence_source_indices or memory.recovered_source_indices)],
                "evidence_message_indices": [index_map[index] for index in evidence_indices]}
        elif self._resolution is not None:
            transaction["resolution"] = self._resolution
        if phase == "draft" and evidence:
            persistent["initial_s0_append"] = {
                "enabled": True, "replace_previous_evidence": True,
                "source_message_indices": [index_map[self._source_positions[index]]
                                           for index in memory.initial_s0_source_indices],
                "evidence_message_indices": [index_map[index] for index in evidence_indices],
            }
        if self.config.allocation == "racer_s0" and phase == "draft":
            persistent["racer_initial_allocation"] = {
                "schema": "racer-initial-allocation-v1",
                "event_ids": list(memory.initial_s0_event_ids),
                "source_message_indices": [index_map[self._source_positions[index]]
                                           for index in memory.initial_s0_source_indices],
                "protected_evidence": bool(evidence),
            }
        persistent["transaction"] = transaction
        persistent["history_budget_tokens"] = self.config.history_budget_tokens
        persistent["native_evidence_tokens"] = memory.recovery_tokens
        features = self.native.shadow_feature_config
        if features is not None and features.enabled:
            hint["shadow_features"] = {"enabled": True, "prefill_layer": features.prefill_layer}
        self._calls += 1
        self.last_generation_trace = {"schema": "racer-generation-trace-v1",
                                      "attempt_uid": context["attempt_uid"], "request": request,
                                      "racer_backend": self.config.receipt(), "status": "submitted"}
        journal = self.native._http_journal
        if journal is not None:
            journal.append({"schema": "racer-chat-http-v1", "event": "request", "request": request, "retries": 0})
        try:
            response = self._post_json("/v1/chat/completions", request, self.native.timeout_seconds)
            result = self._result(response, memory, context)
        except Exception as failure:
            self.last_generation_trace.update(status="failed", error={
                "type": type(failure).__name__, "message": str(failure)})
            if isinstance(failure, HistoryCapacityInfeasible):
                self.last_generation_trace["capacity_rejection"] = copy.deepcopy(failure.receipt)
                if failure.can_retain_draft(decision_key):
                    del self._messages[ledger_length:]
            if self._tool_binder is not None:
                # Binding work has already reached the engine even when chat
                # fails. Drain only this decision's unreported tool attempts.
                self.last_generation_trace["racer_tool_cost"] = self._consume_tool_cost(decision_key)
            if isinstance(failure, SGLangTransportError):
                self.backend.abort_history_request(context["attempt_uid"], self._session_id)
            raise
        finally:
            if journal is not None:
                journal.append({"schema": "racer-chat-http-v1", "event": "response",
                                "response": locals().get("response"), "retries": 0})
        self._decision_key = decision_key
        self._resolution = None
        self._held_retention = self._retention_receipt(response)
        self._current_retention = self._current_retention_receipt(response)
        self.last_generation_trace.update(status="completed", response=response)
        return result

    def _retention_receipt(self, response):
        """Keep held history; engine indices use the carrier-removed ledger frame."""
        report = (response.get("metadata") or {}).get("kv_memory_report") or {}
        held = (report.get("racer_transaction") or {}).get("regeneration_mandatory_history")
        if held is None:
            return None  # An engine without the receipt keeps the submit-and-see behaviour.
        tokens, indices = held.get("tokens"), held.get("source_message_indices")
        if (type(tokens) is not int or tokens < 0 or not isinstance(indices, list)
                or any(type(index) is not int for index in indices)
                or held.get("release") != "replaced_source_message"):
            raise SGLangEventNativeError("RACER held retention receipt is invalid")
        # The engine emits these indices after removing tool carriers. They
        # already address our internal ledger, before the binder's index_map.
        return {"tokens": tokens, "source_message_indices": sorted(indices)}

    def _current_retention_receipt(self, response):
        """Keep current history in the same carrier-removed ledger frame."""
        report = (response.get("metadata") or {}).get("kv_memory_report") or {}
        current = report.get("racer_current_mandatory_history")
        if current is None:
            return None
        if not isinstance(current, dict):
            raise SGLangEventNativeError("RACER current retention receipt is invalid")
        tokens, indices = current.get("tokens"), current.get("source_message_indices")
        if (type(tokens) is not int or tokens < 0 or not isinstance(indices, list)
                or any(type(index) is not int or index < 0 for index in indices)
                or len(set(indices)) != len(indices)
                or current.get("release") not in {"replaced_source_message", "never"}):
            raise SGLangEventNativeError("RACER current retention receipt is invalid")
        return {"tokens": tokens, "source_message_indices": sorted(indices),
                "release": current["release"]}

    def backend_capacity_constraints(self, *, session_id, decision_key, stage):
        """Expose one engine receipt in canonical source-message coordinates."""
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("RACER capacity requires a nonempty session_id")
        if not isinstance(decision_key, str) or not decision_key:
            raise ValueError("RACER capacity requires a nonempty decision_key")
        if stage not in {"draft", "regeneration"}:
            raise ValueError("RACER capacity stage must be draft or regeneration")
        if self._closed:
            raise SGLangEventNativeError("RACER session is closed")
        if self._logical_session_id is None:
            if stage == "regeneration":
                raise SGLangEventNativeError("RACER regeneration has no held decision")
            return None
        if session_id != self._logical_session_id:
            raise SGLangEventNativeError("RACER capacity belongs to another session")
        if stage == "regeneration":
            if decision_key != self._decision_key:
                raise SGLangEventNativeError("RACER regeneration belongs to another decision")
            receipt = self._held_retention
            provenance = "engine_held_checkpoint_receipt"
        else:
            if self._decision_key is not None and self._resolution not in {"commit", "discard"}:
                raise SGLangEventNativeError("RACER previous decision is unresolved")
            if decision_key == self._decision_key:
                raise SGLangEventNativeError("RACER draft reuses the previous decision")
            receipt = self._current_retention if self._resolution == "commit" else self._held_retention
            provenance = ("engine_current_resident_receipt" if self._resolution == "commit"
                          else "engine_held_checkpoint_receipt")
        if receipt is None or receipt["tokens"] == 0:
            return None
        ledger_to_source = {ledger: source for source, ledger in enumerate(self._source_positions)}
        mapped = set()
        for ledger_index in receipt["source_message_indices"]:
            source = ledger_to_source.get(ledger_index)
            if source is not None:
                mapped.add(source)
        # A draft carries its prior pending pages forward. Only a regeneration
        # with recovery_append replaces source messages and can interrupt them.
        release = receipt.get("release", "replaced_source_message")
        if stage == "draft" or not mapped:
            release = "never"
        return BackendCapacityConstraints(
            session_id=session_id, decision_key=decision_key, stage=stage,
            history_budget_tokens=self.config.history_budget_tokens,
            mandatory_history_tokens=receipt["tokens"],
            mandatory_source_indices=tuple(sorted(mapped)), release=release,
            provenance=provenance)

    def regeneration_capacity(self, memory):
        """Receipt when a regeneration of the held generation cannot fit, else None.

        The engine restores the held checkpoint and must keep its mandatory
        history (CommitKV pending pages) inside this request's history target,
        max(1, B - evidence), which also bounds the engine's effective target.
        When it cannot, the engine would reject the regeneration, so it is not
        submitted and the held generation stays committable.  Recovering the
        source of a protected page's message releases the protection.
        """
        held = self._held_retention
        if held is None or not isinstance(memory, PersistentMemory):
            return None
        target = max(1, self.config.history_budget_tokens - memory.recovery_tokens)
        recovered = sorted({self._source_positions[index]
                             for index in (memory.native_evidence_source_indices
                                           or memory.recovered_source_indices)})
        released = bool(set(recovered) & set(held["source_message_indices"]))
        mandatory = 0 if released else held["tokens"]
        if mandatory <= target:
            return None
        return {"schema": "racer-regeneration-capacity-v1",
                "status": "capacity_exhausted",
                "history_budget_tokens": self.config.history_budget_tokens,
                "native_evidence_tokens": memory.recovery_tokens,
                "history_target_tokens": target,
                "mandatory_history_tokens": held["tokens"],
                "mandatory_source_message_indices": held["source_message_indices"],
                "recovered_source_message_indices": recovered,
                "engine_request_submitted": False,
                "source": "engine_held_checkpoint_receipt"}

    def _result(self, response, memory, context):
        metadata = response.get("metadata") or {}
        report = metadata.get("kv_memory_report") or {}
        lifecycle = report.get("history_kv_lifecycle") or {}
        if (lifecycle.get("session_id") != self._session_id
                or lifecycle.get("full_history_reprefill_performed") is not False
                or lifecycle.get("persistent_session_enabled") is not True):
            raise SGLangEventNativeError("Missing verified persistent lifecycle receipt")
        if self.config.allocation == "racer_s0" and context.get("phase") == "draft":
            initial = report.get("racer_initial_allocation") or {}
            if (initial.get("schema") != "racer-initial-allocation-v1"
                    or initial.get("applied") is not True
                    or initial.get("backend_native_selection_preserved") is not True
                    or initial.get("event_ids") != list(memory.initial_s0_event_ids)
                    or type(initial.get("evidence_tokens")) is not int
                    or initial["evidence_tokens"] < 0):
                raise SGLangEventNativeError("Missing verified RACER initial protection receipt")
        generation = metadata.get("racer_generation") or {}
        ids = generation.get("output_token_ids")
        logprobs = generation.get("output_token_logprobs")
        if (not isinstance(ids, list) or not ids or any(type(i) is not int or i < 0 for i in ids)
                or not isinstance(logprobs, list) or len(ids) != len(logprobs)):
            raise SGLangEventNativeError("RACER requires original output token IDs and logprobs")
        if any(isinstance(p, bool) or not isinstance(p, (int, float)) or not math.isfinite(p) for p in logprobs):
            raise SGLangEventNativeError("RACER output logprobs must be finite")
        transaction = lifecycle.get("transaction") or report.get("racer_transaction") or {}
        if transaction.get("decision_id") != context["decision_key"]:
            raise SGLangEventNativeError("Missing RACER decision transaction receipt")
        accounting = generation.get("accounting") or {}
        resident = accounting.get("resident_prompt_tokens")
        history = accounting.get("active_history_tokens")
        evidence = accounting.get("native_evidence_tokens")
        combined = accounting.get("history_and_evidence_tokens")
        if (type(resident) is not int or type(history) is not int or history < 0
                or type(evidence) is not int or evidence < 0
                or type(combined) is not int or combined != history + evidence):
            raise SGLangEventNativeError("RACER requires actual KV token accounting")
        if combined > self.config.history_budget_tokens:
            raise SGLangEventNativeError("Served history plus native evidence exceeds the declared budget")
        shadow = generation.get("shadow_features")
        if self.native.shadow_feature_config is not None and self.native.shadow_feature_config.enabled:
            if not isinstance(shadow, dict) or shadow.get("schema") != "event-native-shadow-features-v1":
                raise SGLangEventNativeError("Requested detector features were not returned")
            prefill = shadow.get("prefill") or {}
            hidden = prefill.get("hidden")
            if (prefill.get("status") != "captured" or prefill.get("readout") != "decoder_layer_output"
                    or (prefill.get("position") or {}).get("kind") != "prompt_last"
                    or type(prefill.get("layer")) is not int or prefill["layer"] < 0
                    or not isinstance(hidden, list) or not hidden
                    or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x)
                           for x in hidden)):
                raise SGLangEventNativeError("RACER needs a captured final-prefill detector feature")
        usage = {"prompt_tokens": resident, "completion_tokens": len(ids), "total_tokens": resident + len(ids)}
        finish = response["choices"][0]["finish_reason"]
        if finish not in {"stop", "length", "tool_calls"}:
            raise SGLangEventNativeError("Persistent generation did not finish normally")
        stats = {"racer_backend": self.config.receipt(), "racer_served_usage": usage,
                 "racer_accounting": accounting, "kv_memory_report": report,
                 "shadow_features": shadow, "generation_calls": 1,
                 "eos_token_ids": list(self.native.eos_token_ids),
                 "source_archive_contains_internal_evidence": False}
        if self._tool_binder is not None:
            stats["racer_tool_cost"] = self._consume_tool_cost(context["decision_key"])
            stats["racer_tool_plan"] = copy.deepcopy(memory.tool_plan)
        return SGLangEventNativeGenerationResult(tuple(ids), "stop" if finish == "tool_calls" else finish,
                                                 tuple(logprobs), stats)

    def resolve_decision(self, response, *, result, record):
        selected = next((i for i, row in enumerate(record["generation_trace"]) if not row["discarded"]), None)
        last_completed = next((i for i in reversed(range(len(record["generation_trace"])))
                               if record["generation_trace"][i].get("status", "completed") == "completed"), None)
        changed = bool(record.get("commit_transform", {}).get("changed")
                       or record.get("commit_validation", {}).get("synthetic_abstention"))
        self._resolution = "discard" if changed or selected != last_completed else "commit"
        ids = result.token_ids
        if ids and result.finish_reason == "stop" and ids[-1] in self.native.eos_token_ids:
            ids = ids[:-1]
        committed = copy.deepcopy(response)
        if any(row.get("draft_protocol") == "acebench-text-actions-v1"
               for row in record["generation_trace"]):
            # ACE executes the original action text; parsed calls only serve
            # recovery queries and are absent from the next source archive.
            committed["tool_calls"] = []
        if self.benchmark == "toolsandbox" and committed.get("tool_calls"):
            # ToolSandbox rebuilds an executed tool-call message from its calls
            # and drops prose emitted with them (as raw_actor_history does for
            # the proxy). This only shapes the expected echo; a commit still
            # replays the raw generated text.
            committed["content"] = ""
        self._pending_commit = {"response": committed, "resolution": self._resolution,
            "raw_text": self.tokenizer.decode(list(ids), skip_special_tokens=False,
                                               clean_up_tokenization_spaces=False)}
        return {"schema": "racer-client-commit-v1", "resolution_on_next_decision": self._resolution,
                "selected_generation_index": selected, "executed_drafts": 0,
                "committed_action_prefilled_from_next_source": self._resolution == "discard"}

    def session_cache_info(self):
        result = {"policy": self.session_cache_policy,
                "session_id": self._logical_session_id,
                "native_session_id": self._session_id,
                "generation_calls": self._calls, "canonical_internal_messages": len(self._messages),
                "source_messages": len(self._source), "closed": self._closed}
        if self._tool_binder is not None:
            result["tool_transport"] = self._tool_transport_receipt(self._tool_attempts)
        return result

    def close_session(self):
        if self._session_id is not None and not self._closed:
            self.backend.close_history_session(self._session_id)
        self._closed = True
        self.native.close_session()


def _action_signature(message):
    calls = []
    for call in message.get("tool_calls") or []:
        function = call["function"]
        args = function.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except ValueError:
                pass
        calls.append({"name": function["name"], "arguments": args})
    return _json({"content": message.get("content") or "", "tool_calls": calls})
