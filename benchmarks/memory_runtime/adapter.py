"""Bridge immutable events and a measured budget into the existing proxy.

The 1088 adapter leaves the legacy turn encoder intact. Recovery is an exact
text packet; all raw suffix tokens are rebuilt by the normal chat request.
"""

from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable
from .tokenization import TOOL_SCHEMA_PROFILE, serving_tools

_shared_path = str(Path(__file__).resolve().parents[2] / "python")
sys.path.insert(0, _shared_path)
try:
    from history_memory.evidence import EVIDENCE_VERSION, evidence_message
    from history_memory.events import EventStore
    from history_memory.packing import native_ids
    from .policy import ConversationMemory, PolicyInputError, RuntimeConfig
finally:
    # python/agent is a different training package from the harness agent/
    # namespace. Loading this optional interface must not change its imports.
    sys.path.remove(_shared_path)

RUNTIME_VERSION = "a-event-runtime-v1"
MODES = {"legacy", "protect", "recover_once", "persistent", "no_gist", "full_shared"}


class RuntimeAdapter:
    def __init__(self, config: dict, token_counter: Callable[[list, Any], int]):
        self.mode = config["mode"]
        if self.mode not in MODES:
            raise ValueError(f"Unknown runtime mode: {self.mode}")
        self.run_id = config.get("run_id")
        if not isinstance(self.run_id, str) or not self.run_id:
            raise ValueError("Runtime config requires an explicit run_id")
        self.bytes_per_kv_token = config["bytes_per_kv_token"]
        if type(self.bytes_per_kv_token) is not int or self.bytes_per_kv_token <= 0:
            raise ValueError("bytes_per_kv_token must be a positive measured integer")
        self.config = RuntimeConfig(
            mode="protect" if self.mode == "legacy" else self.mode,
            history_budget_bytes=config["history_budget_bytes"],
            workspace_budget_bytes=config["workspace_budget_bytes"],
            lease_decisions=config.get("lease_decisions", 3),
            max_retrieved_events=config.get("max_retrieved_events", 2),
        )
        self._token_counter = token_counter
        self._states: dict[tuple, ConversationMemory] = {}
        self._lock = threading.RLock()

    @classmethod
    def from_config(cls, path: str, tokenizer_path: str) -> "RuntimeAdapter":
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)

        def count(messages, tools):
            # Recent Transformers versions return BatchEncoding by default;
            # its len() is the number of fields, not the token sequence length.
            return len(native_ids(tokenizer, messages, tools=serving_tools(tools), generation=True))

        return cls(json.loads(Path(path).read_text(encoding="utf-8")), count)

    def apply(self, messages, assembled, counts, eval_context, tools=None):
        with self._lock:
            return self._apply(messages, assembled, counts, eval_context, tools)

    def _apply(self, messages, assembled, counts, eval_context, tools):
        started = time.perf_counter()
        if not isinstance(eval_context, dict):
            raise PolicyInputError("A runtime needs explicit benchmark request context")
        if any(key in eval_context for key in ("gold", "oracle", "target_action", "hidden_state")):
            raise PolicyInputError("Privileged labels cannot enter runtime context")
        task_id = eval_context.get("task_id")
        run_id = eval_context.get("run_id", self.run_id)
        attempt = eval_context.get("attempt_id", eval_context.get("attempt"))
        if not isinstance(task_id, str) or not task_id or run_id != self.run_id or attempt is None:
            raise PolicyInputError("Require task_id, matching run_id, and explicit attempt/attempt_id")
        decision = eval_context.get("decision_id")
        if decision is None:
            if not {"user_turn", "step"} <= eval_context.keys():
                raise PolicyInputError("Require decision_id or explicit user_turn and step")
            decision = json.dumps([eval_context["user_turn"], eval_context["step"]])
        state_key = (run_id, task_id, str(attempt))
        # IDs use the explicit task identity so auxiliary text is identical
        # across paired arms. The state registry also isolates run and attempt.
        store = EventStore.from_messages(task_id, messages)
        last_anchor = max(
            (i for i, message in enumerate(messages) if message.get("role") in {"user", "tool"}),
            default=-1,
        )
        # The legacy proxy keeps the entire input block after the preceding
        # assistant, including consecutive user messages / parallel returns.
        source_cutoff = next(
            (i + 1 for i in range(last_anchor, -1, -1) if messages[i].get("role") == "assistant"),
            0,
        )
        cutoff = int(counts["current_start_out_index"])
        base = [dict(message) for message in assembled]
        raw = [message for message in base if not message.get("c2kv_key_hash")]
        prefix_raw = [message for message in base[:cutoff] if not message.get("c2kv_key_hash")]
        common_prefix = [message for message in prefix_raw if message.get("role") == "system"]
        common = common_prefix + [message for message in base[cutoff:] if not message.get("c2kv_key_hash")]
        common_tokens = self._token_counter(common, tools)

        if self.mode == "no_gist":
            base = common[:]
            cutoff = len(common_prefix)
            raw = common[:]
        elif self.mode != "full_shared" and any(m.get("role") != "system" for m in prefix_raw):
            raise PolicyInputError("1088 budget adapter requires plain gist history, without hybrid raw history")

        visible = {
            event.event_id for event in store.events
            if self.mode == "full_shared" or event.kind == "instruction"
            or all(index >= source_cutoff for index in event.source_indices)
        }
        packets: dict[tuple, dict | None] = {}
        costs: dict[tuple, int] = {}

        def packet_cost(event_ids):
            ids = tuple(event_ids)
            if ids not in costs:
                packet = evidence_message(store, ids)
                packets[ids] = packet
                # SGLang removes gist carriers before raw chat templating.
                # Measure the complete raw view, including template boundaries.
                view = [m for m in base[:cutoff] if not m.get("c2kv_key_hash")]
                if packet is not None:
                    view.append(packet)
                view.extend(m for m in base[cutoff:] if not m.get("c2kv_key_hash"))
                delta = self._token_counter(view, tools) - self._token_counter(raw, tools)
                if delta < 0:
                    raise ValueError("Evidence renderer produced a negative token delta")
                costs[ids] = delta * self.bytes_per_kv_token
            return costs[ids]

        selection = None
        if self.mode == "legacy":
            selected_ids = ()
        else:
            memory = self._states.setdefault(state_key, ConversationMemory(task_id, self.config))
            selection = memory.prepare(store, packet_cost, visible, str(decision))
            selected_ids = selection.selected_event_ids
        evidence_bytes = packet_cost(selected_ids)
        packet = packets[selected_ids]

        records = list(counts.get("compressed_records") or [])
        kept_indices = set()
        remaining = self.config.history_budget_bytes - evidence_bytes
        # Preserve the legacy doc-0 anchor, then prefer newer complete blocks.
        priority = ([0] + list(range(len(records) - 1, 0, -1))) if records else []
        for index in priority:
            size = int(records[index]["record"]["gist_len"]) * self.bytes_per_kv_token
            if size <= remaining:
                kept_indices.add(index)
                remaining -= size
        keep_out = {records[index]["out_index"] for index in kept_indices}
        discarded = [record for i, record in enumerate(records) if i not in kept_indices]
        out = []
        old_to_new = {}
        packet_index = None
        for index, message in enumerate(base):
            if index == cutoff and packet is not None:
                packet_index = len(out)
                out.append(packet)
            if message.get("c2kv_key_hash") and index not in keep_out:
                continue
            old_to_new[index] = len(out)
            out.append(message)
        if cutoff == len(base) and packet is not None:
            packet_index = len(out)
            out.append(packet)
        retained_records = []
        for index, record in enumerate(records):
            if index in kept_indices:
                retained_records.append({**record, "out_index": old_to_new[record["out_index"]]})
        gist_tokens = sum(int(record["record"]["gist_len"]) for record in retained_records)
        original_tokens = sum(int(record["record"].get("original_seq_len", 0)) for record in retained_records)
        full_raw_tokens = self._token_counter([m for m in out if not m.get("c2kv_key_hash")], tools)
        raw_history_tokens = full_raw_tokens - common_tokens
        if raw_history_tokens < 0:
            raise ValueError("Raw history accounting became negative")
        history_bytes = (gist_tokens + raw_history_tokens) * self.bytes_per_kv_token
        if self.mode != "full_shared" and history_bytes > self.config.history_budget_bytes:
            raise ValueError("Measured active history exceeds the frozen byte cap")
        inserted_system = not any(m.get("role") == "system" for m in messages)
        blocks = [{
            "key_hash": record["record"]["key_hash"],
            "source_indices": [i - int(inserted_system) for i in record["source_indices"]],
            "gist_tokens": int(record["record"]["gist_len"]),
        } for record in retained_records]
        selected_sources = {i for event_id in selected_ids for i in store.event(event_id).source_indices}
        metadata = {
            "version": RUNTIME_VERSION, "evidence_version": EVIDENCE_VERSION,
            "tool_schema_profile": TOOL_SCHEMA_PROFILE, "c2kv_tools_dump_expected": "full",
            "mode": self.mode, "run_id": run_id, "task_id": task_id,
            "attempt_id": attempt, "decision_id": str(decision),
            "bytes_per_kv_token": self.bytes_per_kv_token,
            "byte_geometry_verified_by_backend": False,
            "history_budget_bytes": self.config.history_budget_bytes,
            "workspace_budget_bytes": self.config.workspace_budget_bytes,
            "budget_applies": self.mode != "full_shared",
            "active_history_bytes": history_bytes,
            "evidence_bytes": evidence_bytes, "gist_tokens": gist_tokens,
            "raw_history_tokens": raw_history_tokens,
            "common_raw_prompt_tokens": common_tokens,
            "total_raw_prompt_tokens": full_raw_tokens,
            "selected_event_ids": list(selected_ids),
            "protected_event_ids": list(selection.protected_event_ids) if selection else [],
            "retrieved_event_ids": list(selection.retrieved_event_ids) if selection else [],
            "retained_event_ids": list(selection.retained_event_ids) if selection else [],
            "policy": dict(selection.metadata) if selection else {},
            "block_refs": blocks,
            "overlapping_gist_keys": [b["key_hash"] for b in blocks if selected_sources.intersection(b["source_indices"])],
            "evicted_gist_keys": [record["record"]["key_hash"] for record in discarded],
            "eviction_reason": "absolute_history_budget" if discarded else None,
            "evidence_out_index": packet_index,
            "suffix_policy": "fresh_raw_suffix_with_cached_unchanged_gist",
            "controller_wall_sec": time.perf_counter() - started,
        }
        updated = dict(counts)
        updated.update({
            "memory_runtime": metadata, "compressed_records": retained_records,
            "gist_tokens": gist_tokens, "original_tokens": original_tokens,
            "n_gist_messages": len(retained_records), "compressed": len(retained_records),
            "n_docs": len(retained_records),
            "dropped_docs": int(counts.get("dropped_docs", 0)) + len(discarded),
            "current_start_out_index": old_to_new.get(cutoff, len(out)),
            "history_raw": sum(m.get("role") != "system" and not m.get("c2kv_key_hash") for m in out[:old_to_new.get(cutoff, len(out))]),
        })
        denominator = updated.get("history_packed_original_tokens")
        if denominator is not None:
            updated["history_dropped_original_tokens"] = denominator - original_tokens
            updated["history_retained_fraction"] = original_tokens / denominator if denominator else None
        return out, updated
