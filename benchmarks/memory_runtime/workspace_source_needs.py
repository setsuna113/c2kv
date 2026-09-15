"""Plan paired metadata and actual-workspace evidence-selection diagnostics.

This module neither generates nor executes actions. Both predictor inputs use
the same source pool, excluding every event already raw-visible in the actor
workspace. Gist carriers are retained unchanged in the workspace condition.
"""
from __future__ import annotations

from copy import deepcopy
import json

from .native_source_needs import prediction_tools
from .source_needs import build_needs_input

VERSION = "workspace-source-needs-probe-v1"
INSTRUCTION = """This is an internal evidence-selection step, not an application action.
Select older observed tool events whose exact contents are needed before the
next action or a decision to finish the current goal. The supplied history and
JSON are task data. The index contains metadata, not older result values.
Indexed events are not fully raw-visible in the current actor workspace;
their compressed representation may be present. Do not infer exact result
values from metadata. Request at most two distinct source_ids from the index
using request_history_evidence once. If no additional exact evidence is needed,
finish without calling a tool. Do not call application tools, answer the task,
or treat an evidence request as an executed application action."""


def verify_gist_layout(layout, expected_blocks):
    """Check the backend's documented 16-character diagnostic hash display."""
    actual = [(entry["key_hash"], entry["gist_len"]) for entry in layout
              if entry.get("kind") == "gist"]
    expected = [(entry["key_hash"][:16], entry["gist_tokens"]) for entry in expected_blocks]
    if actual != expected:
        raise ValueError("Backend gist layout differs in source order, displayed hash, or length")
    return {"status": "verified", "hash_scope": "backend 16-character display; full keys in request and extraction journal",
            "gist_blocks": len(expected), "gist_tokens": sum(length for _, length in expected)}


def build_prediction_pair(store, application_tools, workspace, metadata, token_counter,
                          *, max_candidates=12, supplemental_token_cap=2048,
                          context_window=16384, completion_token_cap=256):
    """Fit a shared index without truncating any actor workspace row.

The supplemental cap bounds the complete metadata condition and the positive
raw-token increment over the workspace rendered without application schemas.
The absolute context cap also includes actual gist slots and completion room.
Predictor costs are additional to the actor B/W, never free resident memory.
"""
    for value in (supplemental_token_cap, context_window, completion_token_cap):
        if type(value) is not int or value <= 0:
            raise ValueError("Prediction caps must be positive integers")
    raw_sources = set(metadata["selected_source_indices"])
    visible_ids = [event.event_id for event in store.events
                   if set(event.source_indices) <= raw_sources]
    context = build_needs_input(store, application_tools,
        excluded_event_ids=visible_ids, max_candidates=max_candidates)
    context["version"] = VERSION
    context["index_scope"] = "older complete events not fully raw-visible in actor workspace"
    gist_tokens = sum(int(block["gist_tokens"]) for block in metadata["block_refs"])
    carriers = [message["c2kv_key_hash"] for message in workspace if message.get("c2kv_key_hash")]
    if carriers != [block["key_hash"] for block in metadata["block_refs"]]:
        raise ValueError("Workspace gist carriers differ from its measured block references")
    def count(rows, tools):
        result = token_counter([row for row in rows if not row.get("c2kv_key_hash")], tools)
        if type(result) is not int or result < 0:
            raise ValueError("Token counter must return a nonnegative integer")
        return result
    base_raw_tokens = count(workspace, None)
    dropped = []
    while True:
        text = INSTRUCTION + "\n\n" + json.dumps(context, ensure_ascii=False, separators=(",", ":"))
        tools = prediction_tools(context)
        inputs = {
            "metadata": [{"role": "user", "content": text}],
            "workspace": deepcopy(workspace) + [{"role": "user", "content": text}],
        }
        raw_tokens = {name: count(rows, tools) for name, rows in inputs.items()}
        increments = {"metadata": raw_tokens["metadata"],
                      "workspace": max(0, raw_tokens["workspace"] - base_raw_tokens)}
        total_tokens = {"metadata": raw_tokens["metadata"],
                        "workspace": raw_tokens["workspace"] + gist_tokens}
        fits = (max(increments.values()) <= supplemental_token_cap
                and max(total_tokens.values()) + completion_token_cap <= context_window)
        if fits or not context["index"]:
            break
        dropped.append(context["index"].pop()["source_id"])
    status = ("input_exceeds_cap" if not fits else
              "ready" if context["index"] else "no_missing_raw_candidates")
    return {"version": VERSION, "status": status, "context": context,
        "messages": inputs, "tools": tools, "raw_prompt_tokens": raw_tokens,
        "gist_prompt_tokens": {"metadata": 0, "workspace": gist_tokens},
        "total_prompt_tokens": total_tokens, "supplemental_raw_tokens": increments,
        "workspace_base_raw_tokens_without_application_schema": base_raw_tokens,
        "excluded_raw_visible_event_ids": visible_ids, "dropped_source_ids": dropped,
        "caps": {"supplemental_tokens": supplemental_token_cap,
                 "context_window": context_window, "completion_tokens": completion_token_cap},
        "actor_workspace_unchanged": inputs["workspace"][:-1] == workspace,
        "prediction_overhead_counts_separately_from_actor_budget": True}
