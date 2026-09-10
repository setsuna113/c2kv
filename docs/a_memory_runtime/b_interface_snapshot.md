# Shared history interface

Implementation: `python/history_memory/` in the `task/b-history-training` worktree.
This is a CPU data/packing contract. It is not wired into a trainer or serving
backend yet. A owns retrieval, workspace budgets and evidence leases; B consumes
the observable views produced by that policy.

## Immutable events

```python
from history_memory.events import EventStore

store = EventStore.from_messages(session_id, observable_messages)
event = store.event(event_id)
messages = [message.to_dict() for message in store.event_messages(event_id)]
```

Use an explicit run/task/rollout-scoped `session_id`. Inputs use OpenAI
`role`, `tool_calls`, and `tool_call_id`. Message JSON snapshots are owned and
immutable; `to_dict()` returns a copy. Legacy function/parts dialects require
a source adapter. Non-text inputs and unsupported developer-role templating
raise errors at packing rather than silently losing content.

`EventRecord` fields are `event_id`, `kind`, `source_indices`, `complete`,
`tool_call_ids`, `missing_tool_call_ids`. Kinds are `instruction`, `user`,
`assistant`, and `tool_event`. IDs are `session_id:m<source_message_index>`.
Each source message belongs to exactly one event. A call event owns the
assistant message and every matching result, including out-of-order parallel
results. `complete` means all results arrived; an error result also completes
a call. It does not mean the requested operation succeeded. An unmatched,
duplicate, or ambiguous result is an input error. Incomplete events stay raw.

A can map each event's `source_indices` to its existing legacy gist documents
using its own `BlockRef`. Old `checkpoint-1088` inputs must retain their existing
turn serialization. B's new event serialization is for freshly trained B/C
checkpoints only.

## Views and packing

```python
from history_memory.packing import select_view, pack_memory

view = select_view(store, recent_tool_events=1,
                   restored_event_ids=lease_event_ids,
                   pinned_event_ids=active_binding_event_ids)
packed = pack_memory(store, view, tokenizer, tools=visible_tools,
                     max_chunk_tokens=768, chunk_overlap=64)
```

`MemoryView(gist_event_ids, raw_event_ids)` must cover every visible event.
Unknown IDs, missing coverage and incomplete gist events are rejected. Explicit
raw/gist overlap is allowed and counted. `select_view` retains instructions,
the current user request, the event containing the latest message, pending
calls and the requested number of latest complete tool events. Additional
active bindings and recover/retain/evict decisions come from A. Release a raw
lease by omitting its ID on the next view; the event source remains unchanged.

The encoder receives an immutable typed event envelope containing the original
roles and call/result IDs. It does not receive current workspace state, the
target, future messages, or evaluator metadata. JSON object arguments are parsed
once for rendering; the source snapshot keeps its original representation.
Native Qwen templates omit call IDs, so the encoder envelope is necessary for
parallel result association. Raw workspace messages retain original source
order and use the native template. The native template's omission of raw call
IDs remains a property of that base-model protocol; A must retain the structured
messages for binding/recovery, and parallel-result protocol parity must be
checked when connecting the actual adapters.

Long events are split without dropping tokens. `EncoderChunk` records parent
`event_id`, `part_index`, `source_indices`, and half-open source token offsets.
The overlap is part of encoder cost. Chunks currently split the event token
stream; the model adapter must preserve their grouping and source mapping.
Exceeding `max_chunks`, `max_raw_tokens`, or a target budget raises
`PackingBudgetError`; no partial target or first-plus-tail selection is emitted.

`PackedMemory` provides `system_input_ids`, `workspace_input_ids`,
`raw_source_indices`, `chunks`, `gist_layout(ratio)`, `costs(ratio)`, and a small
`causal_mask(ratio, target_tokens=...)` reference. Tools and system policy remain
in the native raw prefix. Gist precedes the workspace; event and chunk order is
chronological. Dynamic-interleave positions use each source interval's end
minus one, offset by system length and preceding encoded source lengths.
Raw positions start after those source spans, not after the number of gist
KV entries. Ordinary queries see the past prefix/gists and causal raw/target
tokens. The tensor adapter should construct an equivalent efficient mask.

`chunk.encoding_key(parameter_version=..., ratio=...)` includes exact encoder
token IDs and excludes runtime position offsets. Cache pre-RoPE gist keys and
rotate on placement. Advance the parameter version and release cached training
graphs after every optimizer update. The packing module does not own or detach
autograd graphs. Actual tensor gradients and train/serve projection parity need
verification in the integration step.

`pack_target(tokenizer, assistant_message)` returns a complete native assistant
continuation. `training_sequence(packed, target_ids)` masks workspace labels
with `-100` and labels only that continuation. The trainer must normalize token
loss per decision and apply the matched decision weight.

## Dataset boundary

`dataset.py` and `agent/build_history_memory_data.py` consume normalized visible
conversation records. Assign and validate source/task/template/session splits
before expanding assistant decisions. Preserve tools, all assistant target
types, source provenance, and unchanged target JSON. The paired C/B API gives
the lifecycle planner only the visible `EventStore` and decision index; equal
decision targets, repetition counts and weights are shared by both arms.

Required row fields are `source`, `session_id`, `task_id`, `template_id`, `split`,
and `messages`; `tools` is optional. Event stores namespace the session as the
compact JSON pair `[source, session_id]`, while provenance retains the original
session ID. Exports include `event_store_session_id` for reconstruction. A
stateful planner must scope leases by `store.session_id` or reset on a session
change. `build_paired_records(rows, planner, repetitions=1)` calls the planner
once per decision, before repetition. The planner returns `LifecycleSelection`
or a complete `MemoryView`, which is validated against the current prefix.
The CLI exports static C records; it does not manufacture a lifecycle policy.

Token counters are not measurements of tensor bytes, HBM or wall time. Those
measurements belong to the runtime/training integration and cost calibration.
