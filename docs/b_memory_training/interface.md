# Shared history interface

Implementation: `python/history_memory/` in the `task/b-history-training` worktree.
The CPU data/packing contract is consumed by `agent/train_history_memory.py`
and the differentiable `history_memory.runtime.HistoryMemoryModel`. A owns
retrieval, workspace budgets and evidence leases; B consumes the observable
views produced by the frozen shared policy. The legacy serving profile remains
separate from these newly trained event-native checkpoints.

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

`MemoryView(gist_event_ids, raw_event_ids, evidence_event_ids=())` must cover every visible event.
Unknown IDs, missing coverage and incomplete gist events are rejected. Explicit
raw/gist overlap is allowed and counted. `select_view` retains instructions,
the current user request, the event containing the latest message, pending
calls and the requested number of latest complete tool events. Additional
active bindings and recover/retain/evict decisions come from A. Additional
restored/pinned events outside that fixed workspace are marked as
`evidence_event_ids`, a subset of `raw_event_ids`. Release a raw
lease by omitting its ID on the next view; the event source remains unchanged.

The encoder receives an immutable typed event envelope containing the original
roles and call/result IDs. It does not receive current workspace state, the
target, future messages, or evaluator metadata. JSON object arguments are parsed
once for rendering; the source snapshot keeps its original representation.
Native Qwen templates omit call IDs, so the encoder envelope is necessary for
parallel result association. `evidence.py` is copied unchanged from A and emits
the shared `history-evidence-v1` packet. `raw_workspace_messages(store, view)`
places that single user packet before the ordinary current/recent workspace,
after leading system messages. Evidence-owned events appear only in the packet;
the remaining native messages keep original source order. Packing uses
system/tools prefix, gist KV, evidence packet, current raw, then the assistant
generation prefix. The packet preserves call IDs, roles, arguments, results,
completeness and source indices. It is included in token/budget accounting.

The raw layout profile is `event-native-evidence-v1`. A's old 1088 compatibility
layout retains its legacy current-message boundary and coarse turn gist. A
complete evidence event can then overlap a result still present in current raw
and with the old gist; A counts that duplication. This B layout does not change
those legacy boundaries and does not claim strict layout parity with 1088.
Only the evidence renderer is shared across those profiles. The new B/C
checkpoints must use the event-native profile at training and serving time.

Long events are split without dropping tokens. `EncoderChunk` records parent
`event_id`, `part_index`, `source_indices`, and half-open source token offsets.
The overlap is part of encoder cost. Chunks currently split the event token
stream; the tensor adapter preserves their grouping and source mapping.
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
tokens. The tensor adapter constructs the equivalent physical causal mask,
independently of the source positions used by RoPE.

`chunk.encoding_key(parameter_version=..., ratio=...)` includes exact encoder
token IDs and excludes runtime position offsets. Cache pre-RoPE gist keys and
rotate on placement. Advance the parameter version and release cached training
graphs after every optimizer update. The packing module does not own or detach
autograd graphs. `HistoryMemoryModel` shares differentiable extraction graphs
within one forward batch, releases them after that batch, and advances the
parameter version after each optimizer update. Native tokens use base QKV;
only gist embedding/QKV parameters are trainable. CPU tests cover nonzero gist
gradients, independent-versus-shared extraction gradient equivalence, source
position placement, and checkpointed-versus-direct decoder execution. A serving
adapter must consume this same projection and packing profile before these
checkpoints can be evaluated through A0/A1.

`pack_target(tokenizer, assistant_message)` returns a complete native assistant
continuation. `training_sequence(packed, target_ids)` masks workspace labels
with `-100` and labels only that continuation. The trainer normalizes token
loss per decision and applies the matched decision weight across accumulation
and all DDP ranks.

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
The legacy `--input ... --output ...` CLI mode exports static C records.
The `--output-dir` mode creates the paired prepared corpus from normalized rows
or the G-source adapters in `sources.py`. It uses the copied fixed A policy,
which receives only the visible prefix. The manifest records the tokenizer,
packing and policy configuration, source exclusions, actual mixture, paired
exposure counts, memory costs, and content digests of both corpus files.

`PreparedCorpus(path, tokenizer, arm)` reconstructs and checks each arm's packed
input without duplicating all session prefixes on disk. `training.py` supplies
the DDP optimizer loop and checkpoints containing the frozen base, trainable
FP32 gist weights, optimizer/scheduler, per-rank RNG, and exact data cursor.
See [h200.md](h200.md) for source preparation, the two-arm launcher, packaging,
and explicit resume commands.

Preparation token/byte estimates are not measurements of HBM or wall time.
Training additionally logs elapsed update time and CUDA peak allocation;
end-to-end serving latency and residency still require the A0/A1 evaluation.
