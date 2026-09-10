# ACEBench deployment patch

Upstream pin: **ACEBench/ACEBench @ 56dd66c** (`update readme`; shallow clone
`tmp/baselines/acebench`, 2026-09-05).

`0001-endpoint-env-and-model-registry.patch` makes the harness usable
against an OpenAI-compatible endpoint under an arbitrary served model name.
Upstream keys every client on the model NAME (`"gpt" in name` -> `GPT_*`,
`deepseek`, `qwen`, `kimi`; anything else raises `Unknown model name` or
leaves `base_url` unbound) and only names listed in
`model_inference/inference_map.py` are runnable at all.

* `inference_map.py` — names in `ACEBENCH_API_MODELS` (comma-separated) are
  registered as `APIModelInference`.  `CommonInference` is lazy, so an
  API-only C2KV run does not import the upstream local-vLLM implementation;
  selecting an upstream local-model name still imports vLLM at construction.
* `apimodel_inference.py`, `multi_turn/APIModel_agent.py`,
  `multi_step/APIModel_agent.py` — the evaluated agent's clients read
  `ACEBENCH_AGENT_BASE_URL` / `ACEBENCH_AGENT_API_KEY` first (the arm proxy).
  With `ACEBENCH_ROLE_HISTORY_V1=1`, inference passes the scene's structured
  `dialogue_history` instead of its legacy flattened transcript.
* `model_inference/role_history.py` — constructs one canonical OpenAI message
  per structured entry (`user` -> user, `agent` -> assistant, `execution` ->
  tool) and keeps API definitions in the system message. It never infers roles
  by splitting message text.
* `multi_turn/APIModel_user.py` — the USER SIMULATOR reads its own
  `ACEBENCH_USER_BASE_URL` / `ACEBENCH_USER_API_KEY` (the raw upstream, full
  mode).  Same split as the ToolSandbox patch: a simulator routed through
  the arm proxy turns every number into an agent+user joint degradation.

Without the endpoint overrides and role-history flag the upstream paths run
unchanged.

Apply from the ACEBench checkout root:
`git apply --unidiff-zero benchmarks/acebench_patches/0001-endpoint-env-and-model-registry.patch`

The adapter enables `ACEBENCH_ROLE_HISTORY_V1=1` and advertises capability
`acebench_role_history_v1`; matrix preflight rejects non-full ACEBench arms
without that marker. The user simulator remains on its independent raw
endpoint, and the patch does not alter `eval_main.py` or scorer behavior.

Evaluated-agent requests carry `c2kv_eval_context` with the complete official
row ID. With structured history, `user_turn` counts prior user entries from
zero and `step` counts agent entries since the latest user entry; execution
entries do not advance that counter. Repeating the same visible prefix keeps
the same identity, with `attempt=0`. Simulator requests carry no such context.

`ACEBENCH_EVENT_NATIVE_V1=1` enables an experimental request-shape bridge for
an event-native server configured with `--benchmark acebench`. It requires
`temperature=0` and `top_p=1`, maps the same output cap from `max_tokens` to
`max_completion_tokens`, and adds `store=False` and `seed=0`. The official
handler forwards its configured generation parameters to child agents only
in this explicit mode. The generic proxy keeps its existing generation
arguments and the official textual action protocol.

The request-shape flag alone covers the initial agent request. For textual
action continuation, also set `ACEBENCH_TEXT_ACTIONS_V1=1` and run the finite
server with `--benchmark acebench --source-profile acebench-text-actions-v1`.
The textual source flag requires both preceding flags. It records a sidecar
receipt after the official decoder and executor return, binds it to the
new execution history row, and passes it as `c2kv_ace_source`. Assistant text,
observation content, roles, and tool identifiers keep their existing values.
The patch does not change execution or scoring outcomes.

The source adapter groups a submitted action batch and its aggregate official
observation. One submitted call can dispatch to multiple official instances
while exposing only the last result, so receipt slots do not claim a separate
observation for every underlying API invocation. Missing receipts are rejected;
decode errors, unsupported actions, and inconsistent return shapes/counts stay
as incomplete events that must remain raw and cannot supply recovery evidence.

The draft parser accepts a conservative common subset of the official routers:
complete `[Name(keyword=value), ...]` text beginning with `[` and containing a
closing bracket on its first line, with finite scalar or recursive list values.
It does not evaluate expressions. Unsupported grammar produces no recovery
query, while the original final text is still returned to the official client.
Internal parsed calls never appear as executable OpenAI `tool_calls`; discarded
drafts stay inside the finite runner. The adapter and upstream patch receive a
separate source identity alongside the unchanged native method contract.

This profile supports `full_original` and the existing exact-controller routes.
`static` is explicitly unavailable because its inner training controller uses
the native source protocol. Model-free protocol checks are recorded in
`outputs/a_memory_runtime_20260907/acebench_text_protocol_v1/validation.json`.
An actual-model complete task, formal corpus overlap audit, and fixed official
test split are still required before reporting formal ACEBench quality results.
