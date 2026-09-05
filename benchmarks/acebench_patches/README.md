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
