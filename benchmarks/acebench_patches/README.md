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
  registered as `APIModelInference`.
* `apimodel_inference.py`, `multi_turn/APIModel_agent.py`,
  `multi_step/APIModel_agent.py` — the evaluated agent's clients read
  `ACEBENCH_AGENT_BASE_URL` / `ACEBENCH_AGENT_API_KEY` first (the arm proxy).
* `multi_turn/APIModel_user.py` — the USER SIMULATOR reads its own
  `ACEBENCH_USER_BASE_URL` / `ACEBENCH_USER_API_KEY` (the raw upstream, full
  mode).  Same split as the ToolSandbox patch: a simulator routed through
  the arm proxy turns every number into an agent+user joint degradation.

Without the overrides the upstream name-keyed chains run unchanged.

Apply from the ACEBench checkout root:
`git apply benchmarks/acebench_patches/0001-endpoint-env-and-model-registry.patch`

Read before quoting a number: the agent receives the whole dialogue as ONE
user message (`MULTI_TURN_AGENT_PROMPT_USER_*` embeds the
`user:/agent:/execution:` transcript as text), so the arm proxy never sees
compressible history — every arm assembles zero docs and an ACEBench column
is a full-arm number for every arm.  See `adapters/acebench_adapter.py`.
