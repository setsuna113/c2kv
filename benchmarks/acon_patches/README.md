# ACON deployment patch

Upstream pin: **microsoft/acon** (clone used to cut the patch:
`tmp/baselines/acon`, cloned 2026-09-03; the patched file
`src/productive_agents/llm.py` has not changed upstream since the last
release the bench audited).

`0001-openai-base-url-env.patch` makes the ACON runners reproducible for the
bench: `productive_agents.llm.vLLM` (the client every non-OpenAI model name
falls to in `agents/utils.py:LLMManager.create_llm`) hard-codes
`http://localhost:8000/v1` / `token-abc`; the patch makes it read
`ACON_OPENAI_BASE_URL` / `ACON_OPENAI_API_KEY` (same defaults when unset).
`benchmarks/adapters/acon_adapter.py` exports the arm proxy there.  Nothing
else in the runners changes: prompts, memory, decoding options
(`presence_penalty 0.5`, `enable_thinking false`, seed 42) stay ACON's.

Apply from the acon checkout root:
`git apply benchmarks/acon_patches/0001-openai-base-url-env.patch`

Both runners then need only the served model name to NOT contain `gpt`,
`o1`, `o3`, `o4` or `gemini` (those names are routed to the OpenAI / Gemini
clients instead).  `c2kv-agent` is fine.

Runner prerequisites (see the upstream READMEs under `experiments/`):

* 8-objective QA — `pip install smolagents`; the Search-R1 wiki-18 BM25 index
  (`PeterJinGo/wiki-18-bm25-index`, `PeterJinGo/wiki-18-corpus`) under
  `experiments/smolagents/search/database/wikipedia`; the retriever server
  `python search/retriever_server.py --index_path search/database/wikipedia/bm25`
  running before the run.  Data = the shipped `data/nq_multi_8` (100/100).
* AppWorld — `pip install -e` the StonyBrookNLP/appworld checkout,
  `appworld install --repo`, `appworld download data`, and the produced
  `data/` moved into `experiments/appworld/`.  The official scorer is the
  `appworld` CLI on the same venv PATH.

ACON's own compression arm (`--co_config_path`) is NOT wired: with
`model_type: local` its compressor is an in-process vLLM, not the served
endpoint, and the bench's ruling is one served 4B for policy and compressor.
The bench text arm for ACON stays `--arm acon_hist` / `acon_obs` in
`benchmarks/textarms.py`.
