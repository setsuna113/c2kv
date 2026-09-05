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

Apply `0002-unknown-api-cost.patch` with `git apply --ignore-whitespace`
(the pinned source uses CRLF). Unknown model tariffs return
JSON `null`; AppWorld prints that API cost is unavailable and still records
token counts. This fixes the upstream missing `gpt-4o` fallback without
assigning a hosted-model price to local inference.

Apply `0003-bm25-lazy-imports.patch` with `git apply --ignore-space-change`
before launching the 8-objective QA
retriever. The shipped server eagerly imports its optional dense-retrieval
stack even when `retrieval_method="bm25"`. The patch defers those imports to
the dense and external-corpus paths, so a stored-document Lucene index needs
only Pyserini and the FastAPI server dependencies. Dense retrieval behavior is
unchanged when its optional dependencies are installed.

Pyserini `0.44.0` also exports its impact and HNSW searchers eagerly from
`pyserini.search.lucene`, which imports the neural encoder stack before the
BM25 `LuceneSearcher` can be used. Apply `0004-pyserini-sparse-imports.patch`
from the virtual environment's `site-packages` directory. This task-local
sparse installation deliberately stops exporting those two optional searcher
families; its BM25 API and Lucene implementation are unchanged.

Both runners then need only the served model name to NOT contain `gpt`,
`o1`, `o3`, `o4` or `gemini` (those names are routed to the OpenAI / Gemini
clients instead).  `c2kv-agent` is fine.

Runner prerequisites (see the upstream READMEs under `experiments/`):

* 8-objective QA — `pip install smolagents`; the Search-R1 wiki-18 BM25 index
  (`PeterJinGo/wiki-18-bm25-index`, `PeterJinGo/wiki-18-corpus`) under
  `experiments/smolagents/search/database/wikipedia`; the retriever server
  `python search/retriever_server.py --index_path search/database/wikipedia/bm25`
  running before the run.  Data = the shipped `data/nq_multi_8` (100/100).
* AppWorld — install `appworld` (validated with `0.1.3.post1`), run
  `appworld install` and `appworld download data`. Set `APPWORLD_ROOT` to
  the directory containing `data/`, or place it in `experiments/appworld/`.
  The adapter creates a private harness per run, links immutable data, and
  gives generation and official scoring the same selected task list.
  The official scorer is the `appworld` CLI beside the runner Python.

ACON's own compression arm (`--co_config_path`) is NOT wired: with
`model_type: local` its compressor is an in-process vLLM, not the served
endpoint, and the bench's ruling is one served 4B for policy and compressor.
The bench text arm for ACON stays `--arm acon_hist` / `acon_obs` in
`benchmarks/textarms.py`.
