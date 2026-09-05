# Modular benchmark layer for the C2KV / experiment-D line

> Read `docs/c2kv_semantics.md` first. It records where the paper, the
> training checkpoint, this proxy and the SGLang server disagree, which
> proxy/server switch covers each difference, and which numbers in this repo
> were produced under which regime. Serve from the SGLang fork branch
> `task/bdf-pilot` (its `c2kv/c2kv_serving_semantics.md` is the
> server-side twin of that document).

Replaces the single-dataset custom evaluation with a modular design that can
run the same *arms* (full / c2kv / hybrid / repair arms) against the registered
agent benchmarks:

| Benchmark | Source on the NPU server | Env |
|---|---|---|
| τ²-bench | `~/benchmarks/tau2` (sierra-research/tau2-bench) | `~/envs/bench312` (Python 3.12, `uv`) |
| BFCL v3/v4 | `~/benchmarks/gorilla/berkeley-function-call-leaderboard` | `~/envs/bench` |
| ToolSandbox | `~/benchmarks/ToolSandbox` | `~/envs/bench` |
| AppWorld (ACON runner) | `$ACON_DIR/experiments/appworld` (microsoft/acon + StonyBrookNLP/appworld) | acon venv |
| 8-objective QA (ACON runner) | `$ACON_DIR/experiments/smolagents` (microsoft/acon, `data/nq_multi_8`) | acon venv + retriever server |
| ACEBench | `$ACEBENCH_DIR` (ACEBench/ACEBench) | acon/bench venv |

## Architecture

```
benchmarks/run.py          <- CLI, proxy lifecycle, summary envelope
        |  RunContext (adapters/base.py)
        v
adapters/<name>.py         <- NAME / add_arguments(parser) / run(ctx)
        |  builds + runs the official harness command
        v
benchmark runner (tau2 CLI / bfcl_eval / tool_sandbox CLI /
                  ACON run.py, run_all.py / ACEBench generate.py)
        |  OpenAI-compatible HTTP
        v
benchmarks/proxy.py        <- single integration point for ALL benchmarks
        |  arm-aware message assembly:
        |    full   : pass messages through unchanged
        |    c2kv   : history messages -> /v1/c2kv/extract -> c2kv_key_hash
        |    hybrid : top-k tail raw, rest gist (existing repo semantics)
        |    (repair arms plug in here later: append raw KV of block k,
        |     selective recompute, offset correction, ...)
        v
SGLang (kvoffload-sglang-c2kv, --enable-c2kv) serving the gist checkpoint
```

Key properties:

* Benchmarks run **unmodified** at the protocol level — they only point
  `base_url` at the proxy.  One vendored patch exists and is REQUIRED for
  ToolSandbox (`toolsandbox_patches/0001`: route the agent through
  `OPENAI_BASE_URL` and the user simulator through its own
  `TOOLSANDBOX_USER_BASE_URL` — upstream hard-codes api.openai.com, so a
  vanilla clone produces no TS numbers).  tau2/BFCL run unpatched.
* The arm semantics live in `proxy.py` + `arms.py`; serving requests go
  through `backends/sglang.py`. The former dataset-specific history API
  runner has been removed. The D and G entrypoints share this benchmark layer.
* Metrics are computed on **raw transcripts** in `metrics.py`, so every
  benchmark gets the same two-column evaluation:
  * **Protocol column**: every model turn is parsed for tool calls
    (`<tool_call>` text or OpenAI `tool_calls`); a call is legal iff the name
    is in the advertised tool set and arguments JSON-parse and satisfy the
    advertised JSON schema.
  * **Semantic column**: the benchmark's own official scorer (tau2 reward /
    BFCL AST+execution checker / ToolSandbox dialogue similarity).
  * **Cost columns**: per-request wall latency and gist-vs-original token
    accounting from the proxy request log.  Every run gets the run-level
    block (`request_log_summary`); the PER-TASK join is per adapter and
    every summary states which it got in `cost_join`:

    | Benchmark | `cost_join` | why |
    |---|---|---|
    | `acon_qa`, `acon_appworld` | **joined** | each task's `llm_history.json` is verbatim what the client sent, so both conversation ids are exact |
    | `tau2` | not joinable | the agent system message is not in `results.json`, and the litellm wire form of an assistant tool-call message (`content: None` vs `""`) is unpinned |
    | `bfcl` | not joinable | the steady-state id needs the first assistant message verbatim; the result file keeps only the decoded `model_responses` (the verbatim log needs `--include-input-log`, which the pinned argv does not pass) |
    | `toolsandbox` | not joinable | `result_summary.json` holds scores only, no messages |
    | `acebench` | not joinable | the whole transcript rides in ONE growing user message, so the conversation id changes every turn (and every arm is a full arm here anyway) |

    The join is self-checking: it keys on `proxy.conversation_id`, so a
    wrong key matches nothing and `cost_join` says so — it can never
    produce a wrong number.  TTFT is NOT measured — the serving path is
    non-streaming.

    A joined summary states the join's own denominator as a NUMBER, not
    only in the `cost_join` prose: `n_cost_joined` of `n` tasks.  A task
    whose requests were not matched carries no cost field at all, so
    `wall_sec_mean` / `gist_tokens_mean` / `original_tokens_mean` are means
    over those `n_cost_joined` tasks while `semantic_score` is a mean over
    all `n` — never compare the two halves without reading it.  The three
    joined fields `metrics.aggregate` does not mean are rolled up next to
    them: `n_cost_requests` (requests attributed to tasks), `n_docs_max`,
    and `dropped_docs_total` (history docs turn packing dropped inside the
    joined tasks; `None`, never `0`, when the join measured nothing).

Rendering note: historical assistant `tool_calls` are re-rendered into the
training dialect (`content + "\n\n" + "Action:\n" + <tool_call>` minified,
`train_data_multiturn._normal_agent_message`); the chat template's native
tool_calls branch is deliberately not used (kept as a future A/B).

Doc packing (2026-09-02): compressed history is packed the way the
checkpoint was trained (`--doc-packing turn`, one `Previous turn / [User
query] / [Assistant output]` doc per turn, split at `--max-doc-length`,
tail-selected to `--max-doc-num` with the doc-0 anchor). `--doc-packing
message` reproduces the pre-2026-09 per-message format. Repair arms take an
explicit placement (`append_keep_ledger` / `append_tail` / `in_place`) and
extract the raw KV in full context (`repair_extract` messages form).

## Pluggable contract

### The adapter contract (`adapters/base.py`)

One contract for every benchmark.  An adapter module exposes exactly three
names:

* `NAME` — the `--benchmark` value it answers to (`NAMES` when one module
  serves several, as `acon_adapter` does for `acon_qa`/`acon_appworld`);
* `add_arguments(parser)` — **its own** CLI flags.  A flag two adapters read
  (`--num-workers`, `--max-tasks`, `--max-iter`, `--bench-python`) stays in
  run.py's core block, because argparse refuses a duplicate option string;
* `run(ctx) -> summary dict`, given a `RunContext(base_url, user_base_url,
  out_dir, model, arm, run_name, request_log, options)`.

The adapter owns everything about reaching its harness: the `/v1` suffix
(`base.v1()` is the one implementation) and any working-directory change
(BFCL `os.chdir`s into the gorilla checkout inside its own `try/finally`).
run.py owns only the proxy lifecycle, the git-sha suffix on `--run-name` /
`--out`, and the summary envelope; `ADAPTERS` in run.py maps
`--benchmark` values to modules.  New benchmark = one adapter file + one
`ADAPTERS` entry.

**run.py is driven by CLI only** — the flag table IS the interface, and
`test_run_dispatch.py` pins every flag's name, default and type.  The argv
each adapter hands its external harness is likewise pinned, against literal
lists, in `test_adapter_contract.py`: those command lines define what every
recorded number means.

Each adapter also keeps a standalone `python benchmarks/adapters/<name>.py`
recipe (driving one harness without run.py); it goes through the same
`run_*` function as `run(ctx)`, so the two argv paths cannot drift.

> **BFCL standalone-CLI argv change (2026-09-05).** Before the registry
> refactor, `adapters/bfcl_adapter.py --run-ids a,b` re-implemented its own
> argv and appended `--run-ids` to **both** `bfcl generate` and `bfcl
> evaluate`.  It now shares `run()`'s path: the boolean `--run-ids` for
> generate, **`--partial-eval` for evaluate** (this BFCL vintage's evaluate
> has no `--run-ids`; it silently scored the WHOLE category), plus the
> terminal-state gate, which can exit non-zero.  `run.py --benchmark bfcl`
> is unaffected — it already sent `--partial-eval`.  Consequence: a subset
> number produced by the OLD standalone recipe is a full-category score and
> must not be compared with one produced now.

### Any endpoint, any model

The benchmarks are wired to **any** OpenAI-compatible serving stack and
**any** served model name — nothing about c2kv, the checkpoint, or the
port layout is baked into the adapters:

* `--upstream` (required) points at whatever serves `/v1/chat/completions`
  (+ the `/v1/c2kv/*` endpoints when an arm needs them). No default on
  purpose: a default once silently aimed sglang runs at the hf_server port.
* `--model` (default `c2kv-agent`) is the served model name handed to the
  tau2 agent/user LLMs and the BFCL handler. Rename the served model,
  re-run, nothing else changes.
* ToolSandbox selects its agent/user implementations by *role key*
  (`--ts-agent` / `--ts-user`, default `GPT_4_o_2024_05_13` →
  `openai_api_agent`/`openai_api_user`); the endpoint itself is pure
  `OPENAI_BASE_URL`.
* Benchmark checkout locations are overridable: `$TAU2_DIR`,
  `$TS_DIR` (defaults `~/benchmarks/tau2`, `~/benchmarks/ToolSandbox`);
  BFCL runs in-process from the caller's cwd (the gorilla checkout).
* Extension seams, one file each: new serving stack → `backends/<name>.py`
  (+ register in `backends/__init__.py`), never benchmark-side changes; new
  compression arm → `arms.py` only; new benchmark → `adapters/<name>.py`
  (NAME / add_arguments / run) + one `ADAPTERS` entry in `run.py`.

## Usage (on the server)

```bash
# 1. serve the checkpoint from the consolidated source (no deployment patches).
SGLANG_DIR=/path/to/sglang-c2kv DEVICE=1 PORT=35020 \
  QUERY_PROJECTION=base bash benchmarks/ops/launch_sgl1088.sh

# 2. in another terminal, run.py owns the arm proxy and the official adapter.
~/envs/bench/bin/python benchmarks/run.py --benchmark tau2 --arm c2kv \
  --backend sglang --upstream http://127.0.0.1:35020 \
  --doc-packing turn --max-doc-length 512 --max-doc-num 12 \
  --out results/bench/tau2_c2kv
```

For checkpoint-1088, use `--enable-c2kv --c2kv-query-proj base` and
`--disable-cuda-graph`. `base` is the paper/original lowercase-qkv rule;
`gist` reproduces the later local fork used by G training. Select the mode
from the checkpoint provenance (see `docs/c2kv_semantics.md`), and record
both the configured and effective modes in every run.

The in-repo Flask `hf_server` is RETIRED from the evaluation path: it
survives only as the `hfserver` contrast backend (`backends/hfserver.py`)
for A/B checks and D-side tooling. Do not baseline new numbers on it.

## ACON runners (AppWorld, 8-objective QA) and ACEBench (2026-09-05)

Three more benchmarks ride the same proxy through the same seam (one
adapter file + one dispatch branch each): `adapters/acon_adapter.py`
(`--benchmark acon_appworld | acon_qa`) and `adapters/acebench_adapter.py`
(`--benchmark acebench`).  Each external harness needs one vendored patch
that only moves its endpoint into the environment — `acon_patches/`,
`acebench_patches/` (README + `git apply` recipe in each).

```bash
# 8-objective QA (ACON's shipped data/nq_multi_8 test split, 100 tasks; the
# Search-R1 wiki-18 BM25 retriever server must be up, see acon_patches/README.md)
~/envs/bench/bin/python benchmarks/run.py --benchmark acon_qa --arm c2kv   --upstream http://127.0.0.1:35000 --acon-dir ~/baselines/acon   --bench-python ~/envs/acon/bin/python --out results/bench/qa_c2kv
#   smoke: --max-tasks 5   (= run.py --limit)   or   --task-ids nq_multi8_test_2200

# AppWorld test_normal (168 tasks; official scorer `appworld evaluate` runs after)
~/envs/bench/bin/python benchmarks/run.py --benchmark acon_appworld --arm c2kv   --upstream http://127.0.0.1:35000 --acon-dir ~/baselines/acon   --bench-python ~/envs/acon/bin/python --out results/bench/appworld_c2kv
#   smoke: --task-ids <one id> ; split: --split dev

# ACEBench agent group (user simulator = same served model at --user-upstream)
~/envs/bench/bin/python benchmarks/run.py --benchmark acebench --arm full   --upstream http://127.0.0.1:35000 --acebench-dir ~/baselines/acebench   --acebench-category agent --num-workers 4 --out results/bench/ace_full
```

Semantic columns: QA = ACON's EM (F1 alongside, `f1_mean`); AppWorld = the
official `appworld evaluate` pass/fail (the runner's own `success` flag is
carried as `agent_reported_success` only); ACEBench = the official checker
per item (agent categories: end-to-end; `per_category` carries the header
accuracies incl. process accuracy).  `protocol_legal` is `None` on all
three: the ACON agents act through Python code and ACEBench's tools are
prompt text, so the shared tool-call schema column does not apply.
Terminal-state gates: every task of the split / pin list / data file must
be scored or the run fails.

Read before quoting:

* **ACEBench cannot exercise any arm.** Its agent request is system + ONE
  user message carrying the whole transcript as text, so under the
  training rule there is no history and every arm — KV or text —
  assembles zero docs (`n_docs` = 0 on every request-log row).  An
  ACEBench column is a full-arm number for every arm; run it with
  `--arm full` and treat it as a tool-calling sanity column.
* **AppWorld is in the ckpt-1088 training pool** (31.5 % of records,
  `fork/task/d-repair-v2 inv_1088/a3_train_pool_benchmarks.json`); its
  rows on that checkpoint carry the same CONTAMINATED label as tau2.  The
  QA task (NaturalQuestions + wiki-18) is not in the pool.
* ACON observations are long: read `dropped_docs` / `n_docs` in the
  request log — turn packing keeps doc 0 + the last `--max-doc-num`-1
  docs and DROPS the rest, exactly as the checkpoint was trained.
* Protocol deviations to label: ACEBench user simulator = served model
  (upstream default `gpt-4o`), `--temperature 0` (upstream 0.7), `en` only;
  ACON decoding options stay ACON's (`presence_penalty 0.5`, seed 42).

## Arm registry

See `arms.py`. Arms are declarative: name, which history is compressed,
ratio, and for repair arms the block policy plus placement
(`c2kv_repair` = D-harness corr/keepG, `c2kv_repair_tail` = raw_erratum_tail,
`c2kv_repair_inplace` = replaceG; hybrid twins). The proxy refuses unknown
arms and unknown placements.

### Constrained-decoding arms (`cd_full`, `cd_c2kv`) — known confound

`cd_full` / `cd_c2kv` set `constrain_tools=True`, which makes
`backends/sglang.py` `prepare_chat` add
`response_format={"type": "structural_tag"}` **and** rewrite `out["tools"]`
with `_inline_refs` (inline `$ref`/`$defs`, map the loose benchmark types
`dict`/`any`/`int`/`str`/`list`/… onto JSON-schema types, strip keywords
xgrammar rejects). That rewrite is not confined to a grammar channel: SGLang
compiles the grammar **from `request.tools`**, and the chat template renders
the same `request.tools` into the prompt prologue. There is no second field
to put a repaired schema in.

So a `cd_*` row differs from its unconstrained twin (`full` / `c2kv`) in two
ways at once — the decoding constraint and the rendered tool definitions —
and any H1 delta is the sum of both. This is a property of the server's
request schema, not a proxy bug, and it is **not fixed**: changing it would
mean either shipping unrepaired schemas to xgrammar (grammar compilation
fails) or a server-side change adding a separate grammar input. Read H1 as
"structural-tag decoding *with normalised tool schemas* vs the baseline",
never as an isolated decoding effect. `c2kv_tools_dump` in the request log is
the per-request record of what the server actually rendered.

### History-KV eviction arms

**These arms have NOT been run against a live server or a model. Everything
below is the request contract, verified only by unit tests against a fake
`post_json` (`test_history_kv_arms.py`); no number here comes from a run.**

`history_kv_streamingllm_r312`, `history_kv_h2o_r312`,
`history_kv_snapkv_r312`, `history_kv_pyramidkv_r312` port the upstream
history-KV eviction baselines (kvoffload-sglang
`c2kv_eval.adapters.bfcl_history_kv_baselines` +
`scripts/run_history_kv_baselines.sh`). They compress nothing into gists: the
completed history is compressed by **token eviction inside the server**. The
`r312` suffix is the retention ratio in per-mille — 0.312, the
`HISTORY_KV_RETENTION_RATIO` default of the upstream run script; the other
knobs also take that script's defaults (`recent_window` 64, `kernel_size` 5,
`pooling` avgpool, `h2o_recent_fraction` 0.5). `snapkv` and `pyramid` are
accepted as aliases and normalized to `snapkv_persistent` / `pyramidkv`, the
same normalization the server does.

One proxy chat request reproduces the upstream client's sequence for the
`repair_extract` backend (upstream's default):

1. `POST /v1/c2kv/repair_extract` — the full-context `messages` /
   `target_index` form (system + the completed history as one span, plus the
   request's `tools`), `repair_mode="history_kv_<method>"`,
   `raw_kv_position_mode="rotated"`, `extract_source="model_prefill"`,
   `history_kv_method`, `history_kv_retention_ratio`,
   `history_kv_recent_window` / `_kernel_size` / `_pooling` /
   `_h2o_recent_fraction`. The server prefills the span, computes the
   attention scores H2O/SnapKV/PyramidKV need, selects the surviving token
   slots and stores them as one repair entry.
2. the chat request replaces the history messages with the upstream carrier
   message verbatim — `{"role": "user", "content": "[runtime <method>
   compressed history kv]", "c2kv_repair_only_key_hashes": [key_hash],
   "c2kv_use_gist_projection": false}` — and echoes the accounting in
   `c2kv_kv_memory_hint` (`full_equivalent_history_tokens`,
   `active_history_kv_tokens`, `active_raw_repair_tokens`,
   `history_kv_method`, `estimated: false`).

The server's echo comes back in `metadata.kv_memory_report` and
`normalize_response` flattens it into request-log columns:
`history_kv_method`, `history_kv_backend`, `history_kv_runtime_status`,
`history_kv_span_tokens`, `history_kv_selected_tokens`,
`history_kv_active_tokens`, `history_kv_full_equivalent_tokens`, and on the
physical path `history_kv_eviction_ok` / `_error`, `history_kv_freed_slots`,
`history_kv_kept_tokens`, `history_kv_history_tokens`,
`history_kv_freed_bytes`, `history_kv_selection_reason`. Every one of those
is the server's own number. If the server answers the extract without
echoing `history_kv_method`, the request fails hard
(`history_kv_extract_failed`) rather than silently running as a full-history
arm — upstream's `--strict-runtime-eviction`.

**Deviations from the upstream client** (all forced by the proxy having no
tokenizer, which is deliberate — cf. "Raw token counts are NOT estimated
here" in `proxy.py`):

1. *Compressed span.* Upstream tokenizes the exact server chat prompt with a
   local tokenizer and compresses the `[span_start, span_end)` slice of those
   `input_ids`. Here the span is the completed history rendered by the
   proxy's own turn packing (`_normalize_history_message` + `_turn_docs`,
   the same text the `c2kv` arm gists), sent through the `messages` /
   `target_index` form so the server still prefills it with the system block
   and tool schemas in context and measures the span itself. The KV is
   therefore the base model's KV for that text at its true absolute
   positions, but the text is the bench's history rendering, not the
   benchmark client's raw chat rendering.
2. *Budget.* Upstream computes `history_kv_target_tokens` from its own
   tokenizer's history length; these arms send `history_kv_retention_ratio`
   and let the server resolve it against the span it measured
   (`qwen3.generate_raw_repair_kv`: `ceil(requested_span_tokens * ratio)`).
   An arm may still pin an absolute `target_tokens`.
3. *History boundary.* Upstream compresses `messages[:latest_user_query]`,
   system message included. These arms split at the bench's
   `_history_cutoff` and keep the system message raw and outside the span
   (the bench never compresses system prompts), so a history-KV row is
   comparable with every other arm's row on the same request.
4. *Projection regime.* The carrier message sends
   `c2kv_use_gist_projection: false` because upstream sends it, but a
   history-KV request is a repair-only request, which is exactly the case the
   reconciled server changed (D7 / `c2kv_serving_semantics.md` #1: a
   repair-only history is now served under the `--c2kv-query-proj` default
   instead of always base). Which projection a row actually ran under is a
   server decision — read `c2kv_query_proj_effective` and
   `c2kv_query_proj_source` per row, exactly as on `c2kv_repair_inplace`.
5. *`snapkv_refresh` is not ported.* Upstream serves it only under
   `--allow-client-fallback` as a client-side textual proxy (keep the most
   recent whole history units that fit the token budget, drop the rest) and
   labels it "not SnapKV-persistent". It is a text-level policy needing a
   tokenizer, so it belongs to `textarms.py`, which this change does not
   touch.

**The `physical_eviction` backend is implemented but no arm is registered
for it.** `Arm(history_kv={"backend": "physical_eviction", "target_tokens":
N, ...})` shapes the request the upstream `--runtime-history-kv-backend
physical_eviction` path expects: the history stays raw text and
`c2kv_kv_memory_hint.history_kv_eviction` carries `method`,
`history_message_count` (the server resolves the token range itself in its
own frame — `serving_chat._resolve_history_kv_eviction_range`),
`target_tokens`, `retention_ratio` and the four selection knobs. It is not
registered because:

* the scheduler reads `config["target_tokens"]` only and never derives it
  from `retention_ratio` (`scheduler._select_history_kv_eviction_indices`,
  `mem_cache/history_kv_eviction.py`), so a retention-based physical arm
  would need a server-side change; `history_kv_spec` refuses such an arm
  instead of silently sending a budget that degenerates to "keep
  everything";
* the path needs server flags the bench launcher does not set
  (`--disable-radix-cache` — a shared prefix aborts the request with
  `PHYSICAL_HISTORY_KV_EVICTION_SHARED_PREFIX_UNSUPPORTED` — a page-aligned
  `--page-size`, and `--enable-streaming-session` for `persistent_session`).

Persistent sessions, when such an arm is declared, use the minimal
per-conversation state the proxy can hold: one `/open_session`
(`streaming: true`) per `proxy.conversation_id`, its id put on
`session_params.id` of every chat request of that conversation. Two
limitations follow and are not fixable inside a stateless HTTP proxy:
`conversation_id` shifts once after a conversation grows past its first
message (so a conversation opens two sessions, the second starting from an
empty prefix), and sessions are never closed — there is no
end-of-conversation signal — so they live until the server restarts.

### CacheBlend arms (KV reuse with selective recompute)

**These arms have NOT been run against a live server or a model. Everything
below is the request contract, verified only by unit tests against a fake
`post_json` (`test_cacheblend_arm.py`) and by the engine's own unit tests in
the SGLang fork (`test/registered/unit/test_c2kv_cacheblend.py`); no number
here comes from a run.  They need the fork branch `task/c2kv-cacheblend`
(`c2kv/c2kv_serving_semantics.md` section 10).**

`cacheblend_r16` and `cacheblend_r15_k` port CacheBlend (Yao et al., EuroSys
2025, arXiv 2405.16444) as the REAL mechanism, not the "oracle" of the
2026-09-03 design doc (`docs/cacheblend_port_design.md`, a full dense prefill
keeping 16 % of it, which must never be printed as CacheBlend): the
completed history's KV is the per-chunk STANDALONE KV (each chunk prefilled
alone, positions 0..n-1, K kept pre-RoPE and rotated to the chunk's absolute
position at blend time) with the `recomp_ratio` highest-deviation tokens
recomputed in context — layers 0..`check_layer` fully, the deviation
(`sum((fresh - old)^2)` over kv heads x head dim of V, or of K for `_k`)
measured once at `check_layer`, every later layer computed only for the
selected rows.  `r16` = the EuroSys artifact's `recomp_ratio 0.16`,
`check_layers=[1]`, V-deviation; `r15_k` = the LMCache-monorepo lineage
(K-deviation, 0.15).  Nothing is gisted and NOTHING is evicted: the served
KV is the whole span, so resident bytes are 1x raw (the Pareto anchor of the
baseline table) and the saving is compute, reported per row as
`cacheblend_recomputed_tokens` / `cacheblend_span_tokens`
(`cacheblend_effective_recomp_ratio`).

One proxy chat request:

1. `POST /v1/c2kv/repair_extract` in the multi-message form — system + tools
   + ONE MESSAGE PER HISTORY DOC (`target_index`..`target_end_index`), with
   `kv_reuse_method="cacheblend"`, `repair_mode="cacheblend"`,
   `raw_kv_position_mode="rotated"`, `extract_source="model_prefill"` and the
   `cacheblend_*` knobs.  The server renders the prologue and the docs like a
   chat prompt, takes each doc's rendered message as one chunk
   (`chunking: "doc"`; `"grid"` sends `cacheblend_chunk_tokens` and the
   server cuts a token grid across the span instead), runs the blend, and
   stores the whole span as one repair entry.  The server MUST echo
   `kv_reuse_method="cacheblend"` and a `cacheblend` accounting block, or
   the request fails hard (`kv_reuse_extract_failed`) rather than silently
   running as a full-history arm — the same rule as the history-KV arms.
2. the chat request replaces the history messages with a repair-only carrier
   `{"role": "user", "content": "[cacheblend reused history kv]",
   "c2kv_repair_only_key_hashes": [key_hash], "c2kv_repair_placement":
   "in_place"}` and sends the accounting in `c2kv_kv_memory_hint`
   (`active_raw_repair_tokens` = span, `active_recomputed_raw_tokens` =
   recomputed, plus the `cacheblend_*` provenance).  No
   `c2kv_use_gist_projection` is sent: the projection regime is the server's
   `--c2kv-query-proj` decision, read per row from
   `c2kv_query_proj_effective` / `c2kv_query_proj_source`.

The echo comes back in `metadata.kv_memory_report` and `normalize_response`
flattens it into request-log columns: `kv_reuse_method`, `kv_reuse_backend`,
`cacheblend_span_tokens`, `cacheblend_recomputed_tokens`,
`cacheblend_effective_recomp_ratio`, `cacheblend_recomp_ratio`,
`cacheblend_check_layer`, `cacheblend_metric`, `cacheblend_mask`,
`cacheblend_chunking`, `cacheblend_chunk_count`, `cacheblend_deviation_max`,
`cacheblend_deviation_selected_min`, `cacheblend_cache_hit`,
`kv_reuse_active_tokens`, `kv_reuse_recomputed_tokens`.  The proxy request
log row also carries `kv_reuse` (method, chunking, n_history_messages,
n_history_docs).

**Deviations from the official artifact** (all recorded in the fork's
semantics doc section 10; label them on any row):

1. *Chunks are the bench's turn docs* (`_normalize_history_message` +
   `_turn_docs`, the same units the `c2kv` arm gists), each rendered as its
   own chat message — not the artifact's dataset passages and not a
   256-token grid (`chunking: "grid"` exists for that, unregistered).
2. *The system/tool prologue is prefilled fresh* in the same forward (the
   bench never compresses system prompts; the artifact caches the system
   prompt as chunk 0), and the current turn is served by the normal extend
   path over the entry rather than as a `last_len` suffix inside one forward.
3. *The chunk cache is materialised per request* from the chunk tokens
   (identical values, no storage) — the entry is the same, but no
   wall-clock / TTFT number from this path is CacheBlend's.  TTFT is not
   measured here anyway.
4. *The default attention mask is exactly causal* for the scattered
   recompute queries; the artifact's `LowerTriangularFromBottomRightMask`
   (which lets a selected token attend to later keys) is opt-in
   (`mask: "bottom_right"`), not registered as an arm.
5. *First turn:* nothing completed → no extract, no hint, the row carries no
   `cacheblend_*` columns (same as the history-KV arms).

## Relation to experiment D

The teacher-forced D harness (`agent/d_kv_intervene.py`,
`agent/d_paired_analysis.py`) stays untouched: it measures rescue on the
frozen C→W trigger set with exact statistics. This layer answers the
complementary question — *end-to-end benchmark performance under each arm* —
which the deep-research transfer manual asks every method to report
(`docs/bdf_pilot_runbook.md`, `expD_repair_methods_transfer_manual.md`).
