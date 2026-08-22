# B/D/F pilot runbook

Operational guide for the three first-round pilots that ship on this branch.
Everything here was developed and tested off-server; the numbers these
harnesses produce do not exist yet.

- **B** — chunking policy: does the way history is cut into chunks change how
  much next-action behaviour survives compression, at equal gist budget?
- **D** — in-place KV repair vs rollback: on a failure the full cache gets
  right and the compressed cache gets wrong, how much is repairable without
  replaying the trajectory, and what does each repair cost?
- **F** — speculative compaction: keep the current segment compressed *and*
  deferred as two branches, run both one step, and ask whether picking between
  them beats simply retrying the compressed one.

All three answer the same four questions and nothing more (§6).

---

## 1. Deployment gate — run this first, every time

Nothing below is meaningful until the test suite passes on the box that will
run it. Two of the gates are load-bearing: the RoPE reposition properties
(every arm assembles caches by rotating KV into place) and the bit-equality
assertions (a sham arm that is not bit-identical to no-op silently invents
effects).

```bash
cd <repo>
export PYTHONPATH="$(pwd)/python:$(pwd)/python/inference:$(pwd)/agent:${PYTHONPATH:-}"
python -m pytest python/train python/models agent metrology -q
```

Expected: everything green except environment-dependent failures for packages
this box lacks (`bfcl_eval` is the known one). Reference run off-server:
**468 passed, 2 failed (both `bfcl_eval`-missing), 9 skipped**.

The four suites that must show as *executed*, not skipped:

| File | Guards |
|---|---|
| `python/models/test_rope_reposition.py` | store-then-rotate equivalence, position accounting on both `process_context_input_ids` paths, blend/concat consistency |
| `agent/test_generate_sampling_kwargs.py` | the greedy default kwargs are byte-identical to before the sampling switch existed |
| `agent/test_d_kv_intervene_torch.py` | corr slice == sequential-full slots; no-gist recompute rebuilds exactly; mechanical sham is an exact identity |
| `agent/test_f_timing_fork_gpu.py` | both branches place the current turn at the same position; shared-prefix gist KV agrees across branches |

If they report SKIPPED, torch is not visible to pytest — fix that before
running anything.

---

## 2. Frozen state

Every run binds its inputs by sha256 and stamps them into each output row, so
a row can always be traced back to the exact frozen inputs. Freeze before you
run; do not edit a frozen file and re-run on top of it.

| Artifact | Produced by | Consumed by |
|---|---|---|
| `configs/bdf_pilot/b_prereg.md` | hand | B (record the eval manifest sha in it) |
| `configs/bdf_pilot/d_prereg.md` | hand | D |
| `configs/bdf_pilot/f_prereg.md` | hand | F (`--prereg_file`, sha stamped per row) |
| `configs/bdf_pilot/d_cw_manifest.json` + `results/d/bundles_batch_tf.jsonl` | `agent/extract_cw_triggers.py` | D |
| `configs/bdf_pilot/d_sham_plan.json` | `agent/d_sham_plan.py` | D (sham arm) |
| `configs/bdf_pilot/d_neutral_corpus.txt` | in repo | D sham plan (sha-bound) |
| eval qid manifest (n=200 layer) | the nested eval manifest | B `QID_MANIFEST`, F `QID_MANIFEST` |

**Checkpoint pinning.** All three lines must run against the *same* checkpoint
path, and that path plus its sha goes in the interpretation record. Mixing
checkpoints across lines breaks the shared-benchmark discipline the whole
design rests on.

---

## 3. B — chunking policy

Four arms at equal gist budget, plus one shared full/truncate reference:

| Arm | Policy |
|---|---|
| `P-fixed` | fixed 1024-token chunks (the budget reference arm) |
| `P-turn` | one chunk per agent turn (current default; in-distribution reference) |
| `P-struct` | structural — a tool call and its observation are never split |
| `P-delay` | agent-turn, but the most recent turn stays raw instead of compressed |

```bash
QID_MANIFEST=configs/bdf_pilot/<eval200>.json \
SPLIT_MANIFEST_FILE=<frozen split manifest> \
MODEL_PATH=<pinned checkpoint> \
BASE_MODEL=./models/Qwen3-4B-Instruct-2507 \
DATASET_PATH=./datasets/agent-llm-traces \
ARMS=P-fixed,P-turn,P-struct,P-delay RATIOS=8 \
OUTPUT_DIR=./outputs/b_pilot \
bash agent/run_b_pilot_npu.sh
```

Outputs `<arm>.jsonl` + `<arm>.summary.json` per arm, a shared `ref.jsonl`, and
`analysis.{json,md}`.

**Budget declaration.** Arms are only comparable at equal gist budget. The
analyzer reports each arm's mean `gist_tokens` against `P-fixed` and marks any
arm deviating by more than 5% as VOID. `P-delay` is exempt by construction —
it holds a turn back as raw tokens, accounted separately in `raw_recent_tokens`
and reported as a cost, never folded into the gist count.

**Content identity.** All arms re-chunk one frozen text stream, so content is
identical by construction; only chat-template wrapper overhead differs. The
analyzer checks presented tokens across arms and, above 2% spread, reweights by
presented-token decile before comparing.

**Reading the result.** The pre-registered primary contrasts are P-struct vs
P-fixed, P-struct vs P-turn, and P-turn vs P-fixed. Everything else is
exploratory and Holm-corrected. Because the checkpoint was trained under the
agent-turn policy, the other arms are out-of-distribution for it — so a
non-default arm *winning* is conservative evidence, while a non-default arm
*losing* is not attributable and must not be reported as "that policy is
worse".

---

## 4. D — repair vs rollback

Five arms on the same set of failure points. Triggers come from paired rows
where the full cache is right and the compressed cache is wrong.

| Arm | Intervention |
|---|---|
| `none` | none (defines the failure set) |
| `sham` | equal-byte neutral text injected through the identical path |
| `corr` | append-only correction of one mid-position document, downstream untouched |
| `corr_re` | same correction, then recompute everything downstream |
| `full` | full recompute of the same transcript (ceiling) |

```bash
# 1. extract and freeze the trigger set (torch-free; --bind_docs needs torch)
python agent/extract_cw_triggers.py \
  --full_rows <full-arm rows> --compressed_rows <c2kv-arm rows> \
  --batch batch-TF --ckpt_path <pinned checkpoint> --ratio 8 \
  --max_doc_length 768 \
  --out_bundles results/d/bundles_batch_tf.jsonl \
  --out_manifest configs/bdf_pilot/d_cw_manifest.json --bind_docs

# 2. freeze the sham plan
python agent/d_sham_plan.py \
  --manifest configs/bdf_pilot/d_cw_manifest.json \
  --out configs/bdf_pilot/d_sham_plan.json

# 3. smoke (identity sentinels), then the arms
PHASE=smoke MODEL_PATH=<pinned> bash agent/run_d_pilot_npu.sh
PHASE=arms  MODEL_PATH=<pinned> bash agent/run_d_pilot_npu.sh
```

**Grid consistency is enforced, not assumed.** The manifest records the doc
grid it was frozen under (`max_doc_length`, `ratio`) and which harness dialect
the rows came from. `d_kv_intervene.py` refuses to start if the run's geometry
differs, because a trigger is a statement about one specific context: rebuild
it differently and the intervention lands somewhere else, with nothing
downstream to reveal the swap. Note that D runs the history harness, whose doc
budget convention is 768/16 — deliberately *not* the joint harness's 1024/24.
Do not "align" them.

**Reading the result.** The comparison that matters is `corr_re` vs `sham`, not
vs `none` — sham absorbs the effect of the intervention machinery itself and of
the extra bytes. Report both levels of the denominator separately (what
fraction of failures are in the repairable subset × what fraction of that
subset gets repaired); never just the product. A "repair" that produces
incoherent output is not a repair: the coherence triple (protocol-legal parse
rate, repetition degeneration, output-length drift) gates every rescue count.

---

## 5. F — speculative compaction

Two branches per example, generated once each; the remaining arms are derived
at analysis time from those two outputs at zero extra compute.

| Arm | Meaning |
|---|---|
| `F0` | compress the current segment now |
| `F2` | defer — keep it raw for this step |
| `F1` | compress now, two sampled rollouts, pick by deterministic checks |
| `F3` | pick between the compressed and deferred branch by deterministic checks |
| `F4` | pick at random (the floor F3 has to beat) |
| `F5` | either branch correct (ceiling) |

```bash
# pass 1: greedy core -- no dependency on the sampling switch
MODEL_PATH=<pinned> QID_MANIFEST=<eval manifest> MAX_EXAMPLES=200 \
ARM_SET=greedy_core OUTPUT_FILE=./outputs/f_pilot/greedy_core.jsonl \
bash agent/run_f_pilot_npu.sh

# pass 2: sampled arms (adds F1 and the sampled F3)
ARM_SET=sampled TEMPERATURE=0.7 TOP_P=0.95 GEN_SEED=20260822 \
OUTPUT_FILE=./outputs/f_pilot/sampled.jsonl bash agent/run_f_pilot_npu.sh
```

**Memory honesty.** Inside the speculation window both branches are alive, so
the step costs *more* memory than not compressing at all (roughly 1.125x the
segment). The saving only materialises after one branch is committed. Report it
that way; never present compression savings as buying extra branches.

**Reading the result.** The comparison that matters is F3 vs F1: same number of
rollouts, same selection rule, the only difference being whether the diversity
comes from sampling or from memory state. F3 beating F0 alone would only show
that two tries beat one. Because the deferred branch decodes over a longer
cache, equal-rollout and equal-GPU-time ledgers disagree — both are reported,
neither is "the" answer.

---

## 6. What a first round can and cannot conclude

Each line answers exactly four questions:

1. Is there headroom at all, measured against its own sham/random floor?
2. Does it beat the simple baseline (retry, truncation, full rebuild)?
3. What does it cost, in bytes and GPU-seconds?
4. Which failures benefit — is the effect concentrated somewhere legible?

Only then does it become sensible to add learned triggers, learned repair,
wider beams, or richer checkpoints.

A direction stops for exactly five reasons: the implementation is invalid, no
headroom exists, a simple baseline dominates it, the cost is unacceptable, or
something else deserves the compute more. Resemblance to published work is not
one of them; that is an argument about how to describe a result, not about
whether to measure it.

Every result table carries its n, its floor, and the note that no difference
below the detectable threshold is a ranking. Pilot runs are mechanism probes:
they size effects and validate machinery, they do not settle directions.

---

## 7. Scheduling

Ordering, cheapest and most independent first:

1. **Off-accelerator, no queue**: extract and freeze D's trigger set, freeze the
   sham plan, freeze the eval manifest, run the deployment gate.
2. **D** — reuses existing rows for two of five arms, so only three new arms run.
3. **B** — four arms plus one shared reference; the result feeds the data-pipeline
   decision for the next training run, so it has the earliest downstream deadline.
4. **F** — greedy core first (two generations per example); the sampled pass is a
   separate invocation and can slip without touching code.

Time estimates from a battery of this size have been off by a factor of two
before. Measure one route before committing to a schedule, and treat any
published estimate here as a starting guess.
