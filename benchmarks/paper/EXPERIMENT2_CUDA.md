# Experiment 2 on CUDA

This is the existing paper runner with a fixed, 40-cell closed-loop matrix: four
benchmarks (BFCL Base, BFCL Long Context, ACEBench Agent, AppWorld test-normal)
times Full, budgeted HiAgent/ACON, five history-KV methods, bare C2KV, and
`c2kv_goal_rescue_r8`. It excludes C1, newer Goal-composition candidates, tool
compression, budget sweeps, and expanded recovery capacity. Copy the entire
`c2kv-paper` repository and its matching `sglang-paper` fork; no separate
benchmark package is needed.

```bash
git clone --branch share/tracy-experiment2-cuda-20260920 https://github.com/setsuna113/c2kv.git c2kv-paper
git clone --branch paper/benchmarks-cuda-20260917 https://github.com/setsuna113/kvoffload-sglang-c2kv.git sglang-paper
cd c2kv-paper
cp benchmarks/paper/experiment2_cuda.json /absolute/path/to/experiment2.local.json
```

Edit **every** `/ABSOLUTE/PATH/TO/...` value in the local JSON to match the
machine. `checkpoint` is **CHECKPOINT: the selected C1000/checkpoint-1000**
identified by `experiments/history_system/configs/current_algorithm.json`;
verify the checkpoint's `config.json` against that selection before running.
The three Python paths are the SGLang server, benchmark, and AppWorld
environments; the benchmark and official harness directories must also exist.
Use an empty output directory on local Linux storage. Install the dependencies
for both repositories and the official harnesses as described in
[`benchmarks/paper/README.md`](README.md).

The supplied CUDA settings are reference settings for an RTX PRO 6000 run,
**not measured RTX PRO 6000 results**: `flashinfer`, 131072 context/total
tokens, 512-token prefill chunks, `mem_fraction_static=0.8`,
`c2kv_pool_fraction=0.1`, and one ordinary active request. Check capacity and
engine readiness on the target GPU before full execution. The runner starts
and supervises its own SGLang server for each cell.

```bash
CFG=/absolute/path/to/experiment2.local.json
ENGINE=/absolute/path/to/sglang-paper
RESULTS=/absolute/path/to/experiment2-output
PAPER_PY=/absolute/path/to/venv-bench/bin/python

# One official BFCL task per selected arm; functional smoke, not a quality score.
"$PAPER_PY" -m benchmarks.paper.experiment2_smoke \
  --config "$CFG" --sglang-source "$ENGINE" --output "$RESULTS/smoke" \
  --arms full,c2kv_native_r4,c2kv_goal_rescue_r8 --task-id multi_turn_base_26
# Use --arms all to exercise every configured BFCL Base method.

"$PAPER_PY" -m benchmarks.paper prepare \
  --config "$CFG" --sglang-source "$ENGINE" --output "$RESULTS/formal"

# Optional selected cells, followed by the rest of the same prepared matrix.
"$PAPER_PY" -m benchmarks.paper run \
  --config "$CFG" --sglang-source "$ENGINE" --output "$RESULTS/formal" \
  --stage closed_loop \
  --cells bfcl_base__full,bfcl_base__history_kv_h2o_r25_persistent_b768,bfcl_base__c2kv_native_r4,bfcl_base__c2kv_goal_rescue_r8
"$PAPER_PY" -m benchmarks.paper run \
  --config "$CFG" --sglang-source "$ENGINE" --output "$RESULTS/formal" \
  --stage closed_loop
"$PAPER_PY" -m benchmarks.paper aggregate \
  --output "$RESULTS/formal" --stage closed_loop
```

`matrix.csv`, `commands.json`, and `config.resolved.json` record the exact
planned cells and deployment. Each completed cell has
`closed_loop/<cell_id>/complete.json`, an official `summary_<arm>.json`, and
`measurement_summary.json`; `comparison.csv` and `comparison.json` combine
those completed cells. Read `semantic_score` together with `scored_tasks`,
`stage`, `comparison_basis`, and the source files. Results from one run are
`preliminary, n=1`. The smoke output is separate from the formal output.

The B0 history capacity is 768 tokens, including recovered evidence; text
methods cap actor-visible history at 768 and the five history-KV methods use
the explicit 768-token capacity interface. Bare C2KV uses ratio 4 with no
recovery; Goal uses ratio 8, S0 allocation, the frozen T02 detector, and one
bounded recovery. Their difference is **not** a detector-only ablation. The
native dispatcher is named `benchmarks.paper.c1` for historical reasons;
`--arm c2kv_goal_rescue_r8` selects the Goal candidate, not a C1 arm.

BFCL task selection here defaults to the complete official 200-task split per
variant. The paper's 174-task held-out comparison excludes 26 task families
shared with detector training/calibration. This example does not define or
silently apply that exclusion; report the full-split result under its own
denominator. Any held-out comparison needs the same frozen task-family list
applied to every method.
