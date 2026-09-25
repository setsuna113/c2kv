# Experiment 2 on CUDA

This is the existing paper runner with a fixed closed-loop matrix across
BFCL Base, BFCL Long Context, ACEBench Agent, AppWorld test-normal, tau2, and
ToolSandbox. The selected recovery method is now **PendingVerified**,
`c2kv_pending_verified_r8` (`pending_verified`, not `pending_verified_static`).
It replaces legacy Goal in this example. Full, budgeted HiAgent/ACON, five
history-KV methods, and bare C2KV remain available. The matrix excludes C1,
other candidate variants, tool compression, budget sweeps, and expanded
recovery capacity. Copy the entire
`c2kv-paper` repository and its matching `sglang-paper` fork; no separate
benchmark package is needed.

PendingVerified is enabled on the first five benchmark axes, including tau2.
The current candidate registry and delivery-profile guard exclude ToolSandbox;
its PendingVerified cell is explicitly listed in `unsupported_cells.json`.
ToolSandbox runs the other nine methods. The example therefore resolves to
59 runnable cells, not a claimed complete six-by-ten matrix.

```bash
git clone --branch share/tracy-experiment2-cuda-20260920 https://github.com/setsuna113/c2kv.git c2kv-paper
git clone --branch share/tracy-experiment2-cuda-20260920 https://github.com/setsuna113/kvoffload-sglang-c2kv.git sglang-paper
cd c2kv-paper
cp benchmarks/paper/experiment2_cuda.json /absolute/path/to/experiment2.local.json
```

Edit **every** `/ABSOLUTE/PATH/TO/...` value in the local JSON to match the
machine. `checkpoint` is **CHECKPOINT: the selected C1000/checkpoint-1000**
identified by `experiments/history_system/configs/current_algorithm.json`.
Reuse that checkpoint, or download the pinned copy with your existing access
to the private `Jasonning/c2kv` repository:

```bash
hf download Jasonning/c2kv \
  --revision ee456cfd36d88c9844d26895c3d7b3d218cdf36b \
  --include 'b_history/arm-C/seed-42/checkpoint-1000/*' \
  --local-dir /absolute/path/to/weights
```

Set `checkpoint` to the resulting `weights/b_history/arm-C/seed-42/checkpoint-1000`
directory. Its `config.json` SHA256 is
`15e14bfa5853ce7e74ef3cd6d65fc4b4411378c3111aebcfdc8ed272f5424655`;
`model.safetensors` SHA256 is
`b3ea50a5d7e4fa5225d8c5317686062fc4d09c6e4ecff984576fe75f411e846f`.
The frozen risk-detector JSON is included in the code. PendingVerified uses lexical
retrieval and does not require a separate embedding-model download.

The Python paths select the SGLang server, paper/BFCL runner, AppWorld, tau2,
and ToolSandbox environments; the official harness directories must also exist.
Use an empty output directory on local Linux storage. The validated CUDA stack
uses Python 3.11, PyTorch `2.9.1+cu129`, Transformers `5.3.0`,
`flashinfer-python==0.6.7.post2`, and `sglang-kernel==0.4.1`. Install this engine
checkout's `python/` package into that server environment (not stock SGLang).
The benchmark environment additionally needs the BFCL checkout installed
editable, `numpy==1.26.4`, `openpyxl==3.1.5` for the ACEBench scorer,
and the native runtime's `torch`, `transformers`,
`safetensors`, and `requests`. Reuse the existing environments if they match.
The separate AppWorld environment used Python 3.12 and `appworld==0.1.3.post1`.

For the other harnesses, apply the shipped
[ACEBench endpoint/role-history patches](../acebench_patches/README.md) and
[ACON AppWorld patches](../acon_patches/README.md). For this matrix, ACEBench
needs patches `0001` and `0002`; ACON needs `0001`, `0002`, `0006`, and `0007`
(the retrieval/Smolagents patches are for other benchmarks). Pass each patch's
absolute path when running `git apply` inside the external checkout.
AppWorld also requires
`appworld install` and `appworld download data`; `appworld_root` is the directory
containing `data/`. These are harness dependencies, not extra memory algorithms.
The current adapter applies its native-runtime hooks to private run copies.
Do not install the root training `requirements.txt` into the serving environment.

For tau2, use the official checkout and a separate Python 3.12 or 3.13
environment as described in [the tau2 integration guide](README.md#tau2-matrix-cells).
The example selects `airline`, `base`, one trial per task. For ToolSandbox,
apply both [deployment patches](../toolsandbox_patches/README.md) and use its
separate environment (`openai==1.17.0`, `httpx==0.27.2` in the existing
integration). The example selects the frozen `three_distraction_tools_129`
suite. Scenarios that need RapidAPI use the existing `RAPID_API_KEY` environment
variable, or `TOOLSANDBOX_ENV_FILE` pointing to a private file containing that
assignment; keep credentials outside the repository.

The supplied CUDA settings are reference settings for an RTX PRO 6000 run,
**not measured RTX PRO 6000 results**: `flashinfer`, 131072 context/total
tokens, 512-token prefill chunks, `mem_fraction_static=0.8`,
`c2kv_pool_fraction=0.1`, and one ordinary active request. Check capacity and
engine readiness on the target GPU before full execution. The runner starts
and supervises its own SGLang server for each cell.
Decode CUDA graphs are disabled in this delivery config: the local ACEBench
Full smoke stalled in graph decode, while eager CUDA proceeded past the same
point. FlashInfer CUDA attention remains enabled. Do not enable graphs for a
formal run without validating the complete target workload on that GPU.
RTX PRO 6000 Blackwell is [SM120](https://developer.nvidia.com/cuda/gpus);
use kernels built for that device, as supported by
[FlashInfer 0.6.7](https://github.com/flashinfer-ai/flashinfer/blob/v0.6.7/README.md).

```bash
CFG=/absolute/path/to/experiment2.local.json
ENGINE=/absolute/path/to/sglang-paper
RESULTS=/absolute/path/to/experiment2-output
PAPER_PY=/absolute/path/to/venv-bench/bin/python

# One official BFCL task per selected arm; functional smoke, not a quality score.
"$PAPER_PY" -m benchmarks.paper.experiment2_smoke \
  --config "$CFG" --sglang-source "$ENGINE" --output "$RESULTS/smoke" \
  --arms full,c2kv_native_r4,c2kv_pending_verified_r8 --task-id multi_turn_base_26
# Use --arms all to exercise every configured BFCL Base method.

"$PAPER_PY" -m benchmarks.paper prepare \
  --config "$CFG" --sglang-source "$ENGINE" --output "$RESULTS/formal"

# Optional selected cells, followed by the rest of the same prepared matrix.
"$PAPER_PY" -m benchmarks.paper run \
  --config "$CFG" --sglang-source "$ENGINE" --output "$RESULTS/formal" \
  --stage closed_loop \
  --cells bfcl_base__full,bfcl_base__c2kv_native_r4,bfcl_base__c2kv_pending_verified_r8,tau2__c2kv_pending_verified_r8,toolsandbox__full,toolsandbox__c2kv_native_r4
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
the explicit 768-token capacity interface. The historical `r25` registry names
remain in some cell IDs, but `history_budget_tokens=768` replaces their
fractional allowance. Bare C2KV uses ratio 4 with no recovery. PendingVerified
uses ratio 8, the original Goal allocation/recovery and Pending STOP review,
the frozen T02 detector, and verified source-backed field corrections that
do not add a regeneration. Its protocol is `c2kv-verified-binding-v1`, with
`verified-binding-rules-v1` recorded in the profile and ready manifest.
It retains the original B0 budget and bounded recovery; no budget overlay is
requested for native arms. This is **not** a detector-only ablation. The
native dispatcher is named `benchmarks.paper.c1` for historical reasons;
`--arm c2kv_pending_verified_r8` selects PendingVerified, not a C1 arm.
Use a fresh output directory for this revision; do not resume legacy Goal results
as PendingVerified.

BFCL task selection here defaults to the complete official 200-task split per
variant. The paper's 174-task held-out comparison excludes 26 task families
shared with detector training/calibration. This example does not define or
silently apply that exclusion; report the full-split result under its own
denominator. Any held-out comparison needs the same frozen task-family list
applied to every method.

The [previous CUDA validation receipt](validation_receipts/tracy_cuda_20260920.json)
records 17 single-task cases from the September 20 code and **legacy Goal**,
including source revisions, server arguments, artifact hashes, and budget
checks. It is historical evidence, not validation of PendingVerified or this
updated matrix. That validation used an RTX 4090
Laptop GPU with 16 GB, not an RTX PRO 6000. Local smoke capacity was reduced;
PyramidKV, CommitKV, and AgentKV also used `--cpu-offload-gb 2` in the smoke
helper to fit model weights. This leaves the 768-token method budget unchanged
but makes those timings unsuitable for GPU-only performance comparison.
The receipt is functional validation, not a full-benchmark quality table.
The September 21 PR refresh does not start WSL, CUDA tests, or benchmark runs.
Only lightweight configuration checks are performed. No new hardware results
or PendingVerified quality claim is made by this delivery.
