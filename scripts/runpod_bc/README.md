# RunPod B/C paired run, 2026-09-11 (seed 42)

Operational scripts used for the first real B/C history-memory training on two
RunPod community pods (1x NVIDIA H100 NVL 94 GB each, one arm per pod, host
CUDA 12.8). They are recorded here for provenance; paths are the pod layout.

| File | Role |
|---|---|
| `pod_setup.sh` | training env: `.venv-b-history` with torch 2.9.0 **cu128** (host driver 570 cannot load the cu130 wheel named in `docs/b_memory_training/h200.md`), pinned deps, CPU smoke, base model download |
| `b_history_runpod.env` | launcher env actually used: `NPROC_PER_NODE=1`, `GRADIENT_ACCUMULATION_STEPS=16` (effective batch 32), `REQUIRE_H200=0`, `SAVE_STEPS=250`, W&B online (`c2kv-history-b`) |
| `pod_eval_setup.sh` | eval env: A snapshot `snapshot/a-memory-runtime-20260911` @1e5f185 (clean worktree), gorilla/BFCL @6ea5797 (same pin as the NPU `~/benchmarks/gorilla`), separate `.venv-bc-eval` because `bfcl_eval` pins `numpy==1.26.4`; `soundfile` had to be added by hand (undeclared `qwen_agent` dependency) |
| `dev-tasks.json` | frozen dev set = `configs/bfcl_dev_v3_mt.json` (128 `multi_turn_base` ids, seed 42, 2026-08-25) plus the `split: dev` field required by `history_memory.checkpoint_eval` |
| `smoke-tasks.json` | first 8 dev ids; pipeline smoke and timing only, never a selection input |
| `eval_queue.sh` | per-arm (step, ratio) priority queue run alongside training on the same GPU: ratio 8 first, ratio 4 in the gaps; one manifest per checkpoint, `agent/eval_history_memory.py --dry-run` then run |
| `select_bc.py` | pre-registered selection over the six official summaries: argmax `(correct_r8 + correct_r4) / 256` among {500, 750, 1000}, ties to the earlier step, a step lacking either ratio is disqualified; per-ratio winners reported as secondary |
| `finalize.sh` | after training `completed` and the six dev evals: run `select_bc.py`, upload the winner and checkpoint-1098 (model files only, no `optimizer.pt`) plus provenance to `Jasonning/c2kv/b_history/arm-{C,B}/seed-42/`, verify the HF tree against local sizes, write `FINALIZED.json` |

Decisions taken during the run, in order: only the selected checkpoint is
pushed (cloud storage); ratio 8 is the deliverable and ratio 4 is also
evaluated; selection combines both ratios; checkpoint-1098 is shipped
unevaluated so no eval tail follows training (its dev evaluation, if wanted,
happens elsewhere). The candidate set was fixed before any dev score existed
for the excluded step. A first offline-W&B attempt (76 min) was discarded and
restarted online; steps 1-2 reproduced bit-identical losses.
