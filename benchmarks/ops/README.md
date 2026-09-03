# Ops: live launchers for the SGLang eval path

The two scripts vendored here are the **only** ops tooling still on the
live SGLang evaluation path (verified 2026-09-03 by auditing every script
under `~/bench_queue/`, `~/bench_logs/` and `~/bench_results/` on the NPU
server). Everything else there is hf_server-era or one-off history — see
"Not vendored" below. Server-side live copies remain the deployed
originals; these in-repo copies are the version-controlled reference with
absolute paths demoted to env-overridable defaults.

## Scripts

* `launch_sgl1088.sh` — boots the SGLang c2kv server (default dev3 :35000,
  checkpoint-1088 as `c2kv-agent`, `--enable-c2kv --c2kv-query-proj gist
  --disable-cuda-graph`). Env overrides: `SGLANG_DIR`, `PYTHON_BIN`,
  `MODEL_PATH`, `PORT`, `DEVICE`. The in-repo copy adds
  `--c2kv-query-proj gist` over the server original because
  `benchmarks/README.md` mandates training-consistent projections and the
  deployed fork predates the flag.
* `launch_sgl_proxy.sh <arm> <port> <upstream_port> [suffix] [extra...]` —
  (re)starts the arm proxy in front of the SGLang server, kills the previous
  proxy on the port, logs under `$LOG_DIR`. Env overrides: `REPO_DIR`,
  `LOG_DIR`, `PY`.

## Environment map (NPU server)

| Component | venv / python | Notes |
|---|---|---|
| SGLang server | `~/envs/sgl/bin/python` | tree from `SGLANG_DIR` via PYTHONPATH precedence |
| proxy.py / BFCL | `~/envs/bench/bin/python` | BFCL must run with cwd inside the gorilla checkout |
| τ²-bench | `~/envs/bench312/bin/python` | `TAU2_DIR=~/benchmarks/tau2` |
| ToolSandbox | `~/envs/benchts/bin/python` | `TS_DIR=~/benchmarks/ToolSandbox` |
| hf_server (contrast only) | `~/envs/c2kv/bin/python` | retired from the eval path |

NPU env: `source /usr/local/Ascend/cann-8.5.0/set_env.sh` +
`/usr/local/Ascend/nnal/atb/set_env.sh`, unset http(s)_proxy, keep
`NO_PROXY=127.0.0.1,localhost`.

Port conventions: SGLang servers 35000+ (35000 dev3, a second instance was
seen on 35002), per-task arm proxies 34200+ (hash of the task name), ad-hoc
proxies 34100+/35100+. Device affinity: the sgl server pins
`ASCEND_RT_VISIBLE_DEVICES` (3 in the default script); respect the team
device-allocation rules before changing it.

## Not vendored (audited 2026-09-03, deliberately out of repo)

* `~/bench_queue/{worker,run_one_task,delayed_worker}.sh` and
  `~/bench_logs/{launch_hf,launch_proxy}.sh` — the hf_server-era task
  queue. Structurally superseded by `benchmarks/run.py` (which owns proxy
  lifecycle, adapter dispatch and terminal-state checks in one command);
  kept server-side only as the record of how the Aug-31 `bx_*` runs were
  produced. If the hfserver contrast backend is ever rerun, run it through
  `run.py --backend hfserver`, not the queue.
* `~/bench_results/bfcl_arm.py` — mirror of
  `benchmarks/adapters/bfcl_adapter.py`; the adapter is canonical (adds
  terminal-state enforcement). The mirror's extra `--partial-eval` /
  `bfcl_scores.log` append were queue-only conveniences.
* `~/bench_logs/sgl_deploy/*.patch` (7 files) — 2 of 7
  (`split_qkv_rmsnorm_rope`, negative-extend clamp) are load-bearing and
  already encoded in `benchmarks/backends/sglang_patches/0001-*.patch`;
  the other 5 are stale working-copy patches superseded by the fork's own
  NPU commits. `0001` is the deployment source of record.
* `~/bench_results/collect_all_r2.sh` — one-shot collector for the frozen
  Aug-30 r2 matrix.
