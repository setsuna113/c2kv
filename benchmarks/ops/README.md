# Serving and validation

Use the consolidated `task/bdf-pilot` branches of both repositories. Serve
from an explicit `SGLANG_DIR`; the launchers no longer select an old patched
tarball. Historical patches in `backends/sglang_patches/` are reproduction
material and are not applied to the consolidated tree.

`launch_sgl1088.sh` runs the server in the foreground. `MODEL_PATH`,
`PYTHON_BIN`, `SGLANG_DIR`, `DEVICE`, `PORT`, `QUERY_PROJECTION`, pool sizes
and context length are environment overrides. For checkpoint-1088 the
default is `QUERY_PROJECTION=base`. `gist` is the later local-fork rule;
see `docs/c2kv_semantics.md` before selecting it for a checkpoint.

`launch_sgl_proxy.sh <arm> <port> <upstream_port> [suffix] [extra...]` starts
a standalone proxy from this checkout. An occupied port is an error; the
launcher does not stop another service. Normally use `benchmarks/run.py`,
which owns the proxy for one benchmark run.

## Environment map

| Component | NPU-host Python/environment |
|---|---|
| SGLang server | `/home/liuyancheng/envs/sgl/bin/python` |
| Torch-backed D/G tests | `/home/liuyancheng/envs/c2kv/bin/python` |
| BFCL / proxy | `/home/liuyancheng/envs/bench/bin/python` |
| tau2 | `/home/liuyancheng/envs/bench312/bin/python` |
| ToolSandbox | `/home/liuyancheng/envs/benchts/bin/python` |

NPU processes need the CANN and ATB environment setup already sourced by
`launch_sgl1088.sh`. The launcher uses `--disable-cuda-graph`; graph mode
must not be assumed to preserve per-request projection routing.

## Gates

Run the CPU contracts from the C2KV repository:

```bash
python -m pytest agent metrology python/train python/models benchmarks -q
```

Then check the algorithm against a pinned original checkout (the original
source is an input, not a vendored duplicate):

```bash
python benchmarks/ops/check_algorithm_parity.py --help
```

On the NPU host, run the checkpoint integration gates in an unused output
directory and an available device. This starts and stops only its own
server/proxy process groups:

```bash
python benchmarks/ops/validate_npu.py \
  --sglang-dir /path/to/sglang-c2kv \
  --model /home/liuyancheng/checkpoints_upstream/checkpoint-1088 \
  --device 1 --out /path/to/new-validation-output
```

The default runs base and gist sequentially. Each mode checks full, C2KV,
hybrid, repair placements, history-KV methods and CacheBlend, with two
identical cold/warm requests per arm. `summary.json`, raw responses, request
logs and server logs persist. Failure stops the next mode. These synthetic
requests test integration; they are not official benchmark scores.

For a server already launched by the caller:

```bash
python benchmarks/ops/server_smoke.py --upstream http://127.0.0.1:35020 \
  --query-projection base --out /path/to/new-smoke-output
```

After these gates pass, use the registered adapters in `benchmarks/run.py`
for bounded official smoke cases, then full matrices. Use separate output
directories for checkpoint, query mode and packing regime. Do not merge
historical message-packing or unrecorded projection modes into a new run.
