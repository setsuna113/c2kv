#!/usr/bin/env bash
set -Eeuo pipefail

: "${CKPT:?set CKPT=/absolute/path/to/gist/checkpoint}"

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
GU_BASE="${GU_BASE:-/inspire/hdd/global_user/yanjunchi-24040}"
BENCH_ROOT="${BENCH_ROOT:-$GU_BASE/bench-sglang-h200}"
RESULT_ROOT_ENV="${RESULT_ROOT:-}"
CKPT_NAME="${CKPT_NAME:-}"
PROFILE_PYTHON="${PROFILE_PYTHON:-python3}"
C2KV_CHECKPOINT_PROFILE="${C2KV_CHECKPOINT_PROFILE:-}"
C2KV_REFERENCE_PROFILE="${C2KV_REFERENCE_PROFILE:-}"
C2KV_RUN_CONFIG="${C2KV_RUN_CONFIG:-}"
C2KV_TRAIN_MANIFEST="${C2KV_TRAIN_MANIFEST:-}"
C2KV_QUERY_PROJ_OVERRIDE="${C2KV_QUERY_PROJ:-}"
SETUP="${SETUP:-1}"
INSTALL_SYSTEM_DEPS="${INSTALL_SYSTEM_DEPS:-1}"
RESUME="${RESUME:-1}"
OVERWRITE="${OVERWRITE:-0}"
SMOKE="${SMOKE:-0}"
ARMS="${ARMS:-full c2kv c2kv16 hybrid}"
BENCHMARKS="${BENCHMARKS:-tau2 bfcl toolsandbox}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen/Qwen3-4B-Instruct-2507-FC}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-34000}"
PROXY_PORT_BASE="${PROXY_PORT_BASE:-34100}"
TAU2_TASK_SET="${TAU2_TASK_SET:-airline}"
BFCL_CATEGORIES="${BFCL_CATEGORIES:-multi_turn_base}"
TAU2_SMOKE_TASKS="${TAU2_SMOKE_TASKS:-1}"
TAU2_SMOKE_TRIALS="${TAU2_SMOKE_TRIALS:-1}"
TAU2_SMOKE_MAX_STEPS="${TAU2_SMOKE_MAX_STEPS:-12}"
TAU2_SMOKE_TIMEOUT_SEC="${TAU2_SMOKE_TIMEOUT_SEC:-300}"
BFCL_SMOKE_RUN_IDS="${BFCL_SMOKE_RUN_IDS:-multi_turn_base_1}"
TOOLSANDBOX_SMOKE_SCENARIO="${TOOLSANDBOX_SMOKE_SCENARIO:-send_message_with_contact_content_cellular_off_multiple_user_turn}"
# TOOLSANDBOX_FULL=1 forces benchmarks/run.py --full (whole ToolSandbox suite)
# even under SMOKE=1. Default 0 keeps today's behaviour exactly: --full in a
# normal run, --ts-scenarios in a smoke run. run.py rejects --full combined
# with --ts-scenarios, so the two are mutually exclusive.
# A non-smoke run ALREADY passes --full, so TOOLSANDBOX_FULL only has an
# effect together with SMOKE=1 (the script says so at run time).
TOOLSANDBOX_FULL="${TOOLSANDBOX_FULL:-0}"
# Serving knobs are owned by launch_sglang_h200.sh; they are resolved here too
# so the run manifest records the value that was actually served rather than a
# literal that can drift away from the launcher.
C2KV_POOL_FRACTION="${C2KV_POOL_FRACTION:-0.06}"
MATRIX_FEATURES="${MATRIX_FEATURES:-cacheblend_repair_extract_v1}"
export C2KV_POOL_FRACTION

SGLANG_URL="${SGLANG_URL:-git@github.com:setsuna113/kvoffload-sglang-c2kv.git}"
# Consolidated bdf-pilot tip: preserves the tool-aware segment insertion point
# and the serving/cache contracts used by the H200 matrix.
SGLANG_COMMIT="${SGLANG_COMMIT:-078744c594934c93230ce93c6be5142ba03d47f9}"
TAU2_URL="${TAU2_URL:-git@github.com:sierra-research/tau2-bench.git}"
TAU2_COMMIT="${TAU2_COMMIT:-a2c024725189473d2d7cea3a5cfdbcc67478e41f}"
BFCL_URL="${BFCL_URL:-git@github.com:ShishirPatil/gorilla.git}"
BFCL_COMMIT="${BFCL_COMMIT:-6ea57973c7a6097fd7c5915698c54c17c5b1b6c8}"
TOOLSANDBOX_URL="${TOOLSANDBOX_URL:-git@github.com:apple/ToolSandbox.git}"
TOOLSANDBOX_COMMIT="${TOOLSANDBOX_COMMIT:-165848b9a78cead7ca7fe7c89c688b58e6501219}"

SGLANG_REPO="$BENCH_ROOT/kvoffload-sglang-c2kv"
TAU2_DIR="$BENCH_ROOT/tau2"
BFCL_MONOREPO="$BENCH_ROOT/gorilla"
BFCL_DIR="$BFCL_MONOREPO/berkeley-function-call-leaderboard"
TOOLSANDBOX_DIR="$BENCH_ROOT/ToolSandbox"
SGLANG_VENV="$BENCH_ROOT/venv-sglang"
TAU2_VENV="$BENCH_ROOT/venv-tau2"
BFCL_VENV="$BENCH_ROOT/venv-bfcl"
TOOLSANDBOX_VENV="$BENCH_ROOT/venv-toolsandbox"
SGLANG_PYTHON="$SGLANG_VENV/bin/python"
UV_BIN="${UV_BIN:-/inspire/hdd/project/wuliqifa/yanjunchi-24040/uv/uv}"
UV_CACHE_DIR="${UV_CACHE_DIR:-$GU_BASE/uv-cache}"
BENCH_PYTHON_VERSION="${BENCH_PYTHON_VERSION:-3.12}"

CKPT_PATH=$(realpath "$CKPT")
if [[ ! -f "$CKPT_PATH/config.json" || ! -f "$CKPT_PATH/model.safetensors" ]]; then
  echo "FATAL: CKPT is not a complete checkpoint: $CKPT_PATH" >&2
  exit 2
fi
if [[ -z "$CKPT_NAME" ]]; then
  CKPT_NAME="$(basename "$(dirname "$CKPT_PATH")")_$(basename "$CKPT_PATH")"
fi
RESULT_ROOT=${RESULT_ROOT_ENV:-$GU_BASE/bench_results/$CKPT_NAME}
GATE_DIR="$RESULT_ROOT/gates"
mkdir -p "$RESULT_ROOT" "$GATE_DIR" "$BENCH_ROOT" "$UV_CACHE_DIR"
RESOLVED_PROFILE="$RESULT_ROOT/checkpoint_profile.resolved.json"
RESOLVED_PROFILE_SHELL="$RESULT_ROOT/checkpoint_profile.resolved.sh"
PROFILE_ARGS=(
  --checkpoint "$CKPT_PATH"
  --out "$RESOLVED_PROFILE"
  --shell-out "$RESOLVED_PROFILE_SHELL"
  --require-serving-e2e
)
if [[ -n "$C2KV_CHECKPOINT_PROFILE" ]]; then
  PROFILE_ARGS+=(--profile "$C2KV_CHECKPOINT_PROFILE")
fi
if [[ -n "$C2KV_REFERENCE_PROFILE" ]]; then
  PROFILE_ARGS+=(--reference-profile "$C2KV_REFERENCE_PROFILE")
fi
if [[ -n "$C2KV_RUN_CONFIG" ]]; then
  PROFILE_ARGS+=(--run-config "$C2KV_RUN_CONFIG")
fi
if [[ -n "$C2KV_TRAIN_MANIFEST" ]]; then
  PROFILE_ARGS+=(--train-manifest "$C2KV_TRAIN_MANIFEST")
fi
if [[ -n "$C2KV_QUERY_PROJ_OVERRIDE" ]]; then
  PROFILE_ARGS+=(--query-projection "$C2KV_QUERY_PROJ_OVERRIDE")
fi
"$PROFILE_PYTHON" "$REPO_ROOT/benchmarks/checkpoint_profile.py" "${PROFILE_ARGS[@]}"
# shellcheck disable=SC1090
source "$RESOLVED_PROFILE_SHELL"
export C2KV_QUERY_PROJ C2KV_RESOLVED_PROFILE="$RESOLVED_PROFILE"

log() {
  printf '[%s] %s\n' "$(date -Is)" "$*"
}

ensure_repo() {
  local name=$1 url=$2 commit=$3 path=$4
  if [[ -e "$path" && ! -d "$path/.git" ]]; then
    echo "FATAL: $name path exists and is not a git repository: $path" >&2
    exit 2
  fi
  if [[ ! -d "$path/.git" ]]; then
    log "clone $name from $url"
    git clone "$url" "$path"
  fi
  if ! git -C "$path" cat-file -e "$commit^{commit}" 2>/dev/null; then
    git -C "$path" fetch origin
  fi
  if ! git -C "$path" diff --quiet || ! git -C "$path" diff --cached --quiet; then
    echo "FATAL: dedicated $name checkout is dirty: $path" >&2
    exit 2
  fi
  git -C "$path" checkout --quiet "$commit"
  log "$name pinned to $(git -C "$path" rev-parse HEAD)"
}

setup_venv() {
  local name=$1 venv=$2 install_path=$3 commit=$4
  local marker="$venv/.c2kv-install-marker"
  local expected="$install_path|$commit"
  if [[ -f "$marker" && "$(cat "$marker")" == "$expected" && -x "$venv/bin/python" ]]; then
    log "$name venv already installed"
    ensure_adapter_pins "$install_path" "$venv"
    return
  fi
  if [[ -d "$venv" && ! -x "$venv/bin/python" ]]; then
    echo "FATAL: broken venv: $venv" >&2
    exit 2
  fi
  if [[ ! -d "$venv" ]]; then
    "$UV_BIN" venv --python "$BENCH_PYTHON_VERSION" "$venv"
  fi
  "$UV_BIN" pip install --python "$venv/bin/python" -e "$install_path"
  if [[ "$install_path" == "$BFCL_DIR" ]]; then
    # qwen_agent imports soundfile while BFCL's pyproject at this pin does not
    # declare it, even though its model-config import graph eagerly loads Qwen.
    "$UV_BIN" pip install --python "$venv/bin/python" "soundfile==0.13.1"
  fi
  if [[ "$install_path" == "$TOOLSANDBOX_DIR" ]]; then
    # ToolSandbox pins openai==1.17.0, whose client still passes the removed
    # httpx `proxies` argument; keep the compatible HTTP client explicitly.
    "$UV_BIN" pip install --python "$venv/bin/python" "httpx==0.27.2"
  fi
  ensure_adapter_pins "$install_path" "$venv"
  printf '%s\n' "$expected" > "$marker"
}

ensure_adapter_pins() {
  local install_path=$1 venv=$2
  if [[ "$install_path" == "$BFCL_DIR" ]] && ! "$venv/bin/python" -c 'import soundfile' >/dev/null 2>&1; then
    "$UV_BIN" pip install --python "$venv/bin/python" "soundfile==0.13.1"
  fi
  if [[ "$install_path" == "$TOOLSANDBOX_DIR" ]] && ! "$venv/bin/python" -c 'import httpx; assert httpx.__version__ == "0.27.2"' >/dev/null 2>&1; then
    "$UV_BIN" pip install --python "$venv/bin/python" "httpx==0.27.2"
  fi
}

ensure_host_libnuma() {
  if ldconfig -p 2>/dev/null | grep -q 'libnuma\.so\.1'; then
    return
  fi
  if [[ "$INSTALL_SYSTEM_DEPS" != "1" ]]; then
    echo "FATAL: libnuma.so.1 is required by sgl_kernel; install libnuma1 or set INSTALL_SYSTEM_DEPS=1" >&2
    exit 2
  fi
  if ! command -v apt-get >/dev/null 2>&1; then
    echo "FATAL: libnuma.so.1 is missing and apt-get is unavailable" >&2
    exit 2
  fi
  log "installing host dependency libnuma1 for sgl_kernel"
  apt-get update -qq
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq libnuma1
}

compiler_has_concepts() {
  local compiler=$1
  command -v "$compiler" >/dev/null 2>&1 || return 1
  printf '#include <concepts>\n' | "$compiler" -x c++ -std=c++20 -E - >/dev/null 2>&1
}

ensure_host_cpp20_toolchain() {
  if compiler_has_concepts g++; then
    return
  fi
  local candidate=""
  if compiler_has_concepts g++-12; then
    candidate=g++-12
  elif compiler_has_concepts g++-11; then
    candidate=g++-11
  elif compiler_has_concepts g++-10; then
    candidate=g++-10
  else
    if [[ "$INSTALL_SYSTEM_DEPS" != "1" ]]; then
      echo "FATAL: sglang JIT needs a host compiler with C++20 <concepts>" >&2
      exit 2
    fi
    if ! command -v apt-get >/dev/null 2>&1; then
      echo "FATAL: no C++20 host compiler and apt-get is unavailable" >&2
      exit 2
    fi
    log "installing host dependency g++-10 for sglang JIT"
    apt-get update -qq
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq g++-10
    candidate=g++-10
  fi

  local prefix=${candidate#g++}
  local toolchain_bin="$BENCH_ROOT/toolchain-bin"
  mkdir -p "$toolchain_bin"
  for tool in gcc g++ cc c++; do
    ln -sf "/usr/bin/${tool}${prefix}" "$toolchain_bin/$tool"
  done
  export CC="$toolchain_bin/gcc"
  export CXX="$toolchain_bin/g++"
  export CUDAHOSTCXX="$toolchain_bin/g++"
  export PATH="$toolchain_bin:$PATH"
  log "using C++20 host toolchain $candidate via $toolchain_bin"
}

wait_for_health() {
  local timeout=${1:-1800}
  local server_pid=${2:-}
  local started=$SECONDS
  while true; do
    if [[ -n "$server_pid" ]] && ! kill -0 "$server_pid" 2>/dev/null; then
      return 1
    fi
    if "$SGLANG_PYTHON" - "$HOST" "$PORT" <<'PY' >/dev/null 2>&1
import sys
import urllib.request
urllib.request.urlopen(f"http://{sys.argv[1]}:{sys.argv[2]}/health", timeout=3).close()
PY
    then
      return 0
    fi
    if (( SECONDS - started >= timeout )); then
      return 1
    fi
    sleep 5
  done
}

SERVER_WRAPPER_PID=""

stop_server() {
  if [[ -n "$SERVER_WRAPPER_PID" ]] && kill -0 "$SERVER_WRAPPER_PID" 2>/dev/null; then
    kill "$SERVER_WRAPPER_PID" 2>/dev/null || true
    wait "$SERVER_WRAPPER_PID" 2>/dev/null || true
  fi
  SERVER_WRAPPER_PID=""
}

start_server() {
  local log_path=$1
  log "launching sglang at $BASE_URL (log=$log_path)"
  (
    cd "$BENCH_ROOT"
    exec env HOST="$HOST" PORT="$PORT" CKPT="$CKPT_PATH" \
      SGLANG_VENV="$SGLANG_VENV" SERVED_MODEL_NAME="$SERVED_MODEL_NAME" \
      SGLANG_LOG="$log_path" C2KV_RESOLVED_PROFILE="$RESOLVED_PROFILE" \
      C2KV_PROFILE_WORK_DIR="$RESULT_ROOT/server_profile" \
      bash "$REPO_ROOT/benchmarks/launch_sglang_h200.sh"
  ) >"${log_path%.log}_launcher.out" 2>&1 &
  SERVER_WRAPPER_PID=$!
  if ! wait_for_health "${HEALTH_TIMEOUT_SEC:-1800}" "$SERVER_WRAPPER_PID"; then
    echo "FATAL: sglang did not become healthy; log=$log_path" >&2
    tail -100 "$log_path" >&2 || true
    exit 2
  fi
}

cleanup() {
  local status=$?
  stop_server
  exit "$status"
}
trap cleanup EXIT INT TERM

if [[ "$SETUP" != "0" ]]; then
  ensure_repo sglang "$SGLANG_URL" "$SGLANG_COMMIT" "$SGLANG_REPO"
  ensure_repo tau2 "$TAU2_URL" "$TAU2_COMMIT" "$TAU2_DIR"
  ensure_repo bfcl "$BFCL_URL" "$BFCL_COMMIT" "$BFCL_MONOREPO"
  ensure_repo toolsandbox "$TOOLSANDBOX_URL" "$TOOLSANDBOX_COMMIT" "$TOOLSANDBOX_DIR"

  setup_venv sglang "$SGLANG_VENV" "$SGLANG_REPO/python" "$SGLANG_COMMIT"
  setup_venv tau2 "$TAU2_VENV" "$TAU2_DIR" "$TAU2_COMMIT"
  setup_venv bfcl "$BFCL_VENV" "$BFCL_DIR" "$BFCL_COMMIT"
  setup_venv toolsandbox "$TOOLSANDBOX_VENV" "$TOOLSANDBOX_DIR" "$TOOLSANDBOX_COMMIT"
else
  for python_bin in "$SGLANG_PYTHON" "$TAU2_VENV/bin/python" "$BFCL_VENV/bin/python" "$TOOLSANDBOX_VENV/bin/python"; do
    if [[ ! -x "$python_bin" ]]; then
      echo "FATAL: SETUP=0 but missing venv python: $python_bin" >&2
      exit 2
    fi
  done
fi
ensure_host_libnuma
ensure_host_cpp20_toolchain

BASE_URL="http://$HOST:$PORT"
export NO_PROXY="127.0.0.1,localhost"
export no_proxy="$NO_PROXY"
export BFCL_DIR
export TAU2_DIR
export TOOLSANDBOX_DIR
export SERVED_MODEL_NAME

log "static checkpoint gate"
"$SGLANG_PYTHON" "$REPO_ROOT/benchmarks/sglang_smoke.py" checkpoint \
  --checkpoint "$CKPT_PATH" --out "$GATE_DIR/checkpoint.json"

SGLANG_LOG="$RESULT_ROOT/sglang.log"
start_server "$SGLANG_LOG"

log "S1 extract gate"
"$SGLANG_PYTHON" "$REPO_ROOT/benchmarks/sglang_smoke.py" service \
  --base-url "$BASE_URL" --out "$GATE_DIR/S1_extract.json"

log "S2 tool-call gate"
"$SGLANG_PYTHON" "$REPO_ROOT/benchmarks/sglang_smoke.py" tools \
  --base-url "$BASE_URL" --served-model-name "$SERVED_MODEL_NAME" \
  --out "$GATE_DIR/S2_tools.json"

log "variable-length flex attention gate"
"$SGLANG_PYTHON" "$REPO_ROOT/benchmarks/sglang_smoke.py" flex \
  --base-url "$BASE_URL" --checkpoint "$CKPT_PATH" --server-log "$SGLANG_LOG" \
  --out "$GATE_DIR/flex_attention_lengths.json"

log "S3 proxy-arm gate"
"$SGLANG_PYTHON" "$REPO_ROOT/benchmarks/sglang_smoke.py" proxy \
  --base-url "$BASE_URL" --served-model-name "$SERVED_MODEL_NAME" \
  --checkpoint "$CKPT_PATH" --checkpoint-profile "$RESOLVED_PROFILE" \
  --log-dir "$GATE_DIR" --out "$GATE_DIR/S3_proxy.json"

log "S6 tools-through-proxy gate"
"$SGLANG_PYTHON" "$REPO_ROOT/benchmarks/sglang_smoke.py" tools-proxy \
  --base-url "$BASE_URL" --served-model-name "$SERVED_MODEL_NAME" \
  --checkpoint "$CKPT_PATH" --checkpoint-profile "$RESOLVED_PROFILE" \
  --log-dir "$GATE_DIR" \
  --out "$GATE_DIR/S6_tools_through_proxy.json"

read -r -a BENCHMARK_LIST <<<"$BENCHMARKS"
read -r -a ARM_LIST <<<"$ARMS"
if [[ "$OVERWRITE" == "1" ]]; then
  echo "FATAL: generic matrix evidence is immutable; choose a new RESULT_ROOT instead of OVERWRITE=1" >&2
  exit 2
fi

MATRIX_SPEC="$RESULT_ROOT/matrix.json"
MANIFEST="$RESULT_ROOT/run_manifest.json"
"$SGLANG_PYTHON" - "$RESOLVED_PROFILE" "$MATRIX_SPEC" "$MANIFEST" \
  "$CKPT_PATH" "$REPO_ROOT" "$SGLANG_REPO" "$TAU2_DIR" "$BFCL_MONOREPO" \
  "$TOOLSANDBOX_DIR" "$BASE_URL" "$SERVED_MODEL_NAME" "$PROXY_PORT_BASE" \
  "$NUM_WORKERS" "$SMOKE" "$TAU2_TASK_SET" "$TAU2_SMOKE_TASKS" \
  "$TAU2_SMOKE_TRIALS" "$TAU2_SMOKE_MAX_STEPS" "$TAU2_SMOKE_TIMEOUT_SEC" \
  "$BFCL_CATEGORIES" "$BFCL_SMOKE_RUN_IDS" "$TOOLSANDBOX_FULL" \
  "$TOOLSANDBOX_SMOKE_SCENARIO" "$C2KV_POOL_FRACTION" "$MATRIX_FEATURES" \
  "$TAU2_VENV/bin/python" "$BFCL_VENV/bin/python" "$TOOLSANDBOX_VENV/bin/python" \
  "${#BENCHMARK_LIST[@]}" "${BENCHMARK_LIST[@]}" "${ARM_LIST[@]}" <<'PY'
import datetime
import json
import shlex
import subprocess
import sys

args = sys.argv[1:]
(profile_path, spec_path, manifest_path, checkpoint, repo, sglang, tau2, bfcl,
 toolsandbox, base_url, model, proxy_port, workers, smoke, tau2_task_set,
 tau2_smoke_tasks, tau2_smoke_trials, tau2_smoke_steps, tau2_smoke_timeout,
 bfcl_categories, bfcl_smoke_ids, toolsandbox_full, toolsandbox_scenario,
 pool_fraction, feature_text, tau2_python, bfcl_python,
 toolsandbox_python) = args[:28]
benchmark_count = int(args[28])
benchmark_names = args[29:29 + benchmark_count]
arms = args[29 + benchmark_count:]
with open(profile_path, encoding="utf-8") as handle:
    profile = json.load(handle)
features = [
    item
    for token in shlex.split(feature_text.replace(",", " "))
    for item in [token.strip()]
    if item
]

common_args = [
    "--num-workers", workers,
    "--doc-packing", profile["serving"]["doc_packing"],
    "--max-doc-length", str(profile["serving"]["max_doc_length"]),
    "--max-doc-num", str(profile["serving"]["max_doc_num"]),
]
benchmark_configs = {}
for name in benchmark_names:
    if name == "tau2":
        run_args = ["--task-set", tau2_task_set]
        if smoke == "1":
            run_args += [
                "--max-tasks", tau2_smoke_tasks,
                "--tau2-num-trials", tau2_smoke_trials,
                "--tau2-max-steps", tau2_smoke_steps,
                "--tau2-timeout", tau2_smoke_timeout,
            ]
        benchmark_configs[name] = {
            "runner_python": tau2_python,
            "run_args": run_args,
            "options": {"tau2_dir": tau2},
        }
    elif name == "bfcl":
        run_args = ["--categories", bfcl_categories]
        if smoke == "1":
            run_args += ["--run-ids", bfcl_smoke_ids]
        benchmark_configs[name] = {
            "runner_python": bfcl_python,
            "run_args": run_args,
            "options": {"bfcl_dir": f"{bfcl}/berkeley-function-call-leaderboard"},
        }
    elif name == "toolsandbox":
        run_args = (
            ["--ts-scenarios", toolsandbox_scenario]
            if smoke == "1" and toolsandbox_full != "1"
            else ["--full"]
        )
        benchmark_configs[name] = {
            "runner_python": toolsandbox_python,
            "run_args": run_args,
            "options": {"toolsandbox_dir": toolsandbox},
        }
    else:
        raise SystemExit(f"FATAL: H200 setup has no runner for benchmark {name!r}")

spec = {
    "schema_version": 1,
    "profile": {**profile, "path": profile_path},
    "defaults": {
        "backend": "sglang",
        "upstream": base_url,
        "user_upstream": base_url,
        "model": model,
        "proxy_port": int(proxy_port),
        "run_args": common_args,
    },
    "features": features,
    "arms": arms,
    "benchmarks": benchmark_configs,
}

def commit(path):
    return subprocess.check_output(
        ["git", "-C", path, "rev-parse", "HEAD"], text=True
    ).strip()

manifest = {
    "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "checkpoint_profile": {
        "path": profile_path,
        "kind": profile["profile_kind"],
        "fingerprint": profile["profile_fingerprint"],
        "query_projection": profile["serving"]["query_projection"],
        "doc_packing": profile["serving"]["doc_packing"],
        "max_doc_length": profile["serving"]["max_doc_length"],
        "max_doc_num": profile["serving"]["max_doc_num"],
        "compression_ratios": profile["serving"]["compression_ratios"],
        "doc_mode": profile["training"]["doc_mode"],
        "tools_in_system": profile["training"]["tools_in_system"],
    },
    "checkpoint": checkpoint,
    "c2kv_commit": commit(repo),
    "sglang_commit": commit(sglang),
    "benchmark_commits": {
        "tau2": commit(tau2), "bfcl": commit(bfcl), "toolsandbox": commit(toolsandbox),
    },
    "endpoint": base_url,
    "served_model_name": model,
    "server": {
        "features": features,
        "c2kv_pool_fraction": float(pool_fraction),
        "cuda_graph": "disabled",
    },
    "matrix_spec": spec_path,
    "reporting_note": "execution status is recorded per cell; constructing this manifest is not a pass",
}
with open(spec_path, "w", encoding="utf-8") as handle:
    json.dump(spec, handle, indent=2, ensure_ascii=False)
    handle.write("\n")
with open(manifest_path, "w", encoding="utf-8") as handle:
    json.dump(manifest, handle, indent=2, ensure_ascii=False)
    handle.write("\n")
PY

MATRIX_ARGS=(
  --matrix "$MATRIX_SPEC"
  --out "$RESULT_ROOT"
  --plan-out "$RESULT_ROOT/matrix_plan.json"
  --execute
)
if [[ "$RESUME" == "1" ]]; then
  MATRIX_ARGS+=(--resume)
fi
log "execute generic matrix (${#BENCHMARK_LIST[@]} benchmarks x ${#ARM_LIST[@]} arms)"
"$SGLANG_PYTHON" "$REPO_ROOT/benchmarks/matrix.py" "${MATRIX_ARGS[@]}"

if [[ "$SMOKE" == "1" ]]; then
  printf '%s\n' '{"gate":"S5_smoke","passed":true,"status":"all_requested_minimal_cells_completed"}' \
    >"$GATE_DIR/S5_smoke.json"
fi

log "matrix complete: $RESULT_ROOT"
