#!/usr/bin/env bash
set -Eeuo pipefail

: "${CKPT:?set CKPT=/absolute/path/to/gist/checkpoint}"

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
GU_BASE="${GU_BASE:-/inspire/hdd/global_user/yanjunchi-24040}"
BENCH_ROOT="${BENCH_ROOT:-$GU_BASE/bench-sglang-h200}"
RESULT_ROOT_ENV="${RESULT_ROOT:-}"
CKPT_NAME="${CKPT_NAME:-}"
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
TAU2_SMOKE_TASKS="${TAU2_SMOKE_TASKS:-2}"
BFCL_SMOKE_RUN_IDS="${BFCL_SMOKE_RUN_IDS:-multi_turn_base_1}"
TOOLSANDBOX_SMOKE_SCENARIO="${TOOLSANDBOX_SMOKE_SCENARIO:-send_message_with_contact_content_cellular_off_multiple_user_turn}"
# TOOLSANDBOX_FULL=1 forces benchmarks/run.py --full (whole ToolSandbox suite)
# even under SMOKE=1. Default 0 keeps today's behaviour exactly: --full in a
# normal run, --toolsandbox-scenarios in a smoke run. run.py rejects --full
# combined with --toolsandbox-scenarios, so the two are mutually exclusive.
# A non-smoke run ALREADY passes --full, so TOOLSANDBOX_FULL only has an
# effect together with SMOKE=1 (the script says so at run time).
TOOLSANDBOX_FULL="${TOOLSANDBOX_FULL:-0}"
# Serving knobs are owned by launch_sglang_h200.sh; they are resolved here too
# so the run manifest records the value that was actually served rather than a
# literal that can drift away from the launcher.
C2KV_POOL_FRACTION="${C2KV_POOL_FRACTION:-0.06}"
C2KV_QUERY_PROJ="${C2KV_QUERY_PROJ:-gist}"
export C2KV_POOL_FRACTION C2KV_QUERY_PROJ

SGLANG_URL="${SGLANG_URL:-git@github.com:setsuna113/kvoffload-sglang-c2kv.git}"
# task/c2kv-serve-align tip: renders the segment insertion point with `tools`
# (4d08 rendered it without, so every gist landed at a tool-free offset) and
# adds --c2kv-query-proj.
SGLANG_COMMIT="${SGLANG_COMMIT:-718a654e3df356e262c3318a095e1efd91c23512}"
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

summary_is_complete() {
  local summary=$1
  "$SGLANG_PYTHON" - "$summary" <<'PY'
import json
import sys
try:
    data = json.load(open(sys.argv[1], encoding="utf-8"))
except Exception:
    raise SystemExit(1)
raise SystemExit(0 if int(data.get("n") or 0) > 0 and data.get("semantic_score") is not None else 1)
PY
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
CURRENT_POOL_RATIO=""

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
      SGLANG_LOG="$log_path" \
      bash "$REPO_ROOT/benchmarks/launch_sglang_h200.sh"
  ) >"${log_path%.log}_launcher.out" 2>&1 &
  SERVER_WRAPPER_PID=$!
  if ! wait_for_health "${HEALTH_TIMEOUT_SEC:-1800}" "$SERVER_WRAPPER_PID"; then
    echo "FATAL: sglang did not become healthy; log=$log_path" >&2
    tail -100 "$log_path" >&2 || true
    exit 2
  fi
}

arm_pool_ratio() {
  "$SGLANG_PYTHON" - "$REPO_ROOT/benchmarks" "$1" <<'PY'
import sys
sys.path.insert(0, sys.argv[1])
from arms import get_arm
arm = get_arm(sys.argv[2])
print(arm.ratio if arm.compress_history else 0)
PY
}

proxy_regime() {
  # doc_packing / max_docs / max_doc_length, read from proxy.py itself so the
  # manifest cannot claim a segmentation regime the proxy does not run.
  "$SGLANG_PYTHON" - "$REPO_ROOT/benchmarks" <<'PY'
import sys
sys.path.insert(0, sys.argv[1])
import proxy
print(proxy.DOC_PACKING)
print(proxy.MAX_DOCS)
print(proxy.MAX_DOC_LENGTH)
PY
}

ensure_pool_ratio() {
  local arm=$1 ratio
  ratio=$(arm_pool_ratio "$arm")
  if [[ "$ratio" == "0" ]]; then
    return
  fi
  if [[ -n "$CURRENT_POOL_RATIO" && "$CURRENT_POOL_RATIO" != "$ratio" ]]; then
    log "restarting sglang to isolate C2KV ratio $CURRENT_POOL_RATIO -> $ratio"
    stop_server
    sleep 2
    start_server "$RESULT_ROOT/sglang_${arm}_ratio${ratio}.log"
  fi
  CURRENT_POOL_RATIO=$ratio
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
CURRENT_POOL_RATIO=8  # all pre-matrix extract gates use ratio 8

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
  --log-dir "$GATE_DIR" --out "$GATE_DIR/S3_proxy.json"

log "S6 tools-through-proxy gate"
"$SGLANG_PYTHON" "$REPO_ROOT/benchmarks/sglang_smoke.py" tools-proxy \
  --base-url "$BASE_URL" --served-model-name "$SERVED_MODEL_NAME" \
  --checkpoint "$CKPT_PATH" --log-dir "$GATE_DIR" \
  --out "$GATE_DIR/S6_tools_through_proxy.json"

cat >"$GATE_DIR/S4_repair.json" <<'EOF'
{
  "gate": "S4_repair",
  "passed": true,
  "status": "skipped",
  "reason": "repair arms are disabled by policy in this matrix; /v1/c2kv/repair_extract exists at the pinned commit but no repair arm is wired into benchmarks/arms.py"
}
EOF

read -r -a BENCHMARK_LIST <<<"$BENCHMARKS"
read -r -a ARM_LIST <<<"$ARMS"
CELL_INDEX=0
for arm in "${ARM_LIST[@]}"; do
  arm_is_complete=1
  for benchmark in "${BENCHMARK_LIST[@]}"; do
    if ! summary_is_complete "$RESULT_ROOT/${benchmark}_${arm}/summary_${arm}.json"; then
      arm_is_complete=0
      break
    fi
  done
  if [[ "$arm_is_complete" == "1" ]]; then
    log "skip complete arm $arm"
    continue
  fi
  ensure_pool_ratio "$arm"
  for benchmark in "${BENCHMARK_LIST[@]}"; do
    case "$benchmark" in
      tau2) RUN_PYTHON="$TAU2_VENV/bin/python" ;;
      bfcl) RUN_PYTHON="$BFCL_VENV/bin/python" ;;
      toolsandbox) RUN_PYTHON="$TOOLSANDBOX_VENV/bin/python" ;;
      *) echo "FATAL: unknown benchmark $benchmark" >&2; exit 2 ;;
    esac

    cell="$RESULT_ROOT/${benchmark}_${arm}"
    summary="$cell/summary_${arm}.json"
    if [[ -e "$cell" && "$RESUME" == "0" ]]; then
      echo "FATAL: RESUME=0 and target cell already exists: $cell" >&2
      exit 2
    fi
    if [[ "$OVERWRITE" == "1" && -e "$cell" ]]; then
      rm -rf -- "$cell"
    elif [[ -f "$summary" && "$RESUME" == "1" ]] && summary_is_complete "$summary"; then
      log "skip complete cell $benchmark/$arm"
      continue
    elif [[ -d "$cell" ]]; then
      rm -rf -- "$cell"
    fi

    proxy_port=$(( PROXY_PORT_BASE + CELL_INDEX ))
    CELL_INDEX=$(( CELL_INDEX + 1 ))
    log "run $benchmark/$arm on proxy port $proxy_port"
    set -o pipefail
    case "$benchmark" in
      tau2)
        extra=()
        if [[ "$SMOKE" == "1" ]]; then
          extra+=("--max-tasks" "$TAU2_SMOKE_TASKS")
        fi
        "$RUN_PYTHON" "$REPO_ROOT/benchmarks/run.py" \
          --benchmark tau2 --arm "$arm" --upstream "$BASE_URL" \
          --proxy-port "$proxy_port" --out "$cell" \
          --task-set "$TAU2_TASK_SET" --num-workers "$NUM_WORKERS" \
          --run-name "${CKPT_NAME}_tau2_${arm}" \
          "${extra[@]}" 2>&1 | tee "$RESULT_ROOT/${benchmark}_${arm}.log"
        ;;
      bfcl)
        extra=()
        if [[ "$SMOKE" == "1" ]]; then
          extra+=("--run-ids" "$BFCL_SMOKE_RUN_IDS")
        fi
        "$RUN_PYTHON" "$REPO_ROOT/benchmarks/run.py" \
          --benchmark bfcl --arm "$arm" --upstream "$BASE_URL" \
          --proxy-port "$proxy_port" --out "$cell" \
          --categories "$BFCL_CATEGORIES" \
          --served-model-name "$SERVED_MODEL_NAME" \
          "${extra[@]}" 2>&1 | tee "$RESULT_ROOT/${benchmark}_${arm}.log"
        ;;
      toolsandbox)
        extra=()
        if [[ "$SMOKE" == "1" && "$TOOLSANDBOX_FULL" != "1" ]]; then
          extra+=("--toolsandbox-scenarios" "$TOOLSANDBOX_SMOKE_SCENARIO")
        else
          if [[ "$TOOLSANDBOX_FULL" == "1" && "$SMOKE" != "1" ]]; then
            echo "[note] TOOLSANDBOX_FULL=1 is redundant outside SMOKE=1: the full suite is already the default"
          fi
          extra+=("--full")
        fi
        "$RUN_PYTHON" "$REPO_ROOT/benchmarks/run.py" \
          --benchmark toolsandbox --arm "$arm" --upstream "$BASE_URL" \
          --proxy-port "$proxy_port" --out "$cell" \
          --num-workers "$NUM_WORKERS" \
          "${extra[@]}" 2>&1 | tee "$RESULT_ROOT/${benchmark}_${arm}.log"
        ;;
    esac
  done
done

if [[ "$SMOKE" == "1" ]]; then
  printf '%s\n' '{"gate":"S5_smoke","passed":true,"status":"all_requested_minimal_cells_completed"}' \
    >"$GATE_DIR/S5_smoke.json"
fi

MANIFEST="$RESULT_ROOT/run_manifest.json"
mapfile -t PROXY_REGIME < <(proxy_regime)
"$SGLANG_PYTHON" - "$MANIFEST" "$CKPT_PATH" "$CKPT_NAME" "$REPO_ROOT" \
"$SGLANG_REPO" "$TAU2_DIR" "$BFCL_MONOREPO" "$TOOLSANDBOX_DIR" \
"$BASE_URL" "$SERVED_MODEL_NAME" "$SMOKE" "$NUM_WORKERS" "$PORT" \
"$C2KV_POOL_FRACTION" "$C2KV_QUERY_PROJ" \
"${PROXY_REGIME[0]}" "${PROXY_REGIME[1]}" "${PROXY_REGIME[2]}" \
"${#BENCHMARK_LIST[@]}" "${BENCHMARK_LIST[@]}" "${ARM_LIST[@]}" <<'PY'
import datetime
import json
import subprocess
import sys

argv = sys.argv[1:]
(out, ckpt, ckpt_name, repo, sglang, tau2, bfcl, toolsandbox, url, model, smoke,
 workers, port, pool_fraction, query_proj, doc_packing, max_docs,
 max_doc_length) = argv[:18]
benchmark_count = int(argv[18])
benchmarks = argv[19:19 + benchmark_count]
arms = argv[19 + benchmark_count:]
def commit(path):
    return subprocess.check_output(["git", "-C", path, "rev-parse", "HEAD"], text=True).strip()
manifest = {
    "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "checkpoint": ckpt,
    "checkpoint_name": ckpt_name,
    "c2kv_commit": commit(repo),
    "sglang_commit": commit(sglang),
    "benchmark_commits": {
        "tau2": commit(tau2),
        "bfcl": commit(bfcl),
        "toolsandbox": commit(toolsandbox),
    },
    "endpoint": url,
    "served_model_name": model,
    "benchmarks": benchmarks,
    "arms": arms,
    "smoke": smoke == "1",
    "num_workers": int(workers),
    "sglang_flags": {
        "enable_c2kv": True,
        "c2kv_pool_fraction": float(pool_fraction),
        "c2kv_max_tokens": 4096,
        "c2kv_query_proj": query_proj,
        "tool_call_parser": "qwen25",
        "mem_fraction_static": 0.8,
        "disable_piecewise_cuda_graph": True,
        "disable_piecewise_cuda_graph_reason": "Qwen3 C2KV data-dependent branch is incompatible with piecewise tracing",
        "disable_cuda_graph": True,
        "disable_cuda_graph_reason": "the per-token gist/base projection mask is not part of CUDA-graph capture, so a captured decode would revert to the base projections",
        "attention_backend": "sglang_default",
        "port": int(port),
    },
    "proxy_segmentation": {
        "doc_packing": doc_packing,
        "max_docs": int(max_docs),
        "max_doc_length": int(max_doc_length),
        "history_cutoff": "after_last_assistant",
    },
    "sglang_pool_isolation": {
        "restart_on_ratio_change": True,
        "reason": "the C2KV pool hashes input ids but not compression ratio; mixed-ratio arms would collide",
    },
    "repair": {
        "enabled": False,
        "reason": "repair arms are disabled by policy in this matrix; /v1/c2kv/repair_extract exists at the pinned commit but no repair arm is wired into benchmarks/arms.py",
    },
    "reporting_note": "preliminary, n=1; no historical NPU comparisons",
}
with open(out, "w", encoding="utf-8") as handle:
    json.dump(manifest, handle, indent=2, ensure_ascii=False)
    handle.write("\n")
PY

"$SGLANG_PYTHON" "$REPO_ROOT/benchmarks/summarize_matrix.py" \
  --root "$RESULT_ROOT" --manifest "$MANIFEST" \
  --output-json "$RESULT_ROOT/matrix_summary.json" \
  --output-md "$RESULT_ROOT/MATRIX_SUMMARY.md"

log "matrix complete: $RESULT_ROOT"
