"""Prepare a local, non-launching T02 production staging directory."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA = "t02-local-staging-v1"
DESIGN_SCHEMA = "t02-expanded-collection-design-v1"
LABEL_SCHEMA = "t02-labeled-dataset-v1"
DEFAULT_OUTPUT = "outputs/history_system_search/evidence_sets_v1/prepared"
EXPECTED_BUDGET = {
    "eligible_task_variants": 104,
    "target_states": 120,
    "target_train_states": 80,
    "target_calibration_states": 40,
    "minimum_source_families": 26,
    "maximum_candidate_states_per_task_variant": 2,
    "maximum_selected_states_per_source_family": 8,
    "max_complete_branch_executions": 360,
    "candidate_snapshot_cap": 208,
}


class PrepareError(ValueError):
    """A staging input does not match the authorized T02 contract."""


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def _read(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PrepareError(f"cannot read JSON input {path}: {error}") from error


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _find_mappings(value: Any, key: str) -> list[Mapping[str, Any]]:
    found: list[Mapping[str, Any]] = []
    if isinstance(value, Mapping):
        item = value.get(key)
        if isinstance(item, Mapping):
            found.append(item)
        for child in value.values():
            found.extend(_find_mappings(child, key))
    elif isinstance(value, list):
        for child in value:
            found.extend(_find_mappings(child, key))
    return found


def _validate_inputs(repo: Path, design_path: Path) -> tuple[dict[str, Path], dict[str, Any]]:
    design = _read(design_path)
    if not isinstance(design, dict) or design.get("schema") != DESIGN_SCHEMA:
        raise PrepareError(f"design schema must be {DESIGN_SCHEMA}")
    if design.get("launch_authorized") is not False or design.get("resource_coordination_required") is not True:
        raise PrepareError("design must remain gated on manual resource coordination")
    if design.get("status") != "budget_authorized_waiting_for_user_resource_coordination":
        raise PrepareError("design is not in the authorized waiting state")
    for name, expected in EXPECTED_BUDGET.items():
        if design.get(name) != expected:
            raise PrepareError(f"design {name} must equal {expected}")
    first = design.get("first_state_validation", {})
    if first != {
        "mode": "first_production_state_three_branches",
        "included_in_branch_budget": True,
        "abort_on_contract_failure": True,
        "separate_full_smoke_default": False,
    }:
        raise PrepareError("first production state validation contract changed")
    bindings = design.get("source_bindings", {})
    if not isinstance(bindings, Mapping) or set(bindings) != {"tasks", "audit", "D128", "F128"}:
        raise PrepareError("design must bind tasks, audit, D128, and F128")
    inputs: dict[str, Path] = {"design": design_path}
    for name, binding in bindings.items():
        if not isinstance(binding, Mapping):
            raise PrepareError(f"source binding {name} must be an object")
        path = repo / binding.get("path", "")
        if not path.is_file() or _sha(path) != binding.get("sha256"):
            raise PrepareError(f"source binding {name} is missing or has the wrong sha256")
        inputs[name] = path
    runtime = design.get("runtime_template", {})
    runtime_path = repo / runtime.get("path", "")
    if not runtime_path.is_file() or _sha(runtime_path) != runtime.get("sha256"):
        raise PrepareError("runtime template is missing or has the wrong sha256")
    inputs["runtime_template"] = runtime_path
    manifest = _read(inputs["tasks"])
    task_ids = manifest.get("task_ids", []) if isinstance(manifest, Mapping) else []
    if len(task_ids) != 104 or len(set(task_ids)) != 104:
        raise PrepareError("expanded task manifest must contain 104 unique task IDs")
    audit = _read(inputs["audit"])
    clean = audit.get("clean_source_families", []) if isinstance(audit, Mapping) else []
    if len(clean) != 26:
        raise PrepareError("family audit must contain 26 verified source families")
    categories = {"base", "long_context", "miss_func", "miss_param"}
    grouped: dict[int, set[str]] = {}
    for task_id in task_ids:
        match = re.fullmatch(r"multi_turn_(base|long_context|miss_func|miss_param)_(\d+)", task_id)
        if match is None:
            raise PrepareError(f"invalid expanded BFCL task ID: {task_id}")
        grouped.setdefault(int(match.group(2)), set()).add(match.group(1))
    audited_groups = {row.get("source_group") for row in clean if isinstance(row, Mapping)}
    if len(grouped) != 26 or set(grouped) != audited_groups or any(value != categories for value in grouped.values()):
        raise PrepareError("task variants do not form four-category blocks for the 26 audited families")
    protected = set()
    for key in ("D128", "F128"):
        for task_id in _read(inputs[key]).get("task_ids", []):
            match = re.search(r"_(\d+)$", task_id)
            if match:
                protected.add(int(match.group(1)))
    if protected.intersection(grouped):
        raise PrepareError("expanded families overlap D128 or F128")
    smoke = _read(runtime_path)
    local_models = _find_mappings(smoke.get("resolved_configs"), "local_models")
    if len(local_models) != 1 or set(local_models[0]) != {"embedding", "reranker", "selector"}:
        raise PrepareError("runtime template must bind one complete local_models config")
    for role, settings in local_models[0].items():
        if (
            not isinstance(settings, Mapping)
            or not str(settings.get("model_name_or_path", "")).startswith("/home/liuyancheng/")
            or settings.get("device") != "npu:0"
            or settings.get("local_files_only") is not True
        ):
            raise PrepareError(f"runtime {role} model is not bound to existing local NPU weights")
    host_cap = design.get("exact_snapshot_host_byte_cap")
    minimum_host = design.get("minimum_available_host_bytes")
    if type(host_cap) is not int or host_cap <= 0 or type(minimum_host) is not int or minimum_host < host_cap:
        raise PrepareError("design host-memory caps are missing or inconsistent")
    return inputs, {
        "design": design,
        "runtime": smoke,
        "local_models": dict(local_models[0]),
    }


def _gate(*, port: bool) -> str:
    port_case = '\n    --port) PORT="$2"; shift 2 ;;' if port else ""
    port_check = '\n[[ "$PORT" =~ ^[0-9]+$ ]] || { echo "--port is required" >&2; exit 64; }' if port else ""
    return f'''APPROVED=0
DEVICE=""
PORT=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --resource-coordination-approved) APPROVED=1; shift ;;
    --physical-device) DEVICE="$2"; shift 2 ;;{port_case}
    *) echo "unknown argument: $1" >&2; exit 64 ;;
  esac
done
[[ "$APPROVED" == 1 ]] || {{ echo "manual resource coordination is required" >&2; exit 64; }}
[[ "$DEVICE" =~ ^[0-7]$ ]] || {{ echo "--physical-device must be 0..7" >&2; exit 64; }}{port_check}
'''


def _scripts(checkpoint: str, *, host_cap: int, minimum_host: int) -> dict[str, str]:
    engine = f'''#!/usr/bin/env bash
set -eo pipefail
{_gate(port=True)}
set +u
source /usr/local/Ascend/cann-8.5.0/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
set -u
ROOT="$(cd "$(dirname "${{BASH_SOURCE[0]}}")/.." && pwd)"
AVAILABLE_KIB="$(awk '/^MemAvailable:/ {{print $2}}' /proc/meminfo)"
[[ "$AVAILABLE_KIB" =~ ^[0-9]+$ ]] || {{ echo "cannot read MemAvailable" >&2; exit 69; }}
(( AVAILABLE_KIB >= {minimum_host // 1024} )) || {{ echo "insufficient host memory for exact snapshots" >&2; exit 69; }}
export ASCEND_RT_VISIBLE_DEVICES="$DEVICE" C2KV_ENABLE_EXACT_STATE=1
export C2KV_EXACT_MAX_SNAPSHOTS=208 C2KV_EXACT_MAX_HOST_BYTES={host_cap}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo
export NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export PYTHONPATH="$ROOT/sglang/python:${{PYTHONPATH:-}}"
exec /home/liuyancheng/envs/sgl/bin/python -m sglang.launch_server \\
  --model-path {checkpoint} --served-model-name d3-c1000 --model-impl sglang \\
  --device npu --attention-backend ascend --dtype bfloat16 --random-seed 0 --enable-c2kv \\
  --c2kv-gist-type dynamic-interleave --c2kv-gist-param qkv --c2kv-query-proj base \\
  --c2kv-pool-fraction 0.05 --c2kv-shadow-feature-layer -2 --enable-return-hidden-states \\
  --mem-fraction-static 0.55 --max-total-tokens 65536 --context-length 131072 \\
  --max-running-requests 1 --page-size 128 --chunked-prefill-size 256 \\
  --disable-radix-cache --disable-overlap-schedule --disable-cuda-graph \\
  --host 127.0.0.1 --port "$PORT"
'''
    collect = f'''#!/usr/bin/env bash
set -eo pipefail
{_gate(port=True)}
set +u
source /usr/local/Ascend/cann-8.5.0/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
set -u
ROOT="$(cd "$(dirname "${{BASH_SOURCE[0]}}")/.." && pwd)"
export ASCEND_RT_VISIBLE_DEVICES="$DEVICE"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo
export NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export PYTHONPATH="$ROOT/history_system/runtime/python:$ROOT/history_system/runtime:$ROOT/history_system:$ROOT/sglang/python:${{PYTHONPATH:-}}"
exec /home/liuyancheng/envs/sgl/bin/python "$ROOT/history_system/t02_bfcl.py" \\
  --bfcl-root /home/liuyancheng/benchmarks/gorilla/berkeley-function-call-leaderboard \\
  --bfcl-dependency-path /home/liuyancheng/envs/bench/lib/python3.11/site-packages \\
  --task-manifest "$ROOT/configs/tasks.json" --family-audit "$ROOT/configs/audit.json" \\
  --d128-manifest "$ROOT/configs/D128.json" --f128-manifest "$ROOT/configs/F128.json" \\
  --design "$ROOT/configs/runtime.production.json" --checkpoint {checkpoint} \\
  --backend-url "http://127.0.0.1:$PORT" --out "$ROOT/run" --seed 0 \\
  --target-states 120 --train-states 80 --min-task-groups 26 \\
  --max-states-per-group 8 --max-states-per-task 2 \\
  --max-branch-executions 360 --consumed-complete-branch-executions 0 \\
  --candidate-snapshot-cap 208 --backend-snapshot-cap 208 \\
  --resource-coordination-approved
'''
    train = f'''#!/usr/bin/env bash
set -eo pipefail
{_gate(port=False)}
ROOT="$(cd "$(dirname "${{BASH_SOURCE[0]}}")/.." && pwd)"
LABELS="$ROOT/run/labels.json"
[[ -f "$LABELS" ]] || {{ echo "missing labels: $LABELS" >&2; exit 66; }}
set +u
source /usr/local/Ascend/cann-8.5.0/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
set -u
export ASCEND_RT_VISIBLE_DEVICES="$DEVICE"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo
export NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export PYTHONPATH="$ROOT/history_system/runtime/python:$ROOT/history_system/runtime:$ROOT/history_system:${{PYTHONPATH:-}}"
/home/liuyancheng/envs/sgl/bin/python "$ROOT/history_system/t02_prepare.py" extract-prefill-contract \\
  --labels "$LABELS" --output "$ROOT/configs/prefill_contract.json"
run_training() {{
  local model="$1"; shift
  /home/liuyancheng/envs/sgl/bin/python -c 'import runpy,sys; sys.path.append(sys.argv.pop(1)); sys.argv[0]="set_training"; runpy.run_module("benchmarks.memory_runtime.recovery.set_training",run_name="__main__")' \\
    /home/liuyancheng/envs/bench/lib/python3.11/site-packages "$model" "$LABELS" "$@"
}}
# The risk comparator must not gate the gain-model mainline, or vice versa.
FIT_STATUS=0
run_training c4 "$ROOT/training/c4" --tokenizer {checkpoint} --local-models "$ROOT/configs/local_models.c4.json" || FIT_STATUS=1
run_training c1 "$ROOT/training/c1_risk.json" --prefill-contract "$ROOT/configs/prefill_contract.json" || FIT_STATUS=1
exit "$FIT_STATUS"
'''
    return {"engine.sh": engine, "collect.sh": collect, "train.sh": train}


def prepare(repo: Path, design_path: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise PrepareError(f"output already exists: {output}")
    inputs, resolved = _validate_inputs(repo, design_path)
    checkpoint = resolved["runtime"].get("checkpoint_selection", {}).get("path")
    if not isinstance(checkpoint, str) or not checkpoint.startswith("/home/liuyancheng/"):
        raise PrepareError("runtime template lacks the existing NPU checkpoint path")
    output.mkdir(parents=True)
    names = {"design": "design.json", "runtime_template": "runtime_template.json", "tasks": "tasks.json", "audit": "audit.json", "D128": "D128.json", "F128": "F128.json"}
    receipts = {}
    for key, filename in names.items():
        target = output / "configs" / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(inputs[key], target)
        receipts[key] = {"path": f"configs/{filename}", "sha256": _sha(target)}
    local_models = resolved["local_models"]
    _write(output / "configs" / "local_models.c4.json", local_models)
    production = copy.deepcopy(resolved["runtime"])
    production.update(
        name="T02 expanded exact-state collection",
        candidate_id="t02_expanded_exact_state",
        purpose="Collect 120 family-disjoint T02 states for C1/C4 training",
        run_id_template="t02_expanded104_exact_state_v1",
    )
    production.setdefault("limits", {})["tasks"] = 104
    production["base_release_lineage"] = production.pop("source", None)
    production["source"] = {
        "runtime_template_sha256": _sha(inputs["runtime_template"]),
        "source_manifest": "source_files.json",
    }
    _write(output / "configs" / "runtime.production.json", production)
    receipts["runtime_production"] = {
        "path": "configs/runtime.production.json",
        "sha256": _sha(output / "configs" / "runtime.production.json"),
    }
    design = resolved["design"]
    for name, content in _scripts(
        checkpoint,
        host_cap=design["exact_snapshot_host_byte_cap"],
        minimum_host=design["minimum_available_host_bytes"],
    ).items():
        path = output / "scripts" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8", newline="\n")
        path.chmod(0o755)
    contract = {
        "schema": SCHEMA,
        "status": "prepared_waiting_for_source_freeze_and_resource_coordination",
        "launch_authorized": False,
        "resource_coordination_approved": False,
        "source_freeze_complete": False,
        "inputs": receipts,
        "models": {"actor": checkpoint, **{key: value["model_name_or_path"] for key, value in local_models.items()}},
        "manual_entrypoints": ["scripts/engine.sh", "scripts/collect.sh", "scripts/train.sh"],
        "runtime_arguments": {"physical_device": "required at execution", "engine_port": "required by engine and collect"},
        "first_state_validation": resolved["design"]["first_state_validation"],
        "complete_branch_budget": 360,
        "exact_snapshot_host_byte_cap": design["exact_snapshot_host_byte_cap"],
        "minimum_available_host_bytes": design["minimum_available_host_bytes"],
        "automatic_reruns": 0,
    }
    _write(output / "launch_contract.json", contract)
    return contract


def extract_prefill_contract(labels: Path, output: Path) -> dict[str, Any]:
    value = _read(labels)
    if not isinstance(value, Mapping):
        raise PrepareError("labels must be a JSON object")
    rows = value.get("rows")
    if value.get("schema") != LABEL_SCHEMA or not isinstance(rows, list) or not rows:
        raise PrepareError("labels must be a nonempty t02-labeled-dataset-v1 artifact")
    if any(not isinstance(row, Mapping) for row in rows):
        raise PrepareError("every labeled row must be a JSON object")
    questions = [row.get("q") for row in rows]
    if any(not isinstance(question, Mapping) for question in questions):
        raise PrepareError("every labeled row must contain a q object")
    contracts = [question.get("prefill_contract") for question in questions]
    if any(not isinstance(item, Mapping) or not item for item in contracts):
        raise PrepareError("every labeled row must contain q.prefill_contract")
    encoded = {json.dumps(item, sort_keys=True, separators=(",", ":"), allow_nan=False) for item in contracts}
    if len(encoded) != 1:
        raise PrepareError("labeled rows mix prefill contracts")
    contract = dict(contracts[0])
    _write(output, contract)
    return contract


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_cmd = commands.add_parser("prepare")
    prepare_cmd.add_argument("--design", type=Path, default=Path("experiments/history_system/configs/t02.expanded.design.json"))
    prepare_cmd.add_argument("--output", type=Path, default=Path(DEFAULT_OUTPUT))
    contract_cmd = commands.add_parser("extract-prefill-contract")
    contract_cmd.add_argument("--labels", type=Path, required=True)
    contract_cmd.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    repo = _root()
    if args.command == "prepare":
        result = prepare(repo, (repo / args.design).resolve(), (repo / args.output).resolve())
    else:
        result = extract_prefill_contract(args.labels.resolve(), args.output.resolve())
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
