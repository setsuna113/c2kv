"""Build frozen GP candidates for the mixed20 dev pack (server-side).

Reads a build plan JSON with explicit per-config card/engine assignment,
materializes one frozen candidate directory per GP switch set, and writes
design.json files that satisfy runner.py's frozen-design validation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path


def sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


# These files are rewritten per candidate; they MUST be real copies, because
# writing through a hardlink would corrupt the snapshot and every other
# candidate sharing that inode.
OVERWRITTEN_CONFIGS = {
    "configs/controller.json",
    "configs/eval_policy.json",
    "configs/eval_capacity.json",
}


def copy_runtime(src: Path, dst: Path) -> None:
    for item in sorted(src.rglob("*")):
        if item.is_dir():
            continue
        parts = item.parts
        if "__pycache__" in parts or ".pytest_cache" in parts or item.suffix == ".pyc":
            continue
        relative = item.relative_to(src)
        target = dst / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if relative.as_posix() in OVERWRITTEN_CONFIGS:
            shutil.copyfile(item, target)
            continue
        try:
            os.link(item, target)
        except OSError:
            shutil.copyfile(item, target)


def merged_mappings(run_root: Path, plan: dict, mappings: list[dict]) -> list[dict]:
    """Merge into any existing build output, updating entries by name.

    Without this, a partial rebuild of the same plan_id would drop the
    mappings of every previously built candidate of that plan.
    """
    path = run_root / ("build." + plan.get("plan_id", "latest") + ".json")
    if not path.exists():
        return mappings
    existing = {
        m["name"]: m
        for m in json.loads(path.read_text(encoding="utf-8")).get("mappings", [])
    }
    for entry in mappings:
        existing[entry["name"]] = entry
    return sorted(existing.values(), key=lambda m: m["name"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    plan = read(args.plan)
    run_root = Path(plan["run_root"])
    hs = run_root / "history_system"
    current = read(hs / "configs/current_algorithm.json")
    tasks = read(plan["tasks_file"])
    lineage = read(plan["lineage_file"])
    task_ids = tasks["task_ids"]

    sys.path[:0] = [str(hs / "runtime/python"), str(hs / "runtime")]
    from benchmarks.memory_runtime.recovery.experiment_config import configure_controller

    base_controller = read(hs / "runtime/configs/controller.json")
    mappings = []
    for row in plan["assignments"]:
        name = row["name"]
        engine_port = int(row["engine_port"])
        plan_id = plan["plan_id"]
        controller = configure_controller(base_controller, row["gp"])
        switches = controller["gp_experiments"]
        identity = hashlib.sha256(
            json.dumps(switches, sort_keys=True).encode()
        ).hexdigest()[:12]
        candidate_id = "gp_" + identity
        cand_root = run_root / "candidates"
        directory = cand_root / f"{plan_id}__{candidate_id}"
        marker = directory / "engine_port.txt"
        if directory.exists():
            if marker.exists() and marker.read_text(encoding="utf-8").strip() == str(engine_port):
                mappings.append({"name": name, "candidate_id": candidate_id,
                                 "directory": str(directory), "engine_port": engine_port,
                                 "reuse": True})
                continue
            directory = cand_root / f"{plan_id}__{candidate_id}_p{engine_port}"
        runtime_dir = directory / "runtime"
        copy_runtime(hs / "runtime", runtime_dir)
        save(runtime_dir / "configs/controller.json", controller)
        marker.write_text(str(engine_port) + "\n", encoding="utf-8")

        policy = read(hs / "runtime/configs/eval_policy.json")
        policy["policy_id"] = "a-history-gp-v1-" + candidate_id
        policy["policy"]["history_budget_bytes"] = 113246208
        policy["policy"]["workspace_budget_bytes"] = 113246208
        save(runtime_dir / "configs/eval_policy.json", policy)
        capacity = read(hs / "runtime/configs/eval_capacity.json")
        save(runtime_dir / "configs/eval_capacity.json", capacity)

        design = {
            key: current[key] for key in (
                "route", "compression_policy", "history_view_protocol",
                "decode_strategy", "prefill_chunk_size", "sampling",
                "session_cache_policy", "retry_contract")}
        design.update({
            "schema": "a-history-system-candidate-design-v1",
            "status": "frozen", "state": "frozen",
            "candidate_id": candidate_id,
            "run_id_template": "a_history_gp_v1_" + candidate_id,
            "evaluation_stage": "development_search",
            "launch_authorized": True,
            "acceptance_parameters_frozen": False,
            "task_ids": task_ids,
            "task_manifest_sha256": sha(plan["tasks_file"]),
            "limits": {**current["limits"], "tasks": len(task_ids)},
            "ratio": current["ratio"],
            "checkpoint_selection": current["checkpoint_selection"],
            "runtime": {
                "controller": "configs/controller.json",
                "eval_policy": "configs/eval_policy.json",
                "eval_capacity": "configs/eval_capacity.json",
                "shadow_feature_config": "configs/shadow_features.json",
                **{key: current["runtime"][key] for key in (
                    "server_module", "official_worker_module", "device", "dtype",
                    "bfcl_python", "npu_allocator_metrics")},
                "generation_backend": "sglang",
                "sglang_backend_url": "http://127.0.0.1:" + str(engine_port),
                "sglang_timeout_seconds": 10800,
            },
            "resolved_configs": {
                "controller": controller,
                "eval_policy": policy,
                "eval_capacity": capacity,
            },
            "source_files": {
                p.relative_to(runtime_dir).as_posix(): sha(p)
                for p in runtime_dir.rglob("*") if p.is_file()
            },
            "task_and_scorer_lineage": {
                "bfcl_root_default": plan["benchmark_dir"],
                "source_bindings": {
                    file_name: {"remote_sha256": digest}
                    for file_name, digest in lineage["files"].items()},
            },
            "search_contract": {
                "B0_history_bytes": 113246208,
                "history_token_budget": 768,
                "quality_target": "Peer mean parity working target; no delta gate for search",
                "speed_is_auxiliary": True,
                "full_task_rollout": True,
                "new_cohort_planned_reference_not_automatic_retry": True,
            },
            "cpu_validation": {
                "source": "gp_search_v1 snapshot of d3-sglang-20260915/history_system",
                "note": "GP interfaces CPU-validated 2026-09-15 per iteration_plan",
            },
            "cumulative_time_limit_seconds": None,
            "automatic_reruns": 0,
        })
        save(directory / "design.json", design)
        save(directory / "tasks.json", tasks)
        mappings.append({"name": name, "candidate_id": candidate_id,
                         "directory": str(directory), "engine_port": engine_port,
                         "reuse": False})
    save(run_root / ("build." + plan.get("plan_id", "latest") + ".json"),
         {"schema": "a-history-gp-build-v1", "mappings": merged_mappings(run_root, plan, mappings)})
    print(json.dumps({"built": len([m for m in mappings if not m["reuse"]]),
                      "reused": len([m for m in mappings if m["reuse"]]),
                      "mappings": mappings}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
