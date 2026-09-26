"""Keep one T02 shard's live snapshots on its originating engine.

The coordinator owns task claims and the global plan.  This worker owns the
in-process BFCL environments, actors, and backend snapshot IDs for its shard;
none of those live objects are serialized or transferred to another worker.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse

import t02
import t02_bfcl


CONTRACT_SCHEMA = "t02-parallel-contract-v1"
READY_SCHEMA = "t02-parallel-worker-ready-v1"
STATUS_SCHEMA = "t02-parallel-worker-status-v1"


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_write_json(path: Path, value: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(encoded)
    temporary.replace(path)
    return hashlib.sha256(encoded).hexdigest()


def _resolve(root: Path, value: Any, name: str, *, optional: bool = False) -> Path | None:
    if value is None and optional:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"contract common_args.{name} must be a nonempty path")
    path = Path(value)
    return (path if path.is_absolute() else root / path).resolve()


def _worker_definition(contract: Mapping[str, Any], worker_id: str) -> dict[str, Any]:
    workers = contract.get("workers")
    if not isinstance(workers, list):
        raise ValueError("parallel contract workers must be a list")
    matches = [row for row in workers if isinstance(row, Mapping) and row.get("worker_id") == worker_id]
    if len(matches) != 1:
        raise ValueError(f"parallel contract must define worker exactly once: {worker_id}")
    worker = dict(matches[0])
    if type(worker.get("physical_device")) is not int or worker["physical_device"] not in range(7):
        raise ValueError(f"worker {worker_id} has an invalid physical_device")
    if type(worker.get("port")) is not int or not 0 < worker["port"] < 65536:
        raise ValueError(f"worker {worker_id} has an invalid port")
    task_ids = worker.get("task_ids")
    if (not isinstance(task_ids, list)
            or any(not isinstance(task_id, str) or not task_id for task_id in task_ids)
            or len(task_ids) != len(set(task_ids))):
        raise ValueError(f"worker {worker_id} task_ids must be unique nonempty strings")
    return worker


def _validate_dynamic_task_universe(
    root: Path,
    contract: Mapping[str, Any],
    frozen_tasks: set[str],
    manifest_task_ids: Sequence[str],
) -> None:
    """Bind either the full manifest or an explicit exact-recovery subset."""

    recovery_path = contract.get("recovery_contract_path")
    if recovery_path is None:
        if frozen_tasks != set(manifest_task_ids):
            raise ValueError("dynamic source queue must cover the frozen task manifest exactly")
        return
    if not isinstance(recovery_path, str) or not recovery_path:
        raise ValueError("recovery_contract_path must be a nonempty relative path")
    path = (root / recovery_path).resolve()
    if Path(recovery_path).is_absolute() or not path.is_relative_to(root):
        raise ValueError("recovery contract must stay inside the frozen package")
    recovery = _read_json(path)
    if recovery.get("schema") != "t02-exact-recovery-contract-v1":
        raise ValueError("recovery contract schema is invalid")
    source_tasks = recovery.get("source_tasks")
    if not isinstance(source_tasks, list):
        raise ValueError("recovery contract source_tasks must be a list")
    expected = {
        row.get("task_id") for row in source_tasks if isinstance(row, Mapping)
    }
    if (len(expected) != len(source_tasks)
            or any(not isinstance(task_id, str) or not task_id for task_id in expected)):
        raise ValueError("recovery source task IDs must be unique nonempty strings")
    if not expected.issubset(manifest_task_ids):
        raise ValueError("recovery source tasks are outside the original frozen manifest")
    if frozen_tasks != expected:
        raise ValueError("dynamic source queue differs from the exact recovery task subset")


def _load_contract(root: Path, worker_id: str, backend_url: str) -> tuple[dict[str, Any], dict[str, Any]]:
    contract = _read_json(root / "contract.json")
    if not isinstance(contract, Mapping) or contract.get("schema") != CONTRACT_SCHEMA:
        raise ValueError(f"parallel contract schema must be {CONTRACT_SCHEMA}")
    common = contract.get("common_args")
    policy = contract.get("frozen_policy")
    if not isinstance(common, Mapping) or not isinstance(policy, Mapping) or not policy:
        raise ValueError("parallel contract requires common_args and frozen_policy")
    worker = _worker_definition(contract, worker_id)
    parsed = urlparse(backend_url)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost"}:
        raise ValueError("parallel worker backend must be a loopback HTTP endpoint")
    if parsed.port != worker["port"]:
        raise ValueError("parallel worker backend port differs from its frozen worker definition")
    if common.get("seed") != 0:
        raise ValueError("parallel T02 requires the frozen seed=0 contract")
    if common.get("max_states_per_task") != 2:
        raise ValueError("parallel T02 requires max_states_per_task=2")
    return dict(contract), worker


def _verify_lineage(root: Path, contract: Mapping[str, Any]) -> tuple[
        dict[str, Path | None], list[str], dict[str, str]]:
    common = contract["common_args"]
    paths: dict[str, Path | None] = {
        name: _resolve(root, common.get(name), name, optional=(name == "bfcl_dependency_path"))
        for name in (
            "bfcl_root", "bfcl_dependency_path", "task_manifest", "family_audit",
            "d128_manifest", "f128_manifest", "design", "checkpoint",
        )
    }
    task_ids, family_bindings, family_receipt = t02_bfcl.load_family_bindings(
        paths["task_manifest"], paths["family_audit"]
    )
    official_receipt = t02_bfcl.verify_official_source_files(
        paths["bfcl_root"], paths["task_manifest"]
    )
    policy = contract["frozen_policy"]
    if policy.get("family_bindings") != family_receipt:
        raise ValueError("frozen policy family bindings differ from verified inputs")
    if policy.get("official_source_files") != official_receipt:
        raise ValueError("frozen policy official source receipt differs from verified inputs")
    design_sha256 = hashlib.sha256(paths["design"].read_bytes()).hexdigest()
    if policy.get("design_sha256") != design_sha256:
        raise ValueError("frozen policy design digest differs from the worker design")
    if policy.get("sampling") != {"mode": "greedy", "temperature": 0, "seed": 0}:
        raise ValueError("frozen policy sampling contract changed")
    return paths, task_ids, family_bindings


def _write_status(path: Path, phase: str, **values: Any) -> None:
    _atomic_write_json(path, {
        "schema": STATUS_SCHEMA,
        "phase": phase,
        "updated_at_epoch": time.time(),
        **values,
    })


def _abort_error(root: Path) -> RuntimeError | None:
    path = root / "abort.json"
    if not path.exists():
        return None
    try:
        value = _read_json(path)
    except (OSError, json.JSONDecodeError) as error:
        return RuntimeError(f"parallel abort marker is unreadable: {error}")
    reason = value.get("reason") if isinstance(value, Mapping) else None
    return RuntimeError(f"parallel coordinator aborted: {reason or 'unspecified'}")


def _wait_for_global_plan(root: Path, status_path: Path, worker_id: str) -> dict[str, Any]:
    plan_path = root / "global_plan.json"
    while True:
        aborted = _abort_error(root)
        if aborted is not None:
            raise aborted
        if plan_path.exists():
            try:
                value = _read_json(plan_path)
            except (OSError, json.JSONDecodeError):
                time.sleep(0.25)
                continue
            if not isinstance(value, dict):
                raise ValueError("global_plan.json must contain an object")
            return value
        _write_status(status_path, "waiting_for_global_plan", worker_id=worker_id)
        time.sleep(1.0)


def _wait_for_pilot(root: Path, plan: Mapping[str, Any], status_path: Path,
                    worker_id: str) -> None:
    marker_path = root / "pilot_passed.json"
    pilot_id = plan["states"][0]["state_id"]
    plan_sha256 = t02._digest(plan)
    while True:
        aborted = _abort_error(root)
        if aborted is not None:
            raise aborted
        if marker_path.exists():
            try:
                marker = _read_json(marker_path)
            except (OSError, json.JSONDecodeError):
                time.sleep(0.25)
                continue
            if (not isinstance(marker, Mapping)
                    or marker.get("schema") != "t02-parallel-pilot-passed-v1"
                    or marker.get("plan_sha256") != plan_sha256
                    or marker.get("pilot_state_id") != pilot_id
                    or marker.get("complete_branch_executions") != len(t02.BRANCH_IDS)):
                raise ValueError("pilot marker differs from the global plan or branch contract")
            return
        _write_status(status_path, "waiting_for_pilot", worker_id=worker_id,
                      pilot_state_id=pilot_id)
        time.sleep(1.0)


def _merge_partial_results(plan: Mapping[str, Any],
                           partials: Sequence[Mapping[str, Any]], *,
                           require_complete_states: bool = False) -> dict[str, Any]:
    """Merge disjoint calls made by one worker, preserving global plan order."""

    plan_sha256 = t02._digest(plan)
    capabilities = None
    snapshots: list[Any] = []
    results: list[Any] = []
    state_ids: set[str] = set()
    snapshot_ids: set[str] = set()
    result_keys: set[tuple[str, str]] = set()
    for partial in partials:
        if partial.get("plan_sha256") != plan_sha256 or partial.get("partial_worker_result") is not True:
            raise ValueError("worker partial result differs from the global plan contract")
        current = partial.get("adapter_capabilities")
        if capabilities is None:
            capabilities = current
        elif current != capabilities:
            raise ValueError("worker adapter capabilities changed between pilot and remainder")
        current_ids = partial.get("executed_state_ids")
        if (not isinstance(current_ids, list) or len(current_ids) != len(set(current_ids))
                or state_ids.intersection(current_ids)):
            raise ValueError("worker partial state IDs repeat")
        current_id_set = set(current_ids)
        if partial.get("complete_branch_executions") != len(partial.get("results", [])):
            raise ValueError("worker partial branch count differs from its results")
        state_ids.update(current_ids)
        for snapshot in partial.get("snapshots", []):
            state_id = snapshot.get("state_id")
            if state_id not in current_id_set or state_id in snapshot_ids:
                raise ValueError("worker partial snapshots repeat")
            snapshot_ids.add(state_id)
            snapshots.append(snapshot)
        for result in partial.get("results", []):
            key = (result.get("state_id"), result.get("branch_id"))
            if result.get("state_id") not in current_id_set or key in result_keys:
                raise ValueError("worker partial branches repeat")
            result_keys.add(key)
            results.append(result)
    ordered_ids = [state["state_id"] for state in plan["states"] if state["state_id"] in state_ids]
    if snapshot_ids != state_ids:
        raise ValueError("worker partials do not contain one snapshot per started state")
    if len(results) > len(t02.BRANCH_IDS) * len(state_ids):
        raise ValueError("worker partials exceed three branches per state")
    if require_complete_states and len(results) != len(t02.BRANCH_IDS) * len(state_ids):
        raise ValueError("worker final result does not contain three branches per state")
    return {
        "schema": t02.RESULT_SET_SCHEMA,
        "plan_sha256": plan_sha256,
        "adapter_capabilities": capabilities,
        "complete_branch_executions": len(results),
        "branch_cap": plan["max_complete_branch_executions"],
        "authorized_complete_branch_cap": plan.get(
            "authorized_complete_branch_cap", plan["max_complete_branch_executions"]),
        "snapshots": snapshots,
        "results": results,
        "executed_state_ids": ordered_ids,
        "partial_worker_result": True,
    }


def run_worker(root: str | Path, worker_id: str, backend_url: str) -> int:
    """Collect leased sources, retain live snapshots, and execute selected states."""

    import t02_parallel
    from t02_runtime import build_actor

    root = Path(root).resolve()
    contract, worker = _load_contract(root, worker_id, backend_url)
    paths, manifest_task_ids, family_bindings = _verify_lineage(root, contract)
    dynamic = t02_parallel.dynamic_source_queue(contract)
    if dynamic:
        t02_parallel.dynamic_source_limits(contract)
    assigned = set(worker["task_ids"])
    if not assigned.issubset(manifest_task_ids):
        raise ValueError(f"worker {worker_id} contains a task outside the frozen manifest")
    other_tasks = {
        task_id
        for row in contract["workers"]
        if row.get("worker_id") != worker_id
        for task_id in row.get("task_ids", [])
    }
    if assigned.intersection(other_tasks):
        raise ValueError("parallel worker task shards overlap")
    frozen_tasks = assigned | other_tasks
    if dynamic:
        _validate_dynamic_task_universe(root, contract, frozen_tasks, manifest_task_ids)

    worker_root = root / "workers" / worker_id
    worker_root.mkdir(parents=True, exist_ok=True)
    status_path = worker_root / "status.json"
    if (worker_root / "results.json").exists() or (worker_root / "ready.json").exists():
        raise ValueError(f"worker output already exists: {worker_root}")

    dependency = paths["bfcl_dependency_path"]
    old_cwd = Path.cwd()
    adapter = None
    owned_state_ids: set[str] = set()
    claimed_task_ids: set[str] = set()
    source_task_terminals: dict[str, dict[str, Any]] = {}
    failure_scope = "source_collection"
    global_abort_required = True
    try:
        os.environ["BFCL_PROJECT_ROOT"] = str((worker_root / "bfcl_state").resolve())
        bfcl_root = paths["bfcl_root"]
        if str(bfcl_root) not in sys.path:
            sys.path.insert(0, str(bfcl_root))
        if dependency is not None and str(dependency) not in sys.path:
            sys.path.append(str(dependency))
        os.chdir(bfcl_root)
        bindings = t02_bfcl.OfficialBFCLBindings()
        adapter = t02_bfcl.ExactBFCLBranchAdapter(
            frozen_policy=contract["frozen_policy"],
            artifact_dir=worker_root / "official_outcomes",
            family_bindings=family_bindings,
        )
        shared_models: list[Any | None] = [None]

        def actor_factory(task_id: str, task_index: int):
            actor = build_actor(
                design_path=paths["design"], checkpoint=paths["checkpoint"],
                backend_url=backend_url,
                output_dir=worker_root / "actors" / f"{task_index:03d}-{t02_bfcl._safe_name(task_id)}",
            )
            controller = actor.runner.controller
            if shared_models[0] is None:
                shared_models[0] = controller.backends
            else:
                while controller is not None:
                    if hasattr(controller, "backends"):
                        controller.backends = shared_models[0]
                    controller = getattr(controller, "base", None)
            return actor

        while True:
            aborted = _abort_error(root)
            if aborted is not None:
                raise aborted
            claim = t02_parallel.claim_task(root, worker_id)
            if claim is None:
                break
            if not isinstance(claim, Mapping):
                raise TypeError("claim_task must return an object or None")
            task_id, task_index = claim.get("task_id"), claim.get("task_index")
            allowed_tasks = frozen_tasks if dynamic else assigned
            if task_id not in allowed_tasks or task_id in claimed_task_ids:
                raise ValueError(f"coordinator returned an invalid or repeated task claim: {task_id}")
            if type(task_index) is not int or task_index < 0:
                raise ValueError("coordinator returned an invalid global task_index")
            claimed_task_ids.add(task_id)
            _write_status(status_path, "source_task_started", worker_id=worker_id,
                          task_id=task_id, task_index=task_index)
            task, ground_truth = bindings.load_task(task_id)
            environment = t02_bfcl.BFCLTaskEnvironment(task, ground_truth, bindings=bindings)
            actor = actor_factory(task_id, task_index)
            candidates = adapter.discover_task(
                environment, actor, seed=contract["common_args"]["seed"],
                max_states=contract["common_args"]["max_states_per_task"],
            )
            source_terminal = adapter.source_task_terminal(task_id)
            if source_terminal is not None:
                source_task_terminals[task_id] = source_terminal
            new_ids = {row["state_id"] for row in candidates}
            if owned_state_ids.intersection(new_ids):
                raise ValueError("parallel worker discovered a duplicate state_id")
            owned_state_ids.update(new_ids)
            t02_parallel.complete_task(
                root, worker_id, task_id, candidates, adapter.cost_summary(),
                source_terminal=source_terminal,
            )
            _write_status(status_path, "source_task_completed", worker_id=worker_id,
                          task_id=task_id, task_index=task_index,
                           candidate_count=len(owned_state_ids),
                           generation_calls=adapter.cost_summary(),
                           source_terminal=source_terminal)

        if not dynamic and claimed_task_ids != assigned:
            missing = sorted(assigned - claimed_task_ids)
            raise RuntimeError(f"worker queue ended before its frozen shard completed: {missing}")
        ready = {
            "schema": READY_SCHEMA,
            "worker_id": worker_id,
            "claimed_task_ids": sorted(claimed_task_ids),
            "candidate_state_ids": sorted(owned_state_ids),
            "candidate_count": len(owned_state_ids),
            "generation_calls": adapter.cost_summary(),
            "source_task_terminals": source_task_terminals,
        }
        _atomic_write_json(worker_root / "ready.json", ready)
        if dynamic and not claimed_task_ids:
            _write_status(
                status_path,
                "completed_no_work",
                worker_id=worker_id,
                claimed_task_count=0,
                candidate_count=0,
            )
            return 0
        plan = _wait_for_global_plan(root, status_path, worker_id)
        failure_scope = "global_contract"
        t02.check_plan(plan)
        if plan.get("frozen_subsequent_policy") != contract["frozen_policy"]:
            raise ValueError("global plan changed the frozen continuation policy")
        planned_ids = [state["state_id"] for state in plan["states"]]
        own_ids = [state_id for state_id in planned_ids if state_id in owned_state_ids]
        adapter.prune(own_ids)
        pilot_id = plan["states"][0]["state_id"]
        is_pilot_owner = pilot_id in own_ids
        completed_parts: list[Mapping[str, Any]] = []

        def write_cumulative(partial: Mapping[str, Any]) -> None:
            combined = _merge_partial_results(plan, [*completed_parts, partial])
            _atomic_write_json(worker_root / "partial_results.json", combined)

        if is_pilot_owner:
            failure_scope = "pilot_gate"
            _write_status(status_path, "executing_pilot", worker_id=worker_id,
                          pilot_state_id=pilot_id)
            pilot = t02.execute_plan(
                plan, adapter, state_ids=[pilot_id], on_progress=write_cumulative
            )
            if (pilot.get("executed_state_ids") != [pilot_id]
                    or pilot.get("complete_branch_executions") != len(t02.BRANCH_IDS)):
                raise RuntimeError("pilot did not complete exactly three official branches")
            completed_parts.append(pilot)
            _atomic_write_json(root / "pilot_passed.json", {
                "schema": "t02-parallel-pilot-passed-v1",
                "plan_sha256": t02._digest(plan),
                "pilot_state_id": pilot_id,
                "worker_id": worker_id,
                "complete_branch_executions": len(t02.BRANCH_IDS),
                "result_sha256": t02._digest(pilot),
            })
        else:
            failure_scope = "pilot_gate"
            _wait_for_pilot(root, plan, status_path, worker_id)

        remaining_ids = [state_id for state_id in own_ids if state_id != pilot_id]
        _write_status(status_path, "executing", worker_id=worker_id,
                       selected_state_ids=own_ids, selected_state_count=len(own_ids),
                       pilot_state_id=pilot_id, pilot_owner=is_pilot_owner)
        failure_scope = "branch_execution"
        global_abort_required = False
        remainder = t02.execute_plan(
            plan, adapter, state_ids=remaining_ids, on_progress=write_cumulative
        )
        failure_scope = "finalization"
        global_abort_required = True
        completed_parts.append(remainder)
        results = _merge_partial_results(
            plan, completed_parts, require_complete_states=True
        )
        _atomic_write_json(worker_root / "partial_results.json", results)
        _atomic_write_json(worker_root / "results.json", results)
        _write_status(status_path, "completed", worker_id=worker_id,
                      selected_state_count=len(own_ids),
                      complete_branch_executions=results["complete_branch_executions"])
        return 0
    except BaseException as error:
        failure = {
            "schema": STATUS_SCHEMA,
            "phase": "failed",
            "worker_id": worker_id,
            "error_type": type(error).__name__,
            "error": str(error),
            "candidate_state_ids": sorted(owned_state_ids),
            "failure_scope": failure_scope,
            "global_abort_required": global_abort_required,
            "updated_at_epoch": time.time(),
        }
        if adapter is not None:
            failure["generation_calls"] = adapter.cost_summary()
        _atomic_write_json(status_path, failure)
        if global_abort_required and not (root / "abort.json").exists():
            _atomic_write_json(root / "abort.json", {
                "schema": "t02-parallel-abort-v1",
                "phase": "worker",
                "worker_id": worker_id,
                "error_type": type(error).__name__,
                "error": str(error),
                "updated_at_epoch": time.time(),
            })
        raise
    finally:
        if adapter is not None:
            adapter.close()
        os.chdir(old_cwd)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--backend-url", required=True)
    args = parser.parse_args(argv)
    return run_worker(args.root, args.worker_id, args.backend_url)


if __name__ == "__main__":
    raise SystemExit(main())
