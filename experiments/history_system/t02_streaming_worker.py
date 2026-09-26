"""Run one bounded T02 streaming-recovery worker on an existing local engine."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Mapping, Sequence

import t02
import t02_bfcl
import t02_streaming_recovery as streaming
from t02_bfcl import validate_training_feature_contract
from t02_parallel_worker import (
    _atomic_write_json,
    _load_contract,
    _verify_lineage,
    _write_status,
)


WORKER_DONE_SCHEMA = "t02-streaming-worker-done-v1"


def _file_sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _partial_count(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return 0
    count = value.get("complete_branch_executions") if isinstance(value, Mapping) else None
    return count if type(count) is int and count >= 0 else 0


def _backend_healthy(backend_url: str) -> bool:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(backend_url.rstrip("/") + "/health", timeout=3) as response:
            return response.status == 200
    except OSError:
        return False


def _backend_fatal(error: BaseException, backend_url: str) -> bool:
    kind = f"{type(error).__module__}.{type(error).__name__}"
    return "HttpTransport" in kind or not _backend_healthy(backend_url)


def run_worker(root: str | Path, worker_id: str, backend_url: str) -> int:
    """Collect and fully branch one source at a time; never launch an engine."""

    from t02_runtime import build_actor

    root = Path(root).resolve()
    contract, _worker = _load_contract(root, worker_id, backend_url)
    paths, manifest_task_ids, family_bindings = _verify_lineage(root, contract)
    recovery = json.loads(
        (root / "recovery" / "recovery_contract.json").read_text(encoding="utf-8")
    )
    source_plan = json.loads(
        (root / "recovery" / "source_plan.json").read_text(encoding="utf-8")
    )
    streaming.validate_recovery_contract(recovery)
    frozen_tasks = {row["task_id"] for row in recovery["source_tasks"]}
    if not frozen_tasks.issubset(manifest_task_ids):
        raise ValueError("streaming recovery contains a task outside the frozen BFCL manifest")

    worker_root = root / "workers" / worker_id
    worker_root.mkdir(parents=True, exist_ok=True)
    status_path = worker_root / "status.json"
    done_path = worker_root / "streaming_done.json"
    if done_path.exists():
        raise ValueError(f"streaming worker output already exists: {done_path}")

    old_cwd = Path.cwd()
    dependency = paths["bfcl_dependency_path"]
    claimed: list[str] = []
    completed: list[str] = []
    failed: list[str] = []
    stop_for_pilot = False
    stop_for_backend = False
    try:
        os.environ["BFCL_PROJECT_ROOT"] = str((worker_root / "bfcl_state").resolve())
        bfcl_root = paths["bfcl_root"]
        if str(bfcl_root) not in sys.path:
            sys.path.insert(0, str(bfcl_root))
        if dependency is not None and str(dependency) not in sys.path:
            sys.path.append(str(dependency))
        os.chdir(bfcl_root)
        bindings = t02_bfcl.OfficialBFCLBindings()
        shared_models: list[Any | None] = [None]

        def actor_factory(task_id: str, task_index: int, task_root: Path):
            actor = build_actor(
                design_path=paths["design"],
                checkpoint=paths["checkpoint"],
                backend_url=backend_url,
                output_dir=task_root / "actor",
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

        while not stop_for_pilot and not stop_for_backend:
            if not _backend_healthy(backend_url):
                stop_for_backend = True
                break
            claim = streaming.claim_task(root, worker_id)
            if claim is None:
                break
            task_id = claim["task_id"]
            task_index = claim["task_index"]
            claimed.append(task_id)
            task_root = streaming.task_artifact_dir(worker_root, claim)
            task_root.mkdir(parents=True, exist_ok=False)
            task_status_path = task_root / "status.json"
            plan_path = task_root / "plan.json"
            partial_path = task_root / "partial_results.json"
            result_path = task_root / "results.json"
            label_path = task_root / "labels.json"
            adapter = None
            plan: dict[str, Any] | None = None
            _atomic_write_json(task_status_path, {
                "schema": streaming.TASK_STATUS_SCHEMA,
                "phase": "source_collection",
                "worker_id": worker_id,
                "task_id": task_id,
                "task_index": task_index,
                "updated_at_epoch": time.time(),
            })
            _write_status(
                status_path,
                "source_task_started",
                worker_id=worker_id,
                task_id=task_id,
                task_index=task_index,
                completed_task_count=len(completed),
                failed_task_count=len(failed),
            )
            try:
                adapter = t02_bfcl.ExactBFCLBranchAdapter(
                    frozen_policy=contract["frozen_policy"],
                    artifact_dir=task_root / "official_outcomes",
                    family_bindings=family_bindings,
                )
                task, ground_truth = bindings.load_task(task_id)
                environment = t02_bfcl.BFCLTaskEnvironment(
                    task, ground_truth, bindings=bindings
                )
                actor = actor_factory(task_id, task_index, task_root)
                candidates = adapter.discover_task(
                    environment,
                    actor,
                    seed=contract["common_args"]["seed"],
                    max_states=contract["common_args"]["max_states_per_task"],
                )
                plan = streaming.build_task_plan(
                    recovery,
                    task_id,
                    candidates,
                    source_plan=source_plan,
                    frozen_subsequent_policy=contract["frozen_policy"],
                )
                plan["training_feature_contract"] = validate_training_feature_contract(
                    plan["states"]
                )
                # The feature receipt is metadata; validate the execution plan again
                # after attaching it so its final digest is the one used everywhere.
                streaming.validate_task_plan(plan, recovery=recovery)
                plan_file_sha256 = _atomic_write_json(plan_path, plan)
                selected_ids = [state["state_id"] for state in plan["states"]]
                adapter.prune(selected_ids)

                parts: list[Mapping[str, Any]] = []

                def persist_progress(partial: Mapping[str, Any]) -> None:
                    merged = streaming.merge_task_results(
                        plan, [*parts, partial], require_complete=False
                    )
                    _atomic_write_json(partial_path, merged)

                pilot_task = task_id == recovery["source_tasks"][0]["task_id"]
                if pilot_task:
                    first_id = plan["states"][0]["state_id"]
                    pilot = streaming.execute_task_plan(
                        plan,
                        adapter,
                        state_ids=[first_id],
                        on_progress=persist_progress,
                    )
                    if pilot["complete_branch_executions"] != 3:
                        raise RuntimeError("deterministic pilot did not complete three branches")
                    parts.append(pilot)
                    streaming.write_pilot_passed(root, worker_id, plan, pilot)
                    remaining_ids = selected_ids[1:]
                else:
                    _atomic_write_json(task_status_path, {
                        "schema": streaming.TASK_STATUS_SCHEMA,
                        "phase": "waiting_for_pilot",
                        "worker_id": worker_id,
                        "task_id": task_id,
                        "task_index": task_index,
                        "plan_sha256": streaming._digest(plan),
                        "updated_at_epoch": time.time(),
                    })
                    streaming.wait_for_pilot(root)
                    remaining_ids = selected_ids

                _atomic_write_json(task_status_path, {
                    "schema": streaming.TASK_STATUS_SCHEMA,
                    "phase": "branch_execution",
                    "worker_id": worker_id,
                    "task_id": task_id,
                    "task_index": task_index,
                    "plan_sha256": streaming._digest(plan),
                    "updated_at_epoch": time.time(),
                })
                if remaining_ids:
                    remainder = streaming.execute_task_plan(
                        plan,
                        adapter,
                        state_ids=remaining_ids,
                        on_progress=persist_progress,
                    )
                    parts.append(remainder)
                results = streaming.merge_task_results(
                    plan, parts, require_complete=True
                )
                result_file_sha256 = _atomic_write_json(result_path, results)
                _atomic_write_json(partial_path, results)
                labels = streaming.label_task_results(plan, results)
                label_file_sha256 = _atomic_write_json(label_path, labels)
                streaming.complete_task(
                    root,
                    worker_id,
                    task_id,
                    plan_path=plan_path,
                    plan_sha256=streaming._digest(plan),
                    plan_file_sha256=plan_file_sha256,
                    result_path=result_path,
                    result_file_sha256=result_file_sha256,
                    label_path=label_path,
                    label_file_sha256=label_file_sha256,
                    complete_branch_executions=results["complete_branch_executions"],
                )
                completed.append(task_id)
                _atomic_write_json(task_status_path, {
                    "schema": streaming.TASK_STATUS_SCHEMA,
                    "phase": "completed",
                    "worker_id": worker_id,
                    "task_id": task_id,
                    "task_index": task_index,
                    "plan_sha256": streaming._digest(plan),
                    "complete_branch_executions": results["complete_branch_executions"],
                    "updated_at_epoch": time.time(),
                })
            except BaseException as error:
                observed = _partial_count(partial_path)
                streaming.fail_task(
                    root,
                    worker_id,
                    task_id,
                    error=error,
                    complete_branch_executions=observed,
                    plan_sha256=streaming._digest(plan) if plan is not None else None,
                    partial_result_path=partial_path if partial_path.exists() else None,
                )
                failed.append(task_id)
                _atomic_write_json(task_status_path, {
                    "schema": streaming.TASK_STATUS_SCHEMA,
                    "phase": "failed_no_retry",
                    "worker_id": worker_id,
                    "task_id": task_id,
                    "task_index": task_index,
                    "plan_sha256": streaming._digest(plan) if plan is not None else None,
                    "complete_branch_executions": observed,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "automatic_retry": False,
                    "updated_at_epoch": time.time(),
                })
                pilot_task = task_id == recovery["source_tasks"][0]["task_id"]
                if pilot_task and not (root / "pilot_passed.json").exists():
                    streaming.mark_pilot_failed(root, worker_id, task_id, error)
                    stop_for_pilot = True
                elif isinstance(error, streaming.PilotGateFailed):
                    stop_for_pilot = True
                elif _backend_fatal(error, backend_url):
                    stop_for_backend = True
            finally:
                if adapter is not None:
                    adapter.close()

            _write_status(
                status_path,
                "streaming",
                worker_id=worker_id,
                last_task_id=task_id,
                completed_task_count=len(completed),
                failed_task_count=len(failed),
                stop_for_pilot_failure=stop_for_pilot,
                stop_for_backend_failure=stop_for_backend,
            )

        done = {
            "schema": WORKER_DONE_SCHEMA,
            "worker_id": worker_id,
            "claimed_task_ids": claimed,
            "completed_task_ids": completed,
            "failed_task_ids": failed,
            "stopped_for_pilot_failure": stop_for_pilot,
            "stopped_for_backend_failure": stop_for_backend,
            "updated_at_epoch": time.time(),
        }
        _atomic_write_json(done_path, done)
        _write_status(
            status_path,
            "completed_with_failures" if failed else "completed",
            worker_id=worker_id,
            claimed_task_count=len(claimed),
            completed_task_count=len(completed),
            failed_task_count=len(failed),
            stopped_for_pilot_failure=stop_for_pilot,
            stopped_for_backend_failure=stop_for_backend,
        )
        return 1 if failed or stop_for_backend else 0
    finally:
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
