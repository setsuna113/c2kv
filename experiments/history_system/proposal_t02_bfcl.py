"""Run the H0-only proposal-matched T02 lifecycle on live official BFCL state.

The default source-start budget is zero.  A real collection therefore requires
both explicit resource coordination approval and an explicit positive source
budget; this module never treats old serialized labels as restorable actor/KV
snapshots.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import proposal_t02
import t02
import t02_bfcl


SUMMARY_SCHEMA = "proposal-t02-bfcl-run-h0-v1"


def collect_live_proposal_plan(*, task_ids: Sequence[str],
                               actor_factory: Callable[[str, int], Any],
                               bindings: Any,
                               forbidden_manifest_paths: Mapping[str, str | Path],
                               frozen_policy: Mapping[str, Any], artifact_dir: str | Path,
                               seed: int, target_states: int, train_states: int,
                               candidate_snapshot_cap: int,
                               max_states_per_group: int, max_states_per_task: int,
                               max_source_task_starts: int,
                               family_bindings: Mapping[str, str],
                               split_bindings: Mapping[str, str],
                               on_progress: Callable[[Mapping[str, Any]], None] | None = None,
                               on_checkpoint: Callable[[Mapping[str, Any]], None] | None = None,
                               adapter: Any | None = None):
    """Acquire bounded process-local snapshots and build one executable plan."""
    if not 0 < candidate_snapshot_cap <= proposal_t02.MAX_NEW_STATES:
        raise ValueError("candidate_snapshot_cap must be in [1, 60]")
    if adapter is None:
        adapter = t02_bfcl.ExactBFCLBranchAdapter(
            frozen_policy=frozen_policy, artifact_dir=artifact_dir,
            family_bindings=family_bindings,
        )
    forbidden, _ = t02.load_forbidden_manifests(forbidden_manifest_paths)
    candidates = []
    source_task_terminals: dict[str, Any] = {}
    group_counts: dict[str, int] = {}
    ordered = t02_bfcl.round_robin_family_tasks(task_ids, family_bindings)
    exhaustion_reason = "source_task_manifest_exhausted"
    last_task_id = None

    def checkpoint(status: str) -> None:
        if on_checkpoint is None:
            return
        on_checkpoint({
            "schema": "proposal-t02-source-checkpoint-h0-v1",
            "status": status,
            "candidate_state_count": len(candidates),
            "candidates": copy.deepcopy(candidates),
            "source_task_terminals": copy.deepcopy(source_task_terminals),
            "source_task_budget": {
                "authorized": max_source_task_starts,
                "actual": adapter.source_collection_tasks,
            },
            "generation_calls": adapter.cost_summary(),
            "last_task_id": last_task_id,
        })

    try:
        checkpoint("in_progress")
        for task_index, task_id in enumerate(ordered):
            if len(candidates) >= candidate_snapshot_cap:
                exhaustion_reason = "candidate_snapshot_cap_exhausted"
                break
            group = family_bindings[task_id]
            if task_id in forbidden or group in forbidden:
                continue
            remaining_group = max_states_per_group - group_counts.get(group, 0)
            if remaining_group <= 0:
                continue
            if adapter.source_collection_tasks >= max_source_task_starts:
                exhaustion_reason = "source_task_start_cap_exhausted"
                break
            task, ground_truth = bindings.load_task(task_id)
            environment = t02_bfcl.BFCLTaskEnvironment(task, ground_truth, bindings=bindings)
            actor = actor_factory(task_id, task_index)
            last_task_id = task_id
            if on_progress:
                on_progress({"phase": "source_task_started", "task_id": task_id,
                             "source_task_starts": adapter.source_collection_tasks + 1,
                             "candidate_count": len(candidates)})
            discovered = adapter.discover_task(
                environment, actor, seed=seed,
                max_states=min(max_states_per_task, remaining_group,
                               candidate_snapshot_cap - len(candidates)),
            )
            candidates.extend(discovered)
            terminal = adapter.source_task_terminal(task_id)
            if terminal is not None:
                source_task_terminals[task_id] = terminal
            group_counts[group] = group_counts.get(group, 0) + len(discovered)
            checkpoint("in_progress")
            if on_progress:
                on_progress({"phase": "source_task_completed", "task_id": task_id,
                             "source_task_starts": adapter.source_collection_tasks,
                             "candidate_count": len(candidates),
                             "generation_calls": adapter.cost_summary()})
            if len(candidates) < target_states:
                continue
            try:
                plan = proposal_t02.build_plan(
                    candidates,
                    forbidden_manifest_paths=forbidden_manifest_paths,
                    frozen_subsequent_policy=frozen_policy,
                    seed=seed, target_states=target_states, train_states=train_states,
                    split_bindings=split_bindings,
                )
            except proposal_t02.ProposalT02Error:
                continue
            plan["training_feature_contract"] = t02_bfcl.validate_training_feature_contract(
                plan["states"]
            )
            adapter.prune([row["state_id"] for row in plan["states"]])
            checkpoint("completed")
            return plan, candidates, adapter
        plan = proposal_t02.build_plan(
            candidates,
            forbidden_manifest_paths=forbidden_manifest_paths,
            frozen_subsequent_policy=frozen_policy,
            seed=seed, target_states=target_states, train_states=train_states,
            split_bindings=split_bindings,
            allow_partial=True,
            incomplete_collection_reason=exhaustion_reason,
        )
        plan["training_feature_contract"] = t02_bfcl.validate_training_feature_contract(
            plan["states"]
        )
        adapter.prune([row["state_id"] for row in plan["states"]])
        checkpoint("completed")
        return plan, candidates, adapter
    except BaseException:
        checkpoint("failed")
        adapter.close()
        raise


def _split_bindings(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, Mapping) and "group_split_bindings" in value:
        value = value["group_split_bindings"]
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) or split not in {"train", "calibration"}
        for key, split in value.items()
    ):
        raise ValueError("split bindings must map canonical groups to train/calibration")
    return dict(value)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bfcl-root", type=Path, required=True)
    parser.add_argument("--bfcl-dependency-path", type=Path)
    parser.add_argument("--task-manifest", type=Path, required=True)
    parser.add_argument("--family-audit", type=Path, required=True)
    parser.add_argument("--d128-manifest", type=Path, required=True)
    parser.add_argument("--f128-manifest", type=Path, required=True)
    parser.add_argument("--design", type=Path, required=True)
    continuation = parser.add_mutually_exclusive_group(required=True)
    continuation.add_argument("--continuation-policy", type=Path)
    continuation.add_argument("--source-plan", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--backend-url", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--split-bindings", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target-states", type=int, default=proposal_t02.MAX_NEW_STATES)
    parser.add_argument("--train-states", type=int, default=40)
    parser.add_argument("--candidate-snapshot-cap", type=int, default=proposal_t02.MAX_NEW_STATES)
    parser.add_argument("--backend-snapshot-cap", type=int, default=proposal_t02.MAX_NEW_STATES)
    parser.add_argument("--max-states-per-group", type=int, default=8)
    parser.add_argument("--max-states-per-task", type=int, default=2)
    parser.add_argument("--max-source-task-starts", type=int, default=0)
    parser.add_argument("--resource-coordination-approved", action="store_true")
    args = parser.parse_args(argv)

    paths = (
        "bfcl_root", "task_manifest", "family_audit", "d128_manifest", "f128_manifest",
        "design", "checkpoint", "out", "split_bindings", "bfcl_dependency_path",
        "continuation_policy", "source_plan",
    )
    for name in paths:
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.resolve())
    if not args.resource_coordination_approved:
        raise ValueError("live proposal T02 requires explicit resource coordination approval")
    if args.max_source_task_starts <= 0:
        raise ValueError(
            "max-source-task-starts defaults to zero; authorize a positive bounded source acquisition budget"
        )
    if args.max_source_task_starts > 104:
        raise ValueError("one proposal collection may start at most 104 source tasks")
    if not 0 < args.target_states <= proposal_t02.MAX_NEW_STATES:
        raise ValueError("target-states must be in [1, 60]")
    if not 0 <= args.train_states <= args.target_states:
        raise ValueError("train-states must be between zero and target-states")
    if not 0 < args.candidate_snapshot_cap <= proposal_t02.MAX_NEW_STATES:
        raise ValueError("candidate snapshot cap must be in [1, 60]")
    if args.candidate_snapshot_cap > args.backend_snapshot_cap:
        raise ValueError("candidate snapshot cap must be within backend cap")
    if args.target_states > args.candidate_snapshot_cap:
        raise ValueError("target states exceed the candidate snapshot cap")
    if not 0 < args.max_states_per_group <= 8 or not 0 < args.max_states_per_task <= 2:
        raise ValueError("state caps are at most 8 per canonical group and 2 per variant")
    if args.out.exists():
        raise ValueError(f"output directory already exists: {args.out}")
    if not (args.bfcl_root / "bfcl_eval").is_dir():
        raise ValueError("bfcl-root does not contain the official bfcl_eval package")

    task_ids, family_bindings, family_receipt = t02_bfcl.load_family_bindings(
        args.task_manifest, args.family_audit
    )
    official_receipt = t02_bfcl.verify_official_source_files(
        args.bfcl_root, args.task_manifest
    )
    splits = _split_bindings(args.split_bindings)
    args.out.mkdir(parents=True)
    os.environ["BFCL_PROJECT_ROOT"] = str((args.out / "bfcl_state").resolve())
    sys.path.insert(0, str(args.bfcl_root))
    if args.bfcl_dependency_path is not None:
        sys.path.append(str(args.bfcl_dependency_path))
    os.chdir(args.bfcl_root)
    bindings = t02_bfcl.OfficialBFCLBindings()
    # This object is the exact old H0 generation/continuation contract.  New
    # collection authority and budgets are intentionally enforced by this CLI
    # and its new summary rather than inherited from legacy fields inside it.
    frozen_policy = proposal_t02.load_policy(
        args.continuation_policy or args.source_plan
    )
    if frozen_policy.get("history_profile", "H0") != "H0":
        raise ValueError("proposal collection is H0-only")
    if frozen_policy.get("sampling") != {"mode": "greedy", "temperature": 0, "seed": 0}:
        raise ValueError("continuation policy must preserve the frozen greedy seed=0 contract")
    design_sha256 = hashlib.sha256(args.design.read_bytes()).hexdigest()
    if frozen_policy.get("design_sha256") != design_sha256:
        raise ValueError("continuation policy design binding differs from --design")
    if frozen_policy.get("family_bindings") != family_receipt:
        raise ValueError("continuation policy family bindings differ from verified inputs")
    if frozen_policy.get("official_source_files") != official_receipt:
        raise ValueError("continuation policy official-source binding differs from verified inputs")

    from t02_runtime import build_actor
    shared_models: list[Any | None] = [None]

    def actor_factory(task_id: str, task_index: int):
        actor = build_actor(
            design_path=args.design, checkpoint=args.checkpoint,
            backend_url=args.backend_url,
            output_dir=args.out / "actors" / f"{task_index:03d}-{t02_bfcl._safe_name(task_id)}",
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

    forbidden = {"D128": args.d128_manifest, "F128": args.f128_manifest}
    source_checkpoint_path = args.out / "source_checkpoint.json"
    result_checkpoint_path = args.out / "results.partial.json"
    adapter = t02_bfcl.ExactBFCLBranchAdapter(
        frozen_policy=frozen_policy, artifact_dir=args.out / "official_outcomes",
        family_bindings=family_bindings,
    )
    plan = None
    try:
        plan, candidates, adapter = collect_live_proposal_plan(
            task_ids=task_ids, actor_factory=actor_factory, bindings=bindings,
            forbidden_manifest_paths=forbidden, frozen_policy=frozen_policy,
            artifact_dir=args.out / "official_outcomes", seed=args.seed,
            target_states=args.target_states, train_states=args.train_states,
            candidate_snapshot_cap=args.candidate_snapshot_cap,
            max_states_per_group=args.max_states_per_group,
            max_states_per_task=args.max_states_per_task,
            max_source_task_starts=args.max_source_task_starts,
            family_bindings=family_bindings, split_bindings=splits,
            adapter=adapter,
            on_checkpoint=lambda value: t02_bfcl._write_json(
                source_checkpoint_path, value,
            ),
            on_progress=lambda value: t02_bfcl._write_json(
                args.out / "progress.json",
                {"schema": "proposal-t02-progress-h0-v1", **value},
            ),
        )
        t02_bfcl._write_json(args.out / "candidates.json", {
            "schema": "proposal-t02-live-candidates-h0-v1", "states": candidates,
        })
        t02_bfcl._write_json(args.out / "plan.json", plan)
        results = proposal_t02.execute_plan(
            plan, adapter,
            on_checkpoint=lambda value: t02_bfcl._write_json(
                result_checkpoint_path, value,
            ),
            on_progress=lambda value: t02_bfcl._write_json(
                args.out / "progress.json",
                {"schema": "proposal-t02-progress-h0-v1", "phase": "continuations", **value},
            ),
        )
        t02_bfcl._write_json(args.out / "results.json", results)
        labels = proposal_t02.label_results(plan, results)
        t02_bfcl._write_json(args.out / "labels.json", labels)
        summary = {
            "schema": SUMMARY_SCHEMA,
            "status": "completed",
            "proposal_protocol": proposal_t02.PROPOSAL_PROTOCOL,
            "history_profile": "H0",
            "source_task_budget": {"authorized": args.max_source_task_starts,
                                   "actual": adapter.source_collection_tasks},
            "collection": plan["collection"],
            "new_state_count": len(plan["states"]),
            "new_continuation_executions": results["complete_continuation_executions"],
            "old_359_360_budget_affected": False,
            "actual_split_counts": dict(Counter(row["split"] for row in plan["states"])),
            "generation_calls": adapter.cost_summary(),
            "plan_sha256": proposal_t02._digest(plan),
            "labels_sha256": proposal_t02._digest(labels),
            "source_checkpoint_sha256": proposal_t02._file_digest(source_checkpoint_path),
            "result_checkpoint_sha256": proposal_t02._file_digest(result_checkpoint_path),
        }
        t02_bfcl._write_json(args.out / "summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False), flush=True)
        return 0
    except BaseException as error:
        failure = {
            "schema": SUMMARY_SCHEMA, "status": "failed",
            "proposal_protocol": proposal_t02.PROPOSAL_PROTOCOL,
            "history_profile": "H0", "error_type": type(error).__name__,
            "error": str(error), "old_359_360_budget_affected": False,
            "source_task_budget": {"authorized": args.max_source_task_starts,
                                   "actual": adapter.source_collection_tasks if adapter else 0},
        }
        if adapter is not None:
            failure["generation_calls"] = adapter.cost_summary()
        if plan is not None:
            failure.update(
                plan_sha256=proposal_t02._digest(plan),
                new_state_count=len(plan["states"]),
                collection=plan["collection"],
            )
        if source_checkpoint_path.is_file():
            source_checkpoint = json.loads(source_checkpoint_path.read_text(encoding="utf-8"))
            failure.update(
                source_checkpoint_path=str(source_checkpoint_path),
                source_checkpoint_sha256=proposal_t02._file_digest(source_checkpoint_path),
                captured_candidate_state_count=source_checkpoint.get("candidate_state_count", 0),
                source_task_terminal_count=len(source_checkpoint.get("source_task_terminals", {})),
            )
        if result_checkpoint_path.is_file():
            result_checkpoint = json.loads(result_checkpoint_path.read_text(encoding="utf-8"))
            failure.update(
                result_checkpoint_path=str(result_checkpoint_path),
                result_checkpoint_sha256=proposal_t02._file_digest(result_checkpoint_path),
                complete_continuation_executions=result_checkpoint.get(
                    "complete_continuation_executions", 0
                ),
            )
        t02_bfcl._write_json(args.out / "summary.json", failure)
        raise
    finally:
        if adapter is not None:
            adapter.close()


if __name__ == "__main__":
    raise SystemExit(main())
