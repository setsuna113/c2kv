"""Collect exact H1/A0 current-turn labels for a frozen C1 risk head.

Each source retains at most one call and one STOP snapshot.  Only A0 is
continued, and only until the intervention's user turn ends.  Later decisions
within that turn use the frozen C0 policy.  These are calibration continuations,
not new complete-task T02 intervention triplets.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Callable, Mapping, Sequence
from urllib.parse import urlparse

import t02
import t02_bfcl as exact
from benchmarks.memory_runtime.always_compress import CapacityInfeasible


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _context(state: Mapping[str, Any]) -> dict:
    result = {**copy.deepcopy(state["q"]), **copy.deepcopy(state["draft"])}
    for source, target in {"task": "goal", "token_logprobs": "draft_logprobs",
                           "text": "draft_text", "tool_calls": "draft_tool_calls"}.items():
        if target not in result and source in result:
            result[target] = result[source]
    result.setdefault("is_stop", state["draft_kind"] == "stop")
    return result


class H1RiskAdapter(exact.ExactBFCLBranchAdapter):
    def discover_task(self, environment, actor, *, seed=0, max_states=2):
        """Capture outcome-blind call/STOP states without an unused A2 proposal."""
        if max_states not in (1, 2):
            raise ValueError("H1 calibration permits at most two snapshots per source")
        self.source_collection_tasks += 1
        found, kinds = [], set()
        try:
            while not environment.finished:
                state = self._invoke_counted(actor, "source_collection_generation_calls",
                                             "hold", environment.next_payload())
                self.source_collection_decisions += 1
                normalized = None
                if state.get("collectable", True):
                    try:
                        normalized = t02._normalize_candidate_state(
                            self._enrich_state(state, environment))
                    except t02.T02Error:
                        pass
                if normalized is not None and normalized["draft_kind"] not in kinds:
                    self.register_state(normalized, environment=environment, actor=actor,
                                        environment_snapshot=environment.capture(),
                                        actor_snapshot=actor.capture())
                    found.append(normalized)
                    kinds.add(normalized["draft_kind"])
                response = self._invoke_counted(actor, "source_collection_generation_calls",
                                                "submit_held", state.get("selected_ids", []))
                environment.commit_response(response)
                if len(found) >= max_states:
                    break
        except CapacityInfeasible as error:
            self._source_task_terminals[environment.task_id] = environment.terminate_capacity_infeasible(
                error, stage="h1_calibration_source")
        if not found:
            actor.close()
            environment.close()
        return found

    def run_a0_turn(self, state: Mapping[str, Any], *, history_binding: Mapping[str, Any],
                    evidence_mode: str) -> dict:
        snapshot = self.capture_state(state, frozen_policy=self.frozen_policy)
        restored = self.restore_state(snapshot, frozen_policy=self.frozen_policy)
        environment, actor = self._active["environment"], self._active["actor"]
        turn_index = environment.turn_index
        previous_valid = state.get("previous_turn_valid")
        before = self.branch_generation_calls
        capacity = None
        try:
            response = self._invoke_counted(actor, "branch_generation_calls", "submit_held", [])
            environment.commit_response(response)
            while not environment.finished and environment.turn_index == turn_index:
                response = self._invoke_counted(actor, "branch_generation_calls", "generate",
                                                environment.next_payload())
                environment.commit_response(response)
        except CapacityInfeasible as error:
            capacity = environment.terminate_capacity_infeasible(error, stage="h1_a0_current_turn")
        if capacity is not None or previous_valid is not True:
            checker = {"valid": None, "status": "capacity_terminated" if capacity is not None
                       else "unknown_previous_prefix", "previous_turn_valid": previous_valid}
            success = None
        else:
            if len(environment.all_model_response) <= turn_index:
                raise RuntimeError("A0 stopped before its current user turn completed")
            checker = environment.bindings.score_turn_prefix(environment.handler,
                environment.all_model_response, environment.ground_truth, environment.task, turn_index)
            valid = checker.get("valid")
            if type(valid) is not bool:
                raise ValueError("Official current-turn checker returned no boolean validity")
            success = valid
        common = {
            "state_id": state["state_id"], "task_id": state["task_id"],
            "task_group_id": state["task_group_id"], "source_history": "H1", "G": "record_bound",
            "source_gp_file_sha256": history_binding["gp_file_sha256"],
            "source_gp_payload_sha256": history_binding["gp_payload_sha256"],
            "evidence_mode": evidence_mode,
        }
        folder = self.artifact_dir / exact._safe_name(state["state_id"])
        raw_path, outcome_path = folder / "observation.json", folder / "A0.turn.json"
        context = _context(state)
        raw = {"schema": "c1-h1-calibration-observation-v1", **common,
               "context": context, "snapshot": snapshot,
               "decision_key": state["decision_key"], "candidate_state": copy.deepcopy(state)}
        raw_sha = exact._write_json(raw_path, raw)
        official = {
            "schema": "c1-h1-a0-turn-outcome-v1", **common, "branch_id": "A0",
            "candidate_ids": [], "snapshot_id": snapshot["snapshot_id"],
            "restore_receipt": restored, "turn_success": success,
            "turn_checker_result": exact._json_copy(environment.bindings.make_json_serializable(checker)),
            "previous_turn_valid": previous_valid,
            "intervention_turn": turn_index, "capacity_termination": capacity,
            "continued_only_until_current_turn_end": True, "task_success": None,
            "subsequent_policy": copy.deepcopy(self.frozen_policy),
            "generation_calls": self.branch_generation_calls - before,
            "model_result_prefix": exact._json_copy(environment.all_model_response[:turn_index + 1]),
            "official_checker": "bfcl_eval.eval_checker.multi_turn_eval.multi_turn_checker.multi_turn_checker",
        }
        official_sha = exact._write_json(outcome_path, official)
        row = {
            "schema": "c1-h1-calibration-row-v1", "state_id": state["state_id"],
            "task_id": state["task_id"], "task_group_id": state["task_group_id"],
            "split": "calibration", "context": context,
            "c1_risk_label": None if success is None else 1 - int(success),
            "c1_label_status": "unknown" if success is None else "known",
            "a0": {"branch_id": "A0", "action": "no_recovery", "turn_success": success},
            "evidence": {
                "observation": {"path": str(raw_path.resolve()), "file_sha256": raw_sha,
                                "payload_sha256": exact._json_digest(raw)},
                "official_outcome": {"path": str(outcome_path.resolve()), "file_sha256": official_sha,
                                     "payload_sha256": exact._json_digest(official)},
            },
        }
        self._release_state(state["state_id"])
        self._active = None
        return row


def collect_sources(*, task_ids: Sequence[str], calibration_groups: Sequence[str],
                    excluded_groups: Mapping[str, Sequence[str]], history_binding: Mapping[str, Any],
                    frozen_policy: Mapping[str, Any], output: Path,
                    environment_factory: Callable, actor_factory: Callable,
                    max_states: int = 38, evidence_mode: str = "production") -> dict:
    """Collect a fixed source sequence, persisting each A0 before the next source."""
    output = output.resolve()
    if output.exists():
        raise FileExistsError("Refusing to overwrite H1 calibration evidence")
    if evidence_mode not in {"production", "synthetic_fixture"}:
        raise ValueError("Unknown calibration evidence mode")
    if type(max_states) is not int or not 1 <= max_states <= 38:
        raise ValueError("H1 calibration state cap must be between 1 and 38")
    groups = set(calibration_groups)
    excluded = set(excluded_groups.get("training", [])) | set(excluded_groups.get("evaluation", []))
    if (not task_ids or len(set(task_ids)) != len(task_ids) or not groups or groups & excluded
            or any(t02.canonical_task_group_id(task) not in groups for task in task_ids)):
        raise ValueError("H1 calibration sources violate their independent group binding")
    gp_path = Path(history_binding["gp_path"])
    gp = json.loads(gp_path.read_text(encoding="utf-8"))
    if (history_binding.get("source_history") != "H1" or history_binding.get("G") != "record_bound"
            or history_binding.get("gp_file_sha256") != _sha(gp_path)
            or history_binding.get("gp_payload_sha256") != exact._json_digest(gp)
            or gp.get("G") != "record_bound" or gp.get("R") != 1
            or gp.get("set_selector") != "candidate_rule"):
        raise ValueError("Calibration requires the frozen H1/R1 C0 source policy")
    if frozen_policy.get("gp_file_sha256") != _sha(gp_path):
        raise ValueError("Continuation policy does not bind the H1 source configuration")
    if evidence_mode == "production":
        from evidence_c1_h1_calibration import H1_SOURCE_HASHES
        observed = {name: _sha(exact.HERE / "runtime" / name) for name in H1_SOURCE_HASHES}
        if observed != H1_SOURCE_HASHES or history_binding.get("h1_source_hashes") != observed:
            raise ValueError("Loaded source runtime is not the frozen H1 implementation")
    output.mkdir(parents=True)
    rows, tasks, failures = [], [], []
    selected_count = 0
    dataset = {"schema": "c1-h1-calibration-dataset-v1", "evidence_mode": evidence_mode,
               "history_binding": dict(history_binding), "excluded_group_ids": dict(excluded_groups),
               "calibration_group_ids": sorted(groups), "planned_task_ids": list(task_ids),
               "state_cap": max_states, "rows": rows, "task_receipts": tasks,
               "failures": failures, "status": "collecting"}
    exact._write_json(output / "dataset.json", dataset)
    for task_index, task_id in enumerate(task_ids):
        if selected_count >= max_states:
            break
        adapter = H1RiskAdapter(frozen_policy=frozen_policy, artifact_dir=output / "evidence")
        environment = actor = None
        discovery_returned = False
        try:
            environment = environment_factory(task_id)
            actor = actor_factory(task_id, task_index)
            if evidence_mode == "production":
                from t02_runtime import T02Actor
                from benchmarks.memory_runtime.recovery.experiment_config import parse_gp_config
                if not isinstance(environment.bindings, exact.OfficialBFCLBindings) or not isinstance(actor, T02Actor):
                    raise TypeError("Production calibration requires official BFCL and a real T02 actor")
                expected = parse_gp_config({**gp, "export_selection_state": True})
                if actor.runner.controller.gp != expected:
                    raise ValueError("Actual actor configuration differs from the frozen H1 policy")
            states = adapter.discover_task(environment, actor, max_states=min(2, max_states - selected_count))
            discovery_returned = True
            selected_count += len(states)
            dataset["selected_state_count"] = selected_count
            task_row = {"task_id": task_id, "selected_state_ids_before_outcomes": [s["state_id"] for s in states]}
            tasks.append(task_row)
            exact._write_json(output / "dataset.json", dataset)
            for state in states:
                rows.append(adapter.run_a0_turn(state, history_binding=history_binding, evidence_mode=evidence_mode))
                exact._write_json(output / "dataset.json", dataset)
            task_row.update(status="completed", costs=adapter.cost_summary(),
                            a0_current_turn_continuations=len(states),
                            source_terminal=adapter.source_task_terminal(task_id))
        except Exception as error:
            failures.append({"task_id": task_id, "error": f"{type(error).__name__}: {error}",
                             "automatic_retry": False})
        finally:
            registered = bool(adapter._task_refcounts)
            adapter.close()
            if not discovery_returned and not registered:
                if actor is not None:
                    actor.close()
                if environment is not None:
                    environment.close()
            exact._write_json(output / "dataset.json", dataset)
    dataset["status"] = "partial_failed" if failures else "completed"
    dataset["state_count"] = len(rows)
    dataset["complete_task_T02_branches"] = 0
    exact._write_json(output / "dataset.json", dataset)
    return dataset


def load_collection_spec(path: Path) -> dict:
    """Resolve immutable inputs before importing the remote model stack."""
    from evidence_c1_h1_calibration import json_file_binding
    spec = json.loads(path.read_text(encoding="utf-8"))
    if spec.get("schema") != "c1-h1-calibration-collection-spec-v1":
        raise ValueError("Unknown H1 calibration collection spec")
    documents = {}
    for key in ("source_design", "t02_labels", "t02_summary", "task_manifest", "d128_manifest", "f128_manifest"):
        reference = spec.get(key, {})
        bound = json_file_binding(reference["path"])
        if bound != reference:
            raise ValueError(f"Frozen calibration input changed: {key}")
        documents[key] = json.loads(Path(bound["path"]).read_text(encoding="utf-8"))
    labels = documents["t02_labels"]
    label_rows = labels.get("rows", [])
    completed = documents["t02_summary"]
    if (labels.get("schema") != "t02-multi-plan-labeled-set-v1"
            or labels.get("state_count") != 118 or len(label_rows) != 118
            or len({row.get("state_id") for row in label_rows}) != 118
            or labels.get("rows_sha256") != exact._json_digest(label_rows)
            or completed.get("schema") != "t02-streaming-recovery-summary-v1"
            or completed.get("phase") != "labels_completed"
            or completed.get("training_allowed") is not True
            or completed.get("combined_exact_state_count") != 118
            or completed.get("actual_split_counts") != {"train": 80, "calibration": 38}
            or completed.get("combined_labels_sha256") != exact._json_digest(labels)):
        raise ValueError("The completed T02 split is required before H1 calibration")
    if (any(row.get("split") not in {"train", "calibration"}
            or row.get("task_group_id") != t02.canonical_task_group_id(row["task_id"])
            for row in label_rows)
            or sum(row["split"] == "train" for row in label_rows) != 80
            or sum(row["split"] == "calibration" for row in label_rows) != 38):
        raise ValueError("T02 labels do not match their canonical 80/38 group split")
    splits = {name: {row["task_group_id"] for row in label_rows if row.get("split") == name}
              for name in ("train", "calibration")}
    if not splits["train"] or not splits["calibration"] or splits["train"] & splits["calibration"]:
        raise ValueError("T02 split binding is incomplete or overlaps")
    evaluation = {t02.canonical_task_group_id(task) for name in ("d128_manifest", "f128_manifest")
                  for task in documents[name]["task_ids"]}
    if (splits["train"] | splits["calibration"]) & evaluation:
        raise ValueError("T02 source groups overlap evaluation")
    source_tasks = [task for task in documents["task_manifest"]["task_ids"]
                    if t02.canonical_task_group_id(task) in splits["calibration"]]
    if len(set(source_tasks)) != len(source_tasks):
        raise ValueError("Source manifest repeats a calibration task")
    if {t02.canonical_task_group_id(task) for task in source_tasks} != splits["calibration"]:
        raise ValueError("Source manifest omits a calibration group")
    task_order = exact.round_robin_family_tasks(source_tasks, {
        task: t02.canonical_task_group_id(task) for task in source_tasks})
    if spec.get("task_ids") != task_order or spec.get("max_states") != 38:
        raise ValueError("Calibration source order or state cap differs from the fixed protocol")
    design = documents["source_design"]
    gp = json.loads(Path(spec["history_binding"]["gp_path"]).read_text(encoding="utf-8"))
    if (design.get("status") != "frozen"
            or design.get("resolved_configs", {}).get("controller", {}).get("gp_experiments") != gp
            or spec.get("frozen_policy", {}).get("design_sha256") != spec["source_design"]["file_sha256"]):
        raise ValueError("The source design does not bind the exact H1 GP configuration")
    if spec.get("automatic_retries") != 0:
        raise ValueError("Calibration does not permit automatic retries")
    return {"spec": spec, "calibration_groups": sorted(splits["calibration"]),
            "excluded_groups": {"training": sorted(splits["train"]), "evaluation": sorted(evaluation)}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    args.output = args.output.resolve()
    resolved = load_collection_spec(args.spec.resolve())
    spec = resolved["spec"]
    if args.validate_only:
        print(json.dumps({"status": "validated", "source_tasks": len(spec["task_ids"]),
                          "state_cap": 38, "model_calls": 0}))
        return 0
    device = spec.get("physical_device")
    endpoint = urlparse(spec.get("backend_url", ""))
    if (os.name == "nt" or type(device) is not int or device not in {0, 1, 2, 3, 4, 6}
            or os.environ.get("ASCEND_RT_VISIBLE_DEVICES") != str(device)
            or os.environ.get("C2KV_STRICT_NONFINITE_SAMPLING") != "1"
            or endpoint.scheme != "http" or endpoint.hostname not in {"127.0.0.1", "localhost"}):
        raise RuntimeError("Production calibration requires its authorized remote NPU environment")
    if args.output.exists():
        raise FileExistsError("Refusing calibration rerun")
    bfcl_root = Path(spec["bfcl_root"]).resolve()
    sys.path.insert(0, str(bfcl_root))
    if spec.get("bfcl_dependency_path"):
        sys.path.append(str(Path(spec["bfcl_dependency_path"]).resolve()))
    os.environ["BFCL_PROJECT_ROOT"] = str(args.output.resolve() / "bfcl_state")
    os.chdir(bfcl_root)
    from t02_runtime import build_actor
    bindings = exact.OfficialBFCLBindings()
    shared_models = [None]

    def environment_factory(task_id):
        task, gold = bindings.load_task(task_id)
        return exact.BFCLTaskEnvironment(task, gold, bindings=bindings)

    def actor_factory(task_id, index):
        actor = build_actor(design_path=spec["source_design"]["path"], checkpoint=spec["checkpoint"],
                            backend_url=spec["backend_url"], output_dir=args.output.resolve() /
                            "actors" / f"{index:03d}-{exact._safe_name(task_id)}")
        controller = actor.runner.controller
        if shared_models[0] is None:
            shared_models[0] = controller.backends
        else:
            while controller is not None:
                if hasattr(controller, "backends"):
                    controller.backends = shared_models[0]
                controller = getattr(controller, "base", None)
        return actor

    result = collect_sources(task_ids=spec["task_ids"], calibration_groups=resolved["calibration_groups"],
        excluded_groups=resolved["excluded_groups"], history_binding=spec["history_binding"],
        frozen_policy=spec["frozen_policy"], output=args.output.resolve(), max_states=38,
        environment_factory=environment_factory, actor_factory=actor_factory)
    print(json.dumps({"status": result["status"], "state_count": result["state_count"],
                      "output": str(args.output.resolve())}))
    return 0 if result["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
