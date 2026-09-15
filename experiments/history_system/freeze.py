"""Freeze one explicit hybrid-system candidate; no model launch."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import tarfile

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CURRENT_ALGORITHM = HERE / "configs/current_algorithm.json"
DEFAULT_BFCL_ROOT = "/home/liuyancheng/benchmarks/gorilla/berkeley-function-call-leaderboard"
CHECKPOINT_FIELDS = ("status", "path", "config_sha256", "selected_arm", "selected_step")
CHECKPOINT_STATUSES = {"candidate_for_selection", "selected"}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def checkpoint_binding(value):
    if not isinstance(value, dict) or any(key not in value for key in CHECKPOINT_FIELDS):
        raise ValueError("Checkpoint binding requires status, path, config_sha256, selected_arm and selected_step")
    binding = {key:value[key] for key in CHECKPOINT_FIELDS}
    if binding["status"] not in CHECKPOINT_STATUSES:
        raise ValueError("Checkpoint binding status must be candidate_for_selection or selected")
    if not isinstance(binding["path"], str) or not binding["path"]:
        raise ValueError("Checkpoint binding path must be explicit")
    if (not isinstance(binding["config_sha256"], str)
            or len(binding["config_sha256"]) != 64
            or any(char not in "0123456789abcdef" for char in binding["config_sha256"])):
        raise ValueError("Checkpoint binding config_sha256 must be a lowercase SHA-256 digest")
    if binding["selected_arm"] not in {"B", "C"}:
        raise ValueError("Checkpoint binding selected_arm must be B or C")
    if type(binding["selected_step"]) is not int or binding["selected_step"] < 0:
        raise ValueError("Checkpoint binding selected_step must be a nonnegative integer")
    if "artifacts_sha256" in value:
        artifacts = value["artifacts_sha256"]
        if not isinstance(artifacts, dict):
            raise ValueError("Checkpoint binding artifacts_sha256 must be an object")
        for name, digest in artifacts.items():
            if not isinstance(name, str):
                raise ValueError("Checkpoint artifact paths must be relative and contained")
            relative = Path(name)
            if not name or relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Checkpoint artifact paths must be relative and contained")
            if (not isinstance(digest, str) or len(digest) != 64
                    or any(char not in "0123456789abcdef" for char in digest)):
                raise ValueError("Checkpoint artifact hashes must be lowercase SHA-256 digests")
        if artifacts.get("config.json", binding["config_sha256"]) != binding["config_sha256"]:
            raise ValueError("Checkpoint config hashes conflict")
        binding["artifacts_sha256"] = artifacts
    return binding


def resolve_checkpoint_binding(current, *, ratio, binding_path=None):
    if binding_path is None:
        if ratio != current["ratio"]:
            raise ValueError("A non-default ratio requires an explicit checkpoint binding")
        return checkpoint_binding(current["checkpoint_selection"])
    return checkpoint_binding(read(binding_path))


def runtime_config_path(runtime_root, reference):
    relative = Path(reference)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Runtime config path must be relative and contained: {reference}")
    path = (runtime_root / relative).resolve()
    if runtime_root.resolve() not in path.parents:
        raise ValueError(f"Runtime config path escapes runtime: {reference}")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--controller", type=Path)
    parser.add_argument("--eval-policy", type=Path)
    parser.add_argument("--eval-capacity", type=Path)
    parser.add_argument("--history-tokens", type=int, default=768)
    parser.add_argument("--ratio", type=int, choices=(4, 8), default=None)
    parser.add_argument("--checkpoint-binding", type=Path)
    parser.add_argument("--shadow-feature-config", type=Path)
    parser.add_argument("--no-shadow-feature-config", action="store_true")
    parser.add_argument("--tasks-manifest", type=Path, default=HERE / "configs/r001.tasks.json")
    parser.add_argument("--source-lineage", type=Path,
                        default=REPO / "outputs/history_system_search/r001/remote_lineage.json")
    parser.add_argument("--bfcl-root-default", default=DEFAULT_BFCL_ROOT)
    parser.add_argument("--validation", type=Path, required=True)
    args = parser.parse_args()
    if not args.candidate.replace("_", "").isalnum() or not 0 < args.history_tokens <= 768:
        raise ValueError("Explicit candidate identity and at-most-B0 budget required")
    validation = read(args.validation)
    if validation["status"] != "passed":
        raise ValueError("Candidate CPU validation must pass before freeze")
    if validation.get("candidate_id", args.candidate) != args.candidate:
        raise ValueError("CPU validation candidate differs from freeze candidate")
    validated_current = validation.get("current_algorithm", {})
    if validated_current.get("sha256") != sha(CURRENT_ALGORITHM):
        raise ValueError("CPU-validated current algorithm has changed")
    current = read(CURRENT_ALGORITHM)
    if current.get("schema") != "a-current-algorithm-v1":
        raise ValueError("Unexpected current algorithm schema")
    source_runtime = HERE / "runtime"
    args.controller = args.controller or runtime_config_path(
        source_runtime, current["runtime"]["controller"])
    args.eval_policy = args.eval_policy or runtime_config_path(
        source_runtime, current["runtime"]["eval_policy"])
    args.eval_capacity = args.eval_capacity or runtime_config_path(
        source_runtime, current["runtime"]["eval_capacity"])
    if args.no_shadow_feature_config and args.shadow_feature_config is not None:
        parser.error("--no-shadow-feature-config conflicts with --shadow-feature-config")
    if not args.no_shadow_feature_config and args.shadow_feature_config is None:
        args.shadow_feature_config = runtime_config_path(
            source_runtime, current["runtime"]["shadow_feature_config"])
    if args.ratio is None:
        args.ratio = current["ratio"]
    if args.ratio not in (4, 8):
        raise ValueError("The selected checkpoint ratio must be 4 or 8")
    selected_checkpoint = resolve_checkpoint_binding(
        current, ratio=args.ratio, binding_path=args.checkpoint_binding)
    out = REPO / "outputs/history_system_search/r001" / args.candidate
    submitted = out / "submitted"
    if submitted.exists():
        raise FileExistsError(submitted)
    source_files = {p.relative_to(source_runtime).as_posix():sha(p) for p in source_runtime.rglob("*")
                    if p.is_file() and "__pycache__" not in p.parts and ".pytest_cache" not in p.parts and p.suffix != ".pyc"}
    if source_files != validation["runtime_source_files"]:
        raise ValueError("CPU-validated active runtime has changed")
    shutil.copytree(source_runtime, submitted / "runtime", ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache"))
    shutil.copyfile(HERE / "runner.py", submitted / "runner.py")
    runtime = submitted / "runtime"
    controller = read(args.controller)
    controller_path = "configs/controller.json"
    policy_path = "configs/eval_policy.json"
    capacity_path = "configs/eval_capacity.json"
    policy = read(args.eval_policy)
    policy["policy_id"] = "a-history-system-r001-" + args.candidate
    policy["policy"]["history_budget_bytes"] = args.history_tokens * 147456
    policy["policy"]["workspace_budget_bytes"] = args.history_tokens * 147456
    capacity = read(args.eval_capacity)
    for path, value in ((controller_path, controller), (policy_path, policy), (capacity_path, capacity)):
        save(runtime / path, value)
    feature_runtime = {}
    if args.shadow_feature_config is not None:
        feature_path = "configs/shadow_features.json"
        save(runtime / feature_path, read(args.shadow_feature_config))
        feature_runtime["shadow_feature_config"] = feature_path
    tasks = read(args.tasks_manifest)
    lineage = read(args.source_lineage)
    design = {key:current[key] for key in ("route", "compression_policy", "history_view_protocol",
        "decode_strategy", "prefill_chunk_size", "sampling", "session_cache_policy", "retry_contract")}
    design.update({
        "schema":"a-history-system-candidate-design-v1", "status":"frozen", "state":"smoke_passed",
        "candidate_id":args.candidate, "run_id_template":"a_history_r001_" + args.candidate,
        "evaluation_stage":"development_search", "launch_authorized":True,
        "acceptance_parameters_frozen":False, "task_ids":tasks["task_ids"],
        "task_manifest_sha256":sha(args.tasks_manifest),
        "limits": {**current["limits"], "tasks":len(tasks["task_ids"])},
        "ratio":args.ratio, "checkpoint_selection":selected_checkpoint,
        "runtime": {"controller":controller_path, "eval_policy":policy_path, "eval_capacity":capacity_path,
            **feature_runtime,
            **{key:current["runtime"][key] for key in ("server_module", "official_worker_module", "device", "dtype", "bfcl_python", "npu_allocator_metrics")}},
        "resolved_configs": {"controller":controller, "eval_policy":policy, "eval_capacity":capacity},
        "source_files": {p.relative_to(runtime).as_posix():sha(p) for p in runtime.rglob("*") if p.is_file()},
        "task_and_scorer_lineage": {"bfcl_root_default":args.bfcl_root_default,
            "source_bindings":{name:{"remote_sha256":digest} for name,digest in lineage["files"].items()}},
        "search_contract": {"B0_history_bytes":113246208, "history_token_budget":args.history_tokens,
            "quality_target":"Peer mean parity working target; no delta gate for search",
            "speed_is_auxiliary":True, "full_task_rollout":True, "new_cohort_planned_reference_not_automatic_retry":True},
        "cpu_validation": {"source":str(args.validation.resolve()), "sha256":sha(args.validation)},
        "cumulative_time_limit_seconds":None, "automatic_reruns":0,
    })
    save(submitted / "design.json", design)
    save(submitted / "tasks.json", tasks)
    spec = importlib.util.spec_from_file_location("frozen_candidate_runner", submitted / "runner.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    runner.ROOT = runtime
    runner.validate(design, allow_development=False)
    preview = runner.preview(design, argparse.Namespace(checkpoint=design["checkpoint_selection"]["path"],
        output="<unique-run-output>", benchmark_dir=design["task_and_scorer_lineage"]["bfcl_root_default"],
        python="/home/liuyancheng/envs/sgl/bin/python", bfcl_python="/home/liuyancheng/envs/bench/bin/python", port_base=28800))
    save(out / "preview.local.json", preview)
    files = {p.relative_to(submitted).as_posix():sha(p) for p in submitted.rglob("*") if p.is_file() and "__pycache__" not in p.parts}
    save(submitted / "package.manifest.json", {"schema":"a-history-system-package-v1", "candidate_id":args.candidate, "files":files})
    archive = out / "package.tar.gz"
    with tarfile.open(archive, "w:gz") as stream:
        for name in sorted([*files, "package.manifest.json"]):
            stream.add(submitted / name, arcname=name, recursive=False)
    receipt = {"status":"frozen_not_launched", "state":"smoke_passed", "candidate_id":args.candidate,
        "archive":str(archive), "archive_sha256":sha(archive), "manifest_sha256":sha(submitted / "package.manifest.json"),
        "design_sha256":sha(submitted / "design.json"), "runner_sha256":sha(submitted / "runner.py"),
        "files":len(files), "tasks":len(tasks["task_ids"]), "history_budget_bytes":args.history_tokens * 147456}
    save(out / "freeze.json", receipt)
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
