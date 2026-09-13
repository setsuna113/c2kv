#!/usr/bin/env python3
"""Benchmark-interpreter worker for the next-compression official driver."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping

VENDOR_ROOT = Path(__file__).resolve().parent
if str(VENDOR_ROOT) not in sys.path:
    sys.path.insert(0, str(VENDOR_ROOT))
PREFIX = "NEXT_BENCH_JSON:"
MANIFEST_SCHEMA = "next-compression-full-benchmark-manifest-v1"
RESULT_SCHEMA = "next-compression-official-benchmark-worker-v1"


def _emit(value: Any) -> None:
    print(PREFIX + json.dumps(value, ensure_ascii=False, separators=(",", ":")), flush=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected JSON objects in {path}")
            rows.append(value)
    return rows


def _versions(names: list[str]) -> dict[str, str | None]:
    result = {}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def _python_receipt(packages: list[str]) -> dict[str, Any]:
    return {"executable": str(Path(sys.executable).resolve()),
            "version": sys.version, "packages": _versions(packages)}


def _insert_source(root: Path) -> None:
    for path in (root, root / "src"):
        if path.is_dir() and str(path) not in sys.path:
            sys.path.insert(0, str(path))


def enumerate_bfcl(root: Path, split: Mapping[str, Any]) -> dict[str, Any]:
    _insert_source(root)
    os.environ.setdefault("BFCL_PROJECT_ROOT", str(root))
    from bfcl_eval.utils import load_dataset_entry, parse_test_category_argument

    category = str(split["category"])
    concrete = list(parse_test_category_argument([category]))
    if not concrete or "format_sensitivity" in concrete:
        raise ValueError(f"BFCL category is empty or non-scoring: {category}")
    ids = []
    counts = {}
    for name in concrete:
        rows = load_dataset_entry(name, include_prereq=False,
                                  include_language_specific_hint=False)
        category_ids = [str(row["id"]) for row in rows]
        if not category_ids or len(category_ids) != len(set(category_ids)):
            raise ValueError(f"invalid BFCL IDs in category {name}")
        ids.extend(category_ids)
        counts[name] = len(category_ids)
    if len(ids) != len(set(ids)):
        raise ValueError("BFCL concrete categories overlap in official IDs")
    return {"task_ids": ids, "concrete_categories": concrete,
            "category_counts": counts,
            "loader": "bfcl_eval.utils.load_dataset_entry",
            "interpreter": _python_receipt(["bfcl-eval", "openai", "httpx"])}


def enumerate_tau2(root: Path, split: Mapping[str, Any]) -> dict[str, Any]:
    _insert_source(root)
    from tau2.runner.helpers import get_tasks

    task_set = str(split["task_set"])
    task_split = str(split["task_split"])
    tasks = get_tasks(task_set, task_split_name=task_split,
                      task_ids=None, num_tasks=None)
    ids = [str(task.id) for task in tasks]
    return {"task_ids": ids, "loader": "tau2.runner.helpers.get_tasks",
            "task_set": task_set, "task_split": task_split,
            "interpreter": _python_receipt(["tau2-bench", "litellm"])}


def enumerate_toolsandbox(root: Path, split: Mapping[str, Any]) -> dict[str, Any]:
    _insert_source(root)
    from tool_sandbox.cli.utils import resolve_scenarios
    from tool_sandbox.common.tool_discovery import ToolBackend

    if split.get("scope") != "all_registered_scenarios":
        raise ValueError("ToolSandbox only supports the frozen full registry scope")
    resolved = resolve_scenarios(desired_scenario_names=None,
                                 preferred_tool_backend=ToolBackend.DEFAULT)
    ids = sorted(str(name) for name in resolved)
    return {"task_ids": ids,
            "loader": "tool_sandbox.cli.utils.resolve_scenarios",
            "preferred_tool_backend": "DEFAULT",
            "interpreter": _python_receipt(["tool-sandbox", "openai"])}


def enumerate_acebench(root: Path, split: Mapping[str, Any]) -> dict[str, Any]:
    category_file = root / "category.py"
    spec = importlib.util.spec_from_file_location("next_bench_ace_category", category_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load ACEBench category map: {category_file}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    category = str(split["category"])
    language = str(split["language"])
    tests = list(module.ACE_DATA_CATEGORY.get(category, [category]))
    ids = []
    sources = []
    for test in tests:
        source = root / "data_all" / f"data_{language}" / f"data_{test}.json"
        answer = source.parent / "possible_answer" / source.name
        rows = _jsonl(source)
        answers = _jsonl(answer)
        row_ids = [str(row["id"]) for row in rows]
        answer_ids = {str(row["id"]) for row in answers}
        missing = set(row_ids) - answer_ids
        if missing:
            raise ValueError(f"ACEBench possible answers missing IDs in {test}: {sorted(missing)[:10]}")
        ids.extend(row_ids)
        sources.append({"test": test, "data": str(source.resolve()),
                        "data_sha256": _sha256(source), "answers": str(answer.resolve()),
                        "answers_sha256": _sha256(answer), "count": len(row_ids)})
    return {"task_ids": ids, "loader": "category.py + official data JSONL",
            "resolved_tests": tests, "source_files": sources,
            "interpreter": _python_receipt([])}


def enumerate_appworld(root: Path, split: Mapping[str, Any],
                       appworld_root: Path) -> dict[str, Any]:
    _insert_source(root)
    os.environ["APPWORLD_ROOT"] = str(appworld_root.resolve())
    from appworld import load_task_ids

    name = str(split["split"])
    ids = [str(task_id) for task_id in load_task_ids(name)]
    dataset = appworld_root / "data" / "datasets" / f"{name}.txt"
    return {"task_ids": ids, "loader": "appworld.load_task_ids",
            "dataset": str(dataset.resolve()), "dataset_sha256": _sha256(dataset),
            "interpreter": _python_receipt(["appworld"])}


def enumerate_source(benchmark: str, root: Path, split: Mapping[str, Any],
                     appworld_root: Path | None) -> dict[str, Any]:
    if benchmark == "bfcl":
        return enumerate_bfcl(root, split)
    if benchmark == "tau2":
        return enumerate_tau2(root, split)
    if benchmark == "toolsandbox":
        return enumerate_toolsandbox(root, split)
    if benchmark == "acebench":
        return enumerate_acebench(root, split)
    if benchmark == "appworld" and appworld_root is not None:
        return enumerate_appworld(root, split, appworld_root)
    raise ValueError(f"unsupported benchmark or missing AppWorld root: {benchmark}")


def _env_with_source(root: Path) -> dict[str, str]:
    env = os.environ.copy()
    prefixes = [str(path.resolve()) for path in (root / "src", root) if path.is_dir()]
    env["PYTHONPATH"] = os.pathsep.join(prefixes + ([env["PYTHONPATH"]]
                                                     if env.get("PYTHONPATH") else []))
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["NO_PROXY"] = "127.0.0.1,localhost"
    env["no_proxy"] = "127.0.0.1,localhost"
    return env


def run_bfcl(row: Mapping[str, Any], endpoint: str, model: str,
             output: Path, ids: list[str]) -> dict[str, Any]:
    root = Path(row["root"])
    _insert_source(root)
    from adapters import bfcl_adapter as adapter

    harness = output / "harness"
    handler = "next-" + hashlib.sha256((model + str(output)).encode()).hexdigest()[:12]
    previous = Path.cwd()
    os.chdir(root)
    try:
        return adapter.run_bfcl(endpoint, categories=row["split"]["category"],
                                mode="both", run_ids=ids, model=model,
                                handler_name=handler, project_root=harness)
    finally:
        os.chdir(previous)


def run_tau2(row: Mapping[str, Any], endpoint: str, model: str,
             user_endpoint: str, user_model: str, output: Path,
             ids: list[str]) -> dict[str, Any]:
    root = Path(row["root"])
    from adapters import tau2_adapter as adapter

    run_name = "next_" + hashlib.sha256(str(output).encode()).hexdigest()[:16]
    simulation = root / "data" / "simulations" / run_name
    if simulation.exists():
        raise FileExistsError(f"refusing tau2 auto-resume state: {simulation}")
    command = adapter.run_command(endpoint, user_endpoint, row["split"]["task_set"],
                                  model, 1, run_name, num_trials=1,
                                  max_steps=200, python=sys.executable)
    command[command.index("--user-llm") + 1] = f"openai/{user_model}"
    command += ["--task-split-name", row["split"]["task_split"],
                "--task-ids", *ids, "--max-retries", "0"]
    env = _env_with_source(root)
    subprocess.run(command, cwd=root, env=env, check=True)
    results = simulation / "results.json"
    raw = json.loads(results.read_text(encoding="utf-8"))
    simulations = raw.get("simulations") or []
    got = [str(item.get("task_id")) for item in simulations]
    if got != ids:
        if set(got) != set(ids):
            raise RuntimeError("tau2 result task IDs differ from frozen selection")
    infra = [str(item.get("task_id")) for item in simulations
             if item.get("termination_reason") == "infrastructure_error"]
    if infra:
        raise RuntimeError(f"tau2 infrastructure_error tasks: {infra[:20]}")
    subprocess.run(adapter.evaluate_command(simulation, python=sys.executable),
                   cwd=root, env=env, check=True)
    updated = simulation / "updated_results.json"
    if not updated.is_file():
        raise RuntimeError("tau2 official evaluator wrote no updated_results.json")
    summary = adapter.collect(updated, domain=row["split"]["domain"])
    harness = output / "harness"
    shutil.copytree(simulation, harness)
    summary["simulation_dir"] = str(harness)
    return summary


def run_toolsandbox(row: Mapping[str, Any], endpoint: str,
                    user_endpoint: str, output: Path,
                    ids: list[str], smoke: bool) -> dict[str, Any]:
    from adapters import toolsandbox_adapter as adapter

    return adapter.run_ts(
        endpoint, output / "harness", test_mode=False, agent=adapter.AGENT,
        user=adapter.AGENT, expected=len(ids), benchmark_dir=Path(row["root"]),
        user_base_url=user_endpoint, scenarios=ids if smoke else None,
        expected_task_ids=ids, python=sys.executable)


def run_acebench(row: Mapping[str, Any], endpoint: str, model: str,
                 user_endpoint: str, user_model: str, output: Path,
                 ids: list[str]) -> dict[str, Any]:
    from adapters import acebench_adapter as adapter

    return adapter.run_acebench(
        endpoint, user_endpoint, output / "harness", acebench_dir=Path(row["root"]),
        category=row["split"]["category"], language=row["split"]["language"],
        model=model, user_model=user_model, num_threads=1, max_dialog_turns=40,
        temperature=0.0, top_p=1.0, max_tokens=1200,
        task_ids=",".join(ids), python=sys.executable)


def run_appworld(row: Mapping[str, Any], endpoint: str, model: str,
                 output: Path, ids: list[str]) -> dict[str, Any]:
    from adapters import acon_adapter as adapter

    data_root = Path(row["source_binding"]["appworld_data"]["root"])
    old = os.environ.get("APPWORLD_ROOT")
    os.environ["APPWORLD_ROOT"] = str(data_root)
    try:
        return adapter.run_appworld(
            endpoint, output / "harness", acon_dir=Path(row["root"]), model=model,
            tag="next_" + hashlib.sha256(str(output).encode()).hexdigest()[:12],
            split=row["split"]["split"], max_iter=50, task_ids=ids,
            python=sys.executable, request_log=None)
    finally:
        if old is None:
            os.environ.pop("APPWORLD_ROOT", None)
        else:
            os.environ["APPWORLD_ROOT"] = old


def _official_run(benchmark: str, row: Mapping[str, Any], endpoint: str,
                  model: str, user_endpoint: str, user_model: str,
                  output: Path, ids: list[str], smoke: bool) -> dict[str, Any]:
    if benchmark == "bfcl":
        return run_bfcl(row, endpoint, model, output, ids)
    if benchmark == "tau2":
        return run_tau2(row, endpoint, model, user_endpoint, user_model, output, ids)
    if benchmark == "toolsandbox":
        return run_toolsandbox(row, endpoint, user_endpoint, output, ids, smoke)
    if benchmark == "acebench":
        return run_acebench(row, endpoint, model, user_endpoint, user_model, output, ids)
    if benchmark == "appworld":
        return run_appworld(row, endpoint, model, output, ids)
    raise ValueError(f"unsupported benchmark: {benchmark}")


def _error_status(error: BaseException) -> str:
    status_code = getattr(error, "status_code", None)
    response = getattr(error, "response", None)
    if status_code is None and response is not None:
        status_code = getattr(response, "status_code", None)
    if status_code == 422 and "c2kv_model_failure" in str(error):
        return "model_failure_unscored_in_denominator"
    return "infra_failed"


def _preserve_partial_tau2(row: Mapping[str, Any], output: Path) -> None:
    run_name = "next_" + hashlib.sha256(str(output).encode()).hexdigest()[:16]
    simulation = Path(row["root"]) / "data" / "simulations" / run_name
    harness = output / "harness"
    if simulation.is_dir() and not harness.exists():
        shutil.copytree(simulation, harness)

def _artifact_inventory(output: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "worker_result.json":
            rows.append({"path": path.relative_to(output).as_posix(),
                         "bytes": path.stat().st_size, "sha256": _sha256(path)})
    return rows


def run_one(args: argparse.Namespace) -> int:
    started = time.monotonic()
    output = Path(args.output).resolve()
    result_path = output / "worker_result.json"
    benchmark = args.benchmark
    try:
        manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
        if manifest.get("schema") != MANIFEST_SCHEMA:
            raise ValueError("unsupported benchmark manifest")
        row = manifest["benchmarks"][benchmark]
        frozen_ids = list(row["task_ids"])
        ids = frozen_ids[:args.smoke_tasks] if args.smoke_tasks else frozen_ids
        if row["user_simulator_required"] and args.endpoint.rstrip("/") == manifest["user_endpoint"].rstrip("/"):
            raise ValueError(f"{benchmark} agent and user-simulator endpoints must differ")
        current = enumerate_source(
            benchmark, Path(row["root"]), row["split"],
            Path(row["source_binding"]["appworld_data"]["root"])
            if benchmark == "appworld" else None)
        if current["task_ids"] != frozen_ids:
            raise RuntimeError("installed official task IDs differ from frozen manifest")
        summary = _official_run(
            benchmark, row, args.endpoint, args.model_alias,
            manifest["user_endpoint"], manifest["user_model_alias"], output,
            ids, args.smoke_tasks is not None)
        score = summary.get("semantic_score")
        if (summary.get("n") != len(ids) or not isinstance(score, (int, float))
                or isinstance(score, bool) or not math.isfinite(float(score))):
            raise RuntimeError("official scorer did not return the exact finite denominator")
        result = {
            "schema": RESULT_SCHEMA, "status": "completed", "scored": True,
            "benchmark": benchmark, "scope": "artifact_scope_smoke" if args.smoke_tasks else "full_frozen_split",
            "frozen_denominator": len(frozen_ids), "execution_denominator": len(ids),
            "task_ids": ids, "official_summary": summary,
            "elapsed_seconds": time.monotonic() - started, "error": None,
        }
        result["artifacts"] = _artifact_inventory(output)
        _write_result(result_path, result)
        return 0
    except BaseException as error:
        if benchmark == "tau2" and "row" in locals():
            _preserve_partial_tau2(row, output)
        if isinstance(error, KeyboardInterrupt):
            status = "interrupted"
        else:
            status = _error_status(error)
        result = {
            "schema": RESULT_SCHEMA, "status": status, "scored": False,
            "benchmark": benchmark, "official_summary": None,
            "elapsed_seconds": time.monotonic() - started,
            "error": {"type": type(error).__name__, "message": str(error)},
            "artifacts": _artifact_inventory(output) if output.exists() else [],
        }
        _write_result(result_path, result)
        return 2


def _write_result(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                    allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    enum = sub.add_parser("enumerate")
    enum.add_argument("--benchmark", required=True,
                      choices=("bfcl", "tau2", "toolsandbox", "acebench", "appworld"))
    enum.add_argument("--root", required=True, type=Path)
    enum.add_argument("--appworld-root", type=Path)
    enum.add_argument("--split-json", required=True)
    run = sub.add_parser("run")
    run.add_argument("--manifest", required=True, type=Path)
    run.add_argument("--benchmark", required=True,
                     choices=("bfcl", "tau2", "toolsandbox", "acebench", "appworld"))
    run.add_argument("--endpoint", required=True)
    run.add_argument("--model-alias", required=True)
    run.add_argument("--output", required=True, type=Path)
    run.add_argument("--smoke-tasks", type=int)
    args = parser.parse_args(argv)
    if args.action == "enumerate":
        split = json.loads(args.split_json)
        _emit(enumerate_source(args.benchmark, args.root.resolve(), split,
                               args.appworld_root.resolve() if args.appworld_root else None))
        return 0
    if args.smoke_tasks is not None and args.smoke_tasks <= 0:
        parser.error("--smoke-tasks must be positive")
    return run_one(args)


if __name__ == "__main__":
    raise SystemExit(main())