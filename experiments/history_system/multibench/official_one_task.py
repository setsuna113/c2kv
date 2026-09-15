#!/usr/bin/env python3
"""Run one frozen official task against one single-task native server."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from contextlib import contextmanager


TASK_SCHEMA = "history-system-official-task-v1"
RESULT_SCHEMA = "history-system-official-result-v1"
SOURCE_PROFILE = "openai-single-task-v1"
BENCHMARKS = frozenset({"tau2", "toolsandbox", "acon_appworld", "acebench"})
COMMON_FIELDS = frozenset({
    "schema", "benchmark", "task_id", "benchmark_dir", "bench_python",
    "user_base_url", "source_binding", "max_new_tokens",
})
OPTIONAL_FIELDS = frozenset({
    "run_name", "split", "tag", "max_steps", "max_iter", "category",
    "language", "max_dialog_turns", "appworld_root",
})

def _resolve_benchmark_root(script: Path) -> Path:
    script = Path(script).resolve()
    candidates = (
        script.parent / "runtime" / "benchmarks",  # frozen submitted/ layout
        script.parents[1] / "runtime" / "benchmarks",  # active history_system/ layout
    )
    required = (
        "adapters/tau2_adapter.py", "adapters/toolsandbox_adapter.py",
        "adapters/acon_adapter.py", "adapters/acebench_adapter.py", "metrics.py",
    )
    for candidate in candidates:
        if all((candidate / relative).is_file() for relative in required):
            return candidate
    raise RuntimeError(f"frozen benchmark runtime is incomplete; searched {candidates!r}")


BENCH_ROOT = _resolve_benchmark_root(Path(__file__))
if str(BENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCH_ROOT))


def _read_object(path: Path, what: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {what}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{what} must be a JSON object")
    return value


def _validate_task(task: dict[str, Any]) -> dict[str, Any]:
    unknown = set(task) - COMMON_FIELDS - OPTIONAL_FIELDS
    missing = COMMON_FIELDS - set(task)
    if unknown or missing:
        raise ValueError(f"task fields mismatch: missing={sorted(missing)!r}, unknown={sorted(unknown)!r}")
    if task.get("schema") != TASK_SCHEMA:
        raise ValueError(f"task.schema must be {TASK_SCHEMA!r}")
    if task.get("benchmark") not in BENCHMARKS:
        raise ValueError("unsupported benchmark")
    for field in ("task_id", "benchmark_dir", "bench_python"):
        if not isinstance(task.get(field), str) or not task[field]:
            raise ValueError(f"task.{field} must be a nonempty string")
    if type(task.get("max_new_tokens")) is not int or task["max_new_tokens"] <= 0:
        raise ValueError("task.max_new_tokens must be a positive integer")
    if not isinstance(task.get("user_base_url"), str):
        raise ValueError("task.user_base_url must be a string")
    if task["benchmark"] != "acon_appworld" and not task["user_base_url"]:
        raise ValueError("the official user simulator requires task.user_base_url")
    if (task["benchmark"] == "tau2"
            and (not isinstance(task.get("run_name"), str) or not task["run_name"])):
        raise ValueError("tau2 requires task.run_name")
    if task.get("split", "test_normal") != "test_normal" and task["benchmark"] == "acon_appworld":
        raise ValueError("AppWorld delivery is frozen to test_normal")
    if (task["benchmark"] == "acon_appworld"
            and (not isinstance(task.get("appworld_root"), str) or not task["appworld_root"])):
        raise ValueError("AppWorld requires task.appworld_root for the frozen data source")
    if task.get("category", "agent") != "agent" and task["benchmark"] == "acebench":
        raise ValueError("ACEBench delivery is frozen to category=agent")
    if task.get("language", "en") != "en" and task["benchmark"] == "acebench":
        raise ValueError("ACEBench delivery is frozen to language=en")
    if task.get("max_steps", 200) != 200 and task["benchmark"] == "tau2":
        raise ValueError("tau2 delivery preserves the official max_steps=200")
    binding = task.get("source_binding")
    if not isinstance(binding, dict) or set(binding) != {"components"}:
        raise ValueError("task.source_binding must contain exactly components")
    components = binding["components"]
    if not isinstance(components, list) or not components:
        raise ValueError("task.source_binding.components must be nonempty")
    return dict(task)


def _verify_source_binding(task: dict[str, Any]) -> list[dict[str, Any]]:
    receipts = []
    benchmark_dir = Path(task["benchmark_dir"]).resolve()
    bound_benchmark_dir = False
    bound_appworld_root = task["benchmark"] != "acon_appworld"
    for component in task["source_binding"]["components"]:
        if not isinstance(component, dict) or not isinstance(component.get("path"), str):
            raise ValueError("source binding component must have a path")
        root = Path(component["path"]).resolve()
        if root == benchmark_dir:
            bound_benchmark_dir = True
        if task["benchmark"] == "acon_appworld" and root == Path(task["appworld_root"]).resolve():
            bound_appworld_root = True
        kind = component.get("kind")
        if kind == "git":
            if set(component) != {"path", "kind", "revision", "required_clean"}:
                raise ValueError("git source component fields differ from the frozen schema")
            revision = component.get("revision")
            if (not isinstance(revision, str) or len(revision) != 40
                    or any(ch not in "0123456789abcdef" for ch in revision.lower())):
                raise ValueError("git source revision must be a 40-character hex commit")
            head = subprocess.run(
                ["git", "-C", str(root), "rev-parse", "HEAD"], check=True,
                capture_output=True, text=True,
            ).stdout.strip()
            if head != revision:
                raise ValueError(f"source revision mismatch at {root}")
            clean = not subprocess.run(
                ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
                check=True, capture_output=True, text=True,
            ).stdout.strip()
            if component["required_clean"] is not True or not clean:
                raise ValueError(f"source checkout is not frozen clean at {root}")
            receipts.append({"path": str(root), "kind": "git", "revision": head, "clean": True})
        elif kind == "files":
            if set(component) != {"path", "kind", "files"}:
                raise ValueError("files source component fields differ from the frozen schema")
            files = component.get("files")
            if not isinstance(files, dict) or not files:
                raise ValueError("files source component must pin at least one file")
            verified = {}
            for relative, expected in files.items():
                if (not isinstance(relative, str) or not relative
                        or Path(relative).is_absolute() or ".." in Path(relative).parts):
                    raise ValueError("source file keys must be safe relative paths")
                if (not isinstance(expected, str) or len(expected) != 64
                        or any(ch not in "0123456789abcdef" for ch in expected.lower())):
                    raise ValueError("source file hash must be sha256 hex")
                path = (root / relative).resolve()
                if root not in path.parents:
                    raise ValueError("source file escapes its component root")
                actual = _sha256(path)
                if actual != expected.lower():
                    raise ValueError(f"source file hash mismatch: {path}")
                verified[relative] = actual
            receipts.append({"path": str(root), "kind": "files", "files": verified})
        else:
            raise ValueError("source binding kind must be git or files")
    if not bound_benchmark_dir:
        raise ValueError("source binding does not cover task.benchmark_dir")
    if not bound_appworld_root:
        raise ValueError("source binding does not cover task.appworld_root")
    return receipts


def _validate_server(manifest: dict[str, Any], task: dict[str, Any], base_url: str) -> dict[str, Any]:
    if manifest.get("status") != "ready":
        raise ValueError("server manifest is not ready")
    if manifest.get("source_profile") != SOURCE_PROFILE:
        raise ValueError(f"server source_profile must be {SOURCE_PROFILE}")
    if manifest.get("benchmark") != task["benchmark"]:
        raise ValueError("server benchmark differs from task")
    if manifest.get("allowed_task_ids") != [task["task_id"]]:
        raise ValueError("server must own exactly this one official task")
    if not isinstance(base_url, str) or not base_url:
        raise ValueError("--base-url must be nonempty")
    if str(manifest.get("base_url", "")).rstrip("/") != base_url.rstrip("/"):
        raise ValueError("--base-url differs from server manifest")
    if not isinstance(manifest.get("model_name"), str) or not manifest["model_name"]:
        raise ValueError("server manifest has no model_name")
    if type(manifest.get("max_new_tokens")) is not int or manifest["max_new_tokens"] <= 0:
        raise ValueError("server manifest has no finite max_new_tokens")
    if manifest["max_new_tokens"] != task["max_new_tokens"]:
        raise ValueError("server max_new_tokens differs from frozen task")
    if task["benchmark"] == "toolsandbox" and manifest["model_name"] != "gpt-4o-2024-05-13":
        raise ValueError("ToolSandbox's fixed official role sends model=gpt-4o-2024-05-13")
    if task["benchmark"] == "acon_appworld" and manifest["max_new_tokens"] != 2048:
        raise ValueError("AppWorld's frozen source sends max_tokens=2048")
    if task["benchmark"] == "acebench" and manifest["max_new_tokens"] != 1200:
        raise ValueError("ACEBench's frozen configuration sends max_tokens=1200")
    return dict(manifest)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact_records(out_dir: Path, paths: list[Path]) -> list[dict[str, Any]]:
    records = []
    seen = set()
    for path in paths:
        path = Path(path)
        if not path.is_file():
            continue
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        try:
            label = str(resolved.relative_to(out_dir.resolve()))
        except ValueError:
            label = str(resolved)
        records.append({"path": label, "bytes": path.stat().st_size, "sha256": _sha256(path)})
    return sorted(records, key=lambda row: row["path"])


@contextmanager
def _pythonpath_first(*roots: Path):
    old = os.environ.get("PYTHONPATH")
    prefix = [str(Path(root).resolve()) for root in roots if Path(root).is_dir()]
    os.environ["PYTHONPATH"] = os.pathsep.join(prefix + ([old] if old else []))
    try:
        yield
    finally:
        if old is None:
            os.environ.pop("PYTHONPATH", None)
        else:
            os.environ["PYTHONPATH"] = old


def _run_tau2(task: dict[str, Any], base_url: str, out_dir: Path,
              model: str) -> tuple[dict[str, Any], list[Path]]:
    from adapters import tau2_adapter as adapter

    bench_dir = Path(task["benchmark_dir"]).resolve()
    run_name = task["run_name"] + "_" + hashlib.sha256(
        str(out_dir.resolve()).encode("utf-8")).hexdigest()[:12]
    simulation = bench_dir / "data" / "simulations" / run_name
    if simulation.exists():
        raise ValueError(f"refusing tau2 auto-resume state at {simulation}")
    command = adapter.run_command(
        base_url, task["user_base_url"], "airline", model, 1, run_name,
        max_tasks=1, num_trials=1, max_steps=None, timeout=None,
        python=task["bench_python"],
    )
    command += ["--task-ids", task["task_id"], "--max-retries", "0"]
    env = adapter.harness_env()
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (
        str((bench_dir / "src").resolve()), env.get("PYTHONPATH"))))
    subprocess.run(command, cwd=bench_dir, env=env, check=True)
    results = simulation / "results.json"
    subprocess.run(adapter.evaluate_command(simulation, python=task["bench_python"]),
                   cwd=bench_dir, env=env, check=True)
    updated = simulation / "updated_results.json"
    if not results.is_file() or not updated.is_file():
        raise RuntimeError("tau2 did not produce results.json and updated_results.json")
    data = _read_object(results, "tau2 results")
    rows = data.get("simulations") or []
    matching = [row for row in rows if str(row.get("task_id")) == task["task_id"]]
    if len(matching) != 1 or matching[0].get("termination_reason") == "infrastructure_error":
        raise RuntimeError("tau2 task has no valid terminal trajectory")
    summary = adapter.collect(updated, domain="airline")
    official = out_dir / "harness"
    official.mkdir(parents=True, exist_ok=False)
    copied = []
    for source in (results, updated, simulation / "config.json"):
        if source.is_file():
            target = official / f"official_{source.name}"
            shutil.copy2(source, target)
            copied.append(target)
    return summary, copied


def _dispatch(task: dict[str, Any], base_url: str, out_dir: Path,
              manifest: dict[str, Any]) -> tuple[dict[str, Any], list[Path]]:
    benchmark = task["benchmark"]
    model = manifest["model_name"]
    harness = out_dir / "harness"
    if benchmark == "tau2":
        return _run_tau2(task, base_url, out_dir, model)
    if harness.exists():
        raise ValueError(f"refusing existing harness output {harness}")
    if benchmark == "toolsandbox":
        from adapters import toolsandbox_adapter as adapter
        summary = adapter.run_ts(
            base_url, harness, test_mode=False,
            agent=adapter.AGENT, user=adapter.AGENT, expected=1,
            benchmark_dir=Path(task["benchmark_dir"]),
            user_base_url=task["user_base_url"], scenarios=[task["task_id"]],
            expected_task_ids=[task["task_id"]], python=task["bench_python"],
        )
        artifacts = list(harness.glob("agent_*/result_summary.json"))
    elif benchmark == "acon_appworld":
        from adapters import acon_adapter as adapter
        acon_dir = Path(task["benchmark_dir"])
        # The archived venv is editable. Put the frozen ACON source ahead of
        # its recorded editable target before the adapter snapshots os.environ.
        old_appworld_root = os.environ.get("APPWORLD_ROOT")
        os.environ["APPWORLD_ROOT"] = str(Path(task["appworld_root"]).resolve())
        try:
            with _pythonpath_first(acon_dir / "src"):
                summary = adapter.run_appworld(
                    base_url, harness, acon_dir=acon_dir, model=model,
                    tag=task.get("tag", f"history_system_{task['task_id']}"),
                    split="test_normal", max_iter=int(task.get("max_iter", 50)),
                    task_ids=[task["task_id"]], python=task["bench_python"],
                )
        finally:
            if old_appworld_root is None:
                os.environ.pop("APPWORLD_ROOT", None)
            else:
                os.environ["APPWORLD_ROOT"] = old_appworld_root
        artifacts = [harness / "selected_tasks.json", Path(summary["evaluation_path"])]
        run_dir = Path(summary["run_dir"]) / f"task_{task['task_id']}"
        artifacts += [run_dir / "results.json", run_dir / "llm_history.json"]
    else:
        from adapters import acebench_adapter as adapter
        summary = adapter.run_acebench(
            base_url, task["user_base_url"], harness,
            acebench_dir=Path(task["benchmark_dir"]), category="agent", language="en",
            model=model, user_model=model, num_threads=1,
            max_dialog_turns=int(task.get("max_dialog_turns", 40)),
            temperature=0.0, top_p=1.0,
            max_tokens=manifest["max_new_tokens"], task_ids=task["task_id"],
            python=task["bench_python"],
        )
        artifacts = [harness / "acebench_work" / "selected_tasks.json"]
        artifacts += list((harness / "acebench_work").glob("**/*_result.json"))
        artifacts += list((harness / "acebench_work").glob("**/*_score.json"))
    return summary, artifacts


def _write_result(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, type=Path)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--server-manifest", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--max-wall-seconds", required=True, type=int)
    args = parser.parse_args(argv)
    if args.max_wall_seconds <= 0:
        parser.error("--max-wall-seconds must be positive")
    args.out.mkdir(parents=True, exist_ok=True)
    result_path = args.out / "result.json"
    if result_path.exists():
        parser.error(f"refusing existing {result_path}")
    started = time.monotonic()
    benchmark = task_id = None
    try:
        task = _validate_task(_read_object(args.task, "task"))
        benchmark, task_id = task["benchmark"], task["task_id"]
        source_receipt = _verify_source_binding(task)
        manifest = _validate_server(
            _read_object(args.server_manifest, "server manifest"), task, args.base_url)
        summary, artifact_paths = _dispatch(task, args.base_url, args.out, manifest)
        score = summary.get("semantic_score")
        scored = (summary.get("n") == 1 and isinstance(score, (int, float))
                  and not isinstance(score, bool) and math.isfinite(float(score)))
        if not scored:
            raise RuntimeError("official adapter did not return exactly one finite semantic score")
        result = {
            "schema": RESULT_SCHEMA, "status": "completed", "benchmark": benchmark,
            "task_id": task_id, "scored": True, "official_score": float(score),
            "official_artifacts": _artifact_records(args.out, artifact_paths),
            "benchmark_summary": summary,
            "source_binding": source_receipt,
            "elapsed_seconds": time.monotonic() - started,
            "max_wall_seconds": args.max_wall_seconds, "error": None,
        }
        _write_result(result_path, result)
        return 0
    except (Exception, SystemExit) as error:
        result = {
            "schema": RESULT_SCHEMA, "status": "infra_failed", "benchmark": benchmark,
            "task_id": task_id, "scored": False, "official_score": None,
            "official_artifacts": [], "benchmark_summary": None,
            "source_binding": None,
            "elapsed_seconds": time.monotonic() - started,
            "max_wall_seconds": args.max_wall_seconds,
            "error": {"type": type(error).__name__, "message": str(error)},
        }
        _write_result(result_path, result)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
