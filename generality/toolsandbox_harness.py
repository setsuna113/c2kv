"""Run and verify one official ToolSandbox scenario through the shared paper adapter."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

try:
    from .process_lifecycle import interruptible
except ImportError:
    from process_lifecycle import interruptible


SCHEMA = "generality-toolsandbox-official-task-v1"
GENERATION_ROOT = Path("/home/liuyancheng/c2kv-generality-20260918")


def paper_source() -> Path:
    return Path(os.environ.get(
        "C2KV_PAPER_SOURCE", GENERATION_ROOT / "src" / "paper_harness")).resolve()


def worker_command(cell: dict, task_id: str, base_url: str,
                   user_base_url: str, out: Path,
                   native_server_dir: Path | None = None) -> list[str]:
    command = [
        cell["python_sgl"], str(Path(__file__).resolve()),
        "--task-id", task_id, "--base-url", base_url,
        "--user-base-url", user_base_url, "--out", str(out),
        "--benchmark-dir", cell["benchmark_dir"],
        "--bench-python", cell["python_bench"],
        "--model", cell["model_name"],
    ]
    if native_server_dir is not None:
        command += ["--native-server-dir", str(native_server_dir)]
    return command


def official_result(path: Path, task_id: str) -> dict | None:
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(result, dict) or result.get("schema") != SCHEMA:
        return None
    score = result.get("semantic_score")
    adapter = result.get("adapter_summary")
    failures = adapter.get("task_failures") if isinstance(adapter, dict) else None
    declared = []
    if isinstance(failures, dict):
        for kind in ("hiagent_invalid_retrieval", "c2kv_capacity_infeasible"):
            if failures.get(kind) == [task_id]:
                declared.append(kind)
            elif failures.get(kind) not in (None, []):
                return None
    failure_kind = declared[0] if len(declared) == 1 else None
    if (result.get("task_id") != task_id or result.get("n") != 1
            or result.get("official_scorer") != "tool_sandbox official CLI"
            or not isinstance(adapter, dict) or adapter.get("n") != 1
            or adapter.get("scenario_ids") != [task_id]
            or not isinstance(adapter.get("scenario_manifest"), dict)
            or adapter["scenario_manifest"].get("scenario_ids") != [task_id]
            or isinstance(score, bool) or not isinstance(score, (int, float))
            or not math.isfinite(score)
            or len(declared) > 1
            or result.get("task_failure_kind") != failure_kind):
        return None
    return result


def completed_task(cell_dir: Path, task_id: str, *, result_reader=None) -> bool:
    reader = result_reader or official_result
    return any(reader(path, task_id) is not None for path in
               (cell_dir / "batches").glob(
                   f"*/toolsandbox_worker/{task_id}/official_summary.json"))


def score_summary(cell: dict, *, result_reader=None) -> dict:
    cell_dir = Path(cell["cell_dir"])
    expected = cell["task_ids"]
    reader = result_reader or official_result
    scored = {}
    for task_id in expected:
        for path in sorted((cell_dir / "batches").glob(
                f"*/toolsandbox_worker/{task_id}/official_summary.json")):
            result = reader(path, task_id)
            if result is not None:
                scored[task_id] = {"task_id": task_id,
                                   "semantic_score": result["semantic_score"],
                                   "source": str(path)}
                if result.get("task_failure_kind") is not None:
                    scored[task_id]["task_failure_kind"] = result["task_failure_kind"]
                break
    pending = [task_id for task_id in expected if task_id not in scored]
    return {
        "schema": "generality-toolsandbox-score-summary-v1",
        "cell_id": cell["cell_id"],
        "n_total": len(expected),
        "n_official_scored": sum("task_failure_kind" not in row
                                  for row in scored.values()),
        "n_method_failures": sum("task_failure_kind" in row
                                 for row in scored.values()),
        "method_failure_task_ids": [task_id for task_id in expected
                                    if task_id in scored
                                    and "task_failure_kind" in scored[task_id]],
        "pending_task_ids": pending,
        "semantic_score": (sum(row["semantic_score"] for row in scored.values())
                           / len(expected) if not pending else None),
        "score_denominator": len(expected),
        "task_rows": [scored[task_id] for task_id in expected if task_id in scored],
    }


def run_task(task_id: str, base_url: str, user_base_url: str, out: Path,
             benchmark_dir: Path, bench_python: str, model: str,
             native_server_dir: Path | None = None) -> dict:
    source = paper_source()
    if not (source / "benchmarks" / "adapters" / "toolsandbox_adapter.py").is_file():
        raise FileNotFoundError("Shared paper ToolSandbox adapter is missing")
    sys.path.insert(0, str(source))
    sys.path.insert(0, str(source / "benchmarks"))
    from benchmarks.adapters import toolsandbox_adapter as adapter

    summary = adapter.run_ts(
        base_url, out, test_mode=False, scenarios=[task_id],
        benchmark_dir=benchmark_dir, python=bench_python, parallel=1,
        user_base_url=user_base_url, model=model, expected=1,
        native_server_dir=native_server_dir,
    )
    score = summary.get("semantic_score")
    if (summary.get("n") != 1 or summary.get("scenario_ids") != [task_id]
            or isinstance(score, bool) or not isinstance(score, (int, float))
            or not math.isfinite(score)):
        raise RuntimeError("Official ToolSandbox did not score the frozen scenario")
    failures = summary.get("task_failures")
    declared = [kind for kind in ("hiagent_invalid_retrieval", "c2kv_capacity_infeasible")
                if isinstance(failures, dict) and failures.get(kind) == [task_id]]
    if len(declared) > 1:
        raise RuntimeError("ToolSandbox task has conflicting declared failure kinds")
    result = {"schema": SCHEMA, "task_id": task_id, "n": 1,
              "semantic_score": float(score),
              "official_scorer": "tool_sandbox official CLI",
              "adapter_summary": summary,
              "task_failure_kind": declared[0] if declared else None}
    out.mkdir(parents=True, exist_ok=True)
    path = out / "official_summary.json"
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")
    return result


@interruptible
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--user-base-url", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--bench-python", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--native-server-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    result = run_task(args.task_id, args.base_url, args.user_base_url, args.out,
                      args.benchmark_dir, args.bench_python, args.model,
                      args.native_server_dir)
    print(json.dumps({"task_id": result["task_id"], "semantic_score":
                      result["semantic_score"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
