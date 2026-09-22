"""Run one official tau2 task through the shared paper adapter on NPU."""
from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import time
import uuid
from pathlib import Path

try:
    from .process_lifecycle import run_owned_worker
    from .upstream_liveness import UpstreamLiveness
    from .text_budget_failures import TEXT_HISTORY_BUDGET_FAILURE_CODES
except ImportError:  # Direct file launch on ascend03.
    from process_lifecycle import run_owned_worker
    from upstream_liveness import UpstreamLiveness
    from text_budget_failures import TEXT_HISTORY_BUDGET_FAILURE_CODES


PAPER = Path(os.environ.get(
    "C2KV_PAPER_SOURCE",
    str(Path.home() / "c2kv-generality-20260918" / "src" / "paper_harness"),
)).resolve()
BUDGET_FAILURE_CODES = frozenset({"decision_cap_reached", "generation_cap_reached"})
TASK_FAILURE_CODES = BUDGET_FAILURE_CODES | TEXT_HISTORY_BUDGET_FAILURE_CODES | {"context_overflow"}

_WORKER = """
import json
import sys
from pathlib import Path

paper_root, request_path, summary_path = map(Path, sys.argv[1:4])
sys.path.insert(0, str(paper_root))
from benchmarks.adapters.tau2_adapter import run_tau2

request = json.loads(request_path.read_text(encoding="utf-8"))
request["out_dir"] = Path(request["out_dir"])
request["tau2_dir"] = Path(request["tau2_dir"])
if request.get("native_server_dir") is not None:
    request["native_server_dir"] = Path(request["native_server_dir"])
summary = run_tau2(**request)
summary_path.write_text(json.dumps(summary, ensure_ascii=False, allow_nan=False),
                        encoding="utf-8")
"""


def _official_result(path: Path, task_id: str,
                     declared_failure: str | None = None) -> dict | None:
    """Require an exact official reward or an adapter-declared task failure."""
    raw = path.with_name("results.json")
    if not raw.is_file() or not path.is_file():
        return None
    try:
        before = json.loads(raw.read_text(encoding="utf-8"))
        after = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(before, dict) or not isinstance(after, dict):
        return None
    old_sims, new_sims = before.get("simulations"), after.get("simulations")
    if not isinstance(old_sims, list) or not isinstance(new_sims, list):
        return None
    if len(old_sims) != 1 or len(new_sims) != 1:
        return None
    original, scored = old_sims[0], new_sims[0]
    if not isinstance(original, dict) or not isinstance(scored, dict):
        return None
    if str(original.get("task_id")) != task_id or str(scored.get("task_id")) != task_id:
        return None
    termination = scored.get("termination_reason")
    if (not isinstance(termination, str) or not termination
            or original.get("termination_reason") != termination):
        return None
    reward_info = scored.get("reward_info")
    reward = reward_info.get("reward") if isinstance(reward_info, dict) else None
    if termination == "infrastructure_error":
        if (declared_failure not in TASK_FAILURE_CODES
                or (reward is not None and (
                    isinstance(reward, bool) or not isinstance(reward, (int, float))
                    or not math.isfinite(reward)))):
            return None
        return {"task_id": task_id, "semantic_score": 0.0,
                "termination": termination, "task_failure_kind": declared_failure,
                "score_source": ("typed_harness_budget_failure"
                                 if declared_failure in BUDGET_FAILURE_CODES else
                                 "typed_harness_method_failure"),
                "official_reward": reward}
    if declared_failure is not None:
        return None
    if isinstance(reward, bool) or not isinstance(reward, (int, float)):
        return None
    if not math.isfinite(reward):
        return None
    return {"task_id": task_id, "semantic_score": reward,
            "termination": termination}


def _declared_task_failure(attempt: Path, task_id: str) -> str | None:
    try:
        summary = json.loads((attempt / "summary.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    failure = summary.get("task_failures") if isinstance(summary, dict) else None
    if not isinstance(failure, dict):
        return None
    code = failure.get(task_id)
    if code in TASK_FAILURE_CODES and failure == {task_id: code}:
        return code
    return None


def _adapter_summary_matches(attempt: Path, task_id: str, official: dict) -> bool:
    path = attempt / "summary.json"
    request_path = attempt / "request.json"
    if not path.is_file() or not request_path.is_file():
        return False
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
        request = json.loads(request_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if (not isinstance(request, dict) or request.get("task_ids") != [task_id]
            or not isinstance(request.get("run_name"), str)
            or not isinstance(summary, dict) or summary.get("task_ids") != [task_id]):
        return False
    rows = summary.get("task_rows")
    failure = official.get("task_failure_kind")
    if failure is not None and summary.get("task_failures") != {task_id: failure}:
        return False
    return (isinstance(rows, list) and len(rows) == 1
            and isinstance(rows[0], dict)
            and rows[0].get("task_id") == task_id
            and rows[0].get("semantic_score") == official["semantic_score"]
            and rows[0].get("termination") == official["termination"]
            and (failure is None or (
                rows[0].get("task_failure_kind") == failure
                and "official_reward" in rows[0]
                and rows[0]["official_reward"] == official["official_reward"])))


def _completed_attempt(output_root: Path, task_id: str) -> tuple[Path, dict] | None:
    for path in sorted((output_root / "attempts").glob(
            "*/official/updated_results.json"), reverse=True):
        attempt = path.parent.parent
        result = _official_result(path, task_id, _declared_task_failure(attempt, task_id))
        if result is not None and _adapter_summary_matches(attempt,
                                                            task_id, result):
            return path, result
    return None


def completed_tau2_task(output_root: Path | str, task_id: str) -> bool:
    """A receipt alone never completes an unscored task."""
    return _completed_attempt(Path(output_root), str(task_id)) is not None


def completed_tau2_result(output_root: Path | str, task_id: str) -> dict | None:
    completed = _completed_attempt(Path(output_root), str(task_id))
    return completed[1] if completed is not None else None


def _write_receipt(path: Path, result: dict) -> None:
    if path.exists():
        history = path.with_name(path.stem + "_history.jsonl")
        with history.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"preserved_at_ns": time.time_ns(),
                                     "previous_raw": path.read_text(encoding="utf-8")})
                         + "\n")
    temporary = path.with_name(path.name + f".{os.getpid()}.{time.time_ns()}.tmp")
    temporary.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                         encoding="utf-8")
    os.replace(temporary, path)


def _success_receipt(output_root: Path, path: Path, task_id: str,
                     official: dict) -> dict:
    attempt = path.parent.parent
    request = json.loads((attempt / "request.json").read_text(encoding="utf-8"))
    result = {**official, "status": "completed", "returncode": 0,
              "run_name": request["run_name"],
              "official_results": str(path.relative_to(output_root)),
              "summary": str((attempt / "summary.json").relative_to(output_root))}
    done = output_root / "done.json"
    try:
        prior = json.loads(done.read_text(encoding="utf-8")) if done.exists() else None
    except (OSError, ValueError):
        prior = None
    if (isinstance(prior, dict) and prior.get("status") == "completed"
            and prior.get("task_id") == task_id
            and prior.get("official_results") == result["official_results"]
            and prior.get("semantic_score") == official["semantic_score"]
            and prior.get("termination") == official["termination"]):
        return prior
    if prior != result:
        _write_receipt(done, result)
    return result


def run_tau2_task(cell: dict, task_id: str, agent_base_url: str,
                  user_base_url: str, output_root: Path | str,
                  *, native_server_dir: Path | str | None = None) -> dict:
    """Run a pinned single task; keep official files for exact resume checks.

    ``output_root`` is the per-task directory. The caller owns the agent proxy
    and the raw engine; this helper owns only the tau2 child process.
    """
    if not isinstance(task_id, str) or not task_id:
        raise ValueError("task_id must be a nonempty string")
    output_root = Path(output_root).resolve()
    if native_server_dir is not None:
        native_server_dir = Path(native_server_dir).resolve()
        if not native_server_dir.is_dir():
            raise ValueError("native_server_dir must be the existing controller server directory")
    existing = _completed_attempt(output_root, task_id)
    if existing is not None:
        return _success_receipt(output_root, existing[0], task_id, existing[1])

    python = cell.get("python_tau2") or cell["python_bench"]
    tau2_dir = Path(cell.get("tau2_dir") or cell["benchmark_dir"]).resolve()
    paper = Path(cell.get("paper_root") or PAPER).resolve()
    attempt = output_root / "attempts" / f"a{time.time_ns()}_{uuid.uuid4().hex}"
    attempt.mkdir(parents=True, exist_ok=False)
    run_name = f"npu_tau2_{uuid.uuid4().hex}"
    request = {
        "base_url": agent_base_url,
        "user_base_url": user_base_url,
        "out_dir": str(attempt),
        "task_set": cell.get("tau2_task_set", "airline"),
        "task_split": cell.get("tau2_split", "base"),
        "task_ids": [task_id],
        "run_name": run_name,
        "model": cell["model_name"],
        "user_model": cell.get("upstream_model_name", "gen-c1000"),
        "native": (cell.get("backend") == "c2kv"
                   or cell.get("condition") == "tracer_history"),
        "agent_max_tokens": cell.get("caps", {}).get("max_completion_tokens", 2048),
        "python": python,
        "tau2_dir": str(tau2_dir),
        "num_workers": 1,
        "num_trials": 1,
        "max_steps": cell.get("tau2_max_steps"),
        "timeout": cell.get("tau2_timeout"),
    }
    if native_server_dir is not None:
        request["native_server_dir"] = str(native_server_dir)
    (attempt / "request.json").write_text(
        json.dumps(request, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join((str(paper), str(paper / "benchmarks")))
    env["no_proxy"] = env["NO_PROXY"] = "127.0.0.1,localhost"
    for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        env.pop(key, None)

    agent_alive = UpstreamLiveness(agent_base_url)
    user_alive = UpstreamLiveness(user_base_url)

    def monitor() -> None:
        agent_alive()
        user_alive()

    started = time.monotonic()
    with (attempt / "benchmark.log").open("wb") as log:
        returncode = run_owned_worker(
            [python, "-c", _WORKER, str(paper), str(attempt / "request.json"),
             str(attempt / "summary.json")],
            cwd=str(paper), env=env, stdout=log, stderr=log,
            stdin=subprocess.DEVNULL, monitor=monitor,
        )
    source = Path(tau2_dir) / "data" / "simulations" / run_name
    official_dir = attempt / "official"
    official_dir.mkdir(exist_ok=True)
    for name in ("results.json", "updated_results.json"):
        if (source / name).is_file():
            shutil.copy2(source / name, official_dir / name)
    official = _official_result(official_dir / "updated_results.json", task_id,
                                _declared_task_failure(attempt, task_id))
    if official is not None and not _adapter_summary_matches(attempt, task_id, official):
        official = None
    result = {"task_id": task_id,
              "status": "completed" if returncode == 0 and official else "infra_error",
              "returncode": returncode, "wall_s": time.monotonic() - started,
              "run_name": run_name,
              "official_results": str((official_dir / "updated_results.json")
                                      .relative_to(output_root)),
              "summary": str((attempt / "summary.json").relative_to(output_root))}
    if result["status"] == "completed":
        result.update(official)
        _write_receipt(output_root / "done.json", result)
    else:
        _write_receipt(output_root / "status.json", result)
    return result
