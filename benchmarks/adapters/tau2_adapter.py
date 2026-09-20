"""Official tau2 CLI adapter with separate agent and user endpoints.

The selected tau2 checkout resolves task identities, runs the simulations,
and scores saved trajectories. Agent requests pass through the arm proxy;
user-simulator requests use the raw endpoint. Benchmark-owned hooks attach
task identity and capture agent-only telemetry for the paper matrix.
"""
from __future__ import annotations

import json
import hashlib
import os
import math
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paper.process_lifecycle import run_owned  # noqa: E402
from metrics import aggregate, protocol_columns_for_turn  # noqa: E402

from adapters.base import RunContext, v1  # noqa: E402

NAME = "tau2"
TAU2_DIR = Path(os.environ.get("TAU2_DIR") or Path.home() / "benchmarks" / "tau2")


def add_arguments(parser) -> None:
    """tau2-only CLI flags (shared ones live in run.py's core block)."""
    parser.add_argument("--benchmark-dir", type=Path, default=None,
                        help="tau2: checkout containing src/tau2")
    parser.add_argument("--task-set", default="airline")
    parser.add_argument("--tau2-task-split", default="base",
                        help="tau2: official task split (default base)")
    parser.add_argument("--tau2-task-ids", default="",
                        help="tau2: comma-separated official task IDs")
    parser.add_argument("--tau2-num-trials", type=int, default=None,
                        help="tau2: trials per selected task (unset keeps the official default)")
    parser.add_argument("--tau2-max-steps", type=int, default=None,
                        help="tau2: per-task turn cap (unset keeps the official default)")
    parser.add_argument("--tau2-timeout", type=int, default=None,
                        help="tau2: per-task wallclock cap in seconds (unset means no cap)")
    parser.add_argument("--tau2-agent-max-tokens", type=int, default=4096,
                        help="tau2: agent completion cap; must match the native replay contract")


def run_command(base_url: str, user_base_url: str, task_set: str, model: str,
                num_workers: int, run_name: str,
                max_tasks: Optional[int] = None,
                num_trials: Optional[int] = None,
                max_steps: Optional[int] = None,
                timeout: Optional[int] = None,
                python: Optional[str] = None,
                task_split: str = "base",
                task_ids: Optional[List[str]] = None,
                user_model: Optional[str] = None,
                agent_max_tokens: Optional[int] = None,
                task_retries: Optional[int] = None,
                llm_retries: Optional[int] = None) -> List[str]:
    """``tau2.cli run`` argv — PINNED: the server scripts quote these
    numbers, so any edit here changes what every historical tau2 row means.
    """
    agent_config = {"api_base": v1(base_url), "api_key": "EMPTY", "temperature": 0.0}
    if agent_max_tokens is not None:
        agent_config["max_tokens"] = agent_max_tokens
    if llm_retries is not None:
        agent_config["num_retries"] = llm_retries
    agent_args = json.dumps(agent_config)
    user_config = {"api_base": v1(user_base_url), "api_key": "EMPTY", "temperature": 0.0}
    if llm_retries is not None:
        user_config["num_retries"] = llm_retries
    user_args = json.dumps(user_config)
    cmd = [
        python or sys.executable, "-m", "tau2.cli", "run",
        "--domain", domain_for_task_set(task_set),
        "--task-set-name", task_set,
        "--agent-llm", f"openai/{model}",
        "--agent-llm-args", agent_args,
        "--user-llm", f"openai/{user_model or model}",
        "--user-llm-args", user_args,
        "--max-concurrency", str(num_workers),
        "--save-to", run_name,
        # Headless mode; run_tau2 refuses any nonempty prior simulation dir.
        "--auto-resume",
    ]
    if task_split != "base":
        cmd += ["--task-split-name", task_split]
    if task_ids:
        cmd += ["--task-ids", *task_ids]
    if num_trials is not None:
        cmd += ["--num-trials", str(num_trials)]
    if max_tasks is not None:
        cmd += ["--num-tasks", str(max_tasks)]
    if max_steps is not None:
        cmd += ["--max-steps", str(max_steps)]
    if timeout is not None:
        cmd += ["--timeout", str(timeout)]
    if task_retries is not None:
        cmd += ["--max-retries", str(task_retries),
                "--hallucination-retries", "0"]
    return cmd


def domain_for_task_set(task_set: str) -> str:
    """Map official task-set variants to the corresponding environment."""
    if task_set in {"telecom_small", "telecom_full"}:
        return "telecom"
    return task_set


def parse_task_ids(value: Any) -> Optional[List[str]]:
    if not value:
        return None
    raw = value.split(",") if isinstance(value, str) else list(value)
    ids = [str(item).strip() for item in raw]
    if not ids or any(not item for item in ids) or len(ids) != len(set(ids)):
        raise ValueError("tau2 task IDs must be nonempty and unique")
    return ids


def harness_env(tau2_dir: Optional[Path] = None, out_dir: Optional[Path] = None,
                *, native: bool = False) -> Dict[str, str]:
    env = {**os.environ,
           "NO_PROXY": "127.0.0.1,localhost", "no_proxy": "127.0.0.1,localhost"}
    if tau2_dir is not None:
        # Select the requested checkout, even when another tau2 editable
        # installation happens to be present in the harness interpreter.
        paths = [str(Path(__file__).resolve().parents[1] / "tau2_instrumentation"),
                 str(Path(__file__).resolve().parents[1]),
                 str(Path(tau2_dir).resolve() / "src")]
        env["PYTHONPATH"] = os.pathsep.join(filter(None, (*paths, env.get("PYTHONPATH"))))
    if out_dir is not None:
        env["C2KV_TAU2_TELEMETRY_PATH"] = str(
            Path(out_dir).resolve() / "measurement" / "harness_events.jsonl")
        env["C2KV_TAU2_NATIVE"] = "1" if native else "0"
    else:
        env.pop("C2KV_TAU2_TELEMETRY_PATH", None)
        env.pop("C2KV_TAU2_NATIVE", None)
    return env


def selected_task_ids(tau2_dir: Path, python: str, *,
                      task_set: str = "airline", split: str = "base",
                      task_ids: Optional[List[str]] = None,
                      max_tasks: Optional[int] = None) -> List[str]:
    """Ask the selected official checkout for the exact ordered task IDs."""
    tau2_dir = Path(tau2_dir).resolve()
    if not (tau2_dir / "src" / "tau2").is_dir():
        raise FileNotFoundError(f"tau2 checkout lacks src/tau2: {tau2_dir}")
    requested = parse_task_ids(task_ids)
    if max_tasks is not None and (type(max_tasks) is not int or max_tasks <= 0):
        raise ValueError("tau2 max_tasks must be a positive integer")
    code = (
        "import json,sys; from tau2.run import get_tasks; "
        "set_name,split,ids,limit=json.loads(sys.argv[1]); "
        "print(json.dumps([str(t.id) for t in get_tasks(set_name, "
        "task_split_name=split, task_ids=ids, num_tasks=limit)]))"
    )
    completed = run_owned(
        [str(python), "-c", code,
         json.dumps([task_set, split, requested, max_tasks])],
        cwd=tau2_dir, env=harness_env(tau2_dir),
        check=True, capture_output=True, text=True)
    try:
        ids = json.loads(completed.stdout)
    except (TypeError, json.JSONDecodeError) as error:
        raise RuntimeError("Official tau2 task resolver returned invalid JSON") from error
    if (not isinstance(ids, list) or not ids
            or any(not isinstance(item, str) or not item for item in ids)
            or len(ids) != len(set(ids))):
        raise RuntimeError("Official tau2 task resolver returned no unique task IDs")
    if requested is not None and set(ids) != set(requested):
        raise RuntimeError("Official tau2 task resolver changed the requested task IDs")
    return ids


def evaluate_command(sims: Path, python: Optional[str] = None) -> List[str]:
    """``tau2.cli evaluate-trajs`` argv (writes updated_results.json)."""
    return [python or sys.executable, "-m", "tau2.cli", "evaluate-trajs",
            "-o", str(sims), str(sims / "results.json")]


def run(ctx: RunContext) -> Dict[str, Any]:
    return run_tau2(
        ctx.base_url, ctx.user_base_url, ctx.out_dir,
        tau2_dir=ctx.opt("benchmark_dir") or TAU2_DIR,
        python=ctx.opt("bench_python", sys.executable),
        run_name=ctx.run_name, model=ctx.model,
        task_set=ctx.opt("task_set", "airline"),
        task_split=ctx.opt("tau2_task_split", "base"),
        task_ids=parse_task_ids(ctx.opt("tau2_task_ids")),
        max_tasks=ctx.opt("max_tasks"), num_workers=ctx.opt("num_workers", 4),
        num_trials=ctx.opt("tau2_num_trials", 1),
        max_steps=ctx.opt("tau2_max_steps"), timeout=ctx.opt("tau2_timeout"),
        record_prefixes=ctx.opt("record_prefixes", ""),
        agent_max_tokens=ctx.opt("tau2_agent_max_tokens", 4096),
    )


def _official_tool_schemas(tau2_dir: Path, python: str, domain: str) -> List[Dict[str, Any]]:
    code = (
        "import json,sys; from tau2.registry import registry; "
        "env=registry.get_env_constructor(sys.argv[1])(); "
        "print(json.dumps([t.openai_schema for t in env.get_tools()]))"
    )
    completed = run_owned(
        [str(python), "-c", code, domain], cwd=tau2_dir,
        env=harness_env(tau2_dir), check=True, capture_output=True, text=True)
    try:
        schemas = json.loads(completed.stdout)
    except (TypeError, json.JSONDecodeError) as error:
        raise RuntimeError("Official tau2 tool resolver returned invalid JSON") from error
    if not isinstance(schemas, list) or any(not isinstance(row, dict) for row in schemas):
        raise RuntimeError("Official tau2 tool resolver returned invalid schemas")
    return schemas


def _terminal_results(path: Path, expected_ids: List[str], trials: int,
                      *, require_reward: bool = True,
                      task_failures: Optional[Dict[str, str]] = None,
                      inspect_failures: bool = False) -> List[Dict[str, Any]]:
    if not path.is_file():
        raise RuntimeError(f"tau2 produced no {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("simulations")
    if not isinstance(rows, list):
        raise RuntimeError(f"tau2 results lack simulations: {path}")
    counts = Counter(str(row.get("task_id")) for row in rows if isinstance(row, dict))
    expected = Counter({task_id: trials for task_id in expected_ids})
    if counts != expected:
        raise RuntimeError(f"tau2 terminal-state mismatch: expected={dict(expected)} got={dict(counts)}")
    actual_trials = {(str(row["task_id"]), row.get("trial")) for row in rows}
    expected_trials = {(task_id, trial) for task_id in expected_ids
                       for trial in range(trials)}
    if actual_trials != expected_trials:
        raise RuntimeError("tau2 terminal-state mismatch: missing or repeated task trial")
    for row in rows:
        termination = row.get("termination_reason")
        if not isinstance(termination, str) or not termination:
            raise RuntimeError(f"tau2 task {row['task_id']} has no terminal reason")
        if (termination == "infrastructure_error" and not inspect_failures
                and row["task_id"] not in (task_failures or {})):
            raise RuntimeError(f"tau2 task {row['task_id']} ended with infrastructure_error")
        if require_reward and row["task_id"] not in (task_failures or {}):
            reward = (row.get("reward_info") or {}).get("reward")
            if type(reward) not in (int, float) or not math.isfinite(float(reward)):
                raise RuntimeError(f"tau2 task {row['task_id']} lacks a finite official reward")
    return rows


def _validate_recorded_prefixes(path: Path, expected_ids: List[str]) -> None:
    """A Full replay source must cover every official task with actor wire data."""
    from measurement.telemetry import canonical_sha256, read_jsonl

    if not path.is_file():
        raise RuntimeError(f"tau2 Full run produced no replay source: {path}")
    rows = [row for row in read_jsonl(path)
            if row.get("event_type") == "recorded_prefix"]
    if not rows:
        raise RuntimeError("tau2 Full replay source has no recorded actor prefixes")
    ids = set()
    for row in rows:
        payload = row.get("replay_payload")
        if (row.get("source_arm") != "full" or not isinstance(payload, dict)
                or row.get("canonical_sha256") != canonical_sha256(payload)):
            raise RuntimeError("tau2 Full replay source has an invalid actor prefix")
        identity = payload.get("c2kv_measurement_session_id")
        if not isinstance(identity, str) or not identity:
            raise RuntimeError("tau2 Full actor prefix lacks official task identity")
        ids.add(identity)
    if ids != set(expected_ids):
        raise RuntimeError(
            f"tau2 Full replay task coverage mismatch: expected={expected_ids} "
            f"recorded={sorted(ids)}")


def _validate_harness_events(path: Path, expected_ids: List[str], trials: int = 1,
                             task_failures: Optional[Dict[str, str]] = None) -> None:
    """Every official task trial must execute once with agent decisions."""
    from measurement.telemetry import read_jsonl

    if not path.is_file():
        raise RuntimeError(f"tau2 produced no harness telemetry: {path}")
    rows = list(read_jsonl(path))
    expected = Counter({task_id: trials for task_id in expected_ids})
    starts = Counter(row.get("episode_id") for row in rows
                     if row.get("event_type") == "episode_start")
    ends = Counter(row.get("episode_id") for row in rows
                   if row.get("event_type") == "episode_end")
    decisions = Counter(row.get("episode_id") for row in rows
                        if row.get("event_type") == "decision")
    if starts != expected or ends != expected:
        raise RuntimeError(
            f"tau2 official task retry or missing episode: starts={dict(starts)} "
            f"ends={dict(ends)} expected={dict(expected)}")
    allowed = set(task_failures or {})
    if (set(decisions) != set(expected_ids) or
            any(row.get("status") != "ok" and row.get("episode_id") not in allowed
                for row in rows if row.get("event_type") == "episode_end") or
            any(row.get("error") and row.get("episode_id") not in allowed
                for row in rows if row.get("event_type") == "decision")):
        raise RuntimeError("tau2 official episode had no agent decision or ended in error")


def _declared_task_failures(out_dir: Path, rows: List[Dict[str, Any]],
                            trials: int) -> Dict[str, str]:
    """Only explicit method/budget codes may turn tau2's generic error into a task loss."""
    from measurement.telemetry import read_jsonl

    failed = {str(row["task_id"]) for row in rows
              if row.get("termination_reason") == "infrastructure_error"}
    if not failed:
        return {}
    if trials != 1:
        return {}  # The proxy's task identity cannot disambiguate trials.
    events = out_dir / "measurement" / "harness_events.jsonl"
    if not events.is_file():
        return {}
    errors = {str(row.get("episode_id")): str(row.get("error"))
              for row in read_jsonl(events) if row.get("event_type") == "decision"
              and row.get("error")}
    declared: Dict[str, str] = {}
    # Proxy logs hash measurement_session instead of recording the raw task ID.
    session_ids = {hashlib.sha256(json.dumps(
        ["measurement_session", task], ensure_ascii=False, sort_keys=True,
        separators=(",", ":")).encode("utf-8")).hexdigest(): task
                   for task in failed}
    for path in (out_dir / "logs").glob("proxy_*.jsonl"):
        for row in read_jsonl(path):
            task = session_ids.get(row.get("conv_id"))
            if (task in failed and task in errors
                    and row.get("status") == "acon_history_budget_exceeded"):
                declared[task] = "acon_history_budget_exceeded"
    # Native single-task servers return this exact API code with HTTP 429.
    # Generic 429, HTTP 502 or connection refusal never proves model exhaustion.
    for task in failed:
        if task in errors and "decision_cap_reached" in errors[task]:
            declared[task] = "decision_cap_reached"
    return declared


def run_tau2(base_url: str, user_base_url: str, out_dir: Path, *,
             tau2_dir: Path = TAU2_DIR, python: Optional[str] = None,
             run_name: str = "c2kv_run", model: str = "c2kv-agent",
             user_model: Optional[str] = None, native: bool = False,
             task_set: str = "airline", task_split: str = "base",
             task_ids: Optional[List[str]] = None, max_tasks: Optional[int] = None,
             num_workers: int = 1, num_trials: Optional[int] = 1,
             max_steps: Optional[int] = None, timeout: Optional[int] = None,
             record_prefixes: str = "", agent_max_tokens: int = 4096) -> Dict[str, Any]:
    """Run and score an official tau2 selection through separate agent/user endpoints.

    ``native`` disables proxy-only session metadata for the single-task native
    controller; the official task identity remains bound by its ready manifest.
    """
    tau2_dir = Path(tau2_dir).resolve()
    out_dir = Path(out_dir).resolve()
    python = str(python or sys.executable)
    if not user_base_url:
        raise ValueError("tau2 requires a separate raw user simulator endpoint")
    if type(num_workers) is not int or num_workers <= 0:
        raise ValueError("tau2 num_workers must be positive")
    trials = 1 if num_trials is None else num_trials
    if type(trials) is not int or trials <= 0:
        raise ValueError("tau2 num_trials must be positive")
    if type(agent_max_tokens) is not int or agent_max_tokens <= 0:
        raise ValueError("tau2 agent_max_tokens must be positive")
    if (native or record_prefixes) and trials != 1:
        raise ValueError("tau2 native/replay source runs require exactly one trial")
    selected = selected_task_ids(
        tau2_dir, python, task_set=task_set, split=task_split,
        task_ids=task_ids, max_tasks=max_tasks)
    if native and len(selected) != 1:
        raise ValueError("tau2 native controller requires exactly one selected task")
    sims = tau2_dir / "data" / "simulations" / run_name
    if sims.exists() and any(sims.iterdir()):
        raise FileExistsError(
            f"tau2 run_name already has official artifacts: {sims}; use a new run_name")
    out_dir.mkdir(parents=True, exist_ok=True)
    command = run_command(
        base_url, user_base_url, task_set, model, num_workers, run_name,
        max_tasks=max_tasks, num_trials=trials, max_steps=max_steps,
        timeout=timeout, python=python, task_split=task_split,
        task_ids=selected, user_model=user_model,
        agent_max_tokens=agent_max_tokens,
        task_retries=0, llm_retries=0)
    protocol = {"suite": task_set, "split": task_split, "task_ids": selected,
                "num_trials": trials, "num_workers": num_workers,
                "max_steps": max_steps, "timeout": timeout,
                "agent_max_tokens": agent_max_tokens,
                "agent_model": model, "user_model": user_model or model,
                "agent_base_url": v1(base_url), "user_base_url": v1(user_base_url),
                "source": str(tau2_dir), "python": python, "command": command,
                "native": native, "record_prefixes": str(record_prefixes) if record_prefixes else None}
    (out_dir / "tau2_protocol.json").write_text(
        json.dumps(protocol, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    env = harness_env(tau2_dir, out_dir, native=native)
    run_owned(command, cwd=tau2_dir, env=env, check=True)
    official = out_dir / "official"
    official.mkdir(parents=True, exist_ok=True)
    if (sims / "results.json").is_file():
        shutil.copy2(sims / "results.json", official / "results.json")
    raw_rows = _terminal_results(sims / "results.json", selected, trials,
                                 require_reward=False, inspect_failures=True)
    task_failures = _declared_task_failures(out_dir, raw_rows, trials)
    _terminal_results(sims / "results.json", selected, trials,
                      require_reward=False, task_failures=task_failures)
    _validate_harness_events(
        out_dir / "measurement" / "harness_events.jsonl", selected, trials,
        task_failures=task_failures)
    # Re-scoring consumes saved trajectories; it is not another model episode.
    run_owned(evaluate_command(sims, python=python), cwd=tau2_dir,
              env=harness_env(tau2_dir), check=True)
    updated = sims / "updated_results.json"
    scored_rows = _terminal_results(updated, selected, trials,
                                   task_failures=task_failures)
    raw_terminal = sorted((str(row["task_id"]), str(row.get("trial")),
                           row["termination_reason"]) for row in raw_rows)
    scored_terminal = sorted((str(row["task_id"]), str(row.get("trial")),
                              row["termination_reason"]) for row in scored_rows)
    if raw_terminal != scored_terminal:
        raise RuntimeError("tau2 re-evaluation changed task/trial terminal identity")
    tools = _official_tool_schemas(tau2_dir, python, domain_for_task_set(task_set))
    summary = collect(updated, domain=domain_for_task_set(task_set), tools=tools,
                      task_failures=task_failures)
    if summary["n"] != len(selected) * trials:
        raise RuntimeError("tau2 collected row count differs from selected task-trials")
    if record_prefixes:
        _validate_recorded_prefixes(Path(record_prefixes), selected)
    summary.update(task_ids=selected, protocol=protocol, cost_join=COST_JOIN)
    summary["task_failures"] = task_failures
    shutil.copy2(updated, official / "updated_results.json")
    return summary


COST_JOIN = ("joinable: tau2 instrumentation adds the official task ID to "
             "agent-only proxy requests and emits per-task harness events")


def collect(results_path: Path, domain: str = "airline",
            tools: Optional[List[Dict[str, Any]]] = None,
            task_failures: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Parse a tau2 results.json into unified rows.

    Verified against real trajectory files: simulations[i].messages carry
    role/content/tool_calls (litellm already parsed our server's tool_calls),
    reward_info.reward is the official semantic score.  Protocol columns are
    recomputed with the shared checker against the domain tool pool.
    """
    import sys as _sys

    _sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from metrics import protocol_columns_for_turn  # noqa: E402

    if tools is None:
        tools = []
        try:
            from tau2.registry import registry

            env = registry.get_env_constructor(domain)()
            tools = [tool.openai_schema for tool in env.get_tools()]
        except Exception as error:  # noqa: BLE001 - standalone collection only
            print(f"WARNING: tau2 tool pool unavailable ({error!r}); "
                  "protocol column degrades to unknown", file=sys.stderr)

    data = json.loads(results_path.read_text(encoding="utf-8"))
    rows: List[Dict[str, Any]] = []
    for sim in data.get("simulations") or []:
        turns = [
            protocol_columns_for_turn(m, tools)
            for m in sim.get("messages") or []
            if m.get("role") == "assistant"
        ]
        first_violations = [
            t["first_violation"] for t in turns if t["first_violation"]
        ]
        reward_info = sim.get("reward_info") or {}
        protocol_legal = None
        if any(t["protocol_legal"] is False for t in turns):
            protocol_legal = False
        elif turns and all(t["protocol_legal"] is True for t in turns):
            protocol_legal = True
        task_id = str(sim.get("task_id"))
        failure = (task_failures or {}).get(task_id)
        rows.append(
            {
                "task_id": task_id,
                "trial": sim.get("trial"),
                "semantic_score": 0.0 if failure else reward_info.get("reward"),
                "official_reward": reward_info.get("reward"),
                "task_failure_kind": failure,
                "protocol_legal": protocol_legal,
                "n_turns": len(turns),
                "n_tool_calls": sum(t["n_tool_calls"] for t in turns),
                "n_illegal_turns": sum(t["protocol_legal"] is False for t in turns),
                "n_unknown_protocol_turns": sum(t["protocol_legal"] is None for t in turns),
                "first_violation": first_violations[0] if first_violations else None,
                "termination": sim.get("termination_reason"),
            }
        )
    from metrics import aggregate  # noqa: E402

    summary = aggregate(rows, cluster_key="task_id")
    summary["task_rows"] = rows
    summary["protocol_tool_pool_size"] = len(tools)
    return summary


def _task_tools(traj: Dict[str, Any]) -> List[Dict[str, Any]]:
    tools = traj.get("tools")
    if isinstance(tools, list):
        return tools
    return []


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-dir", type=Path, default=None)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--user-base-url", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--task-set", default="airline")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-tasks", type=int)
    parser.add_argument("--run-name", default="c2kv_run")
    parser.add_argument("--model", default="c2kv-agent")
    args = parser.parse_args()
    # one code path: standalone use builds the same RunContext run.py builds
    summary = run(RunContext(
        base_url=args.base_url, user_base_url=args.user_base_url,
        out_dir=args.out, model=args.model, arm="full", run_name=args.run_name,
        options={"benchmark_dir": args.benchmark_dir, "task_set": args.task_set,
                 "num_workers": args.num_workers, "max_tasks": args.max_tasks},
    ))
    print(json.dumps(summary, indent=2))
