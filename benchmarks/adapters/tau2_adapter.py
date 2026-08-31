"""τ²-bench adapter for the SGLang C2KV benchmark layer."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from metrics import aggregate, protocol_columns_for_turn  # noqa: E402

DEFAULT_TAU2_DIR = Path.home() / "benchmarks" / "tau2"
TAU2_DIR = DEFAULT_TAU2_DIR


def _benchmark_dir(benchmark_dir: Optional[Path]) -> Path:
    configured = benchmark_dir or Path(
        os.environ.get("TAU2_DIR", str(DEFAULT_TAU2_DIR))
    )
    resolved = configured.expanduser().resolve()
    if not (resolved / "pyproject.toml").is_file() or not (resolved / "src" / "tau2").is_dir():
        raise SystemExit(
            f"FATAL: TAU2_DIR is not a tau2 checkout: {resolved}; "
            "set TAU2_DIR or --benchmark-dir"
        )
    return resolved


def run(
    benchmark_dir: Path,
    base_url: str,
    user_base_url: str,
    out_dir: Path,
    task_set: str = "airline",
    num_workers: int = 4,
    max_tasks: Optional[int] = None,
    run_name: str = "c2kv_run",
) -> Dict[str, Any]:
    """Run tau2 with the agent LLM behind the arm proxy.

    The user simulator uses ``user_base_url`` directly. The adapter copies both
    official trajectory outputs into ``out_dir`` so every reported number has a
    stable source file under the matrix result tree.
    """
    cwd = _benchmark_dir(benchmark_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    agent_args = json.dumps(
        {"api_base": base_url.rstrip("/") + "/v1", "api_key": "EMPTY", "temperature": 0.0}
    )
    user_args = json.dumps(
        {"api_base": user_base_url.rstrip("/") + "/v1", "api_key": "EMPTY", "temperature": 0.0}
    )
    env = {
        **os.environ,
        "NO_PROXY": "127.0.0.1,localhost",
        "no_proxy": "127.0.0.1,localhost",
    }
    domain = task_set.split("_")[0]
    sims = cwd / "data" / "simulations" / run_name
    if sims.exists():
        # The official CLI interactively offers to resume an existing results
        # file. This adapter owns deterministic run names and must never mix a
        # rerun with stale trajectories from a previous server/pool state.
        shutil.rmtree(sims)
    cmd = [
        sys.executable, "-m", "tau2.cli", "run",
        "--domain", domain,
        "--task-set-name", task_set,
        "--agent-llm", "openai/c2kv-agent",
        "--agent-llm-args", agent_args,
        "--user-llm", "openai/c2kv-agent",
        "--user-llm-args", user_args,
        "--max-concurrency", str(num_workers),
        "--save-to", run_name,
    ]
    if max_tasks:
        cmd += ["--num-tasks", str(max_tasks)]
    subprocess.run(cmd, cwd=cwd, env=env, check=True)

    subprocess.run(
        [
            sys.executable, "-m", "tau2.cli", "evaluate-trajs",
            "-o", str(sims), str(sims / "results.json"),
        ],
        cwd=cwd,
        env=env,
        check=True,
    )
    updated = sims / "updated_results.json"
    if not updated.is_file():
        raise SystemExit(f"FATAL: tau2 evaluation produced no {updated}")

    results_copy = out_dir / "tau2_results.json"
    updated_copy = out_dir / "tau2_updated_results.json"
    shutil.copy2(sims / "results.json", results_copy)
    shutil.copy2(updated, updated_copy)
    summary = collect(updated_copy, domain=domain)
    summary.update(
        {
            "benchmark": "tau2",
            "task_set": task_set,
            "num_workers": num_workers,
            "official_results": str(results_copy),
            "official_updated_results": str(updated_copy),
        }
    )
    return summary


def collect(results_path: Path, domain: str = "airline") -> Dict[str, Any]:
    """Parse tau2 official results into unified rows."""
    tools: List[Dict[str, Any]] = []
    try:
        import tau2.registry as registry

        environment = registry.get_env_constructor(domain)()
        tools = [tool.openai_schema for tool in environment.tools.get_tools().values()]
    except Exception as error:  # noqa: BLE001 - protocol column can degrade
        print(
            f"WARNING: tau2 tool pool unavailable ({error!r}); protocol column degrades",
            file=sys.stderr,
        )
        tools = []

    data = json.loads(results_path.read_text(encoding="utf-8"))
    rows: List[Dict[str, Any]] = []
    for simulation in data.get("simulations") or []:
        turns = [
            protocol_columns_for_turn(message, tools)
            for message in simulation.get("messages") or []
            if message.get("role") == "assistant"
        ]
        first_violations = [turn["first_violation"] for turn in turns if turn["first_violation"]]
        reward_info = simulation.get("reward_info") or {}
        rows.append(
            {
                "task_id": str(simulation.get("task_id")),
                "semantic_score": reward_info.get("reward"),
                "protocol_legal": all(turn["protocol_legal"] for turn in turns) if turns else None,
                "n_turns": len(turns),
                "n_tool_calls": sum(turn["n_tool_calls"] for turn in turns),
                "n_illegal_turns": len(first_violations),
                "first_violation": first_violations[0] if first_violations else None,
                "termination": simulation.get("termination_reason"),
            }
        )
    return aggregate(rows, cluster_key="task_id")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-dir", type=Path, default=None)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--user-base-url", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--task-set", default="airline")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-tasks", type=int)
    parser.add_argument("--run-name", default="c2kv_run")
    args = parser.parse_args(argv)
    summary = run(
        args.benchmark_dir,
        args.base_url,
        args.user_base_url,
        args.out,
        args.task_set,
        args.num_workers,
        args.max_tasks,
        args.run_name,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
