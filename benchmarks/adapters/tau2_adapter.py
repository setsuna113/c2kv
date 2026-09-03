"""τ²-bench adapter.

tau2 talks to the agent LLM through LiteLLM, so the arm proxy is plugged in
purely by configuration: an OpenAI-compatible provider entry whose api_base
is the proxy.  The user simulator and any judge calls go to a *separate*
full-mode endpoint (we compress only the agent's view, never the user
simulator, to keep the benchmark semantics intact).

Run recipe (see benchmarks/README.md):
  1. proxy in the requested arm on port P (agent endpoint)
  2. write a litellm provider block + settings JSON
  3. `tau2 run` over the requested task set
  4. `tau2 evaluate-trajs` for the official reward
  5. this adapter parses trajectories + the proxy request log into unified
     rows (per task: official reward as semantic column; protocol columns
     recomputed from raw assistant turns with our shared checker)
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from metrics import aggregate, protocol_columns_for_turn  # noqa: E402

TAU2_DIR = Path(os.environ.get("TAU2_DIR") or Path.home() / "benchmarks" / "tau2")


def run(
    benchmark_dir: Optional[Path],
    base_url: str,
    user_base_url: str,
    out_dir: Path,
    task_set: str = "airline",
    num_workers: int = 4,
    max_tasks: Optional[int] = None,
    run_name: str = "c2kv_run",
    model: str = "c2kv-agent",
) -> Dict[str, Any]:
    """Run tau2 with the agent LLM behind the arm proxy.

    The user simulator points at a separate full-mode endpoint (same served
    model, no compression) so only the agent's context is ever compressed.
    ``benchmark_dir`` (or $TAU2_DIR, or ~/benchmarks/tau2) is the tau2
    checkout that provides both the CLI and the tool registry.
    Official semantics: ``--save-to NAME`` writes
    ``<tau2_dir>/data/simulations/NAME/results.json``; rewards are computed
    by ``tau2 evaluate-trajs`` into ``updated_results.json``, which collect
    then reads.
    """
    tau2_dir = Path(benchmark_dir) if benchmark_dir else TAU2_DIR

    agent_args = json.dumps(
        {"api_base": base_url.rstrip("/") + "/v1", "api_key": "EMPTY", "temperature": 0.0}
    )
    user_args = json.dumps(
        {"api_base": user_base_url.rstrip("/") + "/v1", "api_key": "EMPTY", "temperature": 0.0}
    )
    env = {**os.environ,
           "NO_PROXY": "127.0.0.1,localhost", "no_proxy": "127.0.0.1,localhost"}
    cmd = [
        sys.executable, "-m", "tau2.cli", "run",
        "--domain", task_set.split("_")[0],
        "--task-set-name", task_set,
        "--agent-llm", f"openai/{model}",
        "--agent-llm-args", agent_args,
        "--user-llm", f"openai/{model}",
        "--user-llm-args", user_args,
        "--max-concurrency", str(num_workers),
        "--save-to", run_name,
    ]
    if max_tasks:
        cmd += ["--num-tasks", str(max_tasks)]
    subprocess.run(cmd, cwd=tau2_dir, env=env, check=True)
    sims = tau2_dir / "data" / "simulations" / run_name
    subprocess.run(
        [sys.executable, "-m", "tau2.cli", "evaluate-trajs", "-o", str(sims),
         str(sims / "results.json")],
        cwd=tau2_dir, env=env, check=True,
    )
    updated = sims / "updated_results.json"
    if not updated.exists():
        raise SystemExit(f"FATAL: tau2 evaluation produced no {updated}")
    # terminal-state gate via the shared checker: infra-error simulations
    # are NOT valid terminal states (the old inline len(sims) check counted
    # them as scored)
    import terminal_check  # noqa: E402  (sibling module)

    code = terminal_check.check_tau2(run_name, max_tasks or None)
    if code != 0:
        raise SystemExit(f"FATAL: tau2 terminal-state check failed (rc={code})")
    return collect(updated, domain=task_set.split("_")[0])


def collect(results_path: Path, domain: str = "airline") -> Dict[str, Any]:
    """Parse a tau2 results.json into unified rows.

    Verified against real trajectory files: simulations[i].messages carry
    role/content/tool_calls (litellm already parsed our server's tool_calls),
    reward_info.reward is the official semantic score.  Protocol columns are
    recomputed with the shared checker against the domain tool pool.
    """
    import sys as _sys

    _sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from metrics import protocol_columns_for_turn  # noqa: E402

    tools: List[Dict[str, Any]] = []
    try:
        import tau2.registry as registry

        env = registry.get_env_constructor(domain)()
        tools = [
            tool.openai_schema for tool in env.tools.get_tools().values()
        ]
    except Exception as error:  # noqa: BLE001 - protocol column degrades
        print(f"WARNING: tau2 tool pool unavailable ({error!r}); "
              "protocol column degrades to unknown", file=sys.stderr)
        tools = []

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
        rows.append(
            {
                "task_id": str(sim.get("task_id")),
                "semantic_score": reward_info.get("reward"),
                "protocol_legal": all(t["protocol_legal"] for t in turns) if turns else None,
                "n_turns": len(turns),
                "n_tool_calls": sum(t["n_tool_calls"] for t in turns),
                "n_illegal_turns": len(first_violations),
                "first_violation": first_violations[0] if first_violations else None,
                "termination": sim.get("termination_reason"),
            }
        )
    from metrics import aggregate  # noqa: E402

    return aggregate(rows, cluster_key="task_id")


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
    parser.add_argument("--model", default="c2kv-agent")
    args = parser.parse_args()
    summary = run(
        args.benchmark_dir, args.base_url, args.user_base_url, args.out,
        task_set=args.task_set, num_workers=args.num_workers,
        max_tasks=args.max_tasks, model=args.model,
    )
    print(json.dumps(summary, indent=2))
