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
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from metrics import aggregate, protocol_columns_for_turn  # noqa: E402

TAU2_DIR = Path.home() / "benchmarks" / "tau2"


def run(
    benchmark_dir: Path,
    base_url: str,
    user_base_url: str,
    out_dir: Path,
    task_set: str = "airline",
    num_workers: int = 4,
    max_tasks: Optional[int] = None,
) -> Dict[str, Any]:
    """Run tau2 with the agent LLM behind the arm proxy.

    The user simulator points at a separate full-mode endpoint (same served
    model, no compression) so only the agent's context is ever compressed.
    """
    import os

    out_dir.mkdir(parents=True, exist_ok=True)
    agent_args = json.dumps(
        {"api_base": base_url.rstrip("/") + "/v1", "api_key": "EMPTY", "temperature": 0.0}
    )
    user_args = json.dumps(
        {"api_base": user_base_url.rstrip("/") + "/v1", "api_key": "EMPTY", "temperature": 0.0}
    )
    env = {**os.environ, "NO_PROXY": "127.0.0.1,localhost", "no_proxy": "127.0.0.1,localhost"}
    cmd = [
        sys.executable, "-m", "tau2.cli", "run",
        "--domain", task_set.split("_")[0],
        "--task-set-name", task_set,
        "--agent-llm", "openai/c2kv-agent",
        "--agent-llm-args", agent_args,
        "--user-llm", "openai/c2kv-agent",
        "--user-llm-args", user_args,
        "--max-concurrency", str(num_workers),
        "--save-to", str(out_dir / "trajectories"),
    ]
    if max_tasks:
        cmd += ["--num-tasks", str(max_tasks)]
    subprocess.run(cmd, cwd=TAU2_DIR, env=env, check=True)
    return collect(out_dir)


def collect(out_dir: Path) -> Dict[str, Any]:
    """Parse trajectories into unified rows.  Rewritten on first live run —
    field names are pinned after inspecting one real trajectory file."""
    rows: List[Dict[str, Any]] = []
    traj_dir = out_dir / "trajectories"
    for path in sorted(traj_dir.glob("*.json*")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        reward = data.get("reward")
        task_id = data.get("task_id") or path.stem
        tools = _task_tools(data)
        turns = [
            protocol_columns_for_turn(m, tools)
            for m in data.get("messages") or []
            if m.get("role") == "assistant"
        ]
        rows.append(
            {
                "task_id": task_id,
                "semantic_score": float(reward) if reward is not None else None,
                "protocol_legal": all(t["protocol_legal"] for t in turns) if turns else None,
                "n_turns": len(turns),
            }
        )
    return aggregate(rows, cluster_key="task_id")


def _task_tools(traj: Dict[str, Any]) -> List[Dict[str, Any]]:
    tools = traj.get("tools")
    if isinstance(tools, list):
        return tools
    return []


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-dir", type=Path, default=TAU2_DIR)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--user-base-url", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--task-set", default="airline")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-tasks", type=int)
    args = parser.parse_args()
    summary = run(
        args.benchmark_dir, args.base_url, args.user_base_url, args.out,
        args.task_set, args.num_workers, args.max_tasks,
    )
    print(json.dumps(summary, indent=2))
