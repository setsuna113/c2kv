"""ToolSandbox adapter (official-CLI driver).

The ToolSandbox Python API (Scenario/play/play_and_evaluate) does not expose
a Scenario.discover(); the supported entrypoint is the ``tool_sandbox`` CLI,
which runs scenarios against an OpenAI-compatible endpoint configured via
``OPENAI_BASE_URL`` and writes ``result_summary.json`` per run.  This
adapter drives that CLI and parses the summaries.

Verified metrics fields (per scenario): similarity / milestone_similarity /
minefield_similarity / turn_count.  There is no "main_acc".

Usage (benchts venv on the server):
    python benchmarks/adapters/toolsandbox_adapter.py \
        --base-url http://127.0.0.1:34002/v1 --out results/bench/ts_full
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

TS_DIR = Path(os.environ.get("TS_DIR") or Path.home() / "benchmarks" / "ToolSandbox")
AGENT = "GPT_4_o_2024_05_13"  # openai_api_agent/openai_api_user role keys


def run(base_url: str, out_dir: Path, test_mode: bool = True,
        agent: str = AGENT, user: str = AGENT, expected: int = None,
        benchmark_dir: Path = None, user_base_url: str = "") -> Dict[str, Any]:
    """``user_base_url`` (default: the raw upstream endpoint) routes the
    user simulator OUT of the arm proxy via TOOLSANDBOX_USER_BASE_URL —
    the patched openai_api_user role reads it.  Routing the simulator
    through the compression arm made every historical TS number an
    agent+user joint degradation (audit BLOCKER)."""
    ts_dir = Path(benchmark_dir) if benchmark_dir else TS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    env = {
        **os.environ,
        "OPENAI_API_KEY": "EMPTY",
        "OPENAI_API_KEY_USER": "EMPTY",
        "OPENAI_BASE_URL": base_url.rstrip("/") + "/v1",
        # default: same endpoint the proxy itself fronts (full mode)
        "TOOLSANDBOX_USER_BASE_URL": (user_base_url.rstrip("/") + "/v1") if user_base_url
        else os.environ.get("TOOLSANDBOX_USER_BASE_URL",
                            base_url.rstrip("/") + "/v1"),
        "NO_PROXY": "127.0.0.1,localhost", "no_proxy": "127.0.0.1,localhost",
    }
    cmd = ["tool_sandbox", "--user", user, "--agent", agent,
           "-o", str(out_dir)]
    if test_mode:
        cmd.append("-t")
    completed = subprocess.run(cmd, cwd=ts_dir, env=env)
    if completed.returncode != 0:
        raise SystemExit(f"FATAL: tool_sandbox CLI exited {completed.returncode}")
    summary = collect(out_dir)
    # terminal-state check (acceptance 1): a scenario that never ran must
    # fail the run, not shrink the denominator
    n_scored = summary.get("n") if isinstance(summary, dict) else None
    if expected is not None and n_scored is not None and n_scored < expected:
        raise SystemExit(
            f"FATAL: ts terminal-state check failed: n_scored={n_scored} < n_total={expected}")
    if expected is not None:
        print(f"TERMINAL-STATE ts: n_scored={n_scored} n_total={expected}")
    return summary


def collect(out_dir: Path) -> Dict[str, Any]:
    summaries = sorted(out_dir.glob("agent_*/result_summary.json"))
    if not summaries:
        raise SystemExit(f"FATAL: no result_summary.json under {out_dir} — "
                         "the CLI produced nothing (check TS run logs)")
    from metrics import aggregate  # noqa: E402

    rows: List[Dict[str, Any]] = []
    crashed: List[str] = []
    for path in summaries:
        data = json.loads(path.read_text(encoding="utf-8"))
        for scenario in data.get("per_scenario_results") or []:
            if scenario.get("traceback"):
                # a crashed scenario FAILS the run — _mean used to skip
                # these None rows and the upstream recorded a silent 0
                crashed.append(str(scenario.get("name")))
                continue
            rows.append({
                "task_id": scenario.get("name"),
                # official semantic column: dialogue similarity to the
                # reference (milestone-weighted); minefield = violations
                "semantic_score": scenario.get("similarity"),
                "milestone_similarity": scenario.get("milestone_similarity"),
                "minefield_similarity": scenario.get("minefield_similarity"),
                "turn_count": scenario.get("turn_count"),
                "protocol_legal": None,  # TS has no tool-call legality metric
            })
    if crashed:
        raise SystemExit(
            f"FATAL: ts terminal-state check failed: {len(crashed)} scenario(s) "
            f"crashed (traceback in result_summary): {', '.join(crashed[:10])}")
    return aggregate(rows, cluster_key="task_id")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--full", action="store_true",
                        help="run the full suite instead of test mode")
    parser.add_argument("--agent", default=AGENT,
                        help="agent role key (openai_api_agent config entry)")
    parser.add_argument("--user", default=AGENT,
                        help="user-simulator role key (openai_api_user config entry)")
    parser.add_argument("--ts-dir", type=Path, default=None,
                        help="ToolSandbox checkout (default $TS_DIR or ~/benchmarks/ToolSandbox)")
    args = parser.parse_args()
    summary = run(args.base_url, args.out, test_mode=not args.full,
                  agent=args.agent, user=args.user, benchmark_dir=args.ts_dir)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
