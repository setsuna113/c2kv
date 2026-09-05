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

from adapters.base import RunContext, v1  # noqa: E402

NAME = "toolsandbox"
TS_DIR = Path(os.environ.get("TS_DIR") or Path.home() / "benchmarks" / "ToolSandbox")
AGENT = "GPT_4_o_2024_05_13"  # openai_api_agent/openai_api_user role keys


def add_arguments(parser) -> None:
    """ToolSandbox-only CLI flags (shared ones live in run.py's core block)."""
    parser.add_argument("--full", action="store_true",
                        help="toolsandbox: full suite instead of test mode")
    parser.add_argument("--ts-scenarios", default="",
                        help="toolsandbox: comma-separated scenario names "
                             "for subset runs (-s); overrides --full")
    parser.add_argument("--ts-agent", default="",
                        help="toolsandbox: agent role key (default "
                             "GPT_4_o_2024_05_13 -> openai_api_agent)")
    parser.add_argument("--ts-user", default="",
                        help="toolsandbox: user-simulator role key (same default)")


def cli_command(out_dir: Path, agent: str = AGENT, user: str = AGENT,
                test_mode: bool = True,
                scenarios: "list[str] | None" = None,
                parallel: "str | None" = None) -> List[str]:
    """``tool_sandbox`` argv (PINNED).  ``-s names...`` is the subset form
    (the CLI also takes ``-p`` for parallelism, from $TS_PARALLEL); ``-t``
    is test mode, and a subset run overrides it."""
    cmd = ["tool_sandbox", "--user", user, "--agent", agent, "-o", str(out_dir)]
    if scenarios:
        cmd += ["-s"] + list(scenarios)
        if parallel:
            cmd += ["-p", parallel]
    elif test_mode:
        cmd.append("-t")
    return cmd


def split_scenarios(raw) -> "list[str] | None":
    """``--ts-scenarios a,b`` -> ["a", "b"]; empty -> None (full/test mode)."""
    if not raw:
        return None
    items = ([s.strip() for s in raw.split(",")] if isinstance(raw, str)
             else [str(s).strip() for s in raw])
    return [s for s in items if s] or None


def harness_env(base_url: str, user_base_url: str = "") -> Dict[str, str]:
    """``user_base_url`` (default: the raw upstream endpoint) routes the
    user simulator OUT of the arm proxy via TOOLSANDBOX_USER_BASE_URL —
    the patched openai_api_user role reads it.  Routing the simulator
    through the compression arm made every historical TS number an
    agent+user joint degradation (audit BLOCKER)."""
    return {
        **os.environ,
        "OPENAI_API_KEY": "EMPTY",
        "OPENAI_API_KEY_USER": "EMPTY",
        "OPENAI_BASE_URL": v1(base_url),
        # default: same endpoint the proxy itself fronts (full mode)
        "TOOLSANDBOX_USER_BASE_URL": v1(user_base_url) if user_base_url
        else os.environ.get("TOOLSANDBOX_USER_BASE_URL", v1(base_url)),
        "NO_PROXY": "127.0.0.1,localhost", "no_proxy": "127.0.0.1,localhost",
    }


def run(ctx: RunContext) -> Dict[str, Any]:
    """Drive the official ``tool_sandbox`` CLI against the arm proxy.

    No cost join: see ``COST_JOIN`` below.
    """
    summary = run_ts(
        ctx.base_url, ctx.out_dir,
        test_mode=not ctx.options.get("full", False),
        agent=ctx.opt("ts_agent", AGENT), user=ctx.opt("ts_user", AGENT),
        # the user simulator must NOT ride the arm proxy: route it to the
        # raw upstream endpoint (tau2 already does the same split)
        user_base_url=ctx.user_base_url,
        scenarios=split_scenarios(ctx.opt("ts_scenarios", "")),
    )
    summary["cost_join"] = COST_JOIN
    return summary


# Why ToolSandbox gets no per-task cost columns: ``result_summary.json`` is
# the only artefact this adapter reads and it carries per-scenario SCORES
# (similarity / milestone_similarity / minefield_similarity / turn_count,
# tool_sandbox/cli/utils.py:196-208), no messages — so nothing here can
# rebuild the message prefix ``proxy.conversation_id`` keys on.
COST_JOIN = ("not joinable: result_summary.json holds per-scenario scores "
             "only, no messages to key the request log by")


def run_ts(base_url: str, out_dir: Path, test_mode: bool = True,
           agent: str = AGENT, user: str = AGENT, expected: int = None,
           benchmark_dir: Path = None, user_base_url: str = "",
           scenarios: "list[str] | None" = None) -> Dict[str, Any]:
    """Run the CLI and collect ``result_summary.json``."""
    ts_dir = Path(benchmark_dir) if benchmark_dir else TS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    env = harness_env(base_url, user_base_url)
    cmd = cli_command(out_dir, agent=agent, user=user, test_mode=test_mode,
                      scenarios=scenarios,
                      parallel=os.environ.get("TS_PARALLEL"))
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
    summary = run_ts(args.base_url, args.out, test_mode=not args.full,
                     agent=args.agent, user=args.user, benchmark_dir=args.ts_dir)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
