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

TS_DIR = Path.home() / "benchmarks" / "ToolSandbox"
AGENT = "GPT_4_o_2024_05_13"  # openai_api_agent/openai_api_user role keys


def run(base_url: str, out_dir: Path, test_mode: bool = True,
        agent: str = AGENT, user: str = AGENT, expected: int = None) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    env = {
        **os.environ,
        "OPENAI_API_KEY": "EMPTY",
        "OPENAI_BASE_URL": base_url.rstrip("/") + "/v1",
        "NO_PROXY": "127.0.0.1,localhost", "no_proxy": "127.0.0.1,localhost",
    }
    cmd = ["tool_sandbox", "--user", user, "--agent", agent,
           "-o", str(out_dir)]
    if test_mode:
        cmd.append("-t")
    completed = subprocess.run(cmd, cwd=TS_DIR, env=env)
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
    for path in summaries:
        data = json.loads(path.read_text(encoding="utf-8"))
        for scenario in data.get("per_scenario_results") or []:
            if scenario.get("traceback"):
                rows.append({
                    "task_id": scenario.get("name"),
                    "semantic_score": None,
                    "error": str(scenario.get("exception_type") or "traceback"),
                })
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
    return aggregate(rows, cluster_key="task_id")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--full", action="store_true",
                        help="run the full suite instead of test mode")
    args = parser.parse_args()
    summary = run(args.base_url, args.out, test_mode=not args.full)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
