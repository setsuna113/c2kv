"""ToolSandbox adapter driven by the official CLI."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DEFAULT_TS_DIR = Path.home() / "benchmarks" / "ToolSandbox"
TS_DIR = DEFAULT_TS_DIR
AGENT = "GPT_4_o_2024_05_13"  # openai_api_agent/openai_api_user role keys


def _benchmark_dir() -> Path:
    resolved = Path(os.environ.get("TOOLSANDBOX_DIR", str(DEFAULT_TS_DIR))).expanduser().resolve()
    if not (resolved / "pyproject.toml").is_file() or not (resolved / "tool_sandbox").is_dir():
        raise SystemExit(f"FATAL: TOOLSANDBOX_DIR is not a ToolSandbox checkout: {resolved}")
    return resolved


def run(
    base_url: str,
    out_dir: Path,
    test_mode: bool = True,
    agent: str = AGENT,
    user: str = AGENT,
    scenarios: Optional[Sequence[str]] = None,
    num_workers: int = 4,
) -> Dict[str, Any]:
    """Run ToolSandbox and collect its official per-scenario similarities."""
    cwd = _benchmark_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    selected = list(scenarios or [])
    if selected and not test_mode:
        raise SystemExit("FATAL: ToolSandbox --full cannot be combined with explicit scenarios")

    env = {
        **os.environ,
        # run.py is invoked with an absolute benchmark-venv Python, so make
        # that venv's console script visible without depending on the caller's
        # activated PATH.
        "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", ""),
        "OPENAI_API_KEY": "EMPTY",
        "OPENAI_BASE_URL": base_url.rstrip("/") + "/v1",
        "NO_PROXY": "127.0.0.1,localhost",
        "no_proxy": "127.0.0.1,localhost",
    }
    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parents[1] / "toolsandbox_cli.py"),
        "--user", user,
        "--agent", agent,
        "--parallel", str(num_workers),
        "-o", str(out_dir.resolve()),
    ]
    if selected:
        cmd += ["--scenarios", *selected]
    elif test_mode:
        cmd.append("--test_mode")

    completed = subprocess.run(cmd, cwd=cwd, env=env)
    if completed.returncode != 0:
        raise SystemExit(f"FATAL: tool_sandbox CLI exited {completed.returncode}")
    summary = collect(out_dir)
    if all(row.get("semantic_score") is None for row in summary.get("rows", [])):
        raise SystemExit("FATAL: every ToolSandbox scenario failed before scoring")
    summary.update(
        {
            "benchmark": "toolsandbox",
            "num_workers": num_workers,
            "test_mode": bool(test_mode and not selected),
            "selected_scenarios": selected,
            "error_count": sum(row.get("semantic_score") is None for row in summary.pop("rows", [])),
        }
    )
    return summary


def collect(out_dir: Path) -> Dict[str, Any]:
    summaries = sorted(out_dir.glob("agent_*/result_summary.json"))
    if not summaries:
        raise SystemExit(
            f"FATAL: no result_summary.json under {out_dir} - the official CLI produced nothing"
        )
    from metrics import aggregate  # noqa: E402

    rows: List[Dict[str, Any]] = []
    official_sources: List[str] = []
    for path in summaries:
        official_sources.append(str(path))
        data = json.loads(path.read_text(encoding="utf-8"))
        for scenario in data.get("per_scenario_results") or []:
            if scenario.get("traceback"):
                rows.append(
                    {
                        "task_id": scenario.get("name"),
                        "semantic_score": None,
                        "error": str(scenario.get("exception_type") or "traceback"),
                    }
                )
                continue
            rows.append(
                {
                    "task_id": scenario.get("name"),
                    "semantic_score": scenario.get("similarity"),
                    "milestone_similarity": scenario.get("milestone_similarity"),
                    "minefield_similarity": scenario.get("minefield_similarity"),
                    "turn_count": scenario.get("turn_count"),
                    "protocol_legal": None,
                }
            )
    summary = aggregate(rows, cluster_key="task_id")
    summary["rows"] = rows
    summary["official_results"] = official_sources
    return summary


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--scenarios", nargs="*", default=None)
    args = parser.parse_args(argv)
    summary = run(
        args.base_url,
        args.out,
        test_mode=not args.full,
        scenarios=args.scenarios,
        num_workers=args.num_workers,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
