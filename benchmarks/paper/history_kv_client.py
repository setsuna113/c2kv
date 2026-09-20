"""Run one configured paper history-KV budget cell against an owned upstream.

This client owns only the benchmark/proxy process. The caller owns the engine.
The paper runner remains the source of budget, cell, and adapter commands.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

from benchmarks.history_budget import parse_history_kv_budget
from . import runner
from .artifact_io import atomic_json
from .process_lifecycle import run_owned, unwind_on_termination


PAPER_ROOT = Path(__file__).resolve().parents[2]


def plan(config: dict, benchmark: str, budget_spec: str, output: Path,
         upstream: str, proxy_port: int | None = None,
         checkpoint_profile: Path | None = None,
         bench_python: str | None = None) -> tuple[dict, dict, list[str], Path]:
    """Select one raw-tool closed-loop cell using the shared paper matrix."""
    arm, tokens = parse_history_kv_budget(budget_spec)
    if not upstream.startswith(("http://", "https://")):
        raise ValueError("--upstream must be an HTTP(S) base URL")
    if proxy_port is not None and not 1 <= proxy_port <= 65535:
        raise ValueError("--proxy-port must be in 1..65535")
    resolved = runner.with_history_kv_budget(config, arm, tokens)
    if bench_python:
        resolved = dict(resolved, bench_python=bench_python)
    if proxy_port is not None:
        resolved = dict(resolved, proxy_port=proxy_port)
    selected = [cell for cell in runner.cells(resolved)
                if cell["arm"] == arm and cell["benchmark"] == benchmark
                and cell.get("history_budget_tokens") == tokens
                and cell["tool_context"] == "raw"]
    if len(selected) != 1:
        raise ValueError(f"Expected one configured raw-tool cell for {benchmark}__{arm}_b{tokens}")
    cell = selected[0]
    if runner._unsupported_stage("closed_loop", cell):
        raise ValueError(f"Unsupported closed_loop cell: {cell['cell_id']}")
    directory = output / "closed_loop" / cell["cell_id"]
    profile = checkpoint_profile or output / "deployment_profile.json"
    command = runner.run_command(resolved, cell, directory, profile, "closed_loop")
    command[command.index("--upstream") + 1] = upstream.rstrip("/")
    command.append("--shared-engine")
    if checkpoint_profile is None:
        index = command.index("--checkpoint-profile")
        del command[index:index + 2]  # Let benchmarks.run discover the checkpoint's profile.
    return resolved, cell, command, directory


@unwind_on_termination
def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True,
                        help="paper config with paths for the existing deployment")
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--history-kv-budget", required=True, metavar="ARM=TOKENS")
    parser.add_argument("--upstream", required=True,
                        help="already owned engine base URL; this client never starts it")
    parser.add_argument("--proxy-port", type=int)
    parser.add_argument("--out", type=Path, required=True, help="dedicated output root")
    parser.add_argument("--checkpoint-profile", type=Path)
    parser.add_argument("--bench-python", help="override the config's benchmark interpreter")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        config = json.loads(args.config.read_text(encoding="utf-8"))
        output = args.out.resolve()
        resolved, cell, command, directory = plan(
            config, args.benchmark, args.history_kv_budget, output, args.upstream,
            args.proxy_port, args.checkpoint_profile.resolve() if args.checkpoint_profile else None,
            args.bench_python)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    record = {"cell": cell, "command": command, "config": resolved,
              "paper_root": str(PAPER_ROOT), "upstream": args.upstream.rstrip("/"),
              "checkpoint_profile": (str(args.checkpoint_profile.resolve())
                                     if args.checkpoint_profile else None)}
    if args.dry_run:
        print(json.dumps({"cell_id": cell["cell_id"], "stage": "closed_loop",
                          "directory": str(directory), "command": command,
                          "paper_root": str(PAPER_ROOT)}, indent=2))
        return
    if args.checkpoint_profile and not args.checkpoint_profile.is_file():
        parser.error(f"Missing checkpoint profile: {args.checkpoint_profile}")
    manifest = output / "client_manifest.json"
    if manifest.exists():
        if json.loads(manifest.read_text(encoding="utf-8")) != record:
            raise RuntimeError(f"Existing output has a different resolved cell: {manifest}")
    elif output.exists() and any(output.iterdir()):
        raise RuntimeError(f"Existing non-empty output has no resolved cell: {output}")
    if (directory / "complete.json").exists():
        return
    if (directory / "started.json").exists():
        raise RuntimeError(f"Partial cell {directory}; inspect it before selecting a new output")
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(manifest, record)
    atomic_json(output / "config.resolved.json", resolved)
    atomic_json(output / "commands.json", [dict(cell, command=command)])
    directory.mkdir(parents=True, exist_ok=True)
    atomic_json(directory / "started.json", {**record, "stage": "closed_loop",
                                             "time": time.time()}, exclusive=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(PAPER_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env["BENCH_BFCL_DIR"] = resolved["bfcl_dir"]
    if resolved.get("appworld_root"):
        env["APPWORLD_ROOT"] = resolved["appworld_root"]
    env["C2KV_PAPER_TELEMETRY"] = "1"
    run_owned(command, cwd=PAPER_ROOT, env=env, check=True)
    atomic_json(directory / "complete.json", {"finished_at": time.time()})


if __name__ == "__main__":
    main()
