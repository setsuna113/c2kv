"""Run one official BFCL task through the unchanged experiment-2 interfaces."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import subprocess

from benchmarks.arms import get_arm
from . import runner
from .artifact_io import atomic_json
from .process_lifecycle import run_owned, unwind_on_termination
from .upstream_liveness import UpstreamLiveness


def smoke_command(config, cell, directory, profile, task_id):
    command = runner.run_command(config, cell, directory, profile)
    if runner.is_native_arm(cell["arm"]):
        command += ["--task-ids", task_id]
    else:
        command += ["--run-ids", task_id, "--bfcl-refill-rounds", "0"]
    return command


def scored_result(directory, arm):
    path = directory / f"summary_{arm}.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    if summary.get("n_scored") != 1 or summary.get("semantic_score") is None:
        raise RuntimeError(f"Expected exactly one officially scored task: {path}")
    if summary.get("n_harness_failures", 0):
        raise RuntimeError(f"Harness failure: {path}")
    ledger = summary.get("completion_ledger") or {}
    if ledger.get("remaining") or ledger.get("invalid_rows") or ledger.get("duplicate_rows"):
        raise RuntimeError(f"Incomplete official task ledger: {path}")
    return {"summary": str(path), "n_scored": summary["n_scored"],
            "semantic_score": summary["semantic_score"],
            "n_method_failures": summary.get("n_method_failures", 0),
            "terminal_failures": ledger.get("terminal_failures", {})}


@unwind_on_termination
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--sglang-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arms", default="full,c2kv_native_r4,c2kv_goal_rescue_r8",
                        help="comma-separated configured arms, or all")
    parser.add_argument("--task-id", default="multi_turn_base_26")
    parser.add_argument("--cpu-offload-gb", type=int, default=0,
                        help="optional laptop-only weight offload; the formal runner is unchanged")
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    source = args.sglang_source.resolve()
    output = args.output.resolve()
    candidates = [cell for cell in runner.cells(config) if cell["benchmark"] == "bfcl_base"]
    selected = set(args.arms.split(","))
    if args.arms != "all":
        missing = selected - {cell["arm"] for cell in candidates}
        if missing:
            parser.error(f"Arms are absent from bfcl_base configuration: {sorted(missing)}")
        candidates = [cell for cell in candidates if cell["arm"] in selected]
    if not candidates:
        parser.error("The configuration has no selected BFCL Base cells")
    # Resolve against the actual official split before starting an engine.
    from .c1 import selected_tasks
    selected_tasks(config, "bfcl_base", [args.task_id])
    output.mkdir(parents=True, exist_ok=False)
    config["sglang_source"] = str(source)
    _, profile = runner.prepare(config, output / "plan", source)
    env = dict(os.environ)
    env.update(PYTHONPATH=os.pathsep.join([str(source / "python"), str(runner.ROOT.parent)]),
               BENCH_BFCL_DIR=config["bfcl_dir"],
               C2KV_PAPER_TELEMETRY="1", SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION="0")
    env.setdefault("CUDA_HOME", "/opt/cuda")
    env["PATH"] = str(Path(config["server_python"]).parent) + os.pathsep + env.get("PATH", "")
    receipt = {"qualification": "functional smoke; preliminary, n=1",
               "task_id": args.task_id, "whole_suite_score": False, "cells": []}
    atomic_json(output / "smoke.json", receipt)
    for cell in candidates:
        directory = output / cell["cell_id"]
        directory.mkdir()
        telemetry = ("native_engine_telemetry.jsonl" if runner.is_native_arm(cell["arm"])
                     else "server_telemetry.jsonl")
        env["C2KV_PAPER_TELEMETRY_LOG"] = str(directory / telemetry)
        command = smoke_command(config, cell, directory, profile, args.task_id)
        server_command = runner.server_command(config, source, cell["arm"], "bfcl_base")
        if args.cpu_offload_gb:
            server_command += ["--cpu-offload-gb", str(args.cpu_offload_gb)]
        record = {"cell_id": cell["cell_id"], "arm": cell["arm"],
                  "command": command, "server_command": server_command, "status": "running"}
        receipt["cells"].append(record)
        atomic_json(output / "smoke.json", receipt)
        for port in (config["server_port"], config["proxy_port"]):
            with socket.socket() as probe:
                if probe.connect_ex(("127.0.0.1", port)) == 0:
                    raise RuntimeError(f"Configured port {port} is occupied")
        with (directory / "server.log").open("x") as log:
            server = subprocess.Popen(server_command, env=env, stdout=log,
                                      stderr=subprocess.STDOUT, start_new_session=True)
            try:
                runner.wait_server(server, config["server_port"])
                if get_arm(cell["arm"]).text_history_budget_tokens is not None:
                    runner.require_budget_renderer(config["server_port"])
                with (directory / "driver.log").open("x") as driver_log:
                    run_owned(command, check=True, env=env, cwd=runner.ROOT.parent,
                              stdout=driver_log, stderr=subprocess.STDOUT,
                              monitor=UpstreamLiveness(
                                  f"http://127.0.0.1:{config['server_port']}", process=server))
                record.update(status="completed", **scored_result(directory, cell["arm"]))
            except BaseException as error:
                record.update(status="failed", error=str(error))
                raise
            finally:
                runner.cleanup_cell_processes(None, server)
                atomic_json(output / "smoke.json", receipt)
        print(json.dumps({"cell_id": cell["cell_id"], "status": record["status"]}), flush=True)


if __name__ == "__main__":
    main()
