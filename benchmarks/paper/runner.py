"""Prepare or execute the fixed paper matrix and measured common-prefix replay."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = Path(__file__).with_name("config.json")


def cells(config):
    rows = [dict(method, benchmark=bench["name"], adapter=bench["adapter"],
                 category=bench.get("category", ""),
                 cell_id=bench["name"] + "__" + method["arm"])
            for bench in config["benchmarks"] for method in config["methods"]]
    # Run the final system after every existing comparison/sweep cell.
    return sorted(rows, key=lambda row: row["arm"] == "c2kv_c1_t02_r8")


def server_command(config, source, arm=None):
    """CUDA server flags for one cell.

    Single-flight serving (one running request, one worker, no overlap
    schedule) is what the per-request telemetry attributes peaks to; it is a
    measurement constraint, not an algorithm requirement. The attention
    backend, CUDA graph, and radix cache are execution-path choices:
    ``disable_cuda_graph`` (default true) and ``radix_cache_arms`` (arms that
    serve with SGLang's cross-request prefix cache; default none, ``"*"`` for
    every arm) live in the config so the resolved config records exactly what
    each cell ran with.
    """
    cmd = [config["server_python"], "-m", "sglang.launch_server",
           "--model-path", config["checkpoint"], "--served-model-name", config["model"],
           "--device", "cuda", "--dtype", "bfloat16", "--model-impl", "sglang",
           "--attention-backend", config["attention_backend"],
           "--tool-call-parser", "qwen25",
           "--enable-c2kv", "--c2kv-gist-type", "dynamic-interleave",
           "--c2kv-gist-param", "qkv", "--c2kv-query-proj", "base",
           "--c2kv-pool-fraction", str(config["c2kv_pool_fraction"]),
           "--mem-fraction-static", str(config["mem_fraction_static"]),
           "--context-length", str(config["context_length"]),
           "--max-total-tokens", str(config["max_total_tokens"]),
           "--max-running-requests", "1", "--page-size", "1",
           "--chunked-prefill-size", str(config["chunked_prefill_size"]),
           "--random-seed", str(config["seed"])]
    radix_arms = set(config.get("radix_cache_arms") or ())
    if "*" not in radix_arms and arm not in radix_arms:
        cmd.append("--disable-radix-cache")
    if config.get("disable_cuda_graph", True):
        cmd.append("--disable-cuda-graph")
    cmd += ["--disable-piecewise-cuda-graph", "--disable-overlap-schedule",
            "--enable-streaming-session", "--host", "127.0.0.1",
            "--port", str(config["server_port"])]
    if arm == "c2kv_c1_t02_r8":
        cmd += ["--c2kv-shadow-feature-layer", "-2", "--enable-return-hidden-states"]
    return cmd


def with_port_offset(config, port_offset):
    """Runtime port shift for running several cells on one host (one GPU each).

    The offset is not part of the prepared/resolved config: it changes only
    which local ports a cell's server and proxy bind, never what is measured.
    """
    if not port_offset:
        return config
    return dict(config, server_port=config["server_port"] + port_offset,
                proxy_port=config["proxy_port"] + port_offset)


def run_command(config, cell, directory, profile, stage="closed_loop"):
    if cell["arm"] == "c2kv_c1_t02_r8":
        cmd = [config["bench_python"], "-m", "benchmarks.paper.c1",
               "--config", str(profile.parent / "config.resolved.json"),
               "--benchmark", cell["benchmark"], "--stage", stage,
               "--upstream", f"http://127.0.0.1:{config['server_port']}",
               "--proxy-port", str(config["proxy_port"]),
               "--out", str(directory), "--num-workers", "1"]
        if stage == "common_prefix":
            cmd += ["--prefixes", str(profile.parent / "closed_loop" /
                                      (cell["benchmark"] + "__full") / "full_prefixes.jsonl")]
        return cmd
    cmd = [config["bench_python"], str(ROOT / "run.py"),
           "--benchmark", cell["adapter"], "--arm", cell["arm"],
           "--upstream", f"http://127.0.0.1:{config['server_port']}",
           "--proxy-port", str(config["proxy_port"]), "--backend", "sglang",
           "--model", config["model"], "--checkpoint", config["checkpoint"],
           "--checkpoint-profile", str(profile), "--out", str(directory),
           "--exact-out", "--run-name", cell["cell_id"], "--num-workers", "1",
           "--telemetry-log", str(directory / "proxy_telemetry.jsonl"),
           "--capability-features", "hiagent_trajectory_retrieval_v1"]
    if cell["arm"] == "full":
        cmd += ["--record-prefixes", str(directory / "full_prefixes.jsonl")]
    if cell["adapter"] == "bfcl":
        cmd += ["--categories", cell["category"]]
    else:
        cmd += ["--acon-dir", config["acon_dir"], "--bench-python", config["appworld_python"],
                "--split", config["appworld_split"], "--max-iter", str(config["appworld_max_iter"])]
    return cmd


def prepare(config, output, source):
    from benchmarks.arms import get_arm, history_kv_spec
    if config["device"] != "cuda":
        raise ValueError("The paper benchmark uses CUDA")
    for item in config["methods"]:
        arm = get_arm(item["arm"])
        if arm.native_controller:
            if (arm.name != "c2kv_c1_t02_r8" or item.get("ratio") != 8
                    or config.get("c1", {}).get("detector") != "t02_risk"
                    or config["c1"].get("selector_threshold") != 0.5
                    or config["c1"].get("history_variant") != "H0"
                    or config["c1"].get("recovery_rounds") != 1):
                raise ValueError("The final system must use H0/C1000/ratio8/T02/R1")
            continue
        if arm.repair or arm.recover or arm.hybrid_top_k:
            raise ValueError("The paper matrix excludes recovery and hybrid algorithms")
        if item["method"] == "C2KV" and (arm.ratio != 4 or item.get("ratio") != 4 or not arm.compress_history):
            raise ValueError("The selected bare C2KV arm must use ratio 4")
        if item["method"] in ("H2O", "SnapKV"):
            spec = history_kv_spec(arm)
            if not spec["persistent_session"] or spec["retention_ratio"] != item["retention"]:
                raise ValueError("Persistent history-KV budget differs from matrix")
    config = dict(config)
    config["sglang_source"] = str(source.resolve())
    resolved_path = output / "config.resolved.json"
    if resolved_path.exists():
        try:
            existing = json.loads(resolved_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(
                f"Existing output has an unreadable resolved config: {resolved_path}"
            ) from error
        if existing != config:
            old_methods = existing.get("methods", [])
            new_methods = config.get("methods", [])
            unchanged = {k: v for k, v in existing.items() if k not in ("methods", "c1")}
            candidate = {k: v for k, v in config.items() if k not in ("methods", "c1")}
            additions = new_methods[len(old_methods):]
            append_only = (
                unchanged == candidate and new_methods[:len(old_methods)] == old_methods
                and additions and all(row["arm"] == "c2kv_c1_t02_r8" for row in additions)
                and ("c1" not in existing or existing["c1"] == config.get("c1"))
            )
            if not append_only:
                raise RuntimeError(
                    f"Existing output was prepared with a different config: {resolved_path}"
                )
            previous = output / "config.before_c1_extension.json"
            if not previous.exists():
                previous.write_text(json.dumps(existing, indent=2) + "\n")
    elif output.exists() and any(output.iterdir()):
        raise RuntimeError(
            f"Existing non-empty output has no resolved config: {output}"
        )
    output.mkdir(parents=True, exist_ok=True)
    resolved_path.write_text(json.dumps(config, indent=2) + "\n")
    # This describes deployment, not a claim that the legacy packing was the training layout.
    profile = {"schema_version": 1, "profile_kind": "paper_deployment",
               "model": {"gist_param": "qkv", "gist_type": "dynamic-interleave"},
               "training": {"doc_mode": "history_only", "tools_in_system": True,
                            "compression_ratios": [4, 8]},
               "serving": {key: config[key] for key in
                           ("doc_packing", "max_doc_length", "max_doc_num", "query_projection")},
               "provenance": {"field_sources": {
                   "training": "Arm C checkpoint-1000 config.json",
                   "serving": "accepted portable benchmark, explicit complete-history document budget"}}}
    profile["serving"]["compatible"] = True
    profile_path = output / "deployment_profile.json"
    profile_path.write_text(json.dumps(profile, indent=2) + "\n")
    matrix = cells(config)
    with (output / "matrix.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["cell_id", "benchmark", "method", "arm", "group", "ratio", "retention", "adapter", "category"])
        writer.writeheader()
        writer.writerows(matrix)
    plan = []
    for cell in matrix:
        directory = output / "closed_loop" / cell["cell_id"]
        plan.append(dict(cell, command=run_command(config, cell, directory, profile_path),
                         replay_source=str(output / "closed_loop" / (cell["benchmark"] + "__full") / "full_prefixes.jsonl")))
    (output / "commands.json").write_text(json.dumps(plan, indent=2) + "\n")
    return plan, profile_path


def wait_server(proc, port):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    for _ in range(600):
        if proc.poll() is not None:
            raise RuntimeError("CUDA server exited; see server.log")
        try:
            with opener.open(f"http://127.0.0.1:{port}/health", timeout=2) as response:
                if response.status == 200:
                    return
        except OSError:
            pass
        time.sleep(1)
    raise TimeoutError("CUDA server health timeout")


def cleanup_cell_processes(proxy, server):
    """Stop both owned children, collecting errors so server cleanup always runs."""
    errors = []
    if proxy is not None:
        try:
            proxy.terminate()
            try:
                proxy.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proxy.kill()
                proxy.wait()
        except Exception as error:  # cleanup must continue to the server
            errors.append(("proxy", error))

    import signal
    try:
        try:
            os.killpg(server.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            server.wait(timeout=30)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(server.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            server.wait()
    except Exception as error:
        errors.append(("server", error))

    if errors:
        detail = "; ".join(f"{owner}: {error}" for owner, error in errors)
        raise RuntimeError(f"Cell process cleanup failed: {detail}") from errors[0][1]


def execute(config, plan, output, source, stages, selected, port_offset=0):
    config = with_port_offset(config, port_offset)
    profile_path = output / "deployment_profile.json"
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(source / "python"), str(ROOT.parent), env.get("PYTHONPATH", "")])
    env["BENCH_BFCL_DIR"] = config["bfcl_dir"]
    env["APPWORLD_ROOT"] = config["appworld_root"]
    env["C2KV_PAPER_TELEMETRY"] = "1"
    env["SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION"] = "0"
    env.setdefault("CUDA_HOME", "/opt/cuda")
    env["PATH"] = (str(Path(config["server_python"]).parent) + os.pathsep
                   + env.get("PATH", ""))
    for stage in stages:
        for cell in plan:
            if selected and cell["cell_id"] not in selected:
                continue
            directory = output / stage / cell["cell_id"]
            directory.mkdir(parents=True, exist_ok=True)
            if (directory / "complete.json").exists():
                continue
            if (directory / "started.json").exists():
                raise RuntimeError(f"Partial cell {directory}; inspect it before explicitly selecting a new output directory")
            (directory / "started.json").write_text(json.dumps({
                "stage": stage, "cell": cell, "config": config,
                "server_command": server_command(config, source, cell["arm"]),
                "port_offset": port_offset,
                "sglang_source": str(source), "time": time.time()}, indent=2))
            telemetry_name = ("native_engine_telemetry.jsonl" if cell["arm"] == "c2kv_c1_t02_r8"
                              else "server_telemetry.jsonl")
            env["C2KV_PAPER_TELEMETRY_LOG"] = str(directory / telemetry_name)
            with (directory / "server.log").open("w") as log:
                import socket
                for port in (config["server_port"], config["proxy_port"]):
                    with socket.socket() as probe:
                        if probe.connect_ex(("127.0.0.1", port)) == 0:
                            raise RuntimeError(f"Configured port {port} is already occupied")
                # Own only this process group. Never stop another experiment's server.
                server = subprocess.Popen(server_command(config, source, cell["arm"]), env=env,
                                          stdout=log, stderr=subprocess.STDOUT,
                                          start_new_session=True)
                proxy = None
                run_failure = None
                try:
                    wait_server(server, config["server_port"])
                    if cell["arm"] == "c2kv_c1_t02_r8":
                        subprocess.run(run_command(config, cell, directory, profile_path, stage),
                                       check=True, env=env, cwd=ROOT.parent)
                    elif stage == "closed_loop":
                        # Rebuilt here so a runtime port offset reaches the harness;
                        # without an offset this equals the prepared cell["command"].
                        subprocess.run(run_command(config, cell, directory, profile_path),
                                       check=True, env=env)
                    else:
                        prefixes = Path(cell["replay_source"])
                        if not prefixes.is_file():
                            raise FileNotFoundError(prefixes)
                        proxy_cmd = [config["bench_python"], str(ROOT / "proxy.py"), "--upstream",
                                     f"http://127.0.0.1:{config['server_port']}", "--arm", cell["arm"],
                                     "--backend", "sglang", "--port", str(config["proxy_port"]),
                                     "--doc-packing", config["doc_packing"],
                                     "--max-doc-length", str(config["max_doc_length"]),
                                     "--max-doc-num", str(config["max_doc_num"]), "--query-projection", "base",
                                     "--request-log", str(directory / "proxy_requests.jsonl"),
                                     "--telemetry-log", str(directory / "proxy_telemetry.jsonl")]
                        proxy = subprocess.Popen(proxy_cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
                        wait_server(proxy, config["proxy_port"])
                        subprocess.run([config["bench_python"], "-m", "benchmarks.measurement.replay",
                                        "--prefixes", str(prefixes), "--base-url", f"http://127.0.0.1:{config['proxy_port']}",
                                        "--output", str(directory / "prefix_replay.jsonl"),
                                        "--source-run-id", cell["benchmark"] + "__full",
                                        "--target-run-id", cell["cell_id"]], check=True, env=env, cwd=ROOT.parent)
                except BaseException:
                    run_failure = sys.exc_info()
                finally:
                    try:
                        cleanup_cell_processes(proxy, server)
                    except BaseException as error:
                        if run_failure is None:
                            raise
                        failure = run_failure[1]
                        if hasattr(failure, "add_note"):
                            failure.add_note(f"Additionally, cleanup failed: {error}")
                if run_failure is not None:
                    _, error, traceback = run_failure
                    raise error.with_traceback(traceback)
                (directory / "complete.json").write_text(
                    json.dumps({"finished_at": time.time()}))


def _selected_plan(plan, selected):
    if not selected:
        return list(plan), []
    by_id = {cell["cell_id"]: cell for cell in plan}
    unknown = sorted(set(selected) - set(by_id))
    return [cell for cell in plan if cell["cell_id"] in selected], unknown


def _required_aggregate_artifacts(stage, cell, directory):
    required = [
        directory / "complete.json",
        directory / "proxy_telemetry.jsonl",
        directory / "server_telemetry.jsonl",
    ]
    if stage == "closed_loop":
        required.append(directory / f"summary_{cell['arm']}.json")
        if cell["arm"] == "full":
            required.append(directory / "full_prefixes.jsonl")
        if cell["adapter"] == "acon_appworld":
            required.append(directory / "measurement" / "harness_events.jsonl")
    else:
        required.append(directory / "prefix_replay.jsonl")
    return required


def aggregate_results(config, plan, output, stages, selected):
    """Aggregate exactly the requested matrix slice and emit its coverage."""
    requested, unknown = _selected_plan(plan, selected)
    coverage_path = output / "aggregation_coverage.json"
    entries = []
    missing = []
    for stage in stages:
        for cell in requested:
            directory = output / stage / cell["cell_id"]
            absent = [str(path) for path in
                      _required_aggregate_artifacts(stage, cell, directory)
                      if not path.is_file()]
            entry = {
                "stage": stage,
                "cell_id": cell["cell_id"],
                "status": "missing" if absent else "ready",
                "missing_artifacts": absent,
                "measurement_summary": str(directory / "measurement_summary.json"),
            }
            entries.append(entry)
            if absent:
                missing.append({"stage": stage, "cell_id": cell["cell_id"],
                                "missing_artifacts": absent})
    for stage in stages:
        for cell_id in unknown:
            missing.append({"stage": stage, "cell_id": cell_id,
                            "missing_artifacts": ["cell_id is not in commands.json"]})

    coverage = {
        "schema": "c2kv.paper.aggregation_coverage.v1",
        "requested_stages": list(stages),
        "requested_cells": sorted(selected) if selected else "all",
        "counts": {"requested": len(entries) + len(unknown) * len(stages),
                   "ready": sum(entry["status"] == "ready" for entry in entries),
                   "missing": len(missing), "aggregated": 0},
        "missing": missing,
        "cells": entries,
    }
    coverage_path.parent.mkdir(parents=True, exist_ok=True)
    coverage_path.write_text(json.dumps(coverage, indent=2) + "\n", encoding="utf-8")
    if missing:
        raise RuntimeError(
            f"Requested aggregation slice is incomplete; see {coverage_path}")

    aggregated = 0
    try:
        for entry in entries:
            stage = entry["stage"]
            cell = next(cell for cell in requested
                        if cell["cell_id"] == entry["cell_id"])
            directory = output / stage / cell["cell_id"]
            cmd = [config["bench_python"], "-m", "benchmarks.measurement.aggregate",
                   "--proxy", str(directory / "proxy_telemetry.jsonl"),
                   "--output", str(directory / "measurement_summary.json")]
            harness = directory / "measurement" / "harness_events.jsonl"
            if harness.exists():
                cmd += ["--harness", str(harness)]
            replay = directory / "prefix_replay.jsonl"
            if replay.exists():
                cmd += ["--replay", str(replay)]
            server_log = directory / "server_telemetry.jsonl"
            if server_log.exists():
                cmd += ["--server", str(server_log)]
            subprocess.run(cmd, check=True, cwd=ROOT.parent)
            entry["status"] = "aggregated"
            aggregated += 1
    except BaseException as error:
        coverage["aggregation_error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        coverage["counts"]["aggregated"] = aggregated
        coverage_path.write_text(
            json.dumps(coverage, indent=2) + "\n", encoding="utf-8")
    return requested, coverage_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "run", "aggregate"])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--sglang-source", type=Path, default=ROOT.parent.parent / "sglang-paper")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--stage", choices=["all", "closed_loop", "common_prefix"], default="all")
    parser.add_argument("--cells", default="", help="comma-separated exact cell ids")
    parser.add_argument("--port-offset", type=int, default=0,
                        help="shift server/proxy ports for concurrent single-GPU runners on one host")
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text())
    output = args.output or Path(config["output_root"])
    source = args.sglang_source.resolve()
    if args.action == "aggregate":
        config = json.loads((output / "config.resolved.json").read_text())
        plan = json.loads((output / "commands.json").read_text())
    else:
        plan, _ = prepare(config, output, source)
    if args.action == "prepare":
        print(json.dumps({"matrix": str(output / "matrix.csv"), "closed_loop_cells": len(plan), "replay_cells": len(plan)}, indent=2))
    elif args.action == "run":
        stages = ["closed_loop", "common_prefix"] if args.stage == "all" else [args.stage]
        execute(config, plan, output, source, stages, set(filter(None, args.cells.split(","))),
                port_offset=args.port_offset)
    else:
        stages = ["closed_loop", "common_prefix"] if args.stage == "all" else [args.stage]
        selected = set(filter(None, args.cells.split(",")))
        requested, _ = aggregate_results(
            config, plan, output, stages, selected)
        from .report import write_comparison
        write_comparison(output, requested)


if __name__ == "__main__":
    main()
