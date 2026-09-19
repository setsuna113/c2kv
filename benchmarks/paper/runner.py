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


C1_ARMS = {"c2kv_c1_t02_r8": 8, "c2kv_c1_t02_r4": 4}   # native C1 controller arms and their ratios

EVENT_NATIVE_CHECKPOINT_MARKERS = {
    "history_memory_training_profile": "history-event-base-query-v1",
    "history_memory_packing_version": "history-event-v1",
    "history_memory_raw_layout": "event-native-evidence-v1",
}


def is_c1_arm(arm):
    return arm in C1_ARMS


def is_native_arm(arm):
    return is_c1_arm(arm) or arm == "c2kv_native_r4"


def cells(config):
    rows = [dict({k: v for k, v in method.items() if k != "benchmarks"},
                 benchmark=bench["name"], adapter=bench["adapter"],
                 category=bench.get("category", ""),
                 cell_id=bench["name"] + "__" + method["arm"])
            for bench in config["benchmarks"] for method in config["methods"]
            # an optional per-method benchmark list restricts an ablation to some benchmarks
            if not method.get("benchmarks") or bench["name"] in method["benchmarks"]]
    # Run the final system (and its ablations) after every existing comparison/sweep cell.
    return sorted(rows, key=lambda row: is_c1_arm(row["arm"]))


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
    from benchmarks.arms import get_arm, history_kv_spec
    spec = history_kv_spec(get_arm(arm)) if arm is not None else None
    reference_attention = bool(spec and spec["backend"] == "reference_attention")
    radix_arms = set(config.get("radix_cache_arms") or ())
    if reference_attention or ("*" not in radix_arms and arm not in radix_arms):
        cmd.append("--disable-radix-cache")
    if reference_attention or config.get("disable_cuda_graph", True):
        cmd.append("--disable-cuda-graph")
    cmd += ["--disable-piecewise-cuda-graph", "--disable-overlap-schedule",
            "--enable-streaming-session", "--host", "127.0.0.1",
            "--port", str(config["server_port"])]
    if is_c1_arm(arm):
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
    if is_native_arm(cell["arm"]):
        cmd = [config["bench_python"], "-m", "benchmarks.paper.c1",
               "--config", str(profile.parent / "config.resolved.json"),
               "--arm", cell["arm"],
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
           "--model", config["model"], "--model-family",
           config.get("model_family", "qwen3-4b"), "--checkpoint", config["checkpoint"],
           "--checkpoint-profile", str(profile), "--out", str(directory),
           "--exact-out", "--run-name", cell["cell_id"], "--num-workers", "1",
           "--telemetry-log", str(directory / "proxy_telemetry.jsonl"),
           "--capability-features",
           "hiagent_trajectory_retrieval_v1,acebench_role_history_v1"]
    if cell["arm"] == "full":
        cmd += ["--record-prefixes", str(directory / "full_prefixes.jsonl")]
    if cell["adapter"] == "bfcl":
        cmd += ["--categories", cell["category"]]
    elif cell["adapter"] == "acebench":
        # Keep the matrix on ACEBench Agent rather than its fixed-call splits.
        cmd += ["--acebench-category", cell.get("category") or "agent",
                "--acebench-language", config.get("acebench_language", "en"),
                "--acebench-dir", config["acebench_dir"],
                "--bench-python", config.get("acebench_python", config["bench_python"])]
    elif cell["adapter"] == "toolsandbox":
        cmd += ["--toolsandbox-dir", config["toolsandbox_dir"],
                "--bench-python", config.get("toolsandbox_python", config["bench_python"]),
                "--ts-parallel", "1"]
        scenarios = config.get("toolsandbox_scenarios") or []
        if scenarios:
            cmd += ["--ts-scenarios", ",".join(scenarios)]
        elif config.get("toolsandbox_suite") == "full":
            cmd.append("--full")
        else:
            raise ValueError("ToolSandbox paper cells require suite=full or explicit scenarios")
    else:
        cmd += ["--acon-dir", config["acon_dir"], "--bench-python", config["appworld_python"],
                "--split", config["appworld_split"], "--max-iter", str(config["appworld_max_iter"])]
    return cmd


# Deployment fields whose change would alter what an existing cell measured.
DEPLOYMENT_KEYS = ("model", "checkpoint", "device", "attention_backend", "disable_cuda_graph",
                   "context_length", "max_total_tokens", "chunked_prefill_size", "mem_fraction_static",
                   "c2kv_pool_fraction", "max_running_requests", "seed", "doc_packing", "max_doc_length",
                   "max_doc_num", "query_projection", "appworld_split", "appworld_max_iter", "c1")


def extension_problem(existing, config, source, output):
    """Why an existing output root may NOT be extended with ``config``; None when it may.

    An output root keeps its completed artifacts valid only if every cell it already
    defined keeps the same algorithm and byte-identical server/run commands. Adding
    cells (new arms, new benchmarks, reordered method lists, per-arm radix or
    model-family flags for arms not yet run) is an extension; anything that touches an
    existing cell is a different experiment and needs a new output directory.
    """
    old_cells = {row["cell_id"]: row for row in cells(existing)}
    new_cells = {row["cell_id"]: row for row in cells(config)}
    missing = sorted(set(old_cells) - set(new_cells))
    if missing:
        return f"cells removed: {missing[:3]}"
    if not (set(new_cells) - set(old_cells)):
        return "no new cells"
    for key in DEPLOYMENT_KEYS:
        if key == "c1" and existing.get("c1") is None:
            continue   # the C1 block arrives with the first C1 cell; no old cell used it
        if existing.get(key) != config.get(key):
            return f"deployment field changed: {key}"
    profile = output / "deployment_profile.json"
    # Keys the old config never had (paths for adapters added later) cannot have
    # changed an old cell; fill them from the new config so the old command can
    # still be rendered by the current code.
    old_view = {**config, **existing}
    for cell_id, old in old_cells.items():
        new = new_cells[cell_id]
        # ``method`` is the table label, not the algorithm: it may be renamed while
        # the cell has no artifacts, after which it is frozen with them.
        has_artifacts = any((output / stage / cell_id).exists() for stage in ("closed_loop", "common_prefix"))
        for key in ("arm", "method", "ratio", "retention", "benchmark", "adapter", "category"):
            if key == "method" and not has_artifacts:
                continue
            if old.get(key) != new.get(key):
                return f"{cell_id}: {key} changed"
        if server_command(old_view, source, old["arm"]) != server_command(config, source, new["arm"]):
            return f"{cell_id}: server command changed"
        for stage in ("closed_loop", "common_prefix"):
            directory = output / stage / cell_id
            if run_command(old_view, old, directory, profile, stage) != run_command(config, new, directory, profile, stage):
                return f"{cell_id}: {stage} command changed"
    return None


def prepare(config, output, source):
    from benchmarks.arms import get_arm, history_kv_spec
    if config["device"] != "cuda":
        raise ValueError("The paper benchmark uses CUDA")
    for item in config["methods"]:
        arm = get_arm(item["arm"])
        if arm.name == "c2kv_native_r4":
            if item.get("ratio") != 4 or item["method"] != "C2KV":
                raise ValueError("Native bare C2KV must use ratio4 and the C2KV label")
            continue
        if arm.native_controller:
            if (not is_c1_arm(arm.name) or item.get("ratio") != C1_ARMS[arm.name]
                    or arm.ratio != C1_ARMS[arm.name]
                    or config.get("c1", {}).get("detector") not in {"t02_risk", "d3_hybrid"}
                    or config["c1"].get("selector_threshold") != 0.5
                    or config["c1"].get("history_variant") != "H0"
                    or config["c1"].get("recovery_rounds") != 1):
                raise ValueError("The final system must use H0/C1000/ratio8/R1 with T02 or D3 hybrid (ratio 4 only as its ablation)")
            if item.get("benchmarks") and not set(item["benchmarks"]) <= {b["name"] for b in config["benchmarks"]}:
                raise ValueError(f"Unknown benchmark restriction on {arm.name}: {item['benchmarks']}")
            continue
        if arm.repair or arm.recover or arm.hybrid_top_k:
            raise ValueError("The paper matrix excludes recovery and hybrid algorithms")
        if item["method"] == "C2KV" and (arm.ratio != 4 or item.get("ratio") != 4 or not arm.compress_history):
            raise ValueError("The selected bare C2KV arm must use ratio 4")
        if item["method"] in ("H2O", "SnapKV", "PyramidKV"):
            spec = history_kv_spec(arm)
            expected = {
                "H2O": ("h2o", "physical_eviction"),
                "SnapKV": ("snapkv_persistent", "physical_eviction"),
                "PyramidKV": ("pyramidkv", "reference_attention"),
            }[item["method"]]
            if (spec is None or spec["method"] != expected[0]
                    or spec["backend"] != expected[1]
                    or not spec["persistent_session"]
                    or spec["retention_ratio"] != item["retention"]
                    or spec["target_tokens"] is not None):
                raise ValueError("Persistent history-KV budget differs from matrix")
        if arm.name == "agentfold":
            if item["method"] != "AgentFold" or arm.text_policy != "agentfold" or arm.history_kv:
                raise ValueError("AgentFold paper cell must use the actor folding policy")
        if arm.name in {"commitkv", "agentkv"}:
            spec = history_kv_spec(arm)
            expected_label = {"commitkv": "CommitKV", "agentkv": "AgentKV"}[arm.name]
            if (item["method"] != expected_label or spec is None
                    or spec["method"] != arm.name
                    or spec["backend"] != "reference_attention"
                    or spec["target_tokens"] != 2048
                    or spec["retention_ratio"] is not None
                    or not spec["persistent_session"]
                    or arm.text_policy):
                raise ValueError(
                    f"{expected_label} paper cell must use its 2048-token "
                    "persistent reference-attention runtime"
                )
        if arm.name in {"agentfold", "commitkv", "agentkv"}:
            if config.get("model_family", "qwen3-4b") != "qwen3-4b":
                raise ValueError("AgentFold/CommitKV/AgentKV require model_family=qwen3-4b")
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
            problem = extension_problem(existing, config, source, output)
            if problem:
                raise RuntimeError(
                    f"Existing output was prepared with a different config ({problem}): {resolved_path}"
                )
            previous = output / f"config.before_extension.{int(time.time())}.json"
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
    ids = [row["cell_id"] for row in matrix]
    if len(ids) != len(set(ids)):
        raise ValueError("The paper matrix contains duplicate cell ids")
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
    (output / "unsupported_cells.json").write_text(json.dumps(
        config.get("unsupported_cells") or [], indent=2) + "\n")
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


def _guard_checkpoint_serving_layout(config, cell):
    """Reject event-native checkpoints only on the legacy compression path."""
    from benchmarks.arms import get_arm

    arm = get_arm(cell["arm"])
    if not arm.compress_history or arm.native_controller:
        return

    config_path = Path(config["checkpoint"]).expanduser() / "config.json"
    try:
        checkpoint_config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"Cannot verify the legacy compression serving layout because checkpoint "
            f"config.json is unavailable or invalid: {config_path}"
        ) from error
    event_native = {
        key: value for key, value in EVENT_NATIVE_CHECKPOINT_MARKERS.items()
        if checkpoint_config.get(key) == value
    }
    if event_native:
        detail = ", ".join(f"{key}={value!r}" for key, value in event_native.items())
        raise RuntimeError(
            f"Training/serving mismatch for arm {arm.name!r}: the checkpoint declares "
            f"event-native history ({detail}), but this arm uses legacy turn packing. "
            "Use an implemented native event-packed base arm for this checkpoint; "
            "do not substitute a C1 controller arm."
        )


def _guard_method_actor(cell):
    if cell["arm"] == "agentfold":
        raise RuntimeError(
            "AgentFold is on hold in the paper matrix: the shared experiment "
            "actor has no trained AgentFold joint folding/action policy. "
            "The failed run is an untrained-actor protocol diagnostic, not a "
            "method quality result. Missing intermediate directives must not "
            "silently become no-fold actions or reminder retries."
        )


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
            if (directory / "complete.json").exists():
                continue
            if (directory / "started.json").exists():
                raise RuntimeError(f"Partial cell {directory}; inspect it before explicitly selecting a new output directory")
            _guard_method_actor(cell)
            _guard_checkpoint_serving_layout(config, cell)
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "started.json").write_text(json.dumps({
                "stage": stage, "cell": cell, "config": config,
                "server_command": server_command(config, source, cell["arm"]),
                "port_offset": port_offset,
                "sglang_source": str(source), "time": time.time()}, indent=2))
            telemetry_name = ("native_engine_telemetry.jsonl" if is_native_arm(cell["arm"])
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
                    if is_native_arm(cell["arm"]):
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
                                     "--benchmark", cell["adapter"],
                                     "--backend", "sglang", "--port", str(config["proxy_port"]),
                                     "--doc-packing", config["doc_packing"],
                                     "--max-doc-length", str(config["max_doc_length"]),
                                      "--max-doc-num", str(config["max_doc_num"]), "--query-projection", "base",
                                      "--model-family", config.get("model_family", "qwen3-4b"),
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
        if cell["adapter"] in {"acon_appworld", "acebench"}:
            required.append(directory / "measurement" / "harness_events.jsonl")
        if cell["adapter"] == "toolsandbox":
            required.extend([
                directory / "scenario_manifest.json",
                directory / "toolsandbox_protocol.json",
                directory / "measurement" / "harness_events.jsonl",
            ])
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
