"""C2KV-backend cell driver for the generality experiment (runs on ascend03).

One cell = backend=c2kv x working point x condition x benchmark. The driver
launches the frozen T02-era controller server (our forked controller_runtime)
per task against the long-lived NPU engine, runs the official benchmark
worker, and writes per-task receipts with resume support.

Conditions:
  tracer_history          -> method=proposed, detector=t02_risk, cell threshold,
                             eval_policy K + recovery caps B, ratio 4
  recovery_off_same_initial -> method=c2kv_only, eval_policy K, ratio 4
  compression_full_budget  -> method=c2kv_only, eval_policy B (single budget),
                             ratio 4; only labelled competitive after the
                             available_history audit passes
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

try:
    from .bfcl_results import (
        bfcl_row_is_valid,
        collect_bfcl_results,
        completion_receipt,
        ordered_unique,
    )
except ImportError:
    from bfcl_results import (
        bfcl_row_is_valid,
        collect_bfcl_results,
        completion_receipt,
        ordered_unique,
    )

GENERATION_ROOT = Path("/home/liuyancheng/c2kv-generality-20260918")
C1_DELIVERY = GENERATION_ROOT / "src" / "c1_delivery"
RUNTIME = GENERATION_ROOT / "src" / "generality" / "controller_runtime"
sys.path.insert(0, str(C1_DELIVERY))

import current                     # noqa: E402  (c1_delivery parent module)
import evidence_sets              # noqa: E402
from c1_artifact_binding import bind_risk_artifact  # noqa: E402

RISK_ARTIFACT = C1_DELIVERY / "artifacts" / "c1_risk.t02_v1.json"

WORKER_MODULES = {
    "bfcl": "benchmarks.memory_runtime.event_native_bfcl",
    "acon_appworld": "benchmarks.memory_runtime.event_native_appworld",
}


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _controller_with_binding(cell: dict) -> tuple[dict, dict | None]:
    """T02 risk selector with the cell's calibrated threshold (fixed weights)."""
    if cell["condition"] == "tracer_history":
        config, _ = evidence_sets.build_config(
            history="H0",
            selector="risk",
            selector_artifact=RISK_ARTIFACT,
            selector_threshold=float(cell["threshold"]),
            embedding_model=cell["embedding_model"],
            embedding_device="cpu",
            semantic_query_overflow_policy="task_head_tail_preserve_draft_v1",
        )
        config["local_models"]["embedding"]["dtype"] = "bfloat16"
        config["selector_artifact"], binding = bind_risk_artifact(
            config["selector_artifact"], Path(cell["checkpoint"])
        )
        controller = current._configure_controller(evidence_sets._base_controller(), config)
        # Hard gate: the D3-era post_draft_recovery section rides along in the
        # shared base config but is provably inert under set_selector="risk"
        # (_reconsider_sets only evaluates the legacy gate for legacy_prefill).
        # Fail loudly if any of these bindings ever drift.
        import hashlib
        gp = controller["gp_experiments"]
        assert gp["set_selector"] == "risk", gp["set_selector"]
        assert gp["selection_protocol"] == "evidence_sets_v1", gp["selection_protocol"]
        assert gp["selector_artifact"]["model_kind"] == "c1_risk_logistic"
        assert (hashlib.sha256(RISK_ARTIFACT.read_bytes()).hexdigest()
                == "18a11f73aa1f7d4b0add86eed66ae9e5e129ea4bdfbe0dfad23faf4f7d2fb4ab")
        assert gp["recovery_reserve_tokens"] == 0, gp["recovery_reserve_tokens"]
        return controller, binding
    controller = evidence_sets._base_controller()
    controller.pop("post_draft_recovery", None)
    controller.pop("gp_experiments", None)
    return controller, None


def build_controller_config(cell: dict) -> dict:
    controller, binding = _controller_with_binding(cell)
    if binding is not None:
        _write(Path(cell["cell_dir"]) / "risk_artifact_binding.json", binding)
    return controller


def build_eval_policy(cell: dict, budgets: dict) -> dict:
    wp = budgets["working_points"][cell["working_point"]]
    policy = {
        "history_budget_bytes": wp["history_allowance_bytes"],
        "workspace_budget_bytes": wp["history_allowance_bytes"],
        "lease_decisions": 0,
        "max_retrieved_events": 2,
    }
    if cell["condition"] == "tracer_history":
        policy["recovery_history_bytes"] = wp["common_cap_bytes"]
        policy["recovery_workspace_bytes"] = wp["common_cap_bytes"]
    elif cell["condition"] == "compression_full_budget":
        # competitive control: the bare compressor may use the whole common cap
        policy["history_budget_bytes"] = wp["common_cap_bytes"]
        policy["workspace_budget_bytes"] = wp["common_cap_bytes"]
    return {
        "schema": "a-event-native-eval-policy-v1",
        "policy_id": f"generality-{cell['cell_id']}",
        "policy": policy,
    }


def server_command(cell: dict, task_ids: list[str], out: Path, port: int) -> list[str]:
    design = current.load_config()
    caps = cell["caps"]
    command = [
        cell["python_sgl"], "-m", "benchmarks.memory_runtime.event_native_server",
        "--checkpoint", cell["checkpoint"],
        "--out", str(out / "server"),
        "--run-id", f"{cell['cell_id']}__b{abs(hash(tuple(task_ids))) % 10**8}",
        "--model-name", cell["model_name"],
        "--benchmark", cell["benchmark"],
        "--source-profile",
        "native-v1" if cell["benchmark"] == "bfcl" else "openai-single-task-v1",
        "--view-mode", design["route"],
        "--compression-policy", design["compression_policy"],
        "--history-view-protocol", design["history_view_protocol"],
        "--ratio", str(cell["ratio"]),
        "--max-new-tokens", str(caps["max_completion_tokens"]),
        "--decode-strategy", design["decode_strategy"],
        "--prefill-chunk-size", str(design["prefill_chunk_size"]),
        "--task-ids", ",".join(task_ids),
        "--max-decisions", str(caps["generation_attempts_per_task"] * len(task_ids)),
        "--max-generation-calls", str(caps["generation_attempts_per_task"]),
        "--max-extraction-calls", str(caps["extraction_calls_per_task"]),
        "--eval-policy", str(cell["eval_policy_path"]),
        "--eval-capacity", str(C1_DELIVERY / "runtime/configs/eval_capacity.json"),
        "--s0-config", str(cell["controller_path"]),
        "--max-wall-seconds", str(caps["task_timeout"]),
        "--device", "cpu",
        "--dtype", "bfloat16",
        "--generation-backend", "sglang",
        "--host", "127.0.0.1",
        "--port", str(port),
        "--torch-threads", "4",
        "--sglang-backend-url", cell["sglang_backend_url"],
        "--sglang-timeout-seconds", str(caps["task_timeout"]),
        "--no-raw-snapshot",
    ]
    return command


def bfcl_worker_command(cell: dict, task_ids: list[str], out: Path, port: int) -> list[str]:
    # the official worker reads the frozen allowlist from the server manifest
    del task_ids
    return [
        cell["python_bench"], "-m", "benchmarks.memory_runtime.event_native_bfcl",
        "--server-manifest", str(out / "server" / "ready.json"),
        "--base-url", f"http://127.0.0.1:{port}/v1",
        "--benchmark-dir", cell["benchmark_dir"],
        "--out", str(out / "bfcl_worker"),
        "--max-wall-seconds", str(caps_of(cell)["task_timeout"] * 200),
    ]


def appworld_worker_command(cell: dict, task_id: str, out: Path, port: int) -> list[str]:
    return [
        cell["python_sgl"], "-m", "benchmarks.memory_runtime.event_native_appworld",
        "--server-manifest", str(out / "server" / "ready.json"),
        "--base-url", f"http://127.0.0.1:{port}/v1",
        "--acon-dir", cell["acon_dir"],
        "--appworld-root", cell["appworld_root"],
        "--bench-python", cell["python_appworld"],
        "--out", str(out / "appworld_worker"),
        "--task-id", task_id,
        "--max-iter", "50",
        "--max-wall-seconds", str(caps_of(cell)["task_timeout"]),
    ]


def caps_of(cell: dict) -> dict:
    return cell["caps"]


def worker_command(cell: dict, task_ids: list[str], out: Path, port: int) -> list[str]:
    if cell["benchmark"] == "bfcl":
        return bfcl_worker_command(cell, task_ids, out, port)
    return appworld_worker_command(cell, task_ids[0], out, port)


def validate_chunk(out: Path, task_ids: list[str]) -> tuple[list[str], list[str]]:
    """Split the chunk's official result rows into healthy vs bad tasks.

    BFCL v4 multi_turn rows carry `result` (+ per-turn fields) on success and
    a `traceback` field on inference errors; there is no model_responses
    field in this vintage. Healthy = a row exists and traceback is None.
    Connection errors, terminal controller failures and timeouts all leave a
    traceback and must never be counted as an official zero score.
    """
    import glob
    requested = ordered_unique(task_ids)
    requested_set = set(requested)
    observed: dict[str, bool] = {}
    for path in glob.glob(str(out / "bfcl_worker" / "bfcl" / "result" / "**" / "*.json"),
                          recursive=True):
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            task_id = row.get("id")
            if task_id not in requested_set:
                continue
            # Any valid official row completes the task.  Later traceback rows
            # cannot turn an already valid task back into a retry.
            observed[task_id] = observed.get(task_id, False) or bfcl_row_is_valid(
                row, fc_model=True)
    healthy = [task_id for task_id in requested if observed.get(task_id, False)]
    bad = [task_id for task_id in requested if not observed.get(task_id, False)]
    return healthy, bad


def validate_appworld_summary(out: Path, task_id: str) -> None:
    path = out / "appworld_worker" / "official_summary.json"
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"missing or invalid AppWorld summary for {task_id}") from error
    score = summary.get("semantic_score") if isinstance(summary, dict) else None
    if (not isinstance(summary, dict)
            or summary.get("schema") != "a-event-native-appworld-run-v1"
            or summary.get("status") != "completed"
            or summary.get("task_id") != task_id
            or summary.get("n") != 1
            or not isinstance(score, (int, float)) or isinstance(score, bool)
            or not math.isfinite(score)):
        raise RuntimeError(f"AppWorld summary is not a scored completion for {task_id}")


def appworld_task_completed(cell_dir: Path, task_id: str) -> bool:
    for path in (cell_dir / "batches").glob("*/appworld_worker/official_summary.json"):
        try:
            validate_appworld_summary(path.parent.parent, task_id)
        except RuntimeError:
            continue
        return True
    return False


def run_task(cell: dict, task_ids: list[str], port: int, batch_dirname: str) -> dict:
    """Run one chunk of tasks under a single controller server instance.

    The controller accepts a frozen comma-separated allowlist; the official
    BFCL worker generates/scores exactly that subset. The returned status
    carries per-task health from the official result rows so the caller can
    bisect a poisoned chunk instead of letting one terminal task zero out the
    rest.
    """
    out = Path(cell["cell_dir"]) / "batches" / batch_dirname
    # Attempts are evidence.  Never reuse a name or remove an older attempt:
    # retries legitimately produce duplicate raw rows which canonical rescore
    # resolves without losing provenance.
    out.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join((str(RUNTIME / "python"), str(RUNTIME)))
    env["no_proxy"] = env["NO_PROXY"] = "127.0.0.1,localhost"
    for k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        env.pop(k, None)
    worker_env = env.copy()
    if cell["benchmark"] == "bfcl":
        worker_env["PYTHONPATH"] = str(RUNTIME)
    elif cell["benchmark"] == "acon_appworld":
        # event_native_appworld needs: controller_runtime (for the worker
        # itself), benchmarks/ (for adapters/proxy imports), acon/src (for
        # the ACON harness), and paper_harness/benchmarks (for proxy)
        # event_native_appworld imports ``adapters.acon_adapter`` from the
        # paper harness.  The controller runtime also has an ``adapters``
        # package, so the harness directory must precede runtime/benchmarks.
        worker_env["PYTHONPATH"] = os.pathsep.join((
            "/home/liuyancheng/c2kv-generality-20260918/src/paper_harness/benchmarks",
            str(RUNTIME), str(RUNTIME / "benchmarks"),
            "/home/liuyancheng/baselines/acon/src"))
        worker_env["APPWORLD_ROOT"] = cell.get("appworld_root", "")
    server_log = (out / "controller.log").open("wb")
    worker_log = (out / "benchmark.log").open("wb")
    started = time.monotonic()
    server = worker = None
    status = {"chunk": batch_dirname, "n_tasks": len(task_ids), "status": "started",
              "started_at": started}
    _write(out / "status.json", status)
    try:
        server = subprocess.Popen(
            server_command(cell, task_ids, out, port), cwd=str(RUNTIME), env=env,
            stdout=server_log, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, start_new_session=True,
        )
        ready = out / "server" / "ready.json"
        deadline = time.monotonic() + cell["caps"]["task_timeout"] * len(task_ids)
        while not ready.exists():
            if server.poll() is not None:
                raise RuntimeError(f"controller exited rc={server.returncode}")
            if time.monotonic() > deadline:
                raise TimeoutError("controller readiness timeout")
            time.sleep(2)
        if cell["benchmark"] == "acon_appworld":
            # AppWorld: one task per worker invocation (event_native_appworld)
            for task_id in task_ids:
                worker = subprocess.Popen(
                    appworld_worker_command(cell, task_id, out, port),
                    cwd=str(RUNTIME),
                    env=worker_env, stdout=worker_log, stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL, start_new_session=True,
                )
                rc = worker.wait(timeout=max(60, deadline - time.monotonic()))
                if rc != 0:
                    raise RuntimeError(f"official worker exited rc={rc} for {task_id}")
                validate_appworld_summary(out, task_id)
        else:
            worker = subprocess.Popen(
                bfcl_worker_command(cell, task_ids, out, port), cwd=str(RUNTIME),
                env=worker_env, stdout=worker_log, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, start_new_session=True,
            )
            rc = worker.wait(timeout=max(60, deadline - time.monotonic()))
            if rc != 0:
                raise RuntimeError(f"official worker exited rc={rc}")
        if cell["benchmark"] == "acon_appworld":
            # AppWorld results are evaluation JSONs, not BFCL result rows;
            # the worker's rc=0 + official_summary.json are the health check
            healthy, bad = task_ids, []
        else:
            healthy, bad = validate_chunk(out, task_ids)
        status.update(status="completed" if not bad else "partial",
                      healthy=healthy, bad=bad,
                      wall_s=time.monotonic() - started)
        _write(out / "done.json" if not bad else out / "status.json", status)
        return status
    except Exception as error:  # infra failure: keep receipt, bisect at caller
        # The official BFCL worker can exit nonzero after writing valid rows.
        # Preserve those outputs so the caller retries only tasks that truly
        # lack a completed official result.
        if cell["benchmark"] == "bfcl":
            healthy, bad = validate_chunk(out, task_ids)
            if healthy:
                status.update(
                    status="partial" if bad else "completed",
                    healthy=healthy,
                    bad=bad,
                    error=f"{type(error).__name__}: {error}",
                    wall_s=time.monotonic() - started,
                )
                _write(out / "status.json", status)
                return status
        status.update(status="failed", error=f"{type(error).__name__}: {error}",
                      wall_s=time.monotonic() - started)
        _write(out / "status.json", status)
        return status
    finally:
        for proc, name in ((worker, "worker"), (server, "server")):
            if proc is None:
                continue
            try:
                # kill the whole session group: controllers spawn --serve-child
                # processes that must not outlive the parent and hold ports
                try:
                    os.killpg(os.getpgid(proc.pid), 15)
                except OSError:
                    if proc.poll() is None:
                        proc.terminate()
                try:
                    proc.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(os.getpgid(proc.pid), 9)
                    except OSError:
                        proc.kill()
            except OSError:
                pass
        server_log.close()
        worker_log.close()


def load_cell(cell_json: Path) -> dict:
    cell = json.loads(cell_json.read_text())
    for key, path in (
        ("controller_path", cell.get("controller_path")),
        ("eval_policy_path", cell.get("eval_policy_path")),
    ):
        if key in cell and not Path(cell[key]).is_absolute():
            cell[key] = str((cell_json.parent / cell[key]).resolve())
    return cell


def prepare_cell_files(cell: dict, budgets: dict) -> dict:
    cell_dir = Path(cell["cell_dir"])
    controller, binding = _controller_with_binding(cell)
    policy = build_eval_policy(cell, budgets)
    prepared = dict(cell)
    prepared["controller_path"] = str(cell_dir / "controller.json")
    prepared["eval_policy_path"] = str(cell_dir / "eval_policy.json")
    frozen = {"controller.json": controller, "eval_policy.json": policy,
              "cell.json": prepared}
    if binding is not None:
        frozen["risk_artifact_binding.json"] = binding

    attempts = cell_dir / "batches"
    if attempts.is_dir() and any(attempts.iterdir()):
        for name, expected in frozen.items():
            path = cell_dir / name
            try:
                previous = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise ValueError(f"existing attempts require frozen {name}") from error
            if name == "cell.json":
                # Engine URL is allocated per scheduler run; all other cell
                # fields are part of the frozen experimental contract.
                previous = dict(previous)
                expected = dict(expected)
                previous.pop("sglang_backend_url", None)
                expected.pop("sglang_backend_url", None)
            if previous != expected:
                raise ValueError(f"existing attempts use a different frozen {name}")
        return prepared

    for name, value in frozen.items():
        _write(cell_dir / name, value)
    return prepared


def _attempt_name(task_ids: list[str]) -> str:
    digest = hashlib.sha256("\0".join(task_ids).encode("utf-8")).hexdigest()[:10]
    return f"a{time.time_ns()}_p{os.getpid()}_{digest}"


def _write_bfcl_completion(cell: dict, expected_task_ids: list[str]) -> dict:
    cell_dir = Path(cell["cell_dir"])
    completion = collect_bfcl_results(cell_dir, expected_task_ids, fc_model=True)
    receipt = completion_receipt(completion)
    receipt["cell_id"] = cell["cell_id"]
    receipt["generated_at"] = time.time()
    _write(cell_dir / "bfcl_completion.json", receipt)
    _write(cell_dir / "bfcl_refill.json", {
        "schema": "generality-bfcl-refill-v1",
        "cell_id": cell["cell_id"],
        "task_ids": completion["refill_task_ids"],
        "n_tasks": len(completion["refill_task_ids"]),
        "source": "bfcl_completion.json",
        "generated_at": receipt["generated_at"],
    })
    return completion


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell", type=Path, required=True, help="cell.json path")
    parser.add_argument("--budgets", type=Path, required=True, help="resolved budgets json")
    parser.add_argument("--task-ids", nargs="*", default=None,
                        help="subset override; default: cell manifest")
    parser.add_argument("--port-base", type=int, default=37200)
    parser.add_argument("--chunk", type=int, default=8,
                        help="tasks per controller server instance")
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument(
        "--max-attempts-per-task", type=int, default=4,
        help="bounded attempts in this invocation, including the initial batch",
    )
    parser.add_argument(
        "--audit-results-only", action="store_true",
        help="write exact BFCL completion/refill manifests without launching work",
    )
    args = parser.parse_args(argv)

    if args.max_attempts_per_task < 1:
        parser.error("--max-attempts-per-task must be at least 1")
    cell = load_cell(args.cell)
    expected_task_ids = cell["task_ids"]
    if (not isinstance(expected_task_ids, list) or not expected_task_ids
            or any(not isinstance(task_id, str) or not task_id
                   for task_id in expected_task_ids)
            or len(expected_task_ids) != len(set(expected_task_ids))):
        raise ValueError("cell manifest must contain unique nonempty task IDs")
    requested_task_ids = ordered_unique(
        args.task_ids if args.task_ids is not None else expected_task_ids
    )
    unexpected_requested = sorted(set(requested_task_ids) - set(expected_task_ids))
    if unexpected_requested:
        raise ValueError(
            "requested task IDs are outside the frozen cell manifest: "
            + ",".join(unexpected_requested)
        )
    if args.audit_results_only:
        if cell["benchmark"] != "bfcl":
            raise ValueError("--audit-results-only is supported only for BFCL cells")
        completion = _write_bfcl_completion(cell, expected_task_ids)
        print(json.dumps(completion_receipt(completion), ensure_ascii=False))
        return 0

    budgets = json.loads(args.budgets.read_text())
    # Existing attempts freeze the controller, policy, and cell contract;
    # prepare_cell_files rejects drift before writing any of them.
    cell = prepare_cell_files(cell, budgets)
    task_ids = requested_task_ids
    if args.max_tasks is not None:
        task_ids = task_ids[: args.max_tasks]

    # AppWorld: the controller (single_task_harness_api) requires exactly one
    # frozen task per server instance — chunk size must be 1
    if cell["benchmark"] == "acon_appworld":
        args.chunk = 1

    cell_dir = Path(cell["cell_dir"])
    progress = cell_dir / "progress.jsonl"
    port_counter = [args.port_base]
    attempt_counts = {task_id: 0 for task_id in task_ids}
    valid_task_ids: set[str] = set()
    if cell["benchmark"] == "bfcl":
        completion = _write_bfcl_completion(cell, expected_task_ids)
        valid_task_ids.update(completion["valid_task_ids"])
        task_ids = [task_id for task_id in task_ids if task_id not in valid_task_ids]

    def next_port() -> int:
        """Monotonic port: the scheduler now gives each card 1000 ports, so
        a driver never crosses into a neighbor's range; TIME_WAIT from a
        previous server on the SAME port is handled by the strict probe."""
        import socket
        while True:
            port_counter[0] += 1
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
                try:
                    probe.bind(("127.0.0.1", port_counter[0]))
                    return port_counter[0]
                except OSError:
                    continue

    def run_chunk(chunk: list[str], depth: int) -> None:
        """Run a chunk; on failure or unhealthy tasks, bisect down to singles.

        BFCL tracebacks and missing rows remain retryable; only a structured
        official output completes a task.  Attempts in one invocation are
        bounded by ``--max-attempts-per-task``.
        """
        if not chunk:
            return
        # BFCL resumes from canonical official rows.  AppWorld keeps its
        # existing per-task marker protocol.
        if cell["benchmark"] == "bfcl":
            # A timeout/nonzero worker may finish writing a valid row while
            # run_task is cleaning up its process group.  Refresh from disk
            # before every retry decision so that late output is not rerun.
            refreshed = collect_bfcl_results(cell_dir, expected_task_ids, fc_model=True)
            valid_task_ids.update(refreshed["valid_task_ids"])
            chunk = [
                task_id for task_id in chunk
                if task_id not in valid_task_ids
                and attempt_counts.get(task_id, 0) < args.max_attempts_per_task
            ]
        else:
            chunk = [
                task_id for task_id in chunk
                if not appworld_task_completed(cell_dir, task_id)
                and attempt_counts.get(task_id, 0) < args.max_attempts_per_task
            ]
        if not chunk:
            return
        for task_id in chunk:
            attempt_counts[task_id] = attempt_counts.get(task_id, 0) + 1
        port = next_port()
        name = _attempt_name(chunk)
        result = run_task(cell, chunk, port, name)
        with progress.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")
        healthy = list(result.get("healthy") or [])
        if result["status"] == "completed":
            for task_id in healthy:
                valid_task_ids.add(task_id)
                _write(cell_dir / "tasks" / task_id / "done.json",
                       {"task_id": task_id, "status": "completed"})
            print(json.dumps({"cell": cell["cell_id"], "chunk": len(chunk),
                              "status": "ok", "done": healthy and len(healthy)}), flush=True)
            return
        # partial or failed: bisect
        if len(chunk) == 1:
            task_id = chunk[0]
            # A worker exit without a scored task is not evidence of a model
            # failure.  In particular, old AppWorld cells recorded the
            # acon_adapter ImportError as terminal for every task.
            _write(cell_dir / "tasks" / task_id / "retryable.json",
                   {"task_id": task_id,
                    "status": "retryable",
                    "reason": result.get("error") or "unhealthy_result_rows",
                    "chunk_status": result["status"]})
            print(json.dumps({"cell": cell["cell_id"], "task": task_id,
                              "status": "retryable_recorded"}), flush=True)
            return
        # healthy tasks inside a partial chunk still count
        for task_id in healthy:
            valid_task_ids.add(task_id)
            _write(cell_dir / "tasks" / task_id / "done.json",
                   {"task_id": task_id, "status": "completed"})
        remainder = [t for t in chunk if t not in set(healthy)]
        half = max(1, len(remainder) // 2)
        run_chunk(remainder[:half], depth + 1)
        run_chunk(remainder[half:], depth + 1)

    for i in range(0, len(task_ids), args.chunk):
        run_chunk(task_ids[i:i + args.chunk], 0)

    if cell["benchmark"] == "bfcl":
        completion = _write_bfcl_completion(cell, expected_task_ids)
        _write(cell_dir / "cell_status.json", {
            "cell_id": cell["cell_id"],
            "status": (
                "complete"
                if completion["valid_count"] == completion["expected_count"]
                else "incomplete"
            ),
            "n_completed": completion["valid_count"],
            "n_valid_unique": completion["valid_count"],
            "n_retryable": len(completion["refill_task_ids"]),
            "n_total": completion["expected_count"],
            "raw_result_rows": completion["total_rows"],
            "duplicate_result_rows": completion["duplicate_rows"],
            "finished_at": time.time(),
        })
    else:
        done = sum(1 for t in expected_task_ids
                   if appworld_task_completed(cell_dir, t))
        # Legacy terminal.json files are retained as evidence but do not
        # complete an unscored AppWorld task.
        retryable = len(expected_task_ids) - done
        _write(cell_dir / "cell_status.json", {
            "cell_id": cell["cell_id"],
            "status": "complete" if retryable == 0 else "incomplete",
            "n_completed": done, "n_retryable": retryable,
            "n_terminal": 0, "n_total": len(expected_task_ids),
            "finished_at": time.time(),
        })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
