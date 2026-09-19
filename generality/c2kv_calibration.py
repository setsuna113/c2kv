"""Run one staged C2KV threshold calibration behind a risk-score-only controller.

The output directory must be new.  This command never promotes its threshold
to the canonical matrix receipt; inspect the staged observations first.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import signal
import socket
import subprocess
import time
from pathlib import Path
from urllib.parse import urlparse

try:
    from . import c2kv_cell, cellplan
except ImportError:
    import c2kv_cell  # type: ignore
    import cellplan  # type: ignore

LABELS = ("/home/liuyancheng/c2kv-evidence-sets-20260916/"
          "prepared_v8/run/labels.json")


def load_rows(path: str, state_id: str | None, smoke: bool) -> list[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = [row for row in data.get("rows", []) if row.get("split") == "calibration"]
    if state_id is not None:
        rows = [row for row in rows if row.get("state_id") == state_id]
    if smoke:
        rows = rows[:1]
    if not rows:
        raise ValueError("no calibration rows selected")
    return rows


def plan(args: argparse.Namespace) -> dict:
    cell_path = Path(args.cell).resolve()
    cell = json.loads(cell_path.read_text(encoding="utf-8"))
    if cell.get("backend") != "c2kv" or cell.get("benchmark") != "bfcl":
        raise ValueError("C2KV calibration needs a C2KV BFCL tracer cell")
    if cell.get("condition") != "tracer_history":
        raise ValueError("C2KV calibration needs the tracer history view")
    if cell.get("threshold_status") == "calibrated":
        raise ValueError("Use an uncalibrated source cell; never reuse a matrix launch")
    rows = load_rows(args.labels, args.state_id, args.smoke)
    task_ids = list(dict.fromkeys(row["task_id"] for row in rows))
    if not task_ids:
        raise ValueError("no source tasks")
    out = Path(args.out).resolve()
    calibration_root = c2kv_cell.GENERATION_ROOT / "calibration" / "c2kv"
    if not out.is_relative_to(calibration_root):
        raise ValueError(f"staged output must be under {calibration_root}")
    if out.exists():
        raise FileExistsError(f"staged output already exists: {out}")
    if urlparse(args.sglang_backend_url).port is None:
        raise ValueError("SGLang endpoint needs an explicit port")
    budgets = json.loads((cellplan.CONFIG / "budgets_resolved.json").read_text())
    wp = budgets["working_points"][cell["working_point"]]
    if cell["budget_bytes"] != {
        "K": wp["history_allowance_bytes"],
        "R_max": wp["recovery_allowance_bytes"],
        "B": wp["common_cap_bytes"],
    }:
        raise ValueError("frozen cell budgets differ from resolved calibration budgets")
    staged = copy.deepcopy(cell)
    staged["cell_dir"] = str(out / "controller")
    staged["sglang_backend_url"] = args.sglang_backend_url
    staged["controller_path"] = str(out / "controller.json")
    staged["eval_policy_path"] = str(out / "eval_policy.json")
    controller, binding = c2kv_cell.calibration_controller_config(staged)
    policy = c2kv_cell.build_eval_policy(staged, budgets)
    server = c2kv_cell.server_command(
        staged, task_ids, out / "controller", args.controller_port,
        max_decisions=cell["caps"]["generation_attempts_per_task"] * len(rows),
        max_generation_calls=cell["caps"]["generation_attempts_per_task"] * len(rows),
        max_extraction_calls=cell["caps"]["extraction_calls_per_task"] * len(rows),
        max_wall_seconds=cell["caps"]["task_timeout"] * len(rows),
    )
    calibration = [
        cell["python_bench"], str(Path(__file__).with_name("calibrate.py").resolve()),
        "--backend", "c2kv", "--wp", cell["working_point"],
        "--engine-url", f"http://127.0.0.1:{args.controller_port}",
        "--steps-path", str(out / "controller" / "server" / "steps.jsonl"),
        "--model", cell["model_name"], "--labels", str(Path(args.labels).resolve()),
        "--max-completion-tokens", str(cell["caps"]["max_completion_tokens"]),
        "--out", str(out),
    ]
    if args.state_id:
        calibration += ["--state-id", args.state_id]
    if args.smoke:
        calibration.append("--smoke")
    return {
        "schema": "c2kv-risk-only-calibration-launch-v1",
        "source_cell": str(cell_path), "working_point": cell["working_point"],
        "state_count": len(rows), "state_ids": [row["state_id"] for row in rows],
        "task_ids": task_ids,
        "recovery_disabled_required": True,
        "selector_threshold_role": "inert_parser_value_not_a_calibrated_threshold",
        "controller": controller, "binding": binding, "eval_policy": policy,
        "server_command": server, "calibration_command": calibration,
        "out": str(out),
    }


def _owned_controller_group(group: int, server_out: Path) -> bool:
    """Do not signal a recycled process group after its controller has gone."""
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            fields = (proc / "stat").read_text().rsplit(") ", 1)[1].split()
            if int(fields[2]) != group:
                continue
            argv = [part.decode("utf-8", "replace") for part in
                    (proc / "cmdline").read_bytes().split(b"\0") if part]
            if "-m" not in argv or argv[argv.index("-m") + 1] != (
                    "benchmarks.memory_runtime.event_native_server"):
                continue
            if Path(argv[argv.index("--out") + 1]).resolve() == server_out:
                return True
        except (OSError, ValueError, IndexError):
            continue
    return False


def run(spec: dict, port: int) -> int:
    out = Path(spec["out"])
    with socket.socket() as probe:
        try:
            probe.bind(("127.0.0.1", port))
        except OSError as error:
            raise RuntimeError(f"controller port {port} is occupied") from error
    out.mkdir(parents=True, exist_ok=False)
    for name, value in (
        ("controller.json", spec["controller"]),
        ("risk_artifact_binding.json", spec["binding"]),
        ("eval_policy.json", spec["eval_policy"]),
        ("launch_plan.json", {key: value for key, value in spec.items()
                              if key not in {"controller", "binding", "eval_policy"}}),
    ):
        (out / name).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    env = os.environ.copy()
    env["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    runtime = c2kv_cell.RUNTIME
    env["PYTHONPATH"] = os.pathsep.join((str(runtime / "python"), str(runtime)))
    env["no_proxy"] = env["NO_PROXY"] = "127.0.0.1,localhost"
    for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        env.pop(key, None)
    with (out / "controller.log").open("wb") as log:
        server = subprocess.Popen(
            spec["server_command"], cwd=runtime, env=env,
            stdout=log, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, start_new_session=True)
        group = server.pid  # start_new_session makes this the process-group ID.
        server_out = out / "controller" / "server"
        previous_term = signal.getsignal(signal.SIGTERM)

        def interrupt(_signal, _frame):
            raise KeyboardInterrupt("C2KV calibration launcher received SIGTERM")

        signal.signal(signal.SIGTERM, interrupt)
        try:
            ready = out / "controller" / "server" / "ready.json"
            deadline = time.monotonic() + 600
            while not ready.exists():
                if server.poll() is not None:
                    raise RuntimeError(f"calibration controller exited rc={server.returncode}")
                if time.monotonic() >= deadline:
                    raise TimeoutError("calibration controller readiness timeout")
                time.sleep(2)
            with (out / "calibration.log").open("wb") as calibration_log:
                result = subprocess.run(
                    spec["calibration_command"], cwd=runtime,
                    stdout=calibration_log, stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL, check=False)
            return result.returncode
        finally:
            signal.signal(signal.SIGTERM, previous_term)
            if _owned_controller_group(group, server_out):
                try:
                    os.killpg(group, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            try:
                server.wait(timeout=20)
            except subprocess.TimeoutExpired:
                pass
            if _owned_controller_group(group, server_out):
                try:
                    os.killpg(group, signal.SIGKILL)
                except ProcessLookupError:
                    pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell", required=True)
    parser.add_argument("--sglang-backend-url", required=True)
    parser.add_argument("--controller-port", required=True, type=int)
    parser.add_argument("--out", required=True)
    parser.add_argument("--labels", default=LABELS)
    parser.add_argument("--state-id")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--run", action="store_true",
                        help="launch model-backed calibration after read-only plan preflight")
    args = parser.parse_args(argv)
    spec = plan(args)
    print(json.dumps({key: value for key, value in spec.items()
                      if key not in {"controller", "binding", "eval_policy"}}, indent=2))
    return run(spec, args.controller_port) if args.run else 0


if __name__ == "__main__":
    raise SystemExit(main())
