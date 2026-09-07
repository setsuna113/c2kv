"""One finite development pilot through the existing official BFCL adapter."""
from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def free_loopback_ports(count):
    """Let the OS choose distinct ports; runner checks ownership again."""
    reservations = []
    try:
        for _ in range(count):
            reservation = socket.socket()
            reservation.bind(("127.0.0.1", 0))
            reservations.append(reservation)
        return [reservation.getsockname()[1] for reservation in reservations]
    finally:
        for reservation in reservations:
            reservation.close()


def design_spec(name):
    if name == "first-dev4":
        return dict(
            task_ids=[f"multi_turn_base_{i}" for i in range(4)],
            variants=[("full", "full"), ("legacy", "c2kv4"), ("protect", "c2kv4")],
            task_selection="first four numeric multi_turn_base IDs, without success filtering",
            command_extra=[])
    if name == "lease-dev2":
        return dict(
            task_ids=["multi_turn_base_1", "multi_turn_base_30"],
            variants=[("full", "full"), ("legacy", "c2kv4"), ("protect", "c2kv4"),
                      ("recover_once", "c2kv4"), ("persistent", "c2kv4"),
                      ("no_gist", "full")],
            task_selection=("two previously exposed development tasks chosen for "
                            "layout/lease diagnostics; no held-out claim"),
            command_extra=["--capture-request-views", "--bfcl-temperature", "0.001",
                           "--bfcl-seed", "0"])
    raise ValueError(f"Unknown pilot design: {name}")


def require_utilization_gate(protocol_root):
    directory = protocol_root / "utilization_probe_v1"
    receipt = json.loads((directory / "receipt.json").read_text())
    if (receipt.get("status") != "completed" or receipt.get("chat_attempts") != 24
            or receipt.get("chat_completed") != 24
            or receipt.get("lease_gate_passed") is not True):
        raise SystemExit("Utilization/lease gate did not complete its frozen 24 cells")
    cells = []
    for path in directory.glob("*.json"):
        row = json.loads(path.read_text())
        if isinstance(row, dict) and isinstance(row.get("cell_id"), str):
            cells.append(row)
    if len(cells) != 24 or len({row["cell_id"] for row in cells}) != 24:
        raise SystemExit("Utilization/lease gate lacks exactly 24 original cell JSON files")
    for row in cells:
        metadata = (row.get("counts") or {}).get("memory_runtime") or {}
        if (row.get("status") != "completed"
                or metadata.get("raw_prompt_tokens_verified_by_backend") is not True
                or metadata.get("byte_geometry_verified_by_backend") is not True):
            raise SystemExit("Utilization/lease cell lacks completed token/byte verification")
    return directory / "receipt.json"


def terminate_process_group(proc, grace_seconds=10):
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except OSError:
        pass
    try:
        return proc.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except OSError:
            pass
        return proc.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--proxy-python", required=True)
    parser.add_argument("--bench-python", required=True)
    parser.add_argument("--protocol-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--design", choices=("first-dev4", "lease-dev2"),
                        default="first-dev4")
    args = parser.parse_args()
    first = json.loads((args.protocol_root / "protocol_v1/receipt.json").read_text())
    remainder = json.loads((args.protocol_root / "protocol_remaining_v1/receipt.json").read_text())
    all_rows = first["request_receipts"] + remainder["request_receipts"]
    audit = json.loads((args.protocol_root / "protocol_cpu_audit.json").read_text())
    if audit["status"] != "passed" or not all(r["selected_view_unchanged"] and r["raw_token_parity"] for r in audit["rows"]):
        raise SystemExit("Corrected tokenizer accounting did not reproduce executed protocol views")
    if (remainder["status"] != "completed" or len(all_rows) != 16
            or first["requests_attempted"] + remainder["requests_attempted"] != 16):
        raise SystemExit("Protocol gate did not complete exactly sixteen requests")
    if any(not row["memory_runtime"]["byte_geometry_verified_by_backend"]
           for row in all_rows if row.get("memory_runtime")):
        raise SystemExit("Protocol gate lacks backend byte geometry verification")
    utilization_receipt = (require_utilization_gate(args.protocol_root)
                           if args.design == "lease-dev2" else None)
    if args.out.exists():
        raise SystemExit("Output already exists; this pilot does not rerun or resume")
    args.out.mkdir(parents=True)
    run_id = "a_" + args.out.name
    design = design_spec(args.design)
    task_ids = design["task_ids"]
    variants = design["variants"]
    proxy_ports = free_loopback_ports(len(variants))
    commands = []
    for index, (name, arm) in enumerate(variants):
        command = [args.bench_python, str(ROOT / "benchmarks/run.py"),
                   "--benchmark", "bfcl", "--arm", arm,
                   "--upstream", args.upstream, "--backend", "sglang",
                   "--checkpoint", args.checkpoint, "--reference-profile", "checkpoint-1088",
                   "--query-projection", "base", "--model", "c2kv-agent",
                   "--num-workers", "1", "--categories", "multi_turn_base",
                   "--run-ids", ",".join(task_ids), "--no-upstream-retries",
                   "--proxy-python", args.proxy_python, "--proxy-port", str(proxy_ports[index]),
                   "--out", str(args.out / name), "--exact-out", "--run-name", run_id + "_" + name]
        command += design["command_extra"]
        if name != "full":
            config = json.loads((HERE / f"configs/{name}.json").read_text())
            config["run_id"] = run_id + "_" + name
            path = args.out / (name + ".config.json")
            save(path, config)
            command += ["--memory-runtime-config", str(path), "--memory-tokenizer", args.checkpoint]
        commands.append(dict(variant=name, argv=command))
    manifest = dict(
        schema="a-runtime-bfcl-dev-pilot-v1", run_id=run_id,
        scope="development pilot; preliminary, n=1", task_ids=task_ids,
        task_selection=design["task_selection"],
        variants=[name for name, _ in variants], maximum_tasks=12,
        maximum_wall_seconds=1800, automatic_reruns=0,
        sdk_retries=0, proxy_transport_retries=0, cache_miss_retries=0,
        generation_max_completion_tokens=4096,
        generation_sampling="unchanged official BFCL generate defaults; recorded in harness output",
        protocol_receipts=[str(args.protocol_root / name / "receipt.json") for name in ["protocol_v1", "protocol_remaining_v1"]],
        commands=commands, status="frozen_before_first_official_request", results=[],
    )
    if args.design == "lease-dev2":
        manifest.update(
            design="lease-dev2",
            generation_request_sampling={
                "temperature": 0.001, "seed": 0, "max_completion_tokens": 4096})
        manifest["generation_sampling"] = "explicit request and forwarded sampling logged"
        manifest["protocol_receipts"].append(str(utilization_receipt))
    receipt_path = args.out / "pilot.json"
    save(receipt_path, manifest)
    deadline = time.monotonic() + manifest["maximum_wall_seconds"]
    for item in commands:
        manifest["status"] = "running"
        manifest["active_variant"] = item["variant"]
        save(receipt_path, manifest)
        with (args.out / (item["variant"] + ".out")).open("w") as log:
            environment = dict(os.environ)
            bypass = ",".join(filter(None, [environment.get("NO_PROXY", environment.get("no_proxy", "")), "127.0.0.1", "localhost", "::1"]))
            environment.update(NO_PROXY=bypass, no_proxy=bypass)
            proc = subprocess.Popen(item["argv"], stdout=log, stderr=subprocess.STDOUT,
                                    start_new_session=True, env=environment)
            try:
                returncode = proc.wait(timeout=max(1, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                terminate_process_group(proc)
                manifest["status"] = "wall_budget_exhausted"
                save(receipt_path, manifest)
                raise SystemExit("Finite BFCL wall budget exhausted")
        manifest["results"].append(dict(variant=item["variant"], returncode=returncode))
        if returncode:
            manifest["status"] = "stopped_on_runner_error"
            save(receipt_path, manifest)
            raise SystemExit(f"BFCL runner failed for {item['variant']}: {returncode}")
        request_logs = list((args.out / item["variant"] / "logs").glob("proxy_*.jsonl"))
        requests = [json.loads(line) for path in request_logs for line in path.read_text().splitlines() if line.strip()]
        observed_tasks = {row.get("eval_context", {}).get("task_id") for row in requests}
        invalid_requests = (not requests or not set(task_ids).issubset(observed_tasks))
        if args.design == "lease-dev2":
            invalid_requests = (not requests or observed_tasks != set(task_ids)
                                or any(row.get("status") != "ok" for row in requests))
        if invalid_requests:
            manifest["status"] = ("invalid_missing_model_requests"
                                  if args.design == "first-dev4"
                                  else "invalid_request_coverage_or_runtime")
            if args.design == "lease-dev2":
                first_non_ok = next(
                    (row for row in requests if row.get("status") != "ok"), None)
                if first_non_ok is not None:
                    manifest["first_non_ok_request"] = {
                        "status": first_non_ok.get("status"),
                        "error": first_non_ok.get("error"),
                    }
            save(receipt_path, manifest)
            raise SystemExit("Official scores are invalid: selected tasks did not reach the model proxy")
        save(receipt_path, manifest)
    manifest["status"] = "completed"
    manifest.pop("active_variant", None)
    save(receipt_path, manifest)
    print(json.dumps(dict(status="completed", receipt=str(receipt_path))))


if __name__ == "__main__":
    main()
