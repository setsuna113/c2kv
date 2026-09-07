"""One finite development pilot through the existing official BFCL adapter."""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--proxy-python", required=True)
    parser.add_argument("--bench-python", required=True)
    parser.add_argument("--protocol-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
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
    if args.out.exists():
        raise SystemExit("Output already exists; this pilot does not rerun or resume")
    args.out.mkdir(parents=True)
    run_id = "a_bfcl_dev4_20260907_v1"
    task_ids = [f"multi_turn_base_{i}" for i in range(4)]
    variants = [("full", "full"), ("legacy", "c2kv4"), ("protect", "c2kv4")]
    commands = []
    for index, (name, arm) in enumerate(variants):
        command = [args.bench_python, str(ROOT / "benchmarks/run.py"),
                   "--benchmark", "bfcl", "--arm", arm,
                   "--upstream", args.upstream, "--backend", "sglang",
                   "--checkpoint", args.checkpoint, "--reference-profile", "checkpoint-1088",
                   "--query-projection", "base", "--model", "c2kv-agent",
                   "--num-workers", "1", "--categories", "multi_turn_base",
                   "--run-ids", ",".join(task_ids), "--no-upstream-retries",
                   "--proxy-python", args.proxy_python, "--proxy-port", str(35270 + index),
                   "--out", str(args.out / name), "--exact-out", "--run-name", run_id + "_" + name]
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
        task_selection="first four numeric multi_turn_base IDs, without success filtering",
        variants=[name for name, _ in variants], maximum_tasks=12,
        maximum_wall_seconds=1800, automatic_reruns=0,
        sdk_retries=0, proxy_transport_retries=0, cache_miss_retries=0,
        generation_max_completion_tokens=4096,
        generation_sampling="unchanged official BFCL generate defaults; recorded in harness output",
        protocol_receipts=[str(args.protocol_root / name / "receipt.json") for name in ["protocol_v1", "protocol_remaining_v1"]],
        commands=commands, status="frozen_before_first_official_request", results=[],
    )
    receipt_path = args.out / "pilot.json"
    save(receipt_path, manifest)
    deadline = time.monotonic() + manifest["maximum_wall_seconds"]
    for item in commands:
        manifest["status"] = "running"
        manifest["active_variant"] = item["variant"]
        save(receipt_path, manifest)
        with (args.out / (item["variant"] + ".out")).open("w") as log:
            proc = subprocess.Popen(item["argv"], stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                returncode = proc.wait(timeout=max(1, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGTERM)
                proc.wait(timeout=10)
                manifest["status"] = "wall_budget_exhausted"
                save(receipt_path, manifest)
                raise SystemExit("Finite BFCL wall budget exhausted")
        manifest["results"].append(dict(variant=item["variant"], returncode=returncode))
        if returncode:
            manifest["status"] = "stopped_on_runner_error"
            save(receipt_path, manifest)
            raise SystemExit(f"BFCL runner failed for {item['variant']}: {returncode}")
        save(receipt_path, manifest)
    manifest["status"] = "completed"
    manifest.pop("active_variant", None)
    save(receipt_path, manifest)
    print(json.dumps(dict(status="completed", receipt=str(receipt_path))))


if __name__ == "__main__":
    main()
