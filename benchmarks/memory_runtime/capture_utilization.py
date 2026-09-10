"""One bounded official task to capture exact observable diagnostic prefixes."""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
from pathlib import Path

from official_pilot import ROOT, free_loopback_ports


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--proxy-python", required=True)
    parser.add_argument("--bench-python", required=True)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit("Capture output already exists; no rerun or resume")
    args.out.mkdir(parents=True)
    task_id = "multi_turn_base_1"
    port = free_loopback_ports(1)[0]
    command = [args.bench_python, str(ROOT / "benchmarks/run.py"),
               "--benchmark", "bfcl", "--arm", "full", "--backend", "sglang",
               "--upstream", args.upstream, "--checkpoint", args.checkpoint,
               "--reference-profile", "checkpoint-1088", "--query-projection", "base",
               "--model", "c2kv-agent", "--num-workers", "1",
               "--categories", "multi_turn_base", "--run-ids", task_id,
               "--no-upstream-retries", "--capture-request-views",
               "--bfcl-temperature", "0.001", "--bfcl-seed", "0",
               "--proxy-python", args.proxy_python, "--proxy-port", str(port),
               "--out", str(args.out / "full"), "--exact-out",
               "--run-name", args.out.name]
    receipt = {
        "schema": "a-runtime-utilization-capture-v1",
        "purpose": "capture exact native requests; development, preliminary, n=1",
        "task_id": task_id, "task_selection": "previously exposed development task",
        "requested_prefixes": [{"user_turn": 0, "step": 1}, {"user_turn": 1, "step": 1}],
        "maximum_tasks": 1, "maximum_wall_seconds": 360, "automatic_reruns": 0,
        "temperature": 0.001, "seed": 0, "max_completion_tokens": 4096,
        "command": command, "status": "frozen_before_model_request",
    }
    path = args.out / "receipt.json"
    save(path, receipt)
    environment = dict(os.environ)
    bypass = ",".join(filter(None, [environment.get("NO_PROXY", environment.get("no_proxy", "")), "127.0.0.1", "localhost", "::1"]))
    environment.update(NO_PROXY=bypass, no_proxy=bypass)
    with (args.out / "runner.out").open("w") as log:
        proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                env=environment, start_new_session=True)
        try:
            returncode = proc.wait(timeout=receipt["maximum_wall_seconds"])
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            receipt["status"] = "wall_budget_exhausted"
            save(path, receipt)
            raise SystemExit(receipt["status"])
    receipt["returncode"] = returncode
    rows = []
    for log in (args.out / "full/logs").glob("proxy_*.jsonl"):
        for number, line in enumerate(log.read_text().splitlines(), 1):
            if line.strip():
                rows.append((log, number, json.loads(line)))
    receipt["chat_generations_observed"] = len(rows)
    valid = bool(rows) and returncode == 0 and all(
        row.get("status") == "ok"
        and row.get("eval_context", {}).get("task_id") == task_id
        and row.get("sampling_request", {}).get("temperature") == receipt["temperature"]
        and row.get("sampling_request", {}).get("seed") == receipt["seed"]
        and isinstance(row.get("request_view", {}).get("messages"), list)
        and len(row.get("forwarded_request_views", [])) == 1
        for _, _, row in rows
    )
    prefixes = []
    for context in receipt["requested_prefixes"]:
        hits = [(log, number, row) for log, number, row in rows
                if all(row.get("eval_context", {}).get(key) == value for key, value in context.items())]
        if len(hits) != 1:
            valid = False
            continue
        log, number, row = hits[0]
        view = row["request_view"]
        body = {key: view[key] for key in ("model", "messages", "tools") if key in view}
        body.update(view["sampling"])
        prefixes.append({"case_id": f"{task_id}:t{context['user_turn']}:s{context['step']}",
                         "eval_context": row["eval_context"], "request_body": body,
                         "source_path": str(log.relative_to(args.out)), "source_line": number})
    save(args.out / "prefixes.json", {
        "schema": "a-runtime-utilization-prefixes-v1", "task_id": task_id,
        "scope": "exact captured input prefixes only; no response or scorer data",
        "prefixes": prefixes,
    })
    receipt["status"] = "completed" if valid else "invalid_capture"
    receipt["prefixes_captured"] = len(prefixes)
    save(path, receipt)
    print(json.dumps({"status": receipt["status"], "receipt": str(path)}))
    if not valid:
        raise SystemExit("Capture contract failed; do not generate a replacement automatically")


if __name__ == "__main__":
    main()
