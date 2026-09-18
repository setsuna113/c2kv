"""Generality run monitor (runs on ascend03; invoked by the supervision cron).

One pass: engine health per card, per-cell progress, infra-failure detection
and single retry, status snapshot appended to logs/monitor.jsonl.
Model failures are NEVER retried or masked; they stay as per-task receipts.
"""
from __future__ import annotations

import json
import subprocess
import time
import urllib.request
from pathlib import Path

GENERATION_ROOT = Path("/home/liuyancheng/c2kv-generality-20260918")
LOGS = GENERATION_ROOT / "logs"
RESULTS = GENERATION_ROOT / "results" / "closed_loop"
ENGINE_PORTS = {0: 36200, 1: 36201, 2: 36202, 3: 36203, 4: 36204, 5: 36205, 6: 36206, 7: 36207}


def opener():
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def engine_health() -> dict:
    out = {}
    for card, port in ENGINE_PORTS.items():
        try:
            with opener().open(f"http://127.0.0.1:{port}/health", timeout=4) as r:
                out[card] = "ok" if r.status == 200 else f"http{r.status}"
        except OSError as error:
            out[card] = f"down:{type(error).__name__}"
    return out


def npu_snapshot() -> dict:
    smi = subprocess.run(["npu-smi", "info"], capture_output=True, text=True).stdout
    cards = {}
    for line in smi.splitlines():
        parts = [p.strip() for p in line.split("|")]
        if len(parts) >= 5 and parts[1].isdigit():
            card = int(parts[1])
            hbm = parts[4].split("/")[0].strip()
            cards[card] = {"power_w": parts[3].split()[0], "hbm_mb_used": hbm,
                           "health": parts[2]}
    procs = subprocess.run(
        ["bash", "-lc",
         "npu-smi info | sed -n '/Process id/,$p' | grep -E '^\\| [0-9]' || true"],
        capture_output=True, text=True).stdout
    return {"cards": cards, "process_rows": [l.strip(" |") for l in procs.splitlines() if l.strip()]}


def cell_progress() -> list[dict]:
    rows = []
    for cell_json in sorted(RESULTS.glob("*/*/*/*/cell.json")):
        cell = json.loads(cell_json.read_text())
        cell_dir = Path(cell["cell_dir"])
        entry = {"cell_id": cell["cell_id"]}
        status = cell_dir / "cell_status.json"
        if status.exists():
            entry["cell_status"] = json.loads(status.read_text()).get("status")
            if entry["cell_status"] == "complete":
                rows.append(entry)
                continue
        n_done = n_failed = 0
        for done in cell_dir.glob("batches/*/done.json"):
            n_done += json.loads(done.read_text()).get("n_tasks", 0)
        for st in cell_dir.glob("tasks/*/done.json"):
            n_done += 1
        for st in cell_dir.glob("batches/*/status.json"):
            n_failed += json.loads(st.read_text()).get("n_tasks", 0)
        for st in cell_dir.glob("tasks/*/status.json"):
            n_failed += 1
        entry.update(n_completed=n_done, n_incomplete=n_failed,
                     n_total=len(cell["task_ids"]))
        running = (cell_dir / "progress.jsonl").exists()
        entry["started"] = running
        rows.append(entry)
    return rows


def active_cells() -> list[str]:
    out = subprocess.run(["bash", "-lc",
         "ps -eo args | grep -E 'c2kv_cell|historykv_cell|session_tracer_cell|scheduler[.]py' | grep -v grep | head -30"],
        capture_output=True, text=True).stdout
    return [l.strip()[:160] for l in out.splitlines() if l.strip()]


def main() -> int:
    snapshot = {
        "ts": time.time(),
        "engines": engine_health(),
        "npu": npu_snapshot(),
        "cells": cell_progress(),
        "active": active_cells(),
    }
    with (LOGS / "monitor.jsonl").open("a") as stream:
        stream.write(json.dumps(snapshot, ensure_ascii=False) + "\n")
    print(json.dumps(snapshot, indent=1)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
