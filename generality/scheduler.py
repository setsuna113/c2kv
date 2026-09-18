"""Production scheduler for the generality matrix (runs on ascend03).

Binds cells to cards: one long-lived engine per card, one measured cell per
card at a time. Long benchmarks first (appworld, then bfcl_long, then bfcl
base) — the order carries no adaptive meaning (thresholds/budgets are frozen
before any closed-loop cell starts). Driver types:
  c2kv cells (all 3 conditions)      -> c2kv_cell.py   (controller path)
  h2o/snapkv off conditions          -> historykv_cell.py (proxy path)
  h2o/snapkv tracer cells            -> session_tracer_cell.py (pending build)

Progress/resume: every driver writes per-batch done.json; rerunning a cell
skips completed batches. Infra failures keep receipts and are retried once by
the monitor; model failures are terminal per-task records, never zero-scored.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

GENERATION_ROOT = Path("/home/liuyancheng/c2kv-generality-20260918")
SRC = GENERATION_ROOT / "src"
LOGS = GENERATION_ROOT / "logs"
RESULTS = GENERATION_ROOT / "results" / "closed_loop"
PY_SGL = "/home/liuyancheng/envs/sgl/bin/python"

ENGINE_PORT = {0: 36200, 1: 36201, 2: 36202, 3: 36203, 4: 36204, 5: 36205, 7: 36207}

# driver assignment per (backend, condition)
def driver_for(backend: str, condition: str) -> str:
    if backend == "c2kv":
        return "c2kv"
    if condition in ("recovery_off_same_initial", "compression_full_budget"):
        return "historykv_off"
    return "session_tracer"          # not yet available; queued last


def cells_ready(backend: str, condition: str, benchmark: str) -> bool:
    if backend in ("h2o", "snapkv"):
        if condition == "tracer_history":
            # session tracer needs its calibration threshold receipt AND the
            # embedding backend wiring for RRF retrieval (last open item)
            return False
        # off conditions verified end-to-end on NPU 2026-09-18 (marker-only
        # first hint + logs/ dir + per-task validation)
        if benchmark == "appworld":
            return False  # AppWorld attach still pending its smoke
        return True
    if condition == "tracer_history":
        # every tracer cell needs its calibrated threshold receipt first
        return False
    if benchmark == "appworld":
        # AppWorld paths are enabled only after their smoke passes
        return False
    return True


def enumerate_cells() -> list[dict]:
    rows = []
    for bench_dir in sorted(RESULTS.iterdir()):
        for backend_dir in sorted(bench_dir.iterdir()):
            for wp_dir in sorted(backend_dir.iterdir()):
                for cond_dir in sorted(wp_dir.iterdir()):
                    cell_json = cond_dir / "cell.json"
                    if cell_json.exists():
                        rows.append(json.loads(cell_json.read_text()))
    return rows


BENCH_ORDER = {"appworld": 0, "bfcl_long_context": 1, "bfcl_base": 2}
COND_ORDER = {"tracer_history": 0, "compression_full_budget": 1, "recovery_off_same_initial": 2}


def queue_cells(cells: list[dict], only_ready: bool = True) -> list[dict]:
    def sort_key(c):
        return (BENCH_ORDER.get(c["benchmark_key"], 9),
                COND_ORDER.get(c["condition"], 9))
    ready = [c for c in cells
             if not only_ready or cells_ready(c["backend"], c["condition"], c["benchmark_key"])]
    return sorted(ready, key=sort_key)


def cell_done(cell: dict) -> bool:
    status = Path(cell["cell_dir"]) / "cell_status.json"
    if not status.exists():
        return False
    try:
        return json.loads(status.read_text()).get("status") == "complete"
    except json.JSONDecodeError:
        return False


def launch_cell(cell: dict, card: int, port_offset: int) -> subprocess.Popen:
    driver = driver_for(cell["backend"], cell["condition"])
    cell = dict(cell)
    cell["sglang_backend_url"] = f"http://127.0.0.1:{ENGINE_PORT[card]}"
    cell_path = Path(cell["cell_dir"]) / "cell_launch.json"
    cell_path.write_text(json.dumps(cell, indent=2))
    log = LOGS / f"cell_{cell['cell_id']}_c{card}.log"
    env = os.environ.copy()
    if driver == "c2kv":
        env["ASCEND_RT_VISIBLE_DEVICES"] = str(card)   # controller-side torch_npu import safety
        cmd = [PY_SGL, str(SRC / "generality" / "c2kv_cell.py"),
               "--cell", str(cell_path), "--budgets",
               str(GENERATION_ROOT / "config" / "budgets_resolved.json"),
               "--port-base", str(37000 + port_offset * 1000)]
    elif driver == "historykv_off":
        cmd = [PY_SGL, str(SRC / "generality" / "historykv_cell.py"),
               "--cell", str(cell_path),
               "--proxy-port", str(37400 + port_offset * 20)]
    else:
        raise RuntimeError("session_tracer driver pending")
    stream = log.open("ab")
    return subprocess.Popen(cmd, cwd=str(GENERATION_ROOT), env=env,
                            stdout=stream, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL, start_new_session=True)


def ensure_engines(cards: list[int]) -> None:
    import urllib.request
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    for card in cards:
        port = ENGINE_PORT[card]
        try:
            with opener.open(f"http://127.0.0.1:{port}/health", timeout=3) as r:
                if r.status == 200:
                    continue
        except OSError:
            pass
        tag = f"gen_c{card}"
        log = LOGS / "engines" / f"{tag}.log"
        with log.open("ab") as stream:
            subprocess.Popen(
                ["bash", str(GENERATION_ROOT / "tools" / "launch_engine.sh"),
                 str(card), str(port), tag, "--max-running-requests", "4"],
                stdout=stream, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, start_new_session=True,
            )
        print(f"engine launch: card {card} port {port}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cards", type=int, nargs="*", default=[0, 1, 2, 3, 4, 5, 7])
    parser.add_argument("--include-pending", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-cells", type=int, default=None)
    args = parser.parse_args(argv)

    ensure_engines(args.cards)
    # wait for engines to become healthy before binding cells
    import urllib.request
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    deadline = time.monotonic() + 900
    pending = set(args.cards)
    while pending and time.monotonic() < deadline:
        for card in list(pending):
            try:
                with opener.open(f"http://127.0.0.1:{ENGINE_PORT[card]}/health", timeout=3) as r:
                    if r.status == 200:
                        pending.discard(card)
            except OSError:
                pass
        if pending:
            time.sleep(15)
    if pending:
        print(json.dumps({"event": "engines_not_ready", "cards": sorted(pending)}), flush=True)

    cells = queue_cells(enumerate_cells(), only_ready=not args.include_pending)
    if args.max_cells is not None:
        cells = cells[: args.max_cells]
    running: dict[int, tuple[subprocess.Popen, dict]] = {}
    port_offset = {card: i for i, card in enumerate(args.cards)}
    launched = 0
    while cells or running:
        for card in list(running):
            proc, cell = running[card]
            if proc.poll() is not None:
                print(json.dumps({"event": "cell_exit", "cell": cell["cell_id"],
                                  "card": card, "rc": proc.returncode}), flush=True)
                del running[card]
        while cells and len(running) < len(args.cards):
            cell = cells.pop(0)
            if cell_done(cell):
                print(json.dumps({"event": "cell_skip_done", "cell": cell["cell_id"]}), flush=True)
                continue
            card = next(c for c in args.cards if c not in running)
            proc = launch_cell(cell, card, port_offset[card])
            running[card] = (proc, cell)
            launched += 1
            print(json.dumps({"event": "cell_launch", "cell": cell["cell_id"],
                              "card": card, "pid": proc.pid}), flush=True)
        time.sleep(30)
    print(json.dumps({"event": "scheduler_done", "launched": launched}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
