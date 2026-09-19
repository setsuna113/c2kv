"""Production scheduler for the generality matrix (runs on ascend03).

Binds cells to cards: one long-lived engine per card, one measured cell per
card at a time. Long benchmarks first (appworld, then bfcl_long, then bfcl
base) — the order carries no adaptive meaning (thresholds/budgets are frozen
before any closed-loop cell starts). Driver types:
  c2kv cells (all 3 conditions)      -> c2kv_cell.py   (controller path)
  H2O/SnapKV/PyramidKV off conditions -> historykv_cell.py (proxy path)
  H2O/SnapKV/PyramidKV tracer cells  -> session_tracer_cell.py

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
TARGET_TOKENS = {"K0": 768, "K2": 1536}

ENGINE_PORT = {0: 36200, 1: 36201, 2: 36202, 3: 36203, 4: 36204, 5: 36205, 6: 36206, 7: 36207}

# driver assignment per (backend, condition)
def driver_for(backend: str, condition: str) -> str:
    if backend == "c2kv":
        return "c2kv"
    if condition in ("recovery_off_same_initial", "compression_full_budget"):
        return "historykv_off"
    return "session_tracer"


def calibration_receipt(cell: dict) -> tuple[Path, dict | None]:
    """Return the frozen threshold receipt for a tracer cell.

    Calibration is an input to a closed-loop cell, not an optional runtime
    fallback.  Loading it here keeps the generated cell.json immutable while
    making the launch manifest self-contained.
    """
    path = (GENERATION_ROOT / "calibration" / cell["backend"] /
            cell["working_point"] / "threshold.json")
    if not path.exists():
        return path, None
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return path, None
    if (value.get("backend") != cell["backend"] or
            value.get("working_point") != cell["working_point"] or
            value.get("target_tokens") != TARGET_TOKENS[cell["working_point"]]):
        return path, value
    if value.get("ready_for_matrix") is not True:
        return path, value
    if (isinstance(value.get("threshold"), bool) or
            not isinstance(value.get("threshold"), (int, float))):
        return path, value
    return path, value


def calibration_is_ready(cell: dict, receipt: dict | None) -> bool:
    return bool(receipt and receipt.get("ready_for_matrix") is True and
                receipt.get("backend") == cell["backend"] and
                receipt.get("working_point") == cell["working_point"] and
                receipt.get("target_tokens") == TARGET_TOKENS[cell["working_point"]] and
                not isinstance(receipt.get("threshold"), bool) and
                isinstance(receipt.get("threshold"), (int, float)))


# pyramidkv verified on NPU 2026-09-19 (full task, no traceback, 39x
# reference_attention_ok + pyramidkv_official_headwise evidence) — released.
# pyramidkv x BFCL cells held back 2026-09-19: the engine aborts most
# multi-turn BFCL requests with [finish_abort] PERSISTENT_HISTORY_SESSION_
# UNAVAILABLE (~85% churn, receipts preserved); appworld pyramidkv cells
# produce normally.  Re-enable when the engine session bug is fixed.
# Production pause lifted for AppWorld 2026-09-19 (user directive): h2o/snapkv
# zeros verified REAL (50-turn agent trajectories with genuine API actions; a
# model-initiated complete_task(fail)) — valid data.  Still blocked:
# - ALL bfcl panels: handler decode failure (2568/2599 empty responses, 0 correct)
# - pyramidkv appworld: empty assistant content / "None" actions at B=3200
#   (reference-path output rendering bug)
BLOCKED_BACKENDS: set[str] = set()
BLOCKED_CELL_KEYS = {
    ("pyramidkv", "bfcl_base"), ("pyramidkv", "bfcl_long_context"),
    ("pyramidkv", "appworld"),
    ("h2o", "bfcl_base"), ("h2o", "bfcl_long_context"),
    ("snapkv", "bfcl_base"), ("snapkv", "bfcl_long_context"),
    ("c2kv", "bfcl_base"), ("c2kv", "bfcl_long_context"),
}



def cells_ready(cell: dict) -> bool:
    if cell["backend"] in BLOCKED_BACKENDS:
        return False
    if (cell["backend"], cell.get("benchmark_key")) in BLOCKED_CELL_KEYS:
        return False
    if (cell["backend"], "*") in BLOCKED_CELL_KEYS:
        return False
    # AppWorld uses the same event-native worker as BFCL and performs its
    # official scorer/summary validation per task.  It is therefore queued by
    # the same scheduler once the code path is installed.
    if cell["condition"] != "tracer_history":
        return True
    _, receipt = calibration_receipt(cell)
    return calibration_is_ready(cell, receipt)


def enumerate_cells() -> list[dict]:
    rows = []
    for bench_dir in sorted(RESULTS.iterdir()):
        if not bench_dir.is_dir():
            continue
        for backend_dir in sorted(bench_dir.iterdir()):
            if not backend_dir.is_dir():
                continue
            for wp_dir in sorted(backend_dir.iterdir()):
                if not wp_dir.is_dir():
                    continue
                for cond_dir in sorted(wp_dir.iterdir()):
                    if not cond_dir.is_dir():
                        continue
                    cell_json = cond_dir / "cell.json"
                    if cell_json.exists():
                        rows.append(json.loads(cell_json.read_text()))
    return rows


BENCH_ORDER = {"appworld": 0, "bfcl_long_context": 1, "bfcl_base": 2}
COND_ORDER = {"tracer_history": 0, "compression_full_budget": 1, "recovery_off_same_initial": 2}
# pyramidkv promoted to front per user 2026-09-19 (support fixed, now high priority)
BACKEND_ORDER = {"pyramidkv": 0, "c2kv": 1, "h2o": 2, "snapkv": 3}


def queue_cells(cells: list[dict], only_ready: bool = True) -> list[dict]:
    def sort_key(c):
        return (BACKEND_ORDER.get(c["backend"], 5),
                BENCH_ORDER.get(c["benchmark_key"], 9),
                COND_ORDER.get(c["condition"], 9))
    ready = [c for c in cells if not only_ready or cells_ready(c)]
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
    if cell["condition"] == "tracer_history":
        receipt_path, receipt = calibration_receipt(cell)
        if not calibration_is_ready(cell, receipt):
            raise RuntimeError(
                f"tracer cell requires ready calibration receipt: {receipt_path}")
        cell["threshold"] = receipt["threshold"]
        cell["threshold_status"] = "calibrated"
        cell["calibration_receipt"] = str(receipt_path)
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
    elif driver == "session_tracer":
        cmd = [PY_SGL, str(SRC / "generality" / "session_tracer_cell.py"),
               "--cell", str(cell_path),
               "--port-base", str(38000 + port_offset * 20)]
    else:
        raise RuntimeError(f"unsupported cell driver: {driver}")
    log.parent.mkdir(parents=True, exist_ok=True)
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


def engine_healthy(card: int, timeout: float = 3.0) -> bool:
    import urllib.request
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(f"http://127.0.0.1:{ENGINE_PORT[card]}/health", timeout=timeout) as r:
            return r.status == 200
    except OSError:
        return False


def live_cell_dirs() -> set[str]:
    """cell_dirs that already have a driver process (any scheduler run).

    Prevents double-driving one cell across scheduler restarts: orphaned
    drivers from a previous run keep working on their cell; the new scheduler
    skips those cells instead of launching a duplicate.
    """
    import re
    try:
        out = subprocess.run(
            ["pgrep", "-af",
             "historykv_cell[.]py|c2kv_cell[.]py|session_tracer_cell[.]py"],
            capture_output=True, text=True, timeout=10).stdout
    except (subprocess.SubprocessError, OSError):
        return set()
    dirs = set()
    for line in out.splitlines():
        m = re.search(r"--cell (\S+)", line)
        if not m:
            continue
        try:
            manifest = json.loads(Path(m.group(1)).read_text())
            d = manifest.get("cell_dir")
            if d:
                dirs.add(str(d))
        except (OSError, json.JSONDecodeError):
            continue
    return dirs


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cards", type=int, nargs="*", default=[0, 1, 2, 3, 4, 5, 7])
    parser.add_argument("--include-pending", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-cells", type=int, default=None)
    args = parser.parse_args(argv)

    ensure_engines(args.cards)
    # brief warmup wait for engines, then proceed with whatever is healthy;
    # the per-tick health loop handles cards that recover later
    deadline = time.monotonic() + 120
    pending = set(args.cards)
    while pending and time.monotonic() < deadline:
        pending = {c for c in pending if not engine_healthy(c)}
        if pending:
            time.sleep(15)
    if pending:
        print(json.dumps({"event": "engines_not_ready", "cards": sorted(pending)}), flush=True)

    only_ready = not args.include_pending
    running: dict[int, tuple[subprocess.Popen, dict]] = {}
    port_offset = {card: i for i, card in enumerate(args.cards)}
    launched = 0
    cells: list[dict] = []
    fail_counts: dict[str, int] = {}
    dropped: set[str] = set()
    MAX_FAILS = 6
    engine_launch_at: dict[int, float] = {}
    first_derivation = True
    while True:
        for card in list(running):
            proc, cell = running[card]
            if proc.poll() is not None:
                rc = proc.returncode
                print(json.dumps({"event": "cell_exit", "cell": cell["cell_id"],
                                  "card": card, "rc": rc}), flush=True)
                del running[card]
                if not cell_done(cell):
                    fail_counts[cell["cell_id"]] = fail_counts.get(cell["cell_id"], 0) + 1
                    if fail_counts[cell["cell_id"]] < MAX_FAILS:
                        cells.append(cell)
                        print(json.dumps({"event": "cell_requeue",
                                          "cell": cell["cell_id"], "fails": fail_counts[cell["cell_id"]]}), flush=True)
                    else:
                        dropped.add(cell["cell_id"])
                        print(json.dumps({"event": "cell_dropped_fastfail",
                                          "cell": cell["cell_id"],
                                          "fails": fail_counts[cell["cell_id"]]}), flush=True)
        # per-tick engine health: relaunch crashed engines (throttled), never
        # bind a cell to a card whose engine is down or still booting
        healthy = []
        for card in args.cards:
            if engine_healthy(card):
                healthy.append(card)
                continue
            now = time.monotonic()
            if now - engine_launch_at.get(card, -1e9) > 600:
                engine_launch_at[card] = now
                ensure_engines([card])
            print(json.dumps({"event": "engine_unhealthy", "card": card}), flush=True)
        live = live_cell_dirs()
        # derive work when the queue runs dry; late-arriving calibration
        # receipts unlock tracer cells without a scheduler restart
        if not cells:
            cells = queue_cells(enumerate_cells(), only_ready=only_ready)
            if first_derivation and args.max_cells is not None:
                cells = cells[: args.max_cells]
            cells = [c for c in cells if not cell_done(c)
                     and str(c["cell_dir"]) not in live
                     and c["cell_id"] not in dropped]
            first_derivation = False
            if not cells and not running:
                break
        while cells and len(running) < len(healthy):
            cell = cells.pop(0)
            if cell_done(cell):
                print(json.dumps({"event": "cell_skip_done", "cell": cell["cell_id"]}), flush=True)
                continue
            if str(cell["cell_dir"]) in live:
                print(json.dumps({"event": "cell_skip_live", "cell": cell["cell_id"]}), flush=True)
                continue
            card = next(c for c in healthy if c not in running)
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
