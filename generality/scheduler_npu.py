"""Production scheduler for the generality matrix (runs on ascend03).

Binds cells to cards: one long-lived engine per card; one measured cell per
card by default, with explicitly bounded concurrency for disjoint sessions.
Long benchmarks first (appworld, then bfcl_long, then bfcl
base) — the order carries no adaptive meaning (thresholds/budgets are frozen
before any closed-loop cell starts). Driver types:
  c2kv cells (all 3 conditions)      -> c2kv_cell.py   (controller path)
  H2O/SnapKV/PyramidKV off conditions -> historykv_cell.py (proxy path)
  H2O/SnapKV/PyramidKV tracer cells  -> session_tracer_cell.py

Progress/resume: every driver writes per-batch done.json; rerunning a cell
skips completed batches. At most six cell launches are recorded across
scheduler restarts; model failures are terminal per-task records.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

try:
    from .completion_contract import status_matches_manifest
except ImportError:
    from completion_contract import status_matches_manifest

GENERATION_ROOT = Path("/home/liuyancheng/c2kv-generality-20260918")
SRC = GENERATION_ROOT / "src"
LOGS = GENERATION_ROOT / "logs"
RESULTS = GENERATION_ROOT / "results" / "closed_loop"
PY_SGL = "/home/liuyancheng/envs/sgl/bin/python"
TARGET_TOKENS = {"K0": 768, "K2": 1536}
MAX_ATTEMPTS = 6  # total cell launches, including the first, across restarts
MAX_DRIVERS_PER_CARD = 2
DRIVER_PORT_BASE = 45000
DRIVER_PORT_STRIDE = 1000

ENGINE_PORT = {0: 36200, 1: 36201, 2: 36202, 3: 36203, 4: 36204, 5: 36205, 6: 36206, 7: 36207}

# driver assignment per (backend, condition)
def driver_for(backend: str, condition: str) -> str:
    if backend == "c2kv":
        return "c2kv"
    if condition in ("recovery_off_same_initial", "compression_full_budget"):
        return "historykv_off"
    return "session_tracer"


SUPPORTED_BENCHMARKS = {
    "c2kv": {"bfcl_base", "bfcl_long_context", "appworld"},
    "historykv_off": {"bfcl_base", "bfcl_long_context", "appworld",
                      "toolsandbox", "acebench"},
    "session_tracer": {"bfcl_base", "bfcl_long_context", "appworld"},
}


def driver_supports_cell(cell: dict) -> bool:
    driver = driver_for(cell["backend"], cell["condition"])
    return cell.get("benchmark_key") in SUPPORTED_BENCHMARKS[driver]


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


# Production is paused for all backends pending per-backend NPU validation and
# audit of prior attempt receipts.  A scheduler restart or --include-pending
# must not release the hold.  The narrower cell holds remain conservative
# defaults for any later selective backend release.
BLOCKED_BACKENDS: set[str] = {"c2kv", "h2o", "snapkv", "pyramidkv"}
BLOCKED_CELL_KEYS = {
    ("pyramidkv", "bfcl_base"), ("pyramidkv", "bfcl_long_context"),
    ("pyramidkv", "appworld"),
    ("h2o", "bfcl_base"), ("h2o", "bfcl_long_context"),
    ("snapkv", "bfcl_base"), ("snapkv", "bfcl_long_context"),
    ("c2kv", "bfcl_base"), ("c2kv", "bfcl_long_context"),
}



def cell_blocked(cell: dict) -> bool:
    if not driver_supports_cell(cell):
        return True
    if cell["backend"] in BLOCKED_BACKENDS:
        return True
    if (cell["backend"], cell.get("benchmark_key")) in BLOCKED_CELL_KEYS:
        return True
    if (cell["backend"], "*") in BLOCKED_CELL_KEYS:
        return True
    return False


def cells_ready(cell: dict) -> bool:
    if cell_blocked(cell):
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
# Preserve the existing priority choice from the canonical scheduler entry.
BACKEND_ORDER = {"c2kv": 0, "h2o": 1, "snapkv": 2, "pyramidkv": 9}


def queue_cells(cells: list[dict], only_ready: bool = True) -> list[dict]:
    def sort_key(c):
        return (BACKEND_ORDER.get(c["backend"], 5),
                BENCH_ORDER.get(c["benchmark_key"], 9),
                COND_ORDER.get(c["condition"], 9))
    ready = [c for c in cells if not cell_blocked(c)
             and (not only_ready or cells_ready(c))]
    return sorted(ready, key=sort_key)


def cell_done(cell: dict) -> bool:
    status = Path(cell["cell_dir"]) / "cell_status.json"
    if not status.exists():
        return False
    try:
        return status_matches_manifest(cell, json.loads(status.read_text()))
    except (OSError, ValueError, TypeError):
        return False


def launch_cell(cell: dict, card: int, slot: int) -> subprocess.Popen:
    if cell_blocked(cell):
        raise RuntimeError(f"Cell is held by scheduler policy: {cell['cell_id']}")
    if not 0 <= slot < MAX_DRIVERS_PER_CARD:
        raise ValueError(f"Invalid driver slot: {slot}")
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
    cell["scheduler_port_slot"] = slot
    # The process reads its own immutable manifest.  Reusing cell_launch.json
    # would let a later failed launch rewrite a live driver's card identity.
    launch_dir = Path(cell["cell_dir"]) / "cell_launch_attempts"
    launch_dir.mkdir(parents=True, exist_ok=True)
    cell_path = launch_dir / f"launch-{time.time_ns()}-{os.getpid()}.json"
    with cell_path.open("x") as stream:
        json.dump(cell, stream, indent=2)
    log = LOGS / f"cell_{cell['cell_id']}_c{card}.log"
    port_base = DRIVER_PORT_BASE + (card * MAX_DRIVERS_PER_CARD + slot) * DRIVER_PORT_STRIDE
    env = os.environ.copy()
    if driver == "c2kv":
        env["ASCEND_RT_VISIBLE_DEVICES"] = str(card)   # controller-side torch_npu import safety
        cmd = [PY_SGL, str(SRC / "generality" / "c2kv_cell.py"),
               "--cell", str(cell_path), "--budgets",
               str(GENERATION_ROOT / "config" / "budgets_resolved.json"),
               "--port-base", str(port_base)]
    elif driver == "historykv_off":
        cmd = [PY_SGL, str(SRC / "generality" / "historykv_cell.py"),
               "--cell", str(cell_path),
               "--proxy-port", str(port_base)]
    elif driver == "session_tracer":
        cmd = [PY_SGL, str(SRC / "generality" / "session_tracer_cell.py"),
               "--cell", str(cell_path),
               "--port-base", str(port_base)]
    else:
        raise RuntimeError(f"unsupported cell driver: {driver}")
    # Resolve through the active source release, not the runtime's src symlink.
    cpu_launcher = Path(__file__).resolve().parents[1] / "tools" / "launch_cpu_controller.sh"
    cmd = ["bash", str(cpu_launcher), *cmd]
    log.parent.mkdir(parents=True, exist_ok=True)
    lock_dir = LOGS / "driver_locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    # Prevent duplicate driving of one cell across scheduler restarts while
    # allowing explicitly configured concurrency on the shared engine.
    cell_key = hashlib.sha256(str(Path(cell["cell_dir"]).resolve()).encode()).hexdigest()[:24]
    cmd = ["flock", "-n", str(lock_dir / f"cell-{cell_key}.lock"), *cmd]
    with log.open("ab") as stream:
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


def live_driver_assignments() -> dict[int, dict[str, dict]]:
    """Read cell and card ownership from every live driver, including orphans.

    Unknown ownership is an error: treating a failed process scan as an empty
    machine would allow another cell to start on an occupied card.
    """
    try:
        scan = subprocess.run(
            ["pgrep", "-af",
             "historykv_cell[.]py|c2kv_cell[.]py|session_tracer_cell[.]py"],
            capture_output=True, text=True, timeout=10,
        )
    except (subprocess.SubprocessError, OSError) as error:
        raise RuntimeError("Cannot inspect live cell drivers") from error
    if scan.returncode == 1 and not scan.stdout:
        return {}
    if scan.returncode != 0:
        raise RuntimeError(f"Live driver scan failed: rc={scan.returncode}")
    port_to_card = {port: card for card, port in ENGINE_PORT.items()}
    assignments: dict[int, dict[str, dict]] = {}
    for line in scan.stdout.splitlines():
        process = re.match(r"\s*\d+\s+(\S+)", line)
        if process is None:
            raise RuntimeError(f"Cannot parse live driver process: {line[:160]}")
        # pgrep also reports the `flock ... python driver.py` parent.  Count
        # only the Python child, otherwise one cell consumes two slots.
        if not Path(process.group(1)).name.startswith("python"):
            continue
        match = re.search(r"(?:^|\s)--cell(?:=|\s+)(\S+)", line)
        if match is None:
            raise RuntimeError(f"Live driver has no --cell manifest: {line[:160]}")
        path = Path(match.group(1))
        try:
            cell = json.loads(path.read_text())
            directory = Path(cell["cell_dir"])
            port = urlparse(cell["sglang_backend_url"]).port
            card = port_to_card[port]
            slot = cell.get("scheduler_port_slot")
            if slot is not None and (type(slot) is not int or
                                     not 0 <= slot < MAX_DRIVERS_PER_CARD):
                raise ValueError("invalid scheduler_port_slot")
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
            raise RuntimeError(f"Cannot identify live driver card from {path}") from error
        manifest = path.resolve()
        cell_directory = directory.resolve()
        if (manifest != cell_directory / "cell_launch.json" and
                manifest.parent != cell_directory / "cell_launch_attempts"):
            raise RuntimeError(f"Live driver manifest has mismatched cell_dir: {path}")
        if str(directory) in assignments.setdefault(card, {}):
            raise RuntimeError(f"Duplicate live drivers for cell: {directory}")
        assignments[card][str(directory)] = cell
    return assignments


def live_cell_dirs() -> set[str]:
    return set().union(*(set(cells) for cells in live_driver_assignments().values()))


def unmanaged_event_native_servers(proc_root: Path = Path("/proc")) -> list[dict]:
    """Find project controllers without a live cell driver ancestor.

    A detached event-native supervisor can keep its child and port alive after
    the driver exits.  Its card cannot be safely assigned from driver scans.
    The scheduler reports it and waits for an identity-checked cleanup; it
    never sends signals or changes the retained attempt files.
    """
    try:
        processes = list(proc_root.iterdir())
    except OSError as error:
        raise RuntimeError("Cannot inspect event-native server processes") from error
    roots = ((RESULTS.resolve()), (GENERATION_ROOT / "validation").resolve())

    def details(pid: int) -> tuple[list[str], int, int] | None:
        proc = proc_root / str(pid)
        try:
            argv = [part.decode("utf-8", "replace") for part in
                    (proc / "cmdline").read_bytes().split(b"\0") if part]
            fields = (proc / "stat").read_text().rsplit(") ", 1)[1].split()
            return argv, int(fields[1]), int(fields[19])
        except (OSError, ValueError, IndexError):
            return None  # process exited during the scan

    def managed_by_driver(parent: int) -> bool:
        visited = set()
        while parent > 1 and parent not in visited:
            visited.add(parent)
            info = details(parent)
            if info is None:
                return False
            argv, parent, _ = info
            if any(arg.endswith(("/generality/c2kv_cell.py",
                                   "/generality/historykv_cell.py",
                                   "/generality/session_tracer_cell.py")) for arg in argv):
                return True
        return False

    unmanaged = []
    for proc in processes:
        if not proc.name.isdigit():
            continue
        info = details(int(proc.name))
        if info is None:
            continue
        argv, parent, start_ticks = info
        if "-m" not in argv:
            continue
        index = argv.index("-m")
        if (index + 1 >= len(argv) or
                argv[index + 1] != "benchmarks.memory_runtime.event_native_server"):
            continue
        try:
            out = argv[argv.index("--out") + 1]
        except (ValueError, IndexError):
            continue
        path = Path(out).resolve()
        if not any(os.path.commonpath((str(root), str(path))) == str(root)
                   for root in roots):
            continue
        if managed_by_driver(parent):
            continue
        unmanaged.append({"pid": int(proc.name), "ppid": parent,
                          "start_ticks": start_ticks, "out": out,
                          "role": "child" if "--serve-child" in argv else "supervisor"})
    return sorted(unmanaged, key=lambda row: row["pid"])


def other_scheduler_pids(proc_root: Path = Path("/proc")) -> list[int]:
    """Find an older scheduler that did not acquire the new singleton lock."""
    try:
        processes = list(proc_root.iterdir())
    except OSError as error:
        raise RuntimeError("Cannot inspect scheduler processes") from error
    expected = {(SRC / "generality" / name).resolve()
                for name in ("scheduler.py", "scheduler_npu.py")}
    found = []
    for proc in processes:
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            argv = [part.decode("utf-8", "replace") for part in
                    (proc / "cmdline").read_bytes().split(b"\0") if part]
            cwd = (proc / "cwd").resolve()
        except OSError:
            continue  # process exited during the scan
        for arg in argv[1:]:
            if not arg.endswith(("generality/scheduler.py", "generality/scheduler_npu.py")):
                continue
            candidate = Path(arg)
            if not candidate.is_absolute():
                candidate = cwd / candidate
            if candidate.resolve() in expected:
                found.append(int(proc.name))
                break
    return sorted(found)


def acquire_scheduler_lock():
    import fcntl

    LOGS.mkdir(parents=True, exist_ok=True)
    stream = (LOGS / ".scheduler.lock").open("a+b")
    try:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        stream.close()
        raise RuntimeError("Another generality scheduler holds the singleton lock") from error
    return stream


def free_healthy_slots(cards: list[int], running: dict,
                       occupied: dict[int, dict[str, dict]],
                       max_drivers_per_card: int) -> list[tuple[int, int]]:
    free = []
    for card in cards:
        known = {directory: manifest.get("scheduler_port_slot")
                 for directory, manifest in occupied.get(card, {}).items()}
        for proc, cell, running_card, slot in running.values():
            if running_card == card and str(cell["cell_dir"]) not in known:
                known[str(cell["cell_dir"])] = slot
        slots = [slot for slot in known.values() if slot is not None]
        if len(slots) != len(set(slots)):
            raise RuntimeError(f"Multiple live drivers claim the same port slot on card {card}")
        used = set(slots)
        for slot in known.values():
            if slot is None:
                available = next((index for index in range(MAX_DRIVERS_PER_CARD)
                                  if index not in used), None)
                if available is not None:
                    used.add(available)
        if len(known) >= max_drivers_per_card:
            continue
        free.extend((card, slot) for slot in range(max_drivers_per_card)
                    if slot not in used)
    return free


def task_keys(cell: dict) -> set[tuple[str, str]] | None:
    ids = cell.get("task_ids")
    benchmark = cell.get("benchmark_key")
    if (not isinstance(benchmark, str) or not isinstance(ids, list)
            or not ids or any(not isinstance(item, str) or not item for item in ids)):
        return None
    return {(benchmark, item) for item in ids}


def may_share_engine(cell: dict, card: int, running: dict,
                     occupied: dict[int, dict[str, dict]]) -> bool:
    active = list(occupied.get(card, {}).values())
    active_dirs = set(occupied.get(card, {}))
    active.extend(other for _, other, running_card, _ in running.values()
                  if running_card == card and str(other["cell_dir"]) not in active_dirs)
    if not active:
        return True
    # Only two persistent proxies passed budget and peer-close isolation.
    # Controller/controller and mixed concurrency have not been validated.
    history_proxy = driver_for(cell["backend"], cell["condition"]) == "historykv_off"
    if not history_proxy or any(
            driver_for(other["backend"], other["condition"]) != "historykv_off"
            for other in active):
        return False
    keys = task_keys(cell)
    if keys is None:
        return False
    return all((other_keys := task_keys(other)) is not None and
               keys.isdisjoint(other_keys) for other in active)


def attempt_state_path(cell: dict) -> Path:
    return Path(cell["cell_dir"]) / "scheduler_attempts.json"


def attempt_count(cell: dict) -> int:
    path = attempt_state_path(cell)
    if not path.exists():
        return 0
    try:
        state = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Invalid scheduler attempt receipt: {path}") from error
    count = state.get("launch_attempts")
    if state.get("cell_id") != cell["cell_id"] or type(count) is not int or count < 0:
        raise RuntimeError(f"Invalid scheduler attempt receipt: {path}")
    return count


def reserve_attempt(cell: dict) -> int:
    count = attempt_count(cell) + 1
    if count > MAX_ATTEMPTS:
        raise RuntimeError(f"Cell exceeded scheduler attempt limit: {cell['cell_id']}")
    path = attempt_state_path(cell)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps({"cell_id": cell["cell_id"],
                                     "launch_attempts": count}) + "\n")
    os.replace(temporary, path)
    return count


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cards", type=int, nargs="+", required=True)
    parser.add_argument("--include-pending", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-cells", type=int, default=None)
    parser.add_argument("--max-drivers-per-card", type=int, default=1,
                        choices=range(1, MAX_DRIVERS_PER_CARD + 1))
    args = parser.parse_args(argv)
    if (not args.cards or len(args.cards) != len(set(args.cards)) or
            any(card not in ENGINE_PORT for card in args.cards)):
        parser.error("--cards must contain at least one distinct supported card ID")
    if args.max_cells is not None and args.max_cells < 0:
        parser.error("--max-cells must be nonnegative")
    if args.include_pending and not args.dry_run:
        parser.error("--include-pending is only supported with --dry-run")
    preview = queue_cells(enumerate_cells(), only_ready=not args.include_pending)
    if args.max_cells is not None:
        preview = preview[:args.max_cells]
    preview = [cell for cell in preview if not cell_done(cell)
               and attempt_count(cell) < MAX_ATTEMPTS]
    if args.dry_run:
        print(json.dumps({"event": "scheduler_dry_run", "cells":
                          [cell["cell_id"] for cell in preview]}), flush=True)
        return 0
    if not preview:
        print(json.dumps({"event": "scheduler_no_runnable_cells"}), flush=True)
        return 0
    with acquire_scheduler_lock():
        older = other_scheduler_pids()
        if older:
            raise RuntimeError(f"Another generality scheduler is active: {older}")
        if args.max_cells is not None:
            args.allowed_cell_ids = {cell["cell_id"] for cell in preview}
        return run_scheduler(args)


def run_scheduler(args) -> int:

    unmanaged = unmanaged_event_native_servers()
    if unmanaged:
        raise RuntimeError(f"Unmanaged event-native servers block scheduler start: {unmanaged[:12]}")
    initial_live = live_driver_assignments()
    start_cards = [card for card in args.cards if card not in initial_live]
    if start_cards:
        ensure_engines(start_cards)
    # brief warmup wait for engines, then proceed with whatever is healthy;
    # the per-tick health loop handles cards that recover later
    deadline = time.monotonic() + 120
    pending = set(args.cards) - set(initial_live)
    while pending and time.monotonic() < deadline:
        pending = {c for c in pending if not engine_healthy(c)}
        if pending:
            time.sleep(15)
    if pending:
        print(json.dumps({"event": "engines_not_ready", "cards": sorted(pending)}), flush=True)

    only_ready = not args.include_pending
    running: dict[int, tuple[subprocess.Popen, dict, int, int]] = {}
    launched = 0
    cells: list[dict] = []
    dropped: set[str] = set()
    engine_launch_at: dict[int, float] = {}
    allowed_cell_ids = getattr(args, "allowed_cell_ids", None)
    if args.max_cells is not None and allowed_cell_ids is None:
        allowed_cell_ids = {cell["cell_id"] for cell in
                            queue_cells(enumerate_cells(), only_ready=only_ready)[:args.max_cells]}
    while True:
        unmanaged = unmanaged_event_native_servers()
        if unmanaged:
            print(json.dumps({"event": "unmanaged_event_native_servers",
                              "count": len(unmanaged),
                              "examples": unmanaged[:12]}), flush=True)
            time.sleep(30)
            continue
        for pid in list(running):
            proc, cell, card, slot = running[pid]
            if proc.poll() is not None:
                rc = proc.returncode
                print(json.dumps({"event": "cell_exit", "cell": cell["cell_id"],
                                  "card": card, "rc": rc}), flush=True)
                del running[pid]
                if not cell_done(cell):
                    count = attempt_count(cell)
                    if count < MAX_ATTEMPTS:
                        cells.append(cell)
                        print(json.dumps({"event": "cell_requeue",
                                          "cell": cell["cell_id"], "fails": count}), flush=True)
                    else:
                        dropped.add(cell["cell_id"])
                        print(json.dumps({"event": "cell_dropped_fastfail",
                                          "cell": cell["cell_id"],
                                          "fails": count}), flush=True)
        live_by_card = live_driver_assignments()
        live = set().union(*(set(cells) for cells in live_by_card.values()))
        # A shared engine is never restarted under a live driver.  If it died,
        # let the affected driver fail and preserve its evidence first.
        healthy = []
        for card in args.cards:
            if engine_healthy(card):
                healthy.append(card)
                continue
            if (card in live_by_card or
                    any(running_card == card for _, _, running_card, _ in running.values())):
                print(json.dumps({"event": "engine_unhealthy_with_live_drivers",
                                  "card": card}), flush=True)
                continue
            now = time.monotonic()
            if now - engine_launch_at.get(card, -1e9) > 600:
                engine_launch_at[card] = now
                ensure_engines([card])
            print(json.dumps({"event": "engine_unhealthy", "card": card}), flush=True)
        # derive work when the queue runs dry; late-arriving calibration
        # receipts unlock tracer cells without a scheduler restart
        if not cells:
            cells = queue_cells(enumerate_cells(), only_ready=only_ready)
            if allowed_cell_ids is not None:
                cells = [cell for cell in cells if cell["cell_id"] in allowed_cell_ids]
            cells = [c for c in cells if not cell_done(c)
                     and str(c["cell_dir"]) not in live
                     and c["cell_id"] not in dropped
                     and attempt_count(c) < MAX_ATTEMPTS]
            if not cells and not running:
                if live:
                    print(json.dumps({"event": "waiting_for_live_drivers",
                                      "cards": sorted(live_by_card)}), flush=True)
                    time.sleep(30)
                    continue
                break
        index = 0
        while index < len(cells):
            cell = cells[index]
            if cell_done(cell):
                cells.pop(index)
                print(json.dumps({"event": "cell_skip_done", "cell": cell["cell_id"]}), flush=True)
                continue
            if str(cell["cell_dir"]) in live:
                cells.pop(index)
                print(json.dumps({"event": "cell_skip_live", "cell": cell["cell_id"]}), flush=True)
                continue
            free_slots = free_healthy_slots(
                healthy, running, live_by_card, args.max_drivers_per_card)
            assignment = next(((card, slot) for card, slot in free_slots
                               if may_share_engine(cell, card, running, live_by_card)), None)
            if assignment is None:
                index += 1
                continue
            cells.pop(index)
            card, slot = assignment
            attempt = reserve_attempt(cell)
            proc = launch_cell(cell, card, slot)
            running[proc.pid] = (proc, cell, card, slot)
            live.add(str(cell["cell_dir"]))
            live_by_card.setdefault(card, {})[str(cell["cell_dir"])] = {
                **cell, "scheduler_port_slot": slot}
            launched += 1
            print(json.dumps({"event": "cell_launch", "cell": cell["cell_id"],
                              "card": card, "slot": slot, "pid": proc.pid,
                              "attempt": attempt}), flush=True)
        time.sleep(30)
    print(json.dumps({"event": "scheduler_done", "launched": launched}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
