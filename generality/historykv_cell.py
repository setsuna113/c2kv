"""H2O/SnapKV/PyramidKV off-condition cell driver (runs on ascend03).

Drives the paper-harness measurement proxy + official benchmark adapters
against the long-lived NPU engine for the two bare-backend conditions:

  recovery_off_same_initial  -> arm gen_{method}_{k0|k2} (absolute K tokens)
  compression_full_budget    -> arm gen_{method}_{b0|b2} (absolute B tokens)

No Tracer machinery runs in these cells: no detector, no recovery, no extra
model calls. Per-task progress with resume for both BFCL and AppWorld.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

GENERATION_ROOT = Path("/home/liuyancheng/c2kv-generality-20260918")
PAPER = GENERATION_ROOT / "src" / "paper_harness"

ARM_OF = {
    ("h2o", "K0", "recovery_off_same_initial"): "gen_h2o_k0",
    ("h2o", "K2", "recovery_off_same_initial"): "gen_h2o_k2",
    ("h2o", "K0", "compression_full_budget"): "gen_h2o_b0",
    ("h2o", "K2", "compression_full_budget"): "gen_h2o_b2",
    ("snapkv", "K0", "recovery_off_same_initial"): "gen_snapkv_persistent_k0",
    ("snapkv", "K2", "recovery_off_same_initial"): "gen_snapkv_persistent_k2",
    ("snapkv", "K0", "compression_full_budget"): "gen_snapkv_persistent_b0",
    ("snapkv", "K2", "compression_full_budget"): "gen_snapkv_persistent_b2",
    ("pyramidkv", "K0", "recovery_off_same_initial"): "gen_pyramidkv_k0",
    ("pyramidkv", "K2", "recovery_off_same_initial"): "gen_pyramidkv_k2",
    ("pyramidkv", "K0", "compression_full_budget"): "gen_pyramidkv_b0",
    ("pyramidkv", "K2", "compression_full_budget"): "gen_pyramidkv_b2",
}


def target_tokens_for_cell(cell: dict) -> int:
    """Resolve the absolute K/B target from the frozen cell contract."""
    budget = cell.get("budget_tokens") or {}
    key = "B" if cell["condition"] == "compression_full_budget" else "K"
    try:
        value = int(budget[key])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"cell {cell.get('cell_id')!r} has no resolved budget_tokens[{key!r}]"
        ) from error
    if value < 1:
        raise ValueError(f"cell {cell.get('cell_id')!r} has invalid target {value}")
    return value


def _proxy_opener():
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def resolve_free_port(base: int, out: Path, arm: str) -> int:
    """Pick a genuinely free proxy port.

    The scheduler's per-card port formula collides whenever orphaned drivers
    or proxies from earlier scheduler runs still hold a card's base port; a
    healthy foreign proxy on that port would silently serve a DIFFERENT arm
    (the 2026-09-19 mis-wiring incident).  Bind-probe and walk up by 2 until
    the port is free; a stale-but-alive foreign proxy therefore fails the
    probe and the driver gets its own port.
    """
    import socket
    for candidate in range(base, base + 40, 2):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 0)
            try:
                sock.bind(("127.0.0.1", candidate))
            except OSError:
                print(json.dumps({
                    "event": "proxy_port_busy", "candidate": candidate,
                    "cell_dir": str(out), "arm": arm}), flush=True)
                continue
            return candidate
    raise RuntimeError(f"no free proxy port in [{base}, {base + 40})")


def start_proxy(arm: str, upstream: str, port: int, out: Path,
                target_tokens: int) -> subprocess.Popen:
    import uuid
    attempt_dir = out / "proxy_attempts" / f"a{time.time_ns()}_{uuid.uuid4().hex}"
    attempt_dir.mkdir(parents=True, exist_ok=False)
    log_path = attempt_dir / "proxy.log"
    request_log = attempt_dir / "proxy_requests.jsonl"
    env = os.environ.copy()
    # proxy.py uses bare imports (repair_policy, textarms, arms) resolved from
    # the benchmarks directory itself, plus package imports from the repo root
    env["PYTHONPATH"] = os.pathsep.join((str(PAPER), str(PAPER / "benchmarks")))
    env["no_proxy"] = env["NO_PROXY"] = "127.0.0.1,localhost"
    for k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        env.pop(k, None)
    with log_path.open("wb") as log:
        proc = subprocess.Popen(
            [sys.executable, "-m", "benchmarks.proxy",
             "--upstream", upstream, "--arm", arm, "--backend", "sglang",
             # generation arms carry a target_tokens placeholder; the resolved
             # per-cell K/B allowance must be passed explicitly (proxy.py contract)
             "--history-kv-target-tokens", str(target_tokens),
             "--port", str(port),
             "--request-log", str(request_log),
             "--telemetry-log", str(attempt_dir / "proxy_telemetry.jsonl")],
            cwd=str(PAPER), env=env, stdout=log, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, start_new_session=True,
        )
    print(json.dumps({"event": "proxy_attempt", "arm": arm,
                      "directory": str(attempt_dir)}), flush=True)
    try:
        deadline = time.monotonic() + 120
        opener = _proxy_opener()
        while time.monotonic() < deadline:
            try:
                with opener.open(f"http://127.0.0.1:{port}/health", timeout=3) as r:
                    if r.status == 200:
                        break
            except OSError:
                pass
            if proc.poll() is not None:
                raise RuntimeError(f"proxy exited rc={proc.returncode}; see {log_path}")
            time.sleep(1)
        else:
            raise TimeoutError("proxy health timeout")
        # Fail fast: a proxy that cannot relay one real generation would silently
        # turn every task into a connection failure (the 2026-09-18 incident).
        probe_sid = f"proxy-probe-{int(time.time())}-{os.getpid()}"
        probe = {
            "model": "gen-c1000", "messages": [
                {"role": "user", "content": "Reply with the single word OK."}],
            "temperature": 0, "max_tokens": 8,
            "c2kv_measurement_session_id": probe_sid,
        }
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            data=json.dumps(probe).encode(), headers={"Content-Type": "application/json",
                     "X-C2KV-Measurement-Request-Id": probe_sid})
        try:
            with opener.open(req, timeout=180) as r:
                body = json.load(r)
        finally:
            # release the probe's engine session: a leaked one 400s the next
            # cell's probe on the same engine
            try:
                opener.open(urllib.request.Request(
                    f"http://127.0.0.1:{port}/close_session",
                    data=json.dumps({"session_id": probe_sid}).encode(),
                    headers={"Content-Type": "application/json"}), timeout=10)
            except Exception:
                pass
        text = (body.get("choices") or [{}])[0].get("message", {}).get("content")
        if not text:
            raise RuntimeError("proxy probe returned no content; refusing to run cell")
        # arm-ownership check: a foreign proxy that somehow holds this port would
        # have served the probe but logged it into ITS OWN request log, not ours.
        # Our log must contain the probe under OUR arm or we refuse to run.
        import time as _t
        _deadline = _t.monotonic() + 10
        while _t.monotonic() < _deadline:
            if proc.poll() is not None:
                raise RuntimeError("owned proxy exited before ownership verification")
            try:
                lines = request_log.read_text().splitlines()
                matching = [json.loads(line) for line in lines if line.strip()]
                matching = [entry for entry in matching if entry.get("request_id") == probe_sid]
                if matching:
                    entry = matching[-1]
                    if entry.get("arm") != arm:
                        raise RuntimeError(
                            f"proxy port serves arm {entry.get('arm')!r}, expected {arm!r}; "
                            f"refusing to run cell (mis-wiring guard)")
                    break
            except FileNotFoundError:
                pass
            _t.sleep(0.5)
        else:
            raise RuntimeError("proxy probe left no request-log entry; refusing to run cell")
        return proc
    except BaseException:
        stop(proc)
        raise



def stop(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


# infra signatures in benchmark.log: transient engine/proxy/harness trouble
# that must be retried, never recorded as a model failure
INFRA_MARKERS = (
    "Connection refused", "502 Bad Gateway", "URLError", "ReadTimeout",
    "RemoteDisconnected", "ConnectionReset", "timed out", "timeout",
    "PREFIX_MISMATCH", "FileExistsError", "AdmissionRejected",
    "HTTPConnectionPool", "Max retries exceeded",
)


def looks_infra(log_text: str) -> bool:
    return any(marker in log_text for marker in INFRA_MARKERS)


def cached_terminal(out: Path) -> dict | None:
    """Return only an explicitly classified terminal method receipt."""
    status = out / "status.json"
    if not status.exists():
        return None
    try:
        receipt = json.loads(status.read_text())
    except json.JSONDecodeError:
        return None
    # Old workers used generic rc=1 as ``failed`` even for scorer crashes,
    # import errors, and transport failures.  Those receipts are evidence,
    # not a reason to skip an unscored task on resume.
    return receipt if receipt.get("terminal_kind") == "method_failure" else None


def write_receipt(out: Path, result: dict) -> None:
    previous = out / "status.json"
    if previous.exists():
        with (out / "status_history.jsonl").open("a", encoding="utf-8") as history:
            history.write(json.dumps({"preserved_at_ns": time.time_ns(),
                                      "previous_raw": previous.read_text(encoding="utf-8")}) + "\n")
    (out / "status.json").write_text(json.dumps(result, indent=2))


def write_done(out: Path, result: dict) -> None:
    done = out / "done.json"
    if done.exists():
        with (out / "done_history.jsonl").open("a", encoding="utf-8") as history:
            history.write(json.dumps({"preserved_at_ns": time.time_ns(),
                                      "previous_raw": done.read_text(encoding="utf-8")}) + "\n")
    done.write_text(json.dumps(result, indent=2), encoding="utf-8")


def bfcl_row_healthy(project_root: Path, task_id: str) -> bool:
    """Accept complete output or an explicitly classified benchmark failure."""
    from generality.bfcl_results import bfcl_row_is_valid
    for path in sorted((project_root / "result").glob("**/*.json")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict) or row.get("id") != task_id:
                continue
            if bfcl_row_is_valid(row, fc_model=True):
                return True
    return False


def existing_bfcl_result(out: Path, task_id: str) -> bool:
    """Find a scored row in the legacy or any retained attempt directory."""
    roots = [out / "bfcl", *(out / "attempts").glob("*/bfcl")]
    return any(root.exists() and bfcl_row_healthy(root, task_id) for root in roots)


def new_attempt_root(out: Path, benchmark: str) -> Path:
    """Keep failed harness output while giving each retry a fresh target."""
    import uuid
    attempt = out / "attempts" / f"a{time.time_ns()}_{uuid.uuid4().hex}"
    attempt.mkdir(parents=True, exist_ok=False)
    return attempt / benchmark


def run_bfcl_task(cell: dict, task_id: str, proxy_port: int) -> dict:
    out = Path(cell["cell_dir"]) / "tasks" / task_id
    done = out / "done.json"
    if existing_bfcl_result(out, task_id):
        if done.exists():
            try:
                prior = json.loads(done.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                prior = None
            if isinstance(prior, dict) and prior.get("status") == "completed":
                return prior
        result = {"task_id": task_id, "status": "completed",
                  "summary_degraded": True,
                  "recovered_from_existing_official_row": True}
        write_done(out, result)
        return result
    terminal = cached_terminal(out)
    if terminal is not None:
        return terminal  # recorded model failure: never retried, never zero-scored
    out.mkdir(parents=True, exist_ok=True)
    category = task_id.rsplit("_", 1)[0]
    if category not in ("multi_turn_base", "multi_turn_long_context"):
        raise ValueError(f"cannot derive BFCL category from task id: {task_id}")
    project_root = new_attempt_root(out, "bfcl")
    started = time.monotonic()
    script = f"""
import sys
sys.path.insert(0, {str(PAPER)!r})
from benchmarks.adapters.bfcl_adapter import run_bfcl
summary = run_bfcl(
    'http://127.0.0.1:{proxy_port}/v1',
    categories='{category}', mode='both', run_ids=['{task_id}'],
    model={cell['model_name']!r}, handler_name={cell['handler_name']!r},
    project_root={str(project_root)!r}, num_threads=1,
)
import json
print('SUMMARY:' + json.dumps(summary))
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PAPER)
    env["BFCL_PROJECT_ROOT"] = str(project_root)
    env["no_proxy"] = env["NO_PROXY"] = "127.0.0.1,localhost"
    for k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        env.pop(k, None)
    log_path = project_root.parent / "benchmark.log"
    with log_path.open("wb") as log:
        rc = subprocess.call(
            [cell["python_bench"], "-c", script],
            cwd=str(PAPER), env=env, stdout=log, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
        )
    log_text = log_path.read_text(errors="ignore")
    healthy = bfcl_row_healthy(project_root, task_id) if project_root.exists() else False
    result = {"task_id": task_id, "status": "completed" if rc == 0 else "failed",
              "returncode": rc, "wall_s": time.monotonic() - started}
    if rc == 0 and healthy:
        write_done(out, result)
        return result
    if rc == 0:
        # official row carries a traceback: connection failure mid-task, the
        # measurement is not an official score — infra, retryable
        result["status"] = "failed_validation"
        write_receipt(out, result)
        return result
    if healthy:
        # subprocess crashed after the official row was written (e.g. the
        # single-task summary stdev bug): the row itself is valid
        result["status"] = "completed"
        result["summary_degraded"] = True
        write_done(out, result)
        return result
    result["status"] = "infra_error" if looks_infra(log_text) else "failed_validation"
    write_receipt(out, result)
    return result


def _run_adapter_task(cell: dict, task_id: str, proxy_port: int,
                      benchmark: str) -> dict:
    """Drive one ToolSandbox scenario or ACEBench instance through the paper
    adapters (official CLI + official scorer), with the same receipt semantics
    as the other benchmarks.  The agent rides the arm proxy; the user
    simulator rides the raw engine endpoint (tau2 split)."""
    out = Path(cell["cell_dir"]) / "tasks" / task_id
    done = out / "done.json"
    if done.exists():
        return json.loads(done.read_text())
    terminal = cached_terminal(out)
    if terminal is not None:
        return terminal  # recorded model failure: never retried, never zero-scored
    out.mkdir(parents=True, exist_ok=True)
    attempt_root = new_attempt_root(out, benchmark)
    upstream = cell["sglang_backend_url"].rstrip("/")
    if benchmark == "toolsandbox":
        inner = f"""
import sys, json
sys.path.insert(0, {str(PAPER)!r})
from benchmarks.adapters import toolsandbox_adapter as ts
summary = ts.run_ts(
    'http://127.0.0.1:{proxy_port}/v1', {str(attempt_root)!r},
    test_mode=False, scenarios=['{task_id}'],
    benchmark_dir={cell['benchmark_dir']!r},
    python={cell['python_bench']!r},
    user_base_url='{upstream}/v1', model={cell['model_name']!r})
print('SUMMARY:' + json.dumps(summary, default=str))
"""
    else:
        inner = f"""
import sys, json
sys.path.insert(0, {str(PAPER)!r})
from benchmarks.adapters import acebench_adapter as ace
summary = ace.run_acebench(
    'http://127.0.0.1:{proxy_port}/v1', '{upstream}/v1', {str(attempt_root)!r},
    acebench_dir={cell['benchmark_dir']!r}, category='agent', language='en',
    model={cell['model_name']!r}, task_ids='{task_id}',
    python={cell['python_bench']!r})
print('SUMMARY:' + json.dumps(summary, default=str))
"""
    started = time.monotonic()
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PAPER)
    env["no_proxy"] = env["NO_PROXY"] = "127.0.0.1,localhost"
    for k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        env.pop(k, None)
    log_path = attempt_root.parent / "benchmark.log"
    with log_path.open("wb") as log:
        rc = subprocess.call(
            [cell["python_sgl"], "-c", inner],
            cwd=str(PAPER), env=env, stdout=log, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
        )
    log_text = log_path.read_text(errors="ignore")
    result = {"task_id": task_id, "status": "completed" if rc == 0 else "failed",
              "returncode": rc, "wall_s": time.monotonic() - started}
    if rc == 0:
        (out / "done.json").write_text(json.dumps(result, indent=2))
    else:
        result["status"] = "infra_error" if looks_infra(log_text) else "failed_validation"
        write_receipt(out, result)
    return result


def appworld_attempt_status(root: Path, task_id: str) -> bool | None:
    """Return None for no trajectory; reject only the known generation failure."""
    records = [path for path in root.rglob("results.json")
               if path.parent.name == f"task_{task_id}"] if root.exists() else []
    if not records:
        return None
    log = root.parent / "benchmark.log"
    if log.exists() and "vLLM generation Error:" in log.read_text(
            encoding="utf-8", errors="replace"):
        return False
    for path in records:
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        if (not isinstance(record, dict)
                or record.get("task_id", task_id) != task_id
                or record.get("termination_reason") == "generation_error"):
            return False
    return True


def appworld_result_healthy(root: Path, task_id: str) -> bool:
    return appworld_attempt_status(root, task_id) is True


def appworld_done_healthy(out: Path, task_id: str) -> bool:
    done = out / "done.json"
    try:
        receipt = json.loads(done.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if (not isinstance(receipt, dict) or receipt.get("status") != "completed"
            or receipt.get("task_id") != task_id):
        return False
    if "attempt_root" in receipt:
        relative = receipt["attempt_root"]
        if (not isinstance(relative, str) or Path(relative).is_absolute()
                or ".." in Path(relative).parts):
            return False
        return appworld_result_healthy(out / relative, task_id)
    # Legacy receipts did not identify their attempt. A known bad trajectory
    # in any retained attempt invalidates that ambiguous receipt.
    roots = [out / "appworld", *(out / "attempts").glob("*/appworld")]
    statuses = [appworld_attempt_status(root, task_id) for root in roots]
    statuses = [status for status in statuses if status is not None]
    return bool(statuses) and all(statuses)


def run_appworld_task(cell: dict, task_id: str, proxy_port: int) -> dict:
    out = Path(cell["cell_dir"]) / "tasks" / task_id
    done = out / "done.json"
    if appworld_done_healthy(out, task_id):
        return json.loads(done.read_text())
    terminal = cached_terminal(out)
    if terminal is not None:
        return terminal  # recorded model failure: never retried, never zero-scored
    out.mkdir(parents=True, exist_ok=True)
    attempt_root = new_attempt_root(out, "appworld")
    started = time.monotonic()
    script = f"""
import sys, os
from pathlib import Path
sys.path.insert(0, {str(PAPER)!r})
from benchmarks.adapters import acon_adapter as acon
summary = acon.run_appworld(
    'http://127.0.0.1:{proxy_port}/v1', Path({str(attempt_root)!r}),
    acon_dir=Path({cell['acon_dir']!r}), model={cell['model_name']!r},
    tag='gen', split='test_normal', max_iter=50,
    task_ids=['{task_id}'], python={cell['python_appworld']!r},
)
import json
print('SUMMARY:' + json.dumps(summary, default=str))
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PAPER)
    env["APPWORLD_ROOT"] = cell["appworld_root"]
    env["no_proxy"] = env["NO_PROXY"] = "127.0.0.1,localhost"
    for k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        env.pop(k, None)
    log_path = attempt_root.parent / "benchmark.log"
    with log_path.open("wb") as log:
        rc = subprocess.call(
            [cell["python_sgl"], "-c", script],
            cwd=str(PAPER), env=env, stdout=log, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
        )
    log_text = log_path.read_text(errors="ignore")
    result = {"task_id": task_id, "status": "completed" if rc == 0 else "failed",
              "returncode": rc, "wall_s": time.monotonic() - started}
    if rc == 0 and appworld_result_healthy(attempt_root, task_id):
        result["attempt_root"] = str(attempt_root.relative_to(out))
        write_done(out, result)
    else:
        result["status"] = "infra_error" if looks_infra(log_text) else "failed_validation"
        write_receipt(out, result)
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell", type=Path, required=True)
    parser.add_argument("--proxy-port", type=int, default=37400)
    parser.add_argument("--task-ids", nargs="*", default=None)
    parser.add_argument("--max-tasks", type=int, default=None)
    args = parser.parse_args(argv)

    cell = json.loads(args.cell.read_text())
    expected_ids = cell["task_ids"]
    if (not isinstance(expected_ids, list) or not expected_ids
            or any(not isinstance(task_id, str) or not task_id
                   for task_id in expected_ids)
            or len(expected_ids) != len(set(expected_ids))):
        raise ValueError("cell manifest must contain unique nonempty task IDs")
    arm = ARM_OF[(cell["backend"], cell["working_point"], cell["condition"])]
    cell["handler_name"] = f"c2kv-{arm.replace('_', '-')}"
    target_tokens = target_tokens_for_cell(cell)
    upstream = cell["sglang_backend_url"].rstrip("/")
    proxy = None
    task_ids = args.task_ids if args.task_ids is not None else expected_ids
    if len(task_ids) != len(set(task_ids)) or set(task_ids) - set(expected_ids):
        raise ValueError("requested task IDs must be unique members of the cell manifest")
    if args.max_tasks is not None:
        task_ids = task_ids[: args.max_tasks]
    results = []
    progress = Path(cell["cell_dir"]) / "progress.jsonl"
    cell_dir = Path(cell["cell_dir"])
    # resolve a port this driver will actually own before spawning anything
    proxy_port = resolve_free_port(args.proxy_port, cell_dir, arm)
    if proxy_port != args.proxy_port:
        print(json.dumps({"event": "proxy_port_reassigned",
                          "requested": args.proxy_port, "using": proxy_port,
                          "cell_dir": str(cell_dir), "arm": arm}), flush=True)
    try:
        proxy = start_proxy(arm, upstream, proxy_port, cell_dir, target_tokens)
        for task_id in task_ids:
            if cell["benchmark"] == "bfcl":
                result = run_bfcl_task(cell, task_id, proxy_port)
            elif cell["benchmark"] in ("toolsandbox", "acebench"):
                result = _run_adapter_task(cell, task_id, proxy_port,
                                           cell["benchmark"])
            else:
                result = run_appworld_task(cell, task_id, proxy_port)
            results.append(result)
            with progress.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(result, ensure_ascii=False) + "\n")
            done = sum(1 for r in results if r["status"] == "completed")
            print(json.dumps({"cell": cell["cell_id"], "arm": arm, "task": task_id,
                              "status": result["status"], "done": done,
                              "total": len(task_ids)}), flush=True)
    finally:
        stop(proxy)
    if cell["benchmark"] == "bfcl":
        completed_ids = [task_id for task_id in expected_ids
                         if existing_bfcl_result(cell_dir / "tasks" / task_id, task_id)]
    elif cell["benchmark"] == "appworld":
        completed_ids = [task_id for task_id in expected_ids
                         if appworld_done_healthy(cell_dir / "tasks" / task_id, task_id)]
    else:
        completed_ids = [task_id for task_id in expected_ids
                         if (cell_dir / "tasks" / task_id / "done.json").exists()]
    terminal = [task_id for task_id in expected_ids
                if task_id not in completed_ids
                and cached_terminal(cell_dir / "tasks" / task_id) is not None]
    pending = [task_id for task_id in expected_ids
               if task_id not in completed_ids and task_id not in terminal]
    # a cell is complete when every task ended in a terminal state: scored
    # done.json (model failures are terminal receipts and stay counted here).
    # infra failures (infra_error / failed_validation) keep it incomplete so
    # the scheduler requeues the cell for another infra retry — they are never
    # zero-scored and never silently dropped.
    (Path(cell["cell_dir"]) / "cell_status.json").write_text(json.dumps({
        "cell_id": cell["cell_id"], "arm": arm,
        "status": "complete" if not pending else "incomplete",
        "n_completed": len(completed_ids), "n_terminal": len(terminal),
        "terminal_tasks": terminal, "pending_infra": pending,
        "n_total": len(expected_ids),
        "finished_at": time.time(),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
