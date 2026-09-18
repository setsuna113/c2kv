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
    ("snapkv", "K0", "recovery_off_same_initial"): "gen_snapkv_k0",
    ("snapkv", "K2", "recovery_off_same_initial"): "gen_snapkv_k2",
    ("snapkv", "K0", "compression_full_budget"): "gen_snapkv_b0",
    ("snapkv", "K2", "compression_full_budget"): "gen_snapkv_b2",
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


def start_proxy(arm: str, upstream: str, port: int, out: Path,
                target_tokens: int) -> subprocess.Popen:
    (out / "logs").mkdir(parents=True, exist_ok=True)
    log = (out / "proxy.log").open("wb")
    env = os.environ.copy()
    # proxy.py uses bare imports (repair_policy, textarms, arms) resolved from
    # the benchmarks directory itself, plus package imports from the repo root
    env["PYTHONPATH"] = os.pathsep.join((str(PAPER), str(PAPER / "benchmarks")))
    env["no_proxy"] = env["NO_PROXY"] = "127.0.0.1,localhost"
    for k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        env.pop(k, None)
    proc = subprocess.Popen(
        [sys.executable, "-m", "benchmarks.proxy",
         "--upstream", upstream, "--arm", arm, "--backend", "sglang",
          "--history-kv-target-tokens", str(target_tokens),
         "--port", str(port),
         "--request-log", str(out / "logs" / "proxy_requests.jsonl"),
         "--telemetry-log", str(out / "proxy_telemetry.jsonl")],
        cwd=str(PAPER), env=env, stdout=log, stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL, start_new_session=True,
    )
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
            raise RuntimeError(f"proxy exited rc={proc.returncode}; see {out / 'proxy.log'}")
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
        data=json.dumps(probe).encode(), headers={"Content-Type": "application/json"})
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
    return proc


def stop(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


def run_bfcl_task(cell: dict, task_id: str, proxy_port: int) -> dict:
    out = Path(cell["cell_dir"]) / "tasks" / task_id
    done = out / "done.json"
    if done.exists():
        return json.loads(done.read_text())
    out.mkdir(parents=True, exist_ok=True)
    category = task_id.rsplit("_", 1)[0]
    if category not in ("multi_turn_base", "multi_turn_long_context"):
        raise ValueError(f"cannot derive BFCL category from task id: {task_id}")
    project_root = out / "bfcl"
    shutil.rmtree(project_root, ignore_errors=True)
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
    log_path = out / "benchmark.log"
    with log_path.open("wb") as log:
        rc = subprocess.call(
            [cell["python_bench"], "-c", script],
            cwd=str(PAPER), env=env, stdout=log, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
        )
    result = {"task_id": task_id, "status": "completed" if rc == 0 else "failed",
              "returncode": rc, "wall_s": time.monotonic() - started}
    if rc == 0:
        # validate the official row: rows with a traceback (connection
        # failures, timeouts) must not count as official zero scores
        import glob
        healthy = False
        for path in glob.glob(str(project_root / "result" / "**" / "*.json"),
                              recursive=True):
            for line in Path(path).read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("id") != task_id:
                    continue
                healthy = row.get("traceback") is None
        if not healthy:
            result["status"] = "failed_validation"
            (out / "status.json").write_text(json.dumps(result, indent=2))
            return result
        (out / "done.json").write_text(json.dumps(result, indent=2))
    else:
        (out / "status.json").write_text(json.dumps(result, indent=2))
    return result


def run_appworld_task(cell: dict, task_id: str, proxy_port: int) -> dict:
    out = Path(cell["cell_dir"]) / "tasks" / task_id
    done = out / "done.json"
    if done.exists():
        return json.loads(done.read_text())
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    script = f"""
import sys, os
from pathlib import Path
sys.path.insert(0, {str(PAPER)!r})
from benchmarks.adapters import acon_adapter as acon
summary = acon.run_appworld(
    'http://127.0.0.1:{proxy_port}/v1', Path({str(out / 'appworld')!r}),
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
    log_path = out / "benchmark.log"
    with log_path.open("wb") as log:
        rc = subprocess.call(
            [cell["python_sgl"], "-c", script],
            cwd=str(PAPER), env=env, stdout=log, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
        )
    result = {"task_id": task_id, "status": "completed" if rc == 0 else "failed",
              "returncode": rc, "wall_s": time.monotonic() - started}
    if rc == 0:
        (out / "done.json").write_text(json.dumps(result, indent=2))
    else:
        (out / "status.json").write_text(json.dumps(result, indent=2))
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell", type=Path, required=True)
    parser.add_argument("--proxy-port", type=int, default=37400)
    parser.add_argument("--task-ids", nargs="*", default=None)
    parser.add_argument("--max-tasks", type=int, default=None)
    args = parser.parse_args(argv)

    cell = json.loads(args.cell.read_text())
    arm = ARM_OF[(cell["backend"], cell["working_point"], cell["condition"])]
    cell["handler_name"] = f"c2kv-{arm.replace('_', '-')}"
    target_tokens = target_tokens_for_cell(cell)
    upstream = cell["sglang_backend_url"].rstrip("/")
    proxy = None
    task_ids = args.task_ids or cell["task_ids"]
    if args.max_tasks is not None:
        task_ids = task_ids[: args.max_tasks]
    results = []
    progress = Path(cell["cell_dir"]) / "progress.jsonl"
    try:
        proxy = start_proxy(
            arm, upstream, args.proxy_port, Path(cell["cell_dir"]), target_tokens)
        for task_id in task_ids:
            if cell["benchmark"] == "bfcl":
                result = run_bfcl_task(cell, task_id, args.proxy_port)
            else:
                result = run_appworld_task(cell, task_id, args.proxy_port)
            results.append(result)
            with progress.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(result, ensure_ascii=False) + "\n")
            done = sum(1 for r in results if r["status"] == "completed")
            print(json.dumps({"cell": cell["cell_id"], "arm": arm, "task": task_id,
                              "status": result["status"], "done": done,
                              "total": len(task_ids)}), flush=True)
    finally:
        stop(proxy)
    completed = sum(1 for r in results if r["status"] == "completed")
    (Path(cell["cell_dir"]) / "cell_status.json").write_text(json.dumps({
        "cell_id": cell["cell_id"], "arm": arm,
        "status": "complete" if completed == len(results) and results else "incomplete",
        "n_completed": completed, "n_total": len(results),
        "finished_at": time.time(),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
