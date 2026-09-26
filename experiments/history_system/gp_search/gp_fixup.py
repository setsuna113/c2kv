"""Re-run missing tasks for a run dir and stitch them into the manifest.

Used when a run completed some tasks and died on later ones for an external
infrastructure reason (e.g. port collisions from other users). Each BFCL task
runs in its own server process and session, so a separate invocation is
execution-equivalent; every stitch is recorded explicitly in the manifest.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

RUN_ROOT = Path("/home/liuyancheng/gp_search_v1")
CHECKPOINT = ("/home/liuyancheng/c2kv-b-final-20260912/checkpoints/b_history/"
              "arm-C/seed-42/checkpoint-1000")
BENCHMARK_DIR = "/home/liuyancheng/benchmarks/gorilla/berkeley-function-call-leaderboard"
SGPY = "/home/liuyancheng/envs/sgl/bin/python"
BENCHPY = "/home/liuyancheng/envs/bench/bin/python"
os.environ.pop("http_proxy", None), os.environ.pop("https_proxy", None)
os.environ["NO_PROXY"] = os.environ["no_proxy"] = "127.0.0.1,localhost"
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")


def save(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    temporary.replace(path)


def fixup_one(run_dir: Path, candidate: Path, task_id: str,
              engine_port: int, port_base: int) -> int:
    shard = run_dir / "task_shards" / task_id
    if (shard / "bfcl" / "official_summary.json").exists():
        print(f"{task_id}: already scored; nothing to do")
        return 0
    design = json.loads((candidate / "design.json").read_text(encoding="utf-8"))
    fixup_design = json.loads(json.dumps(design))
    fixup_design["task_ids"] = [task_id]
    fixup_design["limits"] = {**fixup_design["limits"], "tasks": 1}
    fixup_design["run_id_template"] = design["run_id_template"] + "__fixup"
    fixup_design["runtime"] = {
        **fixup_design["runtime"],
        "sglang_backend_url": f"http://127.0.0.1:{engine_port}"}
    design_path = candidate / "design.fixup.json"
    save(design_path, fixup_design)
    tmp = RUN_ROOT / "runs" / ".fixup_tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    command = [
        SGPY, str(RUN_ROOT / "history_system" / "runner.py"), "run",
        "--design", str(design_path),
        "--checkpoint", CHECKPOINT,
        "--output", str(tmp),
        "--benchmark-dir", BENCHMARK_DIR,
        "--python", SGPY,
        "--bfcl-python", BENCHPY,
        "--port-base", str(port_base),
        "--runtime-root", str(candidate / "runtime"),
    ]
    started = time.monotonic()
    with (run_dir.parent / (run_dir.name + ".fixup.log")).open("a") as log:
        log.write(f"== fixup {task_id} {time.strftime('%H:%M:%S')}\n")
        code = subprocess.call(command, stdout=log, stderr=subprocess.STDOUT,
                               stdin=subprocess.DEVNULL)
    fixup_wall = time.monotonic() - started
    if (code != 0 or not (tmp / "task_shards" / task_id / "bfcl"
                          / "official_summary.json").exists()):
        print(f"fixup run failed rc={code} for {task_id}; log kept")
        return 3
    shutil.copytree(tmp / "task_shards" / task_id, shard, dirs_exist_ok=True)
    shutil.rmtree(tmp)
    manifest = json.loads((run_dir / "stage_manifest.json").read_text(encoding="utf-8"))
    outcomes = [row for row in manifest["task_outcomes"]
                if row["task_id"] != task_id]
    outcomes.append({
        "task_id": task_id,
        "outcome": "official_completed_via_fixup",
        "in_fixed_denominator": True,
        "worker_returncode": 0,
        "official_summary": str(shard / "bfcl" / "official_summary.json"),
    })
    manifest["task_outcomes"] = outcomes
    manifest["completed_task_cells"] = len(outcomes)
    fixups = manifest.setdefault("fixups", [])
    fixups.append({
        "task_id": task_id,
        "reason": "external port collision blocked the original attempt",
        "method": "separate single-task runner invocation, shard stitched",
        "fixup_wall_seconds": round(fixup_wall, 1),
    })
    shard_count = len([p for p in (run_dir / "task_shards").iterdir() if p.is_dir()
                       and (p / "bfcl" / "official_summary.json").exists()])
    if shard_count == manifest.get("whole_task_denominator"):
        manifest["status"] = "completed_fixed_manifest"
        manifest["state"] = "terminal"
    manifest["wall_seconds"] = (manifest.get("wall_seconds") or 0) + fixup_wall
    save(run_dir / "stage_manifest.json", manifest)
    print(json.dumps({"run": run_dir.name, "task": task_id,
                      "fixup_wall_seconds": round(fixup_wall, 1),
                      "scored_shards": shard_count,
                      "status": manifest["status"]}))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--task-id", required=True,
                        help="one task id or comma-separated list; "
                             "'auto' fixes every unscored shard")
    parser.add_argument("--engine-port", type=int, default=36110)
    parser.add_argument("--port-base", type=int, default=0,
                        help="0 = pick a fresh dynamic range")
    args = parser.parse_args()
    run_dir = args.run.resolve()
    candidate = args.candidate.resolve()
    shards = run_dir / "task_shards"
    if args.task_id == "auto":
        design_tasks = json.loads(
            (candidate / "design.json").read_text(encoding="utf-8"))["task_ids"]
        scored = {p.name for p in shards.iterdir() if p.is_dir()
                  and (p / "bfcl" / "official_summary.json").exists()}
        tasks = [t for t in design_tasks if t not in scored]
    else:
        tasks = [t.strip() for t in args.task_id.split(",") if t.strip()]
    if not tasks:
        print("no missing tasks")
        return 0
    if args.port_base == 0:
        import socket
        import random
        candidates = list(range(36200, 60000, 7))
        random.shuffle(candidates)
        for base in candidates:
            held = []
            for offset in range(20):
                probe = socket.socket()
                try:
                    probe.bind(("127.0.0.1", base + offset))
                    held.append(probe)
                except OSError:
                    break
            for probe in held:
                probe.close()
            if len(held) == 20:
                args.port_base = base
                break
        if args.port_base == 0:
            raise SystemExit("no free port range")
    print(f"fixing {len(tasks)} task(s) with port_base {args.port_base}")
    for task_id in tasks:
        code = fixup_one(run_dir, candidate, task_id,
                         args.engine_port, args.port_base)
        if code != 0:
            return code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
