"""Monitor the GP search, collect finished groups, advance to the next group.

Run periodically on the server. Exit codes: 0 normal (progress or transition),
2 blocked/anomaly needing an operator, 3 all groups finished.
State lives in <run_root>/progress/: active_group.txt, groups_order.txt,
retry_ledger.json, transitions.log, status.latest.json.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import time
from pathlib import Path

RUN_ROOT = Path("/home/liuyancheng/gp_search_v1")
GROUPS = ["A", "B", "C", "D", "E", "F"]
BASELINE_RUN = "A__gp_52ede92dd4c8"
# Card 0 reuses the engine validated by the parallel sglang session (port 36100).
CARDS = [0, 1, 2, 3, 4]
SGPY = "/home/liuyancheng/envs/sgl/bin/python"
BENCHPY = "/home/liuyancheng/envs/bench/bin/python"
os.environ.pop("http_proxy", None), os.environ.pop("https_proxy", None)
os.environ["NO_PROXY"] = os.environ["no_proxy"] = "127.0.0.1,localhost"


def log(text: str) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with (RUN_ROOT / "progress" / "transitions.log").open("a", encoding="utf-8") as s:
        s.write(f"[{stamp}] {text}\n")


def read(path: Path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def run_status_safe(run_dir: Path):
    manifest = read(run_dir / "stage_manifest.json")
    if manifest is None:
        return None
    if manifest.get("status"):
        return manifest["status"]
    return manifest.get("state")


def driver_alive(group: str) -> bool:
    result = subprocess.run(["pgrep", "-f", f"driver.{group}.card"],
                            capture_output=True, text=True)
    return bool(result.stdout.strip())


def engine_alive(port: int) -> bool:
    import urllib.request
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(f"http://127.0.0.1:{port}/model_info", timeout=10) as r:
            return r.status == 200
    except Exception:
        return False


ENGINE_PORTS = {f"card{c}s0": 36100 + c * 10 for c in range(5)}
ENGINE_PORTS.update({f"card{c}s1": 36150 + c * 10 for c in range(5)})


def run(command: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(command, capture_output=True, text=True, **kwargs)


def launch_drivers(group: str) -> None:
    """Launch every driver plan for the group: dual-slot plans named
    driver.<group>.card<N>s[01].json plus legacy per-card plans."""
    launched = []
    for plan in sorted((RUN_ROOT / "plans").glob(f"driver.{group}.card*.json")):
        if ".next" in plan.name:
            continue
        name = plan.stem.replace(f"driver.{group}.", "")
        slot_match = re.fullmatch(r"card([0-9])(s[01])", name)
        if slot_match:
            card, slot = slot_match.groups()
            extra = ["--slot", slot]
        else:
            card, extra = name.removeprefix("card"), []
        log_file = RUN_ROOT / "progress" / f"driver.{group}.{name}.log"
        subprocess.Popen(
            ["python3", "gp_driver.py", "--plan", str(plan), "--card", card, *extra],
            cwd=RUN_ROOT, stdout=log_file.open("w"), stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, start_new_session=True)
        launched.append(name)
    log(f"launched drivers for group {group}: {launched}")


def group_runs(group: str) -> dict[str, str | None]:
    build = read(RUN_ROOT / f"build.{group}.json")
    if build is None:
        return {}
    out = {}
    for mapping in build["mappings"]:
        run_name = Path(mapping["directory"]).name
        out[run_name] = run_status_safe(RUN_ROOT / "runs" / run_name)
    return out


def collect(group: str) -> bool:
    result = run(["python3", "gp_collect.py", "--run-root", str(RUN_ROOT),
                  "--baseline", BASELINE_RUN, "--out", f"summary_{group}"],
                 cwd=RUN_ROOT)
    ok = result.returncode == 0 and (RUN_ROOT / f"summary_{group}.json").exists()
    log(f"collect group {group}: rc={result.returncode}")
    metrics = run(["/home/liuyancheng/envs/bench/bin/python", "gp_metrics.py",
                   "--group", group], cwd=RUN_ROOT)
    log(f"metrics group {group}: rc={metrics.returncode}")
    return ok


def retry_missing(group: str, missing: list[str]) -> bool:
    ledger_path = RUN_ROOT / "progress" / "retry_ledger.json"
    ledger = read(ledger_path, default={})
    pending = [name for name in missing if ledger.get(f"{group}__{name}", 0) < 1]
    for name in pending:
        ledger[f"{group}__{name}"] = ledger.get(f"{group}__{name}", 0) + 1
    ledger_path.write_text(json.dumps(ledger, indent=2), encoding="utf-8")
    if not pending:
        return False
    # Relaunch every card driver with retry_failed so finished runs are skipped.
    for card in CARDS:
        plan = RUN_ROOT / "plans" / f"driver.{group}.card{card}.json"
        if not plan.exists():
            continue
        data = json.loads(plan.read_text(encoding="utf-8"))
        data["retry_failed"] = True
        plan.write_text(json.dumps(data, indent=2), encoding="utf-8")
    launch_drivers(group)
    log(f"retry launched for {len(pending)} missing runs in group {group}: {pending}")
    return True


def advance(from_groups: list[str], to_group: str) -> bool:
    spec_path = RUN_ROOT / "plans" / f"spec.{to_group}.json"
    if spec_path.exists():
        log(f"advance to {to_group}: spec exists, building")
    else:
        result = run(["python3", "gp_advance.py",
                      "--from-groups", ",".join(from_groups),
                      "--to-group", to_group, "--out", str(spec_path)],
                     cwd=RUN_ROOT)
        if result.returncode != 0:
            log(f"advance to {to_group} FAILED: {result.stdout} {result.stderr}")
            return False
        log(f"advance produced spec for {to_group}: {result.stdout.strip()}")
    alive_cards = sorted({int(slot.removeprefix("card")[0])
                          for slot, port in ENGINE_PORTS.items()
                          if engine_alive(port)})
    cards = ",".join(str(c) for c in alive_cards) or "0,1,2,3,4"
    steps = [
        ["python3", "mkplans.py", "build", "--group", to_group,
         "--rows", str(spec_path), "--cards", cards, "--dual"],
        [SGPY, "gp_build.py", "--plan", str(RUN_ROOT / "plans" / f"build.{to_group}.json")],
        ["python3", "mkplans.py", "drivers", "--group", to_group],
    ]
    for step in steps:
        result = run(step, cwd=RUN_ROOT)
        if result.returncode != 0:
            log(f"advance step failed: {' '.join(step[:4])}: {result.stdout} {result.stderr}")
            return False
    launch_drivers(to_group)
    (RUN_ROOT / "progress" / "active_group.txt").write_text(to_group, encoding="utf-8")
    return True


def main() -> int:
    progress = RUN_ROOT / "progress"
    progress.mkdir(parents=True, exist_ok=True)
    active_path = progress / "active_group.txt"
    if not active_path.exists():
        active_path.write_text("A", encoding="utf-8")
    group = active_path.read_text(encoding="utf-8").strip()
    runs = group_runs(group)
    states = list(runs.values())
    missing = [name for name, status in runs.items()
               if status != "completed_fixed_manifest"]
    engines = {slot: engine_alive(port) for slot, port in ENGINE_PORTS.items()}
    drivers = driver_alive(group)
    status = {
        "schema": "a-history-gp-monitor-status-v1",
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "active_group": group,
        "runs_total": len(runs),
        "runs_terminal": len(runs) - len(missing),
        "missing_runs": missing,
        "drivers_alive": drivers,
        "engines": engines,
        "qualification": "preliminary, n=1",
    }
    import sys

    def finish(code: int, **extra):
        status.update(extra)
        (progress / "status.latest.json").write_text(
            json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(status, ensure_ascii=False))
        return code

    if missing:
        if drivers:
            return finish(0, note="running")
        if not all(engine_alive(ENGINE_PORTS[f"card{c}s0"]) for c in CARDS):
            return finish(2, note="primary engine down and no driver alive",
                          blocked=True)
        retried = retry_missing(group, missing)
        return finish(0 if retried else 2,
                      note="retry launched" if retried
                      else "missing runs already retried once; blocked",
                      blocked=not retried)
    collected = collect(group)
    if not collected:
        return finish(2, note="collection failed", blocked=True)
    index = GROUPS.index(group) if group in GROUPS else -1
    if index == -1 or index + 1 >= len(GROUPS):
        return finish(3, note="all groups terminal; awaiting operator for "
                              "cross-combination", done=True)
    next_group = GROUPS[index + 1]
    from_groups = GROUPS[: index + 1]
    advanced = advance(from_groups, next_group)
    return finish(0 if advanced else 2,
                  note=f"transitioned to {next_group}" if advanced
                  else f"advance to {next_group} failed",
                  blocked=not advanced)


if __name__ == "__main__":
    raise SystemExit(main())
