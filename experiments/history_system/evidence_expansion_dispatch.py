"""Dispatch every shard in one authorized H0/H1/R3 package exactly once."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import evidence_eval_trained as trained
STATE_SCHEMA = "experiment3-expansion-dispatch-v1"
PACKAGE_KINDS = {
    "expansion_contract.json": ("evidence_eval_expansion.py", "expansion_contract.json"),
    "combination_contract.json": (
        "evidence_eval_combinations.py", "combination_contract.json"),
}
def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value
def _save(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
def _load(path: Path):
    name = "_experiment3_dispatch_" + str(abs(hash(path.resolve())))
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load frozen package module: {path}")
    module = importlib.util.module_from_spec(spec)
    dependency_names = ("evidence_eval", "evidence_eval_expansion",
                        "evidence_eval_combinations")
    saved = {key: sys.modules.pop(key) for key in dependency_names if key in sys.modules}
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
        for key in dependency_names:
            sys.modules.pop(key, None)
        sys.modules.update(saved)
    return module


def _package(package: Path, authorization: Path) -> tuple[Any, str, list[dict[str, Any]]]:
    matches = [(module, contract) for marker, (module, contract) in PACKAGE_KINDS.items()
               if (package / marker).is_file()]
    if len(matches) != 1:
        raise ValueError("Package must contain exactly one H0 or H1/R3 contract")
    module_name, contract_name = matches[0]
    module = _load(package / module_name)
    module.verify_package(package)
    module.verify_authorization(package, authorization)
    contract = _read(package / contract_name)
    rows = []
    for row in contract.get("shards", []):
        shard_id = row.get("shard_id")
        device = row.get("preferred_device", row.get("device"))
        wave = row.get("wave", 0)
        if (not isinstance(shard_id, str) or type(device) is not int
                or type(wave) is not int or wave < 0):
            raise ValueError("Frozen shard schedule is malformed")
        shard = package / "shards" / shard_id
        lane = shard / "lanes" / shard_id
        if (lane / "run").exists() or (lane / "results").exists():
            raise FileExistsError(f"Shard already has run state: {shard_id}")
        rows.append({"shard_id": shard_id, "device": device, "wave": wave,
                     "shard_package": str(shard)})
    if not rows or len({row["shard_id"] for row in rows}) != len(rows):
        raise ValueError("Frozen shard schedule is empty or repeats an ID")
    return module, module_name, rows


def _lane(module: Any, package: Path, shard_id: str) -> tuple[Any, dict[str, Any]]:
    shard = package / "shards" / shard_id
    loader = getattr(module, "_load_base_module", None)
    if loader is None:
        loader = module.expansion._load_base_module
    base = loader(shard / "evidence_eval.py")
    lane = _read(shard / "lanes" / shard_id / "lane.json")
    return base, lane


def _occupied(error: RuntimeError) -> bool:
    message = str(error)
    return "acquired by PIDs" in message or "ports are occupied" in message


def dispatch(package: Path, authorization: Path, *, poll_seconds: float = 20.0) -> int:
    package, authorization = package.resolve(), authorization.resolve()
    state_path, log_root = package / "dispatch.json", package / "dispatch_logs"
    if state_path.exists() or log_root.exists():
        raise FileExistsError("Refusing to overwrite existing dispatch state")
    module, module_name, rows = _package(package, authorization)
    trained.ascend_environment()
    trained.enable_strict_sampling()
    log_root.mkdir()
    state = {"schema": STATE_SCHEMA, "status": "running", "package": str(package),
             "authorization": str(authorization), "created_at": _now(),
             "automatic_retries": 0, "automatic_reruns": 0,
             "shards": [{**row, "status": "queued", "pid": None,
                          "returncode": None} for row in rows]}
    _save(state_path, state)
    pending = {row["shard_id"] for row in rows}
    running: dict[str, tuple[subprocess.Popen, Any]] = {}
    while pending or running:
        for row in state["shards"]:
            shard_id = row["shard_id"]
            if shard_id not in pending:
                continue
            earlier = [other for other in state["shards"]
                       if other["device"] == row["device"]
                       and other["wave"] < row["wave"]]
            if any(other["status"] in {"queued", "running"} for other in earlier):
                continue
            try:
                base, lane = _lane(module, package, shard_id)
                base._assert_lane_free(lane)
            except RuntimeError as error:
                if _occupied(error):
                    continue
                row.update(status="dispatch_failed_no_retry", error=str(error),
                           finished_at=_now())
                pending.remove(shard_id)
                _save(state_path, state)
                continue
            except Exception as error:
                row.update(status="dispatch_failed_no_retry", error=str(error),
                           finished_at=_now())
                pending.remove(shard_id)
                _save(state_path, state)
                continue
            command = [sys.executable, str(package / module_name), "run-shard",
                       "--package", str(package), "--shard", shard_id,
                       "--authorization", str(authorization)]
            log = (log_root / f"{shard_id}.log").open("x", encoding="utf-8", newline="\n")
            try:
                process = subprocess.Popen(
                    command, cwd=package, env=os.environ.copy(),
                    stdin=subprocess.DEVNULL, stdout=log,
                    stderr=subprocess.STDOUT)
            except Exception as error:
                log.close()
                row.update(status="dispatch_failed_no_retry", error=str(error),
                           finished_at=_now(), log=str(log.name))
                pending.remove(shard_id)
                _save(state_path, state)
                continue
            row.update(status="running", pid=process.pid, command=command,
                       started_at=_now(), log=str(log.name))
            pending.remove(shard_id)
            running[shard_id] = (process, log)
            _save(state_path, state)
        for shard_id, (process, log) in list(running.items()):
            code = process.poll()
            if code is None:
                continue
            log.close()
            row = next(item for item in state["shards"] if item["shard_id"] == shard_id)
            row.update(status="completed" if code == 0 else "failed_no_retry",
                       returncode=code, finished_at=_now())
            del running[shard_id]
            _save(state_path, state)
        if pending or running:
            time.sleep(poll_seconds)
    failed = [row for row in state["shards"] if row["status"] != "completed"]
    state.update(status="completed" if not failed else "partial_failed_no_retry",
                 finished_at=_now(), failed_shards=[row["shard_id"] for row in failed])
    _save(state_path, state)
    return 0 if not failed else 2


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=20.0)
    args = parser.parse_args(argv)
    return dispatch(args.package, args.authorization, poll_seconds=args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
