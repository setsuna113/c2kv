"""Plan and execute an explicit cross-benchmark matrix through ``run.py``.

The input is a JSON document.  By default this module only writes a plan; an
operator must pass ``--execute`` to launch any benchmark process.  Each cell
has a profile- and argv-bound fingerprint, independent output directory and
status JSON.  A resume can skip only a previously passed cell with the exact
same fingerprint.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Callable, Iterable, Mapping, Optional

import capabilities


SCHEMA_VERSION = 1
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), default=str)


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False,
                                     dir=path.parent, suffix=".tmp") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def _repo_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "nogit"


def _string_list(value: Any, field: str) -> list[str]:
    if value in (None, ""):
        return []
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{field} must be a JSON list of strings")
    return list(value)


def _flag_args(args: Iterable[str]) -> list[str]:
    values = list(args)
    forbidden = {"--benchmark", "--arm", "--backend", "--upstream", "--user-upstream",
                 "--out", "--run-name"}
    for value in values:
        if value in forbidden or any(value.startswith(flag + "=") for flag in forbidden):
            raise ValueError(f"matrix owns {value!r}; remove it from run_args")
    return values


def _append_adapter_path_options(command: list[str], options: Mapping[str, Any]) -> None:
    """Keep path inputs used by preflight identical to the argv run.py sees."""
    for option, flag in (("acon_dir", "--acon-dir"),
                         ("acebench_dir", "--acebench-dir"),
                         ("bench_python", "--bench-python")):
        value = options.get(option)
        if value in (None, ""):
            continue
        if flag in command or any(arg.startswith(flag + "=") for arg in command):
            continue
        command.extend([flag, str(value)])


def _profile(spec: Mapping[str, Any]) -> dict[str, Any]:
    profile = spec.get("profile")
    if not isinstance(profile, Mapping):
        raise ValueError("matrix profile must be an object with a non-empty fingerprint")
    result = dict(profile)
    if not isinstance(result.get("fingerprint"), str) or not result["fingerprint"].strip():
        raise ValueError("matrix profile.fingerprint is required")
    return result


def _cell_id(benchmark: str, arm: str) -> str:
    return f"{benchmark}__{arm}".replace("/", "-")


def _summary_facts(summary_path: Path) -> tuple[bool, dict[str, Any] | str]:
    if not summary_path.is_file():
        return False, f"missing summary: {summary_path}"
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return False, f"invalid summary {summary_path}: {error}"
    n = payload.get("n")
    if not isinstance(n, int) or n <= 0:
        return False, "summary has no positive task count n"
    return True, {
        "n": n,
        "semantic_score": payload.get("semantic_score"),
        "cost_join": payload.get("cost_join"),
        "request_log_summary": payload.get("request_log_summary"),
        "summary_path": str(summary_path),
    }


def build_plan(spec: Mapping[str, Any], output_root: Path, *,
               environ: Optional[Mapping[str, str]] = None) -> dict[str, Any]:
    """Expand a JSON matrix specification into argv-bound, preflighted cells."""
    if spec.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"matrix schema_version must be {SCHEMA_VERSION}")
    profile = _profile(spec)
    defaults = spec.get("defaults") or {}
    if not isinstance(defaults, Mapping):
        raise ValueError("matrix defaults must be an object")
    benchmarks = spec.get("benchmarks")
    if not isinstance(benchmarks, Mapping) or not benchmarks:
        raise ValueError("matrix benchmarks must be a non-empty object")
    arms = _string_list(spec.get("arms"), "arms")
    if not arms:
        raise ValueError("matrix arms must be non-empty")
    global_features = _string_list(spec.get("features"), "features")
    common_args = _flag_args(_string_list(defaults.get("run_args"), "defaults.run_args"))
    repo_commit = _repo_commit()
    root = Path(output_root).resolve()
    cells: list[dict[str, Any]] = []

    for benchmark, raw_config in benchmarks.items():
        if not isinstance(benchmark, str):
            raise ValueError("benchmark names must be strings")
        config = raw_config or {}
        if not isinstance(config, Mapping):
            raise ValueError(f"benchmark config for {benchmark!r} must be an object")
        runner_python = str(config.get("runner_python")
                            or defaults.get("runner_python") or sys.executable)
        backend = str(config.get("backend") or defaults.get("backend") or "sglang")
        upstream = str(config.get("upstream") or defaults.get("upstream") or "")
        if not upstream:
            raise ValueError(f"{benchmark}.upstream or defaults.upstream is required")
        user_upstream = str(config.get("user_upstream")
                            or defaults.get("user_upstream") or "")
        model = str(config.get("model") or defaults.get("model") or "c2kv-agent")
        bench_args = _flag_args(_string_list(config.get("run_args"),
                                              f"benchmarks.{benchmark}.run_args"))
        bench_features = _string_list(config.get("features"),
                                      f"benchmarks.{benchmark}.features")
        options = dict(defaults.get("options") or {})
        extra_options = config.get("options") or {}
        if not isinstance(extra_options, Mapping):
            raise ValueError(f"benchmarks.{benchmark}.options must be an object")
        options.update(extra_options)
        options["runner_python"] = runner_python
        if benchmark in {"acon_qa", "acon_appworld", "acebench"}:
            options.setdefault("bench_python", runner_python)
        features = sorted(set(global_features) | set(bench_features))

        for arm in arms:
            cell_id = _cell_id(benchmark, arm)
            cell_dir = root / "cells" / cell_id
            run_out = cell_dir / "run"
            status_path = root / "status" / f"{cell_id}.json"
            run_name = f"matrix_{cell_id}_{profile['fingerprint'][:12]}"
            command = [runner_python, str(HERE / "run.py"),
                       "--benchmark", benchmark, "--arm", arm,
                       "--backend", backend, "--upstream", upstream,
                       "--proxy-port", str(config.get("proxy_port")
                                             or defaults.get("proxy_port") or 34100),
                       "--out", str(run_out), "--run-name", run_name,
                       "--model", model]
            if user_upstream:
                command += ["--user-upstream", user_upstream]
            command += common_args + bench_args
            _append_adapter_path_options(command, options)
            preflight = capabilities.preflight(
                benchmark, arm, backend, options=options, profile=profile,
                features=features, environ=environ,
            )
            fingerprint = _sha256({
                "schema_version": SCHEMA_VERSION,
                "profile": profile,
                "repo_commit": repo_commit,
                "command": command,
                "features": features,
            })
            # run.py suffixes --out using this same repository's short HEAD.
            summary_dir = Path(str(run_out) + f"_{repo_commit[:7]}")
            cells.append({
                "id": cell_id,
                "benchmark": benchmark,
                "arm": arm,
                "backend": backend,
                "features": features,
                "profile_fingerprint": profile["fingerprint"],
                "cell_fingerprint": fingerprint,
                "command": command,
                "cell_dir": str(cell_dir),
                "status_path": str(status_path),
                "summary_path": str(summary_dir / f"summary_{arm}.json"),
                "preflight": preflight.as_dict(),
            })
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "benchmark_matrix_plan",
        "repo_commit": repo_commit,
        "profile": profile,
        "output_root": str(root),
        "cells": cells,
    }


def _status_matches(status_path: Path, fingerprint: str) -> bool:
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return status.get("state") == "passed" and status.get("cell_fingerprint") == fingerprint


def _prior_profile_conflict(status_path: Path, fingerprint: str) -> bool:
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return status_path.exists()
    prior = status.get("cell_fingerprint")
    return prior is not None and prior != fingerprint


def _status_exists(status_path: Path) -> bool:
    return status_path.is_file()


Runner = Callable[..., subprocess.CompletedProcess]


def execute_plan(plan: Mapping[str, Any], *, resume: bool = False,
                 runner: Runner = subprocess.run) -> int:
    """Run valid cells, persist outcome facts, and return a shell exit status."""
    exit_code = 0
    for cell in plan.get("cells", []):
        status_path = Path(cell["status_path"])
        fingerprint = str(cell["cell_fingerprint"])
        if _prior_profile_conflict(status_path, fingerprint):
            # Preserve the completed cell's evidence.  A mismatch is itself
            # recorded under a new name rather than rewriting that evidence.
            conflict = status_path.with_name(
                f"{status_path.stem}.{fingerprint[:12]}.conflict.json")
            _atomic_json(conflict, {
                "schema_version": SCHEMA_VERSION, "state": "blocked",
                "reason": "existing cell status has a different profile or command fingerprint",
                "cell_fingerprint": fingerprint,
                "prior_status_path": str(status_path),
            })
            exit_code = max(exit_code, 2)
            continue
        if resume and _status_matches(status_path, fingerprint):
            continue
        if not resume and _status_exists(status_path):
            _atomic_json(status_path.with_name(
                f"{status_path.stem}.{fingerprint[:12]}.rerun-blocked.json"), {
                "schema_version": SCHEMA_VERSION, "state": "blocked",
                "reason": "cell already has this fingerprint; pass --resume to reuse it",
                "cell_fingerprint": fingerprint,
                "prior_status_path": str(status_path),
            })
            exit_code = max(exit_code, 2)
            continue
        if not cell["preflight"]["ok"]:
            _atomic_json(status_path, {
                "schema_version": SCHEMA_VERSION, "state": "blocked",
                "reason": "capability preflight failed",
                "cell_fingerprint": fingerprint,
                "preflight": cell["preflight"],
            })
            exit_code = max(exit_code, 2)
            continue

        cell_dir = Path(cell["cell_dir"])
        cell_dir.mkdir(parents=True, exist_ok=True)
        log_path = cell_dir / "matrix_runner.log"
        _atomic_json(status_path, {
            "schema_version": SCHEMA_VERSION, "state": "running",
            "cell_fingerprint": fingerprint, "command": cell["command"],
            "log_path": str(log_path),
        })
        with log_path.open("w", encoding="utf-8") as log:
            completed = runner(cell["command"], cwd=str(REPO_ROOT),
                               stdout=log, stderr=subprocess.STDOUT, check=False)
        valid, facts = _summary_facts(Path(cell["summary_path"]))
        passed = completed.returncode == 0 and valid
        _atomic_json(status_path, {
            "schema_version": SCHEMA_VERSION,
            "state": "passed" if passed else "failed",
            "cell_fingerprint": fingerprint,
            "command": cell["command"],
            "returncode": completed.returncode,
            "log_path": str(log_path),
            "summary": facts if valid else None,
            "failure": None if passed else (
                f"runner returncode={completed.returncode}" if completed.returncode else str(facts)),
        })
        if not passed:
            exit_code = max(exit_code, completed.returncode or 1)
    return exit_code


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"FATAL: cannot read matrix spec {path}: {error}") from error
    if not isinstance(payload, dict):
        raise SystemExit("FATAL: matrix spec root must be an object")
    return payload


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True,
                        help="JSON matrix specification")
    parser.add_argument("--out", type=Path, required=True,
                        help="matrix root; cells and status files live below it")
    parser.add_argument("--plan-out", type=Path, default=None,
                        help="plan JSON path (default: <out>/matrix_plan.json)")
    parser.add_argument("--execute", action="store_true",
                        help="actually run the planned cells; omitted means plan only")
    parser.add_argument("--resume", action="store_true",
                        help="skip only passed cells with the exact same fingerprint")
    args = parser.parse_args(argv)
    try:
        plan = build_plan(_load_json(args.matrix), args.out)
    except ValueError as error:
        raise SystemExit(f"FATAL: invalid matrix specification: {error}") from error
    plan_out = args.plan_out or args.out / "matrix_plan.json"
    _atomic_json(plan_out, plan)
    if not args.execute:
        print(f"planned {len(plan['cells'])} cell(s): {plan_out}")
        return 0
    return execute_plan(plan, resume=args.resume)


if __name__ == "__main__":
    raise SystemExit(main())
