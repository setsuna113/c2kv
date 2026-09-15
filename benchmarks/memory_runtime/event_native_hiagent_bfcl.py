"""Finite official BFCL orchestration for one native HiAgent server.

This wrapper deliberately has its own ready, health, and run schemas.  Native
HiAgent exposes phase calls rather than the ordinary event-native chat API, so
the official BFCL harness must run through ``benchmarks.run`` and its existing
``hiagent_full_native`` proxy bridge.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit
from urllib.request import ProxyHandler, Request, build_opener

from benchmarks.native_hiagent_protocol import load_policy_sampling

from .bfcl_overlap_admission import (
    QUESTION_FILENAME,
    official_question_path,
    validate_overlap_admission,
)


RUN_SCHEMA = "a-event-native-hiagent-bfcl-run-v1"
READY_SCHEMA = "a-event-native-hiagent-server-v1"
HEALTH_SCHEMA = "a-event-native-hiagent-health-v1"
VALIDATION_SCHEMA = "a-event-native-hiagent-bfcl-official-validation-v1"
ARM = "hiagent_full_native"
BACKEND = "event_native_hiagent"
CAPABILITY_FEATURES = ("hiagent_trajectory_retrieval_v1",)
WORKER_MODULE = "benchmarks.memory_runtime.event_native_hiagent_bfcl"
SHARED_CALL_CAP_PER_TASK = 96
CLI_DESCRIPTION = __doc__


def save_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def positive_seconds(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive and finite")
    return parsed


def proxy_port(value: str) -> int:
    parsed = int(value)
    if not 1 <= parsed <= 65535:
        raise argparse.ArgumentTypeError("must be in [1, 65535]")
    return parsed


def task_category(task_ids: Sequence[str]) -> str:
    if (
        isinstance(task_ids, (str, bytes, bytearray))
        or not isinstance(task_ids, Sequence)
        or not task_ids
        or len(task_ids) != len(set(task_ids))
    ):
        raise ValueError("BFCL requires unique explicit task IDs")
    for category in ("multi_turn_base", "multi_turn_long_context"):
        prefix = category + "_"
        if all(
            isinstance(task, str)
            and task.startswith(prefix)
            and task.removeprefix(prefix).isdigit()
            for task in task_ids
        ):
            return category
    raise ValueError("BFCL requires one supported category of explicit task IDs")


def _loopback_base_url(base_url: str) -> str:
    url = urlsplit(base_url)
    if (
        url.scheme != "http"
        or url.hostname not in {"127.0.0.1", "localhost", "::1"}
        or url.username is not None
        or url.password is not None
        or url.path.rstrip("/")
        or url.query
        or url.fragment
    ):
        raise ValueError("base URL must identify a loopback host without a path")
    return base_url.rstrip("/")


def read_health(base_url: str) -> dict[str, Any]:
    normalized = _loopback_base_url(base_url)
    url = urlsplit(normalized)
    health_url = urlunsplit((url.scheme, url.netloc, "/health", "", ""))
    opener = build_opener(ProxyHandler({}))
    with opener.open(Request(health_url, method="GET"), timeout=5) as response:
        value = json.load(response)
    if not isinstance(value, dict):
        raise ValueError("native HiAgent health response must be a JSON object")
    return value


def _resolved(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a nonempty path")
    return str(Path(value).resolve())


def _counter(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def validate_server_identity(
    ready: Mapping[str, Any],
    health: Mapping[str, Any],
    *,
    base_url: str,
    policy_sampling: Path,
    require_fresh: bool = True,
) -> dict[str, Any]:
    """Bind a live phase endpoint to one ready manifest without old API fields."""

    if ready.get("schema") != READY_SCHEMA or ready.get("status") != "ready":
        raise ValueError("a ready native HiAgent server manifest is required")
    if health.get("schema") != HEALTH_SCHEMA or health.get("status") != "ok":
        raise ValueError("a healthy native HiAgent phase endpoint is required")
    normalized_url = _loopback_base_url(base_url)
    if _loopback_base_url(str(ready.get("base_url", ""))) != normalized_url:
        raise ValueError("endpoint base URL differs from the ready manifest")
    if ready.get("benchmark") != "bfcl" or health.get("benchmark") != "bfcl":
        raise ValueError("native HiAgent official worker requires BFCL identity")
    if ready.get("view_mode") != "full_original":
        raise ValueError("native HiAgent official worker requires full_original")
    if health.get("model_name") != ready.get("model_name"):
        raise ValueError("endpoint model identity differs from the ready manifest")

    task_ids = ready.get("allowed_task_ids")
    task_category(task_ids)
    if sorted(health.get("allowed_task_ids", [])) != sorted(task_ids):
        raise ValueError("endpoint task IDs differ from the ready manifest")

    artifacts = ready.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("ready manifest lacks durable artifact paths")
    for key, health_key in (("join", "join_path"), ("steps", "steps_path")):
        if _resolved(health.get(health_key), f"health.{health_key}") != _resolved(
            artifacts.get(key), f"ready.artifacts.{key}"
        ):
            raise ValueError(f"endpoint {health_key} differs from the ready manifest")

    sampling_contract = ready.get("policy_sampling_contract")
    if not isinstance(sampling_contract, Mapping):
        raise ValueError("ready manifest lacks the policy sampling contract")
    sampling_path = policy_sampling.resolve()
    sampling = load_policy_sampling(sampling_path)
    if _resolved(sampling_contract.get("source"), "policy sampling source") != str(
        sampling_path
    ):
        raise ValueError("policy sampling source path differs from the ready manifest")
    if sampling_contract.get("sha256") != sha256_file(sampling_path):
        raise ValueError("policy sampling file hash differs from the ready manifest")
    if sampling_contract.get("sampling") != sampling:
        raise ValueError("policy sampling values differ from the ready manifest")

    if ready.get("proxy_shared_call_cap_per_task") != SHARED_CALL_CAP_PER_TASK:
        raise ValueError("ready manifest lacks the shared per-task 96-call contract")
    if ready.get("proxy_call_cap_owner") != "official proxy":
        raise ValueError("ready manifest assigns the shared call cap to another owner")
    if not isinstance(ready.get("checkpoint"), Mapping):
        raise ValueError("ready manifest lacks checkpoint provenance")
    _resolved(ready.get("checkpoint_path"), "ready.checkpoint_path")
    runtime = ready.get("runtime_contract")
    expected_runtime = {
        "model_loads": 1,
        "actor_and_auxiliary_share_weights": True,
        "actor_and_auxiliary_caches": "separate",
        "generation_serialization": "dispatcher lock",
    }
    if runtime != expected_runtime:
        raise ValueError("ready manifest lacks the native HiAgent runtime identity")

    counters = {
        key: _counter(health.get(key), f"health.{key}")
        for key in (
            "completed_calls",
            "capacity_rejected_tasks",
            "actor_generation_calls",
            "auxiliary_generation_calls",
        )
    }
    if health.get("terminal_failure") is not None:
        raise ValueError("native HiAgent endpoint reports a terminal failure")
    if health.get("deadline_exceeded") is not False:
        raise ValueError("native HiAgent endpoint deadline is exhausted")
    if require_fresh and any(counters.values()):
        raise ValueError("native HiAgent endpoint was already consumed; rerun is disabled")
    for health_key, ready_key in (
        ("actor_generation_calls", "max_actor_generation_calls"),
        ("auxiliary_generation_calls", "max_auxiliary_generation_calls"),
    ):
        cap = ready.get(ready_key)
        if not isinstance(cap, int) or isinstance(cap, bool) or cap <= 0:
            raise ValueError(f"ready manifest has an invalid {ready_key}")
        if counters[health_key] > cap:
            raise ValueError(f"endpoint exceeded {ready_key}")

    return {
        "status": "matched",
        "ready_schema": READY_SCHEMA,
        "health_schema": HEALTH_SCHEMA,
        "base_url": normalized_url,
        "model_name": ready["model_name"],
        "task_ids": list(task_ids),
        "join_path": _resolved(health["join_path"], "health.join_path"),
        "steps_path": _resolved(health["steps_path"], "health.steps_path"),
        "policy_sampling_sha256": sampling_contract["sha256"],
        "fresh_before_official": require_fresh,
        "health_directly_attests_checkpoint_or_sampling": False,
        "identity_chain": (
            "ready base_url plus health model/task and exact durable join/steps paths; "
            "checkpoint and sampling are ready-manifest contracts"
        ),
        "counters": counters,
    }


def build_run_argv(contract: Mapping[str, Any]) -> list[str]:
    ready = contract["server_manifest"]
    sampling = contract["policy_sampling"]["sampling"]
    task_ids = list(contract["task_ids"])
    total_cap = SHARED_CALL_CAP_PER_TASK * len(task_ids)
    argv = [
        "--benchmark",
        "bfcl",
        "--arm",
        ARM,
        "--backend",
        BACKEND,
        "--upstream",
        contract["base_url"],
        "--proxy-port",
        str(contract["proxy_port"]),
        "--proxy-python",
        contract["proxy_python"],
        "--out",
        contract["official_output"],
        "--exact-out",
        "--run-name",
        ready["run_id"],
        "--model",
        ready["model_name"],
        "--checkpoint",
        ready["checkpoint_path"],
        "--categories",
        contract["category"],
        "--run-ids",
        ",".join(task_ids),
        "--num-workers",
        "1",
        "--native-hiagent-policy-sampling",
        contract["policy_sampling"]["path"],
        "--bfcl-temperature",
        str(sampling["temperature"]),
        "--bfcl-seed",
        str(sampling["seed"]),
        "--bfcl-generation-max-tokens",
        str(sampling["max_completion_tokens"]),
        "--no-upstream-retries",
        "--capture-request-views",
        "--max-generation-attempts",
        str(total_cap),
        "--max-generation-attempts-per-task",
        str(SHARED_CALL_CAP_PER_TASK),
        "--max-extraction-attempts",
        "0",
    ]
    if CAPABILITY_FEATURES:
        argv[6:6] = ["--capability-features", ",".join(CAPABILITY_FEATURES)]
    return argv


def _within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def validate_official_artifacts(contract: Mapping[str, Any]) -> dict[str, Any]:
    ready = contract["server_manifest"]
    task_ids = list(contract["task_ids"])
    official_root = Path(contract["official_output"]).resolve()
    summary_path = official_root / f"summary_{ARM}.json"
    if not summary_path.is_file():
        raise ValueError("official run summary is missing")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    expected = len(task_ids)
    for key, value in (
        ("benchmark", "bfcl"),
        ("arm", ARM),
        ("backend", BACKEND),
        ("model", ready["model_name"]),
        ("categories", contract["category"]),
        ("mode", "both"),
        ("scored", True),
        ("n_total", expected),
        ("n_generated", expected),
        ("n_scored", expected),
        ("bfcl_num_threads", 1),
    ):
        if summary.get(key) != value:
            raise ValueError(f"official summary differs for {key}")
    if Path(summary.get("bfcl_project_root", "")).resolve() != official_root:
        raise ValueError("official BFCL project root differs from the run output")
    expected_sampling = contract["policy_sampling"]["sampling"]
    requested = summary.get("generation_requested")
    if not isinstance(requested, Mapping) or any(
        requested.get(key) != expected_sampling[key]
        for key in ("temperature", "seed", "max_completion_tokens")
    ):
        raise ValueError("official handler sampling differs from the server contract")

    preflight = summary.get("preflight")
    requirements = preflight.get("requirements", []) if isinstance(preflight, Mapping) else []
    matching = [
        item
        for item in requirements
        if isinstance(item, Mapping)
        and item.get("code") == "native_hiagent_explicit_contract"
    ]
    if len(matching) != 1 or matching[0].get("satisfied") is not True:
        raise ValueError("native HiAgent run preflight was not satisfied")

    profile = summary.get("checkpoint_profile")
    ready_profile = ready["checkpoint"]
    if not isinstance(profile, Mapping):
        raise ValueError("official summary lacks checkpoint profile")
    for key in (
        "training_arm",
        "parameter_version",
        "corpus_identity",
        "model_geometry",
        "packing_contract",
        "policy_contract",
    ):
        if profile.get(key) != ready_profile.get(key):
            raise ValueError(f"official checkpoint profile differs for {key}")

    score_artifacts = []
    headers = summary.get("official_score_headers")
    if not isinstance(headers, list) or not headers:
        raise ValueError("official score headers are missing")
    if sum(item.get("total_count", 0) for item in headers if isinstance(item, Mapping)) != expected:
        raise ValueError("official score header denominator differs from selected tasks")
    for item in headers:
        if not isinstance(item, Mapping):
            raise ValueError("official score header receipt is invalid")
        path = Path(item.get("path", ""))
        if not path.is_file() or not _within(path, official_root):
            raise ValueError("official score file is missing or outside the run output")
        score_artifacts.append({"path": str(path.resolve()), "sha256": sha256_file(path)})

    evidence = []
    for label, path_value in (
        ("request_log", summary.get("request_log")),
        (
            "attempt_journal",
            (summary.get("attempt_journal") or {}).get("path")
            if isinstance(summary.get("attempt_journal"), Mapping)
            else None,
        ),
    ):
        path = Path(path_value or "")
        if not path.is_file() or not _within(path, official_root):
            raise ValueError(f"{label} is missing or outside the run output")
        evidence.append({"kind": label, "path": str(path.resolve()), "sha256": sha256_file(path)})

    return {
        "schema": VALIDATION_SCHEMA,
        "status": "passed",
        "task_ids": task_ids,
        "n_generated": expected,
        "n_scored": expected,
        "correct_count": summary.get("correct_count"),
        "accuracy_all_correct_required": False,
        "summary": {"path": str(summary_path), "sha256": sha256_file(summary_path)},
        "official_score_artifacts": score_artifacts,
        "trace_artifacts": evidence,
        "sampling": {
            key: expected_sampling[key]
            for key in ("temperature", "seed", "max_completion_tokens")
        },
        "shared_call_limit": {
            "per_task": SHARED_CALL_CAP_PER_TASK,
            "total": SHARED_CALL_CAP_PER_TASK * expected,
            "owner": "official proxy",
            "automatic_retries": 0,
            "automatic_reruns": 0,
        },
    }


def run_benchmarks(argv: Sequence[str]) -> None:
    from benchmarks import run

    run.main(list(argv))


def worker(contract_path: Path) -> None:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if contract.get("schema") != RUN_SCHEMA or contract.get("status") != "prepared":
        raise ValueError("worker requires one prepared native HiAgent BFCL contract")
    expected_argv = build_run_argv(contract)
    if contract.get("run_argv") != expected_argv:
        raise ValueError("native HiAgent official command changed after admission")
    admission = contract.get("overlap_admission")
    if admission is not None:
        worker_admission = validate_overlap_admission(
            Path(contract["overlap_audit_path"]),
            contract["server_manifest"],
            official_question_path(),
            expected_audit_sha256=admission["audit"]["sha256"],
        )
        if worker_admission["task_ids"] != admission["task_ids"]:
            raise ValueError("task selection changed after parent overlap admission")
        save_json(contract_path.parent / "overlap_admission.json", worker_admission)

    previous_bfcl_dir = os.environ.get("BENCH_BFCL_DIR")
    os.environ["BENCH_BFCL_DIR"] = contract["benchmark_dir"]
    try:
        run_benchmarks(expected_argv)
    finally:
        if previous_bfcl_dir is None:
            os.environ.pop("BENCH_BFCL_DIR", None)
        else:
            os.environ["BENCH_BFCL_DIR"] = previous_bfcl_dir
    validation = validate_official_artifacts(contract)
    save_json(Path(contract["official_validation_path"]), validation)


def stop_owned_process(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    if os.name == "posix":
        os.killpg(process.pid, signal.SIGTERM)
    else:
        process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
        process.wait(timeout=5)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=CLI_DESCRIPTION)
    result.add_argument("--server-manifest", type=Path)
    result.add_argument("--base-url")
    result.add_argument("--policy-sampling", type=Path)
    result.add_argument("--benchmark-dir", type=Path)
    result.add_argument("--bfcl-python", type=Path)
    result.add_argument("--proxy-python", type=Path)
    result.add_argument("--proxy-port", type=proxy_port)
    result.add_argument("--out", type=Path)
    result.add_argument("--max-wall-seconds", type=positive_seconds)
    result.add_argument(
        "--overlap-audit",
        type=Path,
        help="Checkpoint-bound BFCL exact-overlap audit; checked before generation",
    )
    result.add_argument("--worker-contract", type=Path, help=argparse.SUPPRESS)
    return result


def interpreter_path(path: Path) -> str:
    """Return an absolute interpreter path without resolving venv symlinks."""
    return os.path.abspath(os.fspath(path))


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    if args.worker_contract is not None:
        worker(args.worker_contract)
        return
    required = (
        args.server_manifest,
        args.base_url,
        args.policy_sampling,
        args.benchmark_dir,
        args.bfcl_python,
        args.proxy_python,
        args.proxy_port,
        args.out,
        args.max_wall_seconds,
    )
    if any(value is None for value in required):
        parser().error(
            "server manifest, base URL, policy sampling, benchmark directory, "
            "BFCL/proxy interpreters, proxy port, output and wall cap are required"
        )
    if args.out.exists():
        raise FileExistsError(f"output already exists: {args.out}")
    if not (args.benchmark_dir / "bfcl_eval").is_dir():
        raise ValueError("benchmark directory does not contain official bfcl_eval")
    for label, path in (("BFCL", args.bfcl_python), ("proxy", args.proxy_python)):
        if not path.is_file():
            raise ValueError(f"{label} interpreter does not exist")

    ready = json.loads(args.server_manifest.read_text(encoding="utf-8"))
    health_before = read_health(args.base_url)
    identity_before = validate_server_identity(
        ready,
        health_before,
        base_url=args.base_url,
        policy_sampling=args.policy_sampling,
        require_fresh=True,
    )
    task_ids = list(ready["allowed_task_ids"])
    category = task_category(task_ids)
    overlap_admission = None
    if args.overlap_audit is not None:
        overlap_admission = validate_overlap_admission(
            args.overlap_audit,
            ready,
            args.benchmark_dir / "bfcl_eval" / "data" / QUESTION_FILENAME,
        )

    started = time.monotonic()
    out = args.out.resolve()
    official_output = out / "official"
    validation_path = out / "official.validation.json"
    sampling_path = args.policy_sampling.resolve()
    contract: dict[str, Any] = {
        "schema": RUN_SCHEMA,
        "status": "prepared",
        "base_url": _loopback_base_url(args.base_url),
        "server_manifest": ready,
        "server_health_before": health_before,
        "server_identity_before": identity_before,
        "policy_sampling": {
            "path": str(sampling_path),
            "sha256": sha256_file(sampling_path),
            "sampling": load_policy_sampling(sampling_path),
        },
        "task_ids": task_ids,
        "category": category,
        "benchmark_dir": str(args.benchmark_dir.resolve()),
        "bfcl_python": interpreter_path(args.bfcl_python),
        "proxy_python": interpreter_path(args.proxy_python),
        "proxy_port": args.proxy_port,
        "official_output": str(official_output),
        "official_validation_path": str(validation_path),
        "maximum_wall_seconds": args.max_wall_seconds,
        "num_official_workers": 1,
        "automatic_retries": 0,
        "automatic_reruns": 0,
        "overlap_audit_path": (
            str(args.overlap_audit.resolve()) if args.overlap_audit is not None else None
        ),
        "overlap_admission": overlap_admission,
        "scope": (
            "Official BFCL generation and scoring through benchmarks.run, its "
            "existing proxy, and the native HiAgent phase bridge"
        ),
    }
    contract["run_argv"] = build_run_argv(contract)
    out.mkdir(parents=True, exist_ok=False)
    contract_path = out / "contract.json"
    save_json(contract_path, contract)

    root = Path(__file__).resolve().parents[2]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(root)
    command = [
        interpreter_path(args.bfcl_python),
        "-m",
        WORKER_MODULE,
        "--worker-contract",
        str(contract_path),
    ]
    process: subprocess.Popen[Any] | None = None
    status = "failed"
    error_receipt: dict[str, str] | None = None
    try:
        with (out / "worker.log").open("x", encoding="utf-8", newline="\n") as log:
            remaining = args.max_wall_seconds - (time.monotonic() - started)
            if remaining <= 0:
                status = "wall_cap_reached"
                raise TimeoutError("wall budget expired before worker admission")
            process = subprocess.Popen(
                command,
                cwd=root,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=os.name == "posix",
                creationflags=(
                    getattr(subprocess, "CREATE_NO_WINDOW", 0)
                    if os.name == "nt"
                    else 0
                ),
            )
            contract.update(status="running", child_pid=process.pid)
            save_json(out / "running.json", contract)
            remaining = args.max_wall_seconds - (time.monotonic() - started)
            try:
                returncode = process.wait(timeout=max(0, remaining))
                status = "completed_official_scored" if returncode == 0 else "failed_worker"
            except subprocess.TimeoutExpired:
                status = "wall_cap_reached"
                stop_owned_process(process)
            contract["worker_returncode"] = process.poll()
        if status == "completed_official_scored":
            validation = validate_official_artifacts(contract)
            health_after = read_health(args.base_url)
            identity_after = validate_server_identity(
                ready,
                health_after,
                base_url=args.base_url,
                policy_sampling=args.policy_sampling,
                require_fresh=False,
            )
            contract["official_validation"] = validation
            contract["server_health_after"] = health_after
            contract["server_identity_after"] = identity_after
    except BaseException as error:
        if status not in {"wall_cap_reached", "failed_worker"}:
            status = "failed_validation"
        error_receipt = {"type": type(error).__name__, "message": str(error)}
    finally:
        if process is not None:
            stop_owned_process(process)
        contract.update(
            status=status,
            error=error_receipt,
            wall_seconds=time.monotonic() - started,
            wall_seconds_final=True,
        )
        try:
            if "server_health_after" not in contract:
                contract["server_health_after"] = read_health(args.base_url)
        except Exception as error:
            contract["server_health_after"] = None
            contract["server_health_after_error"] = {
                "type": type(error).__name__,
                "message": str(error),
            }
        save_json(out / "final.json", contract)

    print(json.dumps({"status": status, "output": str(out)}), flush=True)
    if status != "completed_official_scored":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
