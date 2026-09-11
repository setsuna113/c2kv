"""Bounded native BFCL evaluation and dev-only checkpoint selection."""
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
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


EVAL_SCHEMA = "history-memory-native-bfcl-eval-v1"
TRAINING_PROFILE = "history-event-base-query-v1"
EXPECTED_CHECKPOINT_PROFILE = {
    "history_memory_training_profile": TRAINING_PROFILE,
    "history_memory_packing_version": "history-event-v1",
    "history_memory_raw_layout": "event-native-evidence-v1",
    "history_memory_evidence_version": "history-evidence-v1",
    "history_memory_normal_query": "base",
    "gist_param": "qkv",
    "gist_type": "dynamic-interleave",
    "gist_residual_type": "embed-mean",
}
EVAL_POLICY_SCHEMAS = {
    "a-event-native-eval-policy-v1",
    "a-event-native-eval-policy-v2",
}
OVERRIDE_VIEW_MODES = {
    "capacity_protect",
    "capacity_exact_once",
    "capacity_exact_persistent",
    "full_exact_shared",
    "capacity_exact_no_gist",
    "full_original",
    "ac_gist_static",
    "ac_protect",
    "ac_exact_once",
    "ac_exact_persistent",
    "ac_full_shared",
    "raw_exact_shared",
}
ALWAYS_VIEW_MODES = {
    "ac_gist_static",
    "ac_protect",
    "ac_exact_once",
    "ac_exact_persistent",
    "ac_full_shared",
    "raw_exact_shared",
}


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    data = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _positive_int(value: Any, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _positive_number(value: Any, name: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or value <= 0
    ):
        raise ValueError(f"{name} must be positive and finite")
    return float(value)


def _absolute_file(value: Any, name: str) -> Path:
    path = Path(value) if isinstance(value, (str, os.PathLike)) else None
    if path is None or not path.is_absolute():
        raise ValueError(f"{name} must be an absolute path")
    if not path.is_file():
        raise FileNotFoundError(f"{name} does not exist: {path}")
    return path.resolve()


def _absolute_dir(value: Any, name: str, *, must_exist: bool = True) -> Path:
    path = Path(value) if isinstance(value, (str, os.PathLike)) else None
    if path is None or not path.is_absolute():
        raise ValueError(f"{name} must be an absolute path")
    if must_exist and not path.is_dir():
        raise FileNotFoundError(f"{name} does not exist: {path}")
    return path.resolve()


def inspect_checkpoint(path: Path, expected_arm: str | None = None) -> dict[str, Any]:
    """Validate the saved B/C identity without importing torch or transformers."""
    checkpoint = Path(path).resolve()
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint}")
    try:
        step = int(checkpoint.name.removeprefix("checkpoint-"))
    except ValueError as error:
        raise ValueError(f"checkpoint directory must be named checkpoint-N: {checkpoint}") from error
    if checkpoint.name != f"checkpoint-{step}" or step <= 0:
        raise ValueError(f"checkpoint directory must be named checkpoint-N for positive N: {checkpoint}")

    config_path = checkpoint / "config.json"
    state_path = checkpoint / "trainer_state.json"
    if not config_path.is_file() or not state_path.is_file():
        raise FileNotFoundError("checkpoint requires config.json and trainer_state.json")
    config = _read_json(config_path)
    state = _read_json(state_path)
    if not isinstance(config, Mapping) or not isinstance(state, Mapping):
        raise ValueError("checkpoint config and trainer state must be JSON objects")
    for field, required in EXPECTED_CHECKPOINT_PROFILE.items():
        if config.get(field) != required:
            raise ValueError(f"checkpoint requires {field}={required!r}; got {config.get(field)!r}")
    if config.get("model_type") != "qwen3" or config.get("architectures") != ["Qwen3ForCausalLM"]:
        raise ValueError("checkpoint must use the repository Qwen3ForCausalLM")

    arm = config.get("history_memory_arm")
    if arm not in {"B", "C"}:
        raise ValueError("checkpoint history_memory_arm must be B or C")
    if expected_arm is not None and arm != expected_arm:
        raise ValueError(f"candidate arm {expected_arm} differs from checkpoint arm {arm}")
    seed = config.get("history_memory_seed")
    ratios = config.get("history_memory_supported_ratios")
    packing = config.get("history_memory_packing")
    policy = config.get("history_memory_policy")
    corpus_identity = config.get("history_memory_corpus_identity")
    if type(seed) is not int:
        raise ValueError("checkpoint history_memory_seed must be an integer")
    if (
        not isinstance(ratios, list)
        or not ratios
        or any(type(ratio) is not int or ratio <= 0 for ratio in ratios)
        or len(ratios) != len(set(ratios))
    ):
        raise ValueError("checkpoint must declare unique positive supported ratios")
    if not isinstance(packing, Mapping) or packing.get("ratios") != ratios:
        raise ValueError("checkpoint packing ratios must match supported ratios")
    if not isinstance(policy, Mapping):
        raise ValueError("checkpoint history_memory_policy must be an object")
    if not isinstance(corpus_identity, str) or not corpus_identity:
        raise ValueError("checkpoint must declare history_memory_corpus_identity")

    contract = state.get("contract")
    if not isinstance(contract, Mapping):
        raise ValueError("trainer_state.contract must be an object")
    expected_contract = {
        "arm": arm,
        "seed": seed,
        "corpus_identity": corpus_identity,
        "profile": TRAINING_PROFILE,
    }
    for field, required in expected_contract.items():
        if contract.get(field) != required:
            raise ValueError(
                f"trainer_state.contract.{field} differs from checkpoint config: "
                f"{contract.get(field)!r} vs {required!r}"
            )
    if state.get("training_profile") != TRAINING_PROFILE:
        raise ValueError("trainer_state has a different training profile")
    if state.get("global_step") != step or state.get("parameter_version") != step:
        raise ValueError("checkpoint name, global_step and parameter_version must agree")
    if type(state.get("completed")) is not bool:
        raise ValueError("trainer_state.completed must be boolean")
    planned_steps = contract.get("planned_steps")
    if type(planned_steps) is not int or planned_steps < step:
        raise ValueError("trainer_state.contract.planned_steps must cover the checkpoint step")

    paired_contract = {key: value for key, value in contract.items() if key != "arm"}
    return {
        "path": str(checkpoint),
        "arm": arm,
        "step": step,
        "seed": seed,
        "corpus_identity": corpus_identity,
        "training_profile": TRAINING_PROFILE,
        "parameter_version": step,
        "completed": state["completed"],
        "planned_steps": planned_steps,
        "supported_ratios": list(ratios),
        "config_sha256": _sha256_file(config_path),
        "trainer_state_sha256": _sha256_file(state_path),
        "training_contract_sha256": _canonical_sha256(contract),
        "paired_contract_sha256": _canonical_sha256(paired_contract),
        "contract": dict(contract),
    }


def _load_task_manifest(path: Path, split: str) -> dict[str, Any]:
    value = _read_json(path)
    if not isinstance(value, Mapping):
        raise ValueError("task manifest must be a JSON object")
    if value.get("split") != split:
        raise ValueError(
            f"task manifest split {value.get('split')!r} does not match evaluation split {split!r}"
        )
    if value.get("category") != "multi_turn_base":
        raise ValueError("task manifest category must be multi_turn_base")
    ids = value.get("ids")
    if (
        not isinstance(ids, list)
        or not ids
        or any(
            not isinstance(task_id, str)
            or not task_id.startswith("multi_turn_base_")
            or not task_id.removeprefix("multi_turn_base_").isdigit()
            for task_id in ids
        )
        or len(ids) != len(set(ids))
    ):
        raise ValueError("task manifest ids must be unique multi_turn_base IDs")
    if value.get("n_total") != len(ids):
        raise ValueError("task manifest n_total must equal the fixed ID count")
    if "categories" in value and value["categories"] != ["multi_turn_base"]:
        raise ValueError("task manifest categories must contain only multi_turn_base")
    items = value.get("items")
    if items is not None:
        if not isinstance(items, list) or [item.get("id") for item in items] != ids:
            raise ValueError("task manifest items must preserve the exact ids order")
        if any(item.get("category") != "multi_turn_base" for item in items):
            raise ValueError("task manifest items must use multi_turn_base")
    return {
        "path": str(path),
        "sha256": _sha256_file(path),
        "split": split,
        "category": "multi_turn_base",
        "task_ids": list(ids),
        "fixed_task_count": len(ids),
        "ids_sha256": _canonical_sha256(sorted(ids)),
    }


def _validate_paired_candidates(
    candidates: Sequence[Mapping[str, Any]], *, split: str
) -> None:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        grouped[candidate["arm"]].append(candidate)
    for arm, values in grouped.items():
        steps = [value["checkpoint_profile"]["step"] for value in values]
        if len(steps) != len(set(steps)):
            raise ValueError(f"arm {arm} contains duplicate checkpoint steps")
        identities = {
            (
                value["checkpoint_profile"]["seed"],
                value["checkpoint_profile"]["corpus_identity"],
                value["checkpoint_profile"]["paired_contract_sha256"],
            )
            for value in values
        }
        if len(identities) != 1:
            raise ValueError(f"arm {arm} candidates do not share one training identity")
    if set(grouped) == {"B", "C"}:
        b_by_step = {value["checkpoint_profile"]["step"]: value for value in grouped["B"]}
        c_by_step = {value["checkpoint_profile"]["step"]: value for value in grouped["C"]}
        if split == "dev" and set(b_by_step) != set(c_by_step):
            raise ValueError("B and C candidates must use equal checkpoint step sets")
        pairs = (
            ((step, b_by_step[step], c_by_step[step]) for step in b_by_step)
            if split == "dev"
            else ((None, grouped["B"][0], grouped["C"][0]),)
        )
        for step, b_candidate, c_candidate in pairs:
            b_profile = b_candidate["checkpoint_profile"]
            c_profile = c_candidate["checkpoint_profile"]
            for field in ("seed", "corpus_identity", "paired_contract_sha256"):
                if b_profile[field] != c_profile[field]:
                    location = f" at step {step}" if step is not None else ""
                    raise ValueError(f"B/C training identity differs{location}: {field}")


def load_manifest(path: Path) -> dict[str, Any]:
    """Load and fully preflight one explicit evaluation manifest."""
    manifest_path = Path(path).resolve()
    raw = _read_json(manifest_path)
    if not isinstance(raw, Mapping) or raw.get("schema") != EVAL_SCHEMA:
        raise ValueError(f"manifest schema must be {EVAL_SCHEMA!r}")
    split = raw.get("split")
    if split not in {"dev", "heldout"}:
        raise ValueError("split must be dev or heldout")
    runtime_root = _absolute_dir(raw.get("runtime_root"), "runtime_root")
    for relative in (
        "benchmarks/memory_runtime/event_native_server.py",
        "benchmarks/memory_runtime/event_native_bfcl.py",
    ):
        if not (runtime_root / relative).is_file():
            raise FileNotFoundError(f"runtime_root lacks {relative}")
    bfcl_dir = _absolute_dir(raw.get("bfcl_benchmark_dir"), "bfcl_benchmark_dir")
    if not (bfcl_dir / "bfcl_eval").is_dir():
        raise ValueError("bfcl_benchmark_dir must contain bfcl_eval")
    task_path = _absolute_file(raw.get("task_manifest"), "task_manifest")
    eval_policy = _absolute_file(raw.get("eval_policy"), "eval_policy")
    eval_policy_value = _read_json(eval_policy)
    if not isinstance(eval_policy_value, Mapping) or eval_policy_value.get("schema") not in EVAL_POLICY_SCHEMAS:
        raise ValueError("eval_policy does not use a supported A evaluation policy schema")
    output_dir = _absolute_dir(raw.get("output_dir"), "output_dir", must_exist=False)
    if output_dir.exists():
        raise FileExistsError(f"output_dir already exists: {output_dir}")
    view_mode = raw.get("view_mode")
    if view_mode not in OVERRIDE_VIEW_MODES:
        raise ValueError("view_mode must support an explicit evaluation policy")
    dtype = raw.get("dtype")
    if dtype not in {"float32", "bfloat16", "float16"}:
        raise ValueError("dtype must be float32, bfloat16 or float16")
    device = raw.get("device")
    if not isinstance(device, str) or not device.strip():
        raise ValueError("device must be explicit and nonempty")

    candidates_raw = raw.get("candidates")
    if not isinstance(candidates_raw, list) or not candidates_raw:
        raise ValueError("candidates must be a nonempty list")
    candidates = []
    for index, item in enumerate(candidates_raw):
        if not isinstance(item, Mapping) or item.get("arm") not in {"B", "C"}:
            raise ValueError(f"candidates[{index}] must declare arm B or C")
        checkpoint = _absolute_dir(item.get("checkpoint"), f"candidates[{index}].checkpoint")
        profile = inspect_checkpoint(checkpoint, item["arm"])
        overlap = item.get("overlap_audit")
        overlap_path = None
        if overlap is not None:
            overlap_path = _absolute_file(overlap, f"candidates[{index}].overlap_audit")
        if split == "heldout" and overlap_path is None:
            raise ValueError("heldout candidates require a checkpoint-bound overlap_audit")
        candidates.append(
            {
                "arm": item["arm"],
                "checkpoint": checkpoint,
                "checkpoint_profile": profile,
                "overlap_audit": overlap_path,
                "overlap_audit_sha256": _sha256_file(overlap_path) if overlap_path else None,
            }
        )
    _validate_paired_candidates(candidates, split=split)
    if split == "heldout":
        counts = {arm: sum(candidate["arm"] == arm for candidate in candidates) for arm in {"B", "C"}}
        if counts != {"B": 1, "C": 1}:
            raise ValueError("heldout evaluation requires exactly one checkpoint per arm")

    ratio = _positive_int(raw.get("ratio"), "ratio")
    for candidate in candidates:
        if ratio not in candidate["checkpoint_profile"]["supported_ratios"]:
            raise ValueError(
                f"ratio {ratio} is absent from {candidate['checkpoint']}"
            )
    ready_timeout = _positive_number(raw.get("ready_timeout_seconds"), "ready_timeout_seconds")
    bfcl_wall = _positive_number(raw.get("bfcl_max_wall_seconds"), "bfcl_max_wall_seconds")
    server_wall = _positive_number(raw.get("server_max_wall_seconds"), "server_max_wall_seconds")
    if server_wall < ready_timeout + bfcl_wall:
        raise ValueError("server wall cap must cover ready timeout plus BFCL wall cap")

    task = _load_task_manifest(task_path, split)
    return {
        "schema": EVAL_SCHEMA,
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256_file(manifest_path),
        "split": split,
        "runtime_root": runtime_root,
        "bfcl_benchmark_dir": bfcl_dir,
        "task_manifest": task,
        "eval_policy": eval_policy,
        "eval_policy_sha256": _sha256_file(eval_policy),
        "output_dir": output_dir,
        "view_mode": view_mode,
        "compression_policy": "always-compress-v1" if view_mode in ALWAYS_VIEW_MODES else None,
        "device": device,
        "dtype": dtype,
        "ratio": ratio,
        "max_new_tokens": _positive_int(raw.get("max_new_tokens"), "max_new_tokens"),
        "max_decisions": _positive_int(raw.get("max_decisions"), "max_decisions"),
        "max_generation_calls": _positive_int(
            raw.get("max_generation_calls"), "max_generation_calls"
        ),
        "server_max_wall_seconds": server_wall,
        "bfcl_max_wall_seconds": bfcl_wall,
        "ready_timeout_seconds": ready_timeout,
        "torch_threads": _positive_int(raw.get("torch_threads", 1), "torch_threads"),
        "candidates": candidates,
    }


def select_best(records: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    """Select per arm by BFCL success rate, breaking exact ties by earlier step."""
    winners: dict[str, Mapping[str, Any]] = {}
    for record in records:
        arm = record.get("arm")
        score = record.get("selection_score")
        step = record.get("step")
        if arm not in {"B", "C"}:
            raise ValueError("candidate result arm must be B or C")
        if (
            not isinstance(score, (int, float))
            or isinstance(score, bool)
            or not math.isfinite(float(score))
            or not 0 <= score <= 1
        ):
            raise ValueError("candidate selection_score must be finite in [0, 1]")
        if type(step) is not int or step <= 0:
            raise ValueError("candidate step must be a positive integer")
        current = winners.get(arm)
        if current is None or (-float(score), step) < (
            -float(current["selection_score"]),
            current["step"],
        ):
            winners[arm] = record
    if not winners:
        raise ValueError("candidate results are empty")
    return winners


def _runtime_identity(root: Path) -> dict[str, Any]:
    def git(*arguments: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(root), *arguments],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return result.stdout.strip()

    commit = git("rev-parse", "HEAD")
    dirty = bool(git("status", "--porcelain"))
    if dirty:
        raise ValueError("runtime_root must be a clean published A worktree")
    return {"root": str(root), "git_commit": commit, "dirty": False}


def _process_flags() -> dict[str, Any]:
    if os.name == "posix":
        return {"start_new_session": True}
    return {
        "creationflags": getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        | getattr(subprocess, "CREATE_NO_WINDOW", 0)
    }


def build_server_command(
    config: Mapping[str, Any], candidate: Mapping[str, Any], candidate_dir: Path
) -> list[str]:
    profile = candidate["checkpoint_profile"]
    tasks = config["task_manifest"]["task_ids"]
    command = [
        sys.executable,
        "-m",
        "benchmarks.memory_runtime.event_native_server",
        "--serve-child",
        "--checkpoint",
        str(candidate["checkpoint"]),
        "--out",
        str(candidate_dir / "server"),
        "--run-id",
        f"history-{candidate['arm'].lower()}-step-{profile['step']}",
        "--model-name",
        f"c2kv-history-{candidate['arm'].lower()}-step-{profile['step']}",
        "--benchmark",
        "bfcl",
        "--source-profile",
        "native-v1",
        "--view-mode",
        config["view_mode"],
        "--ratio",
        str(config["ratio"]),
        "--max-new-tokens",
        str(config["max_new_tokens"]),
        "--decode-strategy",
        "incremental",
        "--task-ids",
        ",".join(tasks),
        "--max-decisions",
        str(config["max_decisions"]),
        "--max-generation-calls",
        str(config["max_generation_calls"]),
        "--eval-policy",
        str(config["eval_policy"]),
        "--max-wall-seconds",
        str(config["server_max_wall_seconds"]),
        "--device",
        config["device"],
        "--dtype",
        config["dtype"],
        "--host",
        "127.0.0.1",
        "--port",
        "0",
        "--torch-threads",
        str(config["torch_threads"]),
    ]
    if config["compression_policy"] is not None:
        command.extend(["--compression-policy", config["compression_policy"]])
    command.extend(["--history-view-protocol", "fixed-budget-main"])
    return command


def build_bfcl_command(
    config: Mapping[str, Any], candidate: Mapping[str, Any], candidate_dir: Path, ready: Mapping[str, Any]
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "benchmarks.memory_runtime.event_native_bfcl",
        "--server-manifest",
        str(candidate_dir / "server" / "ready.json"),
        "--base-url",
        ready["base_url"],
        "--benchmark-dir",
        str(config["bfcl_benchmark_dir"]),
        "--out",
        str(candidate_dir / "bfcl"),
        "--max-wall-seconds",
        str(config["bfcl_max_wall_seconds"]),
    ]
    if candidate["overlap_audit"] is not None:
        command.extend(["--overlap-audit", str(candidate["overlap_audit"])])
    return command


def _environment(runtime_root: Path) -> dict[str, str]:
    environment = os.environ.copy()
    paths = [str(runtime_root / "python"), str(runtime_root)]
    if environment.get("PYTHONPATH"):
        paths.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(paths)
    return environment


def _wait_ready(process: subprocess.Popen[Any], ready_path: Path, timeout: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if ready_path.is_file():
            ready = _read_json(ready_path)
            if ready.get("schema") != "a-event-native-server-v1" or ready.get("status") != "ready":
                raise ValueError("native server ready manifest has an invalid identity")
            return ready
        returncode = process.poll()
        if returncode is not None:
            raise RuntimeError(f"native server exited before readiness with code {returncode}")
        time.sleep(0.1)
    raise TimeoutError("native server readiness deadline expired")


def _stop_owned_process(process: subprocess.Popen[Any], timeout: float = 10.0) -> int:
    if process.poll() is None:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
    try:
        return process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
        return process.wait(timeout=5)


def _run_bfcl_process(
    command: Sequence[str], *, cwd: Path, environment: Mapping[str, str], log: Any, wall_cap: float
) -> int:
    """Run A's bounded supervisor with a small outer cleanup deadline."""
    process = subprocess.Popen(
        list(command),
        cwd=cwd,
        env=dict(environment),
        stdout=log,
        stderr=subprocess.STDOUT,
        **_process_flags(),
    )
    try:
        return process.wait(timeout=wall_cap + 15.0)
    except BaseException as error:
        timed_out = isinstance(error, subprocess.TimeoutExpired)
        if process.poll() is None:
            try:
                if os.name == "posix":
                    os.killpg(process.pid, signal.SIGINT)
                elif hasattr(signal, "CTRL_BREAK_EVENT"):
                    process.send_signal(signal.CTRL_BREAK_EVENT)
                else:
                    process.terminate()
            except ProcessLookupError:
                pass
        try:
            process.wait(timeout=10.0)
        except subprocess.TimeoutExpired:
            _stop_owned_process(process)
        if timed_out:
            raise TimeoutError(
                "official BFCL supervisor exceeded its wall cap and cleanup grace"
            ) from error
        raise


def _save_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False, default=str)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _validate_official_summary(summary: Mapping[str, Any], fixed_count: int) -> tuple[int, float]:
    correct = summary.get("correct_count")
    scored = summary.get("n_scored")
    if summary.get("scored") is not True:
        raise ValueError("official BFCL summary is not scored")
    if type(correct) is not int or not 0 <= correct <= fixed_count:
        raise ValueError("official BFCL correct_count is invalid")
    if scored != fixed_count:
        raise ValueError(
            f"official BFCL coverage {scored!r} differs from fixed task count {fixed_count}"
        )
    score = correct / fixed_count
    if not math.isclose(float(summary.get("semantic_score", -1)), score, rel_tol=0, abs_tol=1e-12):
        raise ValueError("official BFCL semantic_score differs from correct_count / fixed task count")
    return correct, score


def read_official_summary(
    summary_path: Path, runner_final_path: Path, fixed_task_count: int
) -> dict[str, Any]:
    """Read a completed official run and derive the only selection score."""
    _positive_int(fixed_task_count, "fixed_task_count")
    summary_path = Path(summary_path)
    runner_final_path = Path(runner_final_path)
    summary = _read_json(summary_path)
    final = _read_json(runner_final_path)
    if not isinstance(summary, Mapping) or not isinstance(final, Mapping):
        raise ValueError("official BFCL summary and final receipt must be JSON objects")
    if final.get("status") != "completed" or final.get("worker_returncode") != 0:
        raise ValueError("official BFCL runner did not complete successfully")
    correct, score = _validate_official_summary(summary, fixed_task_count)
    return {
        "correct_count": correct,
        "n_scored": fixed_task_count,
        "selection_score": score,
        "summary_sha256": _sha256_file(summary_path),
        "runner_final_sha256": _sha256_file(runner_final_path),
    }


def _candidate_output_name(candidate: Mapping[str, Any]) -> str:
    return f"arm-{candidate['arm']}-step-{candidate['checkpoint_profile']['step']}"


def _run_candidate(
    config: Mapping[str, Any], candidate: Mapping[str, Any], runtime: Mapping[str, Any]
) -> dict[str, Any]:
    candidate_dir = config["output_dir"] / _candidate_output_name(candidate)
    candidate_dir.mkdir()
    server_command = build_server_command(config, candidate, candidate_dir)
    environment = _environment(config["runtime_root"])
    result: dict[str, Any] = {
        "schema": "history-memory-native-bfcl-candidate-v1",
        "status": "running",
        "split": config["split"],
        "arm": candidate["arm"],
        "step": candidate["checkpoint_profile"]["step"],
        "checkpoint": candidate["checkpoint_profile"],
        "overlap_audit": (
            {
                "path": str(candidate["overlap_audit"]),
                "sha256": candidate["overlap_audit_sha256"],
            }
            if candidate["overlap_audit"] is not None
            else None
        ),
        "task_manifest": config["task_manifest"],
        "eval_policy": {
            "path": str(config["eval_policy"]),
            "sha256": config["eval_policy_sha256"],
        },
        "runtime": dict(runtime),
        "server_command": server_command,
    }
    _save_json(candidate_dir / "candidate.json", result)
    server = None
    server_log = (candidate_dir / "server.log").open("x", encoding="utf-8", newline="\n")
    try:
        server = subprocess.Popen(
            server_command,
            cwd=config["runtime_root"],
            env=environment,
            stdout=server_log,
            stderr=subprocess.STDOUT,
            **_process_flags(),
        )
        result["server_pid"] = server.pid
        ready = _wait_ready(
            server, candidate_dir / "server" / "ready.json", config["ready_timeout_seconds"]
        )
        if ready.get("checkpoint_path") != str(candidate["checkpoint"]):
            raise ValueError("native server loaded a different checkpoint path")
        if ready.get("allowed_task_ids") != config["task_manifest"]["task_ids"]:
            raise ValueError("native server task identity differs from the fixed manifest")
        if ready.get("checkpoint", {}).get("training_arm") != candidate["arm"]:
            raise ValueError("native server checkpoint arm differs from the candidate")
        bfcl_command = build_bfcl_command(config, candidate, candidate_dir, ready)
        result.update(server_ready=ready, bfcl_command=bfcl_command)
        _save_json(candidate_dir / "candidate.json", result)
        with (candidate_dir / "bfcl.log").open("x", encoding="utf-8", newline="\n") as bfcl_log:
            bfcl_returncode = _run_bfcl_process(
                bfcl_command,
                cwd=config["runtime_root"],
                environment=environment,
                log=bfcl_log,
                wall_cap=config["bfcl_max_wall_seconds"],
            )
        if bfcl_returncode != 0:
            raise RuntimeError(f"official BFCL runner failed with code {bfcl_returncode}")
        summary_path = candidate_dir / "bfcl" / "official_summary.json"
        bfcl_final_path = candidate_dir / "bfcl" / "final.json"
        official = read_official_summary(
            summary_path,
            bfcl_final_path,
            config["task_manifest"]["fixed_task_count"],
        )
        result.update(
            status="completed",
            correct_count=official["correct_count"],
            n_scored=official["n_scored"],
            selection_score=official["selection_score"],
            selection_metric="official_bfcl_complete_task_success_rate",
            official_summary_path=str(summary_path),
            official_summary_sha256=official["summary_sha256"],
            official_runner_final_path=str(bfcl_final_path),
            official_runner_final_sha256=official["runner_final_sha256"],
        )
    except BaseException as error:
        result.update(status="failed", error={"type": type(error).__name__, "message": str(error)})
        raise
    finally:
        if server is not None:
            result["server_returncode"] = _stop_owned_process(server)
        server_log.close()
        server_final_path = candidate_dir / "server" / "final.json"
        if server_final_path.is_file():
            result["server_final_path"] = str(server_final_path)
            result["server_final_sha256"] = _sha256_file(server_final_path)
        _save_json(candidate_dir / "candidate.json", result)
    return result


def run(config: Mapping[str, Any]) -> dict[str, Any]:
    """Execute all prevalidated candidates serially and write final provenance."""
    runtime = _runtime_identity(config["runtime_root"])
    output = config["output_dir"]
    output.mkdir(parents=True)
    run_record: dict[str, Any] = {
        "schema": EVAL_SCHEMA,
        "status": "running",
        "split": config["split"],
        "source_manifest": {
            "path": config["manifest_path"],
            "sha256": config["manifest_sha256"],
        },
        "runtime": runtime,
        "task_manifest": config["task_manifest"],
        "eval_policy": {
            "path": str(config["eval_policy"]),
            "sha256": config["eval_policy_sha256"],
        },
        "evaluation": {
            name: config[name]
            for name in (
                "view_mode",
                "device",
                "dtype",
                "ratio",
                "max_new_tokens",
                "max_decisions",
                "max_generation_calls",
                "server_max_wall_seconds",
                "bfcl_max_wall_seconds",
                "ready_timeout_seconds",
                "torch_threads",
            )
        },
        "candidates": [],
    }
    _save_json(output / "run.json", run_record)
    try:
        for candidate in config["candidates"]:
            run_record["candidates"].append(_run_candidate(config, candidate, runtime))
            _save_json(output / "run.json", run_record)
        if config["split"] == "dev":
            winners = select_best(run_record["candidates"])
            selection = {
                "schema": "history-memory-native-bfcl-selection-v1",
                "split": "dev",
                "selection_metric": "official_bfcl_complete_task_success_rate",
                "denominator": config["task_manifest"]["fixed_task_count"],
                "tie_break": "earlier_checkpoint_step",
                "loss_used_for_selection": False,
                "task_manifest": config["task_manifest"],
                "runtime": runtime,
                "winners": {
                    arm: {
                        "arm": winner["arm"],
                        "step": winner["step"],
                        "checkpoint": winner["checkpoint"]["path"],
                        "correct_count": winner["correct_count"],
                        "n_scored": winner["n_scored"],
                        "selection_score": winner["selection_score"],
                    }
                    for arm, winner in sorted(winners.items())
                },
            }
            _save_json(output / "selection.json", selection)
            run_record["selection_path"] = str(output / "selection.json")
        else:
            evaluation = {
                "schema": "history-memory-native-bfcl-heldout-evaluation-v1",
                "split": "heldout",
                "selection_performed": False,
                "task_manifest": config["task_manifest"],
                "runtime": runtime,
                "results": run_record["candidates"],
            }
            _save_json(output / "evaluation.json", evaluation)
            run_record["evaluation_path"] = str(output / "evaluation.json")
        run_record["status"] = "completed"
    except BaseException as error:
        run_record.update(status="failed", error={"type": type(error).__name__, "message": str(error)})
        raise
    finally:
        _save_json(output / "run.json", run_record)
    return run_record


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    config = load_manifest(args.manifest)
    runtime = _runtime_identity(config["runtime_root"])
    if args.dry_run:
        plan = {
            "schema": EVAL_SCHEMA,
            "status": "validated",
            "split": config["split"],
            "runtime": runtime,
            "task_manifest": config["task_manifest"],
            "candidate_commands": [
                {
                    "arm": candidate["arm"],
                    "step": candidate["checkpoint_profile"]["step"],
                    "server_command": build_server_command(
                        config,
                        candidate,
                        config["output_dir"] / _candidate_output_name(candidate),
                    ),
                }
                for candidate in config["candidates"]
            ],
        }
        print(json.dumps(plan, ensure_ascii=False, indent=2, allow_nan=False, default=str))
        return 0
    run(config)
    return 0


__all__ = [
    "EVAL_SCHEMA",
    "build_bfcl_command",
    "build_server_command",
    "inspect_checkpoint",
    "load_manifest",
    "main",
    "read_official_summary",
    "run",
    "select_best",
]
