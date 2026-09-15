"""Finite standalone loopback server for native HiAgent phase calls."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from benchmarks.native_hiagent_protocol import load_policy_sampling

from .attempt_journal import AttemptJournal, summarize_attempt_journal
from .event_native import (
    inspect_checkpoint,
    load_generator,
    validate_inference_byte_profile,
)
from .event_native_eval_packing import resolve_eval_packing
from .event_native_eval_policy import (
    load_eval_policy,
    resolve_event_native_eval_policy,
)
from .event_native_hiagent import (
    ACTOR_PHASES,
    COMPRESSOR_MAX_COMPLETION_TOKENS,
    build_event_native_hiagent_dispatcher,
    make_hiagent_server,
    validate_native_greedy_sampling,
)


SCHEMA = "a-event-native-hiagent-server-v1"
SUPERVISOR_SCHEMA = "a-event-native-hiagent-server-supervisor-v1"
PROXY_SHARED_CALL_CAP_PER_TASK = 96
VIEW_MODE = "full_original"


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def positive_seconds(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive and finite")
    return parsed


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--checkpoint", type=Path, required=True)
    result.add_argument("--out", type=Path, required=True)
    result.add_argument("--run-id", required=True)
    result.add_argument("--model-name", default="c2kv-event-native-hiagent")
    result.add_argument("--benchmark", choices=("bfcl", "acebench"), default="bfcl")
    result.add_argument("--ratio", type=positive_int, required=True)
    result.add_argument(
        "--policy-sampling",
        type=Path,
        required=True,
        help=(
            "Explicit policy and trajectory-retrieval sampling contract. "
            "The current runtime rejects sampling it cannot implement."
        ),
    )
    result.add_argument(
        "--task-ids", required=True, help="Frozen comma-separated official task IDs."
    )
    result.add_argument("--max-actor-generation-calls", type=positive_int, required=True)
    result.add_argument(
        "--max-auxiliary-generation-calls", type=positive_int, required=True
    )
    result.add_argument("--eval-policy", type=Path)
    result.add_argument("--eval-capacity", type=Path)
    result.add_argument(
        "--decode-strategy",
        choices=("incremental", "full_recompute"),
        default="incremental",
    )
    result.add_argument("--prefill-chunk-size", type=positive_int)
    result.add_argument("--max-wall-seconds", type=positive_seconds, required=True)
    result.add_argument("--device", default="cpu")
    result.add_argument(
        "--npu-allocator-metrics",
        action="store_true",
        help="Measure actor and auxiliary generation windows independently.",
    )
    result.add_argument(
        "--dtype", choices=("float32", "bfloat16", "float16"), default="float32"
    )
    result.add_argument(
        "--host", choices=("127.0.0.1", "::1", "localhost"), default="127.0.0.1"
    )
    result.add_argument("--port", type=int, default=0)
    result.add_argument("--torch-threads", type=positive_int, default=1)
    result.add_argument(
        "--preview",
        action="store_true",
        help="Validate local contracts and print a read-only launch preview.",
    )
    result.add_argument("--serve-child", action="store_true", help=argparse.SUPPRESS)
    return result


def save_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _task_ids(value: str) -> tuple[str, ...]:
    task_ids = tuple(item.strip() for item in value.split(","))
    if not task_ids or not all(task_ids) or len(task_ids) != len(set(task_ids)):
        raise ValueError("task IDs must be nonempty and unique")
    return task_ids


def _validate_allocator_device(args: argparse.Namespace) -> None:
    if args.npu_allocator_metrics and args.device.split(":", 1)[0] != "npu":
        raise ValueError("NPU allocator metrics require an npu device")


def _sampling_contract(path: Path) -> dict[str, Any]:
    content = path.read_bytes()
    sampling = load_policy_sampling(path)
    for phase in sorted(ACTOR_PHASES):
        validate_native_greedy_sampling(sampling, phase=phase)
    return {
        "source": str(path.resolve()),
        "sha256": hashlib.sha256(content).hexdigest(),
        "sampling": sampling,
    }


def preview(
    args: argparse.Namespace,
    *,
    inspect_checkpoint_fn: Callable[[str | Path], dict[str, Any]] = inspect_checkpoint,
) -> dict[str, Any]:
    """Validate the launch contract without creating output or loading weights."""

    _validate_allocator_device(args)
    if not args.run_id.strip() or not args.model_name.strip():
        raise ValueError("run and model identities must be nonempty")
    task_ids = _task_ids(args.task_ids)
    if not 0 <= args.port <= 65535:
        raise ValueError("invalid loopback port")
    if args.prefill_chunk_size is not None and args.decode_strategy != "incremental":
        raise ValueError("prefill chunking requires incremental decode")

    profile = inspect_checkpoint_fn(args.checkpoint)
    expected_kv_bytes = validate_inference_byte_profile(profile, args.dtype)
    if args.ratio not in profile["declared_supported_ratios"]:
        raise ValueError("ratio is absent from checkpoint contract")
    context = profile["model_geometry"].get("max_position_embeddings")
    if type(context) is not int or context <= 0:
        raise ValueError("checkpoint must declare model context")
    runtime_policy = resolve_event_native_eval_policy(
        profile,
        view_mode=VIEW_MODE,
        policy_override=(
            load_eval_policy(args.eval_policy) if args.eval_policy is not None else None
        ),
        source_path=(
            str(args.eval_policy.resolve()) if args.eval_policy is not None else None
        ),
    )
    runtime_packing = resolve_eval_packing(profile, args.eval_capacity)
    sampling_contract = _sampling_contract(args.policy_sampling)
    target_cap = runtime_packing["effective_packing"]["max_target_tokens"]
    actor_cap = sampling_contract["sampling"]["max_completion_tokens"]
    if actor_cap > target_cap:
        raise ValueError("actor generation reservation exceeds checkpoint target cap")
    if COMPRESSOR_MAX_COMPLETION_TOKENS > target_cap:
        raise ValueError("compressor generation reservation exceeds checkpoint target cap")

    return {
        "schema": SCHEMA,
        "status": "preview",
        "launch": False,
        "output_created": False,
        "run_id": args.run_id,
        "model_name": args.model_name,
        "benchmark": args.benchmark,
        "view_mode": VIEW_MODE,
        "allowed_task_ids": list(task_ids),
        "checkpoint_path": str(args.checkpoint.resolve()),
        "checkpoint": profile,
        "ratio": args.ratio,
        "runtime_policy_contract": runtime_policy,
        "runtime_packing_contract": runtime_packing,
        "expected_kv_bytes_per_token": expected_kv_bytes,
        "policy_sampling_contract": sampling_contract,
        "compressor_completion_cap": COMPRESSOR_MAX_COMPLETION_TOKENS,
        "decode_strategy": args.decode_strategy,
        "prefill_chunk_size": args.prefill_chunk_size,
        "max_actor_generation_calls": args.max_actor_generation_calls,
        "max_auxiliary_generation_calls": args.max_auxiliary_generation_calls,
        "max_wall_seconds": args.max_wall_seconds,
        "proxy_shared_call_cap_per_task": PROXY_SHARED_CALL_CAP_PER_TASK,
        "proxy_call_cap_owner": "official proxy",
        "runtime_contract": {
            "model_loads": 1,
            "actor_and_auxiliary_share_weights": True,
            "actor_and_auxiliary_caches": "separate",
            "generation_serialization": "dispatcher lock",
        },
    }


def _default_tokenizer_loader(checkpoint: Path) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(checkpoint), local_files_only=True)


def _default_auxiliary_generator_factory(
    runtime: Any,
    *,
    decode_strategy: str,
    prefill_chunk_size: int | None,
) -> Any:
    from history_memory.inference import EventNativeGenerator

    kwargs: dict[str, Any] = {"decode_strategy": decode_strategy}
    if prefill_chunk_size is not None:
        kwargs["prefill_chunk_size"] = prefill_chunk_size
    return EventNativeGenerator(runtime, **kwargs)


def _default_allocator_wrapper(generator: Any) -> Any:
    from .event_native_allocator import NpuAllocatorMeasuredGenerator

    return NpuAllocatorMeasuredGenerator(generator)


def _artifact_paths(out: Path) -> dict[str, Path]:
    return {
        "startup": out / "startup.json",
        "ready": out / "ready.json",
        "final": out / "final.json",
        "actor_journal": out / "actor_attempts.jsonl",
        "auxiliary_journal": out / "auxiliary_attempts.jsonl",
        "steps": out / "hiagent_steps.jsonl",
        "join": out / "hiagent_join.jsonl",
    }


def _serve(
    args: argparse.Namespace,
    *,
    inspect_checkpoint_fn: Callable[[str | Path], dict[str, Any]] = inspect_checkpoint,
    tokenizer_loader: Callable[[Path], Any] | None = None,
    generator_loader: Callable[..., tuple[Any, dict[str, Any]]] = load_generator,
    auxiliary_generator_factory: Callable[..., Any] | None = None,
    dispatcher_builder: Callable[..., Any] = build_event_native_hiagent_dispatcher,
    server_builder: Callable[..., Any] = make_hiagent_server,
    allocator_wrapper: Callable[[Any], Any] | None = None,
) -> None:
    contract = preview(args, inspect_checkpoint_fn=inspect_checkpoint_fn)
    started = time.monotonic()
    deadline = started + args.max_wall_seconds
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    paths = _artifact_paths(out)
    manifest = {
        **contract,
        "status": "loading",
        "launch": True,
        "output_created": True,
        "stage_deadline_owner": "child monotonic deadline plus parent hard cutoff",
        "artifacts": {name: str(path.resolve()) for name, path in paths.items()},
        "artifact_privacy": (
            "steps and joins are local evaluation evidence and may contain tokenized "
            "request material; they are not service logs"
        ),
        "journal_cost_partition": {
            "actor": "policy and trajectory_retrieval_policy generation only",
            "auxiliary": "compressor generation only",
            "merged": False,
        },
        "stop_reason": None,
    }
    save_json(paths["startup"], manifest)

    server = dispatcher = actor_generator = auxiliary_generator = None
    stop_requested = False
    old_handlers: dict[Any, Any] = {}

    def request_stop(_signum: int, _frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True

    try:
        if args.device.split(":", 1)[0] == "npu":
            import torch_npu  # Register the explicitly selected optional backend.
        try:
            import torch
        except ModuleNotFoundError:
            torch = None
        if torch is not None:
            torch.set_num_threads(args.torch_threads)

        tokenizer = (tokenizer_loader or _default_tokenizer_loader)(args.checkpoint)
        generator_kwargs: dict[str, Any] = {
            "device": args.device,
            "dtype": args.dtype,
            "decode_strategy": args.decode_strategy,
        }
        if args.prefill_chunk_size is not None:
            generator_kwargs["prefill_chunk_size"] = args.prefill_chunk_size
        actor_generator, loaded_profile = generator_loader(
            args.checkpoint, **generator_kwargs
        )
        for key, value in contract["checkpoint"].items():
            if loaded_profile.get(key) != value:
                raise ValueError(f"loaded checkpoint identity changed field {key}")
        if actor_generator.kv_bytes_per_token() != contract["expected_kv_bytes_per_token"]:
            raise ValueError("loaded KV geometry differs from the declared budget")

        auxiliary_generator = (
            auxiliary_generator_factory or _default_auxiliary_generator_factory
        )(
            actor_generator.runtime,
            decode_strategy=args.decode_strategy,
            prefill_chunk_size=args.prefill_chunk_size,
        )
        if auxiliary_generator is actor_generator:
            raise ValueError("actor and auxiliary generators must be distinct")
        if auxiliary_generator.runtime is not actor_generator.runtime:
            raise ValueError("actor and auxiliary generators must share one loaded runtime")
        if auxiliary_generator.kv_bytes_per_token() != contract["expected_kv_bytes_per_token"]:
            raise ValueError("auxiliary KV geometry differs from the declared budget")

        manifest["session_cache_contract"] = {
            "actor": actor_generator.session_cache_policy,
            "auxiliary": auxiliary_generator.session_cache_policy,
            "shared_runtime": True,
            "close_after_each_phase_call": True,
        }
        if args.npu_allocator_metrics:
            wrap = allocator_wrapper or _default_allocator_wrapper
            actor_generator = wrap(actor_generator)
            auxiliary_generator = wrap(auxiliary_generator)
            manifest["allocator_measurement_contract"] = {
                "actor": copy.deepcopy(actor_generator.measurement_contract),
                "auxiliary": copy.deepcopy(auxiliary_generator.measurement_contract),
                "serialization": "shared dispatcher lock",
                "merged": False,
            }
        save_json(paths["startup"], manifest)
        if time.monotonic() >= deadline:
            raise TimeoutError("wall budget expired while loading the model")

        sampling = contract["policy_sampling_contract"]["sampling"]
        dispatcher = dispatcher_builder(
            tokenizer,
            actor_generator=actor_generator,
            auxiliary_generator=auxiliary_generator,
            packing=contract["runtime_packing_contract"]["effective_packing"],
            policy=contract["runtime_policy_contract"]["effective_policy"],
            ratio=args.ratio,
            model_name=args.model_name,
            benchmark=args.benchmark,
            allowed_task_ids=contract["allowed_task_ids"],
            actor_phase_sampling={phase: copy.deepcopy(sampling) for phase in ACTOR_PHASES},
            actor_max_generation_calls=args.max_actor_generation_calls,
            auxiliary_max_generation_calls=args.max_auxiliary_generation_calls,
            deadline_monotonic=deadline,
            actor_journal=AttemptJournal(paths["actor_journal"]),
            auxiliary_journal=AttemptJournal(paths["auxiliary_journal"]),
            join_path=paths["join"],
            steps_path=paths["steps"],
            model_context=contract["checkpoint"]["model_geometry"][
                "max_position_embeddings"
            ],
        )
        server = server_builder(dispatcher, host=args.host, port=args.port)
        server.timeout = 0.25
        for name in ("actor_journal", "auxiliary_journal", "steps", "join"):
            _initialize_empty_jsonl(paths[name])
        manifest["evidence_artifacts_initialized"] = {
            "status": "complete",
            "before_ready": True,
            "mode": "0600",
            "paths": [
                str(paths[name].resolve())
                for name in ("actor_journal", "auxiliary_journal", "steps", "join")
            ],
        }
        save_json(paths["startup"], manifest)
        for signum in (signal.SIGINT, signal.SIGTERM):
            old_handlers[signum] = signal.signal(signum, request_stop)
        host, port = server.server_address[:2]
        url_host = f"[{host}]" if ":" in host else host
        manifest.update(
            status="ready",
            base_url=f"http://{url_host}:{port}",
            ready_elapsed_seconds=time.monotonic() - started,
        )
        save_json(paths["ready"], manifest)
        print(
            json.dumps(
                {
                    "status": "ready",
                    "base_url": manifest["base_url"],
                    "ready_file": str(paths["ready"]),
                }
            ),
            flush=True,
        )

        while not stop_requested and time.monotonic() < deadline:
            health = dispatcher.health()
            if health["status"] == "terminal":
                break
            server.handle_request()
        health = dispatcher.health()
        manifest.update(
            status="stopped",
            dispatcher_health=health,
            stop_reason=(
                "signal"
                if stop_requested
                else "terminal_failure"
                if health["status"] == "terminal"
                else "wall_cap"
            ),
        )
    except BaseException as error:
        manifest.update(
            status="failed",
            stop_reason="error",
            error={"type": type(error).__name__, "message": str(error)},
        )
        raise
    finally:
        if server is not None:
            server.server_close()
        if auxiliary_generator is not None:
            manifest["auxiliary_cache_before_final_close"] = (
                auxiliary_generator.session_cache_info()
            )
            auxiliary_generator.close_session()
            manifest["auxiliary_cache_after_final_close"] = (
                auxiliary_generator.session_cache_info()
            )
        if actor_generator is not None:
            manifest["actor_cache_before_final_close"] = actor_generator.session_cache_info()
            actor_generator.close_session()
            manifest["actor_cache_after_final_close"] = actor_generator.session_cache_info()
        for signum, old_handler in old_handlers.items():
            signal.signal(signum, old_handler)
        manifest["wall_seconds"] = time.monotonic() - started
        manifest["wall_seconds_final"] = True
        manifest["actor_journal_summary"] = _journal_summary(paths["actor_journal"])
        manifest["auxiliary_journal_summary"] = _journal_summary(
            paths["auxiliary_journal"]
        )
        manifest["step_inventory"] = _jsonl_inventory(paths["steps"])
        manifest["join_inventory"] = _jsonl_inventory(paths["join"])
        manifest["allocator_by_phase"] = _allocator_by_phase(
            paths["steps"], enabled=args.npu_allocator_metrics
        )
        if dispatcher is not None:
            manifest["dispatcher_health"] = dispatcher.health()
        save_json(paths["final"], manifest)

    print(
        json.dumps(
            {
                "status": manifest["status"],
                "stop_reason": manifest["stop_reason"],
                "output": str(out),
            }
        ),
        flush=True,
    )


def _read_complete_jsonl(path: Path) -> tuple[list[dict[str, Any]], bool]:
    if not path.exists():
        return [], False
    raw = path.read_bytes()
    truncated = bool(raw) and not raw.endswith(b"\n")
    complete = raw.rpartition(b"\n")[0] if truncated else raw
    rows = []
    for line_number, line in enumerate(complete.splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"{path.name} line {line_number} is invalid JSON") from error
        if not isinstance(row, dict):
            raise ValueError(f"{path.name} line {line_number} must be an object")
        rows.append(row)
    return rows, truncated


def _initialize_empty_jsonl(path: Path) -> None:
    """Create one durable empty evidence stream without replacing prior data."""

    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_BINARY", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _jsonl_inventory(path: Path) -> dict[str, Any]:
    rows, truncated = _read_complete_jsonl(path)
    exists = path.exists()
    return {
        "path": str(path.resolve()),
        "exists": exists,
        "complete_records": len(rows) if exists else None,
        "truncated_tail": truncated if exists else None,
        "bytes": path.stat().st_size if exists else None,
    }


def _journal_summary(path: Path) -> dict[str, Any]:
    if path.exists():
        result = summarize_attempt_journal(path)
        result["exists"] = True
        result["initialized_empty"] = path.stat().st_size == 0
        return result
    return {
        "exists": False,
        "started": None,
        "finished": None,
        "completed": None,
        "failed": None,
        "pending": None,
        "truncated_tail": None,
        "scope": "attempt count is unknown because no initialized journal exists",
    }


def _allocator_by_phase(path: Path, *, enabled: bool) -> dict[str, Any]:
    rows, truncated = _read_complete_jsonl(path)
    exists = path.exists()
    result: dict[str, Any] = {
        "schema": "a-event-native-hiagent-allocator-by-phase-v1",
        "enabled": enabled,
        "source": str(path.resolve()),
        "source_exists": exists,
        "source_truncated_tail": truncated if exists else None,
        "by_phase": {},
    }
    for phase in ("compressor", "policy", "trajectory_retrieval_policy"):
        if not exists:
            result["by_phase"][phase] = {
                "generator_role": "auxiliary" if phase == "compressor" else "actor",
                "step_records": None,
                "generation_attempts": None,
                "measurement_records": None,
                "measurement_status": None,
                "known_peak_allocated_bytes": None,
                "strict_peak_allocated_bytes": None,
                "record_scope": "unknown because hiagent_steps.jsonl was not initialized",
            }
            continue
        phase_rows = [row for row in rows if row.get("phase") == phase]
        attempts = []
        for row in phase_rows:
            record = row.get("runner_record")
            trace = record.get("generation_trace", []) if isinstance(record, Mapping) else []
            for item in trace if isinstance(trace, list) else []:
                generation = item.get("generation") if isinstance(item, Mapping) else None
                stats = generation.get("stats") if isinstance(generation, Mapping) else None
                attempts.append(
                    stats.get("allocator_measurement")
                    if isinstance(stats, Mapping)
                    else None
                )
        known = [value for value in attempts if isinstance(value, Mapping)]
        peaks = [
            value.get("peak_allocated_bytes")
            for value in known
            if type(value.get("peak_allocated_bytes")) is int
        ]
        missing = len(attempts) - len(known)
        result["by_phase"][phase] = {
            "generator_role": "auxiliary" if phase == "compressor" else "actor",
            "step_records": len(phase_rows),
            "generation_attempts": len(attempts),
            "measurement_records": len(known),
            "measurement_status": {
                "ok": sum(value.get("status") == "ok" for value in known),
                "failed": sum(value.get("status") == "failed" for value in known),
                "missing": missing,
            },
            "known_peak_allocated_bytes": max(peaks) if peaks else None,
            "strict_peak_allocated_bytes": (
                max(peaks) if peaks and missing == 0 and len(peaks) == len(attempts) else None
            ),
            "record_scope": "complete allocator records remain in hiagent_steps.jsonl",
        }
    return result


def _supervisor_path(out: Path) -> Path:
    return out.parent / f"{out.name}.supervisor.json"


def _child_command(args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "benchmarks.memory_runtime.event_native_hiagent_server",
        "--serve-child",
        "--checkpoint",
        str(args.checkpoint.resolve()),
        "--out",
        str(args.out.resolve()),
        "--run-id",
        args.run_id,
        "--model-name",
        args.model_name,
        "--benchmark",
        args.benchmark,
        "--ratio",
        str(args.ratio),
        "--policy-sampling",
        str(args.policy_sampling.resolve()),
        "--task-ids",
        args.task_ids,
        "--max-actor-generation-calls",
        str(args.max_actor_generation_calls),
        "--max-auxiliary-generation-calls",
        str(args.max_auxiliary_generation_calls),
        "--decode-strategy",
        args.decode_strategy,
        "--max-wall-seconds",
        str(args.max_wall_seconds),
        "--device",
        args.device,
        "--dtype",
        args.dtype,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--torch-threads",
        str(args.torch_threads),
    ]
    if args.eval_policy is not None:
        command.extend(("--eval-policy", str(args.eval_policy.resolve())))
    if args.eval_capacity is not None:
        command.extend(("--eval-capacity", str(args.eval_capacity.resolve())))
    if args.prefill_chunk_size is not None:
        command.extend(("--prefill-chunk-size", str(args.prefill_chunk_size)))
    if args.npu_allocator_metrics:
        command.append("--npu-allocator-metrics")
    return command


def _hard_stop_owned_child(process: subprocess.Popen[Any]) -> int:
    if process.poll() is None:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
    return process.wait(timeout=5)


def _source_environment() -> dict[str, str]:
    environment = os.environ.copy()
    root = Path(__file__).resolve().parents[2]
    candidates = [str(root / "python"), str(root)]
    candidates.extend(
        item for item in environment.get("PYTHONPATH", "").split(os.pathsep) if item
    )
    seen: set[str] = set()
    source_paths = []
    for item in candidates:
        key = os.path.normcase(os.path.abspath(item))
        if key not in seen:
            seen.add(key)
            source_paths.append(item)
    environment["PYTHONPATH"] = os.pathsep.join(source_paths)
    return environment


def _supervise(
    args: argparse.Namespace,
    *,
    command: list[str] | None = None,
) -> dict[str, Any]:
    _validate_allocator_device(args)
    started = time.monotonic()
    out = args.out.resolve()
    receipt_path = _supervisor_path(out)
    if out.exists():
        raise FileExistsError(f"output already exists: {out}")
    if receipt_path.exists():
        raise FileExistsError(f"supervisor receipt already exists: {receipt_path}")
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    child_command = list(command if command is not None else _child_command(args))
    receipt: dict[str, Any] = {
        "schema": SUPERVISOR_SCHEMA,
        "status": "prepared",
        "output": str(out),
        "maximum_wall_seconds": args.max_wall_seconds,
        "deadline_owner": "parent_process",
        "owned_child_only": True,
        "child_command": child_command,
        "child_pid": None,
        "child_returncode": None,
        "hard_cutoff": None,
        "wall_seconds": 0.0,
        "wall_seconds_final": False,
    }
    save_json(receipt_path, receipt)
    process = None
    status = "failed"
    caught = None
    try:
        remaining = args.max_wall_seconds - (time.monotonic() - started)
        if remaining <= 0:
            status = "hard_wall_cutoff"
            receipt["hard_cutoff"] = {
                "reason": "maximum_wall_seconds",
                "wall_seconds_at_cutoff": time.monotonic() - started,
                "process_action": "not_started",
            }
        else:
            process = subprocess.Popen(
                child_command,
                cwd=Path(__file__).resolve().parents[2],
                env=_source_environment(),
                start_new_session=os.name == "posix",
                creationflags=(
                    getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
                ),
            )
            receipt.update(status="running", child_pid=process.pid)
            save_json(receipt_path, receipt)
            remaining = max(
                0.0, args.max_wall_seconds - (time.monotonic() - started)
            )
            try:
                returncode = process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                status = "hard_wall_cutoff"
                receipt["hard_cutoff"] = {
                    "reason": "maximum_wall_seconds",
                    "wall_seconds_at_cutoff": time.monotonic() - started,
                    "process_action": (
                        "SIGKILL" if os.name == "posix" else "TerminateProcess"
                    ),
                }
                returncode = _hard_stop_owned_child(process)
            else:
                status = "completed" if returncode == 0 else "failed"
            receipt["child_returncode"] = returncode
    except BaseException as error:
        caught = error
        receipt["error"] = {"type": type(error).__name__, "message": str(error)}
    finally:
        if process is not None and process.poll() is None:
            receipt["child_returncode"] = _hard_stop_owned_child(process)
        if out.exists():
            paths = _artifact_paths(out)
            receipt["actor_journal_summary"] = _journal_summary(paths["actor_journal"])
            receipt["auxiliary_journal_summary"] = _journal_summary(
                paths["auxiliary_journal"]
            )
            receipt["step_inventory"] = _jsonl_inventory(paths["steps"])
            receipt["join_inventory"] = _jsonl_inventory(paths["join"])
        receipt.update(
            status=status,
            wall_seconds=time.monotonic() - started,
            wall_seconds_final=True,
        )
        save_json(receipt_path, receipt)
    if caught is not None:
        raise caught
    return receipt


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    if args.preview:
        print(json.dumps(preview(args), ensure_ascii=False, indent=2, allow_nan=False))
        return
    if args.serve_child:
        _serve(args)
        return
    receipt = _supervise(args)
    if receipt["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()


__all__ = [
    "PROXY_SHARED_CALL_CAP_PER_TASK",
    "SCHEMA",
    "VIEW_MODE",
    "main",
    "parser",
    "preview",
]
