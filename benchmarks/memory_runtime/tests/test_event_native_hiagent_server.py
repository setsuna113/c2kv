"""Focused offline contracts for the standalone native HiAgent server CLI."""
from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from benchmarks.memory_runtime import event_native_hiagent_server as server
from benchmarks.memory_runtime.attempt_journal import AttemptJournal
from benchmarks.memory_runtime.tests.test_event_native_policy import contracts


def _profile(checkpoint: Path) -> dict:
    packing, policy = contracts()
    packing.update(
        max_target_tokens=128,
        max_workspace_tokens=3000,
        max_sequence_tokens=4096,
    )
    return {
        "checkpoint": str(checkpoint.resolve()),
        "profile": {},
        "training_arm": "B",
        "parameter_version": 500,
        "corpus_identity": "fixture-corpus",
        "synthetic_cpu_smoke": False,
        "synthetic_inference_fixture": False,
        "training_completed": True,
        "declared_supported_ratios": [4, 8],
        "packing_contract": packing,
        "policy_contract": policy,
        "model_geometry": {
            "num_hidden_layers": 1,
            "num_key_value_heads": 1,
            "num_attention_heads": 1,
            "head_dim": 8,
            "hidden_size": 8,
            "max_position_embeddings": 4096,
        },
        "scope": "synthetic CLI fixture",
    }


def _sampling(path: Path, *, temperature: float = 0.0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema": "a-native-hiagent-policy-sampling-v1",
                "temperature": temperature,
                "seed": 42,
                "max_completion_tokens": 96,
                "top_p": 1.0,
                "top_k": 0,
                "min_p": 0.0,
            }
        ),
        encoding="utf-8",
    )
    return path


def _args(tmp_path: Path, *, temperature: float = 0.0):
    checkpoint = tmp_path / "checkpoint"
    sampling = _sampling(tmp_path / "sampling.json", temperature=temperature)
    return server.parser().parse_args(
        [
            "--checkpoint",
            str(checkpoint),
            "--out",
            str(tmp_path / "out"),
            "--run-id",
            "hiagent-fixture",
            "--model-name",
            "native-fixture",
            "--benchmark",
            "bfcl",
            "--ratio",
            "4",
            "--policy-sampling",
            str(sampling),
            "--task-ids",
            "task-1,task-2",
            "--max-actor-generation-calls",
            "192",
            "--max-auxiliary-generation-calls",
            "192",
            "--max-wall-seconds",
            "30",
        ]
    )


def test_preview_is_read_only_and_sampling_is_explicit(tmp_path):
    args = _args(tmp_path)
    result = server.preview(args, inspect_checkpoint_fn=lambda path: _profile(Path(path)))

    assert result["status"] == "preview"
    assert result["launch"] is False
    assert result["output_created"] is False
    assert result["view_mode"] == "full_original"
    assert result["policy_sampling_contract"]["sampling"]["temperature"] == 0.0
    assert result["proxy_shared_call_cap_per_task"] == 96
    assert result["proxy_call_cap_owner"] == "official proxy"
    assert result["runtime_contract"] == {
        "model_loads": 1,
        "actor_and_auxiliary_share_weights": True,
        "actor_and_auxiliary_caches": "separate",
        "generation_serialization": "dispatcher lock",
    }
    assert not args.out.exists()

    unsupported = _args(tmp_path / "unsupported", temperature=0.001)
    with pytest.raises(ValueError, match="greedy temperature=0"):
        server.preview(
            unsupported,
            inspect_checkpoint_fn=lambda path: _profile(Path(path)),
        )
    assert not unsupported.out.exists()


def test_policy_sampling_argument_is_required():
    with pytest.raises(SystemExit):
        server.parser().parse_args(
            [
                "--checkpoint",
                "checkpoint",
                "--out",
                "out",
                "--run-id",
                "run",
                "--ratio",
                "4",
                "--task-ids",
                "task-1",
                "--max-actor-generation-calls",
                "1",
                "--max-auxiliary-generation-calls",
                "1",
                "--max-wall-seconds",
                "30",
            ]
        )


class _Generator:
    session_cache_policy = "fixture-independent-last-view-v1"

    def __init__(self, runtime):
        self.runtime = runtime
        self.closed = 0

    def kv_bytes_per_token(self):
        return 64

    def session_cache_info(self):
        return {"policy": self.session_cache_policy, "closed": self.closed}

    def close_session(self):
        self.closed += 1


class _Dispatcher:
    def __init__(self):
        self.terminal = False

    def health(self):
        return {
            "status": "terminal" if self.terminal else "ok",
            "terminal_failure": (
                {"code": "fixture_complete"} if self.terminal else None
            ),
            "actor_generation_calls": 0,
            "auxiliary_generation_calls": 0,
            "completed_calls": 0,
        }


class _Server:
    def __init__(self, dispatcher, host, port):
        self.dispatcher = dispatcher
        self.server_address = (host, port or 43123)
        self.timeout = None
        self.closed = False

    def handle_request(self):
        self.dispatcher.terminal = True

    def server_close(self):
        self.closed = True


def test_injected_cpu_serve_loads_once_and_builds_shared_runtime_generators(tmp_path):
    args = _args(tmp_path)
    profile = _profile(args.checkpoint)
    runtime = object()
    actor = _Generator(runtime)
    calls = {"load": 0}
    captured = {}

    def load(checkpoint, **kwargs):
        calls["load"] += 1
        captured["loader"] = {"checkpoint": checkpoint, **kwargs}
        return actor, {**copy.deepcopy(profile), "inference_decode_strategy": "incremental"}

    def auxiliary(shared_runtime, **kwargs):
        captured["auxiliary"] = {"runtime": shared_runtime, **kwargs}
        return _Generator(shared_runtime)

    def build(tokenizer, **kwargs):
        captured["dispatcher"] = kwargs
        assert isinstance(kwargs["actor_journal"], AttemptJournal)
        assert isinstance(kwargs["auxiliary_journal"], AttemptJournal)
        assert kwargs["actor_journal"].path != kwargs["auxiliary_journal"].path
        assert kwargs["actor_generator"] is actor
        assert kwargs["auxiliary_generator"].runtime is actor.runtime
        return _Dispatcher()

    built_server = None

    def make(dispatcher, *, host, port):
        nonlocal built_server
        built_server = _Server(dispatcher, host, port)
        return built_server

    server._serve(
        args,
        inspect_checkpoint_fn=lambda _path: copy.deepcopy(profile),
        tokenizer_loader=lambda checkpoint: {"checkpoint": checkpoint},
        generator_loader=load,
        auxiliary_generator_factory=auxiliary,
        dispatcher_builder=build,
        server_builder=make,
    )

    assert calls["load"] == 1
    assert captured["loader"]["decode_strategy"] == "incremental"
    assert "max_extraction_calls" not in captured["loader"]
    assert captured["auxiliary"] == {
        "runtime": runtime,
        "decode_strategy": "incremental",
        "prefill_chunk_size": None,
    }
    assert set(captured["dispatcher"]["actor_phase_sampling"]) == {
        "policy",
        "trajectory_retrieval_policy",
    }
    assert built_server is not None and built_server.closed
    assert built_server.timeout == 0.25
    assert actor.closed == 1

    ready = json.loads((args.out / "ready.json").read_text(encoding="utf-8"))
    final = json.loads((args.out / "final.json").read_text(encoding="utf-8"))
    assert ready["base_url"] == "http://127.0.0.1:43123"
    assert final["status"] == "stopped"
    assert final["stop_reason"] == "terminal_failure"
    assert final["actor_journal_summary"]["started"] == 0
    assert final["actor_journal_summary"]["initialized_empty"] is True
    assert final["auxiliary_journal_summary"]["started"] == 0
    assert final["auxiliary_journal_summary"]["initialized_empty"] is True
    assert final["step_inventory"]["complete_records"] == 0
    assert final["join_inventory"]["complete_records"] == 0
    assert all(
        (args.out / name).exists()
        for name in (
            "actor_attempts.jsonl",
            "auxiliary_attempts.jsonl",
            "hiagent_steps.jsonl",
            "hiagent_join.jsonl",
        )
    )
    assert final["wall_seconds_final"] is True


def test_failure_before_ready_keeps_missing_attempt_counts_unknown(tmp_path):
    args = _args(tmp_path)
    profile = _profile(args.checkpoint)

    def fail_load(_checkpoint, **_kwargs):
        raise RuntimeError("fixture load failure")

    with pytest.raises(RuntimeError, match="fixture load failure"):
        server._serve(
            args,
            inspect_checkpoint_fn=lambda _path: copy.deepcopy(profile),
            tokenizer_loader=lambda checkpoint: {"checkpoint": checkpoint},
            generator_loader=fail_load,
        )

    final = json.loads((args.out / "final.json").read_text(encoding="utf-8"))
    assert final["status"] == "failed"
    assert final["actor_journal_summary"]["exists"] is False
    assert final["actor_journal_summary"]["started"] is None
    assert final["auxiliary_journal_summary"]["started"] is None
    assert final["step_inventory"]["complete_records"] is None
    assert final["join_inventory"]["complete_records"] is None
    assert final["allocator_by_phase"]["by_phase"]["policy"][
        "generation_attempts"
    ] is None


def test_child_command_preserves_sampling_capacity_and_measurement_flags(tmp_path):
    args = _args(tmp_path)
    args.eval_capacity = tmp_path / "capacity.json"
    args.prefill_chunk_size = 256
    args.device = "npu:0"
    args.npu_allocator_metrics = True
    command = server._child_command(args)
    child = server.parser().parse_args(command[3:])

    assert child.serve_child
    assert child.policy_sampling == args.policy_sampling.resolve()
    assert child.eval_capacity == args.eval_capacity.resolve()
    assert child.prefill_chunk_size == 256
    assert child.max_actor_generation_calls == 192
    assert child.max_auxiliary_generation_calls == 192
    assert child.npu_allocator_metrics


def test_allocator_inventory_stays_partitioned_by_phase(tmp_path):
    path = tmp_path / "hiagent_steps.jsonl"
    rows = [
        {
            "phase": "compressor",
            "runner_record": {
                "generation_trace": [
                    {
                        "generation": {
                            "stats": {
                                "allocator_measurement": {
                                    "status": "ok",
                                    "peak_allocated_bytes": 100,
                                }
                            }
                        }
                    }
                ]
            },
        },
        {
            "phase": "policy",
            "runner_record": {
                "generation_trace": [
                    {
                        "generation": {
                            "stats": {
                                "allocator_measurement": {
                                    "status": "ok",
                                    "peak_allocated_bytes": 220,
                                }
                            }
                        }
                    }
                ]
            },
        },
        {
            "phase": "policy",
            "runner_record": {
                "generation_trace": [{"generation": {"stats": {}}}]
            },
        },
    ]
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    result = server._allocator_by_phase(path, enabled=True)

    compressor = result["by_phase"]["compressor"]
    policy = result["by_phase"]["policy"]
    assert compressor["generator_role"] == "auxiliary"
    assert compressor["strict_peak_allocated_bytes"] == 100
    assert policy["generator_role"] == "actor"
    assert policy["known_peak_allocated_bytes"] == 220
    assert policy["strict_peak_allocated_bytes"] is None
    assert policy["measurement_status"] == {"ok": 1, "failed": 0, "missing": 1}


def test_module_help_does_not_require_model_runtime(tmp_path):
    root = Path(server.__file__).resolve().parents[2]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join((str(root / "python"), str(root)))
    result = subprocess.run(
        [sys.executable, "-m", "benchmarks.memory_runtime.event_native_hiagent_server", "--help"],
        cwd=root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert "--policy-sampling POLICY_SAMPLING" in result.stdout
    assert "--serve-child" not in result.stdout
    assert not (tmp_path / "out").exists()
