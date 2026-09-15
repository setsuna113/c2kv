"""Focused CPU contracts for native HiAgent official BFCL orchestration."""
from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from benchmarks.memory_runtime import event_native_hiagent_bfcl as wrapper


def _sampling(path: Path) -> tuple[Path, dict]:
    value = {
        "schema": "a-native-hiagent-policy-sampling-v1",
        "temperature": 0.0,
        "seed": 42,
        "max_completion_tokens": 4096,
        "top_p": 1.0,
        "top_k": 0,
        "min_p": 0.0,
    }
    path.write_text(json.dumps(value), encoding="utf-8")
    return path, {key: item for key, item in value.items() if key != "schema"}


def _profile(checkpoint: Path) -> dict:
    return {
        "checkpoint": str(checkpoint.resolve()),
        "training_arm": "B",
        "parameter_version": 500,
        "corpus_identity": "fixture-corpus",
        "model_geometry": {"max_position_embeddings": 40960},
        "packing_contract": {"max_sequence_tokens": 40960},
        "policy_contract": {"history_budget_bytes": 1000},
    }


def _ready(tmp_path: Path, sampling_path: Path, sampling: dict) -> dict:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir(exist_ok=True)
    server_out = tmp_path / "server"
    server_out.mkdir(exist_ok=True)
    join = server_out / "hiagent_join.jsonl"
    steps = server_out / "hiagent_steps.jsonl"
    join.write_text("", encoding="utf-8")
    steps.write_text("", encoding="utf-8")
    return {
        "schema": "a-event-native-hiagent-server-v1",
        "status": "ready",
        "launch": True,
        "run_id": "native-hiagent-fixture",
        "model_name": "native-hiagent-model",
        "benchmark": "bfcl",
        "view_mode": "full_original",
        "base_url": "http://127.0.0.1:36100",
        "allowed_task_ids": [
            "multi_turn_long_context_20",
            "multi_turn_long_context_100",
        ],
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint": _profile(checkpoint),
        "policy_sampling_contract": {
            "source": str(sampling_path.resolve()),
            "sha256": wrapper.sha256_file(sampling_path),
            "sampling": sampling,
        },
        "proxy_shared_call_cap_per_task": 96,
        "proxy_call_cap_owner": "official proxy",
        "max_actor_generation_calls": 192,
        "max_auxiliary_generation_calls": 192,
        "runtime_contract": {
            "model_loads": 1,
            "actor_and_auxiliary_share_weights": True,
            "actor_and_auxiliary_caches": "separate",
            "generation_serialization": "dispatcher lock",
        },
        "artifacts": {"join": str(join.resolve()), "steps": str(steps.resolve())},
    }


def _health(ready: dict, **updates) -> dict:
    value = {
        "schema": "a-event-native-hiagent-health-v1",
        "status": "ok",
        "terminal_failure": None,
        "benchmark": "bfcl",
        "model_name": ready["model_name"],
        "allowed_task_ids": sorted(ready["allowed_task_ids"]),
        "completed_calls": 0,
        "capacity_rejected_tasks": 0,
        "deadline_exceeded": False,
        "actor_generation_calls": 0,
        "auxiliary_generation_calls": 0,
        "join_path": ready["artifacts"]["join"],
        "steps_path": ready["artifacts"]["steps"],
    }
    value.update(updates)
    return value


def _contract(tmp_path: Path, ready: dict, sampling_path: Path, sampling: dict) -> dict:
    official = tmp_path / "official"
    value = {
        "schema": wrapper.RUN_SCHEMA,
        "status": "prepared",
        "base_url": ready["base_url"],
        "server_manifest": ready,
        "policy_sampling": {
            "path": str(sampling_path.resolve()),
            "sha256": wrapper.sha256_file(sampling_path),
            "sampling": sampling,
        },
        "task_ids": ready["allowed_task_ids"],
        "category": "multi_turn_long_context",
        "benchmark_dir": str(tmp_path / "bfcl"),
        "proxy_python": str(Path(sys.executable).resolve()),
        "proxy_port": 36101,
        "official_output": str(official.resolve()),
        "official_validation_path": str((tmp_path / "official.validation.json").resolve()),
        "overlap_admission": None,
        "overlap_audit_path": None,
    }
    value["run_argv"] = wrapper.build_run_argv(value)
    return value


def _write_official(contract: dict, *, correct_count: int = 1) -> Path:
    root = Path(contract["official_output"])
    root.mkdir(parents=True)
    log = root / "logs" / "proxy.jsonl"
    journal = root / "logs" / "proxy.attempts.jsonl"
    log.parent.mkdir()
    log.write_text("", encoding="utf-8")
    journal.write_text("", encoding="utf-8")
    score = root / "score" / "native" / "BFCL_v4_multi_turn_long_context_score.json"
    score.parent.mkdir(parents=True)
    score.write_text(
        json.dumps({"accuracy": correct_count / 2, "correct_count": correct_count, "total_count": 2}) + "\n",
        encoding="utf-8",
    )
    ready = contract["server_manifest"]
    sampling = contract["policy_sampling"]["sampling"]
    summary = {
        "benchmark": "bfcl",
        "arm": wrapper.ARM,
        "backend": wrapper.BACKEND,
        "model": ready["model_name"],
        "categories": "multi_turn_long_context",
        "mode": "both",
        "scored": True,
        "n_total": 2,
        "n_generated": 2,
        "n_scored": 2,
        "correct_count": correct_count,
        "bfcl_num_threads": 1,
        "bfcl_project_root": str(root.resolve()),
        "generation_requested": {
            "temperature": sampling["temperature"],
            "seed": sampling["seed"],
            "max_completion_tokens": sampling["max_completion_tokens"],
        },
        "preflight": {
            "requirements": [
                {"code": "native_hiagent_explicit_contract", "satisfied": True}
            ]
        },
        "checkpoint_profile": copy.deepcopy(ready["checkpoint"]),
        "official_score_headers": [
            {"path": str(score.resolve()), "total_count": 2, "correct_count": correct_count}
        ],
        "request_log": str(log.resolve()),
        "attempt_journal": {"path": str(journal.resolve()), "accounting": {}},
    }
    path = root / f"summary_{wrapper.ARM}.json"
    path.write_text(json.dumps(summary), encoding="utf-8")
    return path


def test_current_hiagent_ready_health_and_sampling_form_one_identity_chain(tmp_path):
    sampling_path, sampling = _sampling(tmp_path / "sampling.json")
    ready = _ready(tmp_path, sampling_path, sampling)
    result = wrapper.validate_server_identity(
        ready,
        _health(ready),
        base_url=ready["base_url"],
        policy_sampling=sampling_path,
    )
    assert result["status"] == "matched"
    assert result["health_directly_attests_checkpoint_or_sampling"] is False
    assert result["policy_sampling_sha256"] == wrapper.sha256_file(sampling_path)


@pytest.mark.parametrize(
    ("target", "field", "value", "match"),
    [
        ("ready", "schema", "a-event-native-server-v1", "ready native HiAgent"),
        ("health", "schema", "a-event-native-api-health-v1", "healthy native HiAgent"),
        ("health", "join_path", "other.jsonl", "join_path"),
        ("health", "completed_calls", 1, "already consumed"),
        ("health", "deadline_exceeded", True, "deadline"),
    ],
)
def test_identity_rejects_old_schema_path_drift_and_consumed_endpoint(
    tmp_path, target, field, value, match
):
    sampling_path, sampling = _sampling(tmp_path / "sampling.json")
    ready = _ready(tmp_path, sampling_path, sampling)
    health = _health(ready)
    (ready if target == "ready" else health)[field] = value
    with pytest.raises(ValueError, match=match):
        wrapper.validate_server_identity(
            ready,
            health,
            base_url="http://127.0.0.1:36100",
            policy_sampling=sampling_path,
        )


def test_sampling_hash_mismatch_rejects_before_official_dispatch(tmp_path):
    sampling_path, sampling = _sampling(tmp_path / "sampling.json")
    ready = _ready(tmp_path, sampling_path, sampling)
    sampling_path.write_text(sampling_path.read_text() + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash differs"):
        wrapper.validate_server_identity(
            ready,
            _health(ready),
            base_url=ready["base_url"],
            policy_sampling=sampling_path,
        )


def test_run_argv_uses_native_bridge_and_shared_96_without_retries(tmp_path):
    sampling_path, sampling = _sampling(tmp_path / "sampling.json")
    ready = _ready(tmp_path, sampling_path, sampling)
    contract = _contract(tmp_path, ready, sampling_path, sampling)
    argv = contract["run_argv"]
    assert argv[argv.index("--arm") + 1] == "hiagent_full_native"
    assert argv[argv.index("--backend") + 1] == "event_native_hiagent"
    assert argv[argv.index("--capability-features") + 1] == "hiagent_trajectory_retrieval_v1"
    assert argv[argv.index("--max-generation-attempts") + 1] == "192"
    assert argv[argv.index("--max-generation-attempts-per-task") + 1] == "96"
    assert argv[argv.index("--max-extraction-attempts") + 1] == "0"
    assert "--no-upstream-retries" in argv
    assert "--capture-request-views" in argv
    assert argv[argv.index("--run-ids") + 1] == ",".join(ready["allowed_task_ids"])


def test_official_validation_accepts_scored_failures_and_hashes_artifacts(tmp_path):
    sampling_path, sampling = _sampling(tmp_path / "sampling.json")
    ready = _ready(tmp_path, sampling_path, sampling)
    contract = _contract(tmp_path, ready, sampling_path, sampling)
    _write_official(contract, correct_count=1)
    result = wrapper.validate_official_artifacts(contract)
    assert result["status"] == "passed"
    assert result["n_scored"] == 2
    assert result["correct_count"] == 1
    assert result["accuracy_all_correct_required"] is False
    assert result["shared_call_limit"] == {
        "per_task": 96,
        "total": 192,
        "owner": "official proxy",
        "automatic_retries": 0,
        "automatic_reruns": 0,
    }
    assert len(result["official_score_artifacts"]) == 1
    assert len(result["trace_artifacts"]) == 2


def test_official_validation_accepts_profile_resolved_from_real_b500_metadata(tmp_path):
    from benchmarks import run

    checkpoint = (
        Path(__file__).resolve().parents[3]
        / "tmp/a_memory_runtime_20260912/native_hiagent_preparation_v1/b500_metadata"
    )
    profile_args = run.build_parser().parse_args([
        "--benchmark", "bfcl", "--arm", wrapper.ARM,
        "--backend", wrapper.BACKEND, "--upstream", "http://127.0.0.1:26000",
        "--out", str(tmp_path / "runner"), "--checkpoint", str(checkpoint),
    ])
    resolved = run.resolve_run_profile(profile_args)
    sampling_path, sampling = _sampling(tmp_path / "sampling.json")
    ready = _ready(tmp_path, sampling_path, sampling)
    ready["checkpoint_path"] = str(checkpoint.resolve())
    ready["checkpoint"] = resolved
    contract = _contract(tmp_path, ready, sampling_path, sampling)
    _write_official(contract, correct_count=1)

    result = wrapper.validate_official_artifacts(contract)

    assert result["status"] == "passed"
    assert resolved["parameter_version"] == 500


def test_worker_recomputes_command_and_writes_official_validation(tmp_path, monkeypatch):
    sampling_path, sampling = _sampling(tmp_path / "sampling.json")
    ready = _ready(tmp_path, sampling_path, sampling)
    contract = _contract(tmp_path, ready, sampling_path, sampling)
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    seen = []

    def run_benchmarks(argv):
        seen.append(list(argv))
        _write_official(contract, correct_count=0)

    monkeypatch.setattr(wrapper, "run_benchmarks", run_benchmarks)
    wrapper.worker(contract_path)
    assert seen == [contract["run_argv"]]
    validation = json.loads(Path(contract["official_validation_path"]).read_text())
    assert validation["status"] == "passed"
    assert validation["correct_count"] == 0


def test_identity_failure_precedes_output_and_child_spawn(tmp_path, monkeypatch):
    sampling_path, sampling = _sampling(tmp_path / "sampling.json")
    ready = _ready(tmp_path, sampling_path, sampling)
    ready_path = tmp_path / "ready.json"
    ready_path.write_text(json.dumps(ready), encoding="utf-8")
    benchmark = tmp_path / "bfcl"
    (benchmark / "bfcl_eval").mkdir(parents=True)
    health = _health(ready, steps_path=str(tmp_path / "wrong.jsonl"))
    monkeypatch.setattr(wrapper, "read_health", lambda base_url: health)
    monkeypatch.setattr(
        wrapper.subprocess,
        "Popen",
        lambda *a, **k: pytest.fail("identity failure must precede spawn"),
    )
    out = tmp_path / "run"
    with pytest.raises(ValueError, match="steps_path"):
        wrapper.main(
            [
                "--server-manifest",
                str(ready_path),
                "--base-url",
                ready["base_url"],
                "--policy-sampling",
                str(sampling_path),
                "--benchmark-dir",
                str(benchmark),
                "--bfcl-python",
                sys.executable,
                "--proxy-python",
                sys.executable,
                "--proxy-port",
                "36101",
                "--out",
                str(out),
                "--max-wall-seconds",
                "10",
            ]
        )
    assert not out.exists()


def test_wall_cap_stops_owned_worker_and_persists_terminal_receipt(tmp_path, monkeypatch):
    sampling_path, sampling = _sampling(tmp_path / "sampling.json")
    ready = _ready(tmp_path, sampling_path, sampling)
    ready_path = tmp_path / "ready.json"
    ready_path.write_text(json.dumps(ready), encoding="utf-8")
    benchmark = tmp_path / "bfcl"
    (benchmark / "bfcl_eval").mkdir(parents=True)
    lexical_python = tmp_path / "bench" / "bin" / "python"
    lexical_python.parent.mkdir(parents=True)
    lexical_python.write_text("fixture", encoding="utf-8")
    lexical_proxy_python = tmp_path / "sgl" / "bin" / "python"
    lexical_proxy_python.parent.mkdir(parents=True)
    lexical_proxy_python.write_text("fixture", encoding="utf-8")
    resolved_base_python = tmp_path / "c2kv" / "bin" / "python3.11"
    real_resolve = Path.resolve

    def resolve_venv_link(path, *args, **kwargs):
        if path in (lexical_python, lexical_proxy_python):
            return resolved_base_python
        return real_resolve(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", resolve_venv_link)
    health_responses = iter((_health(ready), _health(ready)))
    monkeypatch.setattr(wrapper, "read_health", lambda base_url: next(health_responses))
    clock = iter((100.0, 100.5, 101.0, 105.0))
    monkeypatch.setattr(wrapper.time, "monotonic", lambda: next(clock))
    processes = []

    class Process:
        pid = 4321

        def __init__(self, command, **kwargs):
            self.command = command
            self.kwargs = kwargs
            self.returncode = None
            self.wait_calls = []
            self.terminated = False
            processes.append(self)

        def poll(self):
            return self.returncode

        def wait(self, timeout):
            self.wait_calls.append(timeout)
            if len(self.wait_calls) == 1:
                raise subprocess.TimeoutExpired(self.command, timeout)
            self.returncode = -15
            return self.returncode

        def terminate(self):
            self.terminated = True

        def kill(self):
            self.returncode = -9

    monkeypatch.setattr(wrapper.subprocess, "Popen", Process)
    out = tmp_path / "run"
    with pytest.raises(SystemExit) as stopped:
        wrapper.main(
            [
                "--server-manifest",
                str(ready_path),
                "--base-url",
                ready["base_url"],
                "--policy-sampling",
                str(sampling_path),
                "--benchmark-dir",
                str(benchmark),
                "--bfcl-python",
                str(lexical_python),
                "--proxy-python",
                str(lexical_proxy_python),
                "--proxy-port",
                "36101",
                "--out",
                str(out),
                "--max-wall-seconds",
                "2",
            ]
        )
    assert stopped.value.code == 1
    assert len(processes) == 1
    assert processes[0].command[0] == str(lexical_python.absolute())
    running = json.loads((out / "running.json").read_text())
    assert running["bfcl_python"] == str(lexical_python.absolute())
    assert running["proxy_python"] == str(lexical_proxy_python.absolute())
    proxy_python_index = running["run_argv"].index("--proxy-python") + 1
    assert running["run_argv"][proxy_python_index] == str(
        lexical_proxy_python.absolute()
    )
    assert processes[0].terminated is True
    assert processes[0].wait_calls == [pytest.approx(1.0), 5]
    final = json.loads((out / "final.json").read_text())
    assert final["status"] == "wall_cap_reached"
    assert final["worker_returncode"] == -15
    assert final["wall_seconds_final"] is True
    assert final["automatic_retries"] == 0
    assert final["automatic_reruns"] == 0


def test_existing_output_is_zero_rerun_gate_before_health(tmp_path, monkeypatch):
    out = tmp_path / "existing"
    out.mkdir()
    monkeypatch.setattr(
        wrapper,
        "read_health",
        lambda base_url: pytest.fail("existing output must fail before health"),
    )
    with pytest.raises(FileExistsError):
        wrapper.main(
            [
                "--server-manifest",
                str(tmp_path / "ready.json"),
                "--base-url",
                "http://127.0.0.1:36100",
                "--policy-sampling",
                str(tmp_path / "sampling.json"),
                "--benchmark-dir",
                str(tmp_path),
                "--bfcl-python",
                sys.executable,
                "--proxy-python",
                sys.executable,
                "--proxy-port",
                "36101",
                "--out",
                str(out),
                "--max-wall-seconds",
                "10",
            ]
        )
