from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from experiments.history_system.peers import dispatch


def _preview_receipt(method: str, device: int, *, blocked: bool = False):
    port, proxy = dispatch.DEFAULT_PORTS[method]
    return {
        "method": method,
        "candidate_id": f"peer_{method}_base10",
        "physical_device": device,
        "port_base": port,
        "proxy_port_base": proxy,
        "state": "blocked" if blocked else "ready",
        "status": "blocked_not_launched" if blocked else "ready_to_launch_no_model_started",
        "blockers": [{"kind": "active_candidate_on_target_lane"}] if blocked else [],
        "model_requests": 0,
        "scorer_calls": 0,
        "network_calls": 0,
        "launches": 0,
    }


def _launch_receipt(method: str, device: int):
    value = _preview_receipt(method, device)
    value.update(
        state="running",
        status="launched_not_completed",
        pid=12345,
        stage_wall_cap_seconds=10800,
        launches=1,
    )
    return value


def test_all_methods_bind_remote_cpu_freezes_and_generate_exact_remote_contract():
    for method in dispatch.METHODS:
        binding = dispatch._load_binding(method)
        port, proxy = dispatch._ports(method, None, None)
        program = dispatch._remote_program(
            binding,
            action="preview",
            device=4,
            port_base=port,
            proxy_port_base=proxy,
        )
        assert binding["receipt"]["whole_task_denominator"] == 10
        assert binding["receipt"]["launches"] == 0
        assert "def device_processes" in program
        assert dispatch.LAUNCHER in program
        assert dispatch.OVERLAY in program
        assert dispatch.B500 in program
        assert repr(dispatch.ACTIVE_CANDIDATES) in program
        assert repr(dispatch.LEGACY_PROCESSES) in program
        assert "automatic_reruns" in program
        assert "def _resolve_policy_sampling" in program
        assert "subprocess.Popen" in program
        assert "for attempt in" not in program
        assert program.count("subprocess.Popen") == 1
        compile(program, f"<peer-{method}-preview>", "exec")
        compile(
            dispatch._observe_program(method, {"pid": 1, "physical_device": 4}),
            f"<peer-{method}-observe>",
            "exec",
        )


def test_hiagent_policy_sampling_falls_back_to_hash_bound_runner_sibling(
    tmp_path: Path,
):
    submitted = tmp_path / "submitted"
    runner = submitted / "tmp/current/native_hiagent_long20_runner.py"
    policy = runner.parent / "policy_sampling.greedy0.json"
    runner.parent.mkdir(parents=True)
    runner.write_text("# bound runner\n", encoding="utf-8")
    policy.write_text('{"temperature": 0, "seed": 0}\n', encoding="utf-8")
    expected = hashlib.sha256(policy.read_bytes()).hexdigest()

    resolved = dispatch._resolve_policy_sampling(
        submitted,
        runner,
        "tmp/stale/policy_sampling.greedy0.json",
        expected,
    )

    assert resolved == policy.resolve()


def test_preview_saves_parameters_without_launch(tmp_path: Path, monkeypatch):
    out = tmp_path / "peer_raw_base10"
    monkeypatch.setattr(dispatch, "_local_output", lambda method: out)
    calls = []

    def fake_execute(code, *, timeout):
        calls.append((code, timeout))
        return _preview_receipt("raw", 4)

    result = dispatch.preview("raw", 4, execute_fn=fake_execute)
    saved = json.loads((out / "preview.latest.json").read_text(encoding="utf-8"))
    assert result == saved
    assert result["launches"] == 0
    assert len(calls) == 1


def test_blocked_launch_records_blocker_without_launch_receipt(
    tmp_path: Path, monkeypatch
):
    out = tmp_path / "peer_text_base10"
    monkeypatch.setattr(dispatch, "_local_output", lambda method: out)

    def fake_execute(code, *, timeout):
        return _preview_receipt("text", 7, blocked=True)

    result = dispatch.launch("text", 7, execute_fn=fake_execute)
    assert result["status"] == "blocked_not_launched"
    assert (out / "launch.blocked.latest.json").is_file()
    assert not (out / "launch.json").exists()


def test_launch_and_observe_write_only_peer_receipts(tmp_path: Path, monkeypatch):
    out = tmp_path / "peer_full_base10"
    monkeypatch.setattr(dispatch, "_local_output", lambda method: out)

    def fake_launch(code, *, timeout):
        return _launch_receipt("full", 4)

    launched = dispatch.launch("full", 4, execute_fn=fake_launch)
    assert json.loads((out / "launch.json").read_text()) == launched

    observation = {
        "schema": "a-history-system-peer-observation-v1",
        "method": "full",
        "candidate_id": "peer_full_base10",
        "fixed_task_denominator": 10,
        "pid": 12345,
        "pid_alive": True,
        "physical_device": 4,
        "stage": None,
        "cells": [],
    }

    def fake_observe(code, *, timeout):
        assert "EXPECTED_LAUNCH" in code
        return observation

    assert dispatch.observe("full", execute_fn=fake_observe) == observation
    assert json.loads((out / "observation.latest.json").read_text()) == observation


def test_dispatch_rejects_unfrozen_methods_devices_ports_and_relaunch(tmp_path: Path, monkeypatch):
    with pytest.raises(ValueError, match="method"):
        dispatch.preview("c0", 4, execute_fn=lambda *_args, **_kwargs: {})
    with pytest.raises(ValueError, match="4 or 7"):
        dispatch.preview("raw", 5, execute_fn=lambda *_args, **_kwargs: {})
    with pytest.raises(ValueError, match="disjoint"):
        dispatch.preview(
            "hiagent",
            4,
            port_base=30000,
            proxy_port_base=30005,
            execute_fn=lambda *_args, **_kwargs: {},
        )
    with pytest.raises(ValueError, match="only to HiAgent"):
        dispatch.preview(
            "raw",
            4,
            proxy_port_base=30020,
            execute_fn=lambda *_args, **_kwargs: {},
        )

    out = tmp_path / "peer_raw_base10"
    monkeypatch.setattr(dispatch, "_local_output", lambda method: out)
    out.mkdir(parents=True)
    (out / "launch.json").write_text("{}")
    with pytest.raises(FileExistsError, match="already"):
        dispatch.launch("raw", 4, execute_fn=lambda *_args, **_kwargs: {})


def test_observe_requires_local_launch_receipt(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        dispatch, "_local_output", lambda method: tmp_path / f"peer_{method}_base10"
    )
    with pytest.raises(FileNotFoundError, match="no launch"):
        dispatch.observe("hiagent", execute_fn=lambda *_args, **_kwargs: {})


def test_official_verification_requires_nonempty_score_headers(tmp_path: Path):
    task_root = tmp_path / "task"
    summary = task_root / "bfcl/official_summary.json"
    summary.parent.mkdir(parents=True)
    summary.write_text(
        json.dumps(
            {
                "scored": True,
                "n_scored": 1,
                "correct_count": 1,
                "official_score_headers": [],
            }
        ),
        encoding="utf-8",
    )

    result = dispatch._verify_official_summary(task_root, summary)
    assert result["official_verified"] is False
    assert result["correct_count"] is None
    assert result["official_verification_reason"] == "missing_official_score_headers"


def test_hiagent_sibling_official_score_is_resolved_from_task_root(tmp_path: Path):
    task_root = tmp_path / "task"
    summary = task_root / "bfcl/official/summary_hiagent_full_native.json"
    score = task_root / "bfcl/score/hiagent/multi_turn/BFCL_score.json"
    summary.parent.mkdir(parents=True)
    score.parent.mkdir(parents=True)
    summary.write_text(
        json.dumps(
            {
                "scored": True,
                "n_scored": 1,
                "correct_count": 1,
                "official_score_headers": [
                    {
                        "path": "/remote/sibling/BFCL_score.json",
                        "correct_count": 1,
                        "total_count": 1,
                        "accuracy": 1.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    score.write_text(
        json.dumps({"correct_count": 1, "total_count": 1, "accuracy": 1.0})
        + "\n",
        encoding="utf-8",
    )

    result = dispatch._verify_official_summary(task_root, summary)
    assert result["official_verified"] is True
    assert result["correct_count"] == 1
    assert len(result["official_score_sha256"]) == 1
    assert result["official_verification_reason"] is None
