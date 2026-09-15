"""Focused plan/execution-state tests for the generic benchmark matrix."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import matrix


def _spec() -> dict:
    return {
        "schema_version": 1,
        "profile": {
            "path": "/profile.json", "profile_fingerprint": "profile-a",
            "checkpoint": {"path": "/checkpoint-588"},
            "serving": {"query_projection": "gist"},
        },
        "defaults": {
            "backend": "sglang", "upstream": "http://127.0.0.1:35000",
            "runner_python": sys.executable,
        },
        "arms": ["c2kv"],
        "benchmarks": {"tau2": {"run_args": ["--task-set", "airline"]}},
    }


def _env(tmp_path: Path) -> dict[str, str]:
    tau2 = tmp_path / "tau2"
    tau2.mkdir()
    return {"HOME": str(tmp_path), "TAU2_DIR": str(tau2)}


def test_plan_is_explicit_and_profile_bound(tmp_path):
    plan = matrix.build_plan(_spec(), tmp_path / "out", environ=_env(tmp_path))
    assert plan["profile"]["profile_fingerprint"] == "profile-a"
    assert len(plan["cells"]) == 1
    cell = plan["cells"][0]
    assert cell["benchmark"] == "tau2" and cell["arm"] == "c2kv"
    assert cell["preflight"]["ok"]
    assert "--task-set" in cell["command"]
    assert "--exact-out" in cell["command"]
    assert cell["command"].count("--checkpoint") == 1
    assert cell["command"].count("--checkpoint-profile") == 1
    assert Path(cell["status_path"]).parent.name == "status"


def test_toolsandbox_option_binds_preflight_and_execution_to_the_same_checkout(tmp_path):
    root = tmp_path / "selected-ts"
    roles = root / "tool_sandbox" / "roles"
    roles.mkdir(parents=True)
    for role, marker in (("agent", "OPENAI_BASE_URL"), ("user", "TOOLSANDBOX_USER_BASE_URL")):
        (roles / f"openai_api_{role}.py").write_text(
            marker + "\nif not openai_response_message.tool_calls:\n")
    spec = _spec()
    spec["benchmarks"] = {"toolsandbox": {"options": {"toolsandbox_dir": str(root)}}}
    cell = matrix.build_plan(spec, tmp_path / "out")["cells"][0]
    assert cell["preflight"]["ok"]
    assert cell["command"][cell["command"].index("--toolsandbox-dir") + 1] == str(root)


def test_execute_requires_summary_and_resume_matches_exact_fingerprint(tmp_path):
    plan = matrix.build_plan(_spec(), tmp_path / "out", environ=_env(tmp_path))
    cell = plan["cells"][0]
    calls = []

    def fake_runner(command, **kwargs):
        calls.append(command)
        summary = Path(cell["summary_path"])
        summary.parent.mkdir(parents=True, exist_ok=True)
        summary.write_text(json.dumps({"n": 1, "semantic_score": 0.0,
                                       "request_log_summary": {"n_ok": 1, "n_error": 0}}),
                           encoding="utf-8")
        kwargs["stdout"].write("runner invoked\n")
        return subprocess.CompletedProcess(command, 0)

    assert matrix.execute_plan(plan, runner=fake_runner) == 0
    assert len(calls) == 1
    assert matrix.execute_plan(plan, resume=True, runner=fake_runner) == 0
    assert len(calls) == 1


def test_summary_requires_clean_requests_scoring_coverage_and_exercised_text_arm(tmp_path):
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({
        "n": 1, "n_scored": 0,
        "request_log_summary": {"n_ok": 1, "n_error": 0},
    }), encoding="utf-8")
    valid, reason = matrix._summary_facts(summary)
    assert not valid and "n_scored" in reason

    summary.write_text(json.dumps({
        "n": 1, "request_log_summary": {"n_ok": 1, "n_error": 1},
    }), encoding="utf-8")
    valid, reason = matrix._summary_facts(summary)
    assert not valid and "n_error" in reason

    summary.write_text(json.dumps({
        "n": 1, "request_log_summary": {"n_ok": 1, "n_error": 0},
        "textarm_summary": {
            "textarm_requests": 1, "compressor_calls": 0, "retrieval_calls": 0,
        },
    }), encoding="utf-8")
    valid, facts = matrix._summary_facts(summary)
    assert valid and facts["method_exercised"] is False
    assert "full-context plumbing" in facts["method_warning"]

    plan = matrix.build_plan(_spec(), tmp_path / "matrix", environ=_env(tmp_path))
    cell = plan["cells"][0]

    def no_trigger_runner(command, **kwargs):
        Path(cell["summary_path"]).parent.mkdir(parents=True, exist_ok=True)
        Path(cell["summary_path"]).write_text(json.dumps({
            "n": 1, "request_log_summary": {"n_ok": 1, "n_error": 0},
            "textarm_summary": {
                "textarm_requests": 1, "compressor_calls": 0, "retrieval_calls": 0,
            },
        }), encoding="utf-8")
        return subprocess.CompletedProcess(command, 0)

    assert matrix.execute_plan(plan, runner=no_trigger_runner) == 3
    status = json.loads(Path(cell["status_path"]).read_text(encoding="utf-8"))
    assert status["state"] == "unexercised"
    assert status["method_exercised"] is False


def test_profile_mismatch_preserves_prior_passed_status(tmp_path):
    env = _env(tmp_path)
    plan = matrix.build_plan(_spec(), tmp_path / "out", environ=env)
    cell = plan["cells"][0]
    status = Path(cell["status_path"])
    status.parent.mkdir(parents=True, exist_ok=True)
    status.write_text(json.dumps({"state": "passed", "cell_fingerprint": "old"}),
                      encoding="utf-8")
    assert matrix.execute_plan(plan, resume=True) == 2
    preserved = json.loads(status.read_text(encoding="utf-8"))
    assert preserved == {"state": "passed", "cell_fingerprint": "old"}
    conflicts = list(status.parent.glob("*.conflict.json"))
    assert len(conflicts) == 1
