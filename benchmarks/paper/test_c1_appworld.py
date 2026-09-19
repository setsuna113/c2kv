import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from benchmarks.paper import c1 as driver
from benchmarks.paper import c1_appworld as bridge


ROOT = Path(__file__).resolve().parents[2]
DELIVERY = ROOT / "experiments" / "history_system"


def _config(tmp_path):
    return {
        "checkpoint": str(tmp_path / "checkpoint-1000"),
        "bench_python": sys.executable,
        "appworld_python": sys.executable,
        "acon_dir": str(tmp_path / "acon"),
        "appworld_root": str(tmp_path / "appworld"),
        "server_port": 34007,
        "proxy_port": 34107,
        "appworld_split": "test_normal",
        "appworld_max_iter": 50,
        "upstream": "http://127.0.0.1:34007",
        "c1": {
            "detector": "t02_risk",
            "selector_threshold": 0.5,
            "history_variant": "H0",
            "recovery_rounds": 1,
            "task_timeout": 120,
        },
    }


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_task_ids_uses_official_appworld_loader_and_configured_root(tmp_path):
    config = _config(tmp_path)
    completed = SimpleNamespace(stdout=json.dumps(["task-a", "task-b"]))
    with mock.patch.object(bridge, "run_owned", return_value=completed) as run:
        assert bridge.task_ids(config) == ("task-a", "task-b")
    command = run.call_args.args[0]
    kwargs = run.call_args.kwargs
    assert command[:2] == [sys.executable, "-c"]
    assert command[-1] == "test_normal"
    assert kwargs["cwd"] == tmp_path / "acon" / "experiments" / "appworld"
    assert kwargs["env"]["APPWORLD_ROOT"] == str((tmp_path / "appworld").resolve())


def test_prepare_appworld_run_uses_configured_data_root_and_restores_env(tmp_path):
    config = _config(tmp_path)
    dataset = tmp_path / "appworld" / "data" / "datasets" / "test_normal.txt"
    dataset.parent.mkdir(parents=True)
    dataset.write_text("task-a\n", encoding="utf-8")
    observed = {}

    def prepare(acon_dir, out_dir, split, task_ids):
        observed.update(
            acon_dir=acon_dir, out_dir=out_dir, split=split, task_ids=task_ids,
            appworld_root=os.environ.get("APPWORLD_ROOT"),
        )
        return tmp_path / "prepared"

    with mock.patch.dict(os.environ, {"APPWORLD_ROOT": "sentinel"}):
        with mock.patch.object(bridge.acon, "prepare_appworld_run", side_effect=prepare):
            result = bridge._prepare_appworld_run(
                config, tmp_path / "official", "test_normal", "task-a",
            )
        assert os.environ["APPWORLD_ROOT"] == "sentinel"
    assert result == tmp_path / "prepared"
    assert observed == {
        "acon_dir": (tmp_path / "acon").resolve(),
        "out_dir": tmp_path / "official",
        "split": "test_normal",
        "task_ids": ["task-a"],
        "appworld_root": str((tmp_path / "appworld").resolve()),
    }


def test_native_ready_url_is_normalized_once_for_acon():
    assert bridge._openai_origin("http://127.0.0.1:34401/v1") == "http://127.0.0.1:34401"
    assert bridge._openai_origin("http://localhost:34401/") == "http://localhost:34401"
    env = bridge.acon.runner_env(bridge._openai_origin("http://127.0.0.1:34401/v1"))
    assert env[bridge.acon.BASE_URL_ENV] == "http://127.0.0.1:34401/v1"
    with pytest.raises(ValueError, match="loopback origin"):
        bridge._openai_origin("http://127.0.0.1:34401/v1/v1")


@pytest.mark.parametrize("detector", ["t02_risk", "d3_hybrid"])
def test_controller_command_binds_one_task_to_native_appworld_endpoint(tmp_path, detector):
    config = _config(tmp_path)
    config["c1"]["detector"] = detector
    controller = tmp_path / "controller.json"
    controller.write_text(json.dumps({"post_draft_recovery": {}}), encoding="utf-8")
    command = bridge.controller_command(
        config, "task-a", tmp_path / "native", DELIVERY, controller,
    )
    assert command[command.index("--benchmark") + 1] == "acon_appworld"
    assert command[command.index("--source-profile") + 1] == "openai-single-task-v1"
    assert command[command.index("--task-ids") + 1] == "task-a"
    assert command[command.index("--model-name") + 1] == f"c1_{detector}"
    assert command[command.index("--port") + 1] == "34107"
    assert command[command.index("--sglang-backend-url") + 1] == config["upstream"]
    assert command[command.index("--out") + 1] == str(
        (tmp_path / "native" / "task_shards" / "task-a" / "server").resolve()
    )
    assert "--tool-memory" not in command


def test_tool_on_controller_command_carries_separate_tool_checkpoint(tmp_path):
    config = _config(tmp_path)
    config["tool_memory"] = "t0:r8"
    config["tool_checkpoint"] = str(tmp_path / "T0" / "checkpoint-500")
    config["tool_budget_tokens"] = 512
    (tmp_path / "controller.json").write_text("{}", encoding="utf-8")
    command = bridge.controller_command(
        config, "task-a", tmp_path / "native", DELIVERY,
        tmp_path / "controller.json")
    assert command[command.index("--view-mode") + 1] == (
        "ac_native_s0_lexical_raw_reserve_failed_operation")
    assert command[command.index("--tool-memory") + 1] == "t0:r8"
    assert command[command.index("--tool-checkpoint") + 1] == str(
        (tmp_path / "T0" / "checkpoint-500").resolve())
    assert command[command.index("--tool-budget-tokens") + 1] == "512"


def test_action_observations_execute_only_each_native_final_draft(tmp_path):
    task = "task-a"
    steps = tmp_path / "steps.jsonl"
    telemetry = tmp_path / "harness.jsonl"
    history = tmp_path / "llm_history.json"
    _write_jsonl(steps, [
        {
            "session_id": f"acon_appworld/{task}/attempt-0",
            "outer_request_id": "c1-step-0",
            "generation_trace": [
                {"discarded": True, "native_draft": {"text": "print('discarded')"}},
                {"discarded": False, "native_draft": {"text": "print('first')"}},
            ],
        },
        {
            "session_id": f"acon_appworld/{task}/attempt-0",
            "outer_request_id": "c1-step-1",
            "generation_trace": [
                {"discarded": False, "native_draft": {"text": "apis.supervisor.complete_task()"}},
            ],
        },
    ])
    _write_jsonl(telemetry, [
        {"event_type": "episode_start", "episode_id": task},
        {"event_type": "decision", "episode_id": task,
         "decision_request_id": "c1-step-0",
         "response": {"raw_response": "print('first')", "action": "print('first')"}},
        {"event_type": "tool_action", "episode_id": task,
         "decision_request_id": "c1-step-0", "action": "print('first')"},
        {"event_type": "decision", "episode_id": task,
         "decision_request_id": "c1-step-1",
         "response": {"raw_response": "apis.supervisor.complete_task()",
                      "action": "apis.supervisor.complete_task()"}},
        {"event_type": "tool_action", "episode_id": task,
         "decision_request_id": "c1-step-1",
         "action": "apis.supervisor.complete_task()"},
        {"event_type": "episode_end", "episode_id": task},
    ])
    history.write_text(json.dumps([[
        {"role": "system", "content": "system"},
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "print('first')"},
        {"role": "user", "content": "first observation"},
        {"role": "assistant", "content": "apis.supervisor.complete_task()"},
    ]]), encoding="utf-8")

    receipt = bridge.validate_task_execution(
        task, steps_path=steps, telemetry_path=telemetry, history_path=history,
    )
    assert receipt["native_decisions"] == receipt["executed_actions"] == 2
    assert receipt["discarded_drafts_not_executed"] == 1
    assert receipt["action_observation_history_bound"] is True

    rows = [json.loads(line) for line in telemetry.read_text().splitlines()]
    rows[2]["action"] = "print('discarded')"
    _write_jsonl(telemetry, rows)
    with pytest.raises(RuntimeError, match="executed action differs"):
        bridge.validate_task_execution(
            task, steps_path=steps, telemetry_path=telemetry, history_path=history,
        )

    rows[2]["action"] = "print('first')"
    rows[1]["episode_id"] = "another-task"
    _write_jsonl(telemetry, rows)
    with pytest.raises(RuntimeError, match="task identity mismatch"):
        bridge.validate_task_execution(
            task, steps_path=steps, telemetry_path=telemetry, history_path=history,
        )

    rows[1]["episode_id"] = task
    rows[2]["decision_request_id"] = "another-native-step"
    _write_jsonl(telemetry, rows)
    with pytest.raises(RuntimeError, match="request join mismatch"):
        bridge.validate_task_execution(
            task, steps_path=steps, telemetry_path=telemetry, history_path=history,
        )

    rows[2]["decision_request_id"] = "c1-step-0"
    _write_jsonl(telemetry, rows)
    history.write_text(json.dumps([[
        {"role": "system", "content": "system"},
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "print('first')"},
        {"role": "assistant", "content": "apis.supervisor.complete_task()"},
    ]]), encoding="utf-8")
    with pytest.raises(RuntimeError, match="action observation"):
        bridge.validate_task_execution(
            task, steps_path=steps, telemetry_path=telemetry, history_path=history,
        )


def test_official_scores_are_required_and_attached_to_aggregate():
    receipts = []
    for task_id, success in (
        ("scenarioa_1", True), ("scenarioa_2", False),
        ("scenariob_1", True), ("scenariob_2", True),
    ):
        receipts.append({
            "unified_metrics": {
                "task_id": task_id, "official_score": float(success),
                "normal_termination": True, "protocol_legal": None,
            },
            "official_summary": {
                "task_id": task_id,
                # A one-task evaluation reports its own SGC. The combined
                # summary must regroup outcomes instead of averaging these.
                "official_aggregate": {
                    "task_goal_completion": 100.0 if success else 0.0,
                    "scenario_goal_completion": 100.0 if success else 0.0,
                },
                "official_task_outcome": {
                    "success": success, "difficulty": 1, "num_tests": 1,
                    "passes": [] if not success else [{"requirement": "ok"}],
                    "failures": [{"requirement": "ok"}] if not success else [],
                },
            },
        })
    summary = bridge.summarize_scores(receipts)
    assert summary["official_scorer"] == "appworld evaluate (state-based unit tests)"
    assert summary["n_scored"] == 4
    assert summary["semantic_score"] == 0.75
    assert summary["task_goal_completion"] == 75.0
    assert summary["scenario_goal_completion"] == 50.0
    assert summary["official_aggregate"] == {
        "task_goal_completion": 75.0,
        "scenario_goal_completion": 50.0,
    }
    assert [row["scenario_id"] for row in summary["task_rows"]] == [
        "scenarioa", "scenarioa", "scenariob", "scenariob",
    ]
    with pytest.raises(ValueError, match="official score"):
        bridge.summarize_scores([{"unified_metrics": {"task_id": "task-c"}}])


@pytest.mark.parametrize("worker_error", [
    RuntimeError("controller finalization failed"),
    subprocess.CalledProcessError(1, ["python", "run_all.py"]),
])
@pytest.mark.parametrize("has_capacity_evidence", [True, False])
def test_capacity_failure_is_scored_zero_and_driver_runs_next_task(
    tmp_path, monkeypatch, worker_error, has_capacity_evidence,
):
    native = tmp_path / "native"
    native.mkdir()
    controller_path = native / "controller.json"
    controller_path.write_text("{}", encoding="utf-8")
    tasks = ["scenarioa_1", "scenariob_1"]
    completed_metrics = {
        "task_id": tasks[1],
        "official_score": 1.0,
        "normal_termination": True,
        "protocol_legal": None,
    }
    completed = {
        "task_id": tasks[1],
        "status": "completed",
        "unified_metrics": completed_metrics,
        "official_summary": {
            "task_id": tasks[1],
            "official_task_outcome": {"success": True, "difficulty": 1},
        },
    }
    calls = []

    def run_task(_config, task_id, *_args):
        calls.append(task_id)
        if task_id == tasks[0]:
            raise worker_error
        return completed, completed_metrics

    monkeypatch.setattr(driver, "load_delivery", lambda: object())
    monkeypatch.setattr(driver, "selected_tasks", lambda *_args, **_kwargs: tasks)
    monkeypatch.setattr(
        driver,
        "prepare_native",
        lambda *_args, **_kwargs: (native, object(), controller_path),
    )
    monkeypatch.setattr(bridge, "run_task", run_task)
    monkeypatch.setattr(
        driver,
        "controller_step_failure",
        lambda task_root: (
            ("method_failure", "capacity_infeasible", "CapacityInfeasible")
            if has_capacity_evidence and task_root.name == tasks[0]
            else None
        ),
    )

    if not has_capacity_evidence:
        with pytest.raises(type(worker_error)):
            driver.run_closed_loop({}, "appworld", tmp_path / "run")
        assert calls == tasks[:1]
        assert not (native / "task_shards" / tasks[0] / "paper_task_result.json").exists()
        return
    result = driver.run_closed_loop({}, "appworld", tmp_path / "run")

    assert result == native
    assert calls == tasks
    first = json.loads(
        (native / "task_shards" / tasks[0] / "paper_task_result.json").read_text()
    )
    assert first["status"] == "method_failure"
    assert first["unified_metrics"] == {
        "task_id": tasks[0],
        "official_score": 0.0,
        "normal_termination": False,
        "protocol_legal": None,
        "method_failure": "capacity_infeasible",
    }
    summary = json.loads((tmp_path / "run" / f"summary_{driver.ARM}.json").read_text())
    assert summary["n"] == 2
    assert summary["semantic_score"] == 0.5
    assert summary["n_method_failures"] == 1
    assert summary["method_failure_task_ids"] == [tasks[0]]
    assert summary["n_harness_failures"] == 0
    assert summary["task_rows"][0]["score_source"] == "method_failure_zero"
    assert summary["task_rows"][1]["score_source"] == "official_appworld"

    # A failed run from the pre-fix driver lacks metrics.task_id. It must remain
    # resumable from the durable receipt rather than requiring the task to rerun.
    del first["unified_metrics"]["task_id"]
    resumed = bridge.summarize_scores([first, completed])
    assert resumed["n"] == 2
    assert resumed["n_official_scored"] == 1
    assert resumed["method_failure_task_ids"] == [tasks[0]]


def test_official_task_outcome_is_saved_with_audit_details(tmp_path):
    path = tmp_path / "evaluation.json"
    outcome = {
        "success": False, "difficulty": 2, "num_tests": 2,
        "passes": [{"requirement": "first", "label": "passed"}],
        "failures": [{"requirement": "second", "trace": "assertion", "label": "failed"}],
    }
    path.write_text(json.dumps({
        "aggregate": {"task_goal_completion": 0.0, "scenario_goal_completion": 0.0},
        "individual": {"3d9a636_1": outcome},
    }), encoding="utf-8")
    assert bridge._official_task_outcome(path, "3d9a636_1") == outcome
    with pytest.raises(RuntimeError, match="exactly the selected task"):
        bridge._official_task_outcome(path, "3d9a636_2")


def test_appworld_sampling_and_actor_cap_reach_native_runtime(tmp_path):
    runtime = DELIVERY / "runtime"
    script = r'''
import json, time
from pathlib import Path
from benchmarks.memory_runtime.event_native_server import _sampling_params_for_benchmark
from benchmarks.memory_runtime.single_task_harness_api import SingleTaskHarnessAPI

class Runner:
    def __init__(self):
        self.max_new_tokens = 4096
        self.seen_cap = None
        self.controller = object()
        self.generator = object()
        self.generation_calls = 0
    def run(self, payload):
        self.seen_cap = self.max_new_tokens
        self.generation_calls += 1
        return {
            "status": "ok",
            "outer_request_id": "c1-test-request",
            "session_id": payload["session_id"],
            "decision_key": payload["decision_key"],
            "generation_trace": [{"native_draft": {"text": "print('final')"}}],
            "generation_usage_total": {
                "prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12,
            },
            "response": {"role": "assistant", "content": "parsed", "tool_calls": [],
                         "reasoning_content": None, "finish_reason": "stop"},
        }

runner = Runner()
api = SingleTaskHarnessAPI(
    runner, run_id="r", model_name="c1_t02_risk", benchmark="acon_appworld",
    view_mode="ac_native_s0_lexical_raw_reserve_failed_operation",
    max_new_tokens=4096, allowed_task_ids=["task-a"], max_decisions=2,
    deadline_monotonic=time.monotonic() + 30, steps_path=Path(r"'''+str(tmp_path / "steps.jsonl")+r'''"),
    compression_policy="always-compress-v1",
)
response = api.handle_chat({
    "model": "c1_t02_risk", "messages": [{"role": "user", "content": "task"}],
    "c2kv_measurement_session_id": "task-a",
    "temperature": 0, "top_p": 1, "seed": 42, "presence_penalty": 0.5,
    "frequency_penalty": 0, "max_tokens": 2048, "stream": False, "n": 1,
    "chat_template_kwargs": {"enable_thinking": False},
})
wrong_identity_status = None
try:
    api.handle_chat({
        "model": "c1_t02_risk", "messages": [{"role": "user", "content": "different"}],
        "c2kv_measurement_session_id": "task-b", "temperature": 0,
        "max_tokens": 2048, "stream": False,
    })
except Exception as error:
    wrong_identity_status = getattr(error, "status", getattr(error, "status_code", None))
print(json.dumps({
    "sampling": _sampling_params_for_benchmark("acon_appworld"),
    "seen_cap": runner.seen_cap, "restored_cap": runner.max_new_tokens,
    "content": response["choices"][0]["message"]["content"],
    "wrong_identity_status": wrong_identity_status,
}))
'''
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join((str(runtime / "python"), str(runtime))),
    }
    process = subprocess.run(
        [sys.executable, "-c", script], cwd=runtime, env=env,
        check=False, capture_output=True, text=True,
    )
    assert process.returncode == 0, process.stderr
    observed = json.loads(process.stdout)
    assert observed == {
        "sampling": {
            "temperature": 0.0, "top_p": 1.0,
            "presence_penalty": 0.5, "seed": 42,
        },
        "seen_cap": 2048,
        "restored_cap": 4096,
        "content": "print('final')",
        "wrong_identity_status": 409,
    }
