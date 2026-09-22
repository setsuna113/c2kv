"""Exact typed text-budget 422s are declared ACEBench task failures (CPU only).

The instrumented generator continues after such a task; the adapter scores it
0 only when its receipt, failed episode, and proxy declaration agree. Runs
without a receipt keep the original generator environment, scorer cwd and
summary.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from adapters import acebench_adapter as B  # noqa: E402
from adapters import acebench_task_failures as F  # noqa: E402
from benchmarks import acebench_cli as hook  # noqa: E402

CODE = "acon_history_budget_exceeded"


class UnprocessableEntityError(Exception):
    """Shape of openai.UnprocessableEntityError without importing the SDK."""

    def __init__(self, body, status_code=422):
        super().__init__(f"Error code: {status_code} - {body!r}")
        self.body, self.status_code = body, status_code


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _conv(session):
    return hashlib.sha256(json.dumps(["measurement_session", session],
                                     separators=(",", ":")).encode()).hexdigest()


def _official_inference(monkeypatch, tmp_path, error):
    """inference -> task_wrapper(multi_turn_inference) raising ``error``."""
    monkeypatch.setenv("C2KV_ACEBENCH_TELEMETRY", str(tmp_path / "events.jsonl"))
    monkeypatch.setenv(F.ENV, str(tmp_path / "failures.jsonl"))

    @hook.task_wrapper
    def multi_turn_inference(question, test_id):
        raise error

    @hook.declared_failure_wrapper
    def inference(question, functions, time, profile, test_case, id):
        return multi_turn_inference(question, id.split("_")[-1])

    return inference


def test_typed_budget_422_ends_only_its_task_with_a_bound_receipt(monkeypatch, tmp_path):
    inference = _official_inference(
        monkeypatch, tmp_path, UnprocessableEntityError({"code": CODE, "type": "method_budget_failure"}))
    result, process = inference("q", [], "", "", {}, "agent_multi_turn_3")
    assert result is hook.DECLARED_TASK_FAILURE and process is hook.DECLARED_TASK_FAILURE
    receipt = json.loads((tmp_path / "failures.jsonl").read_text(encoding="utf-8"))
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert receipt["task_id"] == "agent_multi_turn_3"
    assert receipt["task_failure_kind"] == CODE and receipt["status_code"] == 422
    assert receipt["episode_instance_id"] == events[-1]["episode_instance_id"]
    assert events[-1]["event_type"] == "episode_end" and events[-1]["status"] == "failed"
    written = []
    writer = hook.result_writer_wrapper(lambda self, row, *rest: written.append(row))
    writer(None, {"id": "agent_multi_turn_3", "result": result, "process": process}, "m", "p/")
    writer(None, {"id": "agent_multi_turn_4", "result": [], "process": []}, "m", "p/")
    assert [row["id"] for row in written] == ["agent_multi_turn_4"]


@pytest.mark.parametrize("error", [
    UnprocessableEntityError({"code": "context_window_exceeded"}),
    UnprocessableEntityError({"code": CODE}, status_code=400),
    UnprocessableEntityError(f"{{'error': {{'code': '{CODE}'}}}}"),
    RuntimeError(f"Error code: 422 - {{'code': '{CODE}'}}"),
])
def test_other_errors_still_end_the_generator_unchanged(monkeypatch, tmp_path, error):
    inference = _official_inference(monkeypatch, tmp_path, error)
    with pytest.raises(type(error)) as raised:
        inference("q", [], "", "", {}, "agent_multi_turn_3")
    assert raised.value is error
    assert not (tmp_path / "failures.jsonl").exists()


def test_typed_422_outside_an_agent_episode_is_not_declared(monkeypatch, tmp_path):
    monkeypatch.setenv(F.ENV, str(tmp_path / "failures.jsonl"))
    error = UnprocessableEntityError({"code": CODE})

    @hook.declared_failure_wrapper
    def inference(question, functions, time, profile, test_case, id):
        raise error

    with pytest.raises(UnprocessableEntityError):
        inference("q", [], "", "", {}, "normal_single_turn_single_function_1")
    assert not (tmp_path / "failures.jsonl").exists()


def test_declaration_mode_is_opt_in(monkeypatch):
    monkeypatch.delenv(F.ENV, raising=False)
    assert hook.declared_failure_mode() is False
    monkeypatch.setenv(F.ENV, "/tmp/receipts.jsonl")
    assert hook.declared_failure_mode() is True


_HARNESS = {
    "model_inference/__init__.py": "",
    "model_inference/multi_turn/__init__.py": "",
    "model_inference/multi_step/__init__.py": "",
    "model_inference/multi_turn/execution_role.py":
        "class EXECUTION:\n    def respond(self, history):\n        return history\n",
    "model_inference/multi_step/execution_role_step.py":
        "class EXECUTION_STEP:\n    def respond(self, history):\n        return history\n",
    "model_inference/apimodel_inference.py": '''
import json, os
from model_inference.multi_turn.execution_role import EXECUTION
from model_inference.multi_step.execution_role_step import EXECUTION_STEP

class UnprocessableEntityError(Exception):
    def __init__(self, body):
        super().__init__(f"Error code: 422 - {body!r}")
        self.body, self.status_code = body, 422

class APIModelInference:
    def inference(self, question, functions, time, profile, test_case, id):
        return self.multi_turn_inference(question, {}, functions, [], id.split("_")[-1], time)
    def multi_turn_inference(self, question, initial_config, functions, involved_classes, test_id, time):
        if question == "over budget":
            raise UnprocessableEntityError({"code": "acon_history_budget_exceeded"})
        return [], []
    def multi_step_inference(self, question, initial_config, functions, involved_classes, test_id, time):
        return [], []
    def write_result(self, result, model_name, result_path):
        os.makedirs(result_path, exist_ok=True)
        with open(os.path.join(result_path, "results.json"), "a") as stream:
            stream.write(json.dumps(result) + "\\n")
''',
    # upstream generate.py control flow: any task exception is re-raised
    "generate.py": '''
from concurrent.futures import ThreadPoolExecutor, as_completed
from model_inference.apimodel_inference import APIModelInference
cases = [{"id": f"agent_multi_turn_{i}", "question": q}
         for i, q in enumerate(("ok", "over budget", "ok"))]
def generate_signal(case):
    model = APIModelInference()
    result, process = model.inference(case["question"], [], "", "", case, case["id"])
    model.write_result({"id": case["id"], "result": result, "process": process}, "m", "./result/")
with ThreadPoolExecutor(max_workers=1) as executor:
    for future in as_completed([executor.submit(generate_signal, case) for case in cases]):
        try:
            future.result()
        except Exception as e:
            print(f"Task raised an exception: {e}")
            raise
print("All tasks have been completed.")
''',
}


@pytest.mark.parametrize("declared", [False, True])
def test_official_generator_control_flow_continues_only_when_declared(tmp_path, declared):
    import os
    import subprocess

    harness = tmp_path / "harness"
    for name, text in _HARNESS.items():
        (harness / name).parent.mkdir(parents=True, exist_ok=True)
        (harness / name).write_text(text, encoding="utf-8")
    env = dict(os.environ, ACEBENCH_AGENT_BASE_URL="http://agent/v1",
               C2KV_ACEBENCH_TELEMETRY=str(tmp_path / "events.jsonl"))
    env.pop(F.ENV, None)
    if declared:
        env[F.ENV] = str(tmp_path / "failures.jsonl")
    completed = subprocess.run(
        [sys.executable, str(Path(hook.__file__)), str(harness)],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120)
    rows = [json.loads(line) for line in
            (tmp_path / "result" / "results.json").read_text().splitlines()]
    assert [row["id"] for row in rows] == ["agent_multi_turn_0", "agent_multi_turn_2"]
    if declared:
        assert completed.returncode == 0, completed.stderr
        receipt = json.loads((tmp_path / "failures.jsonl").read_text(encoding="utf-8"))
        assert receipt["task_id"] == "agent_multi_turn_1"
    else:
        assert completed.returncode != 0
        assert "acon_history_budget_exceeded" in completed.stdout
        assert not (tmp_path / "failures.jsonl").exists()


# ---- adapter: generator environment and verified scoring ---------------------

def _checkout(tmp_path, ids=("agent_multi_turn_0", "agent_multi_turn_1")):
    root = tmp_path / "acebench"
    (root / "category.py").parent.mkdir(parents=True)
    (root / "category.py").write_text(
        "ACE_DATA_CATEGORY = {'agent': ['agent_multi_turn']}\n", encoding="utf-8")
    data = root / "data_all" / "data_en"
    _write_jsonl(data / "data_agent_multi_turn.json", [{"id": task} for task in ids])
    _write_jsonl(data / "possible_answer" / "data_agent_multi_turn.json",
                 [{"id": task, "ground_truth": []} for task in ids])
    return root


def _official(results, *, receipt=None, proxy_code=CODE, episode_status="failed"):
    """run_owned fake: generate writes ``results`` (+ evidence), eval scores all rows."""
    calls = []

    def run_owned(command, *, cwd, env, check):
        calls.append((command, Path(cwd), dict(env)))
        out = Path(env["C2KV_ACEBENCH_TELEMETRY"]).parents[1]
        if len(calls) == 1:
            _write_jsonl(B.result_path(cwd, "en", "m", "agent_multi_turn"),
                         [{"id": task, "result": [], "process": []} for task in results])
            if receipt is not None:
                session = "acebench:1:session"
                _write_jsonl(out / "measurement" / "harness_events.jsonl", [
                    {"event_type": "episode_start", "episode_id": "1",
                     "episode_instance_id": session},
                    {"event_type": "episode_end", "episode_id": "1",
                     "episode_instance_id": session, "status": episode_status},
                ])
                _write_jsonl(out / "logs" / "proxy_acon_hist_ut_co_b128_34100.jsonl", [
                    {"conv_id": _conv(session), "status": proxy_code, "error_kind": proxy_code},
                ])
                _write_jsonl(Path(env[F.ENV]), [dict(
                    F.receipt(receipt, CODE, UnprocessableEntityError({"code": CODE}),
                              {"task": "1", "session": session}))])
        else:
            scored = B._jsonl(B.result_path(cwd, "en", "m", "agent_multi_turn"))
            _write_jsonl(B.score_path(cwd, "en", "m", "agent_multi_turn"), [
                {"end_to_end_accuracy": 1.0, "process_accuracy": 1.0,
                 "correct_count": len(scored), "total_count": len(scored)}])
    return calls, run_owned


def test_default_run_keeps_generator_env_scorer_cwd_and_summary(tmp_path, monkeypatch):
    root = _checkout(tmp_path)
    calls, fake = _official(["agent_multi_turn_0", "agent_multi_turn_1"])
    monkeypatch.setattr(B, "run_owned", fake)
    summary = B.run_acebench("http://agent", "http://user", tmp_path / "out",
                             acebench_dir=root, model="m", python="python")
    assert all(F.ENV not in env for _command, _cwd, env in calls)
    assert calls[1][1] == calls[0][1] == tmp_path / "out" / "acebench_work"
    assert summary["n"] == 2 and summary["semantic_score"] == 1.0
    assert not {"task_failures", "n_task_failures", "score_workdir"} & set(summary)


def test_budget_arm_declares_bound_failure_as_zero_and_scores_rest(tmp_path, monkeypatch):
    root = _checkout(tmp_path)
    calls, fake = _official(["agent_multi_turn_0"], receipt="agent_multi_turn_1")
    monkeypatch.setattr(B, "run_owned", fake)
    out = tmp_path / "out"
    summary = B.run_acebench("http://agent", "http://user", out, acebench_dir=root,
                             model="m", python="python", declare_text_budget_failures=True)
    assert calls[0][2][F.ENV] == str(out.resolve() / F.RECEIPTS)
    scorer_cwd = calls[1][1]
    assert scorer_cwd == out / "acebench_score"
    assert [row["id"] for row in B._jsonl(B.data_path(scorer_cwd, "en", "agent_multi_turn"))] == [
        "agent_multi_turn_0"]
    assert summary["n"] == 2 and summary["semantic_score"] == 0.5
    assert summary["task_failures"] == {CODE: ["agent_multi_turn_1"]}
    assert summary["n_official_scored"] == 1 and summary["n_task_failures"] == 1
    assert summary["workdir"] == str(out / "acebench_work")
    # the generation workdir keeps the official generator's own files
    assert not B.score_path(out / "acebench_work", "en", "m", "agent_multi_turn").exists()


@pytest.mark.parametrize("change", ["proxy_code", "episode_ok", "has_result"])
def test_unbound_receipt_fails_the_run(tmp_path, monkeypatch, change):
    root = _checkout(tmp_path)
    results = ["agent_multi_turn_0"] + (["agent_multi_turn_1"] if change == "has_result" else [])
    _calls, fake = _official(
        results, receipt="agent_multi_turn_1",
        proxy_code="hiagent_history_budget_exceeded" if change == "proxy_code" else CODE,
        episode_status="completed" if change == "episode_ok" else "failed")
    monkeypatch.setattr(B, "run_owned", fake)
    with pytest.raises(SystemExit, match="FATAL: ACEBench task failure"):
        B.run_acebench("http://agent", "http://user", tmp_path / "out", acebench_dir=root,
                       model="m", python="python", declare_text_budget_failures=True)


def test_missing_result_without_receipt_is_still_fatal(tmp_path, monkeypatch):
    root = _checkout(tmp_path)
    _calls, fake = _official(["agent_multi_turn_0"])
    monkeypatch.setattr(B, "run_owned", fake)
    with pytest.raises(SystemExit, match="missing result ids: agent_multi_turn_1"):
        B.run_acebench("http://agent", "http://user", tmp_path / "out", acebench_dir=root,
                       model="m", python="python", declare_text_budget_failures=True)


def test_run_declares_only_for_text_history_budget_arms(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(B, "run_acebench", lambda *args, **kwargs: (
        seen.append(kwargs["declare_text_budget_failures"]), {"categories": ["agent_x"]})[1])
    for arm in ("full", "acon_hist_ut_co", "acon_hist_ut_co_b128", "hiagent_full_b192"):
        B.run(SimpleNamespace(base_url="b", user_base_url="u", out_dir=tmp_path, model="m",
                              arm=arm, opt=lambda key, default=None: default))
    assert seen == [False, False, True, True]
