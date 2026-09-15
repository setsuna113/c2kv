"""Model-free checks for the pinned ACEBench identity transport patch."""
from __future__ import annotations

import copy
import importlib
import io
import os
import subprocess
import sys
import tarfile
import time
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from benchmarks.memory_runtime.event_native_api import EventNativeAPI, EventNativeAPIError


ACEBENCH_COMMIT = "56dd66cf6439b0d9655ee1b353e4cd745c6f664e"
PATCH = Path(__file__).parent / "acebench_patches" / "0001-endpoint-env-and-model-registry.patch"


def _upstream_checkout() -> Path:
    configured = os.environ.get("ACEBENCH_TEST_UPSTREAM")
    if configured:
        return Path(configured).resolve()
    return Path(__file__).resolve().parents[2] / "tmp" / "baselines" / "acebench"


@pytest.fixture(scope="session")
def patched_acebench(tmp_path_factory) -> Path:
    upstream = _upstream_checkout()
    if not (upstream / ".git").exists():
        pytest.skip(f"pinned ACEBench checkout is unavailable: {upstream}")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=upstream, check=True,
        text=True, capture_output=True,
    ).stdout.strip()
    assert commit == ACEBENCH_COMMIT

    archive = subprocess.run(
        ["git", "archive", "--format=tar", "HEAD"], cwd=upstream,
        check=True, stdout=subprocess.PIPE,
    ).stdout
    root = tmp_path_factory.mktemp("acebench-patched")
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as bundle:
        bundle.extractall(root, filter="data")
    # A caller may retain test artifacts below another Git worktree. Give the
    # archive its own root so git apply cannot silently filter all patch paths.
    subprocess.run(["git", "init", "--quiet"], cwd=root, check=True, capture_output=True)
    subprocess.run(
        ["git", "apply", "--check", "--unidiff-zero", str(PATCH)],
        cwd=root, check=True, capture_output=True,
    )
    subprocess.run(
        ["git", "apply", "--unidiff-zero", str(PATCH)],
        cwd=root, check=True, capture_output=True,
    )
    return root


@contextmanager
def _ace_modules(root: Path):
    saved = {
        name: module for name, module in sys.modules.items()
        if name == "model_inference" or name.startswith("model_inference.")
    }
    saved_openai = sys.modules.get("openai")
    saved_wcwidth = sys.modules.get("wcwidth")
    saved_dotenv = sys.modules.get("dotenv")
    openai_stub = ModuleType("openai")
    openai_stub.OpenAI = lambda **unused: None
    wcwidth_stub = ModuleType("wcwidth")
    wcwidth_stub.wcswidth = len
    sys.modules["openai"] = openai_stub
    sys.modules["wcwidth"] = wcwidth_stub
    dotenv_stub = ModuleType("dotenv")
    dotenv_stub.load_dotenv = lambda: None
    sys.modules["dotenv"] = dotenv_stub
    for name in saved:
        sys.modules.pop(name, None)
    sys.path.insert(0, str(root))
    try:
        yield SimpleNamespace(
            api=importlib.import_module("model_inference.apimodel_inference"),
            step=importlib.import_module("model_inference.multi_step.APIModel_agent"),
            turn=importlib.import_module("model_inference.multi_turn.APIModel_agent"),
            user=importlib.import_module("model_inference.multi_turn.APIModel_user"),
            history=importlib.import_module("model_inference.role_history"),
        )
    finally:
        for name in list(sys.modules):
            if name == "model_inference" or name.startswith("model_inference."):
                sys.modules.pop(name, None)
        sys.modules.update(saved)
        if saved_openai is None:
            sys.modules.pop("openai", None)
        else:
            sys.modules["openai"] = saved_openai
        if saved_wcwidth is None:
            sys.modules.pop("wcwidth", None)
        else:
            sys.modules["wcwidth"] = saved_wcwidth
        if saved_dotenv is None:
            sys.modules.pop("dotenv", None)
        else:
            sys.modules["dotenv"] = saved_dotenv
        sys.path.remove(str(root))


def _response(content: str):
    return SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(content=content)
    )])


class CaptureClient:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs):
        self.calls.append(copy.deepcopy(kwargs))
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return _response(outcome)


def _context(call):
    return call["extra_body"]["c2kv_eval_context"]


def test_actual_agent_call_sites_emit_exact_stable_identity(
    patched_acebench, monkeypatch,
):
    monkeypatch.setenv("ACEBENCH_AGENT_BASE_URL", "http://agent.invalid/v1")
    monkeypatch.delenv("ACEBENCH_EVENT_NATIVE_V1", raising=False)
    with _ace_modules(patched_acebench) as modules:
        step_client = CaptureClient([
            "[Lookup(id='7')]", "[Lookup(id='7')]", "finish conversation",
        ])
        monkeypatch.setattr(modules.step, "OpenAI", lambda **unused: step_client)
        step = modules.step.APIAgent_step(
            "served-model", "", [{"name": "Lookup"}], temperature=0.25,
            top_p=0.9, max_tokens=77, language="en",
            task_id="agent_multi_step_17",
        )
        first = [{"sender": "user", "recipient": "agent", "message": "Find 7"}]
        assert step.respond(first)["recipient"] == "execution"
        after_execution = first + [
            {"sender": "agent", "recipient": "execution", "message": "[Lookup(id='7')]"},
            {"sender": "execution", "recipient": "agent", "message": {"value": "seven"}},
        ]
        step.respond(after_execution)
        step.respond(after_execution)
        assert [_context(call) for call in step_client.calls] == [
            {"benchmark": "acebench", "task_id": "agent_multi_step_17",
             "user_turn": 0, "step": 0, "attempt": 0},
            {"benchmark": "acebench", "task_id": "agent_multi_step_17",
             "user_turn": 0, "step": 1, "attempt": 0},
            {"benchmark": "acebench", "task_id": "agent_multi_step_17",
             "user_turn": 0, "step": 1, "attempt": 0},
        ]
        assert step_client.calls[0]["temperature"] == 0.25
        assert step_client.calls[0]["top_p"] == 0.9
        assert step_client.calls[0]["max_tokens"] == 77

        turn_client = CaptureClient(["Need more information."])
        monkeypatch.setattr(modules.turn, "OpenAI", lambda **unused: turn_client)
        turn = modules.turn.APIAgent_turn(
            "served-model", "", [], [], language="en",
            task_id="agent_multi_turn_4",
        )
        next_user = [
            {"sender": "user", "recipient": "agent", "message": "Book a trip"},
            {"sender": "agent", "recipient": "user", "message": "Which day?"},
            {"sender": "user", "recipient": "agent", "message": "Tuesday"},
        ]
        turn.respond(next_user)
        assert _context(turn_client.calls[0]) == {
            "benchmark": "acebench", "task_id": "agent_multi_turn_4",
            "user_turn": 1, "step": 0, "attempt": 0,
        }


def test_single_turn_retry_reuses_exact_context(patched_acebench, monkeypatch):
    monkeypatch.setenv("ACEBENCH_AGENT_BASE_URL", "http://agent.invalid/v1")
    monkeypatch.delenv("ACEBENCH_EVENT_NATIVE_V1", raising=False)
    with _ace_modules(patched_acebench) as modules:
        client = CaptureClient([RuntimeError("data_inspection_failed"), "done"])
        handler = object.__new__(modules.api.APIModelInference)
        handler.language = "en"
        handler.client = client
        handler.model_name = "served-model"
        handler.temperature = 0.0
        handler.top_p = 1.0
        handler.max_tokens = 32
        assert handler.single_turn_inference(
            "question", [], "normal", "", "", "normal_42"
        ) == "done"
        assert [_context(call) for call in client.calls] == [
            {"benchmark": "acebench", "task_id": "normal_42",
             "user_turn": 0, "step": 0, "attempt": 0},
        ] * 2


def test_user_simulator_never_receives_eval_context(patched_acebench, monkeypatch):
    monkeypatch.setenv("ACEBENCH_USER_BASE_URL", "http://user.invalid/v1")
    client = CaptureClient(["I need help.", "Tuesday."])
    with _ace_modules(patched_acebench) as modules:
        monkeypatch.setattr(modules.user, "OpenAI", lambda **unused: client)
        user = modules.user.APIUSER("served-user", ["BaseApi"], language="en")
        user.get_init_prompt("Do a task")
        user.step("agent:Which day?")
        user.respond()
    assert len(client.calls) == 2
    assert all("extra_body" not in call for call in client.calls)


class FakeRunner:
    generation_calls = 0

    def __init__(self, content):
        self.content = content
        self.calls = []

    def run(self, payload):
        self.calls.append(copy.deepcopy(payload))
        return {
            "status": "ok",
            "response": {
                "role": "assistant", "content": self.content,
                "tool_calls": [], "finish_reason": "stop",
            },
            "generation_usage_total": {
                "prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13,
            },
        }


class EventNativeClient(CaptureClient):
    def __init__(self, api):
        super().__init__([])
        self.api = api

    def create(self, **kwargs):
        self.calls.append(copy.deepcopy(kwargs))
        payload = copy.deepcopy(kwargs)
        payload.update(payload.pop("extra_body"))
        response = self.api.handle_chat(payload)
        return _response(response["choices"][0]["message"]["content"])


def _event_api(tmp_path, runner, task_id):
    return EventNativeAPI(
        runner, run_id="ace-identity-test", model_name="served-model",
        benchmark="acebench", view_mode="capacity_exact_once", max_new_tokens=32,
        allowed_task_ids=[task_id], max_decisions=2,
        deadline_monotonic=time.monotonic() + 60,
        steps_path=tmp_path / "steps.jsonl",
    )


def test_initial_actual_call_site_reaches_event_native_api(
    patched_acebench, monkeypatch, tmp_path,
):
    monkeypatch.setenv("ACEBENCH_AGENT_BASE_URL", "http://agent.invalid/v1")
    monkeypatch.setenv("ACEBENCH_EVENT_NATIVE_V1", "1")
    monkeypatch.setenv("ACEBENCH_ROLE_HISTORY_V1", "1")
    runner = FakeRunner("[Lookup(id='7')]")
    api = _event_api(tmp_path, runner, "agent_multi_step_17")
    client = EventNativeClient(api)
    with _ace_modules(patched_acebench) as modules:
        monkeypatch.setattr(modules.step, "OpenAI", lambda **unused: client)
        class InitialScene:
            latest = None

            def __init__(self, **unused):
                self.dialogue_history = [
                    {"sender": "user", "recipient": "agent", "message": "Find 7"},
                ]

            def get_inference_message(self):
                return "user:Find 7\n"

            def add_dialogue(self, message):
                self.dialogue_history.append(message)
                InitialScene.latest = message

            def write_message_history(self, *unused):
                return None

        monkeypatch.setattr(modules.api, "Mulit_Step_Scene", InitialScene)
        monkeypatch.setattr(modules.api, "EXECUTION_STEP", lambda **unused: object())
        handler = object.__new__(modules.api.APIModelInference)
        handler.model_name = "served-model"
        handler.temperature = 0.0
        handler.top_p = 1.0
        handler.max_tokens = 32
        handler.max_dialog_turns = 1
        handler.language = "en"
        handler.multi_step_inference(
            "Find 7", {}, [], [], "17", "", "agent_multi_step_17",
        )
    assert InitialScene.latest == {
        "sender": "agent", "recipient": "execution", "message": "[Lookup(id='7')]",
    }
    assert set(client.calls[0]) == {
        "messages", "model", "temperature", "max_completion_tokens",
        "store", "seed", "extra_body",
    }
    assert runner.calls[0]["session_id"] == "acebench/agent_multi_step_17/attempt-0"
    assert runner.calls[0]["decision_key"] == "turn-0/step-0"


def test_event_native_handler_forwards_decode_controls_to_both_agents(
    patched_acebench, monkeypatch,
):
    monkeypatch.setenv("ACEBENCH_EVENT_NATIVE_V1", "1")
    constructed = {}

    class Agent:
        def __init__(self, kind, **kwargs):
            constructed[kind] = kwargs

    class EmptyScene:
        def __init__(self, **unused):
            self.dialogue_history = [
                {"sender": "user", "recipient": "agent", "message": "initial"},
            ]

        def write_message_history(self, *unused):
            return None

    class User:
        def __init__(self, **unused):
            pass

        def get_init_prompt(self, question):
            return question

    with _ace_modules(patched_acebench) as modules:
        monkeypatch.setattr(
            modules.api, "APIAgent_step", lambda **kwargs: Agent("step", **kwargs)
        )
        monkeypatch.setattr(
            modules.api, "APIAgent_turn", lambda **kwargs: Agent("turn", **kwargs)
        )
        monkeypatch.setattr(modules.api, "Mulit_Step_Scene", EmptyScene)
        monkeypatch.setattr(modules.api, "Scene", EmptyScene)
        monkeypatch.setattr(modules.api, "APIUSER", User)
        monkeypatch.setattr(modules.api, "EXECUTION_STEP", lambda **unused: object())
        monkeypatch.setattr(modules.api, "EXECUTION", lambda **unused: object())
        handler = object.__new__(modules.api.APIModelInference)
        handler.model_name = "served-model"
        handler.temperature = 0.0
        handler.top_p = 1.0
        handler.max_tokens = 32
        handler.max_dialog_turns = 0
        handler.language = "en"
        handler.user_model = "served-user"
        handler.multi_step_inference("q", {}, [], [], "1", "", "agent_multi_step_1")
        handler.multi_turn_inference("q", {}, [], [], "2", "", "agent_multi_turn_2")
    for kind, task_id in (("step", "agent_multi_step_1"),
                          ("turn", "agent_multi_turn_2")):
        assert constructed[kind]["task_id"] == task_id
        assert constructed[kind]["temperature"] == 0.0
        assert constructed[kind]["top_p"] == 1.0
        assert constructed[kind]["max_tokens"] == 32


def test_execution_continuation_exposes_textual_tool_dialect_gap(
    patched_acebench, monkeypatch, tmp_path,
):
    monkeypatch.setenv("ACEBENCH_AGENT_BASE_URL", "http://agent.invalid/v1")
    monkeypatch.setenv("ACEBENCH_EVENT_NATIVE_V1", "1")
    runner = FakeRunner("done")
    api = _event_api(tmp_path, runner, "agent_multi_step_17")
    client = EventNativeClient(api)
    history = [
        {"sender": "user", "recipient": "agent", "message": "Find 7"},
        {"sender": "agent", "recipient": "execution", "message": "[Lookup(id='7')]"},
        {"sender": "execution", "recipient": "agent", "message": {"value": "seven"}},
    ]
    with _ace_modules(patched_acebench) as modules:
        monkeypatch.setattr(modules.step, "OpenAI", lambda **unused: client)
        agent = modules.step.APIAgent_step(
            "served-model", "", [], temperature=0.0, top_p=1.0,
            max_tokens=32, language="en", task_id="agent_multi_step_17",
        )
        with pytest.raises(EventNativeAPIError) as error:
            agent.respond(history)
    assert error.value.code == "invalid_messages"
    assert "Unmatched or duplicate tool result" in str(error.value)
    assert api.decisions_reserved == 0
    assert runner.calls == []
