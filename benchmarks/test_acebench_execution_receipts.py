"""CPU seams for the ACEBench textual-action execution receipts."""
from __future__ import annotations

import copy
import importlib
import io
import os
import subprocess
import sys
import tarfile
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


ACEBENCH_COMMIT = "56dd66cf6439b0d9655ee1b353e4cd745c6f664e"
PATCH = Path(__file__).parent / "acebench_patches" / "0001-endpoint-env-and-model-registry.patch"


def _upstream_checkout() -> Path:
    configured = os.environ.get("ACEBENCH_TEST_UPSTREAM")
    if configured:
        return Path(configured).resolve()
    return Path(__file__).resolve().parents[2] / "tmp" / "baselines" / "acebench"


@pytest.fixture(scope="session")
def patched_acebench_receipts(tmp_path_factory) -> Path:
    upstream = _upstream_checkout()
    if not (upstream / ".git").is_dir():
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
    root = tmp_path_factory.mktemp("acebench-execution-receipts")
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as bundle:
        bundle.extractall(root, filter="data")
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
    dependency_stubs = {
        "openai": ModuleType("openai"),
        "wcwidth": ModuleType("wcwidth"),
        "dotenv": ModuleType("dotenv"),
    }
    dependency_stubs["openai"].OpenAI = lambda **unused: None
    dependency_stubs["wcwidth"].wcswidth = len
    dependency_stubs["dotenv"].load_dotenv = lambda: None
    saved_dependencies = {name: sys.modules.get(name) for name in dependency_stubs}
    sys.modules.update(dependency_stubs)
    for name in saved:
        sys.modules.pop(name, None)
    sys.path.insert(0, str(root))
    try:
        yield SimpleNamespace(
            api=importlib.import_module("model_inference.apimodel_inference"),
            history=importlib.import_module("model_inference.role_history"),
            step_agent=importlib.import_module("model_inference.multi_step.APIModel_agent"),
            step_execution=importlib.import_module(
                "model_inference.multi_step.execution_role_step"),
            turn_execution=importlib.import_module(
                "model_inference.multi_turn.execution_role"),
            user=importlib.import_module("model_inference.multi_turn.APIModel_user"),
        )
    finally:
        for name in list(sys.modules):
            if name == "model_inference" or name.startswith("model_inference."):
                sys.modules.pop(name, None)
        sys.modules.update(saved)
        for name, module in saved_dependencies.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
        sys.path.remove(str(root))


class _CaptureClient:
    def __init__(self, contents: list[str]):
        self.contents = list(contents)
        self.calls: list[dict] = []
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs):
        self.calls.append(copy.deepcopy(kwargs))
        content = self.contents.pop(0)
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content=content)
        )])


def _history() -> list[dict]:
    return [
        {"sender": "user", "recipient": "agent", "message": "Do the action"},
        {"sender": "agent", "recipient": "execution", "message": "[Echo(value='x')]"},
    ]


def _enable_text_actions(monkeypatch):
    monkeypatch.setenv("ACEBENCH_TEXT_ACTIONS_V1", "1")
    monkeypatch.setenv("ACEBENCH_ROLE_HISTORY_V1", "1")
    monkeypatch.setenv("ACEBENCH_EVENT_NATIVE_V1", "1")


def test_handler_binds_step_receipt_and_actual_agent_request_exports_only_source(
    patched_acebench_receipts, monkeypatch,
):
    _enable_text_actions(monkeypatch)
    monkeypatch.setenv("ACEBENCH_AGENT_BASE_URL", "http://agent.invalid/v1")
    with _ace_modules(patched_acebench_receipts) as modules:
        history_holder = {}

        class Scene:
            def __init__(self, **unused):
                self.dialogue_history = [{
                    "sender": "user", "recipient": "agent", "message": "Do the action",
                }]
                history_holder["scene"] = self

            def get_inference_message(self):
                return "unused"

            def add_dialogue(self, message):
                self.dialogue_history.append(message)

            def write_message_history(self, *unused):
                return None

        class Agent:
            def __init__(self, **unused):
                pass

            def respond(self, history):
                if history[-1]["sender"] == "user":
                    return {
                        "sender": "agent", "recipient": "execution",
                        "message": "[Echo(value='x')]",
                    }
                return {
                    "sender": "agent", "recipient": "user",
                    "message": "finish conversation",
                }

        class Progress:
            def __enter__(self):
                return self

            def __exit__(self, *unused):
                return False

            def update(self, *unused):
                return None

        monkeypatch.setattr(modules.step_execution, "execute_agent_func_call", lambda **unused: (
            ['{"echo": "x"}', "opaque"], {},
        ))
        monkeypatch.setattr(modules.api, "Mulit_Step_Scene", Scene)
        monkeypatch.setattr(modules.api, "APIAgent_step", Agent)
        monkeypatch.setattr(modules.api, "EXECUTION_STEP", modules.step_execution.EXECUTION_STEP)
        monkeypatch.setattr(modules.api, "tqdm", lambda **unused: Progress())
        handler = object.__new__(modules.api.APIModelInference)
        handler.model_name = "served-model"
        handler.temperature = 0.0
        handler.top_p = 1.0
        handler.max_tokens = 16
        handler.max_dialog_turns = 3
        handler.language = "en"
        handler.multi_step_inference("Do the action", {}, [], [], "1", "", "agent_multi_step_1")

        history = history_holder["scene"].dialogue_history
        execution = history[2]
        assert execution["message"] == [{"echo": "x"}, "opaque"]
        assert execution["c2kv_acebench_execution"] == {
            "version": "acebench-execution-receipt-v1",
            "agent_history_index": 1,
            "execution_message_index": 3,
            "decode_status": "ok",
            "decoded_calls": ["Echo(value='x')"],
            "executor_status": "returned",
            "executor_return_shape": "list",
            "executor_return_count": 2,
        }

        client = _CaptureClient(["finish conversation"])
        monkeypatch.setattr(modules.step_agent, "OpenAI", lambda **unused: client)
        agent = modules.step_agent.APIAgent_step(
            "served-model", "", [], temperature=0.0, top_p=1.0,
            max_tokens=16, language="en", task_id="agent_multi_step_1",
        )
        agent.respond(history)
        request = client.calls[0]
        assert request["extra_body"]["c2kv_ace_source"] == {
            "version": "acebench-text-actions-v1",
            "receipts": [execution["c2kv_acebench_execution"]],
        }
        assert all(
            set(message) == ({"role", "content", "tool_call_id"}
                             if message["role"] == "tool"
                             else {"role", "content"})
            for message in request["messages"]
        )
        assert all("tool_calls" not in message for message in request["messages"])


def test_step_decode_error_and_turn_nonlist_or_decode_exception_preserve_behavior(
    patched_acebench_receipts, monkeypatch,
):
    _enable_text_actions(monkeypatch)
    with _ace_modules(patched_acebench_receipts) as modules:
        step = modules.step_execution.EXECUTION_STEP("m", {}, [], "1", "en")
        monkeypatch.setattr(step, "decode_function_list", lambda unused: (_ for _ in ()).throw(ValueError("bad")))
        monkeypatch.setattr(
            modules.step_execution, "execute_agent_func_call",
            lambda **unused: pytest.fail("step executor must not run after decode failure"),
        )
        history = _history()
        step_message, step_instances = step.respond(history)
        modules.history.bind_execution_receipt(history, step_message)
        assert step_instances == {}
        assert step_message["message"] == "Please do not ask me any questions, use the known conditions to solve the problem"
        assert step_message["c2kv_acebench_execution"] == {
            "version": "acebench-execution-receipt-v1",
            "agent_history_index": 1,
            "execution_message_index": 3,
            "decode_status": "error",
            "decoded_calls": None,
            "executor_status": "not_called",
            "executor_return_shape": None,
            "executor_return_count": None,
        }

        turn = modules.turn_execution.EXECUTION("m", {}, [], "1", "en")
        monkeypatch.setattr(
            modules.turn_execution, "execute_agent_func_call",
            lambda **unused: ("bad", {}),
        )
        turn_message, turn_instances = turn.respond(_history())
        assert turn_instances == {}
        assert turn_message["message"] == ["b", "a", "d"]
        turn_history = _history()
        modules.history.bind_execution_receipt(turn_history, turn_message)
        assert turn_message["c2kv_acebench_execution"] == {
            "version": "acebench-execution-receipt-v1",
            "agent_history_index": 1,
            "execution_message_index": 3,
            "decode_status": "ok",
            "decoded_calls": ["Echo(value='x')"],
            "executor_status": "returned",
            "executor_return_shape": "non_list",
            "executor_return_count": None,
        }

        propagating_turn = modules.turn_execution.EXECUTION("m", {}, [], "1", "en")
        monkeypatch.setattr(
            propagating_turn, "decode_function_list",
            lambda unused: (_ for _ in ()).throw(ValueError("bad")),
        )
        monkeypatch.setattr(
            modules.turn_execution, "execute_agent_func_call",
            lambda **unused: pytest.fail("turn executor must not run after decode failure"),
        )
        with pytest.raises(ValueError, match="bad"):
            propagating_turn.respond(_history())


def test_flag_dependencies_initial_empty_source_and_user_requests_remain_unaffected(
    patched_acebench_receipts, monkeypatch,
):
    with _ace_modules(patched_acebench_receipts) as modules:
        monkeypatch.setenv("ACEBENCH_TEXT_ACTIONS_V1", "1")
        monkeypatch.delenv("ACEBENCH_ROLE_HISTORY_V1", raising=False)
        monkeypatch.delenv("ACEBENCH_EVENT_NATIVE_V1", raising=False)
        with pytest.raises(ValueError, match="requires ACEBENCH_ROLE_HISTORY_V1=1"):
            modules.history.agent_request([], "m", 0.0, 4, 1.0, {"task_id": "t"})

        _enable_text_actions(monkeypatch)
        initial = modules.history.agent_request(
            [{"role": "user", "content": "initial"}], "m", 0.0, 4, 1.0,
            {"task_id": "t"},
        )
        assert initial["extra_body"]["c2kv_ace_source"] == {
            "version": "acebench-text-actions-v1", "receipts": [],
        }

        monkeypatch.setenv("ACEBENCH_USER_BASE_URL", "http://user.invalid/v1")
        client = _CaptureClient(["first user response", "second user response"])
        monkeypatch.setattr(modules.user, "OpenAI", lambda **unused: client)
        user = modules.user.APIUSER("served-user", ["BaseApi"], language="en")
        user.get_init_prompt("task")
        user.step("agent:question")
        user.respond()
        assert len(client.calls) == 2
        assert all("extra_body" not in call for call in client.calls)
