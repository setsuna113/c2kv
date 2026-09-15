"""CPU-only parity checks for the admitted ACEBench text-action grammar."""

from __future__ import annotations

import importlib
import json
from types import SimpleNamespace

import pytest

from benchmarks.memory_runtime.acebench_source import parse_ace_draft
from benchmarks.test_acebench_event_native_identity import _ace_modules


pytest_plugins = ("benchmarks.test_acebench_event_native_identity",)


class _CaptureClient:
    """Return fixed text to an official agent without reaching any API."""

    def __init__(self, content: str) -> None:
        self.content = content
        self.calls: list[dict] = []
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content=self.content)
        )])


def _normalised_calls(draft) -> list[tuple[str, dict]]:
    assert draft.status == "tool_calls"
    return [
        (call["function"]["name"], json.loads(call["function"]["arguments"]))
        for call in draft.tool_calls
    ]


def _normalised_official_calls(decoded_calls: list[str]) -> list[tuple[str, dict]]:
    """Read official execution submissions through the same safe grammar.

    ``decoded_calls`` is text emitted by the pinned decoder, not executable
    input.  This deliberately avoids the official resolver's unsafe branches.
    """

    decoded_draft = parse_ace_draft(
        "[" + ",".join(decoded_calls) + "]", call_id_prefix="official"
    )
    return _normalised_calls(decoded_draft)


def _step_router(modules, content: str) -> tuple[dict, _CaptureClient]:
    client = _CaptureClient(content)
    agent = object.__new__(modules.step.APIAgent_step)
    agent.client = client
    agent.language = "en"
    agent.model_name = "parity-safe-model"
    agent.temperature = 0.0
    agent.top_p = 1.0
    agent.max_tokens = 1
    agent.time = ""
    agent.task_id = "parser-parity"
    agent.functions = []
    return agent.respond(_ROUTING_HISTORY), client


def _turn_router(modules, content: str) -> tuple[dict, _CaptureClient]:
    client = _CaptureClient(content)
    agent = object.__new__(modules.turn.APIAgent_turn)
    agent.client = client
    agent.language = "en"
    agent.model_name = "parity-safe-model"
    agent.temperature = 0.0
    agent.top_p = 1.0
    agent.max_tokens = 1
    agent.task_id = "parser-parity"
    agent.functions = []
    agent.involved_class = ()
    return agent.respond(_ROUTING_HISTORY), client


_SAFE_DRAFTS = (
    "[Lookup(text=\"O'Brien\", quoted='say \"hello\"', unicode='雪', "
    "escaped='line\\\\nslash\\\\\\\\tab\\\\t', enabled=True, empty=False, "
    "missing=None, count=7, ratio=1.25, neg_count=-3, neg_ratio=-2.5, "
    "nested=['x', [True, None, -4, 0.5]])]",
    "[First(label='alpha'), Second(enabled=False, "
    "items=['β', -1, [None, 2.0]], note='\"quoted\"')]",
)

_ROUTING_HISTORY = [
    {"sender": "user", "recipient": "agent", "message": "Use a safe action."},
]


@pytest.mark.parametrize("text", _SAFE_DRAFTS)
def test_safe_drafts_match_pinned_step_turn_decoders_and_routers(
    patched_acebench, text: str,
) -> None:
    """Compare only the common safe literal subset; never call an executor."""

    internal = parse_ace_draft(text, call_id_prefix="internal")
    assert internal.text == internal.content == text
    expected = _normalised_calls(internal)

    with _ace_modules(patched_acebench) as modules:
        step_execution = importlib.import_module(
            "model_inference.multi_step.execution_role_step"
        )
        step_decoder = object.__new__(step_execution.EXECUTION_STEP)
        turn_decoder = object.__new__(modules.turn.APIAgent_turn)

        step_decoded = step_decoder.decode_function_list(text)
        turn_decoded = turn_decoder.decode_function_list(text)
        assert step_decoded == turn_decoded
        assert _normalised_official_calls(step_decoded) == expected

        step_message, step_client = _step_router(modules, text)
        turn_message, turn_client = _turn_router(modules, text)
        assert step_client.calls and turn_client.calls
        assert step_message == {
            "sender": "agent", "recipient": "execution", "message": text,
        }
        assert turn_message == {
            "sender": "agent", "recipient": "execution", "message": text,
        }


@pytest.mark.parametrize(
    "text",
    (
        "[Lookup(total=1 + 2)]",
        "[Lookup(value=Nested())]",
        "[Lookup(value=lambda: 1)]",
        "[Lookup(value=items[0])]",
        "[Lookup(value={'key': 'value'})]",
        "[Lookup(value=name)]",
    ),
)
def test_unsupported_ast_never_admits_partial_calls(text: str) -> None:
    """Unsafe official-resolver branches are intentionally not differentially run."""

    draft = parse_ace_draft(text, call_id_prefix="internal")
    assert draft.status == "malformed"
    assert draft.tool_calls == ()
    assert draft.text == draft.content == text
