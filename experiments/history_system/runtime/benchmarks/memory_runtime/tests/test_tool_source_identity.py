"""Dynamic tool layouts must not masquerade as edits to source history."""
import copy
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "python")]

from history_memory.events import EventStore, RenderedMessages
from benchmarks.memory_runtime.event_native_controls import EventNativeOnePassController
from benchmarks.memory_runtime.event_native_policy import PolicyInputError
from benchmarks.memory_runtime.event_native_s0_policy import EventNativeS0Controller
from benchmarks.memory_runtime.event_native_tool import ToolRegionController, parse_native_tool_spec
from benchmarks.memory_runtime.tests.test_d3_hybrid_recovery import packing_config, policy_config
from benchmarks.memory_runtime.tests.test_event_native_tool import ToolAwareTokenizer, RecordingGenerator


def request():
    return {
        "session_id": "bfcl/hybrid-source", "decision_key": "turn-0/step-0",
        "messages": [{"role": "system", "content": "Use the tools."},
                     {"role": "user", "content": "find files"}],
        "tools": [{"type": "function", "function": {
            "name": name, "description": description,
            "parameters": {"type": "object", "properties": {}}}}
            for name, description in (("find", "find files"), ("weather", "weather forecast"))],
    }


@pytest.mark.parametrize("mode", ["s0", "static", "full_original"])
def test_hybrid_reranks_across_turns_without_rewriting_source(mode):
    tokenizer = ToolAwareTokenizer()
    if mode == "s0":
        inner = EventNativeS0Controller(tokenizer, packing=packing_config(), policy=policy_config())
    else:
        inner = EventNativeOnePassController(tokenizer, packing=packing_config(),
                                            policy=policy_config(), view_mode=mode)
    controller = ToolRegionController(inner, tokenizer, parse_native_tool_spec("t0:r8:hybrid1"),
                                      model_context=100_000, generator=RecordingGenerator())
    first = request()
    p1 = controller.prepare(first, ratio=4, max_new_tokens=4)
    second = copy.deepcopy(first)
    second["decision_key"] = "turn-1/step-0"
    second["messages"] += [{"role": "assistant", "content": "Found them."},
                            {"role": "user", "content": "weather forecast"}]
    p2 = controller.prepare(second, ratio=4, max_new_tokens=4)
    assert p1.plan.info["native_indices"] == [0]
    assert p2.plan.info["native_indices"] == [1]
    assert p1.memory.system_input_ids != p2.memory.system_input_ids
    assert p2.metadata["decision_index"] == 2
    assert controller.prepare(second, ratio=4, max_new_tokens=4).inner is p2.inner
    controller.spec = parse_native_tool_spec("t0:r8")
    with pytest.raises(PolicyInputError, match="reused with different input"):
        controller.prepare(second, ratio=4, max_new_tokens=4)
    controller.spec = parse_native_tool_spec("t0:r8:hybrid1")
    changed = copy.deepcopy(second)
    changed["decision_key"] = "turn-2/step-0"
    changed["messages"][0]["content"] = "Different source instruction."
    with pytest.raises(PolicyInputError, match="truncated or rewritten"):
        controller.prepare(changed, ratio=4, max_new_tokens=4)
    truncated = copy.deepcopy(first)
    truncated["decision_key"] = "turn-3/step-0"
    with pytest.raises(PolicyInputError, match="truncated or rewritten"):
        controller.prepare(truncated, ratio=4, max_new_tokens=4)


def test_source_snapshot_is_owned_and_not_a_wire_annotation():
    source = request()["messages"]
    rendered = RenderedMessages(source, source=source)
    store = EventStore.from_messages("source", rendered)
    source[0]["content"] = "changed after snapshot"
    assert "changed after snapshot" not in store.source_prefix[0]
    ordinary = EventStore.from_messages("source", list(rendered))
    assert ordinary.source_prefix is None
