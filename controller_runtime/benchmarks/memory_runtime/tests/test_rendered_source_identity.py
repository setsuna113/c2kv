"""The bundled NPU controller separates source identity from tool rendering."""
import copy
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "python")]

from history_memory.events import RenderedMessages
from benchmarks.memory_runtime.event_native_s0_policy import EventNativeS0Controller
from benchmarks.memory_runtime.event_native_policy import PolicyInputError
from benchmarks.memory_runtime.tests.test_candidate_allocation import Tokenizer, packing, policy


def test_layout_change_keeps_source_guard_and_rendered_decision_cache():
    controller = EventNativeS0Controller(Tokenizer(), packing=packing(), policy=policy())
    source = [{"role": "system", "content": "Use tools"},
              {"role": "user", "content": "Find files"}]
    first = {"session_id": "hybrid", "decision_key": "d1", "tools": [],
             "messages": RenderedMessages(source, source=source)}
    controller.prepare(first, ratio=4, max_new_tokens=4)
    source += [{"role": "assistant", "content": "Done"},
               {"role": "user", "content": "Weather forecast"}]
    rendered = copy.deepcopy(source)
    rendered[0]["content"] += "\nWeather schema"
    second = dict(first, decision_key="d2", messages=RenderedMessages(rendered, source=source))
    prepared = controller.prepare(second, ratio=4, max_new_tokens=4)
    assert prepared.metadata["decision_index"] == 2
    rendered[0]["content"] += " different view"
    with pytest.raises(PolicyInputError, match="reused with different input"):
        controller.prepare(dict(second, messages=RenderedMessages(rendered, source=source)),
                           ratio=4, max_new_tokens=4)
    source[0]["content"] += " changed source"
    with pytest.raises(PolicyInputError, match="truncated or rewritten"):
        controller.prepare(dict(second, decision_key="d3",
                                messages=RenderedMessages(rendered, source=source)),
                           ratio=4, max_new_tokens=4)
