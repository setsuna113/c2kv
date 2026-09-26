"""New encoder/selector combinations through the actual recovery controller."""
import json

import pytest

from benchmarks.memory_runtime.tests.test_gp_recovery import make_controller
from benchmarks.memory_runtime.tests.test_event_native_recovery import request, draft_call


@pytest.mark.parametrize("scope", ["record_bound", "record_bound_structural"])
def test_bound_encoding_llm_fields_append_and_expire(scope):
    controller = make_controller(G=scope, U="field", R=3, selector="llm",
        selector_catalog="retrieved_fields", selector_max_units=4,
        backend={"type": "openai_compatible", "base_url": "http://unused.invalid/v1",
                 "chat_model": "cpu-fixture"})
    calls = []

    def chat(*, messages, purpose, config):
        payload = json.loads(messages[-1]["content"])
        catalog = payload["catalog"]
        assert len(catalog) <= 32
        calls.append(catalog)
        return {"selected_ids": [catalog[0]["unit_id"]]}

    controller.backends = {"chat": chat}
    prepared = controller.prepare(request(), ratio=4, max_new_tokens=32)
    original = prepared.memory.chunks
    result = controller.reconsider(prepared, [draft_call()], draft_text="lookup violet item-17")
    assert calls and result["regenerate"]
    assert not result["decision"]["selection"]["selector_fallback"]["applied"]
    assert result["memory"].chunks == original
    assert result["decision"]["selected_unit_count"] == 1
    assert result["metadata"]["actual_history_bytes"] <= controller.policy_config.history_budget_bytes
    after = controller.prepare(request("d2"), ratio=4, max_new_tokens=32)
    assert after.metadata["gp_lifecycle"]["protected_unit_ids"] == []
