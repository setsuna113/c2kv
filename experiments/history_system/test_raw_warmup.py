"""Test full-raw warmup against the actual native packer and budget guard."""
import copy
from dataclasses import replace
import pytest

from benchmarks.memory_runtime import event_native
from benchmarks.memory_runtime.always_compress import ALWAYS_COMPRESSION_POLICY
from benchmarks.memory_runtime.event_native_always import NATIVE_S0_MODE
from benchmarks.memory_runtime.event_native_controls import build_event_native_controller
from benchmarks.memory_runtime.raw_warmup import RAW_WARMUP_POLICY, apply_raw_warmup
from benchmarks.memory_runtime.budget_guard import history_budget_receipt
from benchmarks.memory_runtime.tests import test_same_event_bridge_only as fixture
from benchmarks.memory_runtime.result_key_bridge import RESULT_KEY_BRIDGE_POLICY


def owner(bridge_policy=fixture.SAME_EVENT_BRIDGE_ONLY_POLICY):
    config = fixture.s0_config()
    config.update(observed_entity_slot_policy=bridge_policy,
                  raw_warmup_policy=RAW_WARMUP_POLICY)
    return build_event_native_controller(fixture.Tokenizer(), packing=fixture.packing(), policy=fixture.policy(),
        view_mode=NATIVE_S0_MODE, compression_policy=ALWAYS_COMPRESSION_POLICY, s0_config=config)


@pytest.mark.parametrize("bridge_policy", [fixture.SAME_EVENT_BRIDGE_ONLY_POLICY, RESULT_KEY_BRIDGE_POLICY])
def test_warmup_preserves_all_history_raw_and_still_compresses_eligible_sources(bridge_policy):
    controller = owner(bridge_policy)
    prepared = fixture.prepare(controller, "opaque-reference", {"id":"opaque-reference", "amount":"10"})
    assert prepared.metadata["raw_warmup"]["status"] == "full_raw_admitted"
    assert prepared.memory.chunks == () and len(prepared.eligible_chunks) > 0
    assert prepared.metadata["source_coverage"]["complete_history_coverage"] is True
    assert prepared.metadata["compression_ratio"]["n_history"] == 1.0
    assert prepared.metadata["eligible_extraction"]["retained_chunk_count"] == 0
    assert prepared.metadata["eligible_extraction"]["backend_execution_required"] is True
    assert history_budget_receipt(prepared.memory, prepared.metadata, controller, ratio=4, phase="draft")["status"] == "passed"


def test_over_budget_warmup_returns_original_s0_memory_without_displacement():
    controller = fixture.owner(candidate=True)
    prepared = fixture.prepare(controller, "opaque-reference", {"id":"opaque-reference", "amount":"10"})
    messages = fixture.messages({"order_id":"opaque-reference"}, {"id":"opaque-reference", "amount":"10"})
    store = event_native.EventStore.from_messages("bridge-only/session", messages)
    full = prepared.metadata["same_prefix_full_reference"]["full_history_bytes"]
    controller.policy_config = replace(controller.policy_config, history_budget_bytes=full - 1, workspace_budget_bytes=full - 1)
    candidate = apply_raw_warmup(controller, prepared, store, tuple(copy.deepcopy(fixture.TOOLS)), ratio=4, max_new_tokens=8)
    assert candidate.memory is prepared.memory
    assert candidate.metadata["raw_warmup"]["status"] == "full_history_over_budget"


def test_result_key_remains_admitted_after_warmup_exceeds_budget():
    from benchmarks.memory_runtime.tests import test_result_key_bridge as direct
    result = {"booking_id":"opaque-generated-reference", "unrelated_payload":"x" * 5000}
    base = direct.prepare(direct.owner(), result)
    full = base.metadata["same_prefix_full_reference"]["full_history_bytes"]
    assert base.metadata["actual_history_bytes"] < full
    config = direct.s0_config(RESULT_KEY_BRIDGE_POLICY)
    config["raw_warmup_policy"] = RAW_WARMUP_POLICY
    controller = build_event_native_controller(direct.Tokenizer(), packing=direct.packing(),
        policy=direct.policy(budget=full - 1), view_mode=NATIVE_S0_MODE,
        compression_policy=ALWAYS_COMPRESSION_POLICY, s0_config=config)
    candidate = direct.prepare(controller, result)
    assert candidate.metadata["raw_warmup"]["status"] == "full_history_over_budget"
    assert candidate.metadata["same_event_reference"]["status"] == "admitted"
    assert candidate.metadata["same_event_reference"]["relation"] == "direct_result_key_required_consumer"
    assert history_budget_receipt(candidate.memory, candidate.metadata, controller, ratio=4, phase="draft")["status"] == "passed"
