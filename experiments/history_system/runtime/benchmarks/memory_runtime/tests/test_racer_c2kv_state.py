"""C2KV shares residency without replacing RACER's original initial policy."""
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "python")]

from benchmarks.memory_runtime.event_native_always import NATIVE_S0_MODE
from benchmarks.memory_runtime.event_native_controls import build_event_native_controller
from benchmarks.memory_runtime.event_native_s0_policy import S0_CONFIG_DEFAULTS
from benchmarks.memory_runtime.racer.c2kv_state import C2KVResidentPolicy
from benchmarks.memory_runtime.racer.config import BackendConfig
from benchmarks.memory_runtime.tests.test_candidate_allocation import Tokenizer, messages, packing, policy
from history_memory.events import EventStore
from history_memory.packing import pack_memory, select_view


def backend():
    return BackendConfig.parse({"schema": "racer-backend-v1", "backend": "c2kv",
        "policy": "off", "history_budget_tokens": 1024,
        "detector_calibration": "not_used", "allocation": "c2kv_s0"})


def test_v4_flags_report_intrinsic_s0_without_another_c2kv_policy():
    for protection in ("off", "on"):
        selected = BackendConfig.parse({
            "schema": "racer-backend-v4", "backend": "c2kv", "policy": "off",
            "history_budget_tokens": 1024, "backend_config": {"method": "c2kv"},
            "detector_calibration": "not_used", "allocation": "c2kv_s0",
            "extra_protection": protection,
        })
        receipt = C2KVResidentPolicy(SimpleNamespace(), selected)._receipt(
            "s", None, "prepare", "configured_c2kv_policy")
        assert receipt["extra_protection"] == protection
        assert receipt["extra_protection_status"] == "intrinsic_c2kv_s0_preserved"
        assert receipt["initial_allocation"] == "configured_c2kv_policy"


def test_factory_preserves_s0_initial_memory():
    geometry = packing()
    geometry["ratios"] = [8]
    kwargs = dict(packing=geometry, policy=policy(1024), view_mode=NATIVE_S0_MODE,
                  compression_policy="always-compress-v1", benchmark="bfcl")
    wrapped = build_event_native_controller(Tokenizer(), **kwargs, s0_config={**S0_CONFIG_DEFAULTS,
        "racer_backend": {"schema": "racer-backend-v1", "backend": "c2kv",
        "policy": "off", "history_budget_tokens": 1024,
        "detector_calibration": "not_used", "allocation": "c2kv_s0"}})
    original = build_event_native_controller(Tokenizer(), **kwargs, s0_config=S0_CONFIG_DEFAULTS)
    payload = {"session_id": "s", "decision_key": "0", "messages": messages(), "tools": []}
    prepared = wrapped.prepare(payload, ratio=8, max_new_tokens=32)
    expected = original.prepare(payload, ratio=8, max_new_tokens=32)
    assert isinstance(wrapped, C2KVResidentPolicy)
    assert prepared.memory == expected.memory
    assert prepared.eligible_chunks == expected.eligible_chunks
    state = prepared.metadata["c2kv_resident_state"]
    assert state["backend_allocation"] == "c2kv_s0"
    assert state["initial_allocation"] == "configured_c2kv_policy"
    assert state["initial_policy_route"] == expected.metadata["route"]["baseline_identity"]
    assert wrapped.commit_memory(prepared, prepared.memory)["status"] == "committed"


def test_only_explicit_racer_selection_can_re_admit_old_fragments():
    store = EventStore.from_messages("s", messages(), benchmark="bfcl")
    memory = pack_memory(store, select_view(store, recent_tool_events=1), Tokenizer(),
                         max_chunk_tokens=768, chunk_overlap=64)
    assert len(memory.chunks) >= 2
    selected = memory.chunks[0]
    old = tuple(memory.chunks[1:])

    class Policy:
        benchmark = "bfcl"

        def prepare(self, payload, **kwargs):
            key = payload["decision_key"]
            chunks = (selected,) if key != "2" else (selected, old[0])
            return SimpleNamespace(memory=replace(memory, chunks=chunks),
                eligible_chunks=memory.chunks,
                metadata={"session_id": "s", "decision_key": key,
                          "route": {"baseline_identity": "candidate:test"}})

    controller = C2KVResidentPolicy(Policy(), backend())
    payload = {"session_id": "s", "decision_key": "0", "messages": messages()}
    first = controller.prepare(payload, ratio=8, max_new_tokens=32)
    assert first.metadata["c2kv_resident_state"]["initial_policy_route"] == "candidate:test"
    assert controller.commit_memory(first, first.memory)["initial_policy_route"] == "candidate:test"
    second = controller.prepare({**payload, "decision_key": "1"}, ratio=8, max_new_tokens=32)
    assert second.eligible_chunks == (selected,)
    controller.commit_memory(second, second.memory)
    third = controller.prepare({**payload, "decision_key": "2"}, ratio=8, max_new_tokens=32)
    assert third.eligible_chunks == (selected, old[0])
    admitted = third.metadata["c2kv_resident_state"]["newly_admitted_old_fragments"]
    assert len(admitted) == 1
    assert admitted[0]["event_id"] == old[0].event_id


def test_execution_metadata_maps_partial_encoder_units_to_source_events():
    store = EventStore.from_messages("s", messages(), benchmark="bfcl")
    memory = pack_memory(store, select_view(store, recent_tool_events=1), Tokenizer(),
                         max_chunk_tokens=4, chunk_overlap=1)
    source_event = next(event for event in store.events
                        if len(event.source_indices) == 2
                        and any(chunk.event_id == event.event_id for chunk in memory.chunks))
    assert len(source_event.source_indices) == 2
    source_chunks = [chunk for chunk in memory.chunks if chunk.event_id == source_event.event_id]
    assert len(source_chunks) > 1
    first = replace(source_chunks[0], event_id="synthetic-record:0")
    second = replace(source_chunks[1], event_id="synthetic-record:1")
    assert first.source_indices == second.source_indices == source_event.source_indices

    class Policy:
        benchmark = "bfcl"

        def prepare(self, payload, **kwargs):
            return SimpleNamespace(
                memory=replace(memory, chunks=(first,)),
                eligible_chunks=(first, second),
                metadata={"eligible_extraction": {
                    "encoding_event_groups": [[source_event.event_id]],
                    "eligible_source_indices": [source_event.source_indices[0]],
                }},
            )

    controller = C2KVResidentPolicy(Policy(), backend())
    payload = {"session_id": "s", "decision_key": "0", "messages": messages()}
    first_prepared = controller.prepare(payload, ratio=8, max_new_tokens=32)
    controller.commit_memory(first_prepared, first_prepared.memory)
    second_prepared = controller.prepare({**payload, "decision_key": "1"},
                                         ratio=8, max_new_tokens=32)
    assert second_prepared.eligible_chunks == (first,)
    execution = second_prepared.metadata["eligible_extraction"]
    assert execution["eligible_event_ids"] == [source_event.event_id]
    assert execution["eligible_encoder_unit_ids"] == ["synthetic-record:0"]
    assert execution["encoding_event_groups"] == [[source_event.event_id]]
    assert execution["eligible_source_indices"] == [source_event.source_indices[0]]
    assert execution["whole_event_encoded_source_indices"] == list(source_event.source_indices)
