import copy
import json
import sys
from types import SimpleNamespace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "benchmarks"))
from memory_runtime.adapter import RuntimeAdapter
from history_memory.events import EventStore
from history_memory.evidence import evidence_message
from memory_runtime.tokenization import serving_tools


def count(messages, tools):
    return len(json.dumps(messages, separators=(",", ":")))


def adapter(mode, history=5000, workspace=2500):
    return RuntimeAdapter(dict(mode=mode, run_id="test", bytes_per_kv_token=1,
                               history_budget_bytes=history, workspace_budget_bytes=workspace), count)


def context(task="t", decision="d0"):
    return dict(task_id=task, attempt=0, decision_id=decision)


def prefix():
    messages = [
        {"role": "user", "content": "Read id-17 and keep the number"},
        {"role": "assistant", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "read", "arguments": '{"id":"id-17"}'}}]},
        {"role": "tool", "tool_call_id": "c1", "content": '{"number":17}'},
    ]
    assembled = [{"role": "system", "content": "system"},
                 {"role": "user", "content": "old", "c2kv_key_hash": "g0"},
                 {"role": "user", "content": messages[-1]["content"]}]
    counts = dict(current_start_out_index=2, current_raw=1, history_raw=0, system_raw=1,
                  gist_tokens=300, n_docs=1, dropped_docs=0, history_packed_original_tokens=1200,
                  compressed_records=[dict(out_index=1, source_indices=[1, 2], record=dict(key_hash="g0", gist_len=300, original_seq_len=1200))])
    return messages, assembled, counts


@pytest.mark.parametrize("mode", ["legacy", "protect", "recover_once", "persistent", "no_gist", "full_shared"])
def test_first_user_with_no_history_is_exact_identity(mode):
    source = [{"role": "user", "content": "hello"}]
    assembled = [{"role": "system", "content": "system"}] + source
    counts = dict(current_start_out_index=1, compressed_records=[])
    out, result = adapter(mode).apply(source, assembled, counts, context(), [])
    assert out == assembled
    assert result["memory_runtime"]["active_history_bytes"] == 0
    assert result["memory_runtime"]["selected_event_ids"] == []


def test_evidence_preserves_call_binding_and_parses_json_arguments_once():
    source, _, _ = prefix()
    untouched = copy.deepcopy(source)
    packet = evidence_message(EventStore.from_messages("t", source), ["t:m1"])
    content = json.loads(packet["content"].split("\n", 1)[1])
    event = content["events"][0]
    assert event["source_indices"] == [1, 2]
    assert event["messages"][0]["tool_calls"][0]["function"]["arguments"] == {"id": "id-17"}
    assert event["messages"][1]["tool_call_id"] == "c1"
    assert event["messages"][1]["role"] == "tool"
    assert source == untouched


def test_consecutive_current_inputs_are_already_raw_and_not_duplicated():
    source = [{"role": "user", "content": "first"}, {"role": "user", "content": "amendment"}]
    assembled = [{"role": "system", "content": "system"}] + source
    out, result = adapter("no_gist").apply(source, assembled, dict(current_start_out_index=1), context(), [])
    assert out == assembled
    assert result["memory_runtime"]["selected_event_ids"] == []


def test_raw_recency_requires_explicit_full_renderer():
    source, assembled, counts = prefix()
    with pytest.raises(ValueError, match="existing Full renderer"):
        adapter("raw_recency").apply(source, assembled, counts, context(), [])


def test_capacity_gate_keeps_exact_full_at_boundary_and_requires_lazy_compression_above():
    from memory_runtime.capacity import measure_full_history

    source, _, _ = prefix()
    full = [{"role": "system", "content": "system"}, source[0],
            {"role": "assistant", "content": "Action: read(id-17)"},
            {"role": "user", "content": source[-1]["content"]}]
    counts = dict(current_start_out_index=3, compressed_records=[])
    budget = measure_full_history(full, counts, count, [], 1)["active_history_bytes"]

    def forbidden_compression(_):
        raise AssertionError("No compressed assembly is allowed within budget")

    out, result = adapter("capacity_protect", history=budget).apply(
        source, full, counts, context(), [], render_compressed=forbidden_compression)
    assert out == full
    assert result["memory_runtime"]["capacity_gate"]["compression_activated"] is False
    assert result["memory_runtime"]["active_history_bytes"] == budget
    assert result["memory_runtime"]["compressed_assembly_wall_sec"] == 0
    with pytest.raises(ValueError, match="lazy compressed renderer above budget"):
        adapter("capacity_protect", history=budget - 1).apply(
            source, full, counts, context(), [])


def test_capacity_gate_above_budget_reuses_the_existing_protection_view():
    source, compressed, compressed_counts = prefix()
    source = [{"role": "user", "content": "earlier work"},
              {"role": "assistant", "content": "large earlier answer " * 1000}] + source
    full = [{"role": "system", "content": "system"}] + source[:3] + [
        {"role": "assistant", "content": "Action: read(id-17)"},
        {"role": "user", "content": source[-1]["content"]}]
    compressed_counts = copy.deepcopy(compressed_counts)
    compressed_counts["compressed_records"][0]["source_indices"] = [1, 2, 3, 4]
    calls = []

    def compress(messages):
        calls.append(copy.deepcopy(messages))
        return copy.deepcopy(compressed), copy.deepcopy(compressed_counts)

    out, result = adapter("capacity_protect", history=3000).apply(
        source, full, dict(current_start_out_index=5), context(), [], render_compressed=compress)
    expected, expected_counts = adapter("protect", history=3000).apply(
        source, compressed, compressed_counts, context(), [])
    assert calls == [source]
    assert out == expected
    assert result["memory_runtime"]["capacity_gate"]["compression_activated"] is True
    assert result["memory_runtime"]["active_history_bytes"] == expected_counts["memory_runtime"]["active_history_bytes"]
    assert result["memory_runtime"]["active_history_bytes"] <= 3000
    assert result["memory_runtime"]["mode"] == "capacity_protect"


def test_full_capacity_aux_preserves_full_below_cap_and_matches_protection_above():
    source, compressed, compressed_counts = prefix()
    full = [{"role": "system", "content": "system"}, source[0],
            {"role": "assistant", "content": "Action: read(id-17)"},
            {"role": "user", "content": source[-1]["content"]}]
    out, result = adapter("full_capacity_aux").apply(
        source, full, dict(current_start_out_index=3), context(), [])
    assert out == full
    assert result["memory_runtime"]["auxiliary_gate"]["auxiliary_activated"] is False
    assert result["memory_runtime"]["evidence_bytes"] == 0
    assert result["memory_runtime"]["budget_applies"] is False

    source = [{"role": "user", "content": "earlier work"},
              {"role": "assistant", "content": "large earlier answer " * 1000}] + source
    full = full[:1] + source[:2] + full[1:]
    compressed_counts = copy.deepcopy(compressed_counts)
    compressed_counts["compressed_records"][0]["source_indices"] = [1, 2, 3, 4]
    untouched = copy.deepcopy(full)
    protection, protection_counts = adapter("protect", history=3000).apply(
        source, compressed, compressed_counts, context(), [])
    runtime = adapter("full_capacity_aux", history=3000)
    out, result = runtime.apply(source, full, dict(current_start_out_index=5), context(), [])
    meta = result["memory_runtime"]
    protected_meta = protection_counts["memory_runtime"]
    evidence_index = meta["evidence_out_index"]
    assert meta["auxiliary_gate"]["auxiliary_activated"] is True
    assert meta["selected_event_ids"] == protected_meta["selected_event_ids"]
    assert out[evidence_index] == protection[protected_meta["evidence_out_index"]]
    assert out[:evidence_index] + out[evidence_index + 1:] == full == untouched
    assert meta["active_history_bytes"] > 3000
    assert meta["budget_applies"] is False
    assert meta["workspace_budget_applies"] is True
    assert meta["evidence_bytes"] <= 2500
    assert meta["retrieved_event_ids"] == meta["retained_event_ids"] == []
    assert meta["gist_tokens"] == 0
    assert result["compressed_records"] == []
    assert runtime.apply(source, full, dict(current_start_out_index=5), context(), [])[0] == out


def test_full_capacity_aux_measures_actual_insertion_cost_beyond_selection_reference():
    source, _, _ = prefix()
    source = [{"role": "assistant", "content": "historical marker " * 1000}] + source
    full = [{"role": "system", "content": "system"}] + source[:-2] + [
        {"role": "assistant", "content": "Action: read(id-17)"},
        {"role": "user", "content": source[-1]["content"]}]

    def nonadditive_count(messages, tools):
        text = json.dumps(messages)
        return count(messages, tools) + (
            10000 if "historical marker" in text and "source_indices" in text else 0)

    runtime = RuntimeAdapter(dict(mode="full_capacity_aux", run_id="test",
                                  bytes_per_kv_token=1, history_budget_bytes=3000,
                                  workspace_budget_bytes=2500), nonadditive_count)
    with pytest.raises(ValueError, match="measured workspace byte cap"):
        runtime.apply(source, full, dict(current_start_out_index=4), context(), [])


def test_whole_packet_is_counted_and_coarse_overlap_is_kept_until_budget_pressure():
    source, assembled, counts = prefix()
    out, result = adapter("protect").apply(source, assembled, counts, context(), [])
    meta = result["memory_runtime"]
    raw = [m for m in out if not m.get("c2kv_key_hash")]
    common = [assembled[0], assembled[2]]
    assert meta["evidence_bytes"] == count(raw, []) - count(common, [])
    assert meta["active_history_bytes"] == meta["evidence_bytes"] + 300
    assert meta["overlapping_gist_keys"] == ["g0"]
    assert [m["c2kv_key_hash"] for m in out if m.get("c2kv_key_hash")] == ["g0"]

    out2, result2 = adapter("protect", history=meta["evidence_bytes"] + 299).apply(source, assembled, counts, context(), [])
    assert result2["memory_runtime"]["evicted_gist_keys"] == ["g0"]
    assert result2["memory_runtime"]["active_history_bytes"] == meta["evidence_bytes"]
    assert not any(m.get("c2kv_key_hash") for m in out2)


def test_nogist_discards_full_raw_prefix_and_spends_released_history_budget():
    source, _, _ = prefix()
    full = [{"role": "system", "content": "system"}, source[0],
            {"role": "assistant", "content": "Action: read(id-17)"},
            {"role": "user", "content": source[2]["content"]}]
    out, result = adapter("no_gist", workspace=1).apply(source, full, dict(current_start_out_index=3, compressed_records=[]), context(), [])
    meta = result["memory_runtime"]
    assert meta["active_history_bytes"] > 1
    assert meta["active_history_bytes"] <= 5000
    assert meta["selected_event_ids"] == ["t:m0", "t:m1"]
    assert out[0] == full[0] and out[-1] == full[-1]
    assert len(out) == 3


def test_explicit_attempt_is_required_and_tasks_do_not_share_state():
    source, assembled, counts = prefix()
    runtime = adapter("persistent")
    with pytest.raises(ValueError, match="explicit attempt"):
        runtime.apply(source, assembled, counts, {"task_id": "t"}, [])
    runtime.apply(source, assembled, counts, context(), [])
    runtime.apply(source, assembled, counts, context(task="another"), [])
    assert len(runtime._states) == 2


def test_real_token_counter_reads_batchencoding_input_ids(tmp_path, monkeypatch):
    class Encoding(dict):
        def __init__(self):
            super().__init__(input_ids=list(range(41)), attention_mask=[1] * 41)
            self.input_ids = self["input_ids"]

    tokenizer = SimpleNamespace(apply_chat_template=lambda *a, **kw: Encoding())
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **kw: tokenizer)))
    config = dict(mode="protect", run_id="test", bytes_per_kv_token=1,
                  history_budget_bytes=100, workspace_budget_bytes=50)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    runtime = RuntimeAdapter.from_config(str(path), "local")
    assert runtime._token_counter([{"role": "user", "content": "x"}], []) == 41


def test_tool_tokenization_matches_declared_server_fields_without_mutating_source():
    tools = [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"},
              "response": {"description": "Benchmark response schema is not sent to the model"}}}]
    before = copy.deepcopy(tools)
    assert serving_tools(tools) == [{"type": "function", "function": {
        "description": None, "name": "lookup", "parameters": {"type": "object"}, "strict": False}}]
    assert tools == before
