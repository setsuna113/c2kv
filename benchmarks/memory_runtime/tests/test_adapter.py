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
