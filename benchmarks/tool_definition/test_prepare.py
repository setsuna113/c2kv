from __future__ import annotations

import json
from types import SimpleNamespace

from benchmarks.tool_definition import prepare as module


def test_prepare_excludes_malformed_history_with_decision_id_and_reason(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    for name in ("config.json", "tokenizer.json", "model.safetensors"):
        (checkpoint / name).write_text(name, encoding="utf-8")
    source = tmp_path / "decisions.jsonl"
    rows = [
        {"decision_id": "bad-history", "messages": [{"role": "assistant", "tool_calls": [
            {"function": {"name": "search", "arguments": "{"}}]}],
         "tools": [{"type": "function", "function": {"name": "search"}}],
         "gold_tool_calls": [{"name": "search", "arguments": {}}]},
        {"decision_id": "good", "messages": [{"role": "assistant", "tool_calls": [
            {"function": {"name": "search", "arguments": "{}"}}]}],
         "tools": [{"type": "function", "function": {"name": "search"}}],
         "gold_tool_calls": [{"name": "search", "arguments": {}}]},
    ]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    monkeypatch.setattr(module.toolmemory, "load_tool_checkpoint_contract", lambda *_: None)
    monkeypatch.setattr(module.toolmemory, "NativeTokenizer",
                        lambda *_: SimpleNamespace(_load=lambda: object()))
    monkeypatch.setattr(module, "runtime_modules",
                        lambda: (None, None, None, None, lambda *_args, **_kwargs: [1, 2]))
    monkeypatch.setattr(module, "full_tool_spans", lambda *_args, **_kwargs: ((0, 1),))

    def packed(row, _tokenizer, *, layout, ratio, **_kwargs):
        return {"decision_id": row["decision_id"], "layout": layout, "ratio": ratio,
                "resident_kv_tokens": 10, "gold_tool_calls": row["gold_tool_calls"]}

    monkeypatch.setattr(module, "pack_layout", packed)
    monkeypatch.setattr(module, "retrieval_layout",
                        lambda row, tokenizer, *, ratio, **kwargs:
                        packed(row, tokenizer, layout="retrieval", ratio=ratio))
    manifest = module.prepare(source, checkpoint, tmp_path / "out", ratios=(8,))
    excluded = [json.loads(line) for line in (tmp_path / "out" / "excluded.jsonl").read_text(
        encoding="utf-8").splitlines()]
    records = [json.loads(line) for line in (tmp_path / "out" / "records.jsonl").read_text(
        encoding="utf-8").splitlines()]
    assert manifest["decisions"] == 1
    assert manifest["qualification"]["excluded"] == 1
    assert excluded[0]["decision_id"] == "bad-history"
    assert "invalid_history_tool_calls" in excluded[0]["reason"]
    assert len(records) == 5
    assert {record["decision_id"] for record in records} == {"good"}
