from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from benchmarks import toolmemory, toolselection
from benchmarks.tool_definition import evaluate, prepare
from benchmarks.tool_definition.core import pack_layout, sha256_file
from benchmarks.tool_definition.test_core import CharacterTokenizer, decision


ADAPTIVE_POLICY = "last_user_adaptive_v1"


def selector_row(query: str, names: list[str]) -> dict:
    row = decision([])
    row["messages"] = [
        {"role": "system", "content": "Use the visible catalog."},
        {"role": "user", "content": query},
    ]
    row["tools"] = [
        {"type": "function", "function": {
            "name": name, "description": name, "parameters": {}}}
        for name in names
    ]
    return row


@pytest.mark.parametrize(("query", "names", "expected"), (
    ("nothing relevant", ["alpha", "beta", "gamma", "delta"], []),
    ("alpha beta gamma delta", ["alpha", "beta", "gamma", "delta"], [0, 1, 2, 3]),
    ("alpha beta gamma delta epsilon",
     ["alpha", "beta", "gamma", "delta", "epsilon", "zeta"], [0, 1, 2, 3, 4]),
))
def test_adaptive_selector_can_choose_zero_all_and_more_than_three(query, names, expected):
    record = pack_layout(
        selector_row(query, names), CharacterTokenizer(), layout="hybrid", ratio=8,
        k=3, selector_policy=ADAPTIVE_POLICY)
    assert record["native_indices"] == expected
    assert record["k"] is None
    assert record["selector_policy"] == ADAPTIVE_POLICY
    assert record["n_native"] == len(expected)
    assert record["selector_metadata"]["selection_count"] == len(expected)
    assert record["selector_metadata"]["relative_threshold"] == 0.5
    assert record["selector_metadata"]["top_k_cap"] is None


def prepare_fixture(tmp_path, monkeypatch, *, policy: str, row: dict | None = None):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    for name in ("config.json", "tokenizer.json", "model.safetensors"):
        (checkpoint / name).write_text("{}", encoding="utf-8")
    source = tmp_path / "decisions.jsonl"
    source.write_text(json.dumps(row or decision([])) + "\n", encoding="utf-8")
    monkeypatch.setattr(prepare.toolmemory, "load_tool_checkpoint_contract", lambda *_args: {})
    monkeypatch.setattr(prepare.toolmemory.NativeTokenizer, "_load",
                        lambda self: CharacterTokenizer())
    output = tmp_path / "prepared"
    manifest = prepare.prepare(
        source, checkpoint, output, k=3, ratios=(8,), selector_policy=policy)
    return checkpoint, output, manifest


def test_old_default_manifest_and_new_adaptive_manifest_both_load(tmp_path, monkeypatch):
    checkpoint, output, manifest = prepare_fixture(
        tmp_path, monkeypatch, policy=toolselection.DEFAULT_SELECTOR_POLICY)
    assert manifest["selector"] == {"ranker": toolmemory.RANKER, "k": 3, "seed": 42}
    frozen, groups = evaluate.read_manifest(output / "manifest.json", checkpoint)
    assert frozen["selector"] == manifest["selector"]
    assert "selector_policy" not in groups[("recorded-0", 8)]["hybrid"]


def test_adaptive_manifest_freezes_threshold_count_and_no_cap(tmp_path, monkeypatch):
    row = selector_row(
        "alpha beta gamma delta epsilon",
        ["alpha", "beta", "gamma", "delta", "epsilon", "zeta"])
    checkpoint, output, manifest = prepare_fixture(
        tmp_path, monkeypatch, policy=ADAPTIVE_POLICY, row=row)
    assert manifest["selector"] == {
        "ranker": toolmemory.RANKER,
        "seed": 42,
        "policy": ADAPTIVE_POLICY,
        "selector_version": toolselection.SELECTOR_VERSION,
        "selection_count": "adaptive",
        "relative_threshold": 0.5,
        "top_k_cap": None,
        "fixed_control_k": 3,
    }
    _, groups = evaluate.read_manifest(output / "manifest.json", checkpoint)
    hybrid = groups[("recorded-0", 8)]["hybrid"]
    assert hybrid["native_indices"] == [0, 1, 2, 3, 4]
    assert hybrid["n_native"] == 5


@pytest.mark.parametrize(("field", "value"), (
    ("relative_threshold", 0.6),
    ("selection_count", "fixed"),
    ("top_k_cap", 3),
    ("fixed_control_k", 0),
    ("selector_version", "tool-selection-v0"),
    ("k", 3),
))
def test_adaptive_manifest_rejects_changed_threshold_count_cap_or_identity(field, value):
    manifest = {"selector": {
        "ranker": toolmemory.RANKER,
        "seed": 42,
        "policy": ADAPTIVE_POLICY,
        "selector_version": toolselection.SELECTOR_VERSION,
        "selection_count": "adaptive",
        "relative_threshold": 0.5,
        "top_k_cap": None,
        "fixed_control_k": 3,
    }}
    manifest["selector"][field] = value
    with pytest.raises(ValueError, match="adaptive selector contract"):
        evaluate._selector_contract(manifest)


def test_nondefault_fixed_policy_requires_version_identity():
    manifest = {"selector": {
        "ranker": toolmemory.RANKER, "k": 3, "seed": 42,
        "policy": "latest_event_topk_v1",
    }}
    with pytest.raises(ValueError, match="lacks a version identity"):
        evaluate._selector_contract(manifest)
    manifest["selector"]["selector_version"] = toolselection.SELECTOR_VERSION
    assert evaluate._selector_contract(manifest)["policy"] == "latest_event_topk_v1"


def test_read_manifest_rejects_nondefault_record_identity_change(tmp_path, monkeypatch):
    checkpoint, output, manifest = prepare_fixture(
        tmp_path, monkeypatch, policy="latest_event_topk_v1")
    records_path = output / "records.jsonl"
    records = [json.loads(line) for line in records_path.read_text(
        encoding="utf-8").splitlines()]
    hybrid = next(record for record in records if record["layout"] == "hybrid")
    hybrid["selector_metadata"]["selector_version"] = "tool-selection-v0"
    records_path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")
    manifest["records"]["sha256"] = sha256_file(records_path)
    manifest["records"]["bytes"] = records_path.stat().st_size
    (output / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="selector metadata differs"):
        evaluate.read_manifest(output / "manifest.json", checkpoint)


class LiveCharacterTokenizer:
    def native_ids(self, messages):
        return CharacterTokenizer().apply_chat_template(messages, tokenize=True)


def test_prepared_adaptive_selector_matches_live_plan_and_rejects_other_policy():
    row = selector_row(
        "alpha beta gamma delta epsilon",
        ["alpha", "beta", "gamma", "delta", "epsilon", "zeta"])
    record = json.loads(json.dumps(pack_layout(
        row, CharacterTokenizer(), layout="hybrid", ratio=8, k=3,
        selector_policy=ADAPTIVE_POLICY)))
    contract = evaluate._selector_contract({"selector": {
        "ranker": toolmemory.RANKER,
        "seed": 42,
        "policy": ADAPTIVE_POLICY,
        "selector_version": toolselection.SELECTOR_VERSION,
        "selection_count": "adaptive",
        "relative_threshold": 0.5,
        "top_k_cap": None,
        "fixed_control_k": 3,
    }})
    adaptive_spec = toolmemory.ToolMemorySpec(
        ratio=8, layout="hybrid", top_k=3, selector_policy=ADAPTIVE_POLICY)
    live = toolmemory.plan_visible_tool_memory(
        {"messages": row["messages"], "tools": row["tools"]}, adaptive_spec,
        LiveCharacterTokenizer())
    assert live is not None
    evaluate._validate_live_selector(record, SimpleNamespace(info=live.info), contract)

    wrong_spec = toolmemory.ToolMemorySpec(
        ratio=8, layout="hybrid", top_k=3, selector_policy="latest_event_topk_v1")
    wrong = toolmemory.plan_visible_tool_memory(
        {"messages": row["messages"], "tools": row["tools"]}, wrong_spec,
        LiveCharacterTokenizer())
    assert wrong is not None
    with pytest.raises(ValueError, match="live selector"):
        evaluate._validate_live_selector(record, SimpleNamespace(info=wrong.info), contract)
