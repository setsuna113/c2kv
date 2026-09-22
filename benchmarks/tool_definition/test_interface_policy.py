"""The recorded Toucan study freezes one interface policy end to end."""
from __future__ import annotations

import json

import pytest

from benchmarks import toolmemory
from benchmarks.tool_definition import cli, evaluate, prepare
from benchmarks.tool_definition.core import FULL_CONTROL_POLICY, sha256_file
from benchmarks.tool_definition.test_core import CharacterTokenizer, decision


def test_prepare_manifest_and_evaluate_bind_schema_policy(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    for name in ("config.json", "tokenizer.json", "model.safetensors"):
        (checkpoint / name).write_text("{}", encoding="utf-8")
    source = tmp_path / "decisions.jsonl"
    source.write_text(json.dumps(decision([])) + "\n", encoding="utf-8")
    monkeypatch.setattr(prepare.toolmemory, "load_tool_checkpoint_contract",
                        lambda *_args: {})
    monkeypatch.setattr(prepare.toolmemory.NativeTokenizer, "_load",
                        lambda self: CharacterTokenizer())
    prepared = tmp_path / "prepared"
    manifest = prepare.prepare(source, checkpoint, prepared, k=1, ratios=(8,),
                               interface_policy="schema")
    assert manifest["interface_policy"] == "schema"
    assert manifest["interface_render_profile"] == toolmemory.INTERFACE_RENDER_PROFILE
    assert manifest["full_control_policy"] == FULL_CONTROL_POLICY
    frozen, groups = evaluate.read_manifest(prepared / "manifest.json", checkpoint)
    assert frozen["interface_policy"] == "schema"
    assert {record["interface_policy"] for record in groups[("recorded-0", 8)].values()} == {"schema"}
    assert {record["interface_render_profile"]
            for record in groups[("recorded-0", 8)].values()} == {
                toolmemory.INTERFACE_RENDER_PROFILE}
    assert {record["full_control_policy"]
            for record in groups[("recorded-0", 8)].values()} == {FULL_CONTROL_POLICY}
    with pytest.raises(ValueError, match="evaluation interface policy differs"):
        evaluate.evaluate(prepared / "manifest.json", checkpoint, tmp_path / "result",
                          upstream="http://127.0.0.1:9999", max_new_tokens=16)
    assert not (tmp_path / "result").exists()
    manifest.pop("interface_render_profile")
    (prepared / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="prepared tool interface render profile differs"):
        evaluate.read_manifest(prepared / "manifest.json", checkpoint)
    manifest["interface_render_profile"] = "tool-schema-catalog-frame-v1"
    (prepared / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="prepared tool interface render profile differs"):
        evaluate.read_manifest(prepared / "manifest.json", checkpoint)
    manifest["interface_render_profile"] = toolmemory.INTERFACE_RENDER_PROFILE
    manifest.pop("full_control_policy")
    (prepared / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="prepared Full control policy differs"):
        evaluate.read_manifest(prepared / "manifest.json", checkpoint)
    manifest["full_control_policy"] = FULL_CONTROL_POLICY
    manifest.pop("interface_policy")
    manifest.pop("interface_render_profile")
    manifest.pop("full_control_policy")
    (prepared / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="recorded tool interface policy differs"):
        evaluate.read_manifest(prepared / "manifest.json", checkpoint)


def test_schema_record_profile_mismatch_cannot_reuse_old_results(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    for name in ("config.json", "tokenizer.json", "model.safetensors"):
        (checkpoint / name).write_text("{}", encoding="utf-8")
    source = tmp_path / "decisions.jsonl"
    source.write_text(json.dumps(decision([])) + "\n", encoding="utf-8")
    monkeypatch.setattr(prepare.toolmemory, "load_tool_checkpoint_contract",
                        lambda *_args: {})
    monkeypatch.setattr(prepare.toolmemory.NativeTokenizer, "_load",
                        lambda self: CharacterTokenizer())
    prepared = tmp_path / "prepared"
    manifest = prepare.prepare(source, checkpoint, prepared, k=1, ratios=(8,),
                               interface_policy="schema")
    records_path = prepared / "records.jsonl"
    records = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines()]
    records[0]["interface_render_profile"] = "tool-schema-catalog-frame-v1"
    records_path.write_text("\n".join(json.dumps(row) for row in records) + "\n", encoding="utf-8")
    manifest["records"]["sha256"] = sha256_file(records_path)
    manifest["records"]["bytes"] = records_path.stat().st_size
    (prepared / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="recorded tool interface render profile differs"):
        evaluate.read_manifest(prepared / "manifest.json", checkpoint)
    records[0]["interface_render_profile"] = toolmemory.INTERFACE_RENDER_PROFILE
    records[0].pop("full_control_policy")
    records_path.write_text("\n".join(json.dumps(row) for row in records) + "\n", encoding="utf-8")
    manifest["records"]["sha256"] = sha256_file(records_path)
    manifest["records"]["bytes"] = records_path.stat().st_size
    (prepared / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="recorded Full control policy differs"):
        evaluate.read_manifest(prepared / "manifest.json", checkpoint)


def test_default_preparation_keeps_manifest_and_records_without_interface_profile(
    tmp_path, monkeypatch,
):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    for name in ("config.json", "tokenizer.json", "model.safetensors"):
        (checkpoint / name).write_text("{}", encoding="utf-8")
    source = tmp_path / "decisions.jsonl"
    source.write_text(json.dumps(decision([])) + "\n", encoding="utf-8")
    monkeypatch.setattr(prepare.toolmemory, "load_tool_checkpoint_contract",
                        lambda *_args: {})
    monkeypatch.setattr(prepare.toolmemory.NativeTokenizer, "_load",
                        lambda self: CharacterTokenizer())
    prepared = tmp_path / "prepared"
    manifest = prepare.prepare(source, checkpoint, prepared, k=1, ratios=(8,))
    assert "interface_policy" not in manifest
    assert "interface_render_profile" not in manifest
    assert "full_control_policy" not in manifest
    frozen, groups = evaluate.read_manifest(prepared / "manifest.json", checkpoint)
    assert "interface_render_profile" not in frozen
    assert "full_control_policy" not in frozen
    assert all("interface_render_profile" not in row
               for layouts in groups.values() for row in layouts.values())
    assert all("full_control_policy" not in row
               for layouts in groups.values() for row in layouts.values())


def test_cli_forwards_interface_policy_to_prepare_and_evaluate(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(cli, "prepare", lambda *_args, **kwargs: calls.append(("prepare", kwargs)) or {})
    from benchmarks.tool_definition import evaluate as evaluation_module
    monkeypatch.setattr(evaluation_module, "evaluate",
                        lambda *_args, **kwargs: calls.append(("evaluate", kwargs)) or {})
    cli.main(["prepare", "--input", str(tmp_path / "input"),
              "--checkpoint", str(tmp_path / "checkpoint"),
              "--out", str(tmp_path / "prepared"), "--interface-policy", "schema"])
    cli.main(["evaluate", "--manifest", str(tmp_path / "manifest"),
              "--checkpoint", str(tmp_path / "checkpoint"),
              "--out", str(tmp_path / "results"), "--upstream", "http://127.0.0.1:9999",
              "--max-new-tokens", "16", "--interface-policy", "schema",
              "--full-results", str(tmp_path / "donor")])
    assert calls[0][1]["interface_policy"] == "schema"
    assert calls[1][1]["interface_policy"] == "schema"
    assert calls[1][1]["full_results"] == tmp_path / "donor"
