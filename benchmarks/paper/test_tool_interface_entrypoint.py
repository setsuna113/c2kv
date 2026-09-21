"""The schema interface policy is an opt-in tool axis across paper arms."""
from __future__ import annotations

import json
import copy
from argparse import Namespace

import pytest

from benchmarks.paper import runner


def _config():
    return json.loads(runner.DEFAULT_CONFIG.read_text(encoding="utf-8"))


def test_default_matrix_and_legacy_tool_contexts_remain_unchanged():
    config = _config()
    rows = runner.cells(config)
    assert not any("_schema" in row["cell_id"] for row in rows)
    assert not any("tool_interface_policy" in row for row in rows)
    assert {row["cell_id"] for row in rows if row.get("tool_memory")} == {
        f"{benchmark}__full__tools-{context}"
        for benchmark in ("bfcl_base", "bfcl_long_context", "appworld",
                          "acebench_agent", "toolsandbox", "tau2")
        for context in ("t0_r8", "t0_r8_hybrid3")
    }


@pytest.mark.parametrize("context", ["t0_r8_hybrid3_schema", "h2o_r8_hybrid3_schema"])
def test_schema_context_routes_full_text_history_native_and_kv_arms(tmp_path, context):
    config = runner.with_tool_contexts(_config(), [context])
    plans, _ = runner.prepare(config, tmp_path / "paper", tmp_path / "engine")
    spec = runner.tool_contexts(config)[context]["spec"]
    assert spec.endswith(":schema")
    assert any(item["name"] == context and "schema interface policy" in item["provenance"]
               for item in config["tool_contexts"])
    for arm in ("full", "acon_hist_ut_co", "hiagent_full", "c2kv_native_r4",
                "history_kv_h2o_r25_persistent", "history_kv_snapkv_r25_persistent"):
        row = next(row for row in plans if row["benchmark"] == "bfcl_base"
                   and row["arm"] == arm and row["tool_context"] == context)
        assert row["tool_interface_policy"] == "schema"
        assert row["tool_memory"] == spec
        assert row["cell_id"].endswith("__tools-" + context)
        assert row["command"][row["command"].index("--tool-memory") + 1] == spec
        server = runner.server_command(config, tmp_path / "engine", arm, "bfcl_base",
                                       row["tool_checkpoint"], row["tool_memory"])
        assert ("--c2kv-tool-gist-weights" in server) == spec.startswith("t0:")
    csv_text = (tmp_path / "paper" / "matrix.csv").read_text(encoding="utf-8")
    assert "tool_interface_policy" in csv_text


def test_legacy_context_stays_excluded_from_text_arms_and_schema_is_distinct(tmp_path):
    config = runner.with_tool_contexts(_config(), ["t0_r8", "t0_r8_hybrid3_schema"])
    rows = runner.cells(config)
    for arm in ("acon_hist_ut_co", "hiagent_full"):
        selected = {row["tool_context"] for row in rows if row["arm"] == arm}
        assert "t0_r8_hybrid3_schema" in selected
        assert "t0_r8" not in selected
    config["methods"][1]["tool_contexts"] = ["t0_r8"]
    with pytest.raises(ValueError, match="require the schema interface policy"):
        runner.prepare(config, tmp_path / "paper", tmp_path / "engine")


def test_registering_unused_schema_contexts_extends_existing_default_root(tmp_path):
    config = _config()
    old = copy.deepcopy(config)
    old["tool_contexts"] = [entry for entry in old["tool_contexts"]
                            if not entry["name"].endswith("_schema")]
    output = tmp_path / "existing"
    before, _ = runner.prepare(old, output, tmp_path / "engine")
    after, _ = runner.prepare(config, output, tmp_path / "engine")
    assert [cell["cell_id"] for cell in after] == [cell["cell_id"] for cell in before]
    assert [cell["command"] for cell in after] == [cell["command"] for cell in before]
    assert list(output.glob("config.before_extension.*.json"))


def test_native_raw_schema_uses_actor_tokenizer_without_tool_checkpoint(tmp_path, monkeypatch):
    config = runner.with_tool_contexts(_config(), ["h2o_r8_hybrid3_schema"])
    native = next(row for row in runner.cells(config)
                  if row["benchmark"] == "bfcl_base" and row["arm"] == "c2kv_native_r4"
                  and row["tool_context"] == "h2o_r8_hybrid3_schema")
    native_command = runner.run_command(config, native, tmp_path, tmp_path / "profile.json")
    assert native_command[native_command.index("--tool-memory") + 1] == "h2o:r8:hybrid3:schema"
    assert "--tool-checkpoint" not in native_command

    text = next(row for row in runner.cells(config)
                if row["benchmark"] == "bfcl_base" and row["arm"] == "acon_hist_ut_co"
                and row["tool_context"] == "h2o_r8_hybrid3_schema")
    proxy_command = runner.run_command(config, text, tmp_path, tmp_path / "profile.json")
    assert proxy_command[proxy_command.index("--tool-checkpoint") + 1] == text["tool_checkpoint"]

    from benchmarks.paper import c1

    delivery = c1.load_delivery()
    checkpoint = tmp_path / "actor"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}", encoding="utf-8")
    sglang = tmp_path / "sglang"
    (sglang / ".git").mkdir(parents=True)
    args = Namespace(tool_memory="h2o:r8:hybrid3:schema", tool_checkpoint=None,
                     tool_budget_tokens=512, task_timeout=1, port=38922,
                     method="c2kv_native", embedding_batch_size=1,
                     checkpoint=checkpoint, sglang_root=sglang,
                     sglang_backend_url="http://127.0.0.1:38921", benchmark="bfcl")
    monkeypatch.setattr(delivery, "_identities", lambda _: ["bfcl_task"])
    monkeypatch.setattr(delivery, "_benchmark_dir", lambda _: tmp_path)
    assert delivery.validate_args(args) == ["bfcl_task"]
    args.tool_checkpoint = tmp_path / "not-for-raw-kv"
    with pytest.raises(ValueError, match="does not use --tool-checkpoint"):
        delivery.validate_args(args)
    args.tool_checkpoint = None
    args.tool_memory = "h2o:r8"
    with pytest.raises(ValueError, match="disjoint visible tool spans"):
        delivery.validate_args(args)
