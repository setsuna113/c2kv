"""Native tool-schema variants preserve the release cells and route the explicit mode."""
import copy
import json
from unittest import mock

import pytest

from benchmarks.native_tool_schema import (
    DEFAULT_NATIVE_TOOL_SCHEMA, NATIVE_TOOL_SCHEMAS, NativeToolSchema, parse_native_tool_schema,
)
from benchmarks.paper import c1, runner
from benchmarks.paper.candidate_matrix import with_candidate_methods

ARM = "c2kv_goal_pending_r8"


def config():
    base = json.loads(runner.DEFAULT_CONFIG.read_text(encoding="utf-8"))
    return with_candidate_methods(base, ("goal_pending",))


def test_raw_variants_cover_base_and_budget_cells_without_touching_originals(tmp_path):
    original = runner.with_native_history_budget(config(), ARM, 256)
    saved = copy.deepcopy(original)
    output = tmp_path / "results"
    old, profile = runner.prepare(original, output, tmp_path / "engine")
    expanded = runner.with_native_tool_schema(original, ARM, "raw")
    plan, _ = runner.prepare(expanded, output, tmp_path / "engine")
    assert original == saved
    by_id = {row["cell_id"]: row for row in plan}
    assert all(by_id[row["cell_id"]] == row for row in old)
    assert len(plan) == len(old) + 2
    for base_id in (f"bfcl_base__{ARM}", f"bfcl_base__{ARM}_b256"):
        variant = by_id[base_id + "__toolschema-raw"]
        assert variant["arm"] == ARM and variant["tool_schema"] == "raw"
        assert variant["group"] == "toolschema"
        assert variant.get("history_budget_tokens") == by_id[base_id].get("history_budget_tokens")
        for stage in ("closed_loop", "common_prefix"):
            argv = runner.run_command(expanded, variant, output / stage / variant["cell_id"], profile, stage)
            assert argv[argv.index("--tool-schema") + 1] == "raw"
            release = runner.run_command(expanded, by_id[base_id], output / stage / base_id, profile, stage)
            assert "--tool-schema" not in release
    matrix = (output / "matrix.csv").read_text()
    assert "tool_schema" in matrix and "__toolschema-raw" in matrix


def test_default_duplicate_engine_arm_and_unknown_modes_are_rejected():
    cfg = config()
    with pytest.raises(ValueError, match="release default"):
        runner.with_native_tool_schema(cfg, ARM, "sglang-full")
    with pytest.raises(ValueError, match="must be one of"):
        runner.with_native_tool_schema(cfg, ARM, "strict")
    for arm in ("full", "commitkv", "hiagent_full"):
        with pytest.raises(ValueError, match="not a native controller arm"):
            runner.with_native_tool_schema(cfg, arm, "raw")
    with pytest.raises(ValueError, match="requires a configured arm"):
        runner.with_native_tool_schema(cfg, "c2kv_goal_progress_r8", "raw")
    expanded = runner.with_native_tool_schema(cfg, ARM, "raw")
    with pytest.raises(ValueError, match="already exist"):
        runner.with_native_tool_schema(expanded, ARM, "raw")
    for value in ("", ARM, ARM + "=", "=raw"):
        with pytest.raises(ValueError, match="ARM=SCHEMA"):
            parse_native_tool_schema(value)
    assert parse_native_tool_schema(ARM + "=raw") == (ARM, "raw")
    assert NativeToolSchema("sglang-full").cell_suffix() == ""
    assert NativeToolSchema("raw").cli_args() == ["--tool-schema", "raw"]


def test_explicit_schema_on_engine_arm_fails_before_output(tmp_path):
    cfg = config()
    cfg["methods"][0]["tool_schema"] = "raw"
    assert cfg["methods"][0]["arm"] != ARM
    with pytest.raises(ValueError, match="not a native controller arm"):
        runner.prepare(cfg, tmp_path / "results", tmp_path / "engine")
    assert not (tmp_path / "results" / "config.resolved.json").exists()


def test_cli_repeat_and_selected_cell(tmp_path):
    args = ["--output", str(tmp_path / "results"), "--sglang-source", str(tmp_path / "engine"),
            "--candidate-arms", "goal_pending", "--native-history-budget", ARM + "=256",
            "--native-tool-schema", ARM + "=raw"]
    runner.main(["prepare", *args])
    selected = f"bfcl_base__{ARM}_b256__toolschema-raw"
    with mock.patch.object(runner, "execute") as execute:
        runner.main(["run", *args, "--stage", "closed_loop", "--cells", selected])
    assert execute.call_args.args[4:6] == (["closed_loop"], {selected})


def test_delivery_forwards_schema_only_when_explicit(tmp_path):
    original = c1.ARM
    try:
        c1.select_arm(ARM)
        delivery = c1.load_delivery()
        cfg = config()
        cfg["sglang_source"] = str(tmp_path / "engine")
        release = c1.delivery_args(cfg, "bfcl_base", tmp_path, [], delivery)
        assert release.tool_schema == "sglang-full"
        cfg["tool_schema"] = "raw"
        args = c1.delivery_args(cfg, "bfcl_base", tmp_path, [], delivery)
        assert args.tool_schema == "raw"
        assert args.ratio == release.ratio
        cfg["tool_schema"] = "strict"
        with pytest.raises(ValueError, match="must be one of"):
            c1.delivery_args(cfg, "bfcl_base", tmp_path, [], delivery)
    finally:
        c1.select_arm(original)


def test_delivery_modes_mirror_runtime_and_helper():
    delivery = c1.load_delivery()
    from benchmarks.memory_runtime.tokenization import DEFAULT_TOOL_SCHEMA, TOOL_SCHEMA_MODES
    assert delivery.TOOL_SCHEMA_MODES == TOOL_SCHEMA_MODES == NATIVE_TOOL_SCHEMAS
    assert delivery.TOOL_SCHEMA_DEFAULT == DEFAULT_TOOL_SCHEMA == DEFAULT_NATIVE_TOOL_SCHEMA
