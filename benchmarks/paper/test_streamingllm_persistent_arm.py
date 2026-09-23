"""Persistent StreamingLLM joins the absolute history-KV budget interface opt-in."""
import json

import pytest

from benchmarks.arms import get_arm, history_kv_spec
from benchmarks.backends.sglang import SglangBackend
from benchmarks.history_budget import HistoryKVBudget
from benchmarks.paper import runner

ARM = "history_kv_streamingllm_r25_persistent"


def config(budget=256, extra=()):
    cfg = json.loads(runner.DEFAULT_CONFIG.read_text(encoding="utf-8"))
    cfg["history_kv_budget_tokens"] = budget
    cfg["methods"] = [*cfg["methods"], *extra]
    return cfg


def entry(**fields):
    return {"method": "StreamingLLM", "arm": ARM, "benchmarks": ["bfcl_base"], **fields}


def test_arm_is_a_persistent_physical_eviction_spec_like_h2o():
    spec = history_kv_spec(get_arm(ARM))
    h2o = history_kv_spec(get_arm("history_kv_h2o_r25_persistent"))
    assert spec == {**h2o, "method": "streamingllm"}
    assert (spec["backend"], spec["persistent_session"], spec["retention_ratio"],
            spec["target_tokens"]) == ("physical_eviction", True, 0.25, None)
    for tokens in (256, 128):
        budget = HistoryKVBudget(tokens)
        resolved = history_kv_spec(budget.apply(get_arm(ARM)))
        assert resolved == {**spec, "target_tokens": tokens, "retention_ratio": None}
        assert budget.variant_name(ARM) == f"history_kv_streamingllm_persistent_b{tokens}"


def test_default_matrix_has_no_streamingllm_cell(tmp_path):
    plan, _ = runner.prepare(config(), tmp_path / "results", tmp_path / "engine")
    assert not any("streamingllm" in cell["cell_id"] for cell in plan)
    assert ARM not in {method["arm"] for method in config()["methods"]}


def _cells(plan):
    return {cell["cell_id"]: cell for cell in plan
            if cell["cell_id"].startswith("bfcl_base__history_kv_streamingllm")}


def _assert_budget_cell(cell, tokens):
    command = cell["command"]
    assert command[command.index("--arm") + 1] == ARM
    assert command[command.index("--history-kv-target-tokens") + 1] == str(tokens)
    assert cell["history_budget_tokens"] == tokens and "retention" not in cell


def test_explicit_budget_entries_prepare_b256_and_b128(tmp_path):
    extra = [entry(group="budget", history_budget_tokens=256),
             entry(group="budget", history_budget_tokens=128)]
    plan, _ = runner.prepare(config(extra=extra), tmp_path / "results", tmp_path / "engine")
    cells = _cells(plan)
    assert sorted(cells) == ["bfcl_base__history_kv_streamingllm_persistent_b128",
                             "bfcl_base__history_kv_streamingllm_persistent_b256"]
    for tokens in (256, 128):
        _assert_budget_cell(cells[f"bfcl_base__history_kv_streamingllm_persistent_b{tokens}"], tokens)


def test_cli_shared_and_arm_budgets_prepare_b256_and_b128(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config(budget=None, extra=[
        entry(group="main", history_budget_tokens="shared")])), encoding="utf-8")
    output = tmp_path / "results"
    runner.main(["prepare", "--config", str(path), "--output", str(output),
                 "--history-kv-budget-tokens", "256", "--history-kv-budget", f"{ARM}=128"])
    plan = json.loads((output / "commands.json").read_text(encoding="utf-8"))
    cells = _cells(plan)
    assert sorted(cells) == ["bfcl_base__history_kv_streamingllm_persistent_b128",
                             "bfcl_base__history_kv_streamingllm_persistent_b256"]
    for tokens in (256, 128):
        _assert_budget_cell(cells[f"bfcl_base__history_kv_streamingllm_persistent_b{tokens}"], tokens)
    frozen = json.loads((output / "config.resolved.json").read_text(encoding="utf-8"))
    server = runner.server_command(frozen, tmp_path / "engine", ARM, "bfcl_base")
    assert {"--disable-radix-cache", "--enable-streaming-session",
            "--disable-overlap-schedule"} <= set(server)


def test_method_label_must_match_the_registered_selector(tmp_path):
    wrong = entry(group="budget", history_budget_tokens=256, arm="history_kv_h2o_r25_persistent")
    with pytest.raises(ValueError, match="Persistent history-KV budget differs from matrix"):
        runner.prepare(config(extra=[wrong]), tmp_path / "results", tmp_path / "engine")


def test_budget_reaches_the_persistent_eviction_wire():
    spec = history_kv_spec(HistoryKVBudget(128).apply(get_arm(ARM)))
    backend = SglangBackend(lambda *args, **kwargs: pytest.fail("unexpected network call"))
    _, first, _ = backend._apply_history_kv(
        [{"role": "user", "content": "current"}],
        {"spec": spec, "history_out_indices": [], "history_text": "",
         "session_id": "streaming-session"}, None)
    assert first == {"persistent_history_session": {"enabled": True}}
    _, later, _ = backend._apply_history_kv(
        [{"role": "user", "content": "old"},
         {"role": "assistant", "content": "answer"},
         {"role": "user", "content": "current"}],
        {"spec": spec, "history_out_indices": [0, 1], "history_text": "old answer",
         "history_message_count": 2, "history_start_message_count": 0,
         "session_id": "streaming-session"}, None)
    eviction = later["history_kv_eviction"]
    assert (eviction["method"], eviction["target_tokens"], eviction["retention_ratio"],
            eviction["persistent_session"]) == ("streamingllm", 128, None, True)
    assert later["history_kv_backend"] == "physical_eviction"
    assert later["history_kv_method"] == "streamingllm"
    assert later["persistent_history_session"] == {"enabled": True}
