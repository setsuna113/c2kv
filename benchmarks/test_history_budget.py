"""The shared budget interface changes capacity without changing method identity."""
from dataclasses import asdict

import pytest

from benchmarks.arms import get_arm, history_kv_spec
from benchmarks.backends.sglang import SglangBackend
from benchmarks.history_budget import (HistoryKVBudget, parse_history_kv_budget,
                                       parse_history_kv_retention)


@pytest.mark.parametrize("name", ["commitkv", "agentkv",
    "history_kv_h2o_r25_persistent", "history_kv_snapkv_r25_persistent",
    "history_kv_pyramidkv_r25_persistent"])
def test_capacity_override_preserves_selector_and_registry(name):
    original = get_arm(name)
    before = asdict(original)
    resolved = HistoryKVBudget(768).apply(original)
    expected = {**history_kv_spec(original), "target_tokens": 768, "retention_ratio": None}
    assert history_kv_spec(resolved) == expected
    assert asdict(resolved) == {**before, "history_kv": expected}
    assert asdict(get_arm(name)) == before


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "768", None])
def test_budget_rejects_non_positive_integer(value):
    with pytest.raises(ValueError, match="positive integer"):
        HistoryKVBudget(value)


@pytest.mark.parametrize("name", [
    "history_kv_h2o_r25_persistent", "history_kv_snapkv_r25_persistent",
    "history_kv_pyramidkv_r25_persistent", "commitkv",
])
def test_ratio_override_clears_token_target_and_preserves_registry(name):
    original = get_arm(name)
    before = asdict(original)
    resolved = HistoryKVBudget(retention_ratio=0.5).apply(original)
    assert history_kv_spec(resolved) == {
        **history_kv_spec(original), "target_tokens": None, "retention_ratio": 0.5}
    assert asdict(get_arm(name)) == before
    assert HistoryKVBudget(retention_ratio=0.5).cli_args() == [
        "--history-kv-retention-ratio", "0.5"]


@pytest.mark.parametrize("ratio", [0, -0.1, 1.01, True, "0.5", None,
                                    float("nan"), float("inf"), float("-inf")])
def test_ratio_rejects_invalid_value(ratio):
    with pytest.raises(ValueError):
        HistoryKVBudget(retention_ratio=ratio)


def test_budget_modes_are_exclusive_and_names_drop_stale_default_ratio():
    with pytest.raises(ValueError, match="exactly one"):
        HistoryKVBudget()
    with pytest.raises(ValueError, match="exactly one"):
        HistoryKVBudget(768, 0.5)
    assert HistoryKVBudget(768).variant_name(
        "history_kv_h2o_r25_persistent") == "history_kv_h2o_persistent_b768"
    assert HistoryKVBudget(retention_ratio=0.5).variant_name(
        "history_kv_h2o_r25_persistent") == "history_kv_h2o_persistent_r0p5"
    assert HistoryKVBudget(retention_ratio=0.5).variant_name("commitkv") == "commitkv_r0p5"


@pytest.mark.parametrize("value", ["commitkv", "=0.5", "commitkv=0", "commitkv=-0.5",
                                  "commitkv=1.5", "commitkv=nan", "commitkv=inf",
                                  "commitkv=0.5=1", "commitkv=０.５"])
def test_ratio_cli_rejects_malformed_input(value):
    with pytest.raises(ValueError):
        parse_history_kv_retention(value)


def test_ratio_cli_accepts_bounded_fraction():
    assert parse_history_kv_retention("commitkv=0.5") == ("commitkv", 0.5)
    assert parse_history_kv_retention("agentkv=1") == ("agentkv", 1.0)


@pytest.mark.parametrize("name", ["full", "hiagent_full", "acon_hist_ut_co", "c2kv_native_r4"])
def test_other_memory_interfaces_are_not_silently_reinterpreted(name):
    with pytest.raises(ValueError, match="does not support"):
        HistoryKVBudget(768).apply(get_arm(name))


@pytest.mark.parametrize("value", ["commitkv", "=768", "commitkv=0", "commitkv=-1",
                                  "commitkv=1.5", "commitkv=768=1", "commitkv=７６８"])
def test_cli_budget_requires_explicit_arm_and_integer(value):
    with pytest.raises(ValueError):
        parse_history_kv_budget(value)


@pytest.mark.parametrize("name", ["commitkv", "agentkv"])
@pytest.mark.parametrize("tokens", [512, 768, 2048])
def test_budget_reaches_reference_wire_on_first_and_later_turns(name, tokens):
    arm_name, parsed = parse_history_kv_budget(f"{name}={tokens}")
    budget = HistoryKVBudget(parsed)
    arm = budget.apply(get_arm(arm_name))
    spec = history_kv_spec(arm)
    backend = SglangBackend(lambda *args, **kwargs: pytest.fail("unexpected network call"))
    _, first, _ = backend._apply_history_kv(
        [{"role": "user", "content": "current"}],
        {"spec": spec, "history_out_indices": [], "history_text": "",
         "session_id": "budget-session"}, None)
    assert first["history_kv_reference_config"]["target_tokens"] == tokens
    assert first["history_kv_reference_config"]["method"] == name
    assert first["persistent_history_session"] == {"enabled": True}
    _, later, _ = backend._apply_history_kv(
        [{"role": "user", "content": "old"},
         {"role": "assistant", "content": "answer"},
         {"role": "user", "content": "current"}],
        {"spec": spec, "history_out_indices": [0, 1], "history_text": "old answer",
         "history_message_count": 2, "history_start_message_count": 0,
         "session_id": "budget-session"}, None)
    assert later["history_kv_reference_config"]["target_tokens"] == tokens
    assert later["history_kv_eviction"]["target_tokens"] == tokens
    assert later["history_kv_eviction"]["retention_ratio"] is None
    assert later["history_kv_backend"] == "reference_attention"


@pytest.mark.parametrize("name", ["commitkv", "history_kv_pyramidkv_r25_persistent"])
def test_ratio_reaches_reference_wire_without_default_token_target(name):
    arm = HistoryKVBudget(retention_ratio=0.5).apply(get_arm(name))
    spec = history_kv_spec(arm)
    backend = SglangBackend(lambda *args, **kwargs: pytest.fail("unexpected network call"))
    _, first, _ = backend._apply_history_kv(
        [{"role": "user", "content": "current"}],
        {"spec": spec, "history_out_indices": [], "history_text": "",
         "session_id": "ratio-session"}, None)
    assert first["history_kv_reference_config"]["retention_ratio"] == 0.5
    assert first["history_kv_reference_config"]["target_tokens"] is None
    _, later, _ = backend._apply_history_kv(
        [{"role": "user", "content": "old"},
         {"role": "assistant", "content": "answer"},
         {"role": "user", "content": "current"}],
        {"spec": spec, "history_out_indices": [0, 1], "history_text": "old answer",
         "history_message_count": 2, "history_start_message_count": 0,
         "session_id": "ratio-session"}, None)
    assert later["history_kv_reference_config"]["retention_ratio"] == 0.5
    assert later["history_kv_reference_config"]["target_tokens"] is None
    assert later["history_kv_eviction"]["retention_ratio"] == 0.5
    assert "target_tokens" not in later["history_kv_eviction"]
