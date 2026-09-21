from __future__ import annotations

from benchmarks import toolmemory
from benchmarks.tool_definition.core import (FULL_CONTROL_POLICY, full_tool_spans, pack_layout,
                                             random_rank, retrieval_layout)


class CharacterTokenizer:
    def apply_chat_template(self, messages, *, tools=None, tokenize=True,
                            add_generation_prompt=False, **_kwargs):
        rendered = "".join(f"<{item['role']}>{item.get('content') or ''}\n" for item in messages)
        if add_generation_prompt:
            rendered += "<assistant>"
        return [ord(char) + 1 for char in rendered]


def decision(gold):
    return {
        "decision_id": "recorded-0", "source": "recorded fixture",
        "messages": [
            {"role": "system", "content": "Use the visible catalog."},
            {"role": "user", "content": "Search an invoice and then send it."},
            {"role": "assistant", "content": "I will search first."},
        ],
        "tools": [
            {"type": "function", "function": {"name": "search_invoice", "description": "Search invoice"}},
            {"type": "function", "function": {"name": "send_invoice", "description": "Send invoice"}},
            {"type": "function", "function": {"name": "cancel_order", "description": "Cancel order"}},
            {"type": "function", "function": {"name": "get_weather", "description": "Get weather"}},
        ],
        "gold_tool_calls": gold,
    }


def test_layouts_preserve_full_history_and_do_not_read_gold_for_selection():
    tokenizer = CharacterTokenizer()
    with_gold = decision([{"name": "cancel_order", "arguments": {}}])
    without_gold = decision([])
    full = pack_layout(with_gold, tokenizer, layout="full", ratio=8)
    uniform = pack_layout(with_gold, tokenizer, layout="uniform", ratio=8)
    hybrid = pack_layout(with_gold, tokenizer, layout="hybrid", ratio=8, k=1)
    random = pack_layout(with_gold, tokenizer, layout="random", ratio=8, k=1, seed=42)
    for record in (full, uniform, hybrid, random):
        assert record["prompt_sha256"] == full["prompt_sha256"]
        assert record["memory"]["workspace_input_ids"] == full["memory"]["workspace_input_ids"]
    assert uniform["native_indices"] == []
    assert full["native_indices"] == list(range(4))
    assert hybrid["native_indices"] == pack_layout(without_gold, tokenizer,
                                  layout="hybrid", ratio=8, k=1)["native_indices"]
    assert random["native_indices"] == pack_layout(without_gold, tokenizer,
                                  layout="random", ratio=8, k=1, seed=42)["native_indices"]
    assert len(full["memory"]["chunks"]) == 0
    assert len(uniform["memory"]["chunks"]) == 4
    assert len(hybrid["memory"]["chunks"]) == 3


def test_retrieval_only_fits_matched_total_kv_allowance_and_spans_partition_tools():
    row, tokenizer = decision([]), CharacterTokenizer()
    hybrid = pack_layout(row, tokenizer, layout="hybrid", ratio=8, k=1)
    retrieved = retrieval_layout(row, tokenizer, ratio=8,
                                 allowance_tokens=hybrid["resident_kv_tokens"], k=1)
    assert retrieved["resident_kv_tokens"] <= hybrid["resident_kv_tokens"]
    assert retrieved["memory"]["chunks"] == () or retrieved["memory"]["chunks"] == []
    spans = full_tool_spans(row, tokenizer, ratio=8)
    full = pack_layout(row, tokenizer, layout="full", ratio=8)
    assert len(spans) == len(row["tools"])
    assert all(left[1] == right[0] for left, right in zip(spans, spans[1:]))
    assert spans[-1][1] <= len(full["memory"]["system_input_ids"])
    assert random_rank(4, seed=42, decision_id=row["decision_id"]) == random_rank(
        4, seed=42, decision_id=row["decision_id"])


def test_schema_interface_encodes_only_prose_and_keeps_top_k_native_once():
    row, tokenizer = decision([]), CharacterTokenizer()
    original = pack_layout(row, tokenizer, layout="hybrid", ratio=8, k=1)
    schema = pack_layout(row, tokenizer, layout="hybrid", ratio=8, k=1,
                         interface_policy="schema")
    assert schema["interface_policy"] == "schema"
    assert schema["interface_render_profile"] == toolmemory.INTERFACE_RENDER_PROFILE
    assert schema["full_control_policy"] == FULL_CONTROL_POLICY
    assert "interface_render_profile" not in original
    assert schema["native_indices"] == original["native_indices"]
    assert schema["memory"]["chunks"] != original["memory"]["chunks"]
    for chunk in schema["memory"]["chunks"]:
        text = "".join(chr(token - 1) for token in chunk["token_ids"])
        assert 'tool_description' in text
        assert '"parameters"' not in text
        assert '"name"' not in text
    assert schema["memory"]["workspace_input_ids"] == original["memory"]["workspace_input_ids"]
    protocol = "".join(chr(token - 1) for token in schema["memory"]["system_input_ids"])
    assert "# Executable tool interfaces" in protocol
    for index in range(len(row["tools"])):
        name = row["tools"][index]["function"]["name"]
        assert protocol.count('"name":"' + name + '"') == 1
    retrieved = retrieval_layout(row, tokenizer, ratio=8,
                                 allowance_tokens=schema["resident_kv_tokens"], k=1,
                                 interface_policy="schema")
    assert retrieved["interface_policy"] == "schema"
    assert retrieved["interface_render_profile"] == toolmemory.INTERFACE_RENDER_PROFILE
    assert retrieved["resident_kv_tokens"] <= schema["resident_kv_tokens"]


def test_schema_policy_keeps_full_control_identical_to_original_raw_catalog():
    row, tokenizer = decision([]), CharacterTokenizer()
    original = pack_layout(row, tokenizer, layout="full", ratio=8)
    schema = pack_layout(row, tokenizer, layout="full", ratio=8,
                         interface_policy="schema")
    assert schema["interface_policy"] == "schema"
    assert schema["interface_render_profile"] == toolmemory.INTERFACE_RENDER_PROFILE
    assert schema["full_control_policy"] == FULL_CONTROL_POLICY
    assert schema["native_indices"] == original["native_indices"]
    assert schema["memory"] == original["memory"]
    assert schema["resident_kv_tokens"] == original["resident_kv_tokens"]
