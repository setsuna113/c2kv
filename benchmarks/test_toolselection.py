"""CPU contracts for versioned native-tool selection policies."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import toolmemory  # noqa: E402
import toolselection  # noqa: E402


def tool(name: str, description: str) -> dict:
    return {"type": "function", "function": {
        "name": name,
        "description": description,
        "parameters": {"type": "object", "properties": {}},
    }}


TOOLS = [
    tool("book_flight", "Book a flight"),
    tool("cancel_order", "Cancel an order"),
    tool("get_weather", "Weather for a city"),
    tool("send_message", "Send a message"),
]


def old_rank(tools, query):
    tokens = lambda text: re.findall(r"[a-zA-Z0-9_]+", text.lower())
    query_tokens = set(tokens(query))
    if not query_tokens:
        return tuple(range(len(tools)))
    scored = []
    for index, item in enumerate(tools):
        function = item["function"]
        fields = [function["name"], function["description"], function["parameters"]]
        text = " ".join(value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
                        for value in fields if value)
        name_overlap = len(query_tokens & set(tokens(function["name"])))
        text_overlap = len(query_tokens & set(tokens(text)))
        scored.append((-(4.0 * name_overlap + float(text_overlap)), index))
    return tuple(index for _, index in sorted(scored))


def calls(*ids):
    return [{"id": call_id, "type": "function",
             "function": {"name": "inspect", "arguments": "{}"}}
            for call_id in ids]


def test_default_policy_is_byte_compatible_and_matches_frozen_rank():
    spec = toolmemory.parse_tool_memory_spec("t0:r8:hybrid3:schema")
    explicit = toolmemory.ToolMemorySpec(
        ratio=8, layout="hybrid", top_k=3, interface_policy="schema")
    messages = [{"role": "user", "content": "please cancel_order then send_message"}]
    expected_rank = old_rank(TOOLS, messages[0]["content"])

    assert spec == explicit
    assert spec.name == "t0_r8_hybrid3_schema"
    assert "selector_policy" not in spec.as_dict()
    assert "selector_version" not in spec.as_dict()
    assert toolmemory.lexical_rank(TOOLS, messages[0]["content"]) == expected_rank
    assert toolmemory.native_indices(TOOLS, spec, messages) == tuple(sorted(expected_rank[:3]))


def test_parser_versions_nondefault_policy_without_changing_codec_layout():
    text = "t0:r8:hybrid3:schema:selector=latest_event_topk_v1"
    spec = toolmemory.parse_tool_memory_spec(text)
    assert (spec.encoder, spec.ratio, spec.layout, spec.top_k, spec.interface_policy) == (
        "t0", 8, "hybrid", 3, "schema")
    assert spec.selector_policy == "latest_event_topk_v1"
    assert spec.name == "t0_r8_hybrid3_schema_selector_latest_event_topk_v1"
    assert spec.as_dict()["selector_version"] == "tool-selection-v1"


def test_latest_event_query_changes_selection_with_completed_observation():
    spec = toolmemory.parse_tool_memory_spec(
        "t0:r8:hybrid1:selector=latest_event_topk_v1")
    prefix = [
        {"role": "user", "content": "continue"},
        {"role": "assistant", "content": None, "tool_calls": calls("c1")},
    ]
    weather = toolselection.tool_selection(
        TOOLS, spec, [*prefix, {"role": "tool", "tool_call_id": "c1",
                               "content": "get_weather city forecast"}])
    cancel = toolselection.tool_selection(
        TOOLS, spec, [*prefix, {"role": "tool", "tool_call_id": "c1",
                               "content": "cancel_order order"}])
    assert weather["native_indices"] == (2,)
    assert cancel["native_indices"] == (1,)
    assert weather["query_sha256"] != cancel["query_sha256"]


def test_latest_event_ignores_old_turn_and_incomplete_new_batch():
    spec = toolmemory.parse_tool_memory_spec(
        "t0:r8:hybrid1:selector=latest_event_topk_v1")
    messages = [
        {"role": "user", "content": "get_weather"},
        {"role": "assistant", "content": None, "tool_calls": calls("old")},
        {"role": "tool", "tool_call_id": "old", "content": "weather city"},
        {"role": "user", "content": "cancel_order now"},
        {"role": "assistant", "content": None, "tool_calls": calls("a", "b")},
        {"role": "tool", "tool_call_id": "a", "content": "get_weather"},
    ]
    selected = toolselection.tool_selection(TOOLS, spec, messages)
    assert selected["latest_io_present"] is False
    assert selected["native_indices"] == (1,)


def test_adaptive_policy_allows_zero_all_and_more_than_three():
    spec = toolmemory.parse_tool_memory_spec(
        "t0:r8:hybrid3:selector=last_user_adaptive_v1")
    four = [tool(f"tool_{index}", "target" if index < 4 else "other")
            for index in range(5)]
    all_five = [tool(f"tool_{index}", "target") for index in range(5)]
    assert toolselection.tool_selection(
        four, spec, [{"role": "user", "content": "absent"}])["native_indices"] == ()
    assert toolselection.tool_selection(
        four, spec, [{"role": "user", "content": "target"}])["native_indices"] == (0, 1, 2, 3)
    assert toolselection.tool_selection(
        all_five, spec, [{"role": "user", "content": "target"}])["native_indices"] == (0, 1, 2, 3, 4)


def test_nondefault_plan_separates_scored_and_interface_forced_native():
    content = "prefix OPAQUE-TOOL suffix"
    start = content.index("OPAQUE-TOOL")
    payload = {
        "messages": [{"role": "system", "content": content},
                     {"role": "user", "content": "absent"}],
        "tools": TOOLS[:2],
        toolmemory.TOOL_SPANS_FIELD: [{
            "message_index": 0, "start": start, "end": start + len("OPAQUE-TOOL"),
            "source": "fixture",
        }],
    }
    spec = toolmemory.parse_tool_memory_spec(
        "h2o:r8:hybrid3:schema:selector=last_user_adaptive_v1")
    plan = toolmemory.plan_visible_tool_memory(payload, spec)
    assert plan.info["score_selected_native_indices"] == []
    assert plan.info["interface_forced_native_indices"] == [2]
    assert plan.info["native_indices"] == [2]


def test_dynamic_selector_reads_recorded_bfcl_source_events():
    fixture = json.loads((HERE / "fixtures/tool_selector_bfcl_source.json").read_text(
        encoding="utf-8"))
    messages = fixture["messages"]
    spec = toolmemory.parse_tool_memory_spec(
        "t0:r8:hybrid1:selector=latest_event_topk_v1")
    catalog = [tool(name, "") for name in ("mkdir", "ls", "cd", "rm", "zip")]
    hashes = []
    for end, expected in ((3, 0), (5, 1), (7, 2), (9, 1)):
        completed = toolselection.tool_selection(catalog, spec, messages[:end])
        assert completed["native_indices"] == (expected,)
        hashes.append(completed["query_sha256"])
        if end < len(messages):
            assert toolselection.tool_selection(catalog, spec, messages[:end + 1]) == completed
    assert len(set(hashes)) == len(hashes)


def test_adaptive_keeps_unbounded_legacy_query_and_hashes_scored_text():
    spec = toolmemory.parse_tool_memory_spec(
        "t0:r8:hybrid3:selector=last_user_adaptive_v1")
    prefix = " " * (toolselection.MAX_QUERY_COMPONENT_CHARS + 1)
    messages = [{"role": "user", "content": prefix + "send_message"}]
    selected = toolselection.tool_selection(TOOLS, spec, messages)
    assert selected["native_indices"] == (3,)
    assert selected["query_sha256"] == toolselection._query_hash(prefix + "send_message")


def test_dynamic_query_hash_uses_bounded_scoring_components_after_execution():
    spec = toolmemory.parse_tool_memory_spec(
        "t0:r8:hybrid3:selector=latest_event_topk_v1")
    prefix = "get_weather " + " " * toolselection.MAX_QUERY_COMPONENT_CHARS
    messages = [
        {"role": "user", "content": prefix + "cancel_order"},
        {"role": "assistant", "tool_calls": calls("c1")},
        {"role": "tool", "tool_call_id": "c1", "content": "send_message"},
    ]
    selected = toolselection.tool_selection(TOOLS, spec, messages)
    changed = [dict(message) for message in messages]
    changed[0]["content"] = prefix + "book_flight"
    assert toolselection.tool_selection(TOOLS, spec, changed) == selected
