from __future__ import annotations

import json
import math

import pytest

from history_memory.dataset import iter_decisions
from next_compression.exp1_tools import (
    EXP1_PURPOSE,
    GIST_LAYOUTS,
    LayoutPlan,
    build_layout_records,
    exact_mcnemar,
    layout_material,
    lexical_rank,
    native_tool_token_spans,
    outcome,
    pack_layout,
    paired_test,
    query_text,
    random_rank,
    retrieval_native_indices,
    score_rows,
    validate_exp1_record,
)
from next_compression.test_tools import ByteChatTokenizer, _row, _standard_tools
from next_compression.tools import ToolPreparationConfig, pack_tool_memory, tool_variant_material


def _three_tools():
    tools = _standard_tools()
    tools.append(
        {
            "type": "function",
            "function": {
                "name": "book_flight",
                "description": "Book a flight between two airports.",
                "parameters": {
                    "type": "object",
                    "properties": {"origin": {"type": "string"}},
                    "required": ["origin"],
                },
            },
        }
    )
    return tools


def _last_decision(row):
    return list(iter_decisions(row))[-1]


def test_lexical_rank_weighs_name_overlap_four_times_text_overlap():
    tools = _three_tools()
    assert lexical_rank(tools, "please book my flight tomorrow")[0] == 2
    assert lexical_rank(tools, "what is the weather")[0] == 0
    # No overlap keeps catalog order; ties are stable by index.
    assert lexical_rank(tools, "") == (0, 1, 2)
    assert lexical_rank(tools, "zzz") == (0, 1, 2)


def test_query_text_reads_the_latest_user_message():
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "reply"},
        {"role": "user", "content": "second"},
        {"role": "tool", "content": "result"},
    ]
    assert query_text(messages) == "second"
    assert query_text(messages, "all") == "sys\nfirst\nreply\nsecond\nresult"
    with pytest.raises(ValueError):
        query_text(messages, "unknown")


def test_random_rank_is_deterministic_per_decision_and_seed():
    a = random_rank(6, seed=42, decision_id="d-1")
    assert sorted(a) == list(range(6))
    assert a == random_rank(6, seed=42, decision_id="d-1")
    assert a != random_rank(6, seed=43, decision_id="d-1") or a != random_rank(6, seed=42, decision_id="d-2")


def test_layout_material_splits_native_and_t0_documents():
    tools = _three_tools()
    t0 = tool_variant_material(tools, "T0")
    assert layout_material(tools, "uniform") == t0
    hybrid = layout_material(tools, "hybrid", native_indices=(2, 0))
    assert [tool["function"]["name"] for tool in hybrid.native_tools if "function" in tool] == [
        "lookup_weather",
        "book_flight",
    ]
    # Remainder documents are byte-identical to the T0 documents of those tools.
    assert hybrid.documents == (t0.documents[1],)
    full = layout_material(tools, "full")
    assert len(full.native_tools) == 3 and full.documents == ()
    retrieval = layout_material(tools, "retrieval", native_indices=(1,))
    assert len(retrieval.native_tools) == 1 and retrieval.documents == ()
    with pytest.raises(ValueError):
        layout_material(tools, "hybrid", native_indices=(0, 0))
    with pytest.raises(ValueError):
        layout_material(tools, "hybrid", native_indices=(7,))


def test_pack_layout_full_and_hybrid_render_expected_prefixes():
    tokenizer = ByteChatTokenizer()
    row = _row()
    row["tools"] = _three_tools()
    decision = _last_decision(row)
    config = ToolPreparationConfig()

    full = pack_layout(decision, tokenizer, layout="full", config=config)
    assert full.chunks == ()
    system = tokenizer.decode(full.system_input_ids)
    assert "lookup_weather" in system and "book_flight" in system

    hybrid = pack_layout(decision, tokenizer, layout="hybrid", config=config, native_indices=(2,))
    system = tokenizer.decode(hybrid.system_input_ids)
    assert "book_flight" in system and "lookup_weather" not in system
    encoded = "\n".join(tokenizer.decode(chunk.token_ids) for chunk in hybrid.chunks)
    assert "lookup_weather" in encoded and "book_flight" not in encoded
    # The workspace is identical across layouts.
    assert hybrid.workspace_input_ids == full.workspace_input_ids
    uniform = pack_tool_memory(decision.store, decision.tools, tokenizer, variant="T0", config=config)
    assert pack_layout(decision, tokenizer, layout="uniform", config=config) == uniform


def test_native_tool_token_spans_are_exact_with_a_byte_tokenizer():
    tokenizer = ByteChatTokenizer()
    row = _row()
    row["tools"] = _three_tools()
    decision = _last_decision(row)
    config = ToolPreparationConfig()
    full = pack_layout(decision, tokenizer, layout="full", config=config)
    spans = native_tool_token_spans(decision, tokenizer, config=config)
    assert len(spans) == 3
    previous_end = 0
    for (start, end), tool in zip(spans, decision.tools):
        assert start >= previous_end and end > start
        raw = "\n" + json.dumps(tool, ensure_ascii=False, separators=(",", ":"))
        # The byte tokenizer renders the system message as JSON, so the schema
        # text appears in its JSON-escaped form inside the content string.
        expected = json.dumps(raw, ensure_ascii=False)[1:-1]
        assert tokenizer.decode(full.system_input_ids[start:end]) == expected
        previous_end = end
    # The spans partition the catalog region: nothing before the first span
    # or after the last one mentions a schema.
    before = tokenizer.decode(full.system_input_ids[: spans[0][0]])
    after = tokenizer.decode(full.system_input_ids[spans[-1][1] :])
    assert "lookup_weather" not in before and "book_flight" not in after
    assert after.startswith(json.dumps("\n</tools>")[1:-1])


def test_retrieval_admits_ranked_schemas_within_the_allowance():
    tokenizer = ByteChatTokenizer()
    row = _row()
    row["tools"] = _three_tools()
    decision = _last_decision(row)
    config = ToolPreparationConfig()
    ranked = (2, 0, 1)
    everything = pack_layout(decision, tokenizer, layout="full", config=config).costs(8)["resident_kv_tokens"]
    assert retrieval_native_indices(decision, tokenizer, ranked=ranked, allowance_tokens=everything, config=config) == (0, 1, 2)
    nothing = pack_layout(decision, tokenizer, layout="retrieval", config=config).costs(8)["resident_kv_tokens"]
    assert retrieval_native_indices(decision, tokenizer, ranked=ranked, allowance_tokens=nothing, config=config) == ()
    one = pack_layout(decision, tokenizer, layout="retrieval", config=config, native_indices=(2,)).costs(8)["resident_kv_tokens"]
    chosen = retrieval_native_indices(decision, tokenizer, ranked=ranked, allowance_tokens=one, config=config)
    assert 2 in chosen
    assert pack_layout(decision, tokenizer, layout="retrieval", config=config, native_indices=chosen).costs(8)["resident_kv_tokens"] <= one


def test_build_layout_records_covers_every_layout_ratio_and_validates():
    tokenizer = ByteChatTokenizer()
    row = _row()
    row["tools"] = _three_tools()
    decision = _last_decision(row)
    config = ToolPreparationConfig()
    from history_memory.packing import pack_target

    target_ids = pack_target(tokenizer, decision.target)
    plan = LayoutPlan(k_values=(1, 2), seed=7)
    records = build_layout_records(
        decision,
        tokenizer,
        config=config,
        plan=plan,
        session_key="fixture:session-1",
        source="fixture",
        target_ids=target_ids,
        base_metadata={"decision_type": "tool_call", "gold_tool_calls": [
            {"name": "lookup_weather", "arguments": {"city": "TARGET_SECRET_ARGUMENT", "units": "c"}}
        ]},
    )
    keys = {(record["layout"], record["k"], record["ratio"]) for record in records}
    expected = {("full", None, 8), ("full", None, 12), ("uniform", None, 8), ("uniform", None, 12),
                ("t1", None, 8), ("t1", None, 12)}
    for k in (1, 2):
        for ratio in (8, 12):
            expected |= {("hybrid", k, ratio), ("random", k, ratio), ("retrieval", k, ratio)}
    assert keys == expected
    assert len(records) == len(keys)
    for record in records:
        validate_exp1_record(record)
        assert record["split"] == EXP1_PURPOSE
        assert record["decision_id"] == decision.decision_id
        assert record["metadata"]["ranker"].startswith("lexical")
        assert record["metadata"]["token_counts"]["target_tokens"] == len(target_ids)
    full = next(record for record in records if record["layout"] == "full")
    assert len(full["metadata"]["tool_token_spans"]) == 3
    hybrid = next(record for record in records if record["layout"] == "hybrid" and record["k"] == 1)
    # CURRENT_QUERY overlaps no name, so catalog order wins and lookup_weather is native.
    assert hybrid["metadata"]["native_tool_indices"] == [0]
    assert hybrid["metadata"]["selection_hit"] is True
    retrieval = next(record for record in records if record["layout"] == "retrieval" and record["k"] == 1 and record["ratio"] == 8)
    assert retrieval["metadata"]["allowance_source"] == "hybrid.k1.ratio8"
    assert retrieval["metadata"]["token_counts"]["resident_kv_tokens"] <= retrieval["metadata"]["allowance_tokens"]
    with pytest.raises(ValueError):
        LayoutPlan(k_values=(0,))
    with pytest.raises(ValueError):
        LayoutPlan(layouts=("snapkv",))


def test_exact_mcnemar_matches_binomial_tail():
    assert exact_mcnemar(0, 0) == 1.0
    assert exact_mcnemar(1, 0) == 1.0
    assert exact_mcnemar(9, 0) == pytest.approx(2 * (1 / 512))
    expected = 2 * sum(math.comb(21, i) for i in range(7)) / 2 ** 21
    assert exact_mcnemar(15, 6) == pytest.approx(expected)
    assert exact_mcnemar(6, 15) == exact_mcnemar(15, 6)


def _row_out(decision_id, generated, gold, kv=10):
    return {
        "decision_id": decision_id,
        "generated_text": generated,
        "target_text": "",
        "gold_tool_calls": gold,
        "resident_kv_tokens": kv,
    }


CALL = '<tool_call>\n{"name": "lookup", "arguments": {"q": "x"}}\n</tool_call>'
GOLD = [{"name": "lookup", "arguments": {"q": "x"}}]


def test_outcome_and_score_rows_report_explicit_denominators():
    rows = [
        _row_out("a", CALL, GOLD),
        _row_out("b", '<tool_call>\n{"name": "lookup", "arguments": {"q": "y"}}\n</tool_call>', GOLD),
        _row_out("c", "plain answer", []),
        _row_out("d", CALL, []),
        _row_out("e", "<tool_call>{broken", GOLD),
    ]
    metrics = score_rows(rows)
    assert metrics["tool_decisions"] == 3 and metrics["non_tool_decisions"] == 2
    assert metrics["strict_ordered_call_correct"] == 1
    assert metrics["tool_name_correct"] == 2
    assert metrics["false_tool_calls"] == 1
    assert metrics["malformed_outputs"] == 1
    assert outcome(rows[1]) == {
        "tool_decision": True, "call_correct": False, "name_correct": True,
        "false_call": None, "malformed": False,
    }
    with pytest.raises(ValueError):
        score_rows(rows + [rows[0]])


def test_paired_test_counts_discordant_pairs_on_shared_tool_decisions():
    left = score_rows([_row_out("a", CALL, GOLD), _row_out("b", "nope", GOLD), _row_out("c", "x", [])])["outcomes"]
    right = score_rows([_row_out("a", "nope", GOLD), _row_out("b", "nope", GOLD), _row_out("c", "x", [])])["outcomes"]
    result = paired_test(left, right, key="call_correct")
    assert result["n"] == 2 and result["left_only"] == 1 and result["right_only"] == 0
    assert result["difference"] == 1 and result["p_exact_mcnemar"] == 1.0
    assert GIST_LAYOUTS[0] == "full"
