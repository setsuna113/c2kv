"""Contracts of the tool-definition memory axis (benchmarks/toolmemory.py).

Pinned here: the training protocol text and document format (byte-identical
to c2kv-next-training/python/next_compression/tools.py), the fixed lexical
ranker, the chunk cut, the carrier placement / index shift, and the
checkpoint contract.  No model, no server: extraction is a fake.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import toolmemory  # noqa: E402


# ---------------------------------------------------------------------------
# fakes
# ---------------------------------------------------------------------------


class FakeTokenizer:
    """Deterministic stand-in for NativeTokenizer.native_ids.

    A user message renders as [1, <one id per whitespace token>, 2]; a system
    message as [3, ..., 4]; ``tools=`` appends 7 ids per tool so the raw
    prologue is measurable; generation appends [9].
    """

    def native_ids(self, messages, *, tools=None, generation=False):
        ids = []
        for message in messages:
            open_id, close_id = (3, 4) if message.get("role") == "system" else (1, 2)
            body = [100 + (hash(token) % 1000) for token in str(message.get("content", "")).split()]
            ids += [open_id, *body, close_id]
        if tools:
            ids += [5] * (7 * len(tools))
        if generation:
            ids.append(9)
        return tuple(ids)


def fake_extract(token_ids, ratio, projection_set):
    assert projection_set == "tool"
    digest = hashlib.sha256(json.dumps([list(token_ids), ratio]).encode()).hexdigest()
    return {"key_hash": digest, "gist_len": (len(token_ids) + ratio - 1) // ratio,
            "original_seq_len": len(token_ids), "cache_hit": False}


def write_t0_checkpoint(directory: Path, *, ratios=(8, 12), variant="T0", domain="tool",
                        profile=toolmemory.RENDER_PROFILE):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(json.dumps({
        "history_memory_compression_domain": domain,
        "history_memory_variant": variant,
        "history_memory_render_profile": profile,
        "history_memory_supported_ratios": list(ratios),
        "history_memory_corpus_identity": "abc",
    }), encoding="utf-8")
    (directory / "trainer_state.json").write_text(
        json.dumps({"global_step": 1034, "parameter_version": 1034, "completed": True}),
        encoding="utf-8")
    return directory


TOOLS = [
    {"type": "function", "function": {"name": "book_flight", "description": "Book a flight",
                                      "parameters": {"type": "object", "properties": {"dest": {"type": "string"}}}}},
    {"type": "function", "function": {"name": "cancel_order", "description": "Cancel an order",
                                      "parameters": {"type": "object", "properties": {"id": {"type": "string"}}}}},
    {"type": "function", "function": {"name": "get_weather", "description": "Weather for a city",
                                      "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}},
]


def memory(tmp_path, spec_text="t0:r8", tokenizer=None):
    checkpoint = write_t0_checkpoint(tmp_path / "checkpoint-1034")
    spec = toolmemory.parse_tool_memory_spec(spec_text)
    return toolmemory.ToolMemory(spec, checkpoint, fake_extract, tokenizer=tokenizer or FakeTokenizer())


# ---------------------------------------------------------------------------
# spec / contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text, expected", [
    ("", None), ("none", None), ("raw", None),
    ("t0:r8", ("t0", 8, "uniform", 0, "t0_r8")),
    ("t0:r12:uniform", ("t0", 12, "uniform", 0, "t0_r12")),
    ("T0:R8:HYBRID3", ("t0", 8, "hybrid", 3, "t0_r8_hybrid3")),
])
def test_parse_spec(text, expected):
    spec = toolmemory.parse_tool_memory_spec(text)
    if expected is None:
        assert spec is None
    else:
        assert (spec.encoder, spec.ratio, spec.layout, spec.top_k, spec.name) == expected


@pytest.mark.parametrize("text", ["t0:r4", "t1:r8", "t0:r8:hybrid0", "t0:r8:random3", "r8"])
def test_parse_spec_rejects(text):
    with pytest.raises(ValueError):
        toolmemory.parse_tool_memory_spec(text)


def test_checkpoint_contract_rejects_wrong_encoder(tmp_path):
    spec = toolmemory.parse_tool_memory_spec("t0:r8")
    for index, kwargs in enumerate(({"variant": "T1"}, {"domain": "history"},
                                    {"profile": "other"}, {"ratios": (4, 12)})):
        directory = write_t0_checkpoint(tmp_path / f"bad-{index}", **kwargs)
        with pytest.raises(toolmemory.ToolMemoryError, match="tool_checkpoint"):
            toolmemory.load_tool_checkpoint_contract(directory, spec)
    contract = toolmemory.load_tool_checkpoint_contract(write_t0_checkpoint(tmp_path / "ok"), spec)
    assert contract["variant"] == "T0" and contract["trainer_state"]["global_step"] == 1034


# ---------------------------------------------------------------------------
# protocol, documents, ranker, chunks
# ---------------------------------------------------------------------------


def test_protocol_block_is_the_training_text():
    head = toolmemory.TOOL_PROTOCOL_HEAD
    tail = toolmemory.TOOL_PROTOCOL_TAIL
    assert head.startswith("# Tools\n\nYou may call one or more functions")
    assert "Tool definitions are available in compressed memory." in head
    assert head.endswith("<tools>") and tail.startswith("\n</tools>\n\n")
    assert toolmemory.protocol_block([]) == head + tail
    one = toolmemory.protocol_block([TOOLS[0]])
    assert one == head + "\n" + json.dumps(TOOLS[0], ensure_ascii=False, separators=(",", ":")) + tail


def test_documents_match_t0_format():
    snapshots = [toolmemory.tool_snapshot(t) for t in TOOLS]
    docs = toolmemory.t0_documents(snapshots, [2, 0])
    assert docs == ({"type": "tool_definition", "tool_index": 2, "tool": snapshots[2]},
                    {"type": "tool_definition", "tool_index": 0, "tool": snapshots[0]})
    assert toolmemory.document_envelope(docs[0]).startswith('{"type":"tool_definition","tool_index":2,"tool":{')


def test_lexical_ranker_weighs_name_matches_four_times():
    order = toolmemory.lexical_rank(TOOLS, "please cancel_order the order")
    assert order[0] == 1
    assert toolmemory.lexical_rank(TOOLS, "") == (0, 1, 2)
    messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "weather in Paris city"},
                {"role": "assistant", "content": "ok"}, {"role": "user", "content": "book_flight to Rome"}]
    assert toolmemory.query_text(messages) == "book_flight to Rome"
    spec = toolmemory.parse_tool_memory_spec("t0:r8:hybrid1")
    assert toolmemory.native_indices(TOOLS, spec, messages) == (0,)
    assert toolmemory.native_indices(TOOLS, toolmemory.parse_tool_memory_spec("t0:r8"), messages) == ()


def test_document_chunks_cut_with_overlap_and_limits():
    spec = toolmemory.ToolMemorySpec(ratio=8, chunk_tokens=10, chunk_overlap=3, max_chunks=48)
    long_doc = {"type": "tool_definition", "tool_index": 0, "tool": {"name": " ".join(f"w{i}" for i in range(20))}}
    chunks = toolmemory.document_chunks(FakeTokenizer().native_ids, [long_doc], spec)
    ids = FakeTokenizer().native_ids(({"role": "user", "content": toolmemory.document_envelope(long_doc)},))
    assert [c.token_ids for c in chunks][0] == ids[:10]
    assert chunks[1].source_token_start == 7 and chunks[1].token_ids == ids[7:17]
    assert chunks[-1].source_token_end == len(ids)
    assert all(c.event_id == chunks[0].event_id for c in chunks)
    assert [c.part_index for c in chunks] == list(range(len(chunks)))
    with pytest.raises(toolmemory.ToolMemoryError, match="tool_chunks_over_limit"):
        toolmemory.document_chunks(FakeTokenizer().native_ids, [long_doc],
                                   toolmemory.ToolMemorySpec(ratio=8, chunk_tokens=10, chunk_overlap=3, max_chunks=1))


# ---------------------------------------------------------------------------
# plan + carriers
# ---------------------------------------------------------------------------


def test_plan_uniform_compresses_every_tool_and_rewrites_system(tmp_path):
    mem = memory(tmp_path, "t0:r8")
    payload = {"messages": [{"role": "system", "content": "Be helpful"},
                            {"role": "user", "content": "cancel_order 42"}], "tools": TOOLS}
    plan = mem.plan(payload)
    assert plan is not None
    assert plan.messages[0]["content"] == "Be helpful\n\n" + toolmemory.protocol_block([])
    assert plan.messages[1] == payload["messages"][1]
    assert plan.info["n_tools"] == 3 and plan.info["n_native"] == 0 and plan.info["n_documents"] == 3
    assert plan.info["n_chunks"] == 3
    assert plan.info["gist_tokens"] == sum(r["gist_len"] for r in plan.records)
    assert plan.info["raw_tool_prologue_tokens"] == 21
    carriers = plan.carriers()
    assert len(carriers) == 3
    assert all(c["role"] == "user" and c["content"] == "" and c["c2kv_ratio"] == 8 for c in carriers)
    assert [c["c2kv_key_hash"] for c in carriers] == [r["key_hash"] for r in plan.records]
    assert all(toolmemory.is_carrier(c) for c in carriers)
    assert set(toolmemory.strip_carrier_fields(carriers[0])) == {"role", "content", "c2kv_key_hash", "c2kv_ratio"}
    # memo: the same documents extract once per ratio
    mem.plan(payload)
    assert mem.stats["chunk_extracts"] == 3 and mem.stats["chunk_cache_hits"] == 3


def test_plan_hybrid_keeps_top_k_native_and_inserts_system_when_missing(tmp_path):
    mem = memory(tmp_path, "t0:r12:hybrid1")
    payload = {"messages": [{"role": "user", "content": "get_weather in Paris"}], "tools": TOOLS}
    plan = mem.plan(payload)
    assert plan.messages[0]["role"] == "system"
    assert plan.messages[0]["content"] == toolmemory.protocol_block([toolmemory.tool_snapshot(TOOLS[2])])
    assert plan.info["native_indices"] == [2] and plan.info["native_names"] == ["get_weather"]
    assert plan.info["n_documents"] == 2 and plan.spec.ratio == 12
    assert toolmemory.with_protocol_system(plan.messages, plan.protocol) == plan.messages  # idempotent


def test_plan_without_tools_is_none_and_hybrid_over_catalog_fails(tmp_path):
    mem = memory(tmp_path, "t0:r8:hybrid3")
    assert mem.plan({"messages": [{"role": "user", "content": "hi"}]}) is None
    assert mem.stats["skipped_no_tools"] == 1
    with pytest.raises(toolmemory.ToolMemoryError, match="no_compressed_remainder"):
        mem.plan({"messages": [{"role": "user", "content": "hi"}], "tools": TOOLS})


def test_extract_length_mismatch_is_rejected(tmp_path):
    def bad_extract(token_ids, ratio, projection_set):
        return {"key_hash": "k", "gist_len": 1, "original_seq_len": len(token_ids) + 1}
    checkpoint = write_t0_checkpoint(tmp_path / "ckpt")
    mem = toolmemory.ToolMemory(toolmemory.parse_tool_memory_spec("t0:r8"), checkpoint, bad_extract,
                                tokenizer=FakeTokenizer())
    with pytest.raises(toolmemory.ToolMemoryError, match="extract_length"):
        mem.plan({"messages": [{"role": "user", "content": "x"}], "tools": TOOLS[:1]})


def test_insert_carriers_after_system_and_shift_ledger():
    out = [{"role": "system", "content": "s"}, {"role": "user", "content": "h1"},
           {"role": "user", "content": "", "c2kv_key_hash": "gist"}, {"role": "user", "content": "current"}]
    counts = {"current_start_out_index": 3, "task_packet_out_index": 1,
              "compressed_records": [{"out_index": 2, "role": "user"}]}
    carriers = [{"role": "user", "content": "", "c2kv_key_hash": "t1", toolmemory.CARRIER_MARK: {"event_id": "e"}},
                {"role": "user", "content": "", "c2kv_key_hash": "t2", toolmemory.CARRIER_MARK: {"event_id": "e"}}]
    new, new_counts = toolmemory.insert_carriers(out, counts, carriers)
    assert [m.get("c2kv_key_hash") for m in new] == [None, "t1", "t2", None, "gist", None]
    assert new_counts["current_start_out_index"] == 5
    assert new_counts["task_packet_out_index"] == 3
    assert new_counts["compressed_records"][0]["out_index"] == 4
    assert new_counts["tool_memory_carriers"] == 2 and new_counts["tool_memory_insert_at"] == 1
    assert counts["current_start_out_index"] == 3  # input untouched
    again, again_counts = toolmemory.insert_carriers(new, new_counts, carriers)
    assert again == new and again_counts == new_counts  # idempotent


def test_insert_carriers_without_system_goes_first():
    out = [{"role": "user", "content": "u"}]
    new, counts = toolmemory.insert_carriers(out, {"current_start_out_index": 0}, [
        {"role": "user", "content": "", "c2kv_key_hash": "t", toolmemory.CARRIER_MARK: {}}])
    assert toolmemory.is_carrier(new[0]) and new[1]["content"] == "u"
    assert counts["current_start_out_index"] == 1
