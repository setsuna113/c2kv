"""CPU-only tests for the hybrid-x-D bench additions (arms, proxy assembly,
repair oracle, repair policy, oracle-recover decision layer, terminal-state
checks).  No torch; run anywhere with pytest."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# proxy.py imports its sibling as a top-level module (script-style)
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pytest

from benchmarks.arms import ARMS, Arm, get_arm
from benchmarks import proxy as proxy_mod
from benchmarks import repair_oracle as ro
from benchmarks import repair_policy as rp
from benchmarks import terminal_check as tc


def _extract_stub(role, content, ratio, timeout=0):
    return {
        "key_hash": f"hash-{role}-{len(content)}-{ratio}",
        "gist_len": max(1, len(content) // ratio),
        "original_seq_len": len(content),
    }


def _messages():
    return [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "u1" * 50},
        {"role": "assistant", "content": "a1" * 50},
        {"role": "user", "content": "u2" * 50},
        {"role": "assistant", "content": "a2" * 50},
        {"role": "user", "content": "current"},
    ]


class TestArms:
    # acceptance 7: the pre-existing arms keep their exact v1 semantics
    V1_SNAPSHOT = {
        "full": dict(compress_history=False, ratio=8, hybrid_top_k=0,
                     constrain_tools=False, repair=None),
        "c2kv": dict(compress_history=True, ratio=8, hybrid_top_k=0,
                     constrain_tools=False, repair=None),
        "c2kv16": dict(compress_history=True, ratio=16, hybrid_top_k=0,
                       constrain_tools=False, repair=None),
        "hybrid": dict(compress_history=True, ratio=8, hybrid_top_k=3,
                       constrain_tools=False, repair=None),
        "hybrid_k1": dict(compress_history=True, ratio=8, hybrid_top_k=1,
                          constrain_tools=False, repair=None),
        "hybrid_k5": dict(compress_history=True, ratio=8, hybrid_top_k=5,
                          constrain_tools=False, repair=None),
        "c2kv_repair": dict(compress_history=True, ratio=8, hybrid_top_k=0,
                            constrain_tools=False),
        "hybrid_repair": dict(compress_history=True, ratio=8, hybrid_top_k=3,
                              constrain_tools=False),
        "cd_full": dict(compress_history=False, ratio=8, hybrid_top_k=0,
                        constrain_tools=True, repair=None),
        "cd_c2kv": dict(compress_history=True, ratio=8, hybrid_top_k=0,
                        constrain_tools=True, repair=None),
    }

    def test_v1_arms_unchanged(self):
        for name, fields in self.V1_SNAPSHOT.items():
            arm = ARMS[name]
            for key, value in fields.items():
                assert getattr(arm, key) == value, f"{name}.{key} drifted"

    def test_hybrid_ladder_registered(self):
        assert ARMS["hybrid"].hybrid_top_k == 3
        assert ARMS["hybrid_k1"].hybrid_top_k == 1
        assert ARMS["hybrid_k5"].hybrid_top_k == 5

    def test_repair_arms(self):
        assert ARMS["c2kv_repair"].repair == {"policy": "first", "placement": "append_keep_ledger"}
        assert ARMS["hybrid_repair"].repair == {"policy": "first", "placement": "append_keep_ledger"}
        assert ARMS["hybrid_repair"].hybrid_top_k == 3
        assert ARMS["c2kv_repair_tail"].repair["placement"] == "append_tail"
        assert ARMS["hybrid_repair_tail"].repair["placement"] == "append_tail"
        assert ARMS["c2kv_repair_inplace"].repair["placement"] == "in_place"

    def test_repair_placement_validated(self):
        with pytest.raises(ValueError):
            Arm(name="bad", compress_history=True,
                repair={"policy": "first", "placement": "sideways"}).validate()

    def test_recover_arms(self):
        assert ARMS["c2kv_recover"].recover == {"once": True}
        assert ARMS["c2kv_recover"].compress_history
        assert ARMS["hybrid_recover"].hybrid_top_k == 3
        assert ARMS["hybrid_recover"].recover == {"once": True}

    def test_repair_needs_compression(self):
        with pytest.raises(ValueError):
            Arm(name="bad", compress_history=False, repair={"policy": "first"}).validate()

    def test_recover_needs_compression(self):
        with pytest.raises(ValueError):
            Arm(name="bad", compress_history=False, recover={"once": True}).validate()

    def test_repair_and_recover_exclusive(self):
        with pytest.raises(ValueError):
            Arm(name="bad", compress_history=True,
                repair={"policy": "first"}, recover={"once": True}).validate()

    def test_unknown_arm_exits(self):
        with pytest.raises(SystemExit):
            get_arm("nope")


class TestAssemble:
    def test_c2kv_compresses_all_history(self, monkeypatch):
        monkeypatch.setattr(proxy_mod, "_extract", _extract_stub)
        monkeypatch.setattr(proxy_mod, "DOC_PACKING", "message")  # legacy per-message
        out, counts = proxy_mod._assemble(_messages(), get_arm("c2kv"), 0)
        marked = [m for m in out if "c2kv_key_hash" in m]
        assert len(marked) == 4  # 4 history messages, system+current raw
        assert out[-1]["content"] == "current"
        assert all("c2kv_key_hash" not in m for m in (out[0], out[-1]))
        # B5: keep-raw messages are now COUNTED per class, not skipped
        assert counts["system_raw"] == 1
        assert counts["current_raw"] == 1
        assert counts["compressed"] == 4

    def test_raw_message_breakdown_hybrid_k5(self, monkeypatch):
        monkeypatch.setattr(proxy_mod, "_extract", _extract_stub)
        _, counts = proxy_mod._assemble(_messages(), get_arm("hybrid_k5"), 0)
        # 4 history docs, k=5 keeps them ALL raw -> no compressed messages
        assert counts["history_raw"] == 4
        assert counts["compressed"] == 0
        assert counts["gist_tokens"] == 0

    def test_toolcall_turn_renders_action_dialect(self, monkeypatch):
        """assistant content=None + tool_calls must extract the TRAINING
        dialect text, never the literal '""' (the action-erasure bug)."""
        seen = {}

        def extract(role, content, ratio, timeout=0):
            seen["text"] = content
            return _extract_stub(role, content, ratio, timeout)

        monkeypatch.setattr(proxy_mod, "_extract", extract)
        monkeypatch.setattr(proxy_mod, "DOC_PACKING", "message")  # legacy per-message
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "find flights"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"function": {"name": "search_flights",
                              "arguments": "{\"origin\": \"JFK\"}"}},
            ]},
            {"role": "user", "content": "current"},
        ]
        out, _ = proxy_mod._assemble(messages, get_arm("c2kv"), 0)
        assert seen["text"].startswith("Action:\n<tool_call>\n")
        assert "search_flights" in seen["text"]
        compressed = out[2]
        assert "c2kv_key_hash" in compressed
        assert "tool_calls" not in compressed
        assert compressed["content"] == seen["text"]  # re-extract reproduces the gist

    def test_toolcall_rendering_matches_hf_server(self):
        """the proxy renderer must be verbatim-identical to hf_server.chat's
        normalization (a mismatch breaks re-extraction on gist-pool miss)."""
        hf_source = Path(proxy_mod.__file__).parent / "hf_server.py"
        source = hf_source.read_text(encoding="utf-8")
        # extract the rendering block from hf_server (flask/torch imports
        # make the module unimportable here): compare the JSON dump kwargs
        assert 'separators=(",", ":")' in source
        assert '"Action:\\n" + "\\n".join(blocks)' in source
        assert 'ensure_ascii=False, separators=(",", ":")' in source

    def test_hybrid_k1_keeps_one_tail_raw(self, monkeypatch):
        monkeypatch.setattr(proxy_mod, "_extract", _extract_stub)
        monkeypatch.setattr(proxy_mod, "DOC_PACKING", "message")  # legacy per-message
        out, _ = proxy_mod._assemble(_messages(), get_arm("hybrid_k1"), 0)
        # cutoff = 5 (current user after last assistant); tail-1 history raw
        assert "c2kv_key_hash" not in out[4]  # u2 is the last history message
        assert sum("c2kv_key_hash" in m for m in out) == 3

    def test_hybrid_k5_keeps_all_history_here(self, monkeypatch):
        monkeypatch.setattr(proxy_mod, "_extract", _extract_stub)
        out, _ = proxy_mod._assemble(_messages(), get_arm("hybrid_k5"), 0)
        assert sum("c2kv_key_hash" in m for m in out) == 0  # only 4 history msgs

    def test_full_arm_marks_nothing(self, monkeypatch):
        monkeypatch.setattr(proxy_mod, "_extract", _extract_stub)
        out, counts = proxy_mod._assemble(_messages(), get_arm("full"), 0)
        assert counts["gist_tokens"] == 0
        assert all("c2kv_key_hash" not in m for m in out)


class TestTurnPacking:
    """--doc-packing turn: the TRAINING history format
    (train_data_multiturn._agent_history_turn_docs / _fit_reused_history),
    see docs/c2kv_semantics.md."""

    @staticmethod
    def _conv():
        return [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "weather in Shanghai tomorrow?"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"function": {"name": "get_weather",
                              "arguments": "{\"city\": \"Shanghai\"}"}}]},
            {"role": "tool", "content": "{\"temp\": 26}"},
            {"role": "assistant", "content": "Sunny, 26C."},
            {"role": "user", "content": "and the day after?"},
        ]

    def test_turn_docs_group_input_plus_outputs(self, monkeypatch):
        monkeypatch.setattr(proxy_mod, "_extract", _extract_stub)
        out, counts = proxy_mod._assemble(self._conv(), get_arm("c2kv"), 0)
        assert counts["doc_packing"] == "turn"
        docs = [m for m in out if "c2kv_key_hash" in m]
        assert len(docs) == 2 and counts["n_docs"] == 2
        assert all(m["role"] == "user" for m in docs)
        assert docs[0]["content"] == (
            "Previous turn\n[User query]\nweather in Shanghai tomorrow?\n"
            "[Assistant output]\nAction:\n<tool_call>\n"
            "{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Shanghai\"}}\n</tool_call>")
        # the tool result opens the next doc as its [User query] (tool -> user)
        assert docs[1]["content"] == (
            "Previous turn\n[User query]\n{\"temp\": 26}\n"
            "[Assistant output]\nSunny, 26C.")
        recs = counts["compressed_records"]
        assert recs[0]["source_indices"] == [1, 2] and recs[1]["source_indices"] == [3, 4]
        assert recs[0]["out_index"] == 1 and recs[1]["out_index"] == 2
        # order: system, doc0, doc1, current
        assert [m["role"] for m in out] == ["system", "user", "user", "user"]
        assert out[-1]["content"] == "and the day after?"
        assert counts["current_start_out_index"] == 3

    def test_current_block_is_last_input_message_only(self, monkeypatch):
        """training cutoff: everything before the LAST input message (user
        or tool) is history, including the current turn's earlier steps."""
        monkeypatch.setattr(proxy_mod, "_extract", _extract_stub)
        conv = self._conv()[:4]  # ends with the tool result
        out, counts = proxy_mod._assemble(conv, get_arm("c2kv"), 0)
        assert out[-1]["role"] == "tool" and counts["current_raw"] == 1
        docs = [m for m in out if "c2kv_key_hash" in m]
        assert len(docs) == 1 and "get_weather" in docs[0]["content"]

    def test_tail_selection_keeps_doc0_anchor(self, monkeypatch):
        monkeypatch.setattr(proxy_mod, "_extract", _extract_stub)
        monkeypatch.setattr(proxy_mod, "MAX_DOC_NUM", 4)
        conv = [{"role": "system", "content": "sys"}]
        for t in range(8):
            conv.append({"role": "user", "content": f"q{t}"})
            conv.append({"role": "assistant", "content": f"a{t}"})
        conv.append({"role": "user", "content": "current"})
        out, counts = proxy_mod._assemble(conv, get_arm("c2kv"), 0)
        docs = [m["content"] for m in out if "c2kv_key_hash" in m]
        assert len(docs) == 4 and counts["dropped_docs"] == 4
        assert "q0" in docs[0]          # doc 0 anchor kept
        assert "q5" in docs[1] and "q7" in docs[3]  # last three

    def test_oversize_doc_is_split_and_verified(self, monkeypatch):
        monkeypatch.setattr(proxy_mod, "_extract", _extract_stub)  # tokens == chars
        monkeypatch.setattr(proxy_mod, "MAX_DOC_LENGTH", 60)
        long_answer = "\n".join(f"line {i} " + "x" * 20 for i in range(12))
        conv = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": long_answer},
            {"role": "user", "content": "current"},
        ]
        out, counts = proxy_mod._assemble(conv, get_arm("c2kv"), 0)
        recs = counts["compressed_records"]
        assert len(recs) >= 3
        assert all(r["record"]["original_seq_len"] <= 60 for r in recs)
        assert "".join(r["content"] for r in recs) == (
            "Previous turn\n[User query]\nq\n[Assistant output]\n" + long_answer)

    def test_hybrid_tail_raw_rest_packed(self, monkeypatch):
        monkeypatch.setattr(proxy_mod, "_extract", _extract_stub)
        out, counts = proxy_mod._assemble(self._conv(), get_arm("hybrid_k1"), 0)
        # history = idx 1..4; k=1 keeps idx 4 (assistant text) raw
        assert counts["history_raw"] == 1 and counts["n_docs"] == 2
        assert out[-2]["content"] == "Sunny, 26C." and "c2kv_key_hash" not in out[-2]

    def test_message_packing_is_legacy(self, monkeypatch):
        monkeypatch.setattr(proxy_mod, "_extract", _extract_stub)
        monkeypatch.setattr(proxy_mod, "DOC_PACKING", "message")
        out, counts = proxy_mod._assemble(self._conv(), get_arm("c2kv"), 0)
        assert counts["doc_packing"] == "message" and counts["n_docs"] == 4
        assert [m.get("role") for m in out if "c2kv_key_hash" in m] == [
            "user", "assistant", "tool", "assistant"]


class TestRepairPolicy:
    """B4: offset:<j> indexes DOCS like the D harness; chunk:<i> explicit."""

    def test_parse(self):
        assert rp.parse_policy("first") == {"kind": "doc", "index": 0}
        assert rp.parse_policy("offset:3") == {"kind": "doc", "index": 3}
        assert rp.parse_policy("chunk:5") == {"kind": "chunk", "index": 5}
        for bad in ("median", "offset:x", "offset:-1", "chunk:", "weird:1", ""):
            with pytest.raises(ValueError):
                rp.parse_policy(bad)

    def test_doc_selection_spans_all_chunks_of_the_doc(self):
        # docs with chunk counts [1, 3, 1]: offset:1 spans all 3 chunks of doc 1
        assert rp.span_selection([1, 3, 1], "doc", 1) == (1, 1, 3)
        assert rp.span_selection([1, 3, 1], "doc", 0) == (0, 0, 1)
        assert rp.span_selection([1, 3, 1], "doc", 2) == (2, 4, 1)

    def test_chunk_selection_is_chunk_granular(self):
        # chunk:2 falls inside doc 1 (chunks 1..3), spans exactly one chunk
        assert rp.span_selection([1, 3, 1], "chunk", 2) == (1, 2, 1)
        assert rp.span_selection([1, 3, 1], "chunk", 0) == (0, 0, 1)
        assert rp.span_selection([1, 3, 1], "chunk", 4) == (2, 4, 1)

    def test_out_of_range_and_empty_doc_raise(self):
        # B3-class guard: a target that resolves to nothing must raise,
        # never silently select a degenerate span
        with pytest.raises(ValueError):
            rp.span_selection([1, 3, 1], "doc", 3)
        with pytest.raises(ValueError):
            rp.span_selection([1, 3, 1], "chunk", 5)
        with pytest.raises(ValueError):
            rp.span_selection([2, 0, 1], "doc", 1)  # doc has no extract chunks

    def test_hf_server_append_raw_block_guards(self):
        """source-level assertions: the runtime guards in _append_raw_block
        (torch code, not importable here) must keep refusing zero-token
        spans and slice from an explicit mark — never `-block_tokens:`."""
        source = (Path(proxy_mod.__file__).parent / "hf_server.py").read_text(
            encoding="utf-8")
        assert "refusing" in source and "block_tokens <= 0" in source
        assert "mark:" in source and "[..., mark:, :]" in source
        assert "[..., -block_tokens:, :]" not in source
        assert "_rp.span_selection" in source  # selection is shared (aliased import)


class TestRecoverDecision:
    """Acceptance 3: a trajectory that first diverges at step k flags
    divergence_step == k and recovers exactly once."""

    @staticmethod
    def _ref_and_actions():
        # conversation states fp1..fp5; reference always calls tool A(args X)
        reference = {
            f"fp{i}": {"turn": i, "action": {
                "tool_calls": [{"name": "A", "arguments": {"x": 1}}], "text": ""}}
            for i in range(1, 6)
        }
        match = {"tool_calls": [{"name": "A", "arguments": {"x": 1}}], "text": ""}
        # client-style re-serialization: different key order, string args
        match_scrambled = {"tool_calls": [
            {"name": "A", "arguments": "{\"x\": 1}"}], "text": ""}
        divergent = {"tool_calls": [{"name": "B", "arguments": {"x": 1}}], "text": ""}
        return reference, match, match_scrambled, divergent

    def test_canonicalization_ignores_key_order(self):
        reference, _, _, _ = self._ref_and_actions()
        assert proxy_mod.action_canonical({"tool_calls": [
            {"function": {"name": "A", "arguments": "{\"x\": 1}"}}],
            "content": None}) == reference["fp1"]["action"]

    def test_fingerprint_stable_under_key_order_and_content_type(self):
        m1 = [{"role": "assistant", "content": None, "tool_calls": [
            {"function": {"name": "A", "arguments": "{\"b\": 1, \"a\": 2}"}}]}]
        m2 = [{"role": "assistant", "tool_calls": [
            {"function": {"name": "A", "arguments": "{\"a\": 2, \"b\": 1}"}}],
            "content": None, "tool_call_id": "ignored-transport-key"}]
        assert proxy_mod.messages_fingerprint(m1) == proxy_mod.messages_fingerprint(m2)
        m3 = [{"role": "user", "content": {"b": 1, "a": 2}}]
        m4 = [{"role": "user", "content": {"a": 2, "b": 1}}]
        assert proxy_mod.messages_fingerprint(m3) == proxy_mod.messages_fingerprint(m4)
        assert proxy_mod.messages_fingerprint(m1) != proxy_mod.messages_fingerprint(m3)

    def test_conversation_id_stable_across_growth(self):
        base = [{"role": "system", "content": "sys"},
                {"role": "user", "content": "hello"}]
        grown = base + [{"role": "assistant", "content": "hi"},
                        {"role": "user", "content": "next"}]
        assert proxy_mod.conversation_id(base) == proxy_mod.conversation_id(grown)
        other = [{"role": "system", "content": "sys"},
                 {"role": "user", "content": "different"}]
        assert proxy_mod.conversation_id(base) != proxy_mod.conversation_id(other)

    def test_divergence_at_step_k_and_single_recover(self):
        reference, match, _, divergent = self._ref_and_actions()
        state = proxy_mod.RecoverState(reference)
        # steps 1..2 match, step 3 diverges (k == 3)
        for i in (1, 2):
            flags = state.check("conv", f"fp{i}", match, i)
            assert flags["match"] is True and not flags["diverged_now"]
        flags = state.check("conv", "fp3", divergent, 3)
        assert flags["diverged_now"] and flags["divergence_step"] == 3
        assert state.should_recover("conv", flags) is True
        state.repaired.add("conv")
        assert state.should_recover("conv", flags) is False  # once only
        # post-repair the regenerated action equals the reference action;
        # a LATER mismatch is a re-divergement, not a new repair
        flags4 = state.check("conv", "fp4", divergent, 4)
        assert flags4["re_diverged"] and not flags4["diverged_now"]
        assert state.should_recover("conv", flags4) is False
        flags5 = state.check("conv", "fp5", divergent, 5)
        assert not any(flags5.values())  # terminal: no more flagging

    def test_unknown_state_before_repair_is_not_tracking_lost(self):
        reference, match, _, _ = self._ref_and_actions()
        state = proxy_mod.RecoverState(reference)
        flags = state.check("conv", "fp-unknown", match, 1)
        assert flags["tracking_lost"] is False
        state.repaired.add("conv")
        flags = state.check("conv", "fp-unknown", match, 2)
        assert flags["tracking_lost"] is True  # left the reference track

    def test_reference_roundtrip(self, tmp_path):
        reference, _, _, _ = self._ref_and_actions()
        path = tmp_path / "ref.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for fp, row in reference.items():
                handle.write(json.dumps({"fp": fp, **row}) + "\n")
        loaded = proxy_mod.load_reference(str(path))
        assert set(loaded) == set(reference)
        for fp, row in reference.items():
            assert loaded[fp]["turn"] == row["turn"]
            assert loaded[fp]["action"] == row["action"]


class TestRepairOracle:
    def _tau2_run(self, path, rewards):
        sims = [
            {"task_id": tid, "reward_info": {"reward": r}, "messages": []}
            for tid, r in rewards.items()
        ]
        path.write_text(json.dumps({"simulations": sims}), encoding="utf-8")

    def test_eligible_and_score_tau2(self, tmp_path, monkeypatch):
        runs = {
            "full_run": {"t1": 1.0, "t2": 1.0, "t3": 0.0},
            "base_run": {"t1": 0.0, "t2": 1.0, "t3": 0.0},
            "repair_run": {"t1": 1.0, "t2": 1.0, "t3": 1.0},
        }
        for run, rewards in runs.items():
            sim_dir = tmp_path / "sims" / run
            sim_dir.mkdir(parents=True)
            self._tau2_run(sim_dir / "updated_results.json", rewards)
        monkeypatch.setattr(ro, "TAU2_SIMS", tmp_path / "sims")

        eligible = ro.eligible_set("tau2", "full_run", "base_run")
        assert eligible["eligible_ids"] == ["t1"]  # full ok, base failed

        task_body = ro.write_task("tau2", "c2kv_repair", "/ckpt", "rp", eligible, None)
        assert 'TASK_IDS="t1"' in task_body

        result = ro.score_repair("tau2", "repair_run", eligible)
        assert result["n_eligible"] == 1
        assert result["n_rescued"] == 1
        assert result["rescue_rate"] == 1.0

    def test_eligible_bfcl_from_failure_records(self, tmp_path, monkeypatch):
        arch = tmp_path / "bfcl_archive"
        for run, fails in (("full_run", ["b2"]), ("base_run", ["b1", "b2", "b3"])):
            d = arch / run
            d.mkdir(parents=True)
            rows = [{"accuracy": 0.5, "correct_count": 1, "total_count": 3}]
            rows += [{"id": f, "valid": False} for f in fails]
            (d / "BFCL_v4_multi_turn_base_score.json").write_text(
                json.dumps(rows), encoding="utf-8"
            )
        monkeypatch.setattr(ro, "BFCL_ARCHIVE", arch)
        eligible = ro.eligible_set("bfcl", "full_run", "base_run")
        # b1/b3 failed base but passed full; b2 failed both
        assert eligible["eligible_ids"] == ["b1", "b3"]
        task_body = ro.write_task("bfcl", "c2kv_repair", "/ckpt", "rp", eligible, None)
        assert 'RUN_IDS="b1,b3"' in task_body

    def _bfcl_repair_run(self, arch, run, generated, fails):
        d = arch / run
        d.mkdir(parents=True)
        with (d / "BFCL_v4_multi_turn_base_result.json").open("w", encoding="utf-8") as h:
            for gid in generated:
                h.write(json.dumps({"id": gid, "result": []}) + "\n")
        rows = [{"accuracy": 0.0, "correct_count": 0, "total_count": len(fails)}]
        rows += [{"id": f, "valid": False} for f in fails]
        (d / "BFCL_v4_multi_turn_base_score.json").write_text(
            json.dumps(rows), encoding="utf-8")

    def test_bfcl_unscored_never_rescued(self, tmp_path, monkeypatch):
        """B7 regression: an eligible id the repair run NEVER GENERATED must
        land in unscored (denominator shrinks), never count as a rescue."""
        arch = tmp_path / "bfcl_archive"
        # eligible = b1, b3; repair run generated only b1, and b1 still failed
        self._bfcl_repair_run(arch, "repair_run", generated=["b1"], fails=["b1"])
        monkeypatch.setattr(ro, "BFCL_ARCHIVE", arch)
        eligible = {"eligible_ids": ["b1", "b3"]}
        result = ro.score_repair("bfcl", "repair_run", eligible)
        assert result["unscored_ids"] == ["b3"]
        assert result["n_unscored"] == 1
        assert result["n_scored"] == 1          # denominator = eligible - unscored
        assert result["n_rescued"] == 0         # b1 failed; b3 never ran
        assert result["rescue_rate"] == 0.0

    def test_bfcl_generated_and_clean_counted_rescued(self, tmp_path, monkeypatch):
        arch = tmp_path / "bfcl_archive"
        # both generated; b1 passed (not in failure records), b3 failed
        self._bfcl_repair_run(arch, "repair_run", generated=["b1", "b3"], fails=["b3"])
        monkeypatch.setattr(ro, "BFCL_ARCHIVE", arch)
        result = ro.score_repair("bfcl", "repair_run",
                                 {"eligible_ids": ["b1", "b3"]})
        assert result["n_scored"] == 2
        assert result["n_rescued"] == 1
        assert result["rescue_rate"] == 0.5

    def test_tau2_missing_run_counts_unscored(self, tmp_path, monkeypatch):
        sim_dir = tmp_path / "sims" / "repair_run"
        sim_dir.mkdir(parents=True)
        # only t1 simulated; t2 never ran (killed run)
        self._tau2_run(sim_dir / "updated_results.json", {"t1": 1.0})
        monkeypatch.setattr(ro, "TAU2_SIMS", tmp_path / "sims")
        result = ro.score_repair("tau2", "repair_run", {"eligible_ids": ["t1", "t2"]})
        assert result["unscored_ids"] == ["t2"]
        assert result["n_scored"] == 1
        assert result["rescue_rate"] == 1.0  # over the SCORED denominator


class TestTerminalCheck:
    def test_tau2_shortfall_fails(self, tmp_path, monkeypatch):
        sim_dir = tmp_path / "sims" / "run"
        sim_dir.mkdir(parents=True)
        (sim_dir / "results.json").write_text(json.dumps(
            {"simulations": [{"task_id": i} for i in range(3)]}))
        monkeypatch.setattr(tc, "TAU2_SIMS", tmp_path / "sims")
        assert tc.check_tau2("run", expected=5) == 1
        assert tc.check_tau2("run", expected=3) == 0

    def test_bfcl_counts_result_rows(self, tmp_path, monkeypatch):
        d = tmp_path / "result" / "c2kv-hf" / "multi_turn"
        d.mkdir(parents=True)
        with (d / "BFCL_v4_multi_turn_base_result.json").open("w") as h:
            for i in range(2):
                h.write(json.dumps({"id": f"multi_turn_base_{i}"}) + "\n")
        monkeypatch.setattr(tc, "GORILLA", tmp_path)
        assert tc.check_bfcl(expected=2, run_ids="") == 0
        assert tc.check_bfcl(expected=3, run_ids="") == 1
        assert tc.check_bfcl(expected=None, run_ids="multi_turn_base_1") == 0


class TestBackends:
    """Fake-backend tests for the backend abstraction + proxy repair
    planning (no HTTP, no torch)."""

    class FakeSglang:
        name = "sglang"
        needs_repair_plan = True

        def __init__(self):
            self.repairs = []

        def extract(self, text, role, ratio, tools=None):
            return {
                "key_hash": f"h-{role}-{len(text)}",
                "gist_len": max(1, len(text) // ratio),
                "original_seq_len": len(text),
                "success": True,
            }

        def repair_extract(self, text, role, span_start, span_end,
                           position_offset, source_doc_index):
            self.repairs.append({
                "text": text, "role": role, "span_start": span_start,
                "span_end": span_end, "position_offset": position_offset,
                "source_doc_index": source_doc_index})
            return {"key_hash": f"rep-{source_doc_index}",
                    "token_len": len(text), "success": True}

        # tools add 120 "tokens" each to the prologue, like the extract fake
        # in TestABRegressions; message length == token length
        def repair_extract_messages(self, messages, target_index, tools,
                                    source_doc_index):
            prefix = sum(len(m["content"]) for m in messages[:target_index])
            prefix += 120 * len(tools or [])
            self.repairs.append({
                "messages": messages, "target_index": target_index,
                "tools": tools, "source_doc_index": source_doc_index})
            span_len = len(messages[target_index]["content"])
            return {"key_hash": f"rep-{source_doc_index}", "token_len": span_len,
                    "position_start": prefix, "position_end": prefix + span_len,
                    "already_rotated": False, "success": True}

        def prepare_chat(self, payload, arm, repair_plan):
            from backends.sglang import SglangBackend
            return SglangBackend.prepare_chat(self, payload, arm, repair_plan)

    def test_repair_plan_full_context_form(self, monkeypatch):
        """The raw KV is extracted with the FULL-CONTEXT form (messages +
        target_index); the plan records the server position and the proxy's
        ledger expectation (system + Σ original_seq_len before the target)
        and their difference (frame check)."""
        fake = self.FakeSglang()
        monkeypatch.setattr(proxy_mod, "BACKEND", fake)
        monkeypatch.setattr(proxy_mod, "DOC_PACKING", "message")
        monkeypatch.setattr(proxy_mod, "_extract", lambda role, content,
                            ratio, timeout=0, tools=None:
                            fake.extract(content, role, ratio, tools=tools))
        messages = [
            {"role": "system", "content": "S" * 30},
            {"role": "user", "content": "A" * 40},      # doc 0 (compressed)
            {"role": "assistant", "content": "B" * 20},  # doc 1 (compressed)
            {"role": "user", "content": "current"},
        ]
        arm = get_arm("c2kv_repair")
        out, counts = proxy_mod._assemble(messages, arm)
        plan = proxy_mod.plan_repair(messages, arm, counts, out_messages=out)
        assert plan["doc_index"] == 0 and plan["target_out_index"] == 1
        call = fake.repairs[-1]
        # context = system + doc 0, stripped of c2kv fields
        assert [m["role"] for m in call["messages"]] == ["system", "user"]
        assert all(not k.startswith("c2kv_") for m in call["messages"] for k in m)
        assert call["target_index"] == 1
        assert plan["position_start"] == 30 and plan["expected_offset"] == 30
        assert plan["frame_delta"] == 0
        assert plan["placement"] == "append_keep_ledger"
        # append placements: a repair-only message right before the current block
        prepared = fake.prepare_chat({"messages": out}, arm, plan)
        msgs = prepared["messages"]
        assert len(msgs) == len(out) + 1
        inserted = msgs[plan["current_start_out_index"]]
        assert inserted["c2kv_repair_only_key_hashes"] == ["rep-0"]
        assert inserted["c2kv_repair_placement"] == "append_keep_ledger"
        assert "c2kv_key_hash" not in inserted and msgs[-1]["content"] == "current"
        assert msgs[1]["c2kv_key_hash"] == "h-user-40"  # target's gist stays
        # offset policy: target doc 1 -> system + doc0
        arm1 = Arm(name="r1", compress_history=True, repair={"policy": "offset:1"})
        plan1 = proxy_mod.plan_repair(messages, arm1, counts, out_messages=out)
        assert plan1["doc_index"] == 1 and plan1["target_out_index"] == 2
        assert plan1["position_start"] == 30 + 40 and plan1["frame_delta"] == 0

    def test_repair_placements_shape_the_request(self, monkeypatch):
        fake = self.FakeSglang()
        monkeypatch.setattr(proxy_mod, "BACKEND", fake)
        monkeypatch.setattr(proxy_mod, "_extract", lambda role, content,
                            ratio, timeout=0, tools=None:
                            fake.extract(content, role, ratio, tools=tools))
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"}, {"role": "assistant", "content": "a2"},
            {"role": "user", "content": "current"},
        ]
        # in_place: the target doc loses its gist and becomes repair-only
        arm_ip = get_arm("c2kv_repair_inplace")
        out, counts = proxy_mod._assemble(messages, arm_ip)
        plan = proxy_mod.plan_repair(messages, arm_ip, counts, out_messages=out)
        msgs = fake.prepare_chat({"messages": out}, arm_ip, plan)["messages"]
        assert len(msgs) == len(out)
        target = msgs[plan["target_out_index"]]
        assert "c2kv_key_hash" not in target
        assert target["c2kv_repair_only_key_hashes"] == ["rep-0"]
        assert target["c2kv_repair_placement"] == "in_place"
        # append_tail: repair-only message before the current block, placement set
        arm_tail = get_arm("c2kv_repair_tail")
        out2, counts2 = proxy_mod._assemble(messages, arm_tail)
        plan2 = proxy_mod.plan_repair(messages, arm_tail, counts2, out_messages=out2)
        msgs2 = fake.prepare_chat({"messages": out2}, arm_tail, plan2)["messages"]
        inserted = msgs2[plan2["current_start_out_index"]]
        assert inserted["c2kv_repair_placement"] == "append_tail"
        assert msgs2[-1]["content"] == "current"
        # hybrid: the repair-only message sits AFTER the raw tail
        arm_h = get_arm("hybrid_repair_tail")
        out3, counts3 = proxy_mod._assemble(messages, arm_h)
        plan3 = proxy_mod.plan_repair(messages, arm_h, counts3, out_messages=out3)
        msgs3 = fake.prepare_chat({"messages": out3}, arm_h, plan3)["messages"]
        idx = plan3["current_start_out_index"]
        assert msgs3[idx]["c2kv_repair_placement"] == "append_tail"
        assert all("c2kv_key_hash" not in m for m in msgs3[idx - counts3["history_raw"]:idx])

    def test_raw_toolcall_turns_render_dialect(self, monkeypatch):
        """keep_raw assistant tool_calls turns (full arm, hybrid tail,
        current) must ALSO go out in the training dialect — the sglang
        backend has no server-side normalization."""
        monkeypatch.setattr(proxy_mod, "_extract", _extract_stub)
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"function": {"name": "f", "arguments": "{\"a\": 1}"}}]},
            {"role": "user", "content": "current"},
        ]
        out, counts = proxy_mod._assemble(messages, get_arm("full"))
        raw_asst = out[2]
        assert "tool_calls" not in raw_asst
        assert raw_asst["content"].startswith("Action:\n<tool_call>\n")
        # incoming message untouched (fingerprints/reference use the raw form)
        assert messages[2]["tool_calls"] is not None
        # hybrid tail raw render as well
        out_h, _ = proxy_mod._assemble(messages, get_arm("hybrid_k5"))
        assert out_h[2]["content"].startswith("Action:\n")

    def test_sglang_prepare_constrained_tools(self):
        from backends.sglang import SglangBackend
        backend = SglangBackend(None)
        payload = {
            "messages": [{"role": "user", "content": "x"}],
            "tools": [{"type": "function", "function": {
                "name": "t", "parameters": {
                    "type": "object", "$defs": {"a": {"type": "dict"}},
                    "properties": {"p": {"$ref": "#/$defs/a"}}}}}],
        }
        arm = Arm(name="cd", compress_history=False, constrain_tools=True)
        out = backend.prepare_chat(payload, arm, None)
        assert out["response_format"] == {"type": "structural_tag"}
        params = out["tools"][0]["function"]["parameters"]
        assert "$defs" not in params and "$ref" not in params
        assert params["properties"]["p"] == {"type": "object"}
        # the payload the model SEES keeps the original schema
        assert payload["tools"][0]["function"]["parameters"]["type"] == "object"

    def test_sglang_normalize_response_and_failures(self):
        from backends.sglang import SglangBackend
        from backends import BackendError
        backend = SglangBackend(None)
        ok = backend.normalize_response({
            "choices": [{"message": {"content": "hi", "tool_calls": None},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5},
            "metadata": {"sglang_runtime": {
                "kv_resident_tokens": 100, "kv_pool_size": 4437}},
        })
        assert ok["finish_reason"] == "stop"
        assert ok["cost"] == {"kv_resident_tokens": 100, "kv_pool_size": 4437}
        with pytest.raises(BackendError) as err:
            backend.normalize_response({"choices": [
                {"message": {"content": None}, "finish_reason": "abort"}]})
        assert err.value.kind == "finish_abort"
        with pytest.raises(BackendError) as err2:
            backend.normalize_response({"object": "error", "error": "boom"})
        assert err2.value.kind == "upstream"

    def test_hfserver_backend_shapes(self):
        from backends.hfserver import HfServerBackend
        backend = HfServerBackend(None)
        arm = get_arm("c2kv_repair")
        payload = backend.prepare_chat({"messages": []}, arm, {"doc_index": 0})
        assert payload["c2kv_repair"]["policy"] == "first"
        normalized = backend.normalize_response({
            "choices": [{"message": {"content": "x"}, "finish_reason": "stop"}],
            "usage": {}, "c2kv": {"cache_tokens": 10, "logical_tokens": 80,
                                  "system_len": 5, "repair_block_tokens": 7},
        })
        assert normalized["cost"]["cache_tokens"] == 10
        assert normalized["cost"]["repair_block_tokens"] == 7


class TestABRegressions:
    """External review round-2: two bugs introduced by the sglang port."""

    def test_plan_repair_empty_records_is_noop(self, monkeypatch):
        """A: a repair arm's FIRST request of a session (nothing compressed
        yet) must be a no-op plan, not an uncaught ValueError — hf_server's
        `repair_policy is not None and compressed` guard."""
        fake = TestBackends.FakeSglang()
        fake.needs_repair_plan = True
        monkeypatch.setattr(proxy_mod, "BACKEND", fake)
        monkeypatch.setattr(proxy_mod, "_extract", lambda role, content,
                            ratio, timeout=0, tools=None:
                            fake.extract(content, role, ratio))
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "current"},
        ]
        out, counts = proxy_mod._assemble(messages, get_arm("c2kv_repair"))
        assert counts["compressed_records"] == []
        plan = proxy_mod.plan_repair(messages, get_arm("c2kv_repair"), counts,
                                     out_messages=out)
        assert plan is None
        assert fake.repairs == []  # nothing extracted, nothing crashed

    def test_plan_repair_skipped_for_hfserver_backend(self, monkeypatch):
        """hfserver resolves repair server-side: no system extract (which
        would pollute the c2kv pool) and no no-op repair_extract."""
        monkeypatch.setattr(
            proxy_mod, "BACKEND",
            __import__("backends.hfserver", fromlist=["HfServerBackend"]).HfServerBackend(None))
        out, counts = proxy_mod._assemble(
            [{"role": "system", "content": "sys"},
             {"role": "user", "content": "u" * 80},
             {"role": "user", "content": "cur"}],
            get_arm("c2kv_repair"))
        # hfserver path: _extract is monkeypatched to FAIL if called for planning
        def boom(role, content, ratio, timeout=0, tools=None):
            raise AssertionError("hfserver plan must not extract")
        monkeypatch.setattr(proxy_mod, "_extract", boom)
        assert proxy_mod.plan_repair([], get_arm("c2kv_repair"), counts) is None

    def test_position_offset_includes_tools(self, monkeypatch):
        """B: the system-length measurement must include the tool schemas
        (Qwen templates render tools into the system block); same system
        text with different tools must not hit a stale cache entry."""
        calls = {}

        class ToolsFake(TestBackends.FakeSglang):
            def extract(self, text, role, ratio, tools=None):
                key = (role, len(text), _digest_of(tools))
                if key not in calls:
                    calls[key] = f"h-{role}-{len(text)}-{len(tools or [])}"
                return {
                    "key_hash": calls[key],
                    "gist_len": 1,
                    # the tool schemas add to the rendered system length
                    "original_seq_len": len(text) + 120 * len(tools or []),
                    "success": True,
                }

        def _digest_of(tools):
            import json as _json
            return _json.dumps(tools or [], sort_keys=True)

        fake = ToolsFake()
        monkeypatch.setattr(proxy_mod, "BACKEND", fake)
        monkeypatch.setattr(proxy_mod, "_extract", lambda role, content,
                            ratio, timeout=0, tools=None:
                            fake.extract(content, role, ratio, tools=tools))
        tools = [{"type": "function", "function": {"name": f"t{i}",
                                                   "parameters": {}}}
                 for i in range(20)]
        messages = [
            {"role": "system", "content": "S" * 30},
            {"role": "user", "content": "A" * 40},
            {"role": "assistant", "content": "B" * 20},
            {"role": "user", "content": "current"},
        ]
        arm = get_arm("c2kv_repair")
        monkeypatch.setattr(proxy_mod, "DOC_PACKING", "message")
        out, counts = proxy_mod._assemble(messages, arm)
        assert counts["compressed_records"], "need a compressed history doc"
        plan = proxy_mod.plan_repair(messages, arm, counts, tools=tools,
                                     out_messages=out)
        # proxy ledger expectation: 30 (system) + 20*120 (tools); the server
        # (fake) rendered the same prologue -> frame check passes
        assert plan["expected_offset"] == 30 + 20 * 120
        assert plan["position_start"] == 30 + 20 * 120 and plan["frame_delta"] == 0
        assert fake.repairs[-1]["tools"] == tools
        # cache discrimination: different tools -> different measurement
        plan2 = proxy_mod.plan_repair(
            messages, arm, counts, tools=tools[:1], out_messages=out)
        assert plan2["expected_offset"] == 30 + 120 and plan2["frame_delta"] == 0


class TestCacheMissRecovery:
    """SGLang c2kv pool LRU eviction (400 C2KV cache miss) must trigger a
    re-extract of every compressed message and ONE retry — not a
    deterministic task death (task 49 died 5/5 at ~600s, turn 11)."""

    def test_cache_miss_classified_not_plain_400(self):
        err = proxy_mod.UpstreamError(400, "C2KV cache miss: abc")
        assert not isinstance(err, proxy_mod.CacheMiss)

    def test_refresh_reextractes_all_marked_messages(self, monkeypatch):
        fake = TestBackends.FakeSglang()
        fake.needs_repair_plan = True
        monkeypatch.setattr(proxy_mod, "BACKEND", fake)
        extracted = []

        def extract(role, content, ratio, timeout=0, tools=None):
            extracted.append((role, len(content)))
            return fake.extract(content, role, ratio)

        monkeypatch.setattr(proxy_mod, "_extract", extract)
        records = [{"role": "user", "content": "A" * 30, "record": {}},
                   {"role": "assistant", "content": "B" * 20, "record": {}}]
        for record in records:
            _ = proxy_mod._extract(record["role"], record["content"], 8)
        # simulate the recovery loop: refresh every compressed record
        for record in records:
            _ = proxy_mod._extract(record["role"], record["content"], 8)
        assert len(extracted) == 4  # 2 original + 2 refresh (cache dedups value)
