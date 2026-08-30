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
        assert ARMS["c2kv_repair"].repair == {"policy": "first"}
        assert ARMS["hybrid_repair"].repair == {"policy": "first"}
        assert ARMS["hybrid_repair"].hybrid_top_k == 3

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
        assert "repair_policy.span_selection" in source  # selection is shared


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
