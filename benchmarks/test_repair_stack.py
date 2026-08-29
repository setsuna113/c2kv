"""CPU-only tests for the hybrid-x-D bench additions (arms, proxy assembly,
repair oracle).  No torch; run anywhere with pytest."""
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
    def test_hybrid_ladder_registered(self):
        assert ARMS["hybrid"].hybrid_top_k == 3
        assert ARMS["hybrid_k1"].hybrid_top_k == 1
        assert ARMS["hybrid_k5"].hybrid_top_k == 5

    def test_repair_arms(self):
        assert ARMS["c2kv_repair"].repair == {"policy": "first"}
        assert ARMS["hybrid_repair"].repair == {"policy": "first"}
        assert ARMS["hybrid_repair"].hybrid_top_k == 3

    def test_repair_needs_compression(self):
        with pytest.raises(ValueError):
            Arm(name="bad", compress_history=False, repair={"policy": "first"}).validate()

    def test_unknown_arm_exits(self):
        with pytest.raises(SystemExit):
            get_arm("nope")


class TestAssemble:
    def test_c2kv_compresses_all_history(self, monkeypatch):
        monkeypatch.setattr(proxy_mod, "_extract", _extract_stub)
        out, gist, orig, n = proxy_mod._assemble(_messages(), get_arm("c2kv"), 0)
        marked = [m for m in out if "c2kv_key_hash" in m]
        assert len(marked) == 4  # 4 history messages, system+current raw
        assert out[-1]["content"] == "current"
        assert all("c2kv_key_hash" not in m for m in (out[0], out[-1]))

    def test_hybrid_k1_keeps_one_tail_raw(self, monkeypatch):
        monkeypatch.setattr(proxy_mod, "_extract", _extract_stub)
        out, *_ = proxy_mod._assemble(_messages(), get_arm("hybrid_k1"), 0)
        # cutoff = 5 (current user after last assistant); tail-1 history raw
        assert "c2kv_key_hash" not in out[4]  # u2 is the last history message
        assert sum("c2kv_key_hash" in m for m in out) == 3

    def test_hybrid_k5_keeps_all_history_here(self, monkeypatch):
        monkeypatch.setattr(proxy_mod, "_extract", _extract_stub)
        out, *_ = proxy_mod._assemble(_messages(), get_arm("hybrid_k5"), 0)
        assert sum("c2kv_key_hash" in m for m in out) == 0  # only 4 history msgs

    def test_full_arm_marks_nothing(self, monkeypatch):
        monkeypatch.setattr(proxy_mod, "_extract", _extract_stub)
        out, gist, *_ = proxy_mod._assemble(_messages(), get_arm("full"), 0)
        assert gist == 0
        assert all("c2kv_key_hash" not in m for m in out)


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
