# -*- coding: utf-8 -*-
"""Torch-gated tests for agent/f_timing_fork.py (tiny random model, CPU).

No weights, no dataset, no network: a 2-layer gist-enabled Qwen3 is built from
config and driven with the deterministic whitespace tokenizer from
``train.train_data_joint`` (the same fixture pattern as
``agent/test_eval_joint_next_action_c2kv.py``:544-570).

Every repo import that pulls torch lives INSIDE a test function or fixture, and
the module-level ``importorskip`` makes the whole file SKIP (not ERROR) on the
torch-free dev box.

Coverage:
a. defer-prefix geometry: ``cache_length == system_length + shared_gist_tokens
   + last_chunk_tokens`` and the raw segment is exactly the last chunk;
b. the position invariant both branches must satisfy, and the guard that
   aborts the run when they do not;
c. the shared prefix is bit-comparable across branches: branch A's leading
   system+shared-gist KV matches branch B's;
d. a monkeypatch spy on ``_prefill_tokens_with_cache_maybe_gist`` proving the
   raw segment is placed at ``system_length + sum(original shared chunk
   lengths)`` with ``use_gist=True``;
e. delayed (uncompressed) history turns: branch A prepends them to the plain
   prompt exactly as joint's ``_generate_with_prefix`` does, and branch B
   refuses to build rather than dropping them;
f. end-to-end ``greedy_core``: two complete rows per eligible example, and a
   second run with ``--resume`` regenerates nothing.

Run from the repo root (WSL venv with torch):
  python -m pytest agent/test_f_timing_fork_gpu.py -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _sub in ("python", "python/inference", "agent"):
    _path = str(_REPO_ROOT / _sub)
    if _path not in sys.path:
        sys.path.insert(0, _path)


# ---------------------------------------------------------------------------
# Fixtures (all heavy imports are function-local).
# ---------------------------------------------------------------------------


MAX_DOC_LENGTH = 16
MAX_DOC_NUM = 8
L_MIN = 4
RATIO = 2


def _tokenizer():
    from train.train_data_joint import _WhitespaceSelfTestTokenizer

    return _WhitespaceSelfTestTokenizer()


def _tiny_gist_model(tokenizer):
    """2-layer gist model, same config recipe as the joint eval's CPU tests."""

    import torch

    from models.qwen3 import Qwen3Config, Qwen3ForCausalLM

    config = Qwen3Config(
        vocab_size=1024,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=2048,
        rope_theta=10000.0,
        rms_norm_eps=1e-6,
        gist_type="dynamic-interleave",
        gist_param="qkv",
        gist_residual_type="embed-mean",
        gist_overlap=2,
        gist_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        attn_implementation="eager",
    )
    torch.manual_seed(0)
    model = Qwen3ForCausalLM(config)
    model.eval()
    return model


def _example(qid="s0:0", session="s0"):
    from train.train_data_joint import JointExample

    return JointExample(
        qid=qid,
        session_id=session,
        tool_documents=[
            "<TOOL> <NAME> get_weather <DESC> weather for one city </TOOL>",
            "<TOOL> <NAME> search_files <DESC> search files under a path </TOOL>",
        ],
        history_documents=[
            " ".join(f"hist{index}word{word}" for word in range(8))
            for index in range(3)
        ],
        current_messages=[{"role": "user", "content": "what is the weather in paris"}],
        answer='Action:\n<tool_call>\n{"name":"get_weather","arguments":{"city":"Paris"}}\n</tool_call>',
        system_prompt="you are a test agent",
        subset="test",
    )


def _args(tmp_path, **overrides):
    import f_timing_fork as FT

    argv = [
        "--model", "dummy-checkpoint",
        "--output_file", str(tmp_path / "f_fork.jsonl"),
        "--prereg_file", str(_REPO_ROOT / "configs" / "bdf_pilot" / "f_prereg.md"),
        "--max_doc_length", str(MAX_DOC_LENGTH),
        "--max_doc_num", str(MAX_DOC_NUM),
        "--max_system_length", "64",
        "--max_prompt_tokens", "64",
        "--max_new_tokens", "4",
        "--override_ratio", str(RATIO),
        "--l_min", str(L_MIN),
        "--device_type", "cpu",
        "--assert_greedy_repeat", "0",
    ]
    for key, value in overrides.items():
        argv.extend([f"--{key}", str(value)])
    return FT.parse_args(argv)


@pytest.fixture()
def harness(tmp_path):
    import f_timing_fork as FT

    tokenizer = _tokenizer()
    model = _tiny_gist_model(tokenizer)
    args = _args(tmp_path)
    example = _example()
    tool_chunks, history_chunks, skip_reason, _meta = FT._joint_chunks(
        tokenizer, example, args
    )
    assert skip_reason is None, skip_reason
    assert len(history_chunks) >= 2, (
        "fixture must produce at least 2 history chunks for E2 to pass; got "
        f"{len(history_chunks)}"
    )
    return {
        "FT": FT,
        "tokenizer": tokenizer,
        "model": model,
        "args": args,
        "example": example,
        "chunks": [*tool_chunks, *history_chunks],
        "history_chunks": history_chunks,
        "tool_chunks": tool_chunks,
    }


# ---------------------------------------------------------------------------
# Eligibility on the fixture.
# ---------------------------------------------------------------------------


def test_fixture_example_is_eligible(harness):
    FT = harness["FT"]
    ok, reason, meta = FT._check_eligibility(
        harness["tokenizer"], harness["example"], harness["args"]
    )
    assert ok is True, reason
    assert meta["history_chunk_count"] == len(harness["history_chunks"])
    assert meta["last_chunk_tokens"] == len(harness["chunks"][-1])
    assert meta["fork_chunk_index"] == len(harness["chunks"]) - 1


def test_short_last_chunk_is_rejected_by_e3(harness, tmp_path):
    FT = harness["FT"]
    args = _args(tmp_path, l_min=MAX_DOC_LENGTH)
    ok, reason, _meta = FT._check_eligibility(
        harness["tokenizer"], harness["example"], args
    )
    if ok:
        pytest.skip("fixture last chunk happens to saturate max_doc_length")
    assert reason.startswith("last_chunk_tokens<")


# ---------------------------------------------------------------------------
# a. defer-prefix geometry
# ---------------------------------------------------------------------------


def test_defer_prefix_cache_length_is_system_plus_shared_gist_plus_raw(harness):
    FT = harness["FT"]
    prefix, reason = FT._build_defer_prefix(
        harness["model"], harness["tokenizer"], harness["example"], harness["args"]
    )
    assert reason is None and prefix is not None
    last_chunk_len = len(harness["chunks"][-1])
    assert prefix["raw_segment_tokens"] == last_chunk_len
    assert prefix["cache_length"] == (
        prefix["system_length"] + prefix["shared_gist_tokens"] + last_chunk_len
    )
    # The gist side really is compressed: fewer gist tokens than shared tokens.
    assert prefix["shared_gist_tokens"] < prefix["shared_original_tokens"]
    # doc_length is the ORIGINAL span, not the cache span.
    assert prefix["doc_length"] == sum(len(chunk) for chunk in harness["chunks"])
    assert prefix["use_gist"] is True
    assert prefix["branch"] == "defer"


def test_compress_now_prefix_has_no_raw_segment(harness):
    FT = harness["FT"]
    prefix, reason = FT._build_compress_now_prefix(
        harness["model"], harness["tokenizer"], harness["example"], harness["args"]
    )
    assert reason is None and prefix is not None
    assert prefix["raw_segment_tokens"] == 0
    assert prefix["branch"] == "compress_now"
    assert prefix["cache_length"] == prefix["system_length"] + prefix["gist_tokens"]


# ---------------------------------------------------------------------------
# b. position invariant
# ---------------------------------------------------------------------------


def test_both_branches_place_the_current_turn_at_the_same_position(harness):
    FT = harness["FT"]
    prefix_a, _ = FT._build_compress_now_prefix(
        harness["model"], harness["tokenizer"], harness["example"], harness["args"]
    )
    prefix_b, _ = FT._build_defer_prefix(
        harness["model"], harness["tokenizer"], harness["example"], harness["args"]
    )
    left = prefix_a["system_length"] + prefix_a["doc_length"]
    right = prefix_b["system_length"] + prefix_b["doc_length"]
    assert left == right
    assert left == prefix_a["system_length"] + sum(
        len(chunk) for chunk in harness["chunks"]
    )
    # Cache lengths, by contrast, MUST differ: defer keeps the last chunk raw.
    assert prefix_b["cache_length"] > prefix_a["cache_length"]


def test_position_invariant_guard_aborts_on_mismatch(harness):
    FT = harness["FT"]
    assert FT._assert_position_invariant(120, 120, "s0:0") == 120
    with pytest.raises(SystemExit, match="implementation-invalid"):
        FT._assert_position_invariant(120, 121, "s0:0")


# ---------------------------------------------------------------------------
# c. shared prefix KV equality
# ---------------------------------------------------------------------------


def test_shared_prefix_gist_kv_matches_across_branches(harness):
    import torch

    FT = harness["FT"]
    prefix_a, _ = FT._build_compress_now_prefix(
        harness["model"], harness["tokenizer"], harness["example"], harness["args"]
    )
    prefix_b, _ = FT._build_defer_prefix(
        harness["model"], harness["tokenizer"], harness["example"], harness["args"]
    )
    shared = prefix_b["system_length"] + prefix_b["shared_gist_tokens"]
    assert shared > prefix_b["system_length"]
    for layer_a, layer_b in zip(prefix_a["cache"].layers, prefix_b["cache"].layers):
        assert torch.allclose(
            layer_a.keys[..., :shared, :], layer_b.keys[..., :shared, :],
            atol=1e-4, rtol=1e-4,
        )
        assert torch.allclose(
            layer_a.values[..., :shared, :], layer_b.values[..., :shared, :],
            atol=1e-4, rtol=1e-4,
        )


# ---------------------------------------------------------------------------
# d. raw-segment placement spy
# ---------------------------------------------------------------------------


def test_raw_segment_is_prefilled_at_the_original_shared_length(harness, monkeypatch):
    FT = harness["FT"]
    calls = []
    original = FT._prefill_tokens_with_cache_maybe_gist

    def _spy(model, input_ids, **kwargs):
        calls.append({"input_length": int(input_ids.shape[1]), **kwargs})
        return original(model, input_ids, **kwargs)

    monkeypatch.setattr(FT, "_prefill_tokens_with_cache_maybe_gist", _spy)
    prefix, reason = FT._build_defer_prefix(
        harness["model"], harness["tokenizer"], harness["example"], harness["args"]
    )
    assert reason is None and prefix is not None
    assert len(calls) == 1
    call = calls[0]
    shared_original = sum(len(chunk) for chunk in harness["chunks"][:-1])
    assert call["past_length"] == prefix["system_length"] + shared_original
    assert call["use_gist"] is True
    assert call["input_length"] == len(harness["chunks"][-1])


# ---------------------------------------------------------------------------
# e. delayed (uncompressed) history turns
# ---------------------------------------------------------------------------


def test_defer_branch_refuses_delayed_history_turns(harness, tmp_path):
    """A delayed turn would put raw content on BOTH sides of the fork point.

    ``--delay_recent_turns`` is a B-line flag the F driver does not expose
    today, so this simulates the day it does: branch B must abort loudly, not
    silently drop the delayed turn.
    """

    FT = harness["FT"]
    args = _args(tmp_path)
    args.delay_recent_turns = 1
    _tools, history_chunks, reason, meta = FT._joint_chunks(
        harness["tokenizer"], harness["example"], args
    )
    assert reason is None
    assert meta["raw_history_ids"], "fixture must actually delay a turn here"
    assert len(history_chunks) >= 2, "E2 must still pass, or the test proves nothing"
    with pytest.raises(SystemExit, match="implementation-invalid"):
        FT._build_defer_prefix(
            harness["model"], harness["tokenizer"], harness["example"], args
        )


def test_generate_branch_prepends_delayed_raw_ids(harness, monkeypatch):
    """Branch A rides delayed turns in front of the current turn (joint :852)."""

    FT = harness["FT"]
    seen = []
    original = FT._generate_from_input_ids

    def _spy(model, tokenizer, **kwargs):
        seen.append({
            "input_width": int(kwargs["input_ids"].shape[1]),
            "position_start": int(kwargs["position_ids"][0, 0]),
            "position_width": int(kwargs["position_ids"].shape[1]),
        })
        return original(model, tokenizer, **kwargs)

    monkeypatch.setattr(FT, "_generate_from_input_ids", _spy)

    def _run(raw_ids):
        prefix, reason = FT._build_compress_now_prefix(
            harness["model"], harness["tokenizer"], harness["example"], harness["args"]
        )
        assert reason is None
        assert prefix["raw_history_ids"] == [], "F runs with delay_recent_turns=0"
        if raw_ids is not None:
            prefix["raw_history_ids"] = raw_ids
        return FT._generate_branch(
            harness["model"],
            harness["tokenizer"],
            harness["example"],
            prefix,
            harness["args"],
            branch="compress_now",
            arm_pass="greedy_core",
            rollout_index=0,
        )

    plain = _run(None)
    raw_ids = [7, 11, 13]
    with_raw = _run(raw_ids)

    # The delayed ids are really fed to the model, and they occupy the
    # positions right after the compressed grid -- the start position is
    # unchanged because doc_length still counts only the grid.
    assert seen[1]["input_width"] == seen[0]["input_width"] + len(raw_ids)
    assert seen[1]["position_width"] == seen[0]["position_width"] + len(raw_ids)
    assert seen[1]["position_start"] == seen[0]["position_start"]
    assert plain["raw_recent_tokens"] == 0
    assert with_raw["raw_recent_tokens"] == len(raw_ids)
    # prompt_tokens stays the CURRENT turn only; the residency ledger counts
    # everything that is actually resident.
    assert with_raw["prompt_tokens"] == plain["prompt_tokens"]
    for row in (plain, with_raw):
        assert row["peak_cache_tokens"] == (
            row["cache_tokens"]
            + row["prompt_tokens"]
            + row["raw_recent_tokens"]
            + row["generated_tokens"]
        )


# ---------------------------------------------------------------------------
# f. end-to-end greedy_core + resume
# ---------------------------------------------------------------------------


def _patch_evaluate_boundary(monkeypatch, FT, tokenizer, model, examples):
    monkeypatch.setattr(FT, "_setup_device", lambda device_type: "cpu")
    monkeypatch.setattr(FT, "_resolve_model_checkpoint", lambda path: path)
    monkeypatch.setattr(FT, "_load_tokenizer", lambda args: tokenizer)
    monkeypatch.setattr(FT, "_load_model", lambda args, tok, device: model)
    monkeypatch.setattr(FT, "_load_examples", lambda args: list(examples))


def _read_rows(path):
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_greedy_core_writes_two_complete_rows_and_resume_skips_them(
    harness, tmp_path, monkeypatch
):
    FT = harness["FT"]
    tokenizer = harness["tokenizer"]
    model = harness["model"]
    examples = [_example("s0:0", "s0")]
    _patch_evaluate_boundary(monkeypatch, FT, tokenizer, model, examples)

    args = _args(tmp_path, assert_greedy_repeat=1)
    summary = FT.evaluate(args)
    assert summary["num_eligible"] == 1
    assert summary["arm_set"] == "greedy_core"

    rows = _read_rows(args.output_file)
    live = [row for row in rows if not row.get("skipped")]
    assert len(live) == 2
    branches = sorted(row["branch"] for row in live)
    assert branches == ["compress_now", "defer"]
    for row in live:
        assert row["arm_pass"] == "greedy_core"
        assert row["rollout_index"] == 0
        assert row["qid"] == "s0:0"
        assert row["session_id"] == "s0"
        assert isinstance(row["deterministic_check_pass"], bool)
        assert "pred_action_key" in row and "gold_action_key" in row
        assert row["gold_action_key"] is not None
        assert row["kv_bytes_per_token"] > 0
        assert row["peak_bytes"] == row["peak_cache_tokens"] * row["kv_bytes_per_token"]
        assert row["resident_bytes_measured"] > 0
        assert row["resident_bytes_logical_shared"] > 0
        assert row["git_short_sha"] is None or isinstance(row["git_short_sha"], str)
        assert row["gen_seed_used"] is None  # greedy never seeds
    # Both branches agree on the original prefix length -- the invariant, on disk.
    assert len({row["original_prefix_length"] for row in live}) == 1
    # The honest residency: the fork segment costs MORE than raw alone.
    assert live[0]["fork_segment_logical_ratio"] > 1.0

    # Resume: nothing new is generated.
    second = FT.evaluate(_args(tmp_path, assert_greedy_repeat=0))
    assert second["num_rows_written"] == 0
    assert len(_read_rows(args.output_file)) == len(rows)


def test_ineligible_example_writes_one_skip_row(harness, tmp_path, monkeypatch):
    import dataclasses

    FT = harness["FT"]
    base = _example("s1:0", "s1")
    # JointExample is frozen; one history document violates E2.
    example = dataclasses.replace(base, history_documents=base.history_documents[:1])
    _patch_evaluate_boundary(
        monkeypatch, FT, harness["tokenizer"], harness["model"], [example]
    )
    args = _args(tmp_path)
    summary = FT.evaluate(args)
    assert summary["num_eligible"] == 0
    assert summary["num_ineligible"] == 1
    rows = _read_rows(args.output_file)
    assert len(rows) == 1
    assert rows[0]["skipped"] is True
    assert rows[0]["skip_reason"] == "history_chunks<2"


def test_qid_manifest_fixes_the_order_and_fails_loudly_on_a_missing_qid(
    harness, tmp_path
):
    FT = harness["FT"]
    examples = [_example("s0:1", "s0"), _example("s0:0", "s0")]
    manifest = tmp_path / "qids.json"

    manifest.write_text(json.dumps({"qids": ["s0:0", "s0:1"]}), encoding="utf-8")
    args = _args(tmp_path, qid_manifest=str(manifest))
    assert [ex.qid for ex in FT._select_examples(examples, args)] == ["s0:0", "s0:1"]

    manifest.write_text(json.dumps(["s0:0", "s9:9"]), encoding="utf-8")
    with pytest.raises(SystemExit, match="not reproduced"):
        FT._select_examples(examples, _args(tmp_path, qid_manifest=str(manifest)))


def test_sampled_arm_set_refuses_without_the_sampling_switch(
    harness, tmp_path, monkeypatch
):
    FT = harness["FT"]
    monkeypatch.setattr(FT, "_sampling_supported", lambda: False)
    with pytest.raises(SystemExit, match="sampling switch"):
        FT.evaluate(_args(tmp_path, arm_set="sampled"))


def test_greedy_repeat_check_aborts_on_nondeterminism(harness, monkeypatch):
    FT = harness["FT"]
    with pytest.raises(SystemExit, match="not reproducible"):
        monkeypatch.setattr(
            FT,
            "_generate_branch",
            lambda *a, **k: {"prediction": "drifted output"},
        )
        FT._run_greedy_repeat_check(
            harness["model"],
            harness["tokenizer"],
            harness["example"],
            harness["args"],
            "original output",
            "cpu",
        )
