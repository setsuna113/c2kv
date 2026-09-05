# -*- coding: utf-8 -*-
"""Driver-level tests for the downstream-persistence extension of
agent/d_kv_intervene.py (exploratory prereg addendum 2026-08-23).

No model, no dataset, no device: the harness boundary is monkeypatched away
(the pattern of test_f_timing_fork_gpu.py::_patch_evaluate_boundary), the
frozen state is a tmp manifest/bundles pair with REAL sha256_text_file shas
(the pattern of test_d_recipe_guard.py's honest fixtures), and the cache is a
counting stub.  What stays real: parse_args on both sides, the frozen-state
binding, the continuation loop itself (_downstream_rows, _continuation_block,
_crop_cache), resume, and the row schema.

Coverage (spec test numbers 6-14):
  6.  --downstream_turns 0 is byte-identical to no flag under a frozen clock;
  7.  K>0 refuses corr / full / corr_all / sham_mech;
  8.  K caps at 3;
  9.  _continuation_block: clean extension, prefix mismatch, reconstruction
      mismatch, filtered-span gap, empty block;
  10. span exhaustion emits a counted skip at EVERY unreached offset;
  11. the cache admission check is a counted skip and stops the group;
  12. resume last-group semantics (oom retry, K stamp asymmetry);
  13. row schema and stamps (K=0 rows carry none of the new keys);
  14. a full K>0 run requires the downstream smoke marker.

Run from the repo root:
  python -m pytest agent/test_d_downstream_driver.py -v
"""

from __future__ import annotations

import itertools
import json
import re
import sys
import types
from pathlib import Path

import pytest

pytest.importorskip("torch")

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _sub in ("python", "python/inference", "agent"):
    _path = str(_REPO_ROOT / _sub)
    if _path not in sys.path:
        sys.path.insert(0, _path)

import d_kv_intervene as D  # noqa: E402
import eval_agent_history_c2kv as HH  # noqa: E402
from extract_cw_triggers import sha256_text_file  # noqa: E402
from train.train_data_joint import _WhitespaceSelfTestTokenizer  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class _Example:
    """Attribute surface of CompressHistoryExample that the driver touches."""

    def __init__(self, qid, current_messages, original_messages, answer):
        self.qid = qid
        self.history_messages = [{"role": "user", "content": "earlier turn digest"}]
        self.current_messages = current_messages
        self.original_messages = original_messages
        self.answer = answer
        self.system_prompt = "sys"
        self.tools = []


def _session(session_id, n_spans, span_step=1):
    """Spans of one session; each snapshot extends the previous one exactly.
    span_step > 1 leaves index gaps, the shape of filtered-out spans."""
    conv = [{"role": "user", "content": f"query zero of {session_id}"}]
    examples = []
    span_index = 0
    for _ in range(n_spans):
        examples.append(
            _Example(
                qid=f"{session_id}:{span_index}",
                current_messages=[dict(conv[-1])],
                original_messages=[dict(m) for m in conv],
                answer=f"gold answer {span_index}",
            )
        )
        conv = conv + [
            {"role": "assistant", "content": f"gold action {span_index} of {session_id}"},
            {"role": "user", "content": f"query {span_index + span_step} of {session_id}"},
        ]
        span_index += span_step
    return examples


class _FakeCache:
    def __init__(self, length):
        self._length = length

    def get_seq_length(self):
        return self._length

    def crop(self, length):
        assert length <= self._length
        self._length = length

    def grow(self, added):
        self._length += added


class _StubModel:
    def __init__(self):
        self.device = "cpu"
        self.config = types.SimpleNamespace(
            _attn_implementation="eager", max_position_embeddings=1_000_000
        )


_SYSTEM_LENGTH = 10
_HISTORY_LENGTH = 94
_CACHE_LENGTH = 40  # build-time physical slots; pos_gap0 = 64
_PROMPT_TOKENS = 5
_GENERATED_TOKENS = 3


def _stub_generate_one(model, tokenizer, example, run_args, mode, *, return_state=False):
    row = {
        "qid": example.qid,
        "session_id": example.qid.rsplit(":", 1)[0],
        "mode": mode,
        "ratio": run_args.override_ratio,
        "skipped": False,
        "prediction": f"pred {example.qid}",
        "target": example.answer,
        "tool_name_match": False,
        "prompt_tokens": _PROMPT_TOKENS,
        "generated_tokens": _GENERATED_TOKENS,
        "generate_sec": 0.25,
        "latency_sec": 0.25,
        "tbt_sec": 0.05,
        "cache_tokens": _CACHE_LENGTH,
        "gist_tokens": 8,
        "doc_tokens": 64,
    }
    if not return_state:
        return row
    cache = _FakeCache(_CACHE_LENGTH)
    cache.grow(_PROMPT_TOKENS + _GENERATED_TOKENS - 1)  # generate's in-place growth
    prefix = {
        "cache": cache,
        "system_length": _SYSTEM_LENGTH,
        "history_length": _HISTORY_LENGTH,
        "cache_length": _CACHE_LENGTH,
        "doc_tokens": 64,
        "gist_tokens": 8,
        "use_gist": True,
    }
    return row, prefix


# Every example _generate_with_prefix is handed to score a continuation turn,
# in call order — the driver must pass later[j-1], never the trigger.
_PREFIX_CALLS = []


def _stub_generate_with_prefix(model, tokenizer, example, prefix, run_args, mode):
    _PREFIX_CALLS.append(example.qid)
    prefix["cache"].grow(_PROMPT_TOKENS + _GENERATED_TOKENS - 1)
    return {
        "prediction": f"ds pred {example.qid}",
        "target": example.answer,
        "tool_name_match": True,
        "prompt_tokens": _PROMPT_TOKENS,
        "generated_tokens": _GENERATED_TOKENS,
        "generate_sec": 0.2,
        "latency_sec": 0.2,
        "tbt_sec": 0.04,
    }


def _stub_prefill(model, input_ids, past_key_values, past_length, attn_impl, *, use_gist):
    added = int(input_ids.shape[1])
    past_key_values.grow(added)
    return past_key_values, added, 0.01


_FAKE_CODE_SHA = "1234567890abcdef1234567890abcdef12345678"


def _patch_harness(monkeypatch, examples):
    tokenizer = _WhitespaceSelfTestTokenizer()
    seen = {}

    def _record_load_examples(hargs, tok):
        seen["hargs"] = hargs
        return list(examples), {}

    # git HEAD is unresolvable from a Windows-side worktree mounted into WSL;
    # the driver's FATAL-on-unresolvable contract is unaffected — the stamp
    # itself is what the schema test checks.
    monkeypatch.setattr(
        D.subprocess,
        "run",
        lambda *a, **k: types.SimpleNamespace(stdout=_FAKE_CODE_SHA + "\n"),
    )
    monkeypatch.setattr(HH, "_load_tokenizer", lambda hargs: tokenizer)
    monkeypatch.setattr(HH, "_load_examples", _record_load_examples)
    monkeypatch.setattr(HH, "_setup_device", lambda device_type: "cpu")
    monkeypatch.setattr(HH, "_resolve_model_checkpoint", lambda path: path)
    monkeypatch.setattr(HH, "_load_model", lambda margs, tok, device: _StubModel())
    monkeypatch.setattr(HH, "_generate_one", _stub_generate_one)
    monkeypatch.setattr(HH, "_generate_with_prefix", _stub_generate_with_prefix)
    monkeypatch.setattr(HH, "_prefill_tokens_with_cache_maybe_gist", _stub_prefill)
    monkeypatch.setattr(HH, "_clear_device_cache", lambda device: None)
    return seen


def _frozen_state(tmp_path, cw_qids):
    bundles = tmp_path / "bundles.jsonl"
    bundles.write_text(
        "".join(json.dumps({"qid": qid, "no_downstream": False}) + "\n" for qid in cw_qids),
        encoding="utf-8",
    )
    manifest = {
        "rule_version": "d_cw_v1",
        "batch": "batch-TF-test",
        "cw_qids": list(cw_qids),
        "n_base_paired": 10,
        "bundles_sha256": sha256_text_file(bundles),
        "kv_recipe": {"max_doc_length": 768, "max_doc_num": 16, "ratio": 8},
        "source_dialects": {"history": len(cw_qids)},
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path, bundles


def _driver_args(tmp_path, manifest_path, bundles, out_name="rows.jsonl", **over):
    argv = [
        "--arm", str(over.pop("arm", "none")),
        "--manifest", str(manifest_path),
        "--bundles", str(bundles),
        "--sham_plan", str(tmp_path / "absent_plan.json"),
        "--output_file", str(tmp_path / out_name),
        "--device_type", "cpu",
        "--resume", str(over.pop("resume", "False")),
    ]
    for key, value in over.items():
        argv += [f"--{key}", str(value)]
    return D.parse_args(argv)


def _read_rows(path):
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


# ---------------------------------------------------------------------------
# 6. K=0 byte identity (frozen clock)
# ---------------------------------------------------------------------------


def test_downstream_zero_turns_byte_identical(tmp_path, monkeypatch):
    manifest_path, bundles = _frozen_state(tmp_path, ["s1:0"])
    _patch_harness(monkeypatch, _session("s1", 2))
    ticks = itertools.count()
    monkeypatch.setattr(D.time, "perf_counter", lambda: float(next(ticks)))

    D.evaluate(_driver_args(tmp_path, manifest_path, bundles, out_name="plain.jsonl"))
    D.evaluate(
        _driver_args(
            tmp_path, manifest_path, bundles, out_name="zero.jsonl", downstream_turns=0
        )
    )
    assert (tmp_path / "plain.jsonl").read_bytes() == (tmp_path / "zero.jsonl").read_bytes()


# ---------------------------------------------------------------------------
# 7/8. arm restriction and the K cap
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("arm", ["corr", "full", "corr_all", "sham_mech"])
def test_downstream_arm_restriction_fatal(tmp_path, arm):
    manifest_path, bundles = _frozen_state(tmp_path, ["s1:0"])
    args = _driver_args(
        tmp_path, manifest_path, bundles, arm=arm, downstream_turns=1, qids="s1:0"
    )
    with pytest.raises(SystemExit, match="none/sham/corr_re"):
        D.evaluate(args)


def test_downstream_turns_cap_fatal(tmp_path):
    manifest_path, bundles = _frozen_state(tmp_path, ["s1:0"])
    args = _driver_args(
        tmp_path, manifest_path, bundles, downstream_turns=4, qids="s1:0"
    )
    with pytest.raises(SystemExit, match="caps at 3"):
        D.evaluate(args)


def test_downstream_turns_negative_fatal(tmp_path):
    # A negative K is truthy, so without the guard the whole K>0 machinery
    # would engage while range(1, K+1) stays empty — junk-stamped groups
    # marked done by resume (turns -1 >= -1).
    manifest_path, bundles = _frozen_state(tmp_path, ["s1:0"])
    args = _driver_args(
        tmp_path, manifest_path, bundles, downstream_turns=-1, qids="s1:0"
    )
    with pytest.raises(SystemExit, match="negative"):
        D.evaluate(args)


def test_downstream_forces_max_samples_per_session_zero(tmp_path, monkeypatch):
    # The 4->0 forcing must reach HH._load_examples (the offset-0 identity
    # sentinel is blind to this regression: t* rows are identical either way,
    # only later spans silently vanish into d_ds_no_subsequent_turn skips).
    manifest_path, bundles = _frozen_state(tmp_path, ["s1:0"])
    seen = _patch_harness(monkeypatch, _session("s1", 2))
    D.evaluate(
        _driver_args(
            tmp_path, manifest_path, bundles, downstream_turns=1, qids="s1:0",
            out_name="k1.jsonl",
        )
    )
    assert seen["hargs"].max_samples_per_session == 0
    D.evaluate(_driver_args(tmp_path, manifest_path, bundles, out_name="k0.jsonl"))
    assert seen["hargs"].max_samples_per_session == 4


# ---------------------------------------------------------------------------
# 9. the continuation block diff
# ---------------------------------------------------------------------------


def test_continuation_block_diff():
    prev, nxt, third = _session("s1", 3)

    # clean extension: exact slice, gold action included
    block, skip = D._continuation_block(prev, nxt)
    assert skip is None
    assert block == [
        {"role": "user", "content": "query zero of s1"},
        {"role": "assistant", "content": "gold action 0 of s1"},
    ]

    # a filtered-out intermediate span (gap) still yields the full material
    block, skip = D._continuation_block(prev, third)
    assert skip is None
    assert [m["content"] for m in block] == [
        "query zero of s1",
        "gold action 0 of s1",
        "query 1 of s1",
        "gold action 1 of s1",
    ]

    # non-prefix snapshot -> d_ds_prefix_mismatch
    corrupted = _Example(
        qid=nxt.qid,
        current_messages=[dict(m) for m in nxt.current_messages],
        original_messages=[{"role": "user", "content": "a different opening"}]
        + [dict(m) for m in nxt.original_messages[1:]],
        answer=nxt.answer,
    )
    block, skip = D._continuation_block(prev, corrupted)
    assert block is None and skip == "d_ds_prefix_mismatch"

    # normalization-tail mismatch -> d_ds_conv_reconstruction_mismatch
    drifted = _Example(
        qid=nxt.qid,
        current_messages=[{"role": "user", "content": "not what the snapshot holds"}],
        original_messages=[dict(m) for m in nxt.original_messages],
        answer=nxt.answer,
    )
    block, skip = D._continuation_block(prev, drifted)
    assert block is None and skip == "d_ds_conv_reconstruction_mismatch"

    # lui_next == lui_prev -> empty block, no skip
    same_anchor = _Example(
        qid=nxt.qid,
        current_messages=[dict(m) for m in prev.original_messages[-1:]]
        + [{"role": "assistant", "content": "gold action 0 of s1"}],
        original_messages=[dict(m) for m in prev.original_messages]
        + [{"role": "assistant", "content": "gold action 0 of s1"}],
        answer=nxt.answer,
    )
    block, skip = D._continuation_block(prev, same_anchor)
    assert skip is None
    assert block == []


# ---------------------------------------------------------------------------
# 10/11. counted skips
# ---------------------------------------------------------------------------


def test_no_subsequent_turn_emits_counted_skips_at_every_offset(tmp_path, monkeypatch):
    # s1 has only the trigger span; s2 has one later span.
    manifest_path, bundles = _frozen_state(tmp_path, ["s1:0", "s2:0"])
    _patch_harness(monkeypatch, _session("s1", 1) + _session("s2", 2))
    args = _driver_args(
        tmp_path, manifest_path, bundles, downstream_turns=3, qids="s1:0,s2:0"
    )
    D.evaluate(args)
    rows = _read_rows(args.output_file)
    groups = {}
    for row in rows:
        groups.setdefault(row["qid"], []).append(row)

    exhausted = groups["s1:0"]
    assert [row["d_turn_offset"] for row in exhausted] == [0, 1, 2, 3]
    assert not exhausted[0]["skipped"]
    for row in exhausted[1:]:
        assert row["skipped"] and row["skip_reason"] == "d_ds_no_subsequent_turn"
        assert row["d_ds_scored_qid"] is None
    assert exhausted[-1]["d_ds_terminal"] is True
    assert exhausted[-1]["d_ds_offsets_available"] == 0

    partial = groups["s2:0"]
    assert [row["d_turn_offset"] for row in partial] == [0, 1, 2, 3]
    assert not partial[1]["skipped"]
    assert partial[1]["d_ds_scored_qid"] == "s2:1"
    for row in partial[2:]:
        assert row["skipped"] and row["skip_reason"] == "d_ds_no_subsequent_turn"
    assert partial[-1]["d_ds_terminal"] is True
    assert partial[-1]["d_ds_offsets_available"] == 1


def test_downstream_cache_budget_skip(tmp_path, monkeypatch):
    manifest_path, bundles = _frozen_state(tmp_path, ["s1:0"])
    _patch_harness(monkeypatch, _session("s1", 4))
    args = _driver_args(
        tmp_path,
        manifest_path,
        bundles,
        downstream_turns=3,
        qids="s1:0",
        downstream_max_cache_tokens=100,  # < phys + prompt/decode budgets
    )
    D.evaluate(args)
    rows = _read_rows(args.output_file)
    # offset-0 row, then ONE budget skip; the break stops the group.
    assert [row["d_turn_offset"] for row in rows] == [0, 1]
    assert rows[1]["skipped"] and rows[1]["skip_reason"] == "d_ds_cache_over_budget"
    assert rows[1]["d_ds_terminal"] is True
    assert rows[1]["d_ds_offsets_available"] == 3


def test_downstream_oom_break_then_resume_appends_clean_group(tmp_path, monkeypatch):
    """The in-loop OOM handler as the driver actually writes it (literal
    reason, break, terminal on the skip row), then a real end-to-end resume:
    the retried qid appends a fresh complete group, a converged file stops
    growing, and the analyzer loader reads the retry."""
    manifest_path, bundles = _frozen_state(tmp_path, ["s1:0"])
    _patch_harness(monkeypatch, _session("s1", 4))
    calls = {"n": 0}

    def oom_on_second(model, tokenizer, example, prefix, run_args, mode):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("NPU out of memory. Tried to allocate 1.00 GiB")
        return _stub_generate_with_prefix(model, tokenizer, example, prefix, run_args, mode)

    monkeypatch.setattr(HH, "_generate_with_prefix", oom_on_second)
    args = _driver_args(
        tmp_path, manifest_path, bundles, downstream_turns=3, qids="s1:0",
        resume="True", out_name="oom_resume.jsonl",
    )
    D.evaluate(args)
    rows = _read_rows(args.output_file)
    assert [row["d_turn_offset"] for row in rows] == [0, 1, 2]
    assert not rows[1]["skipped"]
    oom_row = rows[2]
    assert oom_row["skipped"] and oom_row["skip_reason"] == "oom"
    assert oom_row["d_ds_scored_qid"] == "s1:2"
    assert oom_row["d_ds_terminal"] is True
    assert oom_row["d_ds_offsets_available"] == 3
    # the literal reason the driver writes is what resume retries on ...
    assert D._load_done_qids(Path(args.output_file), 3) == set()

    # ... and a resumed invocation appends a fresh complete group
    monkeypatch.setattr(HH, "_generate_with_prefix", _stub_generate_with_prefix)
    D.evaluate(args)
    rows = _read_rows(args.output_file)
    assert [row["d_turn_offset"] for row in rows] == [0, 1, 2, 0, 1, 2, 3]
    assert all(not row["skipped"] for row in rows[3:])
    assert rows[-1]["d_ds_terminal"] is True
    assert D._load_done_qids(Path(args.output_file), 3) == {"s1:0"}

    # converged: a third invocation appends nothing
    size_before = Path(args.output_file).stat().st_size
    D.evaluate(args)
    assert Path(args.output_file).stat().st_size == size_before

    # ... and what the driver wrote is what the analyzer loader keeps
    import d_downstream_analysis as A  # noqa: PLC0415

    assert "oom" in A.BREAK_REASONS
    data = A._load_downstream_arm(str(args.output_file))
    assert set(data["rows"]) == {("s1:0", 0), ("s1:0", 1), ("s1:0", 2), ("s1:0", 3)}
    assert not data["skip_counts"], "the superseded oom group never reaches the counters"


def test_downstream_empty_block_continuation(tmp_path, monkeypatch):
    """A next snapshot that extends only by the assistant action (lui_next ==
    lui_prev) yields an empty block: no prefill, no prologue guard, block
    accounting at zero — and still a scored row, never a skip."""
    u0 = {"role": "user", "content": "query zero of s1"}
    a0 = {"role": "assistant", "content": "gold action 0 of s1"}
    prev = _Example("s1:0", [dict(u0)], [dict(u0)], "gold answer 0")
    nxt = _Example(
        "s1:1", [dict(u0), dict(a0)], [dict(u0), dict(a0)], "gold answer 1"
    )
    manifest_path, bundles = _frozen_state(tmp_path, ["s1:0"])
    _patch_harness(monkeypatch, [prev, nxt])
    args = _driver_args(
        tmp_path, manifest_path, bundles, downstream_turns=1, qids="s1:0"
    )
    D.evaluate(args)
    rows = _read_rows(args.output_file)
    assert [row["d_turn_offset"] for row in rows] == [0, 1]
    cont = rows[1]
    assert not cont["skipped"]
    assert cont["d_ds_scored_qid"] == "s1:1"
    assert cont["d_ds_block_tokens"] == 0
    assert cont["d_ds_block_messages"] == 0
    assert cont["d_ds_block_prefill_sec"] == 0.0
    assert cont["target"] == "gold answer 1"


# ---------------------------------------------------------------------------
# 12. resume: last-group semantics
# ---------------------------------------------------------------------------


def _group_lines(qid, turns, offsets, *, offset0_skipped=False, oom_at=None,
                 terminal=True):
    rows = [{
        "qid": qid,
        "d_turn_offset": 0,
        "skipped": offset0_skipped,
        "d_downstream_turns": turns,
    }]
    for offset in offsets:
        row = {"qid": qid, "d_turn_offset": offset, "skipped": False,
               "d_downstream_turns": turns}
        if oom_at == offset:
            row.update({"skipped": True, "skip_reason": "oom"})
        rows.append(row)
    if terminal and rows[1:]:
        rows[-1]["d_ds_terminal"] = True
    return "".join(json.dumps(row) + "\n" for row in rows)


def test_downstream_resume_group_semantics(tmp_path):
    # complete group -> done
    path = tmp_path / "complete.jsonl"
    path.write_text(_group_lines("s1:0", 3, (1, 2, 3)), encoding="utf-8")
    assert D._load_done_qids(path, 3) == {"s1:0"}

    # group containing an oom row -> retried
    path = tmp_path / "oom.jsonl"
    path.write_text(
        _group_lines("s1:0", 3, (1, 2), oom_at=2, terminal=False), encoding="utf-8"
    )
    assert D._load_done_qids(path, 3) == set()

    # oom group FOLLOWED by a clean complete group -> done (last group wins)
    path = tmp_path / "retried.jsonl"
    path.write_text(
        _group_lines("s1:0", 3, (1, 2), oom_at=2, terminal=False)
        + _group_lines("s1:0", 3, (1, 2, 3)),
        encoding="utf-8",
    )
    assert D._load_done_qids(path, 3) == {"s1:0"}

    # recorded K=1 under launch K=3 -> not done (retried)
    path = tmp_path / "smaller_k.jsonl"
    path.write_text(_group_lines("s1:0", 1, (1,)), encoding="utf-8")
    assert D._load_done_qids(path, 3) == set()

    # recorded K=3 under launch K=1 -> fatal, never silently continued
    path = tmp_path / "larger_k.jsonl"
    path.write_text(_group_lines("s1:0", 3, (1, 2, 3)), encoding="utf-8")
    with pytest.raises(SystemExit, match="larger K"):
        D._load_done_qids(path, 1)

    # K=0 file under legacy semantics: unchanged
    path = tmp_path / "legacy.jsonl"
    path.write_text(
        json.dumps({"qid": "a:1", "skipped": False}) + "\n"
        + json.dumps({"qid": "b:2", "skipped": True, "skip_reason": "oom"}) + "\n",
        encoding="utf-8",
    )
    assert D._load_done_qids(path) == {"a:1"}

    # a K=0 launch pointed at a K>0 file -> fatal (legacy scanning would parse
    # continuation rows as done triggers and corrupt group boundaries)
    path = tmp_path / "k0_onto_downstream.jsonl"
    path.write_text(_group_lines("s1:0", 3, (1, 2, 3)), encoding="utf-8")
    with pytest.raises(SystemExit, match="d_downstream_turns"):
        D._load_done_qids(path)


# ---------------------------------------------------------------------------
# 13. row schema and stamps
# ---------------------------------------------------------------------------


def test_downstream_row_schema_and_stamps(tmp_path, monkeypatch):
    manifest_path, bundles = _frozen_state(tmp_path, ["s1:0"])
    _patch_harness(monkeypatch, _session("s1", 3))
    args = _driver_args(
        tmp_path, manifest_path, bundles, downstream_turns=2, qids="s1:0"
    )
    _PREFIX_CALLS.clear()
    D.evaluate(args)
    rows = _read_rows(args.output_file)
    assert [row["d_turn_offset"] for row in rows] == [0, 1, 2]
    # The examples actually handed to _generate_with_prefix, in order: the
    # continuation must score later[j-1], never the trigger.
    assert _PREFIX_CALLS == ["s1:1", "s1:2"]

    manifest_sha = sha256_text_file(manifest_path)
    for row in rows:
        assert row["d_arm"] == "none"
        assert row["d_mode"] == "c2kv"
        assert row["bundle_manifest_sha256"] == manifest_sha
        assert row["sham_plan_sha256"] is None
        assert row["attn_impl_runtime"] == "eager"
        assert isinstance(row["wall_sec"], float)
        assert row["d_downstream_turns"] == 2
        assert re.fullmatch(r"[0-9a-f]{40}", row["d_code_sha"])

    assert rows[0]["d_ds_scored_qid"] == "s1:0"
    for offset, row in zip((1, 2), rows[1:]):
        assert row["d_ds_scored_qid"] == f"s1:{offset}"
        assert row["d_ds_scored_span_index"] == offset
        # scored against the t*+offset example's own gold answer
        assert row["target"] == f"gold answer {offset}"
        assert row["d_ds_block_tokens"] > 0
        assert row["d_ds_block_messages"] == 2
        assert row["d_ds_block_prefill_sec"] == 0.01
        # physical stays short, logical stays raw, gap stays constant
        assert row["d_ds_logical_tokens"] - row["d_ds_cache_tokens"] == (
            _SYSTEM_LENGTH + _HISTORY_LENGTH - _CACHE_LENGTH
        )
        assert row["d_ds_pos_gap"] == _SYSTEM_LENGTH + _HISTORY_LENGTH - _CACHE_LENGTH
        # structural offset-0 fields are re-carried for per-row cost sums
        assert row["doc_tokens"] == rows[0]["doc_tokens"]
        assert row["gist_tokens"] == rows[0]["gist_tokens"]

    # K=0 rows carry none of the downstream keys
    args = _driver_args(tmp_path, manifest_path, bundles, out_name="k0.jsonl")
    D.evaluate(args)
    (k0_row,) = _read_rows(tmp_path / "k0.jsonl")
    for key in (
        "d_turn_offset", "d_ds_scored_qid", "d_downstream_turns", "d_code_sha",
        "d_ds_terminal", "d_ds_offsets_available",
    ):
        assert key not in k0_row
    assert not any(key.startswith("d_ds_") for key in k0_row)


# ---------------------------------------------------------------------------
# 14. full-run smoke gate
# ---------------------------------------------------------------------------


def test_downstream_full_run_requires_smoke_marker(tmp_path, monkeypatch):
    manifest_path, bundles = _frozen_state(tmp_path, ["s1:0"])
    _patch_harness(monkeypatch, _session("s1", 2))
    monkeypatch.delenv("SKIP_DOWNSTREAM_SMOKE_CHECK", raising=False)

    # full run (no --qids / --max_qids), no marker -> fatal
    args = _driver_args(tmp_path, manifest_path, bundles, downstream_turns=3)
    with pytest.raises(SystemExit, match="smoke"):
        D.evaluate(args)

    # --qids restricts the run: no marker needed
    args = _driver_args(
        tmp_path, manifest_path, bundles, downstream_turns=3, qids="s1:0",
        out_name="smoke_qids.jsonl",
    )
    D.evaluate(args)
    assert _read_rows(tmp_path / "smoke_qids.jsonl")

    # marker present -> proceeds
    marker = tmp_path / "smoke.ok"
    marker.write_text("code_sha=x\n", encoding="utf-8")
    args = _driver_args(
        tmp_path, manifest_path, bundles, downstream_turns=3,
        downstream_smoke_ok=str(marker), out_name="with_marker.jsonl",
    )
    D.evaluate(args)
    assert _read_rows(tmp_path / "with_marker.jsonl")

    # deliberate override -> proceeds
    monkeypatch.setenv("SKIP_DOWNSTREAM_SMOKE_CHECK", "1")
    args = _driver_args(
        tmp_path, manifest_path, bundles, downstream_turns=3,
        out_name="override.jsonl",
    )
    D.evaluate(args)
    assert _read_rows(tmp_path / "override.jsonl")
