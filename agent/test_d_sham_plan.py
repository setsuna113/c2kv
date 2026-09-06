# -*- coding: utf-8 -*-
"""CPU-only unit tests for agent/d_sham_plan.py.

No torch and no real tokenizer: the corpus is tokenised by whitespace and the
decode callback rejoins the words, which is enough to exercise every rule in
the planner (ring arithmetic, per-qid seeding, budget equality, the
neutrality gate) and to check the committed corpus asset itself.

Coverage:
a. k* and ring arithmetic, including wrap-around;
b. budget equality as a MULTISET, not just as a total;
c. per-qid seed determinism and cross-qid dispersion;
d. the neutrality gate fires on forbidden characters and vocabulary, and the
   committed configs/bdf_pilot/d_neutral_corpus.txt passes it;
e. T == 1 boundary (k* = 0, no_downstream true);
f. missing / degenerate qids are reported, not silently dropped.

Run from the repo root:
  python -m pytest agent/test_d_sham_plan.py -v
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _sub in ("python", "python/inference", "agent"):
    _path = str(_REPO_ROOT / _sub)
    if _path not in sys.path:
        sys.path.insert(0, _path)

import d_sham_plan as P  # noqa: E402


CORPUS_PATH = _REPO_ROOT / "configs" / "bdf_pilot" / "d_neutral_corpus.txt"


def _word_corpus(text):
    """Whitespace tokenisation: ids are indices into the word list."""
    words = text.split()
    return list(range(len(words))), (lambda ids: " ".join(words[i] for i in ids))


def _doc_table(**qids):
    """qids: qid -> list of doc lengths."""
    return {
        qid: {"session_id": qid.rsplit(":", 1)[0], "n_docs": len(lens), "doc_lens": list(lens)}
        for qid, lens in qids.items()
    }


@pytest.fixture(scope="module")
def corpus():
    text = CORPUS_PATH.read_text(encoding="utf-8")
    ids, decode = _word_corpus(text)
    return text, ids, decode


# --- a. arithmetic ----------------------------------------------------------


def test_k_star_is_the_median_index():
    assert P.k_star_for(1) == 0
    assert P.k_star_for(2) == 0
    assert P.k_star_for(3) == 1
    assert P.k_star_for(4) == 1
    assert P.k_star_for(16) == 7
    # Both halves must be non-empty for T >= 2 (that is the whole point of k*).
    for n_docs in range(2, 20):
        k = P.k_star_for(n_docs)
        assert 0 <= k < n_docs
        assert n_docs - 1 - k >= 1


def test_ring_slice_wraps_and_keeps_length():
    ids = [10, 11, 12, 13]
    assert P.ring_slice(ids, 0, 4) == [10, 11, 12, 13]
    assert P.ring_slice(ids, 3, 3) == [13, 10, 11]
    # Longer than the corpus: keeps wrapping rather than truncating.
    assert P.ring_slice(ids, 2, 6) == [12, 13, 10, 11, 12, 13]
    assert len(P.ring_slice(ids, 1, 17)) == 17


def test_ring_slice_rejects_empty_corpus():
    with pytest.raises(ValueError):
        P.ring_slice([], 0, 3)
    with pytest.raises(ValueError):
        P.corpus_offset(P.SEED, "s:1", 0)


def test_corpus_offset_is_in_range_and_seed_dependent():
    for qid in ("a:1", "b:2", "c:3"):
        assert 0 <= P.corpus_offset(P.SEED, qid, 997) < 997
    assert P.corpus_offset(P.SEED, "a:1", 997) != P.corpus_offset(P.SEED + 1, "a:1", 997)


# --- b. budget --------------------------------------------------------------


def test_budget_equality_is_by_construction(corpus):
    _, ids, decode = corpus
    table = _doc_table(**{"s:1": [30, 40, 50], "s:2": [12, 7], "s:3": [5, 6, 7, 8, 9]})
    qids = sorted(table)
    plan = P.build_plan(table, qids, ids, decode)
    assert plan["budget"]["typed_tokens_total"] == plan["budget"]["sham_tokens_total"]
    assert plan["budget"]["abs_delta_frac"] == 0.0
    assert plan["budget"]["gate"] == "== 0"
    assert plan["budget"]["gate_passed"] is True


def test_span_length_multiset_matches_the_corr_spans(corpus):
    _, ids, decode = corpus
    table = _doc_table(**{"s:1": [30, 40, 50], "s:2": [12, 7], "s:3": [5, 6, 7, 8, 9]})
    qids = sorted(table)
    plan = P.build_plan(table, qids, ids, decode)
    expected = Counter(
        table[qid]["doc_lens"][P.k_star_for(len(table[qid]["doc_lens"]))] for qid in qids
    )
    observed = Counter(len(entry["sham_token_ids"]) for entry in plan["per_qid"].values())
    assert observed == expected
    for qid in qids:
        entry = plan["per_qid"][qid]
        assert entry["span_len"] == len(entry["sham_token_ids"])
        assert entry["span_len"] == table[qid]["doc_lens"][entry["k_star"]]


# --- c. determinism ---------------------------------------------------------


def test_plan_is_deterministic(corpus):
    _, ids, decode = corpus
    table = _doc_table(**{"s:1": [30, 40, 50], "s:2": [12, 7]})
    qids = sorted(table)
    first = P.build_plan(table, qids, ids, decode)
    second = P.build_plan(table, qids, ids, decode)
    assert first == second


def test_different_qids_draw_different_offsets(corpus):
    _, ids, decode = corpus
    table = _doc_table(**{f"s:{i}": [20, 20, 20] for i in range(24)})
    qids = sorted(table)
    plan = P.build_plan(table, qids, ids, decode)
    offsets = {entry["corpus_offset"] for entry in plan["per_qid"].values()}
    # A per-qid hash must not collapse two dozen qids onto a handful of spans.
    assert len(offsets) >= 20


def test_seed_change_moves_every_span(corpus):
    _, ids, decode = corpus
    table = _doc_table(**{"s:1": [40, 40, 40]})
    base = P.build_plan(table, ["s:1"], ids, decode)
    other = P.build_plan(table, ["s:1"], ids, decode, seed=P.SEED + 1)
    assert base["per_qid"]["s:1"]["corpus_offset"] != other["per_qid"]["s:1"]["corpus_offset"]
    assert base["seed"] != other["seed"]


# --- d. neutrality ----------------------------------------------------------


def test_neutrality_violations_detects_structure_and_vocabulary():
    assert P.neutrality_violations("plain neutral prose about wind and sand") == []
    assert any(v.startswith("char:") for v in P.neutrality_violations('a {"b": 1} c'))
    assert any(v == "word:tool" for v in P.neutrality_violations("the tool moved sand"))
    assert any(v == "word:name" for v in P.neutrality_violations("write the name here"))
    # Substrings must not trip the word gate.
    assert P.neutrality_violations("rapidly observed and recalled") == []


def test_committed_corpus_is_neutral(corpus):
    text, ids, _ = corpus
    assert P.neutrality_violations(text) == []
    assert len(text.split()) > 1500, "the corpus must be long enough to host a 768-token span"
    assert len(ids) == len(text.split())


def test_plan_gate_fails_when_a_span_is_not_neutral(corpus):
    _, ids, _ = corpus
    table = _doc_table(**{"s:1": [10, 10, 10]})
    dirty = lambda span_ids: '{"name": "leak"}'  # noqa: E731
    plan = P.build_plan(table, ["s:1"], ids, dirty)
    assert plan["neutrality"]["gate_passed"] is False
    assert plan["neutrality"]["violating_qids"] == ["s:1"]
    assert plan["per_qid"]["s:1"]["neutrality_violations"]


def test_real_corpus_spans_pass_the_gate(corpus):
    _, ids, decode = corpus
    table = _doc_table(**{f"s:{i}": [128, 256, 512, 768, 64] for i in range(30)})
    plan = P.build_plan(table, sorted(table), ids, decode)
    assert plan["neutrality"]["gate_passed"] is True
    assert plan["neutrality"]["violating_qids"] == []


# --- e/f. boundaries --------------------------------------------------------


def test_single_doc_session_marks_no_downstream(corpus):
    _, ids, decode = corpus
    table = _doc_table(**{"s:1": [42], "s:2": [42, 42]})
    plan = P.build_plan(table, ["s:1", "s:2"], ids, decode)
    assert plan["per_qid"]["s:1"]["k_star"] == 0
    assert plan["per_qid"]["s:1"]["no_downstream"] is True
    assert plan["per_qid"]["s:2"]["no_downstream"] is False


def test_missing_and_degenerate_qids_are_reported(corpus):
    _, ids, decode = corpus
    table = _doc_table(**{"s:1": [10, 10, 10], "s:2": [0, 0, 0]})
    plan = P.build_plan(table, ["s:1", "s:2", "s:404"], ids, decode)
    assert plan["missing_qids"] == ["s:404"]
    assert plan["degenerate_qids"] == ["s:2"]
    assert set(plan["per_qid"]) == {"s:1"}
    assert plan["n_qids"] == 1


def test_inconsistent_doc_table_is_fatal(corpus):
    _, ids, decode = corpus
    table = {"s:1": {"n_docs": 5, "doc_lens": [1, 2, 3]}}
    with pytest.raises(SystemExit, match="n_docs"):
        P.build_plan(table, ["s:1"], ids, decode)


# --- f2. main() exit codes for missing / degenerate coverage -----------------


def _run_main(tmp_path, monkeypatch, corpus, doc_lens_by_qid, cw_qids, extra_argv=()):
    """Drive P.main with a whitespace-tokenizer stub over the real corpus."""
    import json

    text, _, _ = corpus
    words = text.split()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"cw_qids": list(cw_qids)}), encoding="utf-8")
    doc_table_path = tmp_path / "d_doc_ids.json"
    doc_table_path.write_text(
        json.dumps(
            {
                "per_qid": {
                    qid: {
                        "session_id": qid.rsplit(":", 1)[0],
                        "n_docs": len(lens),
                        "doc_lens": list(lens),
                    }
                    for qid, lens in doc_lens_by_qid.items()
                }
            }
        ),
        encoding="utf-8",
    )

    class _Tok:
        def __call__(self, value, add_special_tokens=False):
            return {"input_ids": list(range(len(value.split())))}

        def decode(self, ids, skip_special_tokens=False):
            return " ".join(words[i] for i in ids)

    monkeypatch.setattr(P, "_load_tokenizer", lambda path: _Tok())
    return P.main(
        [
            "--doc_table", str(doc_table_path),
            "--manifest", str(manifest_path),
            "--corpus", str(CORPUS_PATH),
            "--tokenizer", "unused",
            "--out", str(tmp_path / "d_sham_plan.json"),
            *extra_argv,
        ]
    )


def test_missing_qid_is_fatal_by_default(tmp_path, monkeypatch, corpus):
    """d_prereg.md §8-4: a missing qid is FATAL, never a skip."""
    table = {"s:1": [10, 10, 10]}
    assert _run_main(tmp_path, monkeypatch, corpus, table, ["s:1", "s:404"]) == 1


def test_degenerate_qid_is_fatal_by_default(tmp_path, monkeypatch, corpus):
    table = {"s:1": [10, 10, 10], "s:2": [0, 0, 0]}
    assert _run_main(tmp_path, monkeypatch, corpus, table, ["s:1", "s:2"]) == 1


def test_allow_missing_downgrades_to_warning(tmp_path, monkeypatch, corpus):
    import json

    table = {"s:1": [10, 10, 10]}
    assert (
        _run_main(tmp_path, monkeypatch, corpus, table, ["s:1", "s:404"], ["--allow_missing"]) == 0
    )
    # The gap is still recorded in the plan; only the exit code is downgraded.
    plan = json.loads((tmp_path / "d_sham_plan.json").read_text(encoding="utf-8"))
    assert plan["missing_qids"] == ["s:404"]


def test_full_coverage_exits_zero(tmp_path, monkeypatch, corpus):
    table = {"s:1": [10, 10, 10], "s:2": [7, 8]}
    assert _run_main(tmp_path, monkeypatch, corpus, table, ["s:1", "s:2"]) == 0


# --- g. the sha convention the driver asserts on -----------------------------


def test_sha_convention_is_shared_and_newline_insensitive(tmp_path):
    """agent/d_kv_intervene.py FATALs when the plan's recorded manifest sha
    disagrees, and the plan may be frozen on a different OS than the run —
    so the digest must not depend on the line ending."""
    import extract_cw_triggers as X

    assert P._sha256_file is X.sha256_text_file
    lf = tmp_path / "lf.json"
    crlf = tmp_path / "crlf.json"
    lf.write_bytes(b'{"a": 1}\n')
    crlf.write_bytes(b'{"a": 1}\r\n')
    assert X.sha256_text_file(lf) == X.sha256_text_file(crlf)
    assert X.sha256_text_file(lf) != X.sha256_text_file(_write_bytes(tmp_path / "x.json", b'{"a": 2}\n'))


def _write_bytes(path, payload):
    path.write_bytes(payload)
    return path


def test_plan_binds_to_the_manifest_and_corpus_shas(tmp_path, monkeypatch, corpus):
    """End-to-end contract between extract_cw_triggers and d_sham_plan: the
    plan must record exactly the digests the driver recomputes."""
    import json

    import extract_cw_triggers as X

    text, _, _ = corpus
    words = text.split()

    call = '<tool_call>\n{"name": "%s", "arguments": {}}\n</tool_call>'
    rows_full = [
        {"qid": f"s0:{i}", "session_id": "s0", "skipped": False,
         "prediction": call % "get_weather", "target": call % "get_weather",
         "target_tool_name": "get_weather", "has_tool_call": True,
         "tool_name_match": True, "doc_chunks": 5}
        for i in range(3)
    ]
    rows_comp = [dict(row, prediction=call % "wrong_tool", tool_name_match=False) for row in rows_full]

    def _write_jsonl(path, rows):
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        return str(path)

    manifest_path = tmp_path / "manifest.json"
    argv = [
        "--full_rows", _write_jsonl(tmp_path / "full.jsonl", rows_full),
        "--compressed_rows", _write_jsonl(tmp_path / "comp.jsonl", rows_comp),
        "--batch", "batch-TF",
        "--out_bundles", str(tmp_path / "bundles.jsonl"),
        "--out_manifest", str(manifest_path),
        "--out_doc_table", str(tmp_path / "d_doc_ids.json"),
        "--ckpt_path", "./ckpt", "--model_sha", "a", "--eval_code_sha", "b",
        "--chunk_policy", "pilot_v1",
    ]
    manifest = X.run(X.parse_args(argv))
    assert manifest["n_cw"] == 3

    doc_table = {
        "per_qid": {
            qid: {"session_id": "s0", "n_docs": 5, "doc_lens": [40, 55, 63, 71, 48]}
            for qid in manifest["cw_qids"]
        }
    }
    doc_table_path = tmp_path / "d_doc_ids.json"
    doc_table_path.write_text(json.dumps(doc_table), encoding="utf-8")

    class _Tok:
        def __call__(self, value, add_special_tokens=False):
            return {"input_ids": list(range(len(value.split())))}

        def decode(self, ids, skip_special_tokens=False):
            return " ".join(words[i] for i in ids)

    monkeypatch.setattr(P, "_load_tokenizer", lambda path: _Tok())
    plan_path = tmp_path / "d_sham_plan.json"
    exit_code = P.main([
        "--doc_table", str(doc_table_path),
        "--manifest", str(manifest_path),
        "--corpus", str(CORPUS_PATH),
        "--tokenizer", "unused",
        "--out", str(plan_path),
    ])
    assert exit_code == 0
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    assert plan["qid_source_sha256"] == P._sha256_file(manifest_path)
    assert plan["corpus_sha256"] == P._sha256_file(CORPUS_PATH)
    assert plan["doc_table_sha256"] == P._sha256_file(doc_table_path)
    assert sorted(plan["per_qid"]) == sorted(manifest["cw_qids"])
    assert plan["budget"]["gate_passed"] and plan["neutrality"]["gate_passed"]
    # The driver reads exactly these three keys off each entry.
    for entry in plan["per_qid"].values():
        assert {"k_star", "span_len", "sham_token_ids"} <= set(entry)
        assert len(entry["sham_token_ids"]) == entry["span_len"] == 63
