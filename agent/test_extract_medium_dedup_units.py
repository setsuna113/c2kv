# -*- coding: utf-8 -*-
"""Tests for agent/extract_medium_dedup_units.py (G-medium dedup extraction).

The acceptance property (P1-5): the QA extraction units' ``_id`` values ARE
the qids the joint loader produces for the same data, so a dedup removal
entry matches the planner pool by exact string equality.  Also covers the
parts-shaped v2 eval messages (the first one-off extractor's zero-row bug),
the longmagpie shard-local ids (skipped rows consume their row number), and
byte-level idempotence.

Run from the repo root:
  pytest agent/test_extract_medium_dedup_units.py -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _sub in ("python", "agent"):
    _path = str(_REPO_ROOT / _sub)
    if _path not in sys.path:
        sys.path.insert(0, _path)

import extract_medium_dedup_units as emdu  # noqa: E402
from train.train_data_joint_multisource import QADocsJointSource  # noqa: E402


def _write_jsonl(path: Path, rows) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


@pytest.fixture()
def corpus(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    # traces-v2: one eval session (parts-shaped messages) + one train session.
    v2_dir = tmp_path / "v2" / "data"
    v2_dir.mkdir(parents=True)
    span = {
        "span_id": "span-1",
        "start_time": "2026-01-01T00:00:01",
        "status": "ok",
        "attributes": {
            "gen_ai.input.messages": json.dumps([
                {"role": "user", "parts": [{"type": "text", "content": "eval question text"}]}
            ]),
            "gen_ai.output.messages": json.dumps([
                {"role": "assistant", "parts": [{"type": "text", "content": "eval answer text"}]}
            ]),
        },
    }
    pq.write_table(
        pa.table({
            "session_id": ["eval-sess", "train-sess"],
            "spans": [json.dumps([span]), json.dumps([span])],
        }),
        v2_dir / "shard.parquet",
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"taskproxy_disjoint": {"train_session_ids": ["train-sess"], "eval_session_ids": ["eval-sess"]}}),
        encoding="utf-8",
    )

    # hotpotqa jsonl (2 rows).
    hotpotqa = tmp_path / "hotpotqa.jsonl"
    _write_jsonl(hotpotqa, [
        {
            "_id": "hp-0",
            "question": "q0?",
            "answer": "a0",
            "documents": ["Document 1 (title: T0) alpha text", "Document 2 (title: T1) beta text"],
        },
        {"_id": "hp-1", "question": "q1?", "answer": "a1", "documents": ["Document 1 (title: T2) gamma"]},
    ])

    # 2wiki parquet (real format: context is a JSON-encoded string column).
    wiki2_dir = tmp_path / "2wiki"
    wiki2_dir.mkdir()
    pq.write_table(
        pa.table({
            "_id": ["w2-0"],
            "question": ["q?"],
            "answer": ["a"],
            "context": [json.dumps([["TitleA", ["one two"]], ["TitleB", ["three four"]]])],
            "supporting_facts": ['[["TitleA", 0]]'],
        }),
        wiki2_dir / "train.parquet",
    )

    # longmagpie: two shards; shard-a has a skipped row (no "?") in the middle.
    lm_dir = tmp_path / "longmagpie" / "data"
    lm_dir.mkdir(parents=True)

    def lm_row(text):
        return [{"role": "user", "content": text}, {"role": "assistant", "content": "ans"}]

    pq.write_table(
        pa.table({"messages": [
            lm_row("shard a row zero body.What is it?"),
            lm_row("all statements, no question mark"),
            lm_row("shard a row two body.Why is it?"),
        ]}),
        lm_dir / "shard-a.parquet",
    )
    pq.write_table(pa.table({"messages": [lm_row("shard b row zero body.Who is it?")]}), lm_dir / "shard-b.parquet")

    # openswe: one resolved + one unresolved trajectory.
    openswe_dir = tmp_path / "openswe" / "data" / "cfg"
    openswe_dir.mkdir(parents=True)
    pq.write_table(
        pa.table({
            "trajectory_id": ["traj-0", "traj-1"],
            "instance_id": ["inst-0", "inst-1"],
            "resolved": [1, 0],
            "trajectory": [
                [
                    {"role": "user", "content": "fix the bug"},
                    {"role": "assistant", "content": "running", "tool_calls": [
                        {"type": "function", "function": {"name": "run", "arguments": "{\"cmd\": \"ls\"}"}}
                    ]},
                ],
                [{"role": "user", "content": "unresolved task"}],
            ],
        }),
        openswe_dir / "shard-0.parquet",
    )
    return {
        "v2_dir": str(tmp_path / "v2"),
        "manifest": str(manifest),
        "hotpotqa": str(hotpotqa),
        "wiki2": str(wiki2_dir),
        "longmagpie": str(tmp_path / "longmagpie"),
        "openswe": str(tmp_path / "openswe"),
    }


def _run(corpus, tmp_path, out_name="out"):
    out_dir = tmp_path / out_name
    summary = emdu.main([
        "--traces_v2_dir", corpus["v2_dir"],
        "--split_manifest_file", corpus["manifest"],
        "--openswe_dir", corpus["openswe"],
        "--qa_hotpotqa_path", corpus["hotpotqa"],
        "--qa_2wiki_path", corpus["wiki2"],
        "--qa_longmagpie_path", corpus["longmagpie"],
        "--out_dir", str(out_dir),
    ])
    return summary, out_dir


def test_extract_all_sections(corpus, tmp_path):
    summary, out_dir = _run(corpus, tmp_path)
    assert summary == {
        "v2eval_sessions": 1,
        "v2eval_msgs_raw": 2,
        "hotpotqa": 3,
        "2wiki": 2,
        "longmagpie": 4,  # every row with non-empty user text, incl. the loader-skipped one
        "openswe_resolved": 1,
    }
    # v2 eval: only the eval session; parts-shaped messages flattened via
    # dedup's _message_text (the fixed path).
    sessions = _read_jsonl(out_dir / "v2eval_sessions.jsonl")
    assert [row["session_id"] for row in sessions] == ["eval-sess"]
    raw = _read_jsonl(out_dir / "v2eval_msgs_raw.jsonl")
    assert {row["text"] for row in raw} == {"eval question text", "eval answer text"}
    assert all(row["_id"] == "eval-sess" for row in raw)
    # openswe: resolved only; assistant unit carries content + raw arguments.
    openswe = _read_jsonl(out_dir / "openswe_resolved_msgs.jsonl")
    assert [row["trajectory_id"] for row in openswe] == ["traj-0"]
    messages = json.loads(openswe[0]["messages"])
    assert messages[1]["content"] == 'running\n{"cmd": "ls"}'


def test_qa_unit_ids_equal_loader_qids(corpus, tmp_path):
    _, out_dir = _run(corpus, tmp_path)
    qa_units = _read_jsonl(out_dir / "qa_docs_raw.jsonl")
    unit_ids = [row["_id"] for row in qa_units]
    # Exact ids, in file order.
    assert [u for u in unit_ids if u.startswith("qa:hotpotqa:")] == [
        "qa:hotpotqa:hp-0", "qa:hotpotqa:hp-0", "qa:hotpotqa:hp-1",
    ]
    assert [u for u in unit_ids if u.startswith("qa:2wiki:")] == ["qa:2wiki:w2-0", "qa:2wiki:w2-0"]
    assert [u for u in unit_ids if u.startswith("qa:longmagpie:")] == [
        "qa:longmagpie:shard-a:0",
        "qa:longmagpie:shard-a:1",  # skipped by the loader, present in raw units
        "qa:longmagpie:shard-a:2",
        "qa:longmagpie:shard-b:0",
    ]
    # P1-5 acceptance: every qid the LOADER produces appears verbatim among
    # the extraction-unit ids (longmagpie skipped rows are the only extras).
    loader_qids = {
        example.qid
        for example in QADocsJointSource(
            hotpotqa_path=corpus["hotpotqa"],
            wiki2_path=corpus["wiki2"],
            longmagpie_path=corpus["longmagpie"],
        )
    }
    assert loader_qids <= set(unit_ids)
    assert set(unit_ids) - loader_qids == {"qa:longmagpie:shard-a:1"}
    # And the 2wiki unit text is byte-identical to the training document text.
    example = next(
        iter(
            QADocsJointSource(wiki2_path=corpus["wiki2"])
        )
    )
    wiki2_texts = [row["text"] for row in qa_units if row["_id"].startswith("qa:2wiki:")]
    assert wiki2_texts == example.history_documents


def test_extract_idempotent(corpus, tmp_path):
    _, out_a = _run(corpus, tmp_path, out_name="a")
    _, out_b = _run(corpus, tmp_path, out_name="b")
    for name in (
        "v2eval_sessions.jsonl",
        "v2eval_msgs_raw.jsonl",
        "qa_docs_raw.jsonl",
        "openswe_resolved_msgs.jsonl",
    ):
        assert (out_a / name).read_bytes() == (out_b / name).read_bytes()


def test_extractor_requires_inputs(tmp_path):
    with pytest.raises(ValueError, match="nothing to extract"):
        emdu.main(["--out_dir", str(tmp_path / "out")])
    with pytest.raises(ValueError, match="split_manifest_file"):
        emdu.main(["--traces_v2_dir", str(tmp_path), "--out_dir", str(tmp_path / "out")])
