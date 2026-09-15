"""Selection parity tests for the agent trace history source."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("datasets")
pytest.importorskip("torch")

_PYTHON_DIR = Path(__file__).resolve().parents[1]
if str(_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(_PYTHON_DIR))

from train import train_data_multiturn as data


_SESSIONS = ["session-a", "session-b", "session-c", "session-d"]
_ROWS = [
    {"session_id": session, "benchmark": "fixture", "spans": list(range(6))}
    for session in _SESSIONS
]


class _StubSource(data.AgentLLMTracesCompressHistorySource):
    touched: list[str]

    def _split_session_ids(self, sessions):
        return {session["session_id"] for session in sessions}, set()

    def _session_examples(self, session_id, spans):
        self.touched.append(session_id)
        return [
            data.CompressHistoryExample(
                qid=f"{session_id}:{index}",
                history_messages=[{"role": "user", "content": "history"}],
                current_messages=[{"role": "user", "content": "current"}],
                answer="answer",
            )
            for index, _ in enumerate(spans)
        ]


def _source(monkeypatch, tmp_path, *, manifest, **kwargs):
    data_file = tmp_path / "records.parquet"
    monkeypatch.setattr(data, "_find_agent_parquet_files", lambda path: [data_file])
    monkeypatch.setattr(data, "_iter_agent_rows", lambda paths: iter(_ROWS))
    monkeypatch.setattr(data, "_sort_agent_spans", list)
    manifest_file = None
    if manifest:
        manifest_file = tmp_path / "split.json"
        manifest_file.write_text(json.dumps({
            "train_session_ids": _SESSIONS,
            "eval_session_ids": [],
        }), encoding="utf-8")
    _StubSource.touched = []
    source = _StubSource(
        str(tmp_path),
        split="train",
        split_seed=17,
        split_manifest_file=str(manifest_file) if manifest_file else None,
        **kwargs,
    )
    return source, list(_StubSource.touched)


@pytest.mark.parametrize("manifest", [False, True])
def test_selected_source_matches_full_then_filter_with_sampling_and_limit(
        monkeypatch, tmp_path, manifest):
    options = {"max_samples_per_session": 3, "max_records": 10}
    full, _ = _source(monkeypatch, tmp_path, manifest=manifest, **options)

    by_session, touched = _source(
        monkeypatch, tmp_path, manifest=manifest,
        selected_sessions=["session-d"], **options,
    )
    assert [example.qid for example in by_session] == [
        example.qid for example in full if example.qid.startswith("session-d:")
    ]
    # Sampling every earlier split session advances the RNG exactly as before.
    assert touched == _SESSIONS

    wanted = {full.records[4].qid, full.records[-1].qid}
    by_qid, touched = _source(
        monkeypatch, tmp_path, manifest=manifest,
        selected_qids=wanted, **options,
    )
    assert [example.qid for example in by_qid] == [
        example.qid for example in full if example.qid in wanted
    ]
    assert touched == _SESSIONS


def test_unselected_sessions_skip_parsing_when_no_rng_or_limit_dependency(
        monkeypatch, tmp_path):
    selected, touched = _source(
        monkeypatch, tmp_path, manifest=True,
        max_samples_per_session=0, max_records=None,
        selected_sessions=["session-c"],
    )
    assert touched == ["session-c"]
    assert {example.qid.rpartition(":")[0] for example in selected} == {"session-c"}
