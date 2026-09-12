"""Artifact integrity and matched-corpus admission contracts."""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from history_memory.packing import EncoderChunk, MemoryView, PackedMemory
from next_compression.common import CorpusWriter, PreparedDecision, SerializedCorpus, make_record

spec = importlib.util.spec_from_file_location("prepare_next_history", Path(__file__).resolve().parents[2] / "agent" / "prepare_next_history.py")
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)


def record(ratio):
    memory = PackedMemory(MemoryView(("event",), (), ()), (), (5, 6), (),
                          (EncoderChunk("event", 0, (0,), 0, 3, (1, 2, 3)),))
    return make_record(PreparedDecision(memory, (7, 8), ratio, decision_id="decision"),
                       session_key="session", source="test")


def test_serialization_roundtrip_and_tamper_rejected(tmp_path):
    writer = CorpusWriter(tmp_path / "H0", "H0")
    for ratio in (8, 12):
        writer.write(record(ratio))
    writer.finish(tokenizer={"sha256": "fixture"}, preparation={"render_profile": "fixture"}, source_files={}, audit={})
    corpus = SerializedCorpus(tmp_path / "H0", expected_variant="H0")
    assert len(corpus) == 2
    assert corpus[0].memory.chunks[0].token_ids == (1, 2, 3)
    assert corpus[1].ratio == 12
    records = tmp_path / "H0" / "records.jsonl"
    records.write_text(records.read_text().replace('"ratio":8', '"ratio":4'))
    with pytest.raises(ValueError, match="hash/size"):
        SerializedCorpus(tmp_path / "H0")


def test_atomic_pairing_rejects_missing_ratio_and_changed_target():
    with pytest.raises(ValueError, match="Incomplete"):
        list(entry.atomic_records(iter([("H0", record(8))]), ("H0",)))
    altered = record(12)
    altered["target_ids"] = [42]
    with pytest.raises(ValueError, match="same complete continuation"):
        list(entry.atomic_records(iter([("H0", record(8)), ("H0", altered)]), ("H0",)))
    assert len(list(entry.atomic_records(iter([("H0", record(8)), ("H0", record(12))]), ("H0",)))) == 1


def test_source_interleave_retains_all_rows():
    assert list(entry.round_robin(([1, 2, 3], [4], [5, 6]))) == [1, 4, 5, 2, 6, 3]


def test_incomplete_writer_does_not_publish_manifest(tmp_path):
    writer = CorpusWriter(tmp_path / "H0", "H0")
    writer.write(record(8))
    with pytest.raises(ValueError, match="incomplete ratio"):
        writer.finish(tokenizer={}, preparation={}, source_files={}, audit={})
    assert not (tmp_path / "H0" / "manifest.json").exists()
