"""Artifact integrity and matched-corpus admission contracts."""
import hashlib
import importlib.util
import json
import subprocess
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


def test_delivery_rejects_post_packing_action_collapse():
    spec = importlib.util.spec_from_file_location("verify_next_delivery", Path(__file__).resolve().parents[2] / "agent" / "verify_next_delivery.py")
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    verifier.check_history_type_balance({"tool_call": 100, "non_tool_response": 104, "terminal_stop": 102}, 2)
    with pytest.raises(ValueError, match="unbalanced"):
        verifier.check_history_type_balance({"tool_call": 2, "non_tool_response": 100, "terminal_stop": 100}, 2)


def _digest(value):
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _delivery_record(decision_id, ratio, decision_type, *, weighted=False):
    suffix = int(decision_id.rsplit("-", 1)[-1])
    event_id = f"event-{suffix}"
    memory = PackedMemory(
        MemoryView((event_id,), (f"current-{suffix}",), ()),
        (20, 21),
        (30, 31 + suffix),
        (1,),
        (
            EncoderChunk(
                event_id,
                0,
                (0,),
                0,
                4,
                (4 + suffix, 5 + suffix, 6 + suffix, 7 + suffix),
            ),
        ),
    )
    decision = PreparedDecision(
        memory,
        (40 + suffix, 2),
        ratio,
        1.0,
        decision_id,
        (1.0, 2.0) if weighted else None,
    )
    return make_record(
        decision,
        session_key=f"session-{suffix}",
        source="fixture",
        metadata={"decision_type": decision_type},
    )


def _write_delivery_fixture(root):
    decision_types = ("tool_call", "non_tool_response", "terminal_stop")
    for variant in ("H0", "H1", "H2", "H3"):
        writer = CorpusWriter(root / variant, variant)
        decision_ids = [f"history-{index}" for index in range(3)]
        for index, decision_type in enumerate(decision_types):
            for ratio in (8, 12):
                writer.write(
                    _delivery_record(
                        decision_ids[index],
                        ratio,
                        decision_type,
                        weighted=variant == "H3",
                    )
                )
        preparation = {"render_profile": "fixture", "batch_sessions": 1}
        source_files = {}
        audit = {}
        if variant == "H0":
            preparation["config"] = {"h0_decision_ids_count": len(decision_ids)}
            audit = {
                "accepted_distinct_decisions": len(decision_ids),
                "legacy_distinct_decisions_not_emitted": 0,
            }
            source_files["legacy_selected_ids_sha256"] = _digest(
                sorted(decision_ids)
            )
        writer.finish(
            tokenizer={"sha256": "fixture"},
            preparation=preparation,
            source_files=source_files,
            audit=audit,
        )
    for variant in ("T0", "T1"):
        writer = CorpusWriter(root / variant, variant)
        for ratio in (8, 12):
            writer.write(_delivery_record("tool-0", ratio, "tool_call"))
        writer.finish(
            tokenizer={"sha256": "fixture"},
            preparation={"render_profile": "fixture", "batch_sessions": 1},
            source_files={},
            audit={},
        )


def test_parallel_cli_matches_serial_six_variant_verification(tmp_path):
    root = tmp_path / "delivery"
    _write_delivery_fixture(root)
    verifier_path = (
        Path(__file__).resolve().parents[2] / "agent" / "verify_next_delivery.py"
    )
    outputs = {}
    for workers in (1, 3):
        output_path = tmp_path / f"verified-{workers}.json"
        subprocess.run(
            [
                sys.executable,
                str(verifier_path),
                "--data-root",
                str(root),
                "--workers",
                str(workers),
                "--output",
                str(output_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        outputs[workers] = json.loads(output_path.read_text(encoding="utf-8"))
    assert outputs[1] == outputs[3]
    assert outputs[3]["schema"] == "next-compression-delivery-verification-v1"
    assert outputs[3]["status"] == "verified"
