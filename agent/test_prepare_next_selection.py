import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).with_name("prepare_next_selection.py")
SPEC = importlib.util.spec_from_file_location("prepare_next_selection", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
selection = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = selection
SPEC.loader.exec_module(selection)

from history_memory.packing import MemoryView, PackedMemory
from next_compression.common import serialize_memory


def _record(variant, ratio, memory, *, target=(31, 32)):
    record = {
        "decision_id": "decision-a",
        "session_key": '["source","session-a"]',
        "source": "source",
        "split": selection.PURPOSE,
        "ratio": ratio,
        "weight": 1.0,
        "target_ids": list(target),
        "memory": serialize_memory(memory),
        "metadata": {
            "purpose": selection.PURPOSE,
            "variant": variant,
            "decision_type": "tool_call",
            "gold_tool_calls": [
                {
                    "id": "call-a",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": {"q": "x"}},
                }
            ],
            "evaluation_target_weighting": "uniform",
        },
    }
    if variant == "H3":
        record["target_weights"] = [1.0] * len(target)
    return record


def test_session_overlap_is_rejected_across_any_training_variant():
    sessions = {
        variant: {f'{variant}-only'} for variant in selection.VARIANTS
    }
    sessions["T1"].add('["source","session-a"]')

    with pytest.raises(ValueError, match=r"overlap trained corpora.*T1=1"):
        selection.assert_session_disjoint(
            ['["source","session-a"]'], sessions
        )


def test_history_format_keeps_shared_targets_and_paired_inputs():
    static = PackedMemory(MemoryView((), ()), (1,), (2,), (), ())
    a_view = PackedMemory(MemoryView((), ()), (1,), (7,), (), ())
    records = {variant: [] for variant in selection.HISTORY_VARIANTS}
    for ratio in (8, 12):
        records["H0"].append(_record("H0", ratio, static))
        records["H1"].append(_record("H1", ratio, static))
        records["H2"].append(_record("H2", ratio, a_view))
        records["H3"].append(_record("H3", ratio, a_view))

    selection.validate_selection_records(
        records,
        variants=selection.HISTORY_VARIANTS,
        expected_decisions=1,
    )

    broken = copy.deepcopy(records)
    broken["H3"][0]["target_weights"][0] = 3.0
    with pytest.raises(ValueError, match="Non-uniform evaluation weights"):
        selection.validate_selection_records(
            broken,
            variants=selection.HISTORY_VARIANTS,
            expected_decisions=1,
        )


def test_dev_writer_preserves_dev_split_and_top_level_purpose(tmp_path):
    memory = PackedMemory(MemoryView((), ()), (1,), (2,), (), ())
    writer = selection.DevCorpusWriter(tmp_path / "H1", "H1")
    writer.write(_record("H1", 8, memory))
    writer.write(_record("H1", 12, memory))
    manifest = writer.finish(
        tokenizer={"sha256": "a" * 64},
        training_manifest={"loss_profile": "decision-mean-complete-ce-v1"},
        preparation={"render_profile": "event-native-evidence-v1"},
        source_files={},
        provenance={
            "training_session_disjointness": {
                "status": "verified",
                "variants": list(selection.VARIANTS),
            }
        },
        audit={},
    )

    lines = (tmp_path / "H1" / "records.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    assert {json.loads(line)["split"] for line in lines} == {selection.PURPOSE}
    assert manifest["purpose"] == selection.PURPOSE
    assert manifest["split"] == selection.PURPOSE
    disjointness = manifest["provenance"]["training_session_disjointness"]
    assert disjointness["status"] == "verified"
    assert disjointness["variants"] == list(selection.VARIANTS)
