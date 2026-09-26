import argparse
import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).with_name("prepare_exp1_tools.py")
SPEC = importlib.util.spec_from_file_location("prepare_exp1_tools_under_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
prepare = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = prepare
SPEC.loader.exec_module(prepare)

from history_memory.dataset import iter_decisions
from history_memory.packing import pack_target
from next_compression.exp1_tools import EXP1_PURPOSE, EXP1_SCHEMA, LayoutPlan, build_layout_records
from next_compression.test_exp1_tools import _three_tools
from next_compression.test_tools import ByteChatTokenizer, _row
from next_compression.tools import ToolPreparationConfig


def test_quota_and_k_parsers_reject_bad_input():
    assert prepare._positive_quota("tool_call=3,non_tool_response=1") == {
        "tool_call": 3,
        "non_tool_response": 1,
    }
    with pytest.raises(argparse.ArgumentTypeError):
        prepare._positive_quota("unknown=1")
    with pytest.raises(argparse.ArgumentTypeError):
        prepare._positive_quota("tool_call=0")
    assert prepare._int_list("1,3,5") == (1, 3, 5)
    with pytest.raises(argparse.ArgumentTypeError):
        prepare._int_list("1,1")


def test_arguments_require_hybrid_for_retrieval():
    base = [
        "--tool-manifest", "m.json", "--training-dir", "t", "--tokenizer", "tok",
        "--output-dir", "out", "--quotas", "tool_call=2",
    ]
    args = prepare.arguments(base)
    assert args.k == (1, 3, 5) and args.layouts == list(prepare.GIST_LAYOUTS)
    with pytest.raises(SystemExit):
        prepare.arguments(base + ["--layouts", "retrieval", "uniform"])


def _decision_records(tokenizer):
    row = _row()
    row["tools"] = _three_tools()
    decision = list(iter_decisions(row))[-1]
    target_ids = pack_target(tokenizer, decision.target)
    return build_layout_records(
        decision,
        tokenizer,
        config=ToolPreparationConfig(),
        plan=LayoutPlan(k_values=(1,)),
        session_key="fixture:session-1",
        source="fixture",
        target_ids=target_ids,
        base_metadata={"decision_type": "tool_call", "gold_tool_calls": []},
    )


def test_writer_roundtrip_and_verify_existing(tmp_path):
    tokenizer = ByteChatTokenizer()
    records = _decision_records(tokenizer)
    writer = prepare.Exp1Writer(tmp_path / "exp1")
    writer.write_decision(records)
    manifest = writer.finish({"schema": EXP1_SCHEMA, "purpose": EXP1_PURPOSE})
    assert manifest["records"]["count"] == len(records)
    assert manifest["cell_counts"]["hybrid.k1.ratio8"] == 1
    assert manifest["decision_ids"] == [records[0]["decision_id"]]
    lines = (tmp_path / "exp1" / "records.jsonl").read_text(encoding="utf-8").splitlines()
    assert {json.loads(line)["split"] for line in lines} == {EXP1_PURPOSE}
    verified = prepare.verify_existing(tmp_path / "exp1")
    assert verified["status"] == "verified" and verified["records"] == len(records)
    (tmp_path / "exp1" / "records.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        prepare.verify_existing(tmp_path / "exp1")
    with pytest.raises(FileExistsError):
        prepare.Exp1Writer(tmp_path / "exp1")
