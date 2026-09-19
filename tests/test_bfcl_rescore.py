import json

import pytest

from generality.bfcl_rescore import prepare
from generality import rescore


def write_attempt(cell, name, rows):
    path = cell / "batches" / name / "bfcl_worker" / "bfcl" / "result" / "model" / "multi_turn" / "BFCL_v4_multi_turn_base_result.json"
    path.parent.mkdir(parents=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_prepare_deduplicates_and_preserves_wrong_but_valid_result(tmp_path):
    cell = tmp_path / "cell"
    cell.mkdir()
    ids = ["multi_turn_base_1", "multi_turn_base_2"]
    (cell / "cell.json").write_text(json.dumps({"task_ids": ids}))
    write_attempt(cell, "first", [{"id": ids[0], "result": []}, {"id": ids[1], "traceback": "engine crashed"}])
    write_attempt(cell, "retry", [{"id": ids[1], "result": [["wrong()"]]}])
    out = tmp_path / "score"
    receipt = prepare(cell, out)
    rows = [json.loads(line) for line in (out / "result/c2kv-dedup/multi_turn/BFCL_v4_multi_turn_base_result.json").read_text().splitlines()]
    assert [row["id"] for row in rows] == ids
    assert rows[0]["result"] == []
    assert receipt["n_expected"] == 2
    assert receipt["duplicate_rows"] == 1
    assert len(receipt["result_sha256"]) == 64
    with pytest.raises(FileExistsError):
        prepare(cell, out)


def test_prepare_preserves_terminal_traceback_for_official_scorer(tmp_path):
    cell = tmp_path / "cell"
    cell.mkdir()
    task_id = "multi_turn_base_1"
    (cell / "cell.json").write_text(json.dumps({"task_ids": [task_id]}))
    failure = {
        "id": task_id,
        "result": "",
        "traceback": "The input (138237 tokens) is longer than the model's context length (131072 tokens).",
    }
    write_attempt(cell, "first", [failure])

    out = tmp_path / "score"
    receipt = prepare(cell, out)
    result_path = out / "result/c2kv-dedup/multi_turn/BFCL_v4_multi_turn_base_result.json"
    assert json.loads(result_path.read_text(encoding="utf-8")) == failure
    assert receipt["terminal_failures"] == {task_id: "context_overflow"}
    assert json.loads((out / "dedup_manifest.json").read_text())["terminal_failures"] == receipt["terminal_failures"]


def test_incomplete_cell_cannot_publish_score_input(tmp_path):
    cell = tmp_path / "cell"
    cell.mkdir()
    (cell / "cell.json").write_text(json.dumps({"task_ids": ["multi_turn_base_1", "multi_turn_base_2"]}))
    write_attempt(cell, "first", [{"id": "multi_turn_base_1", "result": []}])
    out = tmp_path / "score"
    with pytest.raises(ValueError, match="requires 1 valid results"):
        prepare(cell, out)
    assert not out.exists()


def test_old_fc_handler_decode_error_cannot_publish_score_input(tmp_path):
    cell = tmp_path / "cell"
    cell.mkdir()
    task_id = "multi_turn_base_1"
    (cell / "cell.json").write_text(json.dumps({"task_ids": [task_id]}))
    write_attempt(cell, "old", [{
        "id": task_id, "result": [["<tool_call>...</tool_call>"]],
        "inference_log": [{"step_0": [{"role": "handler_log",
                                      "error": "'str' object has no attribute 'items'"}]}],
    }])
    out = tmp_path / "score"

    with pytest.raises(ValueError, match="requires 1 valid results"):
        prepare(cell, out)
    assert not out.exists()


def test_legacy_rescore_entrypoint_requires_explicit_cell_and_output():
    with pytest.raises(SystemExit) as error:
        rescore.main([])
    assert error.value.code == 2
