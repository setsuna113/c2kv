from __future__ import annotations

import hashlib
import json
from pathlib import Path

from experiments.history_system.reporting.collect_delivery import (
    SAMPLE_LABEL,
    collect_delivery,
)
from experiments.history_system.reporting.render_delivery import render


REPO = Path(__file__).resolve().parents[3]


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _artifact(path: Path) -> dict:
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bytes": path.stat().st_size,
    }


def _official_row(result_root: Path, ordinal: int, task_id: str, score: float) -> dict:
    task_key = f"{ordinal:04d}_tau2_{task_id}"
    result_path = result_root / "task_shards" / task_key / "official" / "result.json"
    steps_path = result_root / "task_shards" / task_key / "server" / "steps.jsonl"
    _write(
        result_path,
        {
            "schema": "history-system-official-result-v1",
            "status": "completed",
            "benchmark": "tau2",
            "task_id": task_id,
            "scored": True,
            "official_score": score,
            "official_artifacts": [],
            "benchmark_summary": {"n": 1, "semantic_score": score},
            "source_binding": [],
            "elapsed_seconds": 1.0,
            "max_wall_seconds": 60,
            "error": None,
        },
    )
    steps_path.parent.mkdir(parents=True, exist_ok=True)
    steps_path.write_text('{"decision_key":"turn-0/step-0"}\n', encoding="utf-8")
    return {
        "ordinal": ordinal,
        "benchmark": "tau2",
        "task_id": task_id,
        "task_key": task_key,
        "max_new_tokens": 4096,
        "in_fixed_denominator": True,
        "outcome": "official_scored",
        "reason": None,
        "official_score": score,
        "scored": True,
        "official_result": {
            "status": "official_scored",
            "official_score": score,
            "scored": True,
            "artifact": _artifact(result_path),
        },
        "server_evidence": {
            "steps": _artifact(steps_path),
            "step_count": 1,
            "generation_count": 1 + ordinal,
            "compression_receipts": [
                {
                    "system_active_history_reduction": 2.0 + 2.0 * ordinal,
                    "errors": [],
                }
            ],
            "compression_receipt_errors": 0,
        },
    }


def _stage(result_root: Path, *, status: str, second_outcome: str = "official") -> None:
    first = _official_row(result_root, 0, "retail-6", 1.0)
    if second_outcome == "official":
        second = _official_row(result_root, 1, "retail-7", 0.0)
    else:
        second = {
            "ordinal": 1,
            "benchmark": "tau2",
            "task_id": "retail-7",
            "task_key": "0001_tau2_retail-7",
            "max_new_tokens": 4096,
            "in_fixed_denominator": True,
            "outcome": (
                "infra_failed_in_denominator"
                if second_outcome == "infra"
                else "not_started_in_denominator"
            ),
            "reason": "synthetic" if second_outcome == "infra" else None,
            "official_score": None,
            "scored": False,
        }
    _write(
        result_root / "stage.json",
        {
            "schema": "a-history-multibench-stage-v1",
            "status": status,
            "suite_id": "cpu-suite",
            "candidate_id": "prefill_gate",
            "checkpoint": {
                "status": "selected",
                "selected_arm": "C",
                "selected_step": 1000,
                "ratio": 8,
                "path": "/frozen/checkpoint-1000",
                "config_sha256": "a" * 64,
            },
            "fixed_denominator": 2,
            "task_manifest_sha256": "b" * 64,
            "task_order": [first["task_key"], second["task_key"]],
            "automatic_reruns": 0,
            "task_outcomes": [first, second],
            "official_scored_tasks": 1 + int(second_outcome == "official"),
            "infra_failed_tasks": int(second_outcome == "infra"),
            "completed_task_cells": 1 + int(second_outcome != "pending"),
            "wall_seconds": 2.0,
            "wall_seconds_final": status != "running_fixed_manifest",
        },
    )


def _collect(*, native=(), suite=(), peers=None):
    return collect_delivery(
        native_outputs=list(native),
        suite_outputs=list(suite),
        algorithm_name="CPU collector test",
        execution_note="Only hash-verified frozen outputs are normalized.",
        peer_sources=peers,
    )


def _render(index: dict) -> str:
    legacy = json.loads(
        (REPO / "experiments/history_system/configs/delivery_20260914.baseline_sources.json")
        .read_text(encoding="utf-8")
    )
    checkpoint = json.loads(
        (REPO / "experiments/history_system/configs/checkpoint.selected.json")
        .read_text(encoding="utf-8")
    )
    return render(index, legacy, checkpoint)


def test_real_completed_c0_is_hash_bound_and_renderer_compatible() -> None:
    root = REPO / "outputs/history_system_search/r001/c0_bridge_memo_b0"
    result = _collect(native=[root])
    cell = result["cells"][0]
    assert cell["status"] == "completed"
    assert cell["official_score"] == 0.45
    assert (cell["n_scored"], cell["n_planned"]) == (20, 20)
    assert cell["sample_label"] == SAMPLE_LABEL
    assert cell["compression"]["full_bytes_over_resident_bytes"] == 1.1143981887292185
    assert len(cell["compression"]["sources"]) == 21
    assert all(Path(source["path"]).is_file() for source in cell["sources"])

    comparison = result["comparison_cells"]
    assert len(comparison) == 1
    assert comparison[0]["cohort"] == "r001 long10 (variant=long)"
    assert comparison[0]["official_score"] == 0.4
    assert comparison[0]["checkpoint"]["selected_step"] == 500

    rendered = _render(result)
    assert "45.00%" in rendered
    assert "None" not in rendered


def test_completed_suite_uses_full_denominator_and_trace_compression(tmp_path: Path) -> None:
    result_root = tmp_path / "returned"
    _stage(result_root, status="completed_fixed_manifest")
    result = _collect(suite=[tmp_path])
    cell = result["cells"][0]
    assert cell["status"] == "completed"
    assert cell["official_score"] == 0.5
    assert (cell["n_scored"], cell["n_planned"]) == (2, 2)
    assert cell["checkpoint"]["ratio"] == 8
    assert cell["compression"]["full_bytes_over_resident_bytes"] == 3.0
    assert cell["compression"]["extra_generations"] == 1
    assert {source["kind"] for source in cell["sources"]} == {
        "official_task_result", "suite_stage", "suite_steps"
    }


def test_running_suite_keeps_partial_count_without_partial_score(tmp_path: Path) -> None:
    result_root = tmp_path / "returned"
    _stage(result_root, status="running_fixed_manifest", second_outcome="pending")
    cell = _collect(suite=[tmp_path])["cells"][0]
    assert cell["status"] == "pending"
    assert cell["official_score"] is None
    assert (cell["n_scored"], cell["n_planned"]) == (1, 2)


def test_infra_failure_is_unknown_not_zero(tmp_path: Path) -> None:
    result_root = tmp_path / "returned"
    _stage(
        result_root,
        status="completed_fixed_manifest_with_infra_failures",
        second_outcome="infra",
    )
    cell = _collect(suite=[tmp_path])["cells"][0]
    assert cell["status"] == "infra_failed"
    assert cell["official_score"] is None
    assert (cell["n_scored"], cell["n_planned"]) == (1, 2)
    assert cell["n_infra_failed"] == cell["n_unscored"] == 1
    assert cell["all_pieces_terminal"] is True
    report = _render(_collect(suite=[tmp_path]))
    assert "无完整总分（已结束，1题未评分）" in report
    assert "（进行中）" not in report


def test_missing_result_directories_emit_pending_cells(tmp_path: Path) -> None:
    native = tmp_path / "native_missing"
    suite = tmp_path / "suite_missing"
    result = _collect(native=[native], suite=[suite])
    assert len(result["cells"]) == 5
    assert all(cell["status"] == "pending" for cell in result["cells"])
    assert all(cell["official_score"] is None for cell in result["cells"])
    assert all(cell["n_scored"] == cell["n_planned"] == 0 for cell in result["cells"])


def test_frozen_native_without_returned_stage_preserves_planned_denominator(tmp_path: Path) -> None:
    root = tmp_path / "d3"
    _write(
        root / "freeze.json",
        {"status": "frozen_not_launched", "candidate_id": "d3_candidate", "tasks": 2},
    )
    _write(
        root / "submitted" / "tasks.json",
        {
            "schema": "a-history-system-task-manifest-v1",
            "manifest_id": "r001_mixed20_subset",
            "stage": "development_search",
            "task_ids": ["multi_turn_base_0", "multi_turn_long_context_0"],
            "fixed_denominator": 2,
        },
    )
    _write(
        root / "submitted" / "design.json",
        {
            "checkpoint_selection": {
                "status": "selected",
                "selected_arm": "C",
                "selected_step": 1000,
                "ratio": 8,
            }
        },
    )
    cell = _collect(native=[root])["cells"][0]
    assert cell["status"] == "pending"
    assert cell["official_score"] is None
    assert (cell["n_scored"], cell["n_planned"]) == (0, 2)
    assert cell["checkpoint"]["selected_step"] == 1000
    assert {source["kind"] for source in cell["sources"]} == {
        "native_design", "native_freeze", "native_task_manifest"
    }


def test_audited_peer_sources_emit_only_completed_long10() -> None:
    peers = REPO / "experiments/history_system/configs/peer_sources.json"
    cells = _collect(peers=peers)["comparison_cells"]
    assert [(cell["method"], cell["official_score"]) for cell in cells] == [
        ("Full (B500)", 0.3),
        ("HiAgent (B500)", 0.6),
        ("Raw (B500)", 0.5),
        ("Text (B500)", 0.3),
    ]
    assert all(cell["cohort"] == "r001 long10 (variant=long)" for cell in cells)
    assert all(cell["n_scored"] == cell["n_planned"] == 10 for cell in cells)
    assert all(cell["checkpoint"]["selected_step"] == 500 for cell in cells)
    assert all(cell["checkpoint"]["ratio"] is None for cell in cells)
    rendered = _render(_collect(peers=peers))
    assert "BFCL same long10 legacy comparison" in rendered
    assert "Raw (B500) | Arm B checkpoint-500 | r001 long10 (variant=long) | 50.00% | 10/10 | completed" in rendered


def test_renderer_keeps_pending_long10_unknown() -> None:
    index = _collect()
    index["comparison_cells"] = [
        {
            "benchmark": "bfcl",
            "method": "D3",
            "cohort": "r001 long10 (variant=long)",
            "official_score": None,
            "n_scored": 0,
            "n_planned": 10,
            "status": "pending",
            "checkpoint": {"selected_arm": "C", "selected_step": 1000, "ratio": 8},
            "sources": [],
        }
    ]
    rendered = _render(index)
    assert "D3 | Arm C checkpoint-1000, ratio=8" in rendered
    assert "未返回 | 0/10 | pending" in rendered
    assert "0.00% | 0/10" not in rendered


def test_ace_recovery_binds_original_generation_and_official_failure_rows(tmp_path):
    import pytest
    from experiments.history_system.reporting.collect_delivery import _ace_recovered_score
    root = tmp_path / "suite"
    result_root = root / "returned"
    task_key = "acebench_0"
    task = result_root / "task_shards" / task_key
    proof_root = root / "returned_scoring" / task_key
    files = {task / "official/result.json": {"status": "infra_failed"},
             task / "server/steps.jsonl": {"generation": "original"},
             task / "data_agent_result.json": {"id": "agent_multi_step_0"},
             task / "data_agent_score.json": {"accuracy": 1.0}}
    bindings = []
    for path, value in files.items():
        _write(path, value)
        remote = "/remote/suite/results/" + path.relative_to(result_root).as_posix()
        bindings.append({"path": remote, "sha256": _artifact(path)["sha256"]})
    _write(proof_root / "result.json", {
        "task_key": task_key, "task_id": "agent_multi_step_0",
        "status": "official_scored_from_existing_generations", "model_calls": 0,
        "generation_trace_unchanged": True, "official_score": 1.0, "artifacts": bindings})
    row = {"benchmark": "acebench", "task_key": task_key, "task_id": "agent_multi_step_0"}
    # Official files are JSONL rather than pretty-printed JSON.
    for path, value in files.items():
        path.write_text(json.dumps(value) + "\n", encoding="utf-8")
    recovered = json.loads((proof_root / "result.json").read_text())
    for binding, path in zip(recovered["artifacts"], files):
        binding["sha256"] = _artifact(path)["sha256"]
    _write(proof_root / "result.json", recovered)
    score, sources = _ace_recovered_score(root, result_root, row)
    assert score == 1.0 and len(sources) == 5
    (task / "server/steps.jsonl").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        _ace_recovered_score(root, result_root, row)


def test_continuation_partition_can_coexist_with_other_lanes():
    import pytest
    from experiments.history_system.reporting.collect_delivery import _merge_suite_pieces
    def piece(ids,parent=None):
        return dict(candidate_id="d3",benchmark="acon_appworld",task_ids=ids,
            continuation_parent_task_ids=parent,status="completed",source_status="completed_fixed_manifest",
            terminal=True,n_planned=len(ids),n_scored=len(ids),scores=[1.0]*len(ids),
            compression_tasks=[],sources=[],checkpoint=None)
    a=piece(["a"],["a","b"]);b=piece(["b"]);other=piece(["c"])
    merged=_merge_suite_pieces([a,b,other])[0]
    assert merged["n_planned"]==merged["n_scored"]==3
    with pytest.raises(ValueError,match="parent denominator"):_merge_suite_pieces([a,other])
    with pytest.raises(ValueError,match="duplicate"):_merge_suite_pieces([a,b,piece(["a"])])
