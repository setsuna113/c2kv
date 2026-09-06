"""Focused regression tests for the matrix2 result harvester."""
from __future__ import annotations

import json
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))

import sg_harvest


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_matrix2_harvest_includes_cacheblend_and_labels_bfcl(
        monkeypatch, tmp_path, capsys):
    matrix2 = tmp_path / "matrix2"
    monkeypatch.setattr(sg_harvest, "HOME", tmp_path)
    monkeypatch.setattr(sg_harvest, "MATRIX2", matrix2)

    for arm in ("cacheblend_r16", "cacheblend_r15_k"):
        _write_json(
            matrix2 / f"ts_{arm}_sha" / f"summary_{arm}.json",
            {"n": 1, "semantic_score": 0.5},
        )

    bfcl_root = matrix2 / "bfcl_cacheblend_r16_sha"
    _write_json(
        bfcl_root / "summary_cacheblend_r16.json",
        {"bfcl_project_root": str(bfcl_root)},
    )
    score = (bfcl_root / "score" /
             "c2kv-cacheblend-r16" / "multi_turn" / "score.json")
    _write_json(score, {"total_count": 2, "correct_count": 1, "accuracy": 0.5})
    stale = (tmp_path / "benchmarks" / "gorilla" /
             "berkeley-function-call-leaderboard" / "score" /
             "c2kv-cacheblend-r16" / "multi_turn" / "stale_score.json")
    _write_json(stale, {"total_count": 9, "correct_count": 0, "accuracy": 0.0})

    report = sg_harvest.matrix2_harvest()
    output = capsys.readouterr().out

    assert set(report["ts"]) >= {"cacheblend_r16", "cacheblend_r15_k"}
    assert report["bfcl"]["cacheblend_r16"]["total_count"] == 2
    assert (
        "| BFCL | cacheblend_r16 | "
        "CacheBlend (artifact: V-dev, r=0.16, doc chunks) | 2 | "
        "acc 0.5 (1/2) |  |"
    ) in output
