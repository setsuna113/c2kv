"""End-to-end differential coverage for preparation performance changes."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = REPO_ROOT / "scripts" / "benchmark_history_prepare.py"
TOKENIZER_CACHE = (
    Path.home()
    / ".cache"
    / "huggingface"
    / "hub"
    / "models--Qwen--Qwen3-4B-Instruct-2507"
)
TRACES_ROOT = REPO_ROOT.parent / "investigation" / "data" / "agent-llm-traces"
BASELINE_REVISION = "ad277e785f6c09186ebabcf8d3bbd39b96d62f0f"


def _tokenizer_snapshot() -> Path | None:
    refs_main = TOKENIZER_CACHE / "refs" / "main"
    if refs_main.is_file():
        candidate = TOKENIZER_CACHE / "snapshots" / refs_main.read_text(encoding="utf-8").strip()
        if (candidate / "tokenizer.json").is_file():
            return candidate
    snapshots = TOKENIZER_CACHE / "snapshots"
    if snapshots.is_dir():
        matches = sorted(path for path in snapshots.iterdir() if (path / "tokenizer.json").is_file())
        if matches:
            return matches[-1]
    return None


def test_prepare_matches_submitted_baseline_end_to_end(tmp_path: Path) -> None:
    tokenizer = _tokenizer_snapshot()
    if tokenizer is None:
        pytest.skip("cached Qwen3-4B-Instruct-2507 tokenizer is unavailable")
    baseline = subprocess.run(
        ["git", "cat-file", "-e", f"{BASELINE_REVISION}^{{commit}}"],
        cwd=REPO_ROOT,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if baseline.returncode:
        pytest.skip("submitted baseline commit is unavailable in this git object database")
    result_path = tmp_path / "result.json"
    command = [
        sys.executable,
        str(BENCHMARK),
        "--suite",
        "equivalence",
        "--tokenizer-path",
        str(tokenizer),
        "--real-scan-rows",
        "96",
        "--work-dir",
        str(tmp_path / "work"),
        "--output-json",
        str(result_path),
    ]
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(result_path.read_text(encoding="utf-8"))

    assert result["baseline_revision"] == BASELINE_REVISION
    assert result["baseline_method"] == "git archive read-only export; separate Python worker"
    assert result["all_exact"] is True
    if (TRACES_ROOT / "data" / "train-00035-of-00039.parquet").is_file():
        assert result["actual_source_fixture"]["kind"] == "real"

    cases = {case["name"]: case for case in result["cases"]}
    assert set(cases) == {
        "stateful-selection",
        "identical-c-b",
        "qa-mixing",
        "tight-presented-cap",
    }
    for case in cases.values():
        assert case["comparison"]["exact"] is True
        assert case["comparison"]["mismatched_files"] == []
        assert case["baseline"]["manifest"] == case["current"]["manifest"]
        assert case["baseline"]["artifacts"]["sha256"] == case["current"]["artifacts"]["sha256"]

    stateful = cases["stateful-selection"]["coverage"]
    assert stateful == {
        "unselected_intermediate_decisions": True,
        "retrieval": True,
        "retention": True,
        "lease_expiry": True,
    }
    identical = cases["identical-c-b"]["coverage"]
    assert identical == {"allow_unchanged_b": True, "all_views_identical": True}
    qa = cases["qa-mixing"]["coverage"]
    assert qa["qa_base_decisions"] == 3
    assert qa["actual_qa_base_decision_fraction"] == pytest.approx(0.15)
    assert qa["actual_qa_base_decision_fraction"] == qa["target_fraction"]
    tight = cases["tight-presented-cap"]["coverage"]
    assert tight["skipped_exposures"] > 0
    assert tight["partial_decision"] is True
    assert tight["long_session_precedes_zero_session"] is True
    assert tight["later_zero_gist_pair_written"] is True
