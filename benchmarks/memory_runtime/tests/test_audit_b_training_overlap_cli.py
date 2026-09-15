"""CPU acceptance tests for the formal-corpus audit CLI."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from benchmarks.memory_runtime import audit_b_training_overlap_cli as cli
from benchmarks.memory_runtime.tests.test_audit_b_training_overlap import (
    _write_bfcl_tasks,
    _write_checkpoint,
    _write_corpus,
    _write_json,
)


CANDIDATE_IDS = [
    "multi_turn_base_01",
    "prompt-only",
    "future-only",
    "bfcl:multi_turn_base_01",
    "development-task",
    "unseen",
]


def _source_sha256(path: Path) -> str:
    payload = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(payload).hexdigest()


def _selection_paths(
    tmp_path: Path, tasks_path: Path, *, dev_source_sha256: str | None = None
) -> tuple[Path, Path]:
    source_sha256 = _source_sha256(tasks_path)
    candidate_path = tmp_path / "candidate-manifest.json"
    dev_path = tmp_path / "development-manifest.json"
    _write_json(
        candidate_path,
        {
            "ids": CANDIDATE_IDS,
            "n_total": len(CANDIDATE_IDS),
            "source_sha256": source_sha256,
        },
    )
    _write_json(
        dev_path,
        {
            "ids": ["development-task"],
            "n_total": 1,
            "source_sha256": dev_source_sha256 or source_sha256,
        },
    )
    return candidate_path, dev_path


def _argv(
    corpus_dir: Path,
    tasks_path: Path,
    candidate_path: Path,
    dev_path: Path,
    output_path: Path,
    *,
    checkpoint_dir: Path | None = None,
) -> list[str]:
    args = [
        "--corpus-dir",
        str(corpus_dir),
        "--bfcl-tasks",
        str(tasks_path),
        "--candidate-ids",
        str(candidate_path),
        "--dev-exclusion-ids",
        str(dev_path),
        "--output",
        str(output_path),
    ]
    if checkpoint_dir is not None:
        args.extend(["--checkpoint-dir", str(checkpoint_dir)])
    return args


def test_cli_writes_checkpoint_bound_audit_from_metadata_only_fixture(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    corpus_dir, manifest_sha256 = _write_corpus(tmp_path)
    tasks_path = _write_bfcl_tasks(tmp_path)
    checkpoint_dir = _write_checkpoint(tmp_path, manifest_sha256)
    candidate_path, dev_path = _selection_paths(tmp_path, tasks_path)
    output_path = tmp_path / "audit.json"

    assert cli.main(
        _argv(
            corpus_dir,
            tasks_path,
            candidate_path,
            dev_path,
            output_path,
            checkpoint_dir=checkpoint_dir,
        )
    ) == 0

    summary = json.loads(capsys.readouterr().out)
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert summary["status"] == report["status"] == "checkpoint_bound"
    assert summary["formal_split_frozen"] is False
    assert report["input_selection"]["dev_union_task_count"] == 1


def test_cli_rejects_mismatched_development_question_source_without_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    corpus_dir, _ = _write_corpus(tmp_path)
    tasks_path = _write_bfcl_tasks(tmp_path)
    candidate_path, dev_path = _selection_paths(
        tmp_path, tasks_path, dev_source_sha256="0" * 64
    )
    output_path = tmp_path / "must-not-exist.json"

    with pytest.raises(SystemExit) as error:
        cli.main(_argv(corpus_dir, tasks_path, candidate_path, dev_path, output_path))

    assert error.value.code == 2
    assert "declared BFCL question-source identity differs" in capsys.readouterr().err
    assert not output_path.exists()


def test_cli_refuses_existing_output_without_overwriting_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    corpus_dir, _ = _write_corpus(tmp_path)
    tasks_path = _write_bfcl_tasks(tmp_path)
    candidate_path, dev_path = _selection_paths(tmp_path, tasks_path)
    output_path = tmp_path / "existing-audit.json"
    original = b'{"preserve":true}\n'
    output_path.write_bytes(original)

    with pytest.raises(SystemExit) as error:
        cli.main(_argv(corpus_dir, tasks_path, candidate_path, dev_path, output_path))

    assert error.value.code == 2
    assert "output already exists" in capsys.readouterr().err
    assert output_path.read_bytes() == original
