"""Contract tests for the bounded ACE formal-corpus overlap reader."""
from __future__ import annotations

from pathlib import Path

import pytest

from benchmarks.memory_runtime.audit_b_training_overlap_ace import audit_ace_formal_corpus
from benchmarks.memory_runtime.tests.test_audit_b_training_overlap import (
    _write_corpus,
    _write_jsonl,
)


def _ace_row(task_id: str, question: str, **extra: object) -> dict[str, object]:
    row: dict[str, object] = {
        "id": task_id,
        "question": question,
        "initial_config": {},
        "path": [],
        "function": [{"name": "Lookup"}],
        "involved_classes": ["MessageApi"],
    }
    row.update(extra)
    return row


def _tasks(
    tmp_path: Path, rows: list[dict[str, object]], *, category: str = "agent_multi_step"
) -> Path:
    path = tmp_path / f"data_{category}.json"
    _write_jsonl(path, rows)
    return path


def _audit(corpus_dir: Path, tasks_path: Path, *, candidates: list[str], development: list[str]):
    return audit_ace_formal_corpus(
        corpus_dir,
        ace_tasks_path=tasks_path,
        benchmark="acebench",
        language="en",
        category="agent_multi_step",
        candidate_ids=candidates,
        dev_task_ids=development,
    )


def test_agent_visible_question_matches_selected_prefix_with_different_task_id_and_excludes_dev(
    tmp_path: Path,
) -> None:
    corpus_dir, _ = _write_corpus(tmp_path)
    prompt_match = "agent_multi_step_prompt_match"
    future_only = "agent_multi_step_future_only"
    development = "agent_multi_step_development"
    unseen = "agent_multi_step_unseen"
    tasks_path = _tasks(
        tmp_path,
        [
            _ace_row(prompt_match, "  make a reminder  "),
            _ace_row(future_only, "Future-only request"),
            _ace_row(development, "Independent development candidate"),
            _ace_row(unseen, "Never used"),
        ],
    )

    report = _audit(
        corpus_dir,
        tasks_path,
        candidates=[prompt_match, future_only, development, unseen],
        development=[development],
    )

    assert report["schema"] == "c2kv-b-training-overlap-ace-v1"
    assert report["formal_split_frozen"] is False
    assert report["ace_source"]["compared_text_field"] == "question"
    assert report["ace_source"]["compared_text_surface"] == "agent_visible_initial_user_query"
    candidate = report["overlap"]["candidate"]
    assert candidate["canonical_exact_task_ids"]["matched_task_ids"] == []
    assert candidate["normalized_exact_user_prompts"]["matched_task_ids"] == [prompt_match]
    assert future_only in report["eligible_candidate_ids"]
    assert unseen in report["eligible_candidate_ids"]
    assert development in report["excluded_candidate_ids"]
    assert future_only not in candidate["normalized_exact_user_prompts"]["matched_task_ids"]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"benchmark": "bfcl"}, "benchmark"),
        ({"language": "zh"}, "language"),
        ({"category": "other"}, "category"),
        ({"category": "agent_multi_turn"}, "agent_multi_turn"),
    ],
)
def test_rejects_non_ace_or_non_agent_multistep_contract(
    tmp_path: Path, kwargs: dict[str, str], message: str
) -> None:
    corpus_dir, _ = _write_corpus(tmp_path)
    task_id = "agent_multi_step_candidate"
    args = {
        "ace_tasks_path": _tasks(tmp_path, [_ace_row(task_id, "A direct user query")]),
        "benchmark": "acebench",
        "language": "en",
        "category": "agent_multi_step",
        "candidate_ids": [task_id],
        "dev_task_ids": [],
    }
    args.update(kwargs)

    with pytest.raises(ValueError, match=message):
        audit_ace_formal_corpus(corpus_dir, **args)


def test_rejects_gold_shaped_row_and_whitespace_only_question(tmp_path: Path) -> None:
    corpus_dir, _ = _write_corpus(tmp_path)
    task_id = "agent_multi_step_candidate"
    gold_shaped = _ace_row(task_id, "A direct user query", ground_truth=[])
    with pytest.raises(ValueError):
        _audit(corpus_dir, _tasks(tmp_path, [gold_shaped]), candidates=[task_id], development=[])

    whitespace_path = _tasks(
        tmp_path / "whitespace", [_ace_row(task_id, " \t\n ")]
    )
    with pytest.raises(ValueError):
        _audit(corpus_dir, whitespace_path, candidates=[task_id], development=[])


def test_rejects_duplicate_official_id_even_when_it_is_not_requested(tmp_path: Path) -> None:
    corpus_dir, _ = _write_corpus(tmp_path)
    requested = "agent_multi_step_requested"
    duplicate = "agent_multi_step_duplicate"
    tasks_path = _tasks(
        tmp_path,
        [
            _ace_row(requested, "Requested user query"),
            _ace_row(duplicate, "First unrequested query"),
            _ace_row(duplicate, "Second unrequested query"),
        ],
    )

    with pytest.raises(ValueError):
        _audit(corpus_dir, tasks_path, candidates=[requested], development=[])
