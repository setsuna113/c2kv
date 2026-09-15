"""End-to-end metadata admission for BFCL exact-overlap exclusions.

These fixtures are synthetic interface fixtures only.  They exercise the
prepared-corpus and checkpoint contracts without labelling any task a formal
held-out result.
"""
from __future__ import annotations

import copy
import hashlib
import json
import shutil
import sys
from types import ModuleType
from pathlib import Path

import pytest

from benchmarks.memory_runtime.audit_b_training_overlap import audit_formal_corpus
from benchmarks.memory_runtime.bfcl_overlap_admission import (
    official_question_path, validate_overlap_admission,
)
from benchmarks.memory_runtime.tests.test_audit_b_training_overlap import (
    _write_checkpoint,
    _write_corpus,
    _write_json,
    _write_jsonl,
)


ELIGIBLE_IDS = ("multi_turn_base_1", "multi_turn_base_2", "multi_turn_base_3")
ID_EXPOSED = "multi_turn_base_01"
PROMPT_EXPOSED = "multi_turn_base_4"


def _write_official_questions(tmp_path: Path, name: str = "questions.jsonl") -> Path:
    """Write input-shaped BFCL questions; no possible answers are present."""
    path = tmp_path / name
    prompts = {
        "multi_turn_base_1": "safe BFCL question one",
        "multi_turn_base_2": "safe BFCL question two",
        "multi_turn_base_3": "safe BFCL question three",
        # This is a complete canonical task-ID overlap with the corpus fixture.
        ID_EXPOSED: "different text cannot make this ID eligible",
        # This is a normalized user-prompt overlap with the corpus fixture.
        PROMPT_EXPOSED: " make a    reminder ",
    }
    _write_jsonl(
        path,
        [
            {"id": task_id, "question": [[{"role": "user", "content": prompt}]]}
            for task_id, prompt in prompts.items()
        ],
    )
    return path


def _bound_inputs(tmp_path: Path) -> dict[str, object]:
    corpus_dir, corpus_identity = _write_corpus(tmp_path)
    questions = _write_official_questions(tmp_path)
    checkpoint = _write_checkpoint(tmp_path, corpus_identity)
    audit = audit_formal_corpus(
        corpus_dir,
        bfcl_tasks_path=questions,
        candidate_ids=[*ELIGIBLE_IDS, ID_EXPOSED, PROMPT_EXPOSED],
        dev_task_ids=[],
        checkpoint_dirs=[checkpoint],
    )
    assert audit["status"] == "checkpoint_bound"
    assert audit["eligible_candidate_ids"] == list(ELIGIBLE_IDS)
    assert set(audit["excluded_candidate_ids"]) == {ID_EXPOSED, PROMPT_EXPOSED}
    audit_path = tmp_path / "checkpoint-bound-audit.json"
    _write_json(audit_path, audit)
    return {
        "audit": audit,
        "audit_path": audit_path,
        "corpus_identity": corpus_identity,
        "checkpoint": checkpoint,
        "questions": questions,
    }


def _ready(checkpoint: Path, corpus_identity: str, task_ids: list[str]) -> dict:
    resolved = str(checkpoint.resolve())
    return {
        "schema": "a-event-native-server-v1",
        "status": "ready",
        "benchmark": "bfcl",
        "source_profile": "native-v1",
        "allowed_task_ids": task_ids,
        "checkpoint_path": resolved,
        "checkpoint": {
            "checkpoint": resolved,
            "corpus_identity": corpus_identity,
            "training_arm": "B",
        },
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_admits_eligible_actual_questions_with_relocated_identical_checkpoint(tmp_path: Path) -> None:
    inputs = _bound_inputs(tmp_path)
    relocated = tmp_path / "relocated-checkpoint"
    shutil.copytree(inputs["checkpoint"], relocated)
    ready = _ready(relocated, inputs["corpus_identity"], list(ELIGIBLE_IDS[:2]))

    admitted = validate_overlap_admission(
        inputs["audit_path"],
        ready,
        inputs["questions"],
        expected_audit_sha256=_sha256(inputs["audit_path"]),
    )

    assert admitted["schema"] == "a-bfcl-overlap-admission-v1"
    assert admitted["status"] == "passed"
    assert admitted["task_ids"] == list(ELIGIBLE_IDS[:2])
    assert admitted["formal_split_frozen"] is False
    assert admitted["audit"]["sha256"] == _sha256(inputs["audit_path"])
    assert {
        key: admitted["bfcl_source"][key] for key in ("sha256", "bytes")
    } == {
        key: inputs["audit"]["bfcl_source"][key] for key in ("sha256", "bytes")
    }
    assert admitted["checkpoint"]["corpus_identity"] == inputs["corpus_identity"]
    bound = inputs["audit"]["checkpoint_binding"]["checkpoints"][0]
    for field in ("config", "trainer_state"):
        assert {
            key: admitted["checkpoint"][field][key] for key in ("sha256", "bytes")
        } == {
            key: bound[field][key] for key in ("sha256", "bytes")
        }


@pytest.mark.parametrize("exposed_id", [ID_EXPOSED, PROMPT_EXPOSED])
def test_rejects_exact_id_or_prompt_exposed_task_before_admission(
    tmp_path: Path, exposed_id: str
) -> None:
    inputs = _bound_inputs(tmp_path)
    ready = _ready(inputs["checkpoint"], inputs["corpus_identity"], [exposed_id])

    with pytest.raises(ValueError):
        validate_overlap_admission(inputs["audit_path"], ready, inputs["questions"])


def test_rejects_question_source_swap_even_when_task_ids_are_unchanged(tmp_path: Path) -> None:
    inputs = _bound_inputs(tmp_path)
    swapped = _write_official_questions(tmp_path, "swapped-questions.jsonl")
    rows = [json.loads(line) for line in swapped.read_text(encoding="utf-8").splitlines()]
    rows[0]["question"][0][0]["content"] = "changed source text"
    _write_jsonl(swapped, rows)
    ready = _ready(inputs["checkpoint"], inputs["corpus_identity"], [ELIGIBLE_IDS[0]])

    with pytest.raises(ValueError):
        validate_overlap_admission(inputs["audit_path"], ready, swapped)


def test_rejects_checkpoint_bytes_or_declared_corpus_identity_that_differ_from_audit(
    tmp_path: Path,
) -> None:
    inputs = _bound_inputs(tmp_path)
    swapped = tmp_path / "modified-checkpoint"
    shutil.copytree(inputs["checkpoint"], swapped)
    config_path = swapped / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["history_memory_seed"] = 18
    _write_json(config_path, config)
    ready = _ready(swapped, inputs["corpus_identity"], [ELIGIBLE_IDS[0]])

    with pytest.raises(ValueError):
        validate_overlap_admission(inputs["audit_path"], ready, inputs["questions"])

    wrong_corpus_ready = copy.deepcopy(
        _ready(inputs["checkpoint"], inputs["corpus_identity"], [ELIGIBLE_IDS[0]])
    )
    wrong_corpus_ready["checkpoint"]["corpus_identity"] = "another-corpus"
    with pytest.raises(ValueError):
        validate_overlap_admission(inputs["audit_path"], wrong_corpus_ready, inputs["questions"])


def test_rejects_corpus_only_audit(tmp_path: Path) -> None:
    inputs = _bound_inputs(tmp_path)
    corpus_only = dict(inputs["audit"])
    corpus_only["status"] = "corpus_only"
    corpus_only["checkpoint_binding"] = {
        "status": "not_provided",
        "corpus_identity": inputs["corpus_identity"],
        "checkpoints": [],
    }
    audit_path = tmp_path / "corpus-only-audit.json"
    _write_json(audit_path, corpus_only)
    ready = _ready(inputs["checkpoint"], inputs["corpus_identity"], [ELIGIBLE_IDS[0]])

    with pytest.raises(ValueError):
        validate_overlap_admission(audit_path, ready, inputs["questions"])


def test_rejects_audit_changed_between_parent_and_worker(tmp_path: Path) -> None:
    inputs = _bound_inputs(tmp_path)
    expected = _sha256(inputs["audit_path"])
    changed = dict(inputs["audit"])
    changed["worker_note"] = "new parent-side audit bytes"
    _write_json(inputs["audit_path"], changed)
    ready = _ready(inputs["checkpoint"], inputs["corpus_identity"], [ELIGIBLE_IDS[0]])

    with pytest.raises(ValueError):
        validate_overlap_admission(
            inputs["audit_path"],
            ready,
            inputs["questions"],
            expected_audit_sha256=expected,
        )


def test_actual_imported_package_version_and_prompt_directory_are_used(tmp_path, monkeypatch):
    config = ModuleType('bfcl_eval.constants.eval_config')
    config.PROMPT_PATH = tmp_path / 'installed-package-data'
    config.VERSION_PREFIX = 'BFCL_v4'
    monkeypatch.setitem(sys.modules, config.__name__, config)
    assert official_question_path() == config.PROMPT_PATH / 'BFCL_v4_multi_turn_base.json'
    config.VERSION_PREFIX = 'BFCL_v5'
    with pytest.raises(ValueError, match='imported BFCL version differs'):
        official_question_path()
