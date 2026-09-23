"""Only the final selected generation view advances resident history."""

from contextlib import contextmanager
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "python")]

from benchmarks.memory_runtime import budget_guard, event_native_step
from benchmarks.memory_runtime.attempt_journal import AttemptJournal
from benchmarks.memory_runtime.event_native_draft import NativeDraft
from benchmarks.memory_runtime.event_native_step import (
    EventNativeDecisionRunner, EventNativeStepError,
)


class Memory:
    def __init__(self, label):
        self.label = label

    def costs(self, ratio):
        return {"resident_kv_tokens": 1}


class Controller:
    def __init__(self, *, regenerate=False, fallback=False):
        self.initial = Memory("initial")
        self.upgraded = Memory("upgraded")
        self.regenerate = regenerate
        self.fallback = fallback
        self.commits = []

    def prepare(self, payload, **kwargs):
        return SimpleNamespace(memory=self.initial, metadata={"decision_index": 1})

    def reconsider(self, prepared, calls, **kwargs):
        return {
            "regenerate": self.regenerate,
            "memory": self.upgraded if self.regenerate else prepared.memory,
            "metadata": prepared.metadata,
            "decision": {"reason": "test_recovery"},
        }

    def validate_commit(self, prepared, calls, **kwargs):
        return {
            "accepted": not self.fallback,
            "fallback": "original" if self.fallback else None,
            "reason": "test_commit",
        }

    def commit_memory(self, prepared, final_memory):
        self.commits.append(final_memory)
        return {"selected_view": final_memory.label}


class Generator:
    def __init__(self, *, fail=False):
        self.fail = fail
        self.calls = []
        self.resolutions = []

    @contextmanager
    def decision_scope(self, *, session_id):
        yield

    def generate(self, memory, *, ratio, max_new_tokens, **kwargs):
        self.calls.append(memory)
        if self.fail:
            raise RuntimeError("generation failed")
        return SimpleNamespace(
            token_ids=(len(self.calls),), token_logprobs=(-0.1,),
            finish_reason="stop", stats={},
        )

    def resolve_decision(self, response, *, result, record):
        self.resolutions.append(response)
        return {"resolved": True}

    def session_cache_info(self):
        return {"status": "empty"}

    def close_session(self):
        return None


@pytest.mark.parametrize(
    "regenerate,fallback,expected",
    [(False, False, "initial"), (True, False, "upgraded"),
     (True, True, "initial")],
)
def test_runner_commits_only_selected_final_view(
    tmp_path, monkeypatch, regenerate, fallback, expected,
):
    monkeypatch.setattr(event_native_step, "memory_to_dict",
                        lambda memory: {"label": memory.label})
    monkeypatch.setattr(budget_guard, "history_budget_receipt",
                        lambda *args, **kwargs: {"status": "passed", "errors": []})
    monkeypatch.setattr(event_native_step, "decode_native_generation",
                        lambda *args, **kwargs: NativeDraft("ok", "ok", (), "text", ""))
    controller = Controller(regenerate=regenerate, fallback=fallback)
    generator = Generator()
    runner = EventNativeDecisionRunner(
        controller, generator, object(), ratio=4, max_new_tokens=8,
        max_generation_calls=2,
        journal=AttemptJournal(tmp_path / "attempts.jsonl"),
    )
    payload = {"session_id": "task", "decision_key": "0", "messages": []}
    record = runner.run(payload)
    assert record["status"] == "ok"
    assert [memory.label for memory in controller.commits] == [expected]
    assert len(generator.resolutions) == 1
    assert record["history_state_commit"] == {"selected_view": expected}
    assert runner.run(payload) == record
    assert len(controller.commits) == 1


def test_failed_generation_never_commits_resident_history(tmp_path, monkeypatch):
    monkeypatch.setattr(event_native_step, "memory_to_dict",
                        lambda memory: {"label": memory.label})
    monkeypatch.setattr(budget_guard, "history_budget_receipt",
                        lambda *args, **kwargs: {"status": "passed", "errors": []})
    controller = Controller()
    generator = Generator(fail=True)
    runner = EventNativeDecisionRunner(
        controller, generator, object(), ratio=4, max_new_tokens=8,
        max_generation_calls=2,
        journal=AttemptJournal(tmp_path / "attempts.jsonl"),
    )
    with pytest.raises(EventNativeStepError, match="generation failed") as failure:
        runner.run({"session_id": "task", "decision_key": "0", "messages": []})
    assert failure.value.record["status"] == "failed"
    assert controller.commits == []
    assert generator.resolutions == []
