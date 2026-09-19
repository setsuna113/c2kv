"""A calibration draft is scored without running a recovery decision."""

from contextlib import contextmanager
from dataclasses import dataclass
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "python")]

from benchmarks.memory_runtime.attempt_journal import AttemptJournal
from benchmarks.memory_runtime.event_native_step import EventNativeDecisionRunner
from benchmarks.memory_runtime.recovery.experiment import GPRecoveryController


def test_calibration_risk_uses_frozen_predictor_without_reconsider(monkeypatch):
    controller = object.__new__(GPRecoveryController)
    controller.set_protocol = True
    controller.gp = {"set_selector": "risk"}
    prepared = SimpleNamespace(
        _store=SimpleNamespace(session_id="bfcl/task/attempt-0/state"),
        metadata={"decision_key": "turn-1/step-2"},
        _checked_result=None,
    )
    controller._prepared = {(prepared._store.session_id, "turn-1/step-2"): prepared}
    contexts = []
    controller.trained_selector = SimpleNamespace(model=SimpleNamespace(
        predict_risk=lambda context: (
            contexts.append(context) or SimpleNamespace(
                available=True, score=0.7, reason=None, missing_fields=()))))
    monkeypatch.setattr(
        "benchmarks.memory_runtime.recovery.set_protocol.context_from_prepared",
        lambda *args: {"held_draft": args[2]},
    )

    risk = controller.calibration_risk(
        prepared, [], draft_text="Done.")

    assert risk["available"] is True
    assert risk["score"] == 0.7
    assert contexts == [{"held_draft": "Done."}]
    assert prepared._checked_result is None


def test_recovery_disabled_records_risk_and_never_reconsiders(tmp_path, monkeypatch):
    from benchmarks.memory_runtime import event_native_step

    monkeypatch.setattr(
        "benchmarks.memory_runtime.budget_guard.history_budget_receipt",
        lambda *args, **kwargs: {"status": "passed"},
    )
    monkeypatch.setattr(event_native_step, "memory_to_dict", lambda memory: {})
    @dataclass
    class Draft:
        status: str = "text"
        text: str = "Done."
        content: str = "Done."
        tool_calls: tuple = ()
        reasoning_content: str | None = None
        reason: str | None = None

    monkeypatch.setattr(event_native_step, "decode_native_generation",
                        lambda *args, **kwargs: Draft())
    memory = SimpleNamespace(costs=lambda ratio: {"resident_kv_tokens": 3})
    calls = []

    class Controller:
        def prepare(self, payload, **kwargs):
            return SimpleNamespace(memory=memory, metadata={"decision_index": 1})

        def calibration_risk(self, prepared, draft_tool_calls, **kwargs):
            calls.append("risk")
            assert kwargs["draft_text"] == "Done."
            return {"available": True, "score": 0.7}

        def reconsider(self, *args, **kwargs):
            raise AssertionError("calibration must not reconsider or regenerate")

    class Generator:
        @contextmanager
        def decision_scope(self, **kwargs):
            yield

        def generate(self, *args, **kwargs):
            calls.append("generate")
            return SimpleNamespace(token_ids=(1,), token_logprobs=(-0.1,),
                                   finish_reason="stop", stats={})

        def session_cache_info(self):
            return {}

        def close_session(self):
            pass

    runner = EventNativeDecisionRunner(
        Controller(), Generator(), object(), ratio=4, max_new_tokens=8,
        max_generation_calls=2,
        journal=AttemptJournal(tmp_path / "attempts.jsonl"),
    )
    record = runner.run({"session_id": "bfcl/task/attempt-0/state",
                         "decision_key": "turn-1/step-2", "recovery_disabled": True})

    assert calls == ["generate", "risk"]
    assert record["risk"]["score"] == 0.7
    assert record["exact_recovery"]["reason"] == "recovery_disabled"
    assert record["generation_attempts"] == 1
