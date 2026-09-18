"""Calibration retry-cache isolation through the actual controller API."""
from __future__ import annotations

import copy
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "generality"), str(ROOT / "controller_runtime"),
               str(ROOT / "controller_runtime" / "python")]

from benchmarks.memory_runtime.event_native_api import EventNativeAPI, EventNativeAPIError  # noqa: E402
from generality import calibrate  # noqa: E402


class FakeRunner:
    def __init__(self):
        self.payloads = []

    def run(self, payload):
        self.payloads.append(copy.deepcopy(payload))
        return {"status": "ok", "session_id": payload["session_id"],
                "decision_key": payload["decision_key"],
                "response": {"role": "assistant", "content": payload["session_id"],
                             "finish_reason": "stop"},
                "generation_usage_total": {"prompt_tokens": 4, "completion_tokens": 1,
                                           "total_tokens": 5}}


class CalibrationControllerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "steps.jsonl"
        self.runner = FakeRunner()
        self.task_id = "multi_turn_base_3"
        self.api = EventNativeAPI(
            self.runner, run_id="test", model_name="qwen", view_mode="static",
            max_new_tokens=128, allowed_task_ids=[self.task_id], max_decisions=10,
            deadline_monotonic=time.monotonic() + 60, steps_path=self.path)

    def request(self, state_id=None, step=3):
        context = {"benchmark": "bfcl", "task_id": self.task_id,
                   "user_turn": 1, "step": step, "attempt": 0}
        if state_id is not None:
            context.update({"calibration_state_id": state_id, "recovery_disabled": True})
        return {"model": "qwen", "messages": [{"role": "user", "content": "Current request"}],
                "tools": [], "temperature": 0, "store": False,
                "max_completion_tokens": 128, "c2kv_eval_context": context}

    def test_two_states_of_same_task_and_decision_run_independently(self):
        first = self.api.handle_chat(self.request("first-state"))
        second = self.api.handle_chat(self.request("second-state"))
        self.assertEqual(len(self.runner.payloads), 2)
        self.assertEqual(self.api.decisions_reserved, 2)
        self.assertNotEqual(first["choices"][0]["message"]["content"],
                            second["choices"][0]["message"]["content"])
        records = [json.loads(line) for line in self.path.read_text().splitlines()]
        self.assertEqual([record["session_id"] for record in records], [
            f"bfcl/{self.task_id}/attempt-0/first-state",
            f"bfcl/{self.task_id}/attempt-0/second-state"])

    def test_overlapping_continuation_and_later_source_target_do_not_conflict(self):
        self.api.handle_chat(self.request("first-state", step=3))
        self.api.handle_chat(self.request("first-state", step=4))
        self.api.handle_chat(self.request("second-state", step=4))
        self.assertEqual(len(self.runner.payloads), 3)
        self.assertEqual(self.api.decisions_reserved, 3)
        self.assertEqual([payload["decision_key"] for payload in self.runner.payloads],
                         ["turn-1/step-3", "turn-1/step-4", "turn-1/step-4"])

    def test_identical_retry_of_same_state_returns_cached_response_without_generation(self):
        request = self.request("first-state")
        first = self.api.handle_chat(request)
        self.assertEqual(self.api.handle_chat(copy.deepcopy(request)), first)
        self.assertEqual(len(self.runner.payloads), 1)
        self.assertEqual(self.api.decisions_reserved, 1)
        self.assertEqual(len(self.path.read_text().splitlines()), 1)

    def test_reusing_same_state_decision_with_different_input_still_conflicts(self):
        request = self.request("first-state")
        self.api.handle_chat(request)
        request["messages"][0]["content"] = "Different request"
        with self.assertRaises(EventNativeAPIError) as error:
            self.api.handle_chat(request)
        self.assertEqual(error.exception.status_code, 409)
        self.assertEqual(error.exception.code, "decision_conflict")
        self.assertEqual(len(self.runner.payloads), 1)

    def test_legacy_task_retry_keeps_original_cache_behavior(self):
        request = self.request()
        first = self.api.handle_chat(request)
        self.assertEqual(self.api.handle_chat(copy.deepcopy(request)), first)
        self.assertEqual(len(self.runner.payloads), 1)
        self.assertEqual(self.runner.payloads[0]["session_id"], f"bfcl/{self.task_id}/attempt-0")

    def test_calibration_client_payload_is_accepted_by_actual_controller_schema(self):
        client = calibrate.EventNativeControllerClient(
            "http://127.0.0.1:1", "qwen", "bfcl", self.path, "source-state", self.task_id)
        with patch.object(calibrate, "_post", side_effect=lambda url, body: self.api.handle_chat(body)):
            response = client.generate({"decision_key": "turn-1/step-3", "messages": [
                {"role": "user", "content": "Current request"}]}, 128)
        self.assertEqual(response["choices"][0]["message"]["content"],
                         f"bfcl/{self.task_id}/attempt-0/source-state")
        self.assertIs(self.runner.payloads[0]["recovery_disabled"], True)


if __name__ == "__main__":
    unittest.main()
