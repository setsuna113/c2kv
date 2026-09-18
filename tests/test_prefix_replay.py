"""Recorded-source replay contracts without a model or BFCL installation."""
from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from generality.prefix_replay import (  # noqa: E402
    PrefixReplayIntegrityError,
    ReplayedEnvironment,
    bind_source_trace,
    restore_bfcl_prefix,
)


def test_current_payload_does_not_open_bfcl_slot_again():
    class Env:
        def __init__(self):
            self.next_payload_calls = 0

        def next_payload(self):
            self.next_payload_calls += 1
            raise AssertionError("current_payload reopened the pending BFCL slot")

    env = Env()
    pending = {"decision_key": "turn-1/step-2", "messages": []}
    replay = ReplayedEnvironment(
        env=env,
        bindings=None,
        task={},
        ground_truth=[],
        turn_index=1,
        prefix_payload_sha256="sha",
        replayed_assistant_messages=1,
        replayed_tool_messages=1,
        previous_turn_valid=True,
        pending_payload=pending,
    )

    assert replay.current_payload() == pending
    assert env.next_payload_calls == 0


def tool_response(name):
    return {"role": "assistant", "content": None, "tool_calls": [{
        "id": f"call-{name}", "type": "function",
        "function": {"name": name, "arguments": "{}"}}]}


def tool_observation(name):
    return {"role": "tool", "content": f"ok:{name}",
            "tool_call_id": f"call-{name}"}


class FakeBFCLTaskEnvironment:
    """Enforce BFCL's pending-slot and stop/global-step transitions."""

    def __init__(self, task, ground_truth, *, bindings, namespace):
        self.task = task
        self.bindings = bindings
        self.handler = bindings.make_handler(namespace)
        self.turn_index = 0
        self.global_step = 0
        self.awaiting_response = False
        self.next_payload_calls = 0
        self.opened_decisions = []
        self.committed_responses = []
        self.executed_actions = []
        self.all_model_response = []
        self.current_turn_actions = []
        self.inference_data = {"message": [
            {"role": "system", "content": "Official instructions"},
            *copy.deepcopy(task["question"][0])], "tools": task["function"]}

    def next_payload(self):
        if self.awaiting_response:
            raise RuntimeError("Commit the outstanding BFCL response first")
        self.next_payload_calls += 1
        self.awaiting_response = True
        key = f"turn-{self.turn_index}/step-{self.global_step}"
        self.opened_decisions.append(key)
        return {"session_id": f"bfcl/{self.task['id']}/attempt-0",
                "decision_key": key,
                "messages": copy.deepcopy(self.inference_data["message"]),
                "tools": copy.deepcopy(self.inference_data["tools"])}

    def commit_response(self, response):
        if not self.awaiting_response:
            raise RuntimeError("No BFCL response is awaiting commit")
        self.awaiting_response = False
        self.global_step += 1
        self.committed_responses.append(copy.deepcopy(response))
        self.inference_data["message"].append(copy.deepcopy(response))
        calls = response.get("tool_calls") or []
        if calls:
            for call in calls:
                name = call["function"]["name"]
                self.executed_actions.append(name)
                self.current_turn_actions.append(name)
                self.inference_data["message"].append(tool_observation(name))
        else:
            self.all_model_response.append(self.current_turn_actions)
            self.current_turn_actions = []
            self.turn_index += 1
            self.inference_data["message"].extend(
                copy.deepcopy(self.task["question"][self.turn_index]))

    def previous_turn_valid(self):
        return self.turn_index == 0 or bool(self.all_model_response[-1])


class FakeBindings:
    environment_class = FakeBFCLTaskEnvironment
    model_style = "fake-openai-completions"

    def make_handler(self, namespace):
        return SimpleNamespace(namespace=namespace)

    def load_task(self, task_id):
        return {"id": task_id, "question": [
            [{"role": "user", "content": "Original request"}],
            [{"role": "user", "content": "Current request"}]],
            "function": [{"name": "hidden_action"}, {"name": "visible_action"},
                         {"name": "new_action"}]}, [["hidden_action()"], ["new_action()"]]


class PrefixReplaySemanticTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.source_root = Path(self.temp.name)
        self.task_id = "multi_turn_base_3"
        self.session_id = f"bfcl/{self.task_id}/attempt-0"
        self.path = (self.source_root / "workers" / "worker-0" / "tasks" /
                     f"source-{self.task_id}" / "actor" / "steps.jsonl")
        self.path.parent.mkdir(parents=True)
        self.row = {
            "state_id": "recorded-state", "task_id": self.task_id,
            "task_group_id": self.task_id, "decision_key": "turn-1/step-3",
            "q": {"session_id": self.session_id, "goal": "Current request",
                  "raw_source_ids": [self.session_id + ":m3", self.session_id + ":m6"],
                  "raw_visible": [tool_response("visible_action"),
                                  tool_observation("visible_action"),
                                  {"role": "user", "content": "Current request"}],
                  "last_action_observation": [tool_response("visible_action"),
                                              tool_observation("visible_action")]}}
        self.records = [
            self.record("turn-0/step-0", tool_response("hidden_action")),
            self.record("turn-0/step-1", tool_response("visible_action")),
            self.record("turn-0/step-2", {"role": "assistant", "content": "done"}),
            self.record("turn-1/step-3", {"role": "assistant", "content": "target-draft"}),
            self.record("turn-0/step-0", tool_response("branch-action")),
            self.record("turn-1/step-3", tool_response("branch-target"))]
        self.write_trace()

    def record(self, decision_key, response):
        return {"session_id": self.session_id, "decision_key": decision_key,
                "status": "ok", "response": response}

    def write_trace(self):
        self.path.write_text("".join(json.dumps(record) + "\n" for record in self.records),
                             encoding="utf-8")

    def restore(self, row=None):
        bound = bind_source_trace(self.row if row is None else row, self.source_root)
        return restore_bfcl_prefix(bound, FakeBindings())

    def test_recorded_stop_consumes_step_and_advances_turn(self):
        replay = self.restore()
        self.assertEqual(replay.env.opened_decisions, [
            "turn-0/step-0", "turn-0/step-1", "turn-0/step-2", "turn-1/step-3"])
        self.assertEqual(replay.env.global_step, 3)
        self.assertEqual(replay.env.turn_index, 1)
        self.assertEqual(replay.env.committed_responses[-1],
                         {"role": "assistant", "content": "done"})
        self.assertEqual(replay.env.executed_actions, ["hidden_action", "visible_action"])
        self.assertEqual(replay.replayed_assistant_messages, 3)
        self.assertEqual(replay.replayed_tool_messages, 2)
        self.assertIs(replay.previous_turn_valid, True)

    def test_restore_current_twice_commit_then_next_preserves_pending_slot(self):
        replay = self.restore()
        first = replay.current_payload()
        second = replay.current_payload()
        self.assertEqual(first, second)
        self.assertEqual(first["decision_key"], "turn-1/step-3")
        self.assertEqual(replay.env.next_payload_calls, 4)
        replay.env.commit_response(tool_response("new_action"))
        following = replay.env.next_payload()
        self.assertEqual(following["decision_key"], "turn-1/step-4")
        self.assertEqual(following["messages"], first["messages"] + [
            tool_response("new_action"), tool_observation("new_action")])

    def test_omitted_raw_history_is_absent_from_initial_and_continuation_model_view(self):
        replay = self.restore()
        expected = [{"role": "system", "content": "Official instructions"},
                    *self.row["q"]["raw_visible"]]
        self.assertEqual(replay.current_payload()["messages"], expected)
        replay.env.commit_response(tool_response("new_action"))
        following = replay.env.next_payload()
        self.assertEqual(following["messages"], expected + [
            tool_response("new_action"), tool_observation("new_action")])
        serialized = json.dumps(following["messages"])
        self.assertNotIn("hidden_action", serialized)
        self.assertNotIn("Original request", serialized)
        self.assertNotIn('"done"', serialized)
        self.assertEqual(replay.env.executed_actions,
                         ["hidden_action", "visible_action", "new_action"])

    def test_current_payload_is_a_copy_of_projected_pending_payload(self):
        replay = self.restore()
        payload = replay.current_payload()
        payload["messages"][0]["content"] = "mutated"
        payload["tools"].clear()
        self.assertEqual(replay.current_payload()["messages"][0]["content"],
                         "Official instructions")
        self.assertEqual(len(replay.current_payload()["tools"]), 3)

    def test_source_event_from_other_session_is_rejected(self):
        row = copy.deepcopy(self.row)
        row["q"]["raw_source_ids"][0] = "bfcl/other/attempt-0:m3"
        with self.assertRaisesRegex(PrefixReplayIntegrityError, "source event identity"):
            self.restore(row)

    def test_source_event_index_outside_reconstructed_history_is_rejected(self):
        row = copy.deepcopy(self.row)
        row["q"]["raw_source_ids"][0] = self.session_id + ":m99"
        with self.assertRaisesRegex(PrefixReplayIntegrityError, "outside reconstructed history"):
            self.restore(row)

    def test_wrong_source_event_selection_is_rejected(self):
        row = copy.deepcopy(self.row)
        row["q"]["raw_source_ids"][0] = self.session_id + ":m1"
        with self.assertRaisesRegex(PrefixReplayIntegrityError, "do not equal q.raw_visible"):
            self.restore(row)

    def test_changed_tool_observation_is_rejected_even_when_q_suffix_matches(self):
        row = copy.deepcopy(self.row)
        row["q"]["raw_visible"][1]["content"] = "fabricated result"
        row["q"]["last_action_observation"][1]["content"] = "fabricated result"
        with self.assertRaisesRegex(PrefixReplayIntegrityError, "tool observations do not equal"):
            self.restore(row)

    def test_changed_tool_call_id_is_rejected(self):
        row = copy.deepcopy(self.row)
        row["q"]["raw_visible"][1]["tool_call_id"] = "other-call"
        row["q"]["last_action_observation"][1]["tool_call_id"] = "other-call"
        with self.assertRaisesRegex(PrefixReplayIntegrityError, "tool observations do not equal"):
            self.restore(row)

    def test_first_target_draft_and_later_branches_do_not_enter_prefix(self):
        # Even an invalid later session must be irrelevant after first target.
        self.records.append({"session_id": "different-session", "decision_key": "bad"})
        self.write_trace()
        bound = bind_source_trace(self.row, self.source_root)
        self.assertEqual(bound["_source_trace"]["prior_decisions"], 3)
        self.assertEqual([record["decision_key"] for record in bound["_recorded_prefix_steps"]],
                         ["turn-0/step-0", "turn-0/step-1", "turn-0/step-2"])
        replay = restore_bfcl_prefix(bound, FakeBindings())
        serialized = json.dumps(replay.env.committed_responses)
        self.assertNotIn("target-draft", serialized)
        self.assertNotIn("branch-action", serialized)
        self.assertNotIn("branch-target", serialized)

    def test_uncommitted_source_decision_before_target_is_rejected(self):
        self.records[1]["status"] = "failed"
        self.write_trace()
        with self.assertRaisesRegex(PrefixReplayIntegrityError, "uncommitted decision"):
            bind_source_trace(self.row, self.source_root)


if __name__ == "__main__":
    unittest.main()

def test_official_checker_uses_bfcl_object_serializer():
    import json
    from types import SimpleNamespace
    from generality.prefix_replay import official_current_turn

    class Directory:
        pass

    directory = Directory()
    def serialize(value):
        assert value['details']['directory'] is directory
        return {'valid': False, 'details': {'directory': '<Directory: example>'}}
    replay = SimpleNamespace(
        bindings=SimpleNamespace(
            score_turn_prefix=lambda *args: {'valid': False, 'details': {'directory': directory}},
            make_json_serializable=serialize),
        env=SimpleNamespace(handler=None, all_model_response=[]),
        ground_truth=[], task={}, turn_index=1)
    outcome = official_current_turn(replay)
    assert outcome['turn_success'] is False
    assert outcome['status'] == 'known'
    assert json.loads(json.dumps(outcome)) == outcome
