"""Calibration client regressions against the SGLang HTTP route shapes."""
from __future__ import annotations

import copy
import json
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "generality"), str(ROOT / "controller_runtime")]

from generality import calibrate  # noqa: E402


class SGLangHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.requests.append((self.path, body))
        if self.path == "/open_session":
            session_id = body["session_id"]
            if session_id in self.server.sessions:
                self.reply({"error": "session already open"}, status=400)
                return
            self.server.sessions.add(session_id)
            self.reply(self.server.open_echo or session_id)
        elif self.path == "/close_session":
            self.server.sessions.discard(body["session_id"])
            self.send_response(200)
            self.end_headers()
        elif self.path == "/v1/chat/completions":
            messages = body.get("messages") or []
            hint = body.get("c2kv_kv_memory_hint") or {}
            count = (hint.get("history_kv_eviction") or {}).get("history_message_count")
            if (isinstance(count, int) and 0 < count < len(messages)
                    and messages[count - 1].get("role") == "tool"
                    and messages[count].get("role") == "tool"):
                self.reply({"error": "completed history splits a tool-response block"}, status=400)
                return
            self.reply(self.server.chat_response)
        else:
            self.reply({"error": "unknown route"}, status=404)

    def reply(self, value, status=200):
        raw = json.dumps(value).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


class CalibrationClientTests(unittest.TestCase):
    def setUp(self):
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), SGLangHandler)
        self.server.requests = []
        self.server.sessions = set()
        self.server.open_echo = None
        self.server.chat_response = {
            "choices": [{"message": {"role": "assistant", "content": "Done."}}]}
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def client(self):
        return calibrate.PersistentSGLangClient(
            self.base + "/v1", "qwen", "bfcl/task/replay-state", "h2o", 768)

    def payload(self):
        return {"decision_key": "turn-1/step-0", "messages": [
            {"role": "system", "content": "Agent"},
            {"role": "user", "content": "Previous request"},
            {"role": "assistant", "content": "Previous answer"},
            {"role": "user", "content": "Current request"}]}

    def test_open_generate_close_uses_string_and_empty_response_contract(self):
        client = self.client()
        try:
            for _ in range(2):
                self.assertEqual(client.generate(self.payload(), 128),
                                 self.server.chat_response)
            self.assertTrue(client.opened)
            self.assertEqual(self.server.sessions, {client.session_id})
        finally:
            client.close()
        client.close()
        self.assertFalse(client.opened)
        self.assertEqual(self.server.sessions, set())
        self.assertEqual([path for path, _ in self.server.requests], [
            "/open_session", "/v1/chat/completions", "/v1/chat/completions",
            "/close_session"])
        opened = self.server.requests[0][1]
        self.assertTrue(opened["streaming"])
        self.assertEqual(opened["capacity_of_str_len"], 0)
        for _, body in self.server.requests[1:3]:
            self.assertEqual(body["session_params"]["id"], client.session_id)
            hint = body["c2kv_kv_memory_hint"]
            self.assertTrue(hint["persistent_history_session"]["enabled"])
            self.assertEqual(hint["history_kv_eviction"]["method"], "h2o")
            self.assertEqual(hint["history_kv_eviction"]["target_tokens"], 768)
        self.assertEqual(self.server.requests[-1][1]["session_id"], client.session_id)

    def test_repeated_source_state_uses_distinct_engine_sessions(self):
        first, second = self.client(), self.client()
        self.assertNotEqual(first.session_id, second.session_id)
        try:
            first.open()
            second.open()
            self.assertEqual(self.server.sessions, {first.session_id, second.session_id})
        finally:
            first.close()
            second.close()

    def test_session_echo_mismatch_is_rejected(self):
        self.server.open_echo = "different-session"
        client = self.client()
        with self.assertRaisesRegex(RuntimeError, "session id echo mismatch"):
            client.open()
        self.assertFalse(client.opened)

    def test_chat_response_still_requires_an_object(self):
        self.server.chat_response = "unexpected string"
        client = self.client()
        try:
            with self.assertRaisesRegex(RuntimeError, "returned a non-object"):
                client.generate(self.payload(), 128)
        finally:
            client.close()

    def test_event_native_response_still_requires_an_object(self):
        self.server.chat_response = "unexpected string"
        client = calibrate.EventNativeControllerClient(self.base, "qwen", "bfcl")
        with self.assertRaisesRegex(RuntimeError, "returned a non-object"):
            client.generate(self.payload(), 128)

    def test_http_error_includes_route_status_and_server_detail(self):
        with self.assertRaisesRegex(RuntimeError, "POST .*missing -> 404: .*unknown route"):
            calibrate._post(self.base + "/missing", {})

    def test_multi_tool_continuation_keeps_trailing_observation_block_complete(self):
        payload = self.payload()
        payload["messages"].extend([
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "first-call", "function": {"name": "lookup", "arguments": "{}"}},
                {"id": "second-call", "function": {"name": "lookup", "arguments": "{}"}}]},
            {"role": "tool", "content": "first result", "tool_call_id": "first-call"},
            {"role": "tool", "content": "second result", "tool_call_id": "second-call"}])
        client = self.client()
        try:
            client.generate(payload, 128)
        finally:
            client.close()
        body = self.server.requests[1][1]
        count = body["c2kv_kv_memory_hint"]["history_kv_eviction"]["history_message_count"]
        self.assertEqual(count, 5)
        self.assertEqual([message["role"] for message in body["messages"][count:]],
                         ["tool", "tool"])


class CalibrationHistoryBoundaryTests(unittest.TestCase):
    def test_single_observation_and_complete_previous_blocks_keep_existing_boundary(self):
        prefix = [{"role": "system"}, {"role": "user"}, {"role": "assistant"}]
        self.assertEqual(calibrate._history_message_span(prefix + [{"role": "tool"}]), (1, 3))
        self.assertEqual(calibrate._history_message_span(prefix + [
            {"role": "tool"}, {"role": "tool"}, {"role": "user"}]), (1, 5))

    def test_replay_receipt_verification_uses_same_complete_block_boundary(self):
        initial = {"decision_key": "turn-0/step-0", "messages": [
            {"role": "system", "content": "Agent"},
            {"role": "user", "content": "Do work"}]}
        first_calls = [{"id": name, "type": "function", "function": {
            "name": "lookup", "arguments": "{}"}} for name in ("first", "second")]
        first_message = {"role": "assistant", "content": None, "tool_calls": first_calls}
        following = {"decision_key": "turn-0/step-1", "messages": initial["messages"] + [
            first_message,
            {"role": "tool", "content": "first result", "tool_call_id": "first"},
            {"role": "tool", "content": "second result", "tool_call_id": "second"}]}
        responses = [{"choices": [{"message": first_message}]},
                     {"choices": [{"message": {"role": "assistant", "content": "Done."}}]}]
        env = SimpleNamespace(finished=False, turn_index=0, next_payload=Mock(return_value=following),
                              close=Mock())
        commits = []
        def commit(response):
            commits.append(response)
            env.finished = len(commits) == 2
        env.commit_response = commit
        replay = SimpleNamespace(env=env, previous_turn_valid=True,
                                 current_payload=lambda: copy.deepcopy(initial),
                                 replayed_assistant_messages=0, replayed_tool_messages=0)
        client = Mock(method="h2o")
        client.generate.side_effect = responses
        row = {"state_id": "state", "task_id": "task", "decision_key": "turn-0/step-0",
               "q": {"session_id": "bfcl/task/attempt-0", "goal": "Do work",
                     "raw_source_ids": ["bfcl/task/attempt-0:m0"],
                     "raw_visible": [initial["messages"][-1]], "last_action_observation": []},
               "_source_trace": {"path": "fixture"}}
        args = SimpleNamespace(backend="h2o", wp="K0", target_tokens=768,
                               source_root="fixture", labels="fixture/labels.json",
                               engine_url="http://127.0.0.1:1", model="qwen", max_completion_tokens=128)
        with patch.object(calibrate, "bind_source_trace", return_value=row), \
                patch.object(calibrate, "restore_bfcl_prefix", return_value=replay), \
                patch.object(calibrate, "PersistentSGLangClient", return_value=client), \
                patch.object(calibrate, "_risk_score", return_value={"available": False}), \
                patch.object(calibrate, "_verify_history_backend", return_value={}) as verify, \
                patch.object(calibrate, "official_current_turn", return_value={
                    "turn_success": True, "status": "known", "checker": {}}):
            calibrate.replay_one(row, args, None, None)
        self.assertEqual([call.args[2] for call in verify.call_args_list], [0, 2])
        self.assertEqual([call.kwargs["continuation"] for call in verify.call_args_list], [False, True])
        self.assertEqual(len(commits[0]["tool_calls"]), 2)
        env.close.assert_called_once()


class CalibrationDraftTests(unittest.TestCase):
    def setUp(self):
        self.artifact = Mock()
        self.artifact.predict_risk.return_value = SimpleNamespace(available=True, score=0.7)
        self.row = {"q": {"goal": "Make a backup", "prefill_contract": {"layer": 16},
                          "last_action_observation": []}}

    def response(self, message):
        return {"id": "draft-response-id", "choices": [{
            "message": message, "hidden_states": [[[0.1, 0.2], [0.3, 0.4]]],
            "logprobs": {"content": [{"logprob": -0.2}, {"logprob": None},
                                     {"logprob": -0.4}]}}],
            "metadata": {"hidden_states": ["wrong-location"]}}

    def context_for(self, response):
        receipt = calibrate._risk_score(response, self.row, self.artifact)
        return receipt, self.artifact.predict_risk.call_args.args[0]

    def test_structured_calls_with_null_content_are_actions_for_risk_and_environment(self):
        call = {"id": "server-call-id", "type": "function", "function": {
            "name": "mkdir", "arguments": '{"dir_name":"backup_tests"}'}}
        response = self.response({"role": "assistant", "content": None, "tool_calls": [call]})
        receipt, context = self.context_for(response)
        executed = calibrate._response_for_environment(response)
        self.assertFalse(context["is_stop"])
        self.assertTrue(context["parse_ok"])
        self.assertEqual(receipt["parse_status"], "tool_calls")
        self.assertEqual(receipt["draft_protocol"], "structured_tool_calls")
        self.assertEqual(context["draft_tool_calls"], [call["function"]])
        self.assertEqual(executed["tool_calls"], [call])

    def test_structured_object_arguments_are_serialized_without_mutating_response(self):
        message = {"role": "assistant", "content": "Planning", "tool_calls": [{
            "id": "call-id", "type": "function", "function": {
                "name": "mkdir", "arguments": {"dir_name": "backup_tests"}}}]}
        response = self.response(message)
        before = json.dumps(response, sort_keys=True)
        _, context = self.context_for(response)
        executed = calibrate._response_for_environment(response)
        self.assertFalse(context["is_stop"])
        self.assertEqual(json.loads(executed["tool_calls"][0]["function"]["arguments"]),
                         {"dir_name": "backup_tests"})
        self.assertEqual(json.dumps(response, sort_keys=True), before)

    def test_native_tagged_call_is_executed_and_detector_receives_same_action(self):
        text = ('I will create the backup directory.\n<tool_call>\n'
                '{"name": "mkdir", "arguments": {"dir_name": "backup_tests"}}\n'
                '</tool_call>')
        response = self.response({"role": "assistant", "content": text, "tool_calls": None})
        receipt, context = self.context_for(response)
        executed = calibrate._response_for_environment(response)
        self.assertFalse(context["is_stop"])
        self.assertTrue(context["parse_ok"])
        self.assertEqual(receipt["draft_protocol"], "native_text")
        self.assertEqual(executed["content"], "I will create the backup directory.")
        self.assertEqual(executed["tool_calls"][0]["id"], "draft-response-id_0")
        self.assertEqual(context["draft_tool_calls"],
                         [call["function"] for call in executed["tool_calls"]])
        self.assertEqual(context["draft_text"], text)

    def test_hidden_and_logprob_features_come_from_choice_prompt_last_position(self):
        receipt, context = self.context_for(self.response(
            {"role": "assistant", "content": "Done."}))
        self.assertEqual(context["prefill_hidden"], [0.3, 0.4])
        self.assertEqual(context["draft_logprobs"], [-0.2, -0.4])
        self.assertEqual(context["prefill_contract"], self.row["q"]["prefill_contract"])
        self.assertEqual(receipt["draft_logprobs_count"], 2)
        self.assertTrue(receipt["feature_hidden_available"])

    def test_plain_text_remains_a_stop_for_both_risk_and_environment(self):
        response = self.response({"role": "assistant", "content": "Done."})
        receipt, context = self.context_for(response)
        executed = calibrate._response_for_environment(response)
        self.assertTrue(context["is_stop"])
        self.assertTrue(context["parse_ok"])
        self.assertEqual(receipt["parse_status"], "text")
        self.assertNotIn("tool_calls", executed)
        self.assertEqual(executed["content"], "Done.")

    def test_malformed_native_call_never_admits_a_partial_action(self):
        text = ('<tool_call>{"name":"mkdir","arguments":{}}</tool_call>'
                '<tool_call>{"name":"copy","arguments":')
        response = self.response({"role": "assistant", "content": text})
        receipt, context = self.context_for(response)
        executed = calibrate._response_for_environment(response)
        self.assertFalse(context["parse_ok"])
        self.assertEqual(receipt["parse_status"], "malformed")
        self.assertEqual(context["draft_tool_calls"], [])
        self.assertNotIn("tool_calls", executed)

    def test_malformed_structured_arguments_are_not_marked_parse_ok(self):
        response = self.response({"role": "assistant", "content": None, "tool_calls": [{
            "id": "call-id", "function": {"name": "mkdir", "arguments": "[1,2]"}}]})
        receipt, context = self.context_for(response)
        executed = calibrate._response_for_environment(response)
        self.assertFalse(context["parse_ok"])
        self.assertEqual(receipt["parse_status"], "malformed")
        self.assertNotIn("tool_calls", executed)


class CalibrationRiskReceiptTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "steps.jsonl"
        self.task_id = "multi_turn_base_3"
        self.state_id = "source-state"
        self.decision_key = "turn-1/step-3"
        self.session_id = f"bfcl/{self.task_id}/attempt-0"

    def record(self, score=0.2, **fields):
        return {"task_id": self.task_id, "decision_key": self.decision_key,
                "risk": {"available": True, "score": score}, **fields}

    def client(self, state_id=None):
        return calibrate.EventNativeControllerClient(
            "http://127.0.0.1:1", "qwen", "bfcl", self.path, state_id, self.task_id)

    def write(self, *records):
        self.path.write_text("".join(json.dumps(record) + "\n" for record in records),
                             encoding="utf-8")

    def test_state_bound_session_cannot_read_other_state_of_same_task(self):
        self.write(self.record(session_id=self.session_id + "/" + self.state_id),
                   self.record(0.9, session_id=self.session_id + "/other-state"))
        receipt = self.client(self.state_id).risk_for(self.decision_key)
        self.assertTrue(receipt["available"])
        self.assertEqual(receipt["score"], 0.2)

    def test_explicit_state_mismatch_is_rejected_even_with_matching_session(self):
        self.write(self.record(calibration_state_id="other-state",
                               session_id=self.session_id + "/" + self.state_id))
        self.assertIsNone(self.client(self.state_id).risk_for(self.decision_key))

    def test_explicit_state_and_task_can_bind_receipt_without_session_field(self):
        self.write(self.record(calibration_state_id=self.state_id))
        receipt = self.client(self.state_id).risk_for(self.decision_key)
        self.assertEqual(receipt["score"], 0.2)

    def test_legacy_receipt_without_state_cannot_bind_a_state_specific_request(self):
        self.write(self.record(session_id=self.session_id))
        self.assertIsNone(self.client(self.state_id).risk_for(self.decision_key))

    def test_single_legacy_task_and_decision_receipt_is_supported(self):
        self.write(self.record())
        receipt = self.client().risk_for(self.decision_key)
        self.assertEqual(receipt["score"], 0.2)

    def test_multiple_legacy_receipts_are_explicitly_unavailable(self):
        self.write(self.record(), self.record(0.9))
        receipt = self.client().risk_for(self.decision_key)
        self.assertFalse(receipt["available"])
        self.assertIsNone(receipt["score"])
        self.assertEqual(receipt["reason"], "ambiguous_risk_receipts")
        self.assertEqual(receipt["matching_receipts"], 2)

    def test_session_identity_supports_runner_receipt_without_task_field(self):
        record = self.record(session_id=self.session_id + "/" + self.state_id)
        record.pop("task_id")
        self.write(record)
        receipt = self.client(self.state_id).risk_for(self.decision_key)
        self.assertEqual(receipt["score"], 0.2)

    def test_explicit_task_mismatch_is_rejected_even_with_matching_session(self):
        self.write(self.record(task_id="other-task",
                               session_id=self.session_id + "/" + self.state_id))
        self.assertIsNone(self.client(self.state_id).risk_for(self.decision_key))

    def test_exact_recovery_selection_fallback_keeps_state_binding(self):
        record = self.record(session_id=self.session_id + "/" + self.state_id)
        record.pop("risk")
        record["exact_recovery"] = {"selection": {"score": 0.4}}
        self.write(record)
        receipt = self.client(self.state_id).risk_for(self.decision_key)
        self.assertEqual(receipt["score"], 0.4)
        self.assertEqual(receipt["source"], "controller_steps_jsonl.exact_recovery.selection")


if __name__ == "__main__":
    unittest.main()
