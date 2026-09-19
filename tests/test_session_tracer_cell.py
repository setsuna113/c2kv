"""Controller regressions with an injected engine and risk predictor."""
from __future__ import annotations

import sys
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "controller_runtime"),
               str(ROOT / "controller_runtime" / "python")]

from generality import session_tracer_cell as driver
from generality import scheduler_npu as scheduler


def response(text, *, active_history=100, splice=True):
    splice_receipt = ({
        "enabled": True,
        "session_id": "session-current",
        "generation_prefix_tokens": 3,
        "retained_body_physical_tokens": 128,
        "freed_physical_page_tokens": 128,
        "full_history_reprefill_performed": False,
        "scope": "verified_generation_prefix_only",
    } if splice else None)
    return {"choices": [{"message": {"content": text},
                         "hidden_states": [[[0.1, 0.2]]],
                         "logprobs": {"content": [{"logprob": -0.2}]}}],
            "metadata": {"kv_memory_report": {
                "active_history_kv_tokens": active_history,
                "history_kv_backend": "physical_eviction",
                "persistent_session_generation_prefix_splice": splice_receipt,
            }}}


class SessionTracerProtocolTests(unittest.TestCase):
    def make_task(self, benchmark, texts, risk_score=0.1):
        risk = Mock()
        risk.artifact = {"feature_contract": {"prefill_contract": {}}}
        risk.predict_risk.return_value = SimpleNamespace(available=True, score=risk_score)
        with patch.object(driver, "C1RiskArtifact", return_value=risk):
            task = driver.SessionTracerTask(
                "http://127.0.0.1:1", "qwen", "test", benchmark,
                 {"backend": "h2o", "threshold": 0.5,
                 "budget_tokens": {"K": 768, "R": 1024, "B": 1792},
                 "page_size": 128,
                 "caps": {"max_completion_tokens": 128}}, tokenizer=None)
        task.engine = Mock()
        task.engine.session_id = "session-current"
        task.engine.generate.side_effect = [response(text) for text in texts]
        return task, risk

    def decide(self, task):
        return task.decide({"messages": [{"role": "system", "content": "Agent"},
                                         {"role": "user", "content": "Find a contact"}],
                            "c2kv_eval_context": {"task_id": "test", "user_turn": 0, "step": 0}})

    def test_appworld_code_is_active_without_fabricated_tool_calls(self):
        code = '```python\nfor name in names:\n    print(apis.contacts.search(name=name))\n```'
        task, risk = self.make_task("acon_appworld", [code])
        parsed, receipt = self.decide(task)
        context = risk.predict_risk.call_args.args[0]
        self.assertFalse(context["is_stop"])
        self.assertEqual(context["draft_tool_calls"], [])
        self.assertEqual(context["draft_text"], code)
        self.assertEqual(parsed.text, code)
        self.assertEqual(receipt["exact_recovery"]["status"], "not_triggered")

    def test_bfcl_text_remains_stop(self):
        task, risk = self.make_task("bfcl", ["Done."])
        self.decide(task)
        self.assertTrue(risk.predict_risk.call_args.args[0]["is_stop"])

    def test_bfcl_tool_call_remains_active(self):
        task, risk = self.make_task("bfcl", ['<tool_call>{"name":"lookup","arguments":{}}</tool_call>'])
        parsed, _ = self.decide(task)
        self.assertFalse(risk.predict_risk.call_args.args[0]["is_stop"])
        self.assertEqual(len(parsed.tool_calls), 1)

    def test_appworld_observation_is_available_to_retrieval_context(self):
        task, risk = self.make_task("acon_appworld", ["print(next_step)"])
        task.decide({
            "messages": [
                {"role": "system", "content": "Agent"},
                {"role": "user", "content": "Find a contact"},
                {"role": "assistant", "content": "print(old_step)"},
                {"role": "user", "content": "Execution result: contact_id=123"},
            ],
            "c2kv_eval_context": {"task_id": "test", "user_turn": 1, "step": 1},
        })
        context = risk.predict_risk.call_args.args[0]
        self.assertEqual(context["last_action_observation"][-1]["role"], "user")
        self.assertIn("contact_id=123", context["last_action_observation"][-1]["content"])

    def test_appworld_risk_can_recover_and_regenerate(self):
        task, _ = self.make_task("acon_appworld", ["print(old_id)", "print(recovered_id)"], 0.9)
        evidence = [{"role": "user", "content": "Archived contact: recovered_id=123"}]
        task._select_candidate = Mock(return_value=(object(), {}))
        task._admit = Mock(return_value=(evidence, {
            "status": "admitted", "packet_resident_tokens": 128,
            "history_target_tokens": 1664}))
        parsed, receipt = self.decide(task)
        self.assertEqual(receipt["exact_recovery"]["status"], "recover")
        self.assertEqual(parsed.text, "print(recovered_id)")
        self.assertEqual(task.engine.generate.call_count, 2)
        task.engine.reopen.assert_not_called()
        self.assertIn(evidence[0], task.engine.generate.call_args.args[0])
        self.assertEqual(task.engine.generate.call_args.kwargs["target_tokens"], 1664)
        self.assertTrue(task.engine.generate.call_args.kwargs["recovery_append"])

    def test_two_recoveries_keep_older_packet_at_its_original_boundary(self):
        task, _ = self.make_task("acon_appworld", ["draft1", "action1", "draft2", "action2"], 0.9)
        first = [{"role": "user", "content": "packet1"}]
        second = [{"role": "user", "content": "packet2"}]
        task._select_candidate = Mock(return_value=(object(), {}))
        receipt = {"status": "admitted", "packet_resident_tokens": 128,
                   "history_target_tokens": 1664}
        task._admit = Mock(side_effect=[(first, receipt), (second, receipt)])
        self.decide(task)
        messages = [{"role": "system", "content": "Agent"},
                    {"role": "user", "content": "Find a contact"},
                    {"role": "assistant", "content": "action1"},
                    {"role": "user", "content": "Execution result"}]
        task.decide({"messages": messages,
                     "c2kv_eval_context": {"task_id": "test", "user_turn": 1, "step": 1}})
        views = [call.args[0] for call in task.engine.generate.call_args_list]
        self.assertEqual(views[1][2], first[0])
        self.assertEqual(views[2][2], first[0])
        self.assertEqual(views[3][2], first[0])
        self.assertEqual(views[3][-1], second[0])
        self.assertEqual(views[2][3:], messages[2:])
        task.engine.reopen.assert_not_called()

    def test_admission_rounds_packet_pages_and_reserves_space_by_shrinking_history(self):
        task, _ = self.make_task("bfcl", [])
        task.last_kv_report = {
            "active_history_kv_tokens": 1550,
            "history_kv_backend": "physical_eviction",
        }
        with patch.object(driver, "render_units", return_value=[{"role": "user", "content": "packet"}]), \
                patch.object(driver, "native_ids", side_effect=[range(100), range(230)]):
            admitted, receipt = task._admit(object(), object(), [{"role": "user", "content": "goal"}])
        self.assertIsNotNone(admitted)
        self.assertEqual(receipt["packet_tokens"], 130)
        self.assertEqual(receipt["packet_resident_tokens"], 256)
        self.assertEqual(receipt["total_after_admission"], 1792)
        self.assertEqual(receipt["history_target_tokens"], 1536)

    def test_admission_does_not_treat_missing_measured_history_as_zero(self):
        task, _ = self.make_task("bfcl", [])
        task.last_kv_report = {}
        with patch.object(driver, "render_units", return_value=[{"role": "user", "content": "packet"}]), \
                patch.object(driver, "native_ids", side_effect=[range(100), range(120)]):
            admitted, receipt = task._admit(object(), object(), [{"role": "user", "content": "goal"}])
        self.assertIsNone(admitted)
        self.assertEqual(receipt["reason"], "missing_measured_history")

    def test_pyramid_budget_uses_reference_full_model_token_equivalent(self):
        task, _ = self.make_task("bfcl", [])
        task.engine.method = "pyramidkv"
        report = {
            "history_kv_backend": "reference_attention",
            "active_history_kv_tokens": 511,
            "reference_history_token_slots": 65408,
            "reference_history_resident_bytes": 123456,
            "canonical_history_tokens": 4000,
        }
        self.assertEqual(task._active_history_tokens(report), 511)
        self.assertIsNone(task._active_history_tokens({
            **report, "history_kv_backend": "physical_eviction"}))
        self.assertIsNone(task._active_history_tokens({
            **report, "reference_history_token_slots": None}))

    def test_regeneration_requires_resident_preserving_splice_receipt(self):
        task, _ = self.make_task("acon_appworld", [], 0.9)
        task.engine.generate.side_effect = [response("draft"), response("action", splice=False)]
        task._select_candidate = Mock(return_value=(object(), {}))
        task._admit = Mock(return_value=([{"role": "user", "content": "packet"}], {
            "status": "admitted", "packet_resident_tokens": 128,
            "history_target_tokens": 1664}))
        with self.assertRaisesRegex(RuntimeError, "resident-preserving"):
            self.decide(task)

    def test_regeneration_rejects_measured_history_plus_packet_above_b(self):
        task, _ = self.make_task("acon_appworld", [], 0.9)
        task.engine.generate.side_effect = [response("draft"), response("action", active_history=1700)]
        task._select_candidate = Mock(return_value=(object(), {}))
        task._admit = Mock(return_value=([{"role": "user", "content": "packet"}], {
            "status": "admitted", "packet_resident_tokens": 128,
            "history_target_tokens": 1664}))
        with self.assertRaisesRegex(RuntimeError, "common-budget"):
            self.decide(task)

    def test_engine_recovery_reuses_session_and_sends_explicit_wire_contract(self):
        engine = driver.EngineSession("http://unused", "qwen", "session-current", "h2o")
        engine._post = Mock(side_effect=["session-current", response("draft"), response("action"), {}])
        messages = [{"role": "system", "content": "Agent"}, {"role": "user", "content": "goal"}]
        engine.generate(messages, [], max_tokens=16, target_tokens=768, history_span=(1, 1))
        engine.generate([*messages, {"role": "user", "content": "packet"}], [],
                        max_tokens=16, target_tokens=1664, history_span=(1, 2),
                        recovery_append=True)
        engine.close()
        calls = engine._post.call_args_list
        self.assertEqual([call.args[0] for call in calls],
                         ["/open_session", "/v1/chat/completions", "/v1/chat/completions", "/close_session"])
        body = calls[2].args[1]
        self.assertEqual(body["session_params"]["id"], "session-current")
        self.assertEqual(body["c2kv_kv_memory_hint"]["persistent_history_session"]["recovery_append"],
                         {"enabled": True})
        self.assertEqual(calls[3].args[1]["session_id"], "session-current")

    def test_pyramid_request_selects_reference_attention_backend(self):
        engine = driver.EngineSession(
            "http://unused", "qwen", "session-current", "pyramidkv")
        engine._post = Mock(side_effect=["session-current", response("draft")])

        engine.generate(
            [{"role": "system", "content": "Agent"},
             {"role": "user", "content": "goal"}],
            [], max_tokens=16, target_tokens=768, history_span=(1, 2))

        body = engine._post.call_args_list[1].args[1]
        hint = body["c2kv_kv_memory_hint"]
        self.assertEqual(hint["history_kv_backend"], "reference_attention")
        self.assertEqual(hint["history_kv_eviction"]["method"], "pyramidkv")

    def test_history_span_keeps_trailing_tool_result_block_outside_eviction(self):
        messages = [
            {"role": "system", "content": "agent"},
            {"role": "user", "content": "goal"},
            {"role": "assistant", "content": "call"},
            {"role": "tool", "content": "part 1"},
            {"role": "tool", "content": "part 2"},
        ]
        self.assertEqual(driver._history_message_span(messages), (1, 3))

    def test_appworld_worker_env_is_constructed_without_bfcl_branch(self):
        env = driver.appworld_worker_env({
            "acon_dir": "/tmp/acon", "appworld_root": "/tmp/appworld"})
        self.assertEqual(env["APPWORLD_ROOT"], "/tmp/appworld")
        self.assertIn(str(Path("/tmp/acon") / "src"), env["PYTHONPATH"])
        self.assertNotIn("HTTP_PROXY", env)
        self.assertNotIn("HTTPS_PROXY", env)


if __name__ == "__main__":
    unittest.main()


def test_stale_batch_done_is_revalidated_and_full_manifest_stamped(tmp_path, monkeypatch):
    config = tmp_path / "config"
    config.mkdir()
    (config / "budgets_resolved.json").write_text(json.dumps({
        "working_points": {"K0": {"kv_token_equivalents": {"K": 8}}}}))
    monkeypatch.setattr(driver, "GENERATION_ROOT", tmp_path)
    cell_dir = tmp_path / "cell"
    cell_dir.mkdir()
    task_id = "task_1"
    cell = {"cell_id": "tracer-1", "cell_dir": str(cell_dir),
            "benchmark": "acon_appworld", "working_point": "K0",
            "threshold": 0.5, "task_ids": [task_id],
            "caps": {"task_timeout": 5}, "python_sgl": "python",
            "python_appworld": "python", "acon_dir": "acon",
            "appworld_root": "appworld"}
    manifest = cell_dir / "cell.json"
    manifest.write_text(json.dumps(cell))
    old_out = cell_dir / "batches" / "000_task_1"
    old_out.mkdir(parents=True)
    (old_out / "done.json").write_text('{"status":"completed"}')

    class Server:
        def serve_forever(self):
            pass

        def shutdown(self):
            pass

    monkeypatch.setattr(driver, "run_server", lambda *args: (Server(), {}))

    def scored_worker(command, **kwargs):
        out = Path(command[command.index("--out") + 1])
        out.mkdir(parents=True)
        (out / "official_summary.json").write_text(json.dumps({
            "schema": "a-event-native-appworld-run-v1", "status": "completed",
            "task_id": task_id, "n": 1, "semantic_score": 0.0}))
        return 0

    monkeypatch.setattr(driver.subprocess, "call", scored_worker)
    assert driver.main(["--cell", str(manifest)]) == 0
    assert list((cell_dir / "batches").glob("000_task_1.prior.*/done.json"))
    status = json.loads((cell_dir / "cell_status.json").read_text())
    assert status["status"] == "complete"
    assert status["n_completed"] == 1
    assert scheduler.cell_done(cell)
