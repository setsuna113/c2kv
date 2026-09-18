"""Controller regressions with an injected engine and risk predictor."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "controller_runtime"),
               str(ROOT / "controller_runtime" / "python")]

from generality import session_tracer_cell as driver


def response(text):
    return {"choices": [{"message": {"content": text},
                         "hidden_states": [[[0.1, 0.2]]],
                         "logprobs": {"content": [{"logprob": -0.2}]}}]}


class SessionTracerProtocolTests(unittest.TestCase):
    def make_task(self, benchmark, texts, risk_score=0.1):
        risk = Mock()
        risk.artifact = {"feature_contract": {"prefill_contract": {}}}
        risk.predict_risk.return_value = SimpleNamespace(available=True, score=risk_score)
        with patch.object(driver, "C1RiskArtifact", return_value=risk):
            task = driver.SessionTracerTask(
                "http://127.0.0.1:1", "qwen", "test", benchmark,
                {"backend": "h2o", "threshold": 0.5,
                 "budget_tokens": {"K": 768, "B": 1792},
                 "caps": {"max_completion_tokens": 128}}, tokenizer=None)
        task.engine = Mock()
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

    def test_appworld_risk_can_recover_and_regenerate(self):
        task, _ = self.make_task("acon_appworld", ["print(old_id)", "print(recovered_id)"], 0.9)
        evidence = [{"role": "user", "content": "Archived contact: recovered_id=123"}]
        task._select_candidate = Mock(return_value=(object(), {}))
        task._admit = Mock(return_value=(evidence, {"status": "admitted"}))
        parsed, receipt = self.decide(task)
        self.assertEqual(receipt["exact_recovery"]["status"], "recover")
        self.assertEqual(parsed.text, "print(recovered_id)")
        self.assertEqual(task.engine.generate.call_count, 2)
        task.engine.reopen.assert_called_once()
        self.assertIn(evidence[0], task.engine.generate.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
