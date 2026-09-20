"""Native C1 budget override contracts without loading model weights."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import history_budget
import run_c1
from benchmarks.memory_runtime.budget_guard import history_budget_receipt


def _value(command: list[str], flag: str) -> str:
    return command[command.index(flag) + 1]


class NativeHistoryBudgetTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.checkpoint = self.root / "checkpoint-1000"
        self.checkpoint.mkdir()
        self.config = {
            "num_hidden_layers": 36,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "history_memory_policy": {"kv_bytes_per_token": 147456},
        }
        self.config_path = self.checkpoint / "config.json"
        self.config_path.write_text(json.dumps(self.config) + "\n", encoding="utf-8")
        self.design = run_c1.current.load_config()
        self.design["checkpoint_selection"]["config_sha256"] = hashlib.sha256(
            self.config_path.read_bytes()).hexdigest()
        self.bfcl = self.root / "bfcl"
        (self.bfcl / "bfcl_eval").mkdir(parents=True)

    def args(self, *, tokens=None, method="c2kv_only", candidate=None,
             preview=False, output=None):
        command = [
            "--method", method,
            "--checkpoint", str(self.checkpoint),
            "--sglang-backend-url", "http://127.0.0.1:38800",
            "--benchmark-dir", str(self.bfcl),
            "--out", str(output or self.root / "run"),
        ]
        if tokens is not None:
            command += ["--history-budget-tokens", str(tokens)]
        if candidate:
            command += ["--candidate-algorithm", candidate]
        if preview:
            command += ["--preview"]
        return run_c1.build_parser().parse_args(command)

    def profile(self, args):
        with (mock.patch.object(run_c1.current, "load_config",
                                side_effect=lambda: copy.deepcopy(self.design)),
              mock.patch.object(run_c1, "bind_risk_artifact",
                                side_effect=lambda artifact, checkpoint: (artifact, {"status": "fixture"}))):
            return run_c1.build_profile(args)

    def command(self, args, controller):
        args.out.mkdir(parents=True, exist_ok=True)
        controller_path = args.out / "controller.json"
        run_c1.save(controller_path, controller)
        with mock.patch.object(run_c1.current, "load_config",
                               side_effect=lambda: copy.deepcopy(self.design)):
            return run_c1.commands_for_task(args, "multi_turn_base_0", controller_path)[0]

    def test_default_profile_and_server_policy_are_unchanged(self):
        args = self.args()
        controller, profile = self.profile(args)
        self.assertNotIn("native_history_budget", profile)
        server = self.command(args, controller)
        self.assertEqual(_value(server, "--eval-policy"), str(
            (run_c1.RUNTIME / "configs/eval_policy.json").resolve()))
        self.assertFalse((args.out / history_budget.POLICY_FILENAME).exists())

    def test_non_bfcl_budget_rejected_at_cli_and_direct_profile(self):
        args = self.args(tokens=768, method="proposed", candidate="goal_pending")
        args.benchmark = "tau2"
        with self.assertRaisesRegex(ValueError, "BFCL only"):
            run_c1.validate_args(args)
        with self.assertRaisesRegex(ValueError, "BFCL only"):
            run_c1.build_profile(args)

    def test_768_tokens_are_byte_equivalent_to_frozen_policy(self):
        args = self.args(tokens=768)
        controller, profile = self.profile(args)
        receipt = profile["native_history_budget"]
        base = json.loads((run_c1.RUNTIME / "configs/eval_policy.json").read_text(encoding="utf-8"))
        self.assertEqual(receipt["kv_bytes_per_token"], 147456)
        self.assertEqual(receipt["history_budget_bytes"], base["policy"]["history_budget_bytes"])
        self.assertEqual(receipt["workspace_budget_bytes"], base["policy"]["workspace_budget_bytes"])
        self.assertEqual(receipt["eval_policy"]["policy"]["lease_decisions"],
                         base["policy"]["lease_decisions"])
        self.assertEqual(receipt["eval_policy"]["policy"]["max_retrieved_events"],
                         base["policy"]["max_retrieved_events"])
        server = self.command(args, controller)
        target = args.out / history_budget.POLICY_FILENAME
        self.assertEqual(_value(server, "--eval-policy"), str(target.resolve()))
        self.assertEqual(json.loads(target.read_text(encoding="utf-8")), receipt["eval_policy"])
        self.assertNotEqual(receipt["eval_policy"]["policy_id"], base["policy_id"])

    def test_256_and_2048_propagate_through_off_and_pending_server_argv(self):
        for method, candidate in (("c2kv_only", None), ("proposed", "goal_pending")):
            for tokens in (256, 2048):
                with self.subTest(method=method, candidate=candidate, tokens=tokens):
                    args = self.args(tokens=tokens, method=method, candidate=candidate,
                                     output=self.root / f"{method}_{tokens}")
                    controller, profile = self.profile(args)
                    receipt = profile["native_history_budget"]
                    server = self.command(args, controller)
                    policy = json.loads(Path(_value(server, "--eval-policy")).read_text(encoding="utf-8"))
                    expected = tokens * receipt["kv_bytes_per_token"]
                    self.assertEqual(policy["policy"]["history_budget_bytes"], expected)
                    self.assertEqual(policy["policy"]["workspace_budget_bytes"], expected)
                    self.assertEqual(receipt["history_budget_bytes"], expected)
                    self.assertEqual(_value(server, "--ratio"), "8")
                    if candidate:
                        self.assertEqual(controller["candidate_algorithm"]["variant"], candidate)
                    self.assertEqual(receipt["override_eval_policy_sha256"], hashlib.sha256(
                        Path(_value(server, "--eval-policy")).read_bytes()).hexdigest())

    def test_draft_and_recovery_guard_use_the_same_resolved_byte_cap(self):
        for tokens, expected_status in ((256, "rejected"), (2048, "passed")):
            args = self.args(tokens=tokens, output=self.root / f"guard_{tokens}")
            controller_config, _ = self.profile(args)
            server = self.command(args, controller_config)
            policy = json.loads(Path(_value(server, "--eval-policy")).read_text(encoding="utf-8"))["policy"]
            controller = SimpleNamespace(
                kv_bytes_per_token=147456,
                policy_config=SimpleNamespace(
                    history_budget_bytes=policy["history_budget_bytes"],
                    workspace_budget_bytes=policy["workspace_budget_bytes"],
                ),
            )
            for phase, resident in (("draft", 1024), ("recovery", 1280)):
                with self.subTest(tokens=tokens, phase=phase):
                    memory = SimpleNamespace(costs=lambda ratio: {
                        "resident_kv_tokens": resident, "gist_tokens": 64,
                    })
                    metadata = {
                        "common_raw_prompt_tokens": 0,
                        "actual_history_bytes": resident * controller.kv_bytes_per_token,
                    }
                    receipt = history_budget_receipt(memory, metadata, controller,
                                                     ratio=8, phase=phase)
                    self.assertEqual(receipt["status"], expected_status)
                    self.assertEqual(receipt["history_budget_bytes"], tokens * 147456)
                    self.assertEqual("active_history_exceeds_budget" in receipt["errors"],
                                     expected_status == "rejected")

    def test_preview_resolves_policy_without_creating_output_or_loading_model(self):
        args = self.args(tokens=256, preview=True)
        controller, profile = self.profile(args)
        target = args.out / history_budget.POLICY_FILENAME
        with mock.patch.object(history_budget, "materialize",
                               side_effect=AssertionError("preview wrote policy")):
            with mock.patch.object(run_c1.current, "load_config",
                                   side_effect=lambda: copy.deepcopy(self.design)):
                server, _ = run_c1.commands_for_task(args, "multi_turn_base_0", args.out / "controller.json")
        self.assertEqual(_value(server, "--eval-policy"), str(target.resolve()))
        self.assertEqual(profile["native_history_budget"]["requested_tokens"], 256)
        self.assertFalse(args.out.exists())

    def test_invalid_values_and_bare_semantics_are_rejected(self):
        for value in (0, -1):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    self.profile(self.args(tokens=value))
        with self.assertRaises(SystemExit):
            self.args(tokens="not-an-int")
        with self.assertRaisesRegex(ValueError, "c2kv_native_r4"):
            self.profile(self.args(tokens=768, method="c2kv_native"))
        args = self.args(tokens=256)
        self.config["history_memory_policy"]["kv_bytes_per_token"] = 1
        self.config_path.write_text(json.dumps(self.config), encoding="utf-8")
        self.design["checkpoint_selection"]["config_sha256"] = hashlib.sha256(
            self.config_path.read_bytes()).hexdigest()
        with self.assertRaisesRegex(ValueError, "KV byte geometry"):
            self.profile(args)


if __name__ == "__main__":
    unittest.main()
