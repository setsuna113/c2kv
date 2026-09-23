import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from benchmarks.paper import runner
from benchmarks.paper.report import write_comparison
from benchmarks.paper.task_subsets import with_task_subsets


class TaskSubsetsTest(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(runner.DEFAULT_CONFIG.read_text())
        self.config["history_kv_budget_tokens"] = 768
        self.subsets = {
            "bfcl_long_context__agentkv_b768": ["multi_turn_long_context_0", "multi_turn_long_context_3"],
            "appworld__history_kv_pyramidkv_persistent_b768": ["3d9a636_1"],
        }

    def test_prepare_filters_cells_and_forwards_exact_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = dict(self.config, task_subsets=self.subsets)
            plan, _ = runner.prepare(config, root, root / "engine")
            self.assertEqual({row["cell_id"] for row in plan}, set(self.subsets))
            self.assertEqual(json.loads((root / "commands.json").read_text()), plan)
            self.assertEqual(json.loads((root / "config.resolved.json").read_text())["task_subsets"], self.subsets)
            for row in plan:
                ids = self.subsets[row["cell_id"]]
                flag = "--run-ids" if row["adapter"] == "bfcl" else "--task-ids"
                self.assertEqual(row["command"][row["command"].index(flag) + 1], ",".join(ids))
                self.assertEqual(row["task_ids"], ids)
                self.assertEqual(row["expected_subset_n"], len(ids))
                self.assertFalse(row["whole_cell_score"])
            self.assertIn("result_scope", (root / "matrix.csv").read_text())
            for row in runner.cells(self.config):
                self.assertNotIn("task_ids", row)

    def test_cli_and_file_forms(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "subsets.json"
            path.write_text(json.dumps(self.subsets))
            from_file = with_task_subsets(self.config, path=path)
            from_cli = with_task_subsets(self.config, [f"{cell}={','.join(ids)}" for cell, ids in self.subsets.items()])
            self.assertEqual(from_file, from_cli)
            with mock.patch("builtins.print") as printed:
                runner.main(["prepare", "--output", str(root / "prepared"),
                             "--history-kv-budget-tokens", "768", "--task-subset-file", str(path)])
            self.assertEqual(json.loads(printed.call_args.args[0])["replay_cells"], 0)

    def test_invalid_selections(self):
        cell = "bfcl_long_context__agentkv_b768"
        for subsets in ({}, {cell: []}, {cell: [""]}, {cell: ["x", "x"]},
                        {cell: [" x"]}, {cell: "x"}, {cell: [1]},
                        {cell: ["x,y"]}, {"unknown": ["x"]},
                        {"bfcl_base__c2kv_native_r4": ["x"]},
                        {"acebench_agent__agentkv": ["x"]}):
            with self.subTest(subsets=subsets), self.assertRaises(ValueError):
                runner.cells(dict(self.config, task_subsets=subsets))
        for values in ([cell], [cell + "=x", cell + "=y"]):
            with self.assertRaises(ValueError):
                with_task_subsets(self.config, values)

    def test_subset_scope_cannot_replace_existing_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runner.prepare(self.config, root, root / "engine")
            original = (root / "commands.json").read_bytes()
            with self.assertRaisesRegex(RuntimeError, "task subset scope changed"):
                runner.prepare(dict(self.config, task_subsets=self.subsets), root, root / "engine")
            self.assertEqual((root / "commands.json").read_bytes(), original)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = dict(self.config, task_subsets=self.subsets)
            runner.prepare(config, root, root / "engine")
            altered = dict(self.subsets, bfcl_long_context__agentkv_b768=["multi_turn_long_context_1"])
            with self.assertRaisesRegex(RuntimeError, "task subset scope changed"):
                runner.prepare(dict(config, task_subsets=altered), root, root / "engine")

    def test_no_replay_or_unknown_selection_can_start_a_server(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = dict(self.config, task_subsets=self.subsets)
            plan, profile = runner.prepare(config, root, root / "engine")
            with mock.patch.object(runner.subprocess, "Popen") as start:
                for stages, selected in ((["closed_loop", "common_prefix"], set()),
                                         (["closed_loop"], {"unknown"})):
                    with self.assertRaises(ValueError):
                        runner.execute(config, plan, root, root / "engine", stages, selected)
                with self.assertRaises(SystemExit):
                    runner.main(["run", "--history-kv-budget-tokens", "768",
                                 "--task-subset", "bfcl_long_context__agentkv_b768=x"])
                start.assert_not_called()
            with self.assertRaises(ValueError):
                runner.run_command(config, plan[0], root, profile, "common_prefix")

    def test_completion_and_summary_retain_scope_and_reject_incomplete_scores(self):
        for n_scored in (2, 1):
            with self.subTest(n_scored=n_scored), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                config = dict(self.config, task_subsets={"bfcl_long_context__agentkv_b768": self.subsets["bfcl_long_context__agentkv_b768"]})
                plan, _ = runner.prepare(config, root, root / "engine")
                cell = plan[0]
                directory = root / "closed_loop" / cell["cell_id"]

                def run(command, **kwargs):
                    self.assertIn("--run-ids", command)
                    (directory / "summary_agentkv.json").write_text(json.dumps({"n_scored": n_scored, "semantic_score": 0.5}))

                with mock.patch.object(runner.subprocess, "Popen", return_value=mock.Mock(pid=123)), \
                        mock.patch.object(runner, "run_owned", side_effect=run), \
                        mock.patch.object(runner, "wait_server"), \
                        mock.patch.object(runner, "cleanup_cell_processes"), \
                        mock.patch("socket.socket") as socket_type:
                    socket_type.return_value.__enter__.return_value.connect_ex.return_value = 1
                    if n_scored == 1:
                        with self.assertRaisesRegex(RuntimeError, "Repair subset scored"):
                            runner.execute(config, plan, root, root / "engine", ["closed_loop"], set())
                        self.assertFalse((directory / "complete.json").exists())
                        continue
                    runner.execute(config, plan, root, root / "engine", ["closed_loop"], set())
                for name in ("task_subset.json", "complete.json", "summary_agentkv.json"):
                    receipt = json.loads((directory / name).read_text())
                    self.assertEqual(receipt["task_ids"], cell["task_ids"])
                    self.assertFalse(receipt["whole_cell_score"])
                    self.assertEqual(receipt["result_scope"], "repair_subset")
                started = json.loads((directory / "started.json").read_text())
                self.assertEqual(started["cell"]["task_ids"], cell["task_ids"])

    def test_table_aggregation_and_direct_report_exclude_subset_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = dict(self.config, task_subsets=self.subsets)
            plan, _ = runner.prepare(config, root, root / "engine")
            for cell in plan:
                directory = root / "closed_loop" / cell["cell_id"]
                directory.mkdir(parents=True)
                for name in ("measurement_summary.json", "complete.json", "task_subset.json"):
                    (directory / name).write_text("{}")
                (directory / f"summary_{cell['arm']}.json").write_text('{"semantic_score": 1.0, "n": 2}')
            stale_plan = [cell for cell in runner.cells(self.config) if cell["cell_id"] in self.subsets]
            for view in (plan, stale_plan):
                with mock.patch.object(runner.subprocess, "run") as run:
                    with self.assertRaisesRegex(RuntimeError, "not whole-cell scores"):
                        runner.aggregate_results(config, view, root, ["closed_loop"], set())
                    run.assert_not_called()
                self.assertEqual(write_comparison(root, view), [])


if __name__ == "__main__":
    unittest.main()
