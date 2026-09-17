import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from benchmarks.paper import runner
from benchmarks.paper.runner import (
    DEFAULT_CONFIG, aggregate_results, cells, cleanup_cell_processes, execute,
    prepare, run_command, server_command,
)
from benchmarks.paper.report import write_comparison


class PaperMatrixTest(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(DEFAULT_CONFIG.read_text())

    def test_only_requested_methods_and_ratio(self):
        rows = cells(self.config)
        self.assertEqual(len(rows), 24)
        self.assertEqual(sum(row["group"] == "main" for row in rows), 18)
        self.assertEqual({row["ratio"] for row in rows if row["method"] == "C2KV"}, {4})
        self.assertEqual({row["method"] for row in rows}, {"Full", "HiAgent", "ACON", "C2KV", "H2O", "SnapKV"})

    def test_preparation_validates_registered_arms_and_full_prefix_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            plan, profile = prepare(self.config, output, output / "sglang")
            self.assertEqual(len(plan), 24)
            self.assertTrue(profile.is_file())
            for row in plan:
                cmd = row["command"]
                self.assertEqual(cmd[cmd.index("--num-workers") + 1], "1")
                self.assertEqual("--record-prefixes" in cmd, row["arm"] == "full")
                self.assertEqual(Path(row["replay_source"]).parent.name, row["benchmark"] + "__full")

    def test_cuda_command_is_single_flight(self):
        cmd = server_command(self.config, Path("sglang"))
        self.assertEqual(cmd[cmd.index("--device") + 1], "cuda")
        self.assertEqual(cmd[cmd.index("--max-running-requests") + 1], "1")
        self.assertIn("--enable-streaming-session", cmd)
        self.assertNotIn("npu", " ".join(cmd))

    def test_execution_path_levers_default_to_conservative_flags(self):
        bare = {key: value for key, value in self.config.items()
                if key not in ("disable_cuda_graph", "radix_cache_arms")}
        cmd = server_command(bare, Path("sglang"), "full")
        self.assertIn("--disable-radix-cache", cmd)
        self.assertIn("--disable-cuda-graph", cmd)
        self.assertIn("--disable-piecewise-cuda-graph", cmd)

    def test_shipped_config_serves_flashinfer_graph_and_radix_for_text_arms(self):
        self.assertEqual(self.config["attention_backend"], "flashinfer")
        self.assertFalse(self.config["disable_cuda_graph"])
        text_arms = {"full", "hiagent_full", "acon_hist_ut_co"}
        self.assertEqual(set(self.config["radix_cache_arms"]), text_arms)
        for row in cells(self.config):
            cmd = server_command(self.config, Path("sglang"), row["arm"])
            self.assertNotIn("--disable-cuda-graph", cmd)
            # Text arms keep SGLang's prefix cache; KV-compression arms reuse
            # KV through their own session/gist mechanisms.
            self.assertEqual("--disable-radix-cache" not in cmd, row["arm"] in text_arms)

    def test_radix_cache_is_per_arm_and_cuda_graph_is_a_config_toggle(self):
        config = dict(self.config, radix_cache_arms=["full"], disable_cuda_graph=False)
        full = server_command(config, Path("sglang"), "full")
        self.assertNotIn("--disable-radix-cache", full)
        self.assertNotIn("--disable-cuda-graph", full)
        self.assertIn("--disable-piecewise-cuda-graph", full)
        compressed = server_command(config, Path("sglang"), "c2kv4")
        self.assertIn("--disable-radix-cache", compressed)
        everywhere = server_command(dict(config, radix_cache_arms=["*"]), Path("sglang"))
        self.assertNotIn("--disable-radix-cache", everywhere)
        self.assertNotIn("--disable-cuda-graph", compressed)
        self.assertEqual(compressed[compressed.index("--max-running-requests") + 1], "1")

    def test_port_offset_shifts_runtime_ports_but_not_the_resolved_config(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            plan, _ = prepare(self.config, output, output / "sglang")
            cell = plan[0]
            shifted = runner.with_port_offset(self.config, 10)
            self.assertEqual(shifted["server_port"], self.config["server_port"] + 10)
            self.assertEqual(shifted["proxy_port"], self.config["proxy_port"] + 10)
            self.assertIs(runner.with_port_offset(self.config, 0), self.config)
            seen = {}

            def popen(*_args, **kwargs):
                return mock.Mock(pid=123)

            def run(cmd, **_kwargs):
                seen["run"] = cmd

            with mock.patch.object(runner.subprocess, "Popen", side_effect=popen),                     mock.patch.object(runner.subprocess, "run", side_effect=run),                     mock.patch.object(runner, "wait_server"),                     mock.patch.object(runner, "cleanup_cell_processes"),                     mock.patch("socket.socket") as socket_type:
                socket_type.return_value.__enter__.return_value.connect_ex.return_value = 1
                execute(self.config, [cell], output, output / "sglang",
                        ["closed_loop"], set(), port_offset=10)
            started = json.loads((output / "closed_loop" / cell["cell_id"] / "started.json").read_text())
            self.assertEqual(started["port_offset"], 10)
            server_cmd = started["server_command"]
            self.assertEqual(server_cmd[server_cmd.index("--port") + 1],
                             str(self.config["server_port"] + 10))
            self.assertEqual(seen["run"][seen["run"].index("--upstream") + 1],
                             f"http://127.0.0.1:{self.config['server_port'] + 10}")
            self.assertEqual(seen["run"][seen["run"].index("--proxy-port") + 1],
                             str(self.config["proxy_port"] + 10))
            resolved = json.loads((output / "config.resolved.json").read_text())
            self.assertEqual(resolved["server_port"], self.config["server_port"])
            differing = [(a, b) for a, b in zip(cell["command"], seen["run"]) if a != b]
            self.assertEqual(len(cell["command"]), len(seen["run"]))
            self.assertEqual(differing, [
                (f"http://127.0.0.1:{self.config['server_port']}",
                 f"http://127.0.0.1:{self.config['server_port'] + 10}"),
                (str(self.config["proxy_port"]), str(self.config["proxy_port"] + 10))])

    def test_prepare_resume_requires_semantically_equal_resolved_config(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "paper"
            source = Path(temporary) / "sglang"
            prepare(self.config, output, source)
            resolved = output / "config.resolved.json"
            equivalent = json.loads(resolved.read_text())
            resolved.write_text(json.dumps(equivalent, sort_keys=True))
            prepare(dict(self.config), output, source)

            before = resolved.read_text()
            commands_before = (output / "commands.json").read_text()
            changed = dict(self.config, seed=self.config["seed"] + 1)
            with self.assertRaisesRegex(RuntimeError, "different config"):
                prepare(changed, output, source)
            self.assertEqual(resolved.read_text(), before)
            self.assertEqual((output / "commands.json").read_text(), commands_before)

    def test_prepare_rejects_nonempty_output_without_resolved_config(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "paper"
            output.mkdir()
            (output / "orphan.txt").write_text("partial")
            with self.assertRaisesRegex(RuntimeError, "no resolved config"):
                prepare(self.config, output, Path(temporary) / "sglang")

    @staticmethod
    def _write_aggregate_inputs(output, stage, cell):
        directory = output / stage / cell["cell_id"]
        directory.mkdir(parents=True)
        for name in ("complete.json", "proxy_telemetry.jsonl",
                     "server_telemetry.jsonl"):
            (directory / name).write_text("{}\n")
        if stage == "closed_loop":
            (directory / f"summary_{cell['arm']}.json").write_text("{}\n")
            if cell["arm"] == "full":
                (directory / "full_prefixes.jsonl").write_text("{}\n")
            if cell["adapter"] == "acon_appworld":
                harness = directory / "measurement" / "harness_events.jsonl"
                harness.parent.mkdir()
                harness.write_text("{}\n")
        else:
            (directory / "prefix_replay.jsonl").write_text("{}\n")

    def test_aggregate_coverage_honors_stage_and_cells_and_fails_missing(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            plan = cells(self.config)[:2]
            selected = {plan[0]["cell_id"]}
            self._write_aggregate_inputs(output, "closed_loop", plan[0])
            with mock.patch.object(runner.subprocess, "run") as run:
                requested, coverage_path = aggregate_results(
                    self.config, plan, output, ["closed_loop"], selected)
            self.assertEqual([cell["cell_id"] for cell in requested], list(selected))
            run.assert_called_once()
            coverage = json.loads(coverage_path.read_text())
            self.assertEqual(coverage["requested_stages"], ["closed_loop"])
            self.assertEqual(coverage["counts"], {
                "requested": 1, "ready": 1, "missing": 0, "aggregated": 1})
            self.assertEqual(coverage["cells"][0]["status"], "aggregated")

            with self.assertRaisesRegex(RuntimeError, "slice is incomplete"):
                aggregate_results(
                    self.config, plan, output, ["closed_loop"],
                    {plan[0]["cell_id"], plan[1]["cell_id"]})
            missing = json.loads(coverage_path.read_text())
            self.assertEqual(missing["counts"]["missing"], 1)
            self.assertEqual(missing["missing"][0]["cell_id"], plan[1]["cell_id"])
            self.assertTrue(any(path.endswith("complete.json") for path in
                                missing["missing"][0]["missing_artifacts"]))

    def test_aggregate_common_prefix_requires_only_requested_stage_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            cell = cells(self.config)[0]
            self._write_aggregate_inputs(output, "common_prefix", cell)
            with mock.patch.object(runner.subprocess, "run") as run:
                aggregate_results(
                    self.config, [cell], output, ["common_prefix"],
                    {cell["cell_id"]})
            command = run.call_args.args[0]
            self.assertIn("--replay", command)
            coverage = json.loads(
                (output / "aggregation_coverage.json").read_text())
            self.assertEqual(coverage["requested_stages"], ["common_prefix"])

    def test_aggregate_cli_passes_exact_stage_and_cells(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            cell = cells(self.config)[0]
            (output / "config.resolved.json").write_text(json.dumps(self.config))
            (output / "commands.json").write_text(json.dumps([cell]))
            coverage = output / "aggregation_coverage.json"
            with mock.patch.object(
                    runner, "aggregate_results", return_value=([cell], coverage)) as aggregate, \
                    mock.patch(
                        "benchmarks.paper.report.write_comparison") as write_comparison:
                runner.main([
                    "aggregate", "--output", str(output),
                    "--stage", "common_prefix", "--cells", cell["cell_id"],
                ])
            aggregate.assert_called_once_with(
                self.config, [cell], output, ["common_prefix"], {cell["cell_id"]})
            write_comparison.assert_called_once_with(output, [cell])

    def test_proxy_timeout_fallback_still_cleans_server(self):
        events = []

        class Proxy:
            def terminate(self):
                events.append("proxy_terminate")

            def wait(self, timeout=None):
                events.append(("proxy_wait", timeout))
                if timeout == 20:
                    raise subprocess.TimeoutExpired("proxy", timeout)
                raise OSError("fixture proxy wait failure after kill")

            def kill(self):
                events.append("proxy_kill")

        class Server:
            pid = 77

            def wait(self, timeout=None):
                events.append(("server_wait", timeout))

        with mock.patch.object(runner.os, "killpg", side_effect=lambda pid, sig: events.append(("server_signal", pid, sig)), create=True):
            with self.assertRaisesRegex(RuntimeError, "proxy"):
                cleanup_cell_processes(Proxy(), Server())
        self.assertIn("proxy_kill", events)
        self.assertTrue(any(isinstance(event, tuple) and event[0] == "server_signal"
                            for event in events))
        self.assertIn(("server_wait", 30), events)

    def test_execute_marks_complete_only_after_successful_cleanup_and_sets_cuda_env(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            plan, _ = prepare(self.config, output, output / "sglang")
            cell = plan[0]
            complete = output / "closed_loop" / cell["cell_id"] / "complete.json"
            seen_env = {}

            def popen(*_args, **kwargs):
                seen_env.update(kwargs["env"])
                return mock.Mock(pid=123)

            def cleanup(_proxy, _server):
                self.assertFalse(complete.exists())

            with mock.patch.dict(os.environ, {"CUDA_HOME": "/custom/cuda"}), \
                    mock.patch.object(runner.subprocess, "Popen", side_effect=popen), \
                    mock.patch.object(runner.subprocess, "run"), \
                    mock.patch.object(runner, "wait_server"), \
                    mock.patch.object(runner, "cleanup_cell_processes", side_effect=cleanup), \
                    mock.patch("socket.socket") as socket_type:
                socket_type.return_value.__enter__.return_value.connect_ex.return_value = 1
                execute(self.config, [cell], output, output / "sglang",
                        ["closed_loop"], set())
            self.assertTrue(complete.is_file())
            self.assertEqual(seen_env["CUDA_HOME"], "/custom/cuda")
            self.assertEqual(Path(seen_env["PATH"].split(os.pathsep)[0]),
                             Path(self.config["server_python"]).parent)

            complete.unlink()
            (complete.parent / "started.json").unlink()
            with mock.patch.object(runner.subprocess, "Popen", side_effect=popen), \
                    mock.patch.object(runner.subprocess, "run"), \
                    mock.patch.object(runner, "wait_server"), \
                    mock.patch.object(runner, "cleanup_cell_processes",
                                      side_effect=RuntimeError("cleanup failed")), \
                    mock.patch("socket.socket") as socket_type:
                socket_type.return_value.__enter__.return_value.connect_ex.return_value = 1
                with self.assertRaisesRegex(RuntimeError, "cleanup failed"):
                    execute(self.config, [cell], output, output / "sglang",
                            ["closed_loop"], set())
            self.assertFalse(complete.exists())

    def test_offline_savings_use_actual_full_peak(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            plan = cells(self.config)[:4]
            for row, peak in ((plan[0], 115), (plan[3], 100)):
                directory = output / "common_prefix" / row["cell_id"]
                directory.mkdir(parents=True)
                (directory / "measurement_summary.json").write_text(json.dumps({
                    "memory": {"request_peak_resident_kv_bytes": {"max": peak}}}))
            report = write_comparison(output, plan)
            compressed = next(row for row in report if row["arm"] == "c2kv4")
            self.assertAlmostEqual(compressed["resident_kv_peak_bytes_saving_vs_full_pct"], 100 * (1 - 100 / 115))
            self.assertIsNone(compressed["semantic_score"])
            self.assertIsNone(compressed["model_ms_per_committed_action"])

    def test_common_prefix_uses_source_denominator_and_algorithm_time(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            plan = cells(self.config)[1:2]
            directory = output / "common_prefix" / plan[0]["cell_id"]
            directory.mkdir(parents=True)
            (directory / "measurement_summary.json").write_text(json.dumps({
                "token_ratios": {"whole": {"ratio_of_sums": None}},
                "common_prefix_token_ratios": {"whole": {"ratio_of_sums": 0.7},
                                                 "history": {"ratio_of_sums": 0.4}},
                "latency_ms": {"request": {"mean": 15},
                               "request_algorithm": {"mean": 12, "p95": 14, "p99": 14}}}))
            row = write_comparison(output, plan)[0]
            self.assertEqual(row["whole_retained_fraction"], 0.7)
            self.assertEqual(row["history_retained_fraction"], 0.4)
            self.assertEqual(row["model_request_mean_ms"], 12)
            self.assertEqual(row["observed_request_mean_ms"], 15)


if __name__ == "__main__":
    unittest.main()
