import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from benchmarks.paper import runner
from benchmarks.paper.runner import (extension_problem, 
    DEFAULT_CONFIG, aggregate_results, cells, cleanup_cell_processes, execute,
    prepare, run_command, server_command,
)
from benchmarks.paper.report import write_comparison


def legacy_matrix_config():
    """Keep historical proxy identities explicit when testing frozen runs."""
    config = json.loads(DEFAULT_CONFIG.read_text())
    config["history_kv_budget_tokens"] = 768
    config["methods"] = [
        {key: value for key, value in method.items()
         if key not in {"history_runtime", "history_backend", "recovery_policy",
                        "compression_ratio"}}
        for method in config["methods"] if method["method"] != "StreamingLLM"
    ]
    for method in config["methods"]:
        if method["method"] == "C2KV":
            method["arm"] = "c2kv4"
            method["ratio"] = 4
            method.pop("history_budget_tokens", None)
    return config


class PaperMatrixTest(unittest.TestCase):
    def setUp(self):
        self.config = legacy_matrix_config()

    def test_exact_prefix_replay_is_rejected_before_starting_server(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan, _ = prepare(self.config, root, root / "engine")
            with mock.patch.object(runner.subprocess, "Popen") as popen:
                with self.assertRaisesRegex(RuntimeError, "exact_generated_prefix"):
                    execute(self.config, plan, root, root / "engine",
                            ["common_prefix"], {"bfcl_base__agentkv_b768"})
                popen.assert_not_called()
            self.assertFalse((root / "common_prefix" / "bfcl_base__agentkv_b768").exists())

    def test_toolsandbox_simulator_has_a_slot_beside_persistent_actor(self):
        for arm in ("history_kv_h2o_r25_persistent", "commitkv", "c2kv_native_r4"):
            command = server_command(self.config, Path("engine"), arm, "toolsandbox")
            self.assertEqual(command[command.index("--max-running-requests") + 1], "2")

    def test_exact_prefix_replay_aggregation_reports_unsupported(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan, _ = prepare(self.config, root, root / "engine")
            with mock.patch.object(runner.subprocess, "run") as run:
                _, path = aggregate_results(self.config, plan, root, ["common_prefix"],
                                            {"bfcl_base__agentkv_b768", "bfcl_base__commitkv_b768"})
                run.assert_not_called()
            receipt = json.loads(path.read_text())
            self.assertEqual(receipt["counts"]["unsupported"], 2)
            self.assertEqual(receipt["counts"]["missing"], 0)
            self.assertTrue(all(row["status"] == "unsupported_protocol" for row in receipt["cells"]))

    def test_agentfold_hold_precedes_launch_and_preserves_history(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "paper"
            plan, _ = prepare(self.config, output, root / "sglang")
            cell_id = "bfcl_base__agentfold"
            with mock.patch.object(runner.subprocess, "Popen") as popen:
                with self.assertRaisesRegex(RuntimeError, "AgentFold is on hold"):
                    execute(self.config, plan, output, root / "sglang",
                            ["closed_loop"], {cell_id})
                popen.assert_not_called()
            directory = output / "closed_loop" / cell_id
            self.assertFalse(directory.exists())
            directory.mkdir(parents=True)
            (directory / "complete.json").write_text("{}")
            with mock.patch.object(runner.subprocess, "Popen") as popen:
                execute(self.config, plan, output, root / "sglang",
                        ["closed_loop"], {cell_id})
                popen.assert_not_called()

    def test_only_requested_methods_and_ratio(self):
        rows = [row for row in cells(self.config) if row["benchmark"] != "tau2"]
        # The tool-context axis adds compressed-tool cells to the raw-tool matrix.
        raw = [row for row in rows if row["tool_context"] == "raw"]
        self.assertEqual(len(raw), 56)
        self.assertEqual(len(rows) - len(raw), 10)
        self.assertEqual(sum(row["group"] == "main" for row in raw), 39)
        self.assertEqual({row["ratio"] for row in rows if row["method"] == "C2KV"}, {4})
        self.assertEqual({row["method"] for row in rows}, {
            "Full", "HiAgent", "ACON", "C2KV", "H2O", "SnapKV", "PyramidKV",
            "AgentFold", "CommitKV", "AgentKV", "C2KV+C1",
        })
        self.assertTrue(all(row["arm"].startswith("c2kv_c1_t02_r") for row in rows[-6:]))
        # The ratio-4 C1 ablation adds ACEBench without changing older cells.
        r4 = [row for row in rows if row["arm"] == "c2kv_c1_t02_r4"]
        self.assertEqual([row["cell_id"] for row in r4],
                         ["bfcl_base__c2kv_c1_t02_r4", "acebench_agent__c2kv_c1_t02_r4"])
        for row in r4:
            self.assertEqual(row["ratio"], 4)
            self.assertNotIn("benchmarks", row)
        self.assertEqual(sum(row["arm"] == "c2kv_c1_t02_r8" for row in rows), 4)

    def test_preparation_validates_registered_arms_and_full_prefix_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            plan, profile = prepare(self.config, output, output / "sglang")
            self.assertEqual(len([row for row in plan if row["benchmark"] != "tau2"]), 66)
            self.assertTrue(profile.is_file())
            for row in plan:
                cmd = row["command"]
                self.assertEqual(cmd[cmd.index("--num-workers") + 1], "1")
                # only the raw-tools Full cell records the canonical replay prefixes
                self.assertEqual("--record-prefixes" in cmd,
                                 row["arm"] == "full" and row["tool_context"] == "raw")
                self.assertEqual("--tool-memory" in cmd, row["tool_context"] != "raw")
                self.assertEqual(Path(row["replay_source"]).parent.name, row["benchmark"] + "__full")
            self.assertEqual(json.loads((output / "unsupported_cells.json").read_text()),
                             self.config["unsupported_cells"])

    def test_explicit_generation_timeout_only_reaches_persistent_history_cells(self):
        config = dict(self.config, generation_timeout=1200)
        rows = cells(config)
        profile = Path("out/deployment_profile.json")
        persistent = next(row for row in rows if row["cell_id"] == "bfcl_base__commitkv_b768")
        ordinary = next(row for row in rows if row["cell_id"] == "bfcl_base__full")

        default_cmd = run_command(
            self.config, persistent, Path("out/default-persistent"), profile)
        persistent_cmd = run_command(config, persistent, Path("out/persistent"), profile)
        ordinary_cmd = run_command(config, ordinary, Path("out/ordinary"), profile)

        # An existing frozen config keeps benchmarks.run's 600-second default.
        self.assertNotIn("--generation-timeout", default_cmd)
        self.assertEqual(
            persistent_cmd[persistent_cmd.index("--generation-timeout") + 1], "1200")
        self.assertNotIn("--generation-timeout", ordinary_cmd)

    def test_cli_timeout_is_frozen_and_cannot_rewrite_existing_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            legacy_config = Path(temporary) / "legacy_config.json"
            legacy_config.write_text(json.dumps(self.config))
            runner.main(["prepare", "--output", str(output),
                         "--config", str(legacy_config),
                         "--history-kv-budget-tokens", "768",
                         "--generation-timeout", "1800"])
            frozen = (output / "config.resolved.json").read_bytes()
            self.assertEqual(json.loads(frozen)["generation_timeout"], 1800)
            plan = json.loads((output / "commands.json").read_text())
            row = next(row for row in plan
                       if row["cell_id"] == "bfcl_long_context__commitkv_b768")
            cmd = row["command"]
            self.assertEqual(cmd[cmd.index("--generation-timeout") + 1], "1800.0")
            with self.assertRaises((ValueError, RuntimeError)):
                runner.main(["prepare", "--output", str(output),
                             "--config", str(legacy_config),
                             "--history-kv-budget-tokens", "768",
                             "--generation-timeout", "3600"])
            self.assertEqual((output / "config.resolved.json").read_bytes(), frozen)
            with self.assertRaises(SystemExit):
                runner.main(["aggregate", "--output", str(output),
                             "--config", str(legacy_config),
                             "--generation-timeout", "3600"])

    def test_prepare_pins_original_opponent_runtime_contracts(self):
        from dataclasses import replace
        from benchmarks.arms import ARMS

        cases = [
            ("agentfold", replace(
                ARMS["agentfold"], text_policy="length_summary_surrogate"),
             "actor folding policy"),
            ("commitkv", replace(
                ARMS["commitkv"],
                history_kv={**ARMS["commitkv"].history_kv, "backend": "physical_eviction"}),
             "reference-attention"),
            ("agentkv", replace(
                ARMS["agentkv"],
                history_kv={**ARMS["agentkv"].history_kv, "backend": "physical_eviction"}),
             "reference-attention"),
            ("history_kv_pyramidkv_r25_persistent", replace(
                ARMS["history_kv_pyramidkv_r25_persistent"],
                history_kv={
                    **ARMS["history_kv_pyramidkv_r25_persistent"].history_kv,
                    "backend": "physical_eviction",
                }),
             "budget differs"),
        ]
        for arm_name, bad_arm, message in cases:
            with self.subTest(arm=arm_name), tempfile.TemporaryDirectory() as temporary:
                with mock.patch.dict(ARMS, {arm_name: bad_arm}):
                    with self.assertRaisesRegex(ValueError, message):
                        prepare(self.config, Path(temporary), Path(temporary) / "sglang")

    def test_agent_and_toolsandbox_are_real_method_matrices(self):
        rows = cells(self.config)
        expected = {"full", "hiagent_full", "acon_hist_ut_co", "c2kv4",
                    "history_kv_h2o_r25_persistent",
                    "history_kv_snapkv_r25_persistent",
                    "history_kv_pyramidkv_r25_persistent",
                    "agentfold", "commitkv", "agentkv"}
        for benchmark in ("acebench_agent", "toolsandbox"):
            actual = {row["arm"] for row in rows if row["benchmark"] == benchmark}
            self.assertEqual(actual, expected | ({"c2kv_c1_t02_r8", "c2kv_c1_t02_r4"}
                                                 if benchmark == "acebench_agent" else set()))
        unsupported = {(row["benchmark"], row["arm"])
                       for row in self.config["unsupported_cells"]}
        self.assertEqual(unsupported, {
            ("toolsandbox", "c2kv_c1_t02_r8"),
        })
        self.assertIn("end-to-end ToolSandbox C1 validation is pending",
                      self.config["unsupported_cells"][0]["reason"])
        ace_c1 = next(row for row in rows
                      if row["cell_id"] == "acebench_agent__c2kv_c1_t02_r8")
        command = run_command(self.config, ace_c1, Path("out/ace-c1"),
                              Path("out/deployment_profile.json"))
        self.assertEqual(command[command.index("--benchmark") + 1], "acebench_agent")
        self.assertEqual(command[command.index("--arm") + 1], "c2kv_c1_t02_r8")
        ace_r4 = next(row for row in rows
                      if row["cell_id"] == "acebench_agent__c2kv_c1_t02_r4")
        r4_command = run_command(self.config, ace_r4, Path("out/ace-c1-r4"),
                                 Path("out/deployment_profile.json"))
        self.assertEqual(r4_command[r4_command.index("--benchmark") + 1], "acebench_agent")
        self.assertEqual(r4_command[r4_command.index("--arm") + 1], "c2kv_c1_t02_r4")

    def test_benchmark_commands_freeze_official_scope_and_single_flight(self):
        rows = cells(self.config)
        profile = Path("out/deployment_profile.json")
        ace = next(row for row in rows if row["cell_id"] == "acebench_agent__full")
        ace_cmd = run_command(self.config, ace, Path("out/ace"), profile)
        self.assertEqual(ace_cmd[ace_cmd.index("--acebench-category") + 1], "agent")
        self.assertEqual(ace_cmd[ace_cmd.index("--acebench-language") + 1], "en")
        self.assertIn("acebench_role_history_v1", ace_cmd[ace_cmd.index("--capability-features") + 1])
        ts = next(row for row in rows if row["cell_id"] == "toolsandbox__full")
        ts_cmd = run_command(self.config, ts, Path("out/ts"), profile)
        self.assertEqual(ts_cmd[ts_cmd.index("--ts-suite") + 1],
                         "three_distraction_tools_129")
        self.assertEqual(len(ts_cmd[ts_cmd.index("--ts-scenarios") + 1].split(",")), 129)
        self.assertEqual(ts_cmd[ts_cmd.index("--ts-parallel") + 1], "1")

    def test_reference_attention_arms_get_static_pool_headroom(self):
        from benchmarks.paper.runner import REFERENCE_ATTENTION_MEM_FRACTION
        full = server_command(self.config, Path("sglang"), "full")
        self.assertEqual(full[full.index("--mem-fraction-static") + 1], str(self.config["mem_fraction_static"]))
        for arm in ("history_kv_pyramidkv_r25_persistent", "agentkv", "commitkv"):
            cmd = server_command(self.config, Path("sglang"), arm)
            self.assertEqual(cmd[cmd.index("--mem-fraction-static") + 1],
                             str(min(float(self.config["mem_fraction_static"]), REFERENCE_ATTENTION_MEM_FRACTION)))
            self.assertIn("--disable-cuda-graph", cmd)
        h2o = server_command(self.config, Path("sglang"), "history_kv_h2o_r25_persistent")
        self.assertEqual(h2o[h2o.index("--mem-fraction-static") + 1], str(self.config["mem_fraction_static"]))

    def test_acebench_cells_serve_two_request_slots(self):
        from benchmarks.paper.runner import ACEBENCH_MAX_RUNNING_REQUESTS
        for arm in ("full", "history_kv_h2o_r25_persistent", "agentkv"):
            ace = server_command(self.config, Path("sglang"), arm, "acebench_agent")
            self.assertEqual(ace[ace.index("--max-running-requests") + 1], str(ACEBENCH_MAX_RUNNING_REQUESTS))
            for other in ("bfcl_base", "appworld", None):
                cmd = server_command(self.config, Path("sglang"), arm, other)
                self.assertEqual(cmd[cmd.index("--max-running-requests") + 1], "1")

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
        from benchmarks.arms import get_arm, history_kv_spec
        text_arms = {"full", "hiagent_full", "acon_hist_ut_co", "agentfold"}
        self.assertEqual(set(self.config["radix_cache_arms"]), text_arms)
        for row in cells(self.config):
            cmd = server_command(self.config, Path("sglang"), row["arm"])
            spec = history_kv_spec(get_arm(row["arm"]))
            reference = bool(spec and spec["backend"] == "reference_attention")
            self.assertEqual("--disable-cuda-graph" in cmd, reference)
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

            with mock.patch.object(runner.subprocess, "Popen", side_effect=popen),                     mock.patch.object(runner, "run_owned", side_effect=run),                     mock.patch.object(runner, "wait_server"),                     mock.patch.object(runner, "cleanup_cell_processes"),                     mock.patch("socket.socket") as socket_type:
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

    def test_execute_rejects_event_native_checkpoint_before_starting_legacy_arm(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            (checkpoint / "config.json").write_text(json.dumps({
                "history_memory_training_profile": "history-event-base-query-v1",
                "history_memory_packing_version": "history-event-v1",
                "history_memory_raw_layout": "event-native-evidence-v1",
            }))
            config = dict(self.config, checkpoint=str(checkpoint))
            output = root / "paper"
            plan, _ = prepare(config, output, root / "sglang")
            cell = next(row for row in plan
                        if row["cell_id"] == "bfcl_base__c2kv4")
            directory = output / "closed_loop" / cell["cell_id"]

            with mock.patch.object(runner.subprocess, "Popen") as popen:
                with self.assertRaisesRegex(
                        RuntimeError,
                        "Training/serving mismatch.*native event-packed base arm"):
                    execute(config, [cell], output, root / "sglang",
                            ["closed_loop"], {cell["cell_id"]})
            popen.assert_not_called()
            self.assertFalse(directory.exists())

            # A completed historical cell is evidence, not a request to rerun the
            # now-invalid serving path.  It must remain byte-for-byte untouched.
            directory.mkdir(parents=True)
            complete = directory / "complete.json"
            complete.write_text('{"historical": true}\n')
            with mock.patch.object(runner.subprocess, "Popen") as popen:
                execute(config, [cell], output, root / "sglang",
                        ["closed_loop"], {cell["cell_id"]})
            popen.assert_not_called()
            self.assertEqual(complete.read_text(), '{"historical": true}\n')

    def test_unselected_legacy_arm_does_not_block_safe_event_native_cells(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            (checkpoint / "config.json").write_text(json.dumps({
                "history_memory_training_profile": "history-event-base-query-v1",
                "history_memory_packing_version": "history-event-v1",
                "history_memory_raw_layout": "event-native-evidence-v1",
            }))
            config = dict(self.config, checkpoint=str(checkpoint))
            output = root / "paper"
            plan, _ = prepare(config, output, root / "sglang")
            legacy = next(row for row in plan
                          if row["cell_id"] == "bfcl_base__c2kv4")
            safe_ids = {
                "bfcl_base__full",
                "bfcl_base__history_kv_h2o_persistent_b768",
                "bfcl_base__c2kv_c1_t02_r8",
            }

            with mock.patch.object(
                    runner.subprocess, "Popen",
                    side_effect=lambda *_args, **_kwargs: mock.Mock(pid=123)) as popen, \
                    mock.patch.object(runner, "run_owned"), \
                    mock.patch.object(runner, "wait_server"), \
                    mock.patch.object(runner, "cleanup_cell_processes"), \
                    mock.patch("socket.socket") as socket_type:
                socket_type.return_value.__enter__.return_value.connect_ex.return_value = 1
                execute(config, plan, output, root / "sglang",
                        ["closed_loop"], safe_ids)

            self.assertEqual(popen.call_count, len(safe_ids))
            self.assertFalse((output / "closed_loop" / legacy["cell_id"]).exists())
            for cell_id in safe_ids:
                self.assertTrue(
                    (output / "closed_loop" / cell_id / "complete.json").is_file())

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
                "requested": 1, "ready": 1, "unsupported": 0, "audit_excluded": 0,
                "missing": 0, "aggregated": 1})
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

    def test_toolsandbox_quality_aggregation_requires_joinable_harness_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            cell = next(row for row in cells(self.config)
                        if row["cell_id"] == "toolsandbox__full")
            directory = output / "closed_loop" / cell["cell_id"]
            directory.mkdir(parents=True)
            for name in ("complete.json", "proxy_telemetry.jsonl",
                         "server_telemetry.jsonl", "summary_full.json"):
                (directory / name).write_text("{}\n")
            with self.assertRaisesRegex(RuntimeError, "slice is incomplete"):
                aggregate_results(self.config, [cell], output,
                                  ["closed_loop"], {cell["cell_id"]})
            missing = json.loads((output / "aggregation_coverage.json").read_text())
            paths = missing["missing"][0]["missing_artifacts"]
            self.assertTrue(any(path.endswith("scenario_manifest.json") for path in paths))
            self.assertTrue(any(path.endswith("harness_events.jsonl") for path in paths))

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
                    mock.patch.object(runner, "run_owned"), \
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
                    mock.patch.object(runner, "run_owned"), \
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
            plan = [row for row in cells(self.config) if row["tool_context"] == "raw"][:4]
            for row, peak in ((plan[0], 115), (plan[3], 100)):
                directory = output / "common_prefix" / row["cell_id"]
                directory.mkdir(parents=True)
                (directory / "complete.json").write_text("{}")
                (directory / "measurement_summary.json").write_text(json.dumps({
                    "memory": {"request_peak_resident_kv_bytes": {"max": peak},
                               "resident_peak_chain": {
                                   "request_peak_c2kv_cache_accounting_available": True}}}))
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
            (directory / "complete.json").write_text("{}")
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


class DefaultHistoryBudgetTest(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(DEFAULT_CONFIG.read_text())
        self.config["history_kv_budget_tokens"] = 768

    def test_default_history_kv_cells_share_one_absolute_budget(self):
        expected = {
            "H2O": "h2o",
            "SnapKV": "snapkv",
            "StreamingLLM": "streamingllm",
            "PyramidKV": "pyramidkv",
            "CommitKV": "commitkv",
            "AgentKV": "agentkv",
        }
        all_benchmarks = {item["name"] for item in self.config["benchmarks"]}
        rows = cells(self.config)
        self.assertFalse(any(row["group"] == "sweep" for row in rows))
        for method, backend in expected.items():
            arm = f"racer_{backend}_off_b768"
            selected = [row for row in rows if row["method"] == method]
            self.assertEqual({row["benchmark"] for row in selected}, all_benchmarks)
            self.assertEqual(len(selected), len(all_benchmarks))
            for row in selected:
                with self.subTest(method=method, benchmark=row["benchmark"]):
                    self.assertEqual(row["group"], "racer")
                    self.assertEqual(row["arm"], arm)
                    self.assertEqual(row["history_runtime"], "racer")
                    self.assertEqual(row["history_backend"], backend)
                    self.assertEqual(row["recovery_policy"], "off")
                    self.assertEqual(row["history_budget_tokens"], 768)
                    self.assertEqual(row["cell_id"], f'{row["benchmark"]}__{arm}')
                    self.assertEqual(row["racer_backend"]["history_budget_tokens"], 768)
                    self.assertNotIn("retention", row)
                    self.assertNotIn("history_retention_ratio", row)
                    command = run_command(self.config, row, Path("out/cell"),
                                          Path("out/deployment_profile.json"))
                    self.assertEqual(command[command.index("--arm") + 1], arm)
                    self.assertEqual(command[command.index("-m") + 1], "benchmarks.paper.c1")
                    self.assertNotIn("--history-kv-target-tokens", command)
                    self.assertNotIn("--history-kv-retention-ratio", command)
        native = [row for row in rows if row["method"] == "C2KV"]
        self.assertEqual({row["benchmark"] for row in native}, all_benchmarks)
        self.assertEqual(len(native), len(all_benchmarks))
        for row in native:
            with self.subTest(method="C2KV", benchmark=row["benchmark"]):
                self.assertEqual(row["arm"], "c2kv_native_r8")
                self.assertEqual(row["cell_id"], f'{row["benchmark"]}__c2kv_native_r8_b768')
                self.assertEqual(row["history_runtime"], "racer")
                self.assertEqual(row["history_backend"], "c2kv")
                self.assertEqual(row["history_allocation"], "c2kv_bare")
                self.assertEqual(row["recovery_policy"], "off")
                self.assertEqual(row["history_budget_tokens"], 768)
                command = run_command(self.config, row, Path("out/cell"),
                                      Path("out/deployment_profile.json"))
                self.assertEqual(command[command.index("--arm") + 1], "c2kv_native_r8")
                self.assertEqual(command[command.index("--history-budget-tokens") + 1], "768")

    def test_frozen_matrix_cannot_resume_with_a_different_absolute_budget(self):
        import copy

        old = copy.deepcopy(self.config)
        old["history_kv_budget_tokens"] = 512
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            old_plan, _ = prepare(old, output, output / "sglang")
            old_resolved = (output / "config.resolved.json").read_bytes()
            old_cell = next(row for row in old_plan
                            if row["cell_id"] == "bfcl_base__racer_commitkv_off_b512")
            marker = output / "closed_loop" / old_cell["cell_id"] / "complete.json"
            marker.parent.mkdir(parents=True)
            marker.write_text("{}\n")
            with self.assertRaisesRegex(RuntimeError, "different config"):
                prepare(self.config, output, output / "sglang")
            self.assertEqual((output / "config.resolved.json").read_bytes(), old_resolved)
            self.assertEqual(marker.read_text(), "{}\n")


class ExtensionRuleTest(unittest.TestCase):
    """An output root may grow (new arms, new benchmarks, reordered methods, flags for
    arms not yet run) but never change a cell it already defined."""

    def setUp(self):
        self.config = legacy_matrix_config()

    def _old(self):
        import copy
        old = copy.deepcopy(self.config)
        # the shape of the first matrix: fewer methods, no per-method benchmark lists,
        # no model_family, radix cache for the three text arms only
        old["methods"] = [dict((k, v) for k, v in m.items() if k != "benchmarks")
                          for m in old["methods"] if m["group"] != "baseline" and m["arm"] in
                          ("full", "hiagent_full", "acon_hist_ut_co", "c2kv4",
                           "history_kv_h2o_r25_persistent", "history_kv_snapkv_r25_persistent",
                           "c2kv_c1_t02_r8")]
        old["benchmarks"] = [b for b in old["benchmarks"] if b["name"] in ("bfcl_base", "bfcl_long_context", "appworld")]
        old.pop("model_family", None)
        old["radix_cache_arms"] = ["full", "hiagent_full", "acon_hist_ut_co"]
        return old

    def test_growth_is_accepted_and_archived(self):
        from benchmarks.paper.runner import extension_problem
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            old_plan, _ = prepare(self._old(), output, output / "sglang")
            done = output / "closed_loop" / old_plan[0]["cell_id"] / "complete.json"
            done.parent.mkdir(parents=True); done.write_text("{}\n")
            self.assertIsNone(extension_problem(json.loads((output / "config.resolved.json").read_text()),
                                                dict(self.config, sglang_source=str((output / "sglang").resolve())),
                                                output / "sglang", output))
            new_plan, _ = prepare(self.config, output, output / "sglang")
            self.assertGreater(len(new_plan), len(old_plan))
            self.assertTrue(set(r["cell_id"] for r in old_plan) <= set(r["cell_id"] for r in new_plan))
            self.assertEqual(done.read_text(), "{}\n")
            self.assertTrue(list(output.glob("config.before_extension.*.json")))
            # old cells keep byte-identical commands
            old_cmd = {r["cell_id"]: r["command"] for r in old_plan}
            for row in new_plan:
                if row["cell_id"] in old_cmd:
                    self.assertEqual(row["command"], old_cmd[row["cell_id"]])

    def test_engine_checkout_may_move_without_new_cells(self):
        """Rolling deployments point the same matrix at a new sglang checkout; only the
        recorded ``sglang_source`` differs, which is not an experiment change."""
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            plan_a, _ = prepare(self.config, output, output / "sglang-v3")
            plan_b, _ = prepare(self.config, output, output / "sglang-v4")
            self.assertEqual([r["cell_id"] for r in plan_a], [r["cell_id"] for r in plan_b])
            resolved = json.loads((output / "config.resolved.json").read_text())
            self.assertEqual(resolved["sglang_source"], str((output / "sglang-v4").resolve()))
            self.assertTrue(list(output.glob("config.before_extension.*.json")))
            # anything else still goes through the extension rule
            import copy
            changed = copy.deepcopy(self.config)
            changed["chunked_prefill_size"] = 256
            with self.assertRaisesRegex(RuntimeError, "different config"):
                prepare(changed, output, output / "sglang-v4")

    def test_touching_an_existing_cell_is_refused(self):
        import copy
        from benchmarks.paper.runner import extension_problem
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            prepare(self._old(), output, output / "sglang")
            existing = json.loads((output / "config.resolved.json").read_text())
            source = output / "sglang"
            changed = copy.deepcopy(self.config)
            changed["chunked_prefill_size"] = 256
            self.assertIn("deployment field", extension_problem(existing, changed, source, output) or "")
            changed = copy.deepcopy(self.config)
            changed["radix_cache_arms"] = [a for a in changed["radix_cache_arms"] if a != "full"]
            self.assertIn("server command changed", extension_problem(existing, changed, source, output) or "")
            changed = copy.deepcopy(self.config)
            changed["benchmarks"] = [b for b in changed["benchmarks"] if b["name"] != "appworld"]
            self.assertIn("cells removed", extension_problem(existing, changed, source, output) or "")
            self.assertEqual(extension_problem(existing, dict(existing), source, output), "no new cells")

    def test_old_root_without_new_adapter_paths_and_label_renames(self):
        """The 45-cell roots predate ``acebench_dir``/``toolsandbox_dir`` and labelled the
        two Full baselines separately; the current code must still render their old
        commands, and a label may change only while the cell has no artifacts."""
        import copy
        from benchmarks.paper.runner import extension_problem
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            source = output / "sglang"
            old = self._old()
            old["methods"] += [
                {"method": "ACEBench Agent baseline", "arm": "full", "group": "baseline",
                 "benchmarks": ["acebench_agent"]},
                {"method": "ToolSandbox baseline", "arm": "full", "group": "baseline",
                 "benchmarks": ["toolsandbox"]}]
            old["benchmarks"] = self.config["benchmarks"]
            for item in old["methods"]:
                if item["arm"] == "c2kv_c1_t02_r8":   # C1 never had the new adapters' cells
                    item["benchmarks"] = ["bfcl_base", "bfcl_long_context", "appworld"]
            for key in ("acebench_dir", "acebench_python", "acebench_language",
                        "toolsandbox_dir", "toolsandbox_python", "toolsandbox_suite",
                        "toolsandbox_scenarios", "unsupported_cells"):
                old.pop(key, None)
            (output / "closed_loop").mkdir(parents=True)
            (output / "config.resolved.json").write_text(json.dumps(old) + "\n")
            new = dict(self.config, sglang_source=str(source.resolve()))
            self.assertIsNone(extension_problem(old, new, source, output))
            new_plan, _ = prepare(self.config, output, source)
            labels = {r["cell_id"]: r["method"] for r in new_plan}
            self.assertEqual(labels["acebench_agent__full"], "Full")
            # once the relabelled cell has artifacts, its label is frozen
            (output / "closed_loop" / "acebench_agent__full").mkdir()
            relabel = copy.deepcopy(self.config)
            for item in relabel["methods"]:
                if item["arm"] == "full":
                    item["method"] = "Full (renamed)"
                if item["arm"] == "c2kv_c1_t02_r4":   # a genuine new cell, so the label check is reached
                    item["benchmarks"] = ["bfcl_base", "bfcl_long_context", "acebench_agent"]
            self.assertIn("method changed", extension_problem(
                json.loads((output / "config.resolved.json").read_text()), relabel, source, output) or "")
            with self.assertRaises(RuntimeError):
                prepare(dict(self.config, chunked_prefill_size=256), output, output / "sglang")
