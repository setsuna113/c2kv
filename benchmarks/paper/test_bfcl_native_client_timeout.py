"""The native BFCL client must wait for a whole decision under a configured deadline.

SZ-p6k BFCL Long B=256 RACER shards (paper 733eee5) died on their first task:
one decision ran past the official client's hard-coded 600 s read timeout while
the server's own deadline was 10800 s.  A configured --generation-timeout now
reaches the official worker; without it the worker command is unchanged.
"""
import json

from benchmarks.paper import c1, runner


def _worker(config, out, delivery, controller):
    args = c1.delivery_args(config, "bfcl_base", out, ["multi_turn_base_0"], delivery)
    return delivery.commands_for_task(args, "multi_turn_base_0", controller)[1]


def test_bfcl_worker_receives_only_a_configured_generation_deadline(tmp_path):
    original = c1.ARM
    c1.select_arm("c2kv_native_r8")
    try:
        config = dict(json.loads(runner.DEFAULT_CONFIG.read_text()), history_kv_budget_tokens=768)
        config = runner.with_native_ratios(config, [8])
        config["methods"] = [method for method in config["methods"]
                             if method["method"] != "C2KV" or method["arm"] == "c2kv_native_r8"]
        config.update(native_arm="c2kv_native_r8", sglang_source=str(tmp_path / "sglang"))
        (tmp_path / "bfcl" / "bfcl_eval").mkdir(parents=True)
        config["bfcl_dir"] = str(tmp_path / "bfcl")
        controller = tmp_path / "controller.json"
        controller.write_text("{}")
        delivery = c1.load_delivery()
        default = _worker(config, tmp_path / "out", delivery, controller)
        configured = _worker(dict(config, generation_timeout=10800.0), tmp_path / "out",
                             delivery, controller)
    finally:
        c1.select_arm(original)
    assert "--generation-timeout" not in default
    assert configured == default + ["--generation-timeout", "10800.0"]
    assert default[default.index("-m") + 1] == "benchmarks.memory_runtime.event_native_bfcl"
