"""An explicit cell tool schema reaches the native server; the default command is untouched."""
from generality import c2kv_cell


def _cell(tmp_path, **extra):
    return {
        "cell_id": "bfcl_base__c2kv__b256__recovery_off_same_initial",
        "python_sgl": "python",
        "checkpoint": str(tmp_path / "checkpoint"),
        "model_name": "c2kv-agent",
        "benchmark": "bfcl",
        "ratio": 8,
        "eval_policy_path": tmp_path / "eval_policy.json",
        "controller_path": tmp_path / "controller.json",
        "sglang_backend_url": "http://127.0.0.1:36100",
        "condition": "recovery_off_same_initial",
        "caps": {"max_completion_tokens": 128, "generation_attempts_per_task": 2,
                 "extraction_calls_per_task": 4, "task_timeout": 60},
        **extra,
    }


def test_tool_schema_is_opt_in_for_cell_server_commands(tmp_path, monkeypatch):
    monkeypatch.setattr(c2kv_cell.current, "load_config", lambda: {
        "route": "ac_native_s0_lexical_raw_reserve_failed_operation",
        "compression_policy": "always-compress-v1",
        "history_view_protocol": "fixed-budget-main",
        "decode_strategy": "incremental", "prefill_chunk_size": 256,
    }, raising=False)
    tasks = ["multi_turn_base_0"]
    default = c2kv_cell.server_command(_cell(tmp_path), tasks, tmp_path / "attempt", 36300)
    assert "--tool-schema" not in default
    raw = c2kv_cell.server_command(_cell(tmp_path, tool_schema="raw"), tasks, tmp_path / "attempt", 36300)
    assert raw[raw.index("--tool-schema") + 1] == "raw"
    without_flag = raw[:raw.index("--tool-schema")] + raw[raw.index("--tool-schema") + 2:]
    assert without_flag == default
