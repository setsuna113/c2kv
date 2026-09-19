import copy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import urllib.error

import pytest

from benchmarks.arms import get_arm
from benchmarks.paper import c1 as paper_c1
from benchmarks.paper.c1 import (ARM, controller_oom_message, controller_step_failure, replay_task_id,
                                 selected_tasks, select_arm, summarize_scores)
from benchmarks.paper.runner import DEFAULT_CONFIG, prepare, server_command
from benchmarks.measurement.telemetry import canonical_sha256, read_jsonl


def test_native_arm_requires_real_controller_and_keeps_bare_c2kv():
    assert get_arm(ARM).native_controller == "c1_t02"
    assert get_arm(ARM).ratio == 8
    assert get_arm("c2kv4").ratio == 4
    config = json.loads(DEFAULT_CONFIG.read_text())
    command = server_command(config, Path("sglang"), ARM)
    assert "--enable-return-hidden-states" in command
    assert "--c2kv-shadow-feature-layer" in command
    assert "--disable-cuda-graph" not in command
    assert "--enable-return-hidden-states" not in server_command(config, Path("sglang"), "full")


def test_paper_delivery_uses_configured_detector_and_defaults_to_d3_hybrid(tmp_path):
    config = json.loads(DEFAULT_CONFIG.read_text())
    config["sglang_source"] = str(tmp_path / "sglang")
    delivery = paper_c1.load_delivery()

    configured = paper_c1.delivery_args(
        config, "bfcl_base", tmp_path / "configured", ["multi_turn_base_0"], delivery
    )
    assert configured.detector == config["c1"].get("detector", "d3_hybrid")

    without_detector = copy.deepcopy(config)
    without_detector["c1"].pop("detector", None)
    defaulted = paper_c1.delivery_args(
        without_detector, "bfcl_base", tmp_path / "defaulted", ["multi_turn_base_0"], delivery
    )
    assert defaulted.detector == "d3_hybrid"

    for detector in ("legacy_prefill", "t02_risk"):
        legacy = copy.deepcopy(config)
        legacy["c1"]["detector"] = detector
        parsed = paper_c1.delivery_args(
            legacy, "bfcl_base", tmp_path / detector, ["multi_turn_base_0"], delivery
        )
        assert parsed.detector == detector


def test_append_final_arm_preserves_old_cells_and_completed_artifacts():
    config = json.loads(DEFAULT_CONFIG.read_text())
    previous = copy.deepcopy(config)
    previous["methods"] = previous["methods"][:-2]   # drop the C1 system and its ratio-4 ablation
    previous.pop("c1")
    with tempfile.TemporaryDirectory() as temporary:
        output = Path(temporary)
        old_plan, _ = prepare(previous, output, output / "sglang")
        completed = output / "closed_loop" / old_plan[0]["cell_id"] / "complete.json"
        completed.parent.mkdir(parents=True)
        completed.write_text('{"old_result": true}\n')
        new_plan, _ = prepare(config, output, output / "sglang")
        assert [row["cell_id"] for row in new_plan[:len(old_plan)]] == [row["cell_id"] for row in old_plan]
        assert completed.read_text() == '{"old_result": true}\n'
        archived = sorted(output.glob("config.before_extension.*.json"))
        assert archived and json.loads(archived[-1].read_text())["methods"] == previous["methods"]
        assert all("benchmarks.paper.c1" in row["command"] for row in new_plan[-4:])
        r4 = next(row for row in new_plan if row["arm"] == "c2kv_c1_t02_r4")
        command = list(r4["command"])
        assert r4["benchmark"] == "bfcl_base"
        assert command[command.index("--arm") + 1] == "c2kv_c1_t02_r4"
        assert "--enable-return-hidden-states" in server_command(config, Path("sglang"), "c2kv_c1_t02_r4")
        # An algorithm change to a previous arm cannot masquerade as extension.
        changed = copy.deepcopy(config)
        changed["methods"][3]["ratio"] = 8
        with pytest.raises(ValueError):
            prepare(changed, output, output / "sglang")


def test_task_selection_uses_the_official_file_and_rejects_foreign_ids():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        data = root / "bfcl_eval" / "data"
        data.mkdir(parents=True)
        (data / "BFCL_v4_multi_turn_base.json").write_text(
            '{"id":"multi_turn_base_1"}\n{"id":"multi_turn_base_2"}\n')
        config = {"bfcl_dir": str(root)}
        assert selected_tasks(config, "bfcl_base") == ["multi_turn_base_1", "multi_turn_base_2"]
        assert selected_tasks(config, "bfcl_base", ["multi_turn_base_2"]) == ["multi_turn_base_2"]
        with pytest.raises(ValueError):
            selected_tasks(config, "bfcl_base", ["multi_turn_base_99"])


def test_replay_keeps_official_task_identity_for_native_evidence_ids():
    rows = [{"replay_payload": {"c2kv_measurement_session_id": "multi_turn_base_26"}}] * 2
    assert replay_task_id(rows, "hashed-proxy-conversation") == "multi_turn_base_26"
    assert replay_task_id([{"replay_payload": {}}], "synthetic").startswith("replay_")


def test_controller_oom_is_a_scored_zero_harness_failure(tmp_path):
    shard = tmp_path / "task_shards" / "multi_turn_long_context_100"
    (shard / "server").mkdir(parents=True)
    assert controller_oom_message(shard) is None
    rows = [
        {"status": "completed", "error": None},
        {"status": "failed", "error": "{'type': 'OutOfMemoryError', "
                                      "'message': 'CUDA out of memory. Tried to allocate 6.00 GiB'}"},
    ]
    (shard / "server" / "steps.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    message = controller_oom_message(shard)
    assert message and "OutOfMemoryError" in message
    receipts = [
        {"task_id": "a", "status": "completed", "unified_metrics": {"official_score": 1.0}},
        {"task_id": "multi_turn_long_context_100", "status": "harness_failure",
         "failure": {"kind": "cuda_oom", "message": message},
         "unified_metrics": {"official_score": 0.0, "harness_failure": "cuda_oom"}},
    ]
    summary = summarize_scores("bfcl_long_context", receipts)
    assert summary["n"] == 2 and summary["semantic_score"] == 0.5
    assert summary["n_harness_failures"] == 1
    assert summary["harness_failure_task_ids"] == ["multi_turn_long_context_100"]
    assert summary["n_method_failures"] == 0


def capacity_evidence(shard, *, session_task=None):
    task = shard.name
    server = shard / "server"
    server.mkdir(parents=True)
    (server / "ready.json").write_text(json.dumps({
        "schema": "a-event-native-server-v1", "status": "ready", "benchmark": "bfcl",
        "allowed_task_ids": [task],
    }))
    row = {
        "schema": "a-event-native-exact-step-v1", "status": "failed",
        "session_id": f"bfcl/{session_task or task}/attempt-0",
        "failure_kind": "method_failure", "failure_code": "c2kv_capacity_infeasible",
        "error": {"type": "CapacityInfeasible", "message":
                  "Native S0 mandatory raw input and minimum whole-event gist cannot fit"},
    }
    (server / "steps.jsonl").write_text(json.dumps(row) + "\n")
    return row


def test_capacity_infeasible_is_a_scored_zero_method_failure(tmp_path):
    shard = tmp_path / "task_shards" / "multi_turn_long_context_101"
    capacity_evidence(shard)
    status, kind, message = controller_step_failure(shard)
    assert (status, kind) == ("method_failure", "capacity_infeasible")
    assert controller_oom_message(shard) is None
    receipts = [{"task_id": "multi_turn_long_context_101", "status": status,
                 "failure": {"kind": kind, "message": message},
                 "unified_metrics": {"official_score": 0.0, status: kind}}]
    summary = summarize_scores("bfcl_long_context", receipts)
    assert summary["semantic_score"] == 0.0 and summary["n_method_failures"] == 1
    assert summary["method_failure_task_ids"] == ["multi_turn_long_context_101"]
    assert summary["n_harness_failures"] == 0


def test_earlier_declared_failure_does_not_classify_a_later_step(tmp_path):
    shard = tmp_path / "task_shards" / "multi_turn_base_164"
    failed = capacity_evidence(shard)
    with (shard / "server" / "steps.jsonl").open("a") as handle:
        handle.write(json.dumps({"status": "completed", "error": None}) + "\n")
    assert failed["status"] == "failed"
    assert controller_step_failure(shard) is None


@pytest.mark.parametrize("change", ["transport", "transport_with_stale_code", "legacy_text",
                                    "missing_code", "other_session", "other_ready_task", "runner_failed"])
def test_capacity_marker_needs_typed_task_bound_failure(tmp_path, change):
    shard = tmp_path / "task_shards" / "multi_turn_base_164"
    row = capacity_evidence(shard)
    ready_path = shard / "server" / "ready.json"
    if change in {"transport", "transport_with_stale_code"}:
        if change == "transport":
            row.pop("failure_kind")
            row.pop("failure_code")
        row["error"] = {"type": "SGLangTransportError", "message":
                        "SGLang native_generate returned HTTP 502: upstream said CapacityInfeasible"}
    elif change == "legacy_text":
        row["error"] = "{'type': 'CapacityInfeasible', 'message': 'cannot fit'}"
    elif change == "missing_code":
        row.pop("failure_code")
    elif change == "other_session":
        row["session_id"] = "bfcl/multi_turn_base_165/attempt-0"
    elif change == "other_ready_task":
        ready = json.loads(ready_path.read_text())
        ready["allowed_task_ids"] = ["multi_turn_base_165"]
        ready_path.write_text(json.dumps(ready))
    else:
        row["failure_kind"] = "runner_failed"
    (shard / "server" / "steps.jsonl").write_text(json.dumps(row) + "\n")
    assert controller_step_failure(shard) is None


@pytest.mark.parametrize("session_task", ["multi_turn_base_164", "multi_turn_base_165"])
def test_prefix_replay_only_tolerates_capacity_for_its_task(tmp_path, monkeypatch, session_task):
    task = "multi_turn_base_164"
    payload = {"messages": [{"role": "user", "content": "fixture"}],
               "temperature": 0.001,
               "c2kv_measurement_session_id": task}
    prefix = {"event_type": "recorded_prefix", "source_arm": "full",
              "conversation_id": "fixture", "prefix_id": "prefix-1",
              "replay_payload": payload, "canonical_sha256": canonical_sha256(payload)}
    prefix_path = tmp_path / "prefix.jsonl"
    prefix_path.write_text(json.dumps(prefix) + "\n")
    native = tmp_path / "run" / "native"
    shard = native / "task_shards" / task
    capacity_evidence(shard, session_task=session_task)
    stopped = []
    delivery = SimpleNamespace(
        commands_for_task=lambda *_args: (["python", "--model-name", "fixture"], []),
        runner=SimpleNamespace(_stop_server=lambda *_args: stopped.append(True)),
    )
    monkeypatch.setattr(paper_c1, "load_delivery", lambda: delivery)
    monkeypatch.setattr(paper_c1, "prepare_native", lambda *_args: (native, object(), tmp_path / "controller.json"))
    monkeypatch.setattr(paper_c1, "_controller_process", lambda *_args: (object(), io.StringIO(), shard))

    class FailingOpener:
        def open(self, *_args, **_kwargs):
            raise urllib.error.HTTPError("fixture", 422, "capacity", {}, io.BytesIO(b"capacity"))

    monkeypatch.setattr(paper_c1, "OPENER", FailingOpener())
    output = tmp_path / "run"
    if session_task != task:
        with pytest.raises(RuntimeError, match="C1 replay HTTP 422"):
            paper_c1.run_common_prefix({"proxy_port": 49001, "c1": {"task_timeout": 1}},
                                       "bfcl_base", output, prefix_path)
        assert not (output / "prefix_replay.jsonl").exists()
    else:
        assert paper_c1.run_common_prefix({"proxy_port": 49001, "c1": {"task_timeout": 1}},
                                          "bfcl_base", output, prefix_path) == native
        rows = list(read_jsonl(output / "prefix_replay.jsonl"))
        assert len(rows) == 1
        assert rows[0]["failure"]["kind"] == "capacity_infeasible"
        assert rows[0]["native_task_id"] == task
        assert rows[0]["replay_attempted"] is True
        assert rows[0]["sampling_contract"]["source"]["temperature"] == 0.001
        assert rows[0]["sampling_contract"]["target"]["temperature"] == 0.0
        summary = json.loads((output / "replay_summary.json").read_text())
        assert summary["declared_failure_tasks"] == 1
        assert summary["attempted_prefixes"] == 1
        assert summary["not_attempted_after_declared_failure"] == 0
    assert stopped == [True]


def test_replay_distinguishes_failed_request_from_later_unattempted_prefix(tmp_path, monkeypatch):
    task = "multi_turn_base_164"
    payload = {"messages": [{"role": "user", "content": "fixture"}],
               "temperature": 0.001, "c2kv_measurement_session_id": task}
    prefixes = [{"event_type": "recorded_prefix", "source_arm": "full",
                 "conversation_id": task, "prefix_id": f"prefix-{index}",
                 "replay_payload": payload, "canonical_sha256": canonical_sha256(payload)}
                for index in range(2)]
    prefix_path = tmp_path / "prefixes.jsonl"
    prefix_path.write_text("".join(json.dumps(row) + "\n" for row in prefixes))
    native = tmp_path / "run" / "native"
    shard = native / "task_shards" / task
    capacity_evidence(shard)
    delivery = SimpleNamespace(
        commands_for_task=lambda *_args: (["python", "--model-name", "fixture"], []),
        runner=SimpleNamespace(_stop_server=lambda *_args: None),
    )
    monkeypatch.setattr(paper_c1, "load_delivery", lambda: delivery)
    monkeypatch.setattr(paper_c1, "prepare_native", lambda *_args: (native, object(), tmp_path / "controller.json"))
    monkeypatch.setattr(paper_c1, "_controller_process", lambda *_args: (object(), io.StringIO(), shard))

    class FailingOpener:
        def open(self, *_args, **_kwargs):
            raise urllib.error.HTTPError("fixture", 422, "capacity", {}, io.BytesIO(b"capacity"))

    monkeypatch.setattr(paper_c1, "OPENER", FailingOpener())
    output = tmp_path / "run"
    paper_c1.run_common_prefix({"proxy_port": 49001, "c1": {"task_timeout": 1}},
                               "bfcl_base", output, prefix_path)
    rows = list(read_jsonl(output / "prefix_replay.jsonl"))
    assert [row["replay_attempted"] for row in rows] == [True, False]
    summary = json.loads((output / "replay_summary.json").read_text())
    assert summary["failed"] == 2
    assert summary["declared_failure_tasks"] == 1
    assert summary["attempted_prefixes"] == 1
    assert summary["not_attempted_after_declared_failure"] == 1


def test_ratio4_ablation_binds_arm_and_ratio_for_summaries():
    try:
        assert select_arm("c2kv_c1_t02_r4") == ("c2kv_c1_t02_r4", 4)
        summary = summarize_scores("bfcl_base", [
            {"task_id": "a", "status": "completed", "unified_metrics": {"official_score": 1.0}}])
        assert (summary["arm"], summary["ratio"]) == ("c2kv_c1_t02_r4", 4)
        with pytest.raises(ValueError):
            select_arm("c2kv_c1_t02_r16")
    finally:
        select_arm("c2kv_c1_t02_r8")
    assert (paper_c1.ARM, paper_c1.RATIO) == ("c2kv_c1_t02_r8", 8)
