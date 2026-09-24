import copy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import urllib.error

import pytest

from benchmarks.arms import get_arm
from benchmarks.paper import c1 as paper_c1, native_extra
from benchmarks.paper.c1 import (ARM, controller_oom_message, controller_step_failure, replay_task_id,
                                 selected_tasks, select_arm, summarize_scores)
from benchmarks.paper.runner import DEFAULT_CONFIG, prepare, resolve_history_kv_budgets, server_command
from benchmarks.measurement.telemetry import canonical_sha256, read_jsonl


def test_native_arm_requires_real_controller_and_keeps_bare_c2kv():
    assert get_arm(ARM).native_controller == "c1_t02"
    assert get_arm(ARM).ratio == 8
    assert get_arm("c2kv4").ratio == 4
    config = dict(json.loads(DEFAULT_CONFIG.read_text()), history_kv_budget_tokens=768)
    command = server_command(config, Path("sglang"), ARM)
    assert "--enable-return-hidden-states" in command
    assert "--c2kv-shadow-feature-layer" in command
    assert "--disable-cuda-graph" not in command
    assert "--enable-return-hidden-states" not in server_command(config, Path("sglang"), "full")


def test_paper_delivery_uses_configured_detector_and_defaults_to_d3_hybrid(tmp_path):
    config = dict(json.loads(DEFAULT_CONFIG.read_text()), history_kv_budget_tokens=768)
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
        assert parsed.embedding_batch_size == 1

    legacy["c1"]["embedding_batch_size"] = 2
    parsed = paper_c1.delivery_args(
        legacy, "bfcl_base", tmp_path / "batch2", ["multi_turn_base_0"], delivery
    )
    assert parsed.embedding_batch_size == 2


def test_append_final_arm_preserves_old_cells_and_completed_artifacts():
    config = dict(json.loads(DEFAULT_CONFIG.read_text()), history_kv_budget_tokens=768)
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
        assert archived and json.loads(archived[-1].read_text())["methods"] == (
            resolve_history_kv_budgets(previous)["methods"])
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
    ace = [{"ace_official_task_id": "agent_multi_turn_1",
            "replay_payload": {"c2kv_measurement_session_id": "acebench:multi_turn_1:abcdef"}}]
    assert replay_task_id(ace, "ace-session") == "agent_multi_turn_1"
    with pytest.raises(ValueError, match="consistent official task ID"):
        replay_task_id([*ace, {"replay_payload": ace[0]["replay_payload"]}], "ace-session")
    with pytest.raises(ValueError, match="consistent official task ID"):
        replay_task_id([{"ace_official_task_id": "../other", "replay_payload": {}}], "ace-session")


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


def capacity_evidence(shard, *, session_task=None, benchmark="bfcl"):
    task = shard.name
    server = shard / "server"
    server.mkdir(parents=True)
    (server / "ready.json").write_text(json.dumps({
        "schema": "a-event-native-server-v1", "status": "ready", "benchmark": benchmark,
        "allowed_task_ids": [task],
    }))
    row = {
        "schema": "a-acebench-event-step-v1" if benchmark == "acebench" else "a-event-native-exact-step-v1", "status": "failed",
        "session_id": f"{benchmark}/{session_task or task}/attempt-0",
        "failure_kind": "method_failure", "failure_code": "c2kv_capacity_infeasible",
        "error": {"type": "CapacityInfeasible", "message":
                  "Native S0 mandatory raw input and minimum whole-event gist cannot fit"},
    }
    (server / "steps.jsonl").write_text(json.dumps(row) + "\n")
    return row


def generation_cap_evidence(shard, *, session_task=None):
    task = shard.name
    server = shard / "server"
    server.mkdir(parents=True)
    (server / "ready.json").write_text(json.dumps({
        "schema": "a-event-native-server-v1", "status": "ready", "benchmark": "tau2",
        "allowed_task_ids": [task],
    }))
    row = {
        "schema": "a-event-native-exact-step-v1", "status": "failed",
        "session_id": f"tau2/{session_task or task}/attempt-0",
        "failure_kind": "budget_exhausted", "failure_code": "generation_cap_reached",
        "error": {"type": "GenerationCallCapExceeded",
                  "message": "Finite generation-call cap exhausted before submission"},
    }
    (server / "steps.jsonl").write_text(json.dumps(row) + "\n")
    (server / "final.json").write_text(json.dumps({
        "status": "stopped", "journal_summary": {"completed": 96, "failed": 0, "pending": 0},
        "api_health": {"terminal_reason": "generation_cap_reached"},
    }) + "\n")
    return row


@pytest.mark.parametrize("change", [
    None, "other_session", "unknown_error", "not_failed", "final_pending",
    "final_failed", "final_status_failed", "final_missing", "cost_summary_error", "terminal_error",
])
def test_tau2_generation_cap_is_task_local_only_for_bound_typed_failure(
        tmp_path, monkeypatch, change):
    cell = tmp_path / "cell"
    native = cell / "native"
    native.mkdir(parents=True)
    calls = []
    monkeypatch.setattr(paper_c1, "load_delivery", lambda: object())
    monkeypatch.setattr(paper_c1, "selected_tasks", lambda *_args: ["5", "6"])
    monkeypatch.setattr(paper_c1, "prepare_native",
                        lambda *_args: (native, None, tmp_path / "controller.json"))

    def run_task(_config, _benchmark, task, _native, _delivery, _controller):
        calls.append(task)
        shard = native / "task_shards" / task
        shard.mkdir(parents=True)
        if task == "6":
            metrics = {"task_id": task, "official_score": 1.0, "normal_termination": True}
            return {"task_id": task, "status": "completed", "unified_metrics": metrics}, metrics
        row = generation_cap_evidence(shard)
        if change == "other_session":
            row["session_id"] = "tau2/another-task/attempt-0"
        elif change == "unknown_error":
            row.pop("failure_kind")
            row.pop("failure_code")
            row["error"] = {"type": "RuntimeError", "message": "unknown runtime failure"}
        elif change == "not_failed":
            row["status"] = "ok"
        (shard / "server" / "steps.jsonl").write_text(json.dumps(row) + "\n")
        final_path = shard / "server" / "final.json"
        if change == "final_missing":
            final_path.unlink()
        elif change in {"final_pending", "final_failed", "final_status_failed",
                        "cost_summary_error", "terminal_error"}:
            final = json.loads(final_path.read_text())
            if change == "final_pending":
                final["journal_summary"]["pending"] = 1
            elif change == "final_failed":
                final["journal_summary"]["failed"] = 1
            elif change == "terminal_error":
                final["api_health"]["terminal_reason"] = "budget_rejection_write_failed"
            elif change == "final_status_failed":
                final["status"] = "failed"
            else:
                final["cost_summary_error"] = "cost aggregation failed"
            final_path.write_text(json.dumps(final) + "\n")
        raise RuntimeError("tau2 task 5 ended with infrastructure_error")

    monkeypatch.setattr(native_extra, "run_task", run_task)
    if change is not None:
        with pytest.raises(RuntimeError, match="infrastructure_error"):
            paper_c1.run_closed_loop({}, "tau2", cell)
        assert calls == ["5"]
        assert not (native / "task_shards" / "5" / "paper_task_result.json").exists()
        assert not (native / "task_shards" / "6").exists()
        return

    assert paper_c1.run_closed_loop({}, "tau2", cell) == native
    assert calls == ["5", "6"]
    capped = json.loads((native / "task_shards" / "5" / "paper_task_result.json").read_text())
    completed = json.loads((native / "task_shards" / "6" / "paper_task_result.json").read_text())
    assert capped["status"] == "method_failure"
    assert capped["failure"]["kind"] == "generation_cap_reached"
    assert capped["unified_metrics"]["official_score"] == 0.0
    assert "not an official reward" in capped["qualification"]
    assert completed["status"] == "completed"
    summary = json.loads((cell / f"summary_{paper_c1.ARM}.json").read_text())
    assert summary["n"] == 2 and summary["semantic_score"] == 0.5
    assert summary["n_method_failures"] == 1
    assert summary["method_failure_task_ids"] == ["5"]
    assert json.loads((native / "result.json").read_text())["status"] == "completed"


def test_unmatched_failed_attempt_still_scores_its_typed_step_zero(tmp_path, monkeypatch):
    """1b32ffc regression: a failed attempt that is no capacity fallback raised
    ValueError past the driver's RuntimeError fallback and aborted the cell."""
    delivery = paper_c1.load_delivery()
    from benchmarks.memory_runtime.attempt_journal import AttemptJournal, summarize_attempt_journal

    cell = tmp_path / "cell"
    native = cell / "native"
    native.mkdir(parents=True)
    calls = []
    monkeypatch.setattr(paper_c1, "load_delivery", lambda: delivery)
    monkeypatch.setattr(paper_c1, "selected_tasks", lambda *_args: ["5", "6"])
    monkeypatch.setattr(paper_c1, "prepare_native",
                        lambda *_args: (native, None, tmp_path / "controller.json"))

    def run_task(_config, _benchmark, task, _native, _delivery, _controller):
        calls.append(task)
        shard = native / "task_shards" / task
        shard.mkdir(parents=True)
        if task == "6":
            metrics = {"task_id": task, "official_score": 1.0, "normal_termination": True}
            return {"task_id": task, "status": "completed", "unified_metrics": metrics}, metrics
        capacity_evidence(shard, benchmark="tau2")
        journal = AttemptJournal(shard / "server" / "attempts.jsonl")
        context = {"task_id": task, "decision_id": "turn-0/step-0"}
        journal.finish(journal.start("generation", 1, "draft", context), "completed")
        journal.finish(journal.start("generation", 2, "draft", context), "failed")
        summary = summarize_attempt_journal(shard / "server" / "attempts.jsonl")
        final = {"status": "stopped", "cost_summary": {"generation_calls": 2},
                 "journal_summary": {key: summary[key] for key in (
                     "schema", "started", "finished", "completed", "failed", "pending",
                     "truncated_tail")}}
        (shard / "server" / "final.json").write_text(json.dumps(final) + "\n")
        with pytest.raises(ValueError, match="Unexpected failed model attempt"):
            delivery.validate_handled_capacity_failures(final, shard / "server")
        # The finalization sites use this wrapper: no capacity fallback, historical error.
        delivery.handled_capacity_failures(final, shard / "server")
        raise AssertionError("an unmatched failed attempt must not be accepted")

    monkeypatch.setattr(native_extra, "run_task", run_task)
    assert paper_c1.run_closed_loop({}, "tau2", cell) == native
    assert calls == ["5", "6"]
    failed = json.loads((native / "task_shards" / "5" / "paper_task_result.json").read_text())
    assert (failed["status"], failed["failure"]["kind"]) == ("method_failure", "capacity_infeasible")
    assert "outside a safe capacity fallback" in failed["failure"]["error"]
    assert json.loads((native / "task_shards" / "6" / "paper_task_result.json").read_text())[
        "status"] == "completed"
    assert delivery.handled_capacity_failures(
        {"journal_summary": {"failed": 0}}, tmp_path / "unused") == 0


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


def test_ace_capacity_failure_uses_its_runtime_schema_and_task_identity(tmp_path):
    shard = tmp_path / "task_shards" / "agent_multi_step_0"
    row = capacity_evidence(shard, benchmark="acebench")
    assert controller_step_failure(shard)[:2] == ("method_failure", "capacity_infeasible")
    row["session_id"] = "acebench/agent_multi_step_1/attempt-0"
    (shard / "server" / "steps.jsonl").write_text(json.dumps(row) + "\n")
    assert controller_step_failure(shard) is None


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


def test_ace_replay_uses_recorded_official_id_and_checks_loaded_controller(tmp_path, monkeypatch):
    task = "agent_multi_turn_1"
    payload = {
        "messages": [{"role": "system", "content": "Visible APIs"},
                     {"role": "user", "content": "Use one API"}],
        "temperature": 0.001, "top_p": 1, "max_tokens": 1000,
        "c2kv_measurement_session_id": "acebench:multi_turn_1:abcdef",
        "c2kv_ace_source": {"version": "acebench-text-actions-v1", "receipts": []},
    }
    prefix = {"event_type": "recorded_prefix", "source_arm": "full",
              "ace_official_task_id": task, "conversation_id": "recorded-conversation",
              "prefix_id": "prefix-1", "replay_payload": payload,
              "canonical_sha256": canonical_sha256(payload)}
    prefix_path = tmp_path / "prefix.jsonl"
    prefix_path.write_text(json.dumps(prefix) + "\n", encoding="utf-8")
    output = tmp_path / "run"
    output.mkdir()
    shard = output / "native" / "task_shards" / task
    shard.mkdir(parents=True)
    (shard / "server").mkdir()
    (shard / "server" / "final.json").write_text(json.dumps({
        "status": "stopped", "journal_summary": {"completed": 1, "failed": 0, "pending": 0},
        "cost_summary": {"generation_calls": 1},
    }))
    observed = {}
    delivery = SimpleNamespace(runner=SimpleNamespace(_stop_server=lambda *_: None))
    monkeypatch.setattr(paper_c1, "load_delivery", lambda: delivery)
    monkeypatch.setattr(paper_c1, "prepare_native",
                        lambda *_: (output / "native", object(), tmp_path / "controller.json"))
    monkeypatch.setattr(paper_c1, "_controller_process",
                        lambda *_: (object(), io.StringIO(), shard))
    monkeypatch.setattr(native_extra, "controller_command",
                        lambda *_: ["python", "--model-name", "c1_d3_hybrid"])
    monkeypatch.setattr(native_extra, "validate_ready_manifest",
                        lambda *args: observed.setdefault("ready", args))

    class Opener:
        def open(self, request, **_kwargs):
            observed["payload"] = json.loads(request.data)
            return io.StringIO('{"id":"replayed-step"}')

    monkeypatch.setattr(paper_c1, "OPENER", Opener())
    paper_c1.run_common_prefix({"proxy_port": 49001, "c1": {"task_timeout": 1}},
                               "acebench_agent", output, prefix_path)
    assert observed["ready"][2] == task
    assert observed["ready"][3] == shard / "server" / "ready.json"
    assert observed["payload"]["c2kv_eval_context"]["task_id"] == task
    assert observed["payload"]["c2kv_ace_source"] == payload["c2kv_ace_source"]
    assert list(read_jsonl(output / "prefix_replay.jsonl"))[0]["native_task_id"] == task


@pytest.mark.parametrize("declared_failure", [False, True])
def test_replay_rejects_cost_failure_even_after_successful_responses(tmp_path, declared_failure):
    (tmp_path / "server").mkdir()
    (tmp_path / "server" / "final.json").write_text(json.dumps({
        "status": "failed", "journal_summary": {"completed": 2, "failed": 0, "pending": 0},
        "cost_summary_error": {"type": "ValueError", "message": "prefix bytes mismatch"},
    }))
    with pytest.raises(RuntimeError, match="cost finalization failed"):
        paper_c1.validate_replay_finalization(tmp_path, declared_failure=declared_failure)


def test_replay_requires_complete_controller_journal(tmp_path):
    (tmp_path / "server").mkdir()
    path = tmp_path / "server" / "final.json"
    final = {"status": "stopped", "journal_summary": {"completed": 1, "pending": 1},
             "cost_summary": {"generation_calls": 1}}
    path.write_text(json.dumps(final))
    with pytest.raises(RuntimeError, match="controller finalization failed"):
        paper_c1.validate_replay_finalization(tmp_path)
    final["journal_summary"]["pending"] = 0
    path.write_text(json.dumps(final))
    paper_c1.validate_replay_finalization(tmp_path)


def test_replay_accepts_only_validated_capacity_fallback(tmp_path):
    server = tmp_path / "server"
    server.mkdir()
    (server / "final.json").write_text(json.dumps({
        "status": "stopped", "journal_summary": {"completed": 1, "failed": 1, "pending": 0},
        "cost_summary": {"generation_calls": 2}}))
    seen = []
    delivery = SimpleNamespace(handled_capacity_failures=lambda final, path: (
        seen.append((final["journal_summary"]["failed"], path)) or 1))
    paper_c1.validate_replay_finalization(tmp_path, delivery=delivery)
    assert seen == [(1, server)]


def test_closed_loop_does_not_mask_cost_failure_as_capacity_failure(tmp_path, monkeypatch):
    task = "multi_turn_base_164"
    native = tmp_path / "native"
    shard = native / "task_shards" / task

    def failed_task(*_args, **_kwargs):
        capacity_evidence(shard)
        (shard / "server" / "final.json").write_text(json.dumps({
            "status": "failed", "cost_summary_error": {"message": "prefix bytes mismatch"},
        }))
        raise RuntimeError("invalid cost inventory")

    delivery = SimpleNamespace(run_task=failed_task)
    monkeypatch.setattr(paper_c1, "load_delivery", lambda: delivery)
    monkeypatch.setattr(paper_c1, "selected_tasks", lambda *_: [task])
    monkeypatch.setattr(paper_c1, "prepare_native",
                        lambda *_: (native, object(), tmp_path / "controller.json"))
    with pytest.raises(RuntimeError, match="invalid cost inventory"):
        paper_c1.run_closed_loop({}, "bfcl_base", tmp_path)
    assert not (shard / "paper_task_result.json").exists()


@pytest.mark.parametrize("captured", [None, False, [{"message_index": 0, "start": 0,
                                                     "end": 10_000, "source": "acebench_function_list"}]])
def test_ace_tool_replay_rejects_missing_or_invalid_capture_before_server(
        tmp_path, monkeypatch, captured):
    payload = {"messages": [{"role": "system", "content": "Visible API"},
                            {"role": "user", "content": "Use it"}],
               "c2kv_measurement_session_id": "acebench:task:session",
               "c2kv_ace_source": {"version": "acebench-text-actions-v1", "receipts": []}}
    if captured is not None:
        payload["c2kv_tool_spans_v1"] = captured
    prefix = {"event_type": "recorded_prefix", "source_arm": "full",
              "ace_official_task_id": "agent_task_1", "conversation_id": "conversation",
              "replay_payload": payload, "canonical_sha256": canonical_sha256(payload)}
    path = tmp_path / "prefix.jsonl"
    path.write_text(json.dumps(prefix) + "\n", encoding="utf-8")
    monkeypatch.setattr(paper_c1, "load_delivery",
                        lambda: pytest.fail("server setup must not start"))
    with pytest.raises(ValueError, match="tool replay"):
        paper_c1.run_common_prefix({"tool_memory": "t0:r8"},
                                   "acebench_agent", tmp_path, path)


def test_ace_tool_replay_accepts_explicit_empty_capture_before_server(tmp_path, monkeypatch):
    payload = {"messages": [{"role": "user", "content": "Proceed"}],
               "c2kv_tool_spans_v1": [],
               "c2kv_measurement_session_id": "acebench:task:session"}
    prefix = {"event_type": "recorded_prefix", "source_arm": "full",
              "ace_official_task_id": "agent_task_1", "conversation_id": "conversation",
              "replay_payload": payload, "canonical_sha256": canonical_sha256(payload)}
    path = tmp_path / "prefix.jsonl"
    path.write_text(json.dumps(prefix) + "\n", encoding="utf-8")

    class PassedCapture(Exception):
        pass

    monkeypatch.setattr(paper_c1, "load_delivery", lambda: (_ for _ in ()).throw(PassedCapture()))
    with pytest.raises(PassedCapture):
        paper_c1.run_common_prefix({"tool_memory": "t0:r8"},
                                   "acebench_agent", tmp_path, path)


@pytest.mark.parametrize("alias", ["none", "raw", "full"])
def test_explicit_off_tool_alias_restores_unmodified_native_config(alias):
    config = {"native_arm": "c2kv_c1_t02_r4", "tool_memory": "t0:r8",
              "tool_checkpoint": "/old/T0", "tool_budget_tokens": 256}
    paper_c1.apply_tool_cli(config, alias, "", None)
    assert config == {"native_arm": "c2kv_c1_t02_r4"}
    with pytest.raises(ValueError, match="active --tool-memory"):
        paper_c1.apply_tool_cli(config, alias, "/orphan/T0", None)


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
