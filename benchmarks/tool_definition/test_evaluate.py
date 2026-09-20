from __future__ import annotations

import hashlib
import json
import pytest
from types import SimpleNamespace

from benchmarks.tool_definition import evaluate as module
from benchmarks.tool_definition.evaluate import (_http_result, _measured_kv,
                                                 _outcome_http)


def test_http_outcome_uses_ordered_names_arguments_and_no_call_denominator():
    gold = [{"name": "search", "arguments": {"q": "a"}},
            {"name": "open", "arguments": {"id": 1}}]
    calls = [{"function": {"name": "search", "arguments": '{"q":"a"}'}},
             {"function": {"name": "open", "arguments": '{"id":1}'}}]
    assert _outcome_http(gold, None, calls)["strict_ordered_call_correct"]
    assert not _outcome_http(gold, None, calls[::-1])["strict_ordered_call_correct"]
    assert _outcome_http([], None, calls)["false_tool_call"]
    assert not _outcome_http([], "plain answer", None)["false_tool_call"]


def test_http_requires_server_generation_measurement_and_raw_receipt():
    normalized = {"cost": {"server_measurement": {
        "generation_active_kv_tokens": 60,
        "generation_active_kv_bytes": 1200,
    }}}
    with pytest.raises(ValueError, match="tool-KV receipt"):
        _measured_kv({}, normalized, "h2o", 60)
    with pytest.raises(ValueError, match="generation-start"):
        _measured_kv({}, {"cost": {}}, "c2kv", None)
    response = {"metadata": {"kv_memory_report": {"tool_kv_eviction": {
        "success": True, "method": "h2o", "history_untouched": True,
        "first_token_after_selection": True, "resident_tokens_by_layer": [60, 57],
        "logical_kv_bytes": 1000, "target_resident_tokens_per_layer": 60,
    }}}}
    assert _measured_kv(response, normalized, "h2o", 60)["resident_kv_tokens_by_layer"] == [60, 57]


def test_full_equivalent_no_op_requires_adequate_allowance_and_full_residency():
    normalized = {"cost": {"server_measurement": {
        "generation_active_kv_tokens": 100, "generation_active_kv_bytes": 2000}}}
    receipt = {"success": True, "method": "h2o", "history_untouched": True,
               "no_op": True, "first_token_after_selection": False,
               "full_prompt_tokens": 100, "resident_tokens_by_layer": [100, 100],
               "logical_kv_bytes": 2000, "target_resident_tokens_per_layer": 100}
    response = {"metadata": {"kv_memory_report": {"tool_kv_eviction": receipt}}}
    assert _measured_kv(response, normalized, "h2o", 100)["full_equivalent_no_op"]
    record = {"decision_id": "d", "source": "recorded", "ratio": 8,
              "k": 1, "seed": 42, "prompt_sha256": "abc", "gold_tool_calls": [],
              "native_indices": [0], "resident_kv_tokens": 100,
              "base_prompt_tokens_without_tool_protocol": 30}
    result = _http_result(record, method="h2o", layout="hybrid", response=response,
                          normalized=normalized, plan=SimpleNamespace(info={"native_indices": [0]}),
                          target=100, full_active_tokens=100, full_active_bytes=2000)
    assert result["full_equivalent_no_op"] and result["R_tool"] == 1
    with pytest.raises(ValueError, match="full-prompt matched allowance"):
        _measured_kv(response, normalized, "h2o", 99)
    receipt["resident_tokens_by_layer"] = [100, 99]
    with pytest.raises(ValueError, match="Full prompt"):
        _measured_kv(response, normalized, "h2o", 100)


def test_http_result_uses_measured_full_anchor_and_flags_exceeded_budget():
    record = {"decision_id": "d", "source": "recorded", "ratio": 8,
              "k": 1, "seed": 42, "prompt_sha256": "abc", "gold_tool_calls": [],
              "native_indices": [0], "resident_kv_tokens": 60,
              "base_prompt_tokens_without_tool_protocol": 30}
    receipt = {"success": True, "method": "h2o", "history_untouched": True,
               "first_token_after_selection": True, "resident_tokens_by_layer": [67, 65],
               "logical_kv_bytes": 1300, "target_resident_tokens_per_layer": 60}
    response = {"metadata": {"kv_memory_report": {"tool_kv_eviction": receipt}}}
    normalized = {"content": "answer", "tool_calls": [], "cost": {"server_measurement": {
        "generation_active_kv_tokens": 67, "generation_active_kv_bytes": 1340}}}
    row = _http_result(record, method="h2o", layout="random", response=response,
                       normalized=normalized, plan=SimpleNamespace(info={"native_indices": [0]}),
                       target=60, full_active_tokens=100, full_active_bytes=2000)
    assert row["R_tool"] == 70 / 37
    assert row["budget_status"] == "exceeds_allowance"
    assert row["raw_response"] is response
    assert row["resident_kv_tokens_by_layer"] == [67, 65]


def test_evaluate_stages_full_history_through_shared_http_adapter(tmp_path, monkeypatch):
    messages = [{"role": "user", "content": "earlier turn"},
                {"role": "user", "content": "choose a tool"}]
    tools = [{"type": "function", "function": {"name": "search", "parameters": {}}},
             {"type": "function", "function": {"name": "open", "parameters": {}}}]
    source = {"decision_id": "d", "source": "recorded", "messages": messages,
              "tools": tools, "gold_tool_calls": [{"name": "search", "arguments": {}}]}
    fingerprint = hashlib.sha256(json.dumps({"messages": messages, "tools": tools},
        ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode("utf-8")).hexdigest()
    natives = {"full": [0, 1], "uniform": [], "hybrid": [0],
               "random": [1], "retrieval": [0]}
    costs = {"full": 100, "uniform": 55, "hybrid": 65,
             "random": 65, "retrieval": 65}
    pair = {layout: {"decision_id": "d", "source": "recorded", "ratio": 8,
                     "layout": layout, "k": 1, "seed": 42,
                     "prompt_sha256": fingerprint,
                     "gold_tool_calls": source["gold_tool_calls"],
                     "native_indices": native, "resident_kv_tokens": costs[layout],
                     "base_prompt_tokens_without_tool_protocol": 30}
            for layout, native in natives.items()}
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(module, "read_manifest", lambda *_: (
        {"checkpoint": {"config_sha256": "frozen-config",
                        "model_files_sha256": {"model.safetensors": "weight-hash"}}},
        {("d", 8): pair}))
    monkeypatch.setattr(module, "_source_rows", lambda *_: {"d": source})
    sent = []
    failures = {}
    reject_malformed_history = [False]

    class FakeBackend:
        extract_tokens = staticmethod(lambda *_: {})

        @staticmethod
        def normalize_response(response):
            return response["normalized"]

    class FakeClient:
        def __init__(self, upstream):
            self.backend = FakeBackend()

        @staticmethod
        def model_id(_requested):
            return "frozen-model"

        @staticmethod
        def _json(method, path):
            assert (method, path) == ("GET", "/model_info")
            return {"model_path": "remote/frozen-model", "c2kv_native_packed": {
                "tool_gist": {"enabled": True, "source": "remote/tool", "identity": "tool-id",
                              "config_sha256": "frozen-config"}}}

        @staticmethod
        def post_json(path, request):
            assert path == "/v1/chat/completions"
            sent.append(request)
            assert "gold_tool_calls" not in json.dumps(request)
            if reject_malformed_history[0] and any(
                call.get("function", {}).get("arguments") == "{"
                for message in request["messages"] for call in message.get("tool_calls", [])
            ):
                raise RuntimeError("SGLang /v1/chat/completions returned HTTP 400: malformed arguments")
            case = request["case"]
            method = case["method"]
            layout = case["layout"]
            if (method, layout) in failures:
                raise failures[(method, layout)]
            active = 67 if method == "h2o" and layout == "random" else costs[layout]
            response = {"metadata": {}, "normalized": {
                "content": None, "tool_calls": [{"function": {
                    "name": "search", "arguments": "{}"}}],
                "finish_reason": "tool_calls", "usage": {},
                "cost": {"server_measurement": {
                    "generation_active_kv_tokens": active,
                    "generation_active_kv_bytes": active * 20}}}}
            if method != "c2kv":
                response["metadata"]["kv_memory_report"] = {"tool_kv_eviction": {
                    "success": True, "method": method, "history_untouched": True,
                    "first_token_after_selection": True,
                    "resident_tokens_by_layer": [active, active],
                    "logical_kv_bytes": active * 20,
                    "target_resident_tokens_per_layer": case["target"]}}
            return response

    class FakeAdapter:
        def __init__(self, spec, _checkpoint, _extract):
            self.spec = spec

        def prepare_full_history_request(self, payload, *, native_override=None,
                                         retrieval_only=False,
                                         target_resident_tokens=None):
            native = (list(native_override) if native_override is not None else
                      [0] if self.spec.layout == "hybrid" else [])
            layout = ("full" if retrieval_only and len(native) == 2 else
                      "retrieval" if retrieval_only else
                      "random" if native_override is not None else self.spec.layout)
            staged = dict(payload)
            staged["messages"] = [{"role": "system", "content": "native protocol"}]
            if self.spec.encoder == "t0" and layout != "full":
                staged["messages"].append({"role": "user", "content": "carrier",
                                           "c2kv_key_hash": "hash"})
            staged["messages"].extend(payload["messages"])
            staged["case"] = {"method": "c2kv" if self.spec.encoder == "t0" else self.spec.encoder,
                              "layout": layout, "target": target_resident_tokens}
            return staged, SimpleNamespace(info={"native_indices": native})

    monkeypatch.setattr(module, "SglangClient", FakeClient)
    monkeypatch.setattr(module.toolmemory, "ToolMemory", FakeAdapter)
    failures[("h2o", "hybrid")] = RuntimeError("bad request")
    report = module.evaluate(manifest_path, tmp_path, tmp_path / "out",
                             upstream="http://localhost:30000", max_new_tokens=32,
                             methods=("c2kv", "h2o"))
    assert report["status"] == "completed_with_errors"
    assert report["result_rows"] == 7
    assert report["grouped_metrics"] == []
    errors = [json.loads(line) for line in (tmp_path / "out" / "errors.jsonl").read_text(
        encoding="utf-8").splitlines()]
    assert errors[0]["decision_id"] == "d" and errors[0]["layout"] == "hybrid"
    failures.clear()
    report = module.evaluate(manifest_path, tmp_path, tmp_path / "out",
                             upstream="http://localhost:30000", max_new_tokens=32,
                             methods=("c2kv", "h2o"), resume=True)
    rows = [json.loads(line) for line in (tmp_path / "out" / "results.jsonl").read_text(
        encoding="utf-8").splitlines()]
    assert report["status"] == "completed"
    assert report["result_rows"] == len(rows) == 8
    assert report["selected_unique_decisions"] == 1
    assert report["selected_decision_ratio_groups"] == 1
    assert set(report["client_source_sha256"]) == {
        "benchmarks/tool_definition/evaluate.py", "benchmarks/tool_definition/core.py",
        "benchmarks/tool_definition/prepare.py", "benchmarks/toolmemory.py",
        "benchmarks/backends/sglang.py"}
    assert "engine_source_revision_unverified" in report["server_identity_scope"]
    assert len(sent) == 9
    assert report["server_model_info"]["c2kv_native_packed"]["tool_gist"]["identity"] == "tool-id"
    assert all(row["outcome"]["strict_ordered_call_correct"] for row in rows)
    assert all(row["full_generation_active_kv_tokens"] == 100 for row in rows)
    assert next(row for row in rows if row["layout"] == "full")["R_tool"] == 1
    assert next(row for row in rows if row["method"] == "h2o" and row["layout"] == "random")[
        "budget_status"] == "exceeds_allowance"
    assert sent[0]["c2kv_kv_memory_hint"]["paper_measurement"]["history_message_count"] == 2
    assert sent[1]["c2kv_kv_memory_hint"]["paper_measurement"]["history_message_count"] == 3
    assert all(request["c2kv_kv_memory_hint"]["paper_measurement"][
        "canonical_source_messages"] == sent[0]["messages"] for request in sent)
    with pytest.raises(ValueError, match="run contract differs"):
        module.evaluate(manifest_path, tmp_path, tmp_path / "out",
                        upstream="http://localhost:30000", max_new_tokens=64,
                        methods=("c2kv", "h2o"), resume=True)
    failures[("h2o", "hybrid")] = module.UpstreamUnavailable("server exited")
    with pytest.raises(module.UpstreamUnavailable):
        module.evaluate(manifest_path, tmp_path, tmp_path / "interrupted",
                        upstream="http://localhost:30000", max_new_tokens=32,
                        methods=("c2kv", "h2o"))
    interrupted = json.loads((tmp_path / "interrupted" / "evaluation.json").read_text(
        encoding="utf-8"))
    assert interrupted["status"] == "interrupted"
    assert interrupted["result_rows"] == 6
    assert interrupted["unresolved_rows"] == 2
    assert interrupted["grouped_metrics"] == []
    failures.clear()
    resumed = module.evaluate(manifest_path, tmp_path, tmp_path / "interrupted",
                              upstream="http://localhost:30000", max_new_tokens=32,
                              methods=("c2kv", "h2o"), resume=True)
    assert resumed["status"] == "completed" and resumed["result_rows"] == 8

    reject_malformed_history[0] = True
    bad_source = {**source, "decision_id": "bad-history", "messages": [
        {"role": "assistant", "tool_calls": [{"function": {
            "name": "search", "arguments": "{"}}]}, *messages]}
    bad_fingerprint = hashlib.sha256(json.dumps({
        "messages": bad_source["messages"], "tools": tools}, ensure_ascii=False,
        sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()
    bad_pair = {layout: {**record, "decision_id": "bad-history",
                         "prompt_sha256": bad_fingerprint}
                for layout, record in pair.items()}
    monkeypatch.setattr(module, "read_manifest", lambda *_: (
        {"checkpoint": {"config_sha256": "frozen-config",
                        "model_files_sha256": {"model.safetensors": "weight-hash"}}},
        {("bad-history", 8): bad_pair, ("d", 8): pair}))
    monkeypatch.setattr(module, "_source_rows", lambda *_: {
        "bad-history": bad_source, "d": source})
    malformed = module.evaluate(manifest_path, tmp_path, tmp_path / "malformed",
                                upstream="http://localhost:30000", max_new_tokens=32,
                                methods=("c2kv", "h2o"))
    assert malformed["status"] == "completed_with_errors"
    assert malformed["result_rows"] == 8
    assert malformed["expected_rows"] == 16
    assert malformed["grouped_metrics"] == []
    bad_errors = [json.loads(line) for line in (tmp_path / "malformed" / "errors.jsonl").read_text(
        encoding="utf-8").splitlines()]
    assert len(bad_errors) == 1
    assert bad_errors[0]["decision_id"] == "bad-history"
    assert bad_errors[0]["phase"] == "full_anchor"
