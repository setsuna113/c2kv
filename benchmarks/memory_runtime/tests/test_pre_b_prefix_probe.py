"""Shared-packet and no-undeclared-generation checks for finite P1 views."""
import copy
import io
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "benchmarks"))
from memory_runtime import pre_b_prefix_probe as planner
from memory_runtime import pre_b_prefix_live as live


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return list(range(len(json.dumps(messages)) // 4 + 1))


def case(turns=2):
    messages = []
    for index in range(turns):
        messages.extend([dict(role="user", content=f"topic {index}"),
                         dict(role="assistant", content=f"answer {index}")])
    messages.append(dict(role="user", content="Continue."))
    return dict(case_id="case", eval_context=dict(task_id="t", attempt=0, decision_id="d"),
                request_view=dict(model="c2kv-agent", messages=messages, tools=[]))


def prepared():
    value = planner.build_views([case()], Tokenizer())
    value["freeze_digest"] = planner.digest(value)
    return value


def test_views_share_exact_e_and_e_only_only_removes_gist():
    value = prepared()
    live.validate(value)
    views = value["views"]
    full, fe, ga, e = [views["case:" + label]["payload"]["messages"]
                       for label in ("F", "FE", "G_all", "E_only")]
    packet_index = views["case:FE"]["counts"]["memory_runtime"]["evidence_out_index"]
    if packet_index is not None:
        assert fe[:packet_index] + fe[packet_index + 1:] == full
        ga_index = views["case:G_all"]["counts"]["memory_runtime"]["evidence_out_index"]
        assert fe[packet_index] == ga[ga_index]
    assert e == [m for m in ga if not m.get("c2kv_key_hash")]
    assert len(value["schedule"]) == 5
    assert all(item["placeholder"].startswith(planner.PLACEHOLDER) for item in value["extraction_manifest"])


def test_native_loss_marks_only_unsupported_view_and_restores_proxy_globals():
    before = planner.proxy.MEMORY_RUNTIME
    value = planner.build_views([case(14)], Tokenizer())
    assert planner.proxy.MEMORY_RUNTIME is before
    assert value["cases"][0]["unsupported_views"]["G_all"]["status"] == "unsupported_by_native_packing"
    assert "case:G_all" not in value["views"]
    assert all("case:" + label in value["views"] for label in ("F", "FE", "E_only", "G_B0"))


def test_payload_edit_is_rejected_before_any_transport(tmp_path, monkeypatch):
    value = prepared()
    value["views"]["case:F"]["payload"]["max_tokens"] = 8192
    monkeypatch.setattr(live, "build_opener", lambda *a: pytest.fail("Opened transport before frozen input validation"))
    with pytest.raises(ValueError, match="artifact changed"):
        live.execute(value, "http://unused", tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_bad_materialization_stops_before_generation_and_preserves_attempt(tmp_path, monkeypatch):
    value = prepared()
    requests = []
    identity = dict(model_path="/home/user/checkpoints_upstream/checkpoint-1088",
                    dtype="bfloat16", device="npu", context_length=16384, tp_size=1,
                    enable_c2kv=True, c2kv_query_proj="base", c2kv_tools_dump="full",
                    attention_backend="ascend", served_model_name="c2kv-agent")

    class Opener:
        def open(self, request, **kwargs):
            if isinstance(request, str):
                return io.BytesIO(json.dumps(identity).encode())
            requests.append(request.full_url)
            return io.BytesIO(json.dumps(dict(success=True, key_hash="real", gist_len=999,
                                              original_seq_len=999)).encode())

    monkeypatch.setattr(live, "build_opener", lambda *a: Opener())
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="extraction differs"):
        live.execute(value, "http://unused", out)
    assert requests == ["http://unused/v1/c2kv/extract"]
    receipt = json.loads((out / "receipt.json").read_text())
    assert receipt["counts"] == dict(generation=0, extraction=1)
    assert receipt["status"] == "failed"
    assert (out / "attempts.jsonl").is_file()
