"""Contracts for the finite off-policy pre-B candidate probe."""
import copy
import io
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "benchmarks"))
from memory_runtime import pre_b_candidate_probe as probe


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return list(range(len(json.dumps(messages, sort_keys=True)) // 4 + 1))


def fixture(source_visible=False):
    first = "current token" if source_visible else "archived token"
    messages = [
        dict(role="user", content=first),
        dict(role="assistant", content="Stored."),
        dict(role="user", content="Continue with current token."),
    ]
    event_id = "task:m0" if not source_visible else None
    response = {
        "content": "Action:",
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "lookup", "arguments": '{"key":"archived token"}'},
            }
        ],
    }
    return dict(
        case_id="fixture-visible" if source_visible else "fixture-gap",
        source_file="archive.jsonl",
        source_line=1,
        source_file_sha256="0" * 64,
        request_id="request-1",
        eval_context=dict(task_id="task", user_turn=1, step=0, attempt=0),
        request_view=dict(model="c2kv-agent", messages=messages, tools=[]),
        archived_first_draft=dict(
            source_pointer="archive.jsonl:1/generation_trace/0/response_view",
            finish_reason="tool_calls",
            response_view=response,
            response_digest=probe.digest(response),
        ),
        archived_gap=dict(
            status="gap",
            reason="missing_unique_complete_source",
            candidate_event_id=event_id,
            judges_action_correctness=False,
        ),
    )


def prepared(source_visible=False):
    value = probe.build_probe([fixture(source_visible)], Tokenizer())
    value["freeze_digest"] = probe.digest(value)
    return value


def test_activated_fixture_uses_one_base_and_only_two_new_generations():
    value = prepared()
    probe.validate(value)
    case = value["cases"][0]
    assert case["activated"] is True
    assert case["old_gap_verified_by_current_detector"] is True
    assert case["base_equivalence"] == {
        "same_messages": True,
        "same_source_cutoff": True,
        "same_visible_source_indices": True,
        "same_selection_and_budget_state": True,
    }
    controls = case["controls"]
    assert controls["no_upgrade"]["new_generation_calls"] == 0
    assert controls["acquire_for_next"]["new_generation_calls"] == 0
    assert controls["same_view_secondgen"]["new_generation_calls"] == 1
    assert controls["incumbent_e_regen"]["new_generation_calls"] == 1
    decision = controls["acquire_for_next"]["decision"]
    assert decision["deferred_lease_acquisition_count"] == 1
    assert decision["current_view_upgrade_count"] == 0
    assert decision["actual_regeneration_count"] == 0
    assert decision["final_action_source"] == "original_draft"
    assert controls["acquire_for_next"]["current_payload_unchanged"] is True
    assert {cell["control"] for cell in value["schedule"]} == {
        "same_view_secondgen",
        "incumbent_e_regen",
    }
    assert len(value["schedule"]) == 2


def test_visible_old_source_marks_branch_nonactivated_without_calls():
    value = prepared(source_visible=True)
    probe.validate(value)
    case = value["cases"][0]
    assert case["activated"] is False
    assert case["old_gap_verified_by_current_detector"] is False
    assert value["schedule"] == []
    assert value["extraction_manifest"] == []
    assert all(
        case["controls"][name]["new_generation_calls"] == 0
        for name in probe.CONTROL_ORDER
    )


def test_privileged_context_is_rejected_before_runtime_preparation():
    case = fixture()
    case["eval_context"]["gold"] = {"answer": "hidden"}
    with pytest.raises(ValueError, match="Privileged"):
        probe.build_probe([case], Tokenizer())


def test_payload_edit_is_rejected_before_transport(tmp_path, monkeypatch):
    value = prepared()
    payload_id = value["schedule"][0]["payload_id"]
    value["views"][payload_id]["payload"]["max_tokens"] = 8192
    monkeypatch.setattr(
        probe,
        "build_opener",
        lambda *args: pytest.fail("Opened transport before P2 validation"),
    )
    with pytest.raises(ValueError, match="artifact changed"):
        probe.execute(value, "http://unused", tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_fake_live_execution_materializes_then_runs_only_second_generations(tmp_path, monkeypatch):
    value = prepared()
    identity = dict(
        model_path="/home/user/checkpoints_upstream/checkpoint-1088",
        dtype="bfloat16",
        device="npu",
        context_length=16384,
        tp_size=1,
        enable_c2kv=True,
        c2kv_query_proj="base",
        c2kv_tools_dump="full",
        attention_backend="ascend",
        served_model_name="c2kv-agent",
    )
    manifest = {item["placeholder"]: item for item in value["extraction_manifest"]}
    posted = []

    class Opener:
        def open(self, request, **kwargs):
            if isinstance(request, str):
                return io.BytesIO(json.dumps(identity).encode())
            payload = json.loads(request.data)
            posted.append((request.full_url, payload))
            if request.full_url.endswith("/v1/c2kv/extract"):
                item = next(
                    value
                    for value in manifest.values()
                    if value["content"] == payload["text"]
                )
                result = dict(
                    success=True,
                    key_hash="real-" + str(len(posted)),
                    original_seq_len=item["original_seq_len"],
                    gist_len=item["gist_len"],
                )
            else:
                cell = value["schedule"][
                    sum(url.endswith("/v1/chat/completions") for url, _ in posted) - 1
                ]
                view = value["views"][cell["payload_id"]]
                result = {
                    "choices": [
                        {
                            "message": {"role": "assistant", "content": "second response"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": len(view["raw_input_ids"]),
                        "completion_tokens": 2,
                        "total_tokens": len(view["raw_input_ids"]) + 2,
                    },
                    "metadata": {
                        "sglang_runtime": {
                            "bytes_per_kv_token": probe.GEOMETRY,
                            "c2kv_query_proj_effective": "base",
                            "c2kv_gist_seen": True,
                        },
                    },
                }
            return io.BytesIO(json.dumps(result).encode())

    monkeypatch.setattr(probe, "build_opener", lambda *args: Opener())
    out = tmp_path / "out"
    state = probe.execute(value, "http://unused", out)
    assert state["status"] == "completed"
    assert state["counts"] == {
        "generation": 2,
        "extraction": len(value["extraction_manifest"]),
    }
    generation_urls = [url for url, _ in posted if url.endswith("/v1/chat/completions")]
    assert len(generation_urls) == 2
    assert {row["control"] for row in map(json.loads, (out / "cells.jsonl").read_text().splitlines())} == {
        "same_view_secondgen",
        "incumbent_e_regen",
    }
