"""No-model contracts for the ACEBench Full-original raw SGLang bridge."""
from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path

import pytest

from benchmarks.memory_runtime.acebench_full_raw_backend import (
    EXTERNAL_CACHE_POLICY,
    FullRawSGLangError,
    RAW_HTTP_JOURNAL_SCHEMA,
    SGLangFullRawGenerator,
)
from benchmarks.memory_runtime.event_native_raw import RuntimeMemoryView, build_raw_control
from benchmarks.memory_runtime.event_native_draft import decode_native_generation
from benchmarks.memory_runtime.tests.test_event_native_policy import Tokenizer, contracts, recovery_sequence
from history_memory.events import EventStore


class _Response:
    status = 200

    def __init__(self, payload):
        self._body = json.dumps(payload).encode("utf-8")

    def read(self, limit=None):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


class _Opener:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def open(self, request, *, timeout):
        self.requests.append(
            {
                "url": request.full_url,
                "method": request.get_method(),
                "timeout": timeout,
                "payload": json.loads(request.data.decode("utf-8")),
            }
        )
        return _Response(self.responses.pop(0))


def _memory():
    packing, policy = contracts()
    packing["max_workspace_tokens"] = 4096
    request = recovery_sequence()[-1]
    return build_raw_control(
        EventStore.from_messages(request["session_id"], request["messages"]),
        Tokenizer(),
        packing=packing,
        policy=policy,
        mode="full_original",
        max_new_tokens=8,
        tools=(),
    ).memory


def _response(memory, *, prompt_tokens=None, output_ids=(71, 72), logprob_ids=None):
    ids = list(output_ids)
    return {
        "text": "['tool', {'x': 1}]",
        "output_ids": ids,
        "meta_info": {
            "prompt_tokens": len(memory.system_input_ids) + len(memory.workspace_input_ids)
            if prompt_tokens is None
            else prompt_tokens,
            "completion_tokens": len(ids),
            "finish_reason": {"type": "stop"},
            "output_token_logprobs": [
                [-0.25 - index, token, None]
                for index, token in enumerate(logprob_ids if logprob_ids is not None else ids)
            ],
            "output_token_logprobs_length": len(ids),
            "cached_tokens": 99,
        },
    }


def _generator(tmp_path: Path, opener, **overrides):
    arguments = dict(
        upstream="http://127.0.0.1:35160",
        model_context=4096,
        max_new_tokens=8,
        max_generation_calls=2,
        timeout_seconds=12,
        eos_token_ids=(151645,),
        eos_source="checkpoint_generation_config",
        journal_path=tmp_path / "raw_http.jsonl",
        opener=opener,
    )
    arguments.update(overrides)
    return SGLangFullRawGenerator(**arguments)


def _journal(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_full_original_posts_exact_raw_ids_and_journals_a_durable_pair(tmp_path):
    memory = _memory()
    opener = _Opener([_response(memory)])
    generator = _generator(tmp_path, opener)

    with generator.decision_scope(session_id="ace/one"):
        result = generator.generate(memory, ratio=4, max_new_tokens=8)

    posted = opener.requests[0]["payload"]
    assert opener.requests[0]["url"] == "http://127.0.0.1:35160/generate"
    assert opener.requests[0]["method"] == "POST" and opener.requests[0]["timeout"] == 12
    assert posted == {
        "input_ids": list(memory.system_input_ids + memory.workspace_input_ids),
        "sampling_params": {"temperature": 0, "top_p": 1, "sampling_seed": 0, "max_new_tokens": 8},
        "return_logprob": True,
        "logprob_start_len": -1,
        "top_logprobs_num": 0,
        "return_text_in_logprobs": False,
        "stream": False,
    }
    assert not any(key.startswith("c2kv_") for key in posted)
    assert result.token_ids == (71, 72)
    assert result.token_logprobs == (-0.25, -1.25)
    assert result.finish_reason == "stop"
    assert result.stats["external_sglang_cache"] == {
        "state": "unknown", "client_owns": False, "flush_or_release_attempted": False
    }
    assert generator.session_cache_info()["policy"] == EXTERNAL_CACHE_POLICY
    assert generator.session_cache_info()["requests_submitted"] == 1
    generator.close_session()

    records = _journal(tmp_path / "raw_http.jsonl")
    assert [record["event"] for record in records] == ["request", "response"]
    assert all(record["schema"] == RAW_HTTP_JOURNAL_SCHEMA for record in records)
    assert records[0]["request"] == posted
    assert records[1]["status"] == "completed" and records[1]["response"]["output_ids"] == [71, 72]


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda memory: replace(memory, chunks=(object(),)), "chunks"),
        (
            lambda memory: replace(
                memory,
                view=replace(memory.view, evidence_event_ids=(memory.view.raw_event_ids[0],)),
            ),
            "evidence_event_ids",
        ),
        (
            lambda memory: replace(
                memory,
                view=replace(memory.view, omitted_event_ids=(memory.view.raw_event_ids[0],)),
            ),
            "omitted_event_ids",
        ),
        (lambda memory: replace(memory, raw_source_indices=(0, 2)), "noncontiguous"),
    ],
)
def test_rejects_non_full_original_memory_before_network(tmp_path, mutate, message):
    memory = mutate(_memory())
    opener = _Opener([])
    generator = _generator(tmp_path, opener)

    with pytest.raises(ValueError, match=message):
        generator.generate(memory, ratio=4, max_new_tokens=8)
    assert opener.requests == []
    assert not (tmp_path / "raw_http.jsonl").exists()


def test_rejects_mismatched_server_counts_and_logprob_ids_after_one_durable_transport(tmp_path):
    memory = _memory()
    opener = _Opener([_response(memory, logprob_ids=(71, 999))])
    generator = _generator(tmp_path, opener, max_generation_calls=1)

    with pytest.raises(FullRawSGLangError, match="token ID"):
        generator.generate(memory, ratio=4, max_new_tokens=8)
    assert len(opener.requests) == 1
    records = _journal(tmp_path / "raw_http.jsonl")
    assert [record["status"] for record in records] == ["started", "failed"]
    assert records[1]["response"]["meta_info"]["output_token_logprobs"][1][1] == 999
    with pytest.raises(RuntimeError, match="cap exhausted"):
        generator.generate(memory, ratio=4, max_new_tokens=8)
    assert len(opener.requests) == 1


def test_rejects_mismatched_server_prompt_count_before_returning_a_result(tmp_path):
    memory = _memory()
    opener = _Opener([_response(memory, prompt_tokens=0)])
    generator = _generator(tmp_path, opener)

    with pytest.raises(FullRawSGLangError, match="prompt_tokens"):
        generator.generate(memory, ratio=4, max_new_tokens=8)
    assert len(opener.requests) == 1
    assert _journal(tmp_path / "raw_http.jsonl")[1]["status"] == "failed"


def test_rejects_non_full_original_layout_even_when_other_fields_look_raw(tmp_path):
    memory = _memory()
    no_gist_view = RuntimeMemoryView(
        gist_event_ids=(),
        raw_event_ids=memory.view.raw_event_ids,
        evidence_event_ids=(),
        omitted_event_ids=(),
        mandatory_raw_event_ids=memory.view.raw_event_ids,
        raw_control_layout="no-gist-native-v1",
    )
    opener = _Opener([])
    with pytest.raises(ValueError, match="full-original-native-v1"):
        _generator(tmp_path, opener).generate(
            replace(memory, view=no_gist_view), ratio=4, max_new_tokens=8
        )
    assert opener.requests == []


def test_parent_native_decoder_removes_only_the_configured_terminal_eos(tmp_path):
    memory = _memory()
    opener = _Opener([_response(memory, output_ids=(71, 151645))])
    result = _generator(tmp_path, opener).generate(memory, ratio=4, max_new_tokens=8)

    class Tokenizer:
        def decode(self, ids, *, skip_special_tokens, clean_up_tokenization_spaces):
            assert ids == [71]
            assert skip_special_tokens is False
            assert clean_up_tokenization_spaces is False
            return "raw text without EOS"

    draft = decode_native_generation(Tokenizer(), result, call_id_prefix="d1_r0")
    assert draft.text == "raw text without EOS"
    assert result.stats["eos_token_ids"] == [151645]
    assert result.stats["eos_source"] == "checkpoint_generation_config"
