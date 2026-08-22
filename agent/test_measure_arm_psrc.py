# -*- coding: utf-8 -*-
"""Tests for agent/measure_arm_psrc.py (P1-4: multi-source arm manifests).

Builds a tiny agent-llm-traces parquet (same synthetic-span style as
python/train/test_train_data_joint.py) plus a tiny hotpotqa jsonl, then
measures an arm whose manifest mixes bare traces qids with ``qa:`` qids —
the input shape that used to die with a RuntimeError before P1-4.

Run from the repo root:
  pytest agent/test_measure_arm_psrc.py -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _sub in ("python", "python/inference", "agent"):
    _path = str(_REPO_ROOT / _sub)
    if _path not in sys.path:
        sys.path.insert(0, _path)

import measure_arm_psrc as maps  # noqa: E402


_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Fetch the current weather for one city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]


def _write_traces(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    span = {
        "span_id": "span-1",
        "start_time": "2026-01-01T00:00:01",
        "status": "ok",
        "attributes": {
            "gen_ai.input.messages": json.dumps([
                {"role": "system", "content": "You are a weather agent."},
                {"role": "user", "content": "Hi there."},
                {"role": "assistant", "content": "Hello! How can I help?"},
                {"role": "user", "content": "What is the weather in Paris?"},
            ]),
            "gen_ai.output.messages": json.dumps([
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": {"city": "Paris"}},
                    }],
                }
            ]),
            "gen_ai.tool.definitions": json.dumps(_TOOLS),
        },
    }
    data_dir = tmp_path / "traces"
    data_dir.mkdir()
    pq.write_table(
        pa.table({
            "benchmark": ["weather-bench"],
            "session_id": ["sess-1"],
            "spans": [json.dumps([span])],
        }),
        data_dir / "shard.parquet",
    )
    manifest = tmp_path / "split_manifest.json"
    manifest.write_text(
        json.dumps({"train_session_ids": ["sess-1"], "eval_session_ids": []}),
        encoding="utf-8",
    )
    return str(data_dir), str(manifest)


def _write_hotpotqa(tmp_path):
    path = tmp_path / "hotpotqa.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for index in range(2):
            handle.write(json.dumps({
                "_id": f"hp-{index}",
                "question": f"Which document answers question {index}?",
                "answer": f"answer {index}",
                "documents": [
                    f"Document 1 (title: Alpha {index}) " + " ".join(f"a{index}-{j}" for j in range(15)),
                    f"Document 2 (title: Beta {index}) " + " ".join(f"b{index}-{j}" for j in range(15)),
                ],
            }, ensure_ascii=False) + "\n")
    return str(path)


def _write_manifest(tmp_path, qids, name="train_manifest_used.json"):
    path = tmp_path / name
    path.write_text(
        json.dumps({"doc_mode": "joint", "split_seed": 42, "train_qids": qids}),
        encoding="utf-8",
    )
    return str(path)


def test_mixed_traces_qa_arm_manifest_resolves(tmp_path):
    dataset_path, split_manifest = _write_traces(tmp_path)
    hotpotqa_path = _write_hotpotqa(tmp_path)
    manifest = _write_manifest(tmp_path, ["sess-1:0", "qa:hotpotqa:hp-0"])
    report = maps.main([
        "--dataset_path", dataset_path,
        "--split_manifest_file", split_manifest,
        "--qa_hotpotqa_path", hotpotqa_path,
        "--tokenizer", "fake",
        "--min_target_tokens", "0",
        "--arm", f"joint={manifest}",
    ])
    arm = report["arms"]["joint"]
    assert arm["doc_mode"] == "joint"
    assert arm["num_examples"] == 2
    assert arm["P_src"] > 0
    assert arm["T_tgt"] > 0


def test_qa_qid_without_qa_path_still_errors_loudly(tmp_path):
    dataset_path, split_manifest = _write_traces(tmp_path)
    manifest = _write_manifest(tmp_path, ["sess-1:0", "qa:hotpotqa:hp-0"])
    with pytest.raises(RuntimeError, match="not found in the loaded source"):
        maps.main([
            "--dataset_path", dataset_path,
            "--split_manifest_file", split_manifest,
            "--tokenizer", "fake",
            "--min_target_tokens", "0",
            "--arm", f"joint={manifest}",
        ])
