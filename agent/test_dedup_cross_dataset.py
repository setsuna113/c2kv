# -*- coding: utf-8 -*-
"""CPU-only offline unit tests for agent/dedup_cross_dataset.py.

No real dataset and no network: train/eval corpora are tiny synthetic
jsonl/parquet files; the BFCL fixture is a two-line file mirroring
``.foreman/ref/bfcl_data`` (``{"id": ..., "question": [[{"role": "user",
"content": ...}], ...]}``).

Coverage:
a. MinHash: a near-duplicate pair (one word changed in a long doc) is flagged
   and a distant pair is not; pure-Python and numpy signature paths agree;
b. exact dedup via normalized-text sha1 (case/whitespace insensitive);
c. removal list touches only train-side units, never eval-side;
d. unit extraction strategies (messages / tools / raw; OpenAI, ShareGPT and
   agent-llm-traces span layouts; jsonl and parquet inputs);
e. --bfcl_dir question-text extraction into eval units;
f. config validation and output schema (caps, counts, sorting).

Run from the repo root (local venv has pyarrow/pytest):
  pytest agent/test_dedup_cross_dataset.py -v
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pytest

# Make agent/ importable when pytest is invoked from the repo root.
_AGENT_DIR = Path(__file__).resolve().parent
if str(_AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(_AGENT_DIR))

import dedup_cross_dataset as dcd  # noqa: E402


def _long_doc(word="careful"):
    sentences = []
    for index in range(40):
        sentences.append(
            f"Step {index}: the agent must perform a {word} verification of artifact "
            f"number {index} before continuing with the deployment pipeline stage {index}."
        )
    return " ".join(sentences)


_DOC_A = _long_doc("careful")
# One word changed in a long doc.
_DOC_B = _DOC_A.replace("a careful verification of artifact number 7", "a thorough verification of artifact number 7", 1)
_DOC_DISTANT = " ".join(
    f"Hypothesis {index}: lattice gauge simulations probe quantum chromodynamics "
    f"observable {index} under nonzero baryon chemical potential run {index}."
    for index in range(40)
)


def _write_jsonl(path, records):
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


def _messages_record(record_id, texts):
    return {
        "id": record_id,
        "messages": [{"role": "user" if i % 2 == 0 else "assistant", "content": text} for i, text in enumerate(texts)],
    }


@pytest.fixture()
def corpora(tmp_path):
    train_path = tmp_path / "train.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    _write_jsonl(train_path, [
        _messages_record("train-rec-1", [_DOC_A]),
        _messages_record("train-rec-2", ["summarize the quarterly report exactly as stored"]),
        _messages_record("train-rec-3", ["a unique training-only instruction about backups"]),
    ])
    _write_jsonl(eval_path, [
        _messages_record("eval-rec-1", [_DOC_B]),
        _messages_record("eval-rec-2", ["Summarize   the quarterly REPORT exactly as stored"]),
        _messages_record("eval-rec-3", [_DOC_DISTANT]),
    ])
    bfcl_dir = tmp_path / "bfcl_data"
    bfcl_dir.mkdir()
    _write_jsonl(bfcl_dir / "BFCL_v4_multi_turn_base.json", [
        {"id": "multi_turn_base_0",
         "question": [[{"role": "user", "content": "Move the file into the temp directory."}],
                      [{"role": "user", "content": "Now sort it by line."}]]},
        {"id": "multi_turn_base_1",
         "question": [[{"role": "user", "content": "a unique training-only instruction about backups"}]]},
    ])
    return {
        "train": str(train_path),
        "eval": str(eval_path),
        "bfcl_dir": str(bfcl_dir),
    }


def _args(corpora, **overrides):
    values = {
        "train_inputs": [f"traincorp={corpora['train']}"],
        "eval_inputs": [f"evalcorp={corpora['eval']}"],
        "bfcl_dir": None,
        "unit": "messages",
        "out": None,
        "threshold": 0.8,
        "num_perm": 128,
        "lsh_bands": 16,
        "lsh_rows": 8,
        "shingle_size": 5,
        "max_shingles": 4096,
        "max_pairs": 10000,
        "max_bucket_pairs": 20000,
        "seed": 42,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


# ---------------------------------------------------------------------------
# MinHash behaviour.
# ---------------------------------------------------------------------------


def test_minhash_flags_near_duplicate_not_distant():
    hasher = dcd._MinHasher(128, 42)
    sig_a = hasher.signature(dcd._shingle_hashes(_DOC_A, 5, 4096))
    sig_b = hasher.signature(dcd._shingle_hashes(_DOC_B, 5, 4096))
    sig_c = hasher.signature(dcd._shingle_hashes(_DOC_DISTANT, 5, 4096))
    assert dcd._estimate_jaccard(sig_a, sig_b, 128) >= 0.8
    assert dcd._estimate_jaccard(sig_a, sig_c, 128) < 0.5


def test_signature_backends_agree():
    shingles = dcd._shingle_hashes(_DOC_A, 5, 4096)
    a, b = dcd._permutation_coefficients(128, 42)
    python_sig = dcd._signature_python(shingles, a, b)
    if dcd.np is None:
        pytest.skip("numpy not available")
    import numpy as np

    numpy_sig = dcd._signature_numpy(shingles, np.asarray(a, dtype=np.uint64), np.asarray(b, dtype=np.uint64))
    assert python_sig == numpy_sig
    # Empty input is handled by both paths.
    assert dcd._signature_python([], a, b) == dcd._empty_signature(128)


def test_shingle_backends_agree():
    if dcd.np is None:
        pytest.skip("numpy not available")
    data = dcd._normalize_text(_DOC_A).encode("utf-8")
    assert dcd._rolling_hashes_python(data, 5) == dcd._rolling_hashes_numpy(data, 5)
    # Non-ASCII text takes the same code path.
    data = dcd._normalize_text("Ünïcodé désaccord — 数据去重测试 " * 30).encode("utf-8")
    assert dcd._rolling_hashes_python(data, 5) == dcd._rolling_hashes_numpy(data, 5)


def test_near_dup_flagged_end_to_end_and_distant_not(corpora):
    result = dcd.dedup(_args(corpora))
    pairs = {(item["train_unit"], item["eval_unit"]) for item in result["near_dup_pairs"]}
    assert ("traincorp:train-rec-1:0", "evalcorp:eval-rec-1:0") in pairs
    assert not any("eval-rec-3" in eval_unit for _, eval_unit in pairs)
    flagged = {item["train_unit"] for item in result["near_dup_pairs"]}
    assert "traincorp:train-rec-3:0" not in flagged
    # est_jaccard is reported and pairs are sorted by it descending.
    estimates = [item["est_jaccard"] for item in result["near_dup_pairs"]]
    assert estimates == sorted(estimates, reverse=True)
    assert all(0.0 <= est <= 1.0 for est in estimates)


# ---------------------------------------------------------------------------
# Exact dedup and removal-list discipline.
# ---------------------------------------------------------------------------


def test_exact_dedup_counts_and_removal(corpora):
    result = dcd.dedup(_args(corpora))
    # "summarize ... as stored" vs "Summarize   the quarterly REPORT ..." is an
    # exact dup after normalization (lowercase + whitespace collapse).
    assert result["exact_dup_counts"] == {"traincorp": {"evalcorp": 1}}
    removal = {item["unit_id"]: item for item in result["removal_list"]}
    assert removal["traincorp:train-rec-2:0"]["match_type"] == "exact"
    assert removal["traincorp:train-rec-2:0"]["best_est_jaccard"] == 1.0


def test_removal_list_never_touches_eval_side(corpora):
    result = dcd.dedup(_args(corpora, bfcl_dir=corpora["bfcl_dir"]))
    removal_ids = {item["unit_id"] for item in result["removal_list"]}
    # The bfcl record multi_turn_base_1 is an exact dup of train-rec-3: the
    # train unit is removed, the eval (bfcl) unit is never removed.
    assert "traincorp:train-rec-3:0" in removal_ids
    assert not any(unit_id.startswith(("evalcorp:", "bfcl:")) for unit_id in removal_ids)
    assert all(item["dataset"] == "traincorp" for item in result["removal_list"])
    # The exact dup across the train x bfcl boundary is counted too.
    assert result["exact_dup_counts"]["traincorp"]["bfcl"] == 1


# ---------------------------------------------------------------------------
# Unit extraction strategies and input formats.
# ---------------------------------------------------------------------------


def test_unit_strategies_and_formats(tmp_path):
    spans = [
        {
            "span_id": "span-1",
            "start_time": "2026-01-01T00:00:01",
            "status": "ok",
            "attributes": {
                "gen_ai.input.messages": json.dumps([
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "trace user text"},
                ]),
                "gen_ai.output.messages": json.dumps([{"role": "assistant", "content": "trace reply"}]),
                "gen_ai.tool.definitions": json.dumps([
                    {"type": "function", "function": {"name": "search", "parameters": {"type": "object"}}}
                ]),
            },
        }
    ]
    import pyarrow as pa
    import pyarrow.parquet as pq

    parquet_path = tmp_path / "traces.parquet"
    pq.write_table(
        pa.table({"session_id": ["sess-1"], "spans": [json.dumps(spans)]}),
        parquet_path,
    )
    sharegpt_path = tmp_path / "sharegpt.jsonl"
    _write_jsonl(sharegpt_path, [{"id": "sg-1", "conversations": [{"from": "human", "value": "sharegpt question"}]}])
    raw_path = tmp_path / "raw.jsonl"
    _write_jsonl(raw_path, [{"id": "raw-1", "text": "raw document body"}])

    messages_units = dcd._load_units([f"traces={parquet_path}", f"sg={sharegpt_path}"], "messages", "train")
    texts = sorted(unit["text"] for unit in messages_units)
    assert texts == ["sharegpt question", "sys", "trace reply", "trace user text"]
    assert {unit["unit_id"] for unit in messages_units} >= {
        "traces:sess-1:0",
        "traces:sess-1:1",
        "traces:sess-1:2",
        "sg:sg-1:0",
    }
    assert all(unit["unit_hash"] == dcd._unit_hash(unit["text"]) for unit in messages_units)

    tool_units = dcd._load_units([f"traces={parquet_path}"], "tools", "train")
    assert len(tool_units) == 1
    assert json.loads(tool_units[0]["text"])["function"]["name"] == "search"

    raw_units = dcd._load_units([f"raw={raw_path}"], "raw", "train")
    assert [unit["text"] for unit in raw_units] == ["raw document body"]


def test_bfcl_dir_extraction(corpora):
    units = dcd._load_bfcl_units(corpora["bfcl_dir"])
    assert [unit["dataset"] for unit in units] == ["bfcl"] * 3
    by_record = {}
    for unit in units:
        by_record.setdefault(unit["record_id"], []).append(unit["text"])
    assert by_record["multi_turn_base_0"] == [
        "Move the file into the temp directory.",
        "Now sort it by line.",
    ]
    assert by_record["multi_turn_base_1"] == ["a unique training-only instruction about backups"]
    assert all(unit["side"] == "eval" for unit in units)


# ---------------------------------------------------------------------------
# Config validation and output schema.
# ---------------------------------------------------------------------------


def test_config_validation(corpora):
    with pytest.raises(ValueError, match="lsh_bands"):
        dcd.dedup(_args(corpora, lsh_bands=10))
    with pytest.raises(ValueError, match="threshold"):
        dcd.dedup(_args(corpora, threshold=0.0))
    with pytest.raises(ValueError, match="name=glob"):
        dcd.dedup(_args(corpora, train_inputs=["no-equals-sign"]))
    with pytest.raises(FileNotFoundError):
        dcd.dedup(_args(corpora, train_inputs=["missing=/no/such/files-*.jsonl"]))


def test_output_schema_and_pair_cap(corpora):
    result = dcd.dedup(_args(corpora, bfcl_dir=corpora["bfcl_dir"], max_pairs=1))
    metadata = result["metadata"]
    assert metadata["unit"] == "messages"
    assert metadata["train_units"] == {"traincorp": 3}
    assert metadata["eval_units"] == {"bfcl": 3, "evalcorp": 3}
    assert metadata["shared_exact_hashes"] == 2
    assert metadata["near_dup_pair_count"] >= len(result["near_dup_pairs"])
    assert len(result["near_dup_pairs"]) <= 1
    assert metadata["removal_count"] == len(result["removal_list"])
    removal_keys = {"unit_id", "dataset", "record_id", "unit_index", "unit_hash", "match_type", "best_est_jaccard", "matched_eval_unit"}
    assert all(set(item) == removal_keys for item in result["removal_list"])
    # JSON-serializable as-is (this is what --out writes).
    json.dumps(result, ensure_ascii=False)


def test_shingle_bottom_k_cap_keeps_near_dup_detection():
    long_a = _long_doc("careful") * 20
    long_b = long_a.replace("artifact number 7", "artifact number seven", 1)
    shingles_a = dcd._shingle_hashes(long_a, 5, 512)
    assert len(shingles_a) <= 512
    hasher = dcd._MinHasher(128, 42)
    sig_a = hasher.signature(shingles_a)
    sig_b = hasher.signature(dcd._shingle_hashes(long_b, 5, 512))
    assert dcd._estimate_jaccard(sig_a, sig_b, 128) >= 0.8
