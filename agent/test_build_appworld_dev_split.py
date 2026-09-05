"""Tests for agent/build_appworld_dev_split.py (tiny parquet fixture)."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

pyarrow = pytest.importorskip("pyarrow")
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_appworld_dev_split as bads  # noqa: E402


# session_id -> benchmark of the fixture corpus.
FIXTURE_ROWS = [
    ("s-appworld-1", "appworld"),
    ("s-appworld-2", "AppWorld"),
    ("s-appworld-3", "appworld"),
    ("s-tau2-airline-1", "tau2_airline"),
    ("s-tau2-retail-1", "tau2_retail"),
    ("s-tau2-telecom-1", "tau2_telecom"),
    ("s-swe-1", "swebench"),
    ("s-browse-1", "browsecompplus"),
    ("s-other-1", "some_other_bench"),
]


def _write_dataset(tmp_path: Path) -> Path:
    dataset = tmp_path / "traces"
    (dataset / "data").mkdir(parents=True, exist_ok=True)
    # Two shards, each carrying two rows per session, to exercise dedup and the
    # sorted-file iteration order.
    for shard, chunk in enumerate((FIXTURE_ROWS[:5], FIXTURE_ROWS[5:])):
        session_ids = [session for session, _ in chunk for _ in (0, 1)]
        benchmarks = [benchmark for _, benchmark in chunk for _ in (0, 1)]
        table = pa.table(
            {
                "session_id": pa.array(session_ids, type=pa.string()),
                "benchmark": pa.array(benchmarks, type=pa.string()),
                "payload": pa.array(["x"] * len(session_ids), type=pa.string()),
            }
        )
        pq.write_table(table, dataset / "data" / f"shard-{shard}.parquet")
    return dataset


def _write_base_manifest(tmp_path: Path, train, evals, name="taskproxy_disjoint") -> Path:
    path = tmp_path / "base_manifest.json"
    path.write_text(
        json.dumps(
            {
                name: {
                    "train_session_ids": sorted(train),
                    "eval_session_ids": sorted(evals),
                },
                "metadata": {"split_name": name},
            }
        ),
        encoding="utf-8",
    )
    return path


def _build(tmp_path: Path, **overrides):
    dataset = overrides.pop("dataset", None) or _write_dataset(tmp_path)
    train = overrides.pop(
        "train",
        [
            "s-appworld-1",
            "s-appworld-2",
            "s-tau2-airline-1",
            "s-tau2-retail-1",
            "s-tau2-telecom-1",
            "s-swe-1",
            "s-browse-1",
            "s-other-1",
        ],
    )
    evals = overrides.pop("evals", ["s-appworld-3", "s-tau2-airline-1x", "s-other-2"])
    base_manifest = overrides.pop("base_manifest", None) or _write_base_manifest(
        tmp_path, train, evals
    )
    argv = [
        "--dataset_path",
        str(dataset),
        "--base_manifest_file",
        str(base_manifest),
        "--out",
        str(tmp_path / "out.json"),
    ]
    for key, value in overrides.items():
        argv += [f"--{key}", str(value)]
    args = bads.parse_args(argv)
    return bads.build_split(args)


def test_train_drops_excluded_benchmarks_eval_keeps_appworld_only(tmp_path):
    manifest, tables = _build(tmp_path)
    split = manifest["appworld_dev"]
    # airline/retail/telecom/swebench/browsecompplus leave the train side;
    # appworld and the unmatched "some_other_bench" stay.
    assert split["train_session_ids"] == ["s-appworld-1", "s-appworld-2", "s-other-1"]
    # eval keeps only sessions whose benchmark matches --eval_include; the two
    # ids absent from the parquet have no benchmark and are dropped.
    assert split["eval_session_ids"] == ["s-appworld-3"]
    # counts key on the RAW benchmark string (matching is case-insensitive, the
    # accounting is not).
    assert tables["train"] == {"AppWorld": 1, "appworld": 1, "some_other_bench": 1}
    assert tables["eval"] == {"appworld": 1}
    assert tables["dropped_train"]["tau2_airline"] == 1
    assert tables["dropped_eval"][bads.UNKNOWN_BENCHMARK] == 2


def test_benchmark_match_is_case_insensitive_substring(tmp_path):
    # "AppWorld" (mixed case) must be recognised on both sides.
    manifest, _ = _build(
        tmp_path,
        train=["s-appworld-2", "s-tau2-airline-1"],
        evals=["s-appworld-2x", "s-appworld-3"],
        exclude_benchmarks="APPWORLD",
    )
    split = manifest["appworld_dev"]
    assert split["train_session_ids"] == ["s-tau2-airline-1"]
    assert split["eval_session_ids"] == ["s-appworld-3"]


def test_metadata_counts_and_sha256_are_deterministic(tmp_path):
    manifest, _ = _build(tmp_path)
    meta = manifest["metadata"]
    split = manifest["appworld_dev"]
    assert meta["num_train_sessions"] == len(split["train_session_ids"]) == 3
    assert meta["num_eval_sessions"] == len(split["eval_session_ids"]) == 1
    assert meta["num_base_train_sessions"] == 8
    assert meta["num_base_eval_sessions"] == 3
    assert meta["exclude_benchmarks"] == [
        "airline",
        "retail",
        "telecom",
        "swebench",
        "browsecompplus",
    ]
    assert meta["eval_include"] == ["appworld"]
    expected = hashlib.sha256(
        "".join(f"{sid}\n" for sid in split["train_session_ids"]).encode("utf-8")
    ).hexdigest()
    assert meta["train_session_ids_sha256"] == expected
    assert meta["num_sessions_without_benchmark"] == 0

    # Same inputs -> byte-identical manifest.
    again, _ = _build(tmp_path)
    assert json.dumps(again, sort_keys=False) == json.dumps(manifest, sort_keys=False)


def test_ids_are_sorted_and_unique(tmp_path):
    manifest, _ = _build(
        tmp_path,
        train=["s-appworld-2", "s-appworld-1", "s-appworld-1"],
        evals=["s-appworld-3"],
    )
    split = manifest["appworld_dev"]
    assert split["train_session_ids"] == ["s-appworld-1", "s-appworld-2"]
    assert split["eval_session_ids"] == ["s-appworld-3"]
    assert manifest["metadata"]["num_base_train_sessions"] == 2


def test_train_eval_overlap_raises(tmp_path):
    with pytest.raises(RuntimeError, match="overlap"):
        _build(
            tmp_path,
            train=["s-appworld-1", "s-appworld-3"],
            evals=["s-appworld-3"],
        )


def test_missing_base_split_name_raises(tmp_path):
    dataset = _write_dataset(tmp_path)
    base = _write_base_manifest(tmp_path, ["s-appworld-1"], ["s-appworld-3"], name="other_name")
    with pytest.raises(KeyError):
        _build(tmp_path, dataset=dataset, base_manifest=base)


def test_empty_eval_include_raises(tmp_path):
    with pytest.raises(ValueError, match="eval_include"):
        _build(tmp_path, eval_include="")


def test_missing_parquet_raises(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        _build(tmp_path, dataset=empty)


def test_cli_writes_manifest_and_prints_table(tmp_path, capsys):
    dataset = _write_dataset(tmp_path)
    base = _write_base_manifest(
        tmp_path,
        ["s-appworld-1", "s-tau2-airline-1", "s-other-1"],
        ["s-appworld-3"],
    )
    out = tmp_path / "nested" / "appworld_dev.json"
    bads.main(
        [
            "--dataset_path",
            str(dataset),
            "--base_manifest_file",
            str(base),
            "--out",
            str(out),
        ]
    )
    manifest = json.loads(out.read_text(encoding="utf-8"))
    assert manifest["appworld_dev"]["train_session_ids"] == ["s-appworld-1", "s-other-1"]
    assert manifest["appworld_dev"]["eval_session_ids"] == ["s-appworld-3"]
    printed = capsys.readouterr().out
    assert "| benchmark | train | eval | dropped_train | dropped_eval |" in printed
    assert "| appworld |" in printed


def test_metadata_counts_unknown_benchmark_sessions(tmp_path):
    # A session present in the parquet but with a NULL `benchmark` maps to
    # UNKNOWN_BENCHMARK: invisible to num_sessions_without_benchmark (it IS in
    # the mapping), so it needs its own counter.
    dataset = tmp_path / "traces_null"
    (dataset / "data").mkdir(parents=True)
    table = pa.table(
        {
            "session_id": pa.array(["s-appworld-3", "s-null-1"], type=pa.string()),
            "benchmark": pa.array(["appworld", None], type=pa.string()),
        }
    )
    pq.write_table(table, dataset / "data" / "shard-0.parquet")
    manifest, _ = _build(
        tmp_path, dataset=dataset, train=["s-null-1"], evals=["s-appworld-3"]
    )
    meta = manifest["metadata"]
    assert manifest["appworld_dev"]["train_session_ids"] == ["s-null-1"]
    assert meta["num_sessions_without_benchmark"] == 0
    assert meta["num_sessions_unknown_benchmark"] == 1
    assert meta["num_parquet_files_with_benchmark_column"] == 1


def test_empty_eval_side_raises(tmp_path):
    # Nothing on the base eval side matches --eval_include -> the manifest would
    # make the whole history eval score zero rows.  Must fail loudly instead.
    with pytest.raises(RuntimeError, match="eval side is empty"):
        _build(tmp_path, train=["s-appworld-1"], evals=["s-tau2-airline-1"])


def test_empty_train_side_raises(tmp_path):
    with pytest.raises(RuntimeError, match="train side is empty"):
        _build(tmp_path, train=["s-swe-1", "s-browse-1"], evals=["s-appworld-3"])


def test_missing_benchmark_column_raises(tmp_path):
    # No `benchmark` column at all: every session would map to <unknown> and the
    # eval side would silently come out empty.  Name the root cause instead.
    dataset = tmp_path / "traces_nobench"
    (dataset / "data").mkdir(parents=True)
    table = pa.table(
        {"session_id": pa.array(["s-appworld-1", "s-appworld-3"], type=pa.string())}
    )
    pq.write_table(table, dataset / "data" / "shard-0.parquet")
    with pytest.raises(RuntimeError, match="benchmark. column"):
        _build(
            tmp_path, dataset=dataset, train=["s-appworld-1"], evals=["s-appworld-3"]
        )


# --- --max_system_tokens: drop eval sessions the harness would truncate -----
#
# The measurement itself (_eval_session_system_tokens) parses agent-llm-traces
# spans through python/train/train_data_multiturn.py, which pulls in
# datasets/torch; it is stubbed here so the filter, the guards and the metadata
# stay covered on a CPU-only host.


def _stub_tokenizer(monkeypatch):
    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained",
        classmethod(lambda cls, *args, **kwargs: object()),
    )


def test_max_system_tokens_is_off_by_default(tmp_path):
    manifest, _ = _build(tmp_path)
    meta = manifest["metadata"]
    assert meta["max_system_tokens"] == 0
    assert meta["num_eval_sessions_before"] == meta["num_eval_sessions_after"] == 1
    assert meta["num_eval_sessions_dropped_system_overflow"] == 0


def test_max_system_tokens_drops_the_overlong_eval_session(tmp_path, monkeypatch):
    _stub_tokenizer(monkeypatch)
    monkeypatch.setattr(
        bads,
        "_eval_session_system_tokens",
        lambda data_files, session_ids, tokenizer: {
            "s-appworld-2": 20480,
            "s-appworld-3": 3000,
        },
    )
    baseline, _ = _build(
        tmp_path, train=["s-appworld-1"], evals=["s-appworld-2", "s-appworld-3"]
    )
    manifest, tables = _build(
        tmp_path,
        train=["s-appworld-1"],
        evals=["s-appworld-2", "s-appworld-3"],
        max_system_tokens=4096,
        tokenizer="./models/Qwen3-4B-Instruct-2507",
    )
    split = manifest["appworld_dev"]
    assert baseline["appworld_dev"]["eval_session_ids"] == ["s-appworld-2", "s-appworld-3"]
    assert split["eval_session_ids"] == ["s-appworld-3"]
    meta = manifest["metadata"]
    assert meta["max_system_tokens"] == 4096
    assert meta["num_eval_sessions_before"] == 2
    assert meta["num_eval_sessions_after"] == meta["num_eval_sessions"] == 1
    assert meta["num_eval_sessions_dropped_system_overflow"] == 1
    # The slice is pinnable: dropping a session moves the hash and shows up in
    # the dropped_eval accounting.
    assert meta["eval_session_ids_sha256"] != baseline["metadata"]["eval_session_ids_sha256"]
    assert tables["dropped_eval"]["AppWorld"] == 1
    # The train side is untouched by the eval-only filter.
    assert split["train_session_ids"] == baseline["appworld_dev"]["train_session_ids"]


def test_max_system_tokens_keeps_unmeasured_sessions(tmp_path, monkeypatch):
    # A session with no usable span never reaches the mapping; only sessions we
    # actually measured may be dropped.
    _stub_tokenizer(monkeypatch)
    monkeypatch.setattr(
        bads, "_eval_session_system_tokens", lambda data_files, session_ids, tokenizer: {}
    )
    manifest, _ = _build(
        tmp_path,
        train=["s-appworld-1"],
        evals=["s-appworld-2", "s-appworld-3"],
        max_system_tokens=16,
        tokenizer="./models/Qwen3-4B-Instruct-2507",
    )
    assert manifest["appworld_dev"]["eval_session_ids"] == ["s-appworld-2", "s-appworld-3"]
    assert manifest["metadata"]["num_eval_sessions_dropped_system_overflow"] == 0


def test_max_system_tokens_emptying_the_eval_side_raises(tmp_path, monkeypatch):
    # An over-tight threshold must abort loudly rather than silently shrink the
    # selection set to nothing.
    _stub_tokenizer(monkeypatch)
    monkeypatch.setattr(
        bads,
        "_eval_session_system_tokens",
        lambda data_files, session_ids, tokenizer: {"s-appworld-3": 5000},
    )
    with pytest.raises(RuntimeError, match="eval side is empty after --max_system_tokens"):
        _build(
            tmp_path,
            max_system_tokens=4096,
            tokenizer="./models/Qwen3-4B-Instruct-2507",
        )


def test_max_system_tokens_without_tokenizer_raises(tmp_path):
    with pytest.raises(ValueError, match="requires --tokenizer"):
        _build(tmp_path, max_system_tokens=4096)
