#!/usr/bin/env python3
"""Differential and bounded performance checks for history corpus preparation.

The driver exports the submitted baseline commit without changing git refs and
runs the baseline and current working tree in separate Python processes.  The
worker uses the same serialized rows, tokenizer snapshot, and configuration for
both implementations.  Timing and tokenizer instrumentation live outside the
prepared manifest so exact artifact equality remains testable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


BASELINE_REVISION = "ad277e785f6c09186ebabcf8d3bbd39b96d62f0f"
ARTIFACT_FILES = ("sessions.jsonl", "paired_decisions.jsonl", "manifest.json")
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TOKENIZER_CACHE = (
    Path.home()
    / ".cache"
    / "huggingface"
    / "hub"
    / "models--Qwen--Qwen3-4B-Instruct-2507"
)
DEFAULT_TRACES_ROOT = (
    REPO_ROOT.parent / "investigation" / "data" / "agent-llm-traces"
)


def _json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _json_load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _tool(name: str = "fetch") -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": "Fetch one named resource.",
            "parameters": {
                "type": "object",
                "properties": {
                    "resource": {"type": "string"},
                    "case_id": {"type": "string"},
                },
                "required": ["resource"],
            },
        },
    }


def _call(call_id: str, resource: str, *, case_id: str | None = None) -> dict[str, Any]:
    arguments = {"resource": resource}
    if case_id is not None:
        arguments["case_id"] = case_id
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": "fetch", "arguments": json.dumps(arguments)},
    }


def _normalized_row(
    session_id: str,
    messages: Sequence[Mapping[str, Any]],
    *,
    source: str = "benchmark:synthetic",
    template_id: str = "history-prepare-differential",
    tools: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    return {
        "session_id": session_id,
        "source": source,
        "split": "train",
        "task_id": session_id,
        "template_id": template_id,
        "messages": [dict(message) for message in messages],
        "tools": [dict(tool) for tool in tools],
    }


def _lease_row(session_id: str = "lease-state") -> dict[str, Any]:
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": "Use tools and preserve cited evidence."},
        {"role": "user", "content": "Fetch alpha.txt for case CASE-17."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [_call("alpha", "alpha.txt", case_id="CASE-17")],
        },
        {
            "role": "tool",
            "tool_call_id": "alpha",
            "content": "CASE-17 alpha.txt payload: " + "evidence " * 24,
        },
        {"role": "assistant", "content": "The alpha evidence was saved."},
        {"role": "user", "content": "Use alpha.txt for CASE-17 now."},
        {"role": "assistant", "content": "CASE-17 is supported by alpha.txt."},
        {"role": "user", "content": "Continue with an unrelated status line."},
        {"role": "assistant", "content": "Status line one."},
        {"role": "user", "content": "Continue the unrelated status."},
        {"role": "assistant", "content": "Status line two."},
        {"role": "user", "content": "Finish the unrelated status."},
        {"role": "assistant", "content": "Status complete."},
        {"role": "user", "content": "Add a final unrelated note."},
        {"role": "assistant", "content": "Final note."},
    ]
    return _normalized_row(
        session_id,
        messages,
        template_id="stateful-lease",
        tools=[_tool()],
    )


def _long_tool_row(
    session_id: str,
    *,
    turns: int,
    payload_words: int = 32,
    source: str = "benchmark:synthetic",
) -> dict[str, Any]:
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": "Inspect resources in order and preserve CASE-LONG."},
        {"role": "user", "content": "Inspect all resources for CASE-LONG."},
    ]
    for index in range(turns):
        call_id = f"call-{index}"
        resource = f"resource-{index}.json"
        messages.extend(
            [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        _call(
                            call_id,
                            resource,
                            case_id="CASE-LONG" if index in {0, turns - 1} else None,
                        )
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": json.dumps(
                        {
                            "resource": resource,
                            "status": "available",
                            "payload": (f"value-{index} " * payload_words).strip(),
                            **({"case_id": "CASE-LONG"} if index == 0 else {}),
                        }
                    ),
                },
            ]
        )
    messages.extend(
        [
            {"role": "user", "content": "Use the CASE-LONG evidence."},
            {"role": "assistant", "content": "CASE-LONG is supported by the inspected resources."},
        ]
    )
    return _normalized_row(
        session_id,
        messages,
        source=source,
        template_id="long-tool-conversation",
        tools=[_tool()],
    )


def _short_row(
    session_id: str,
    *,
    source: str = "benchmark:synthetic",
    answer: str = "ok",
) -> dict[str, Any]:
    return _normalized_row(
        session_id,
        [
            {"role": "user", "content": f"Question for {session_id}"},
            {"role": "assistant", "content": answer},
        ],
        source=source,
        template_id="zero-gist",
    )


def _packing(**overrides: Any) -> dict[str, Any]:
    values: dict[str, Any] = {
        "ratios": [4, 8],
        "recent_tool_events": 0,
        "max_chunk_tokens": 256,
        "chunk_overlap": 16,
        "max_chunks": 128,
        "max_encoder_tokens": 100_000,
        "max_system_tokens": 50_000,
        "max_workspace_tokens": 50_000,
        "max_target_tokens": 32_768,
        "max_sequence_tokens": 100_000,
    }
    values.update(overrides)
    return values


def _sampling(**overrides: Any) -> dict[str, Any]:
    values: dict[str, Any] = {
        "max_rows_per_source": 1_000,
        "max_sessions": 1_000,
        "max_decisions_per_session": 64,
        "max_total_decisions": 10_000,
        "max_presented_tokens_per_arm": 100_000_000,
        "qa_target_fraction": 0.15,
        "repetitions": 1,
        "seed": 42,
    }
    values.update(overrides)
    return values


def _policy(**overrides: Any) -> dict[str, Any]:
    values: dict[str, Any] = {
        "mode": "persistent",
        "history_budget_bytes": 10_000_000,
        "workspace_budget_bytes": 10_000_000,
        "lease_decisions": 3,
        "max_retrieved_events": 2,
    }
    values.update(overrides)
    return values


def _case(
    name: str,
    rows: Sequence[Mapping[str, Any]],
    *,
    packing: Mapping[str, Any] | None = None,
    sampling: Mapping[str, Any] | None = None,
    policy: Mapping[str, Any] | None = None,
    source_audit: Mapping[str, int] | None = None,
    allow_unchanged_b: bool = False,
    fixture: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "name": name,
        "rows": list(rows),
        "packing": dict(packing or _packing()),
        "sampling": dict(sampling or _sampling()),
        "policy": dict(policy or _policy()),
        "kv_bytes_per_token": 1,
        "source_audit": dict(source_audit or {}),
        "allow_unchanged_b": allow_unchanged_b,
        "fixture": dict(fixture or {}),
    }


def _stable_rank(source: str, session_id: str, seed: int) -> str:
    session_key = json.dumps([source, session_id], ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(f"{seed}\0{session_key}".encode("utf-8")).hexdigest()


def _ordered_long_and_zero_ids(seed: int = 42) -> tuple[str, str]:
    source = "benchmark:synthetic"
    candidates = [f"cap-{index:02d}" for index in range(32)]
    ranked = sorted(candidates, key=lambda value: (_stable_rank(source, value, seed), value))
    return ranked[0], ranked[-1]


def _clone_actual_row(row: Mapping[str, Any], suffix: str) -> dict[str, Any]:
    cloned = json.loads(json.dumps(row, ensure_ascii=False, allow_nan=False))
    cloned["session_id"] = f"{row['session_id']}:{suffix}"
    cloned["task_id"] = f"{row['task_id']}:{suffix}"
    metadata = dict(cloned.get("source_metadata") or {})
    metadata["benchmark_clone_suffix"] = suffix
    cloned["source_metadata"] = metadata
    return cloned


def _tokenizer_snapshot(cache_root: Path) -> Path | None:
    if (cache_root / "tokenizer.json").is_file():
        return cache_root.resolve()
    refs_main = cache_root / "refs" / "main"
    if refs_main.is_file():
        revision = refs_main.read_text(encoding="utf-8").strip()
        snapshot = cache_root / "snapshots" / revision
        if (snapshot / "tokenizer.json").is_file():
            return snapshot.resolve()
    snapshots = cache_root / "snapshots"
    if snapshots.is_dir():
        matches = sorted(path for path in snapshots.iterdir() if (path / "tokenizer.json").is_file())
        if matches:
            return matches[-1].resolve()
    return None


def _load_actual_trace(
    traces_root: Path,
    *,
    scan_rows: int,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    provenance: dict[str, Any] = {
        "kind": "synthetic",
        "reason": "no bounded real source fixture was available",
        "traces_root": str(traces_root),
        "scan_rows": scan_rows,
    }
    data = traces_root / "data"
    parquet_files = sorted(
        data.glob("*.parquet"),
        key=lambda path: (
            path.name != "train-00035-of-00039.parquet",
            path.stat().st_size,
            path.name,
        ),
    )
    if not parquet_files:
        return None, provenance
    try:
        import pyarrow.parquet as pq

        sys.path.insert(0, str(REPO_ROOT / "python"))
        from history_memory.sources import SourceRowError, adapt_agent_trace_row
    except (ImportError, OSError) as exc:
        provenance["reason"] = f"real source reader unavailable: {type(exc).__name__}"
        return None, provenance

    best: tuple[int, dict[str, Any], Path, int] | None = None
    seen = 0
    rejected: Counter[str] = Counter()
    columns = [
        "benchmark",
        "subset",
        "dataset",
        "task",
        "session_id",
        "trace_id",
        "id",
        "spans",
    ]
    for path in parquet_files:
        parquet = pq.ParquetFile(path)
        available = set(parquet.schema_arrow.names)
        selected_columns = [column for column in columns if column in available]
        file_row_index = 0
        for batch in parquet.iter_batches(batch_size=min(32, scan_rows), columns=selected_columns):
            for raw in batch.to_pylist():
                if seen >= scan_rows:
                    break
                seen += 1
                try:
                    normalized = adapt_agent_trace_row(raw)
                except SourceRowError as exc:
                    rejected[exc.reason] += 1
                    file_row_index += 1
                    continue
                score = len(normalized["messages"])
                if best is None or score > best[0]:
                    best = (score, normalized, path, file_row_index)
                file_row_index += 1
            if seen >= scan_rows:
                break
        if seen >= scan_rows:
            break
    if best is None:
        provenance["reason"] = "no valid agent trace within the bounded scan"
        provenance["rows_scanned"] = seen
        provenance["rejected_by_reason"] = dict(sorted(rejected.items()))
        return None, provenance
    score, normalized, path, row_index = best
    provenance = {
        "kind": "real",
        "source_family": normalized["source"],
        "source_file": str(path.resolve()),
        "source_row_within_batch": row_index,
        "session_id": normalized["session_id"],
        "messages": score,
        "rows_scanned": seen,
        "rejected_by_reason": dict(sorted(rejected.items())),
    }
    return normalized, provenance


def _equivalence_cases(
    actual_row: Mapping[str, Any] | None,
    actual_provenance: Mapping[str, Any],
) -> list[dict[str, Any]]:
    state_rows = [_lease_row()]
    if actual_row is not None:
        state_rows.append(actual_row)
    else:
        state_rows.append(_long_tool_row("representative-long", turns=7))

    nonqa = [_short_row(f"mix-agent-{index}", source="agent") for index in range(17)]
    qa = [_short_row(f"mix-qa-{index}", source="qa:hotpotqa") for index in range(10)]
    long_id, zero_id = _ordered_long_and_zero_ids()
    tight_rows = [
        _long_tool_row(long_id, turns=6, payload_words=48),
        _short_row(zero_id),
    ]
    return [
        _case(
            "stateful-selection",
            state_rows,
            sampling=_sampling(max_decisions_per_session=5, max_total_decisions=8, repetitions=2),
            policy=_policy(lease_decisions=3),
            allow_unchanged_b=True,
            fixture={
                "kind": "mixed-real-and-synthetic" if actual_row is not None else "synthetic",
                "actual_source": dict(actual_provenance),
                "coverage": ["stateful leases", "unselected intermediate decisions", "targets", "policy metadata"],
            },
        ),
        _case(
            "identical-c-b",
            [_short_row("identical-view")],
            packing=_packing(ratios=[8]),
            allow_unchanged_b=True,
            fixture={"kind": "synthetic", "coverage": ["explicitly allowed C=B view"]},
        ),
        _case(
            "qa-mixing",
            [*nonqa, *qa],
            packing=_packing(ratios=[8]),
            sampling=_sampling(qa_target_fraction=0.15),
            allow_unchanged_b=True,
            fixture={"kind": "synthetic", "coverage": ["QA selection before expansion"]},
        ),
        _case(
            "tight-presented-cap",
            tight_rows,
            sampling=_sampling(repetitions=3),
            allow_unchanged_b=True,
            fixture={
                "kind": "synthetic-adversarial",
                "auto_presented_cap": True,
                "long_session_id": long_id,
                "later_zero_gist_session_id": zero_id,
                "coverage": ["oversized skip", "later zero-gist fit", "partial ratio/repetition cap"],
            },
        ),
    ]


def _performance_case(
    actual_row: Mapping[str, Any] | None,
    actual_provenance: Mapping[str, Any],
    *,
    sessions: int,
    turns: int,
    repetitions: int,
    force_synthetic: bool,
) -> dict[str, Any]:
    if (
        not force_synthetic
        and actual_row is not None
        and len(actual_row.get("messages", ())) >= 8
    ):
        rows = [_clone_actual_row(actual_row, f"bench-{index}") for index in range(sessions)]
        fixture = {"kind": "real-cloned-bounded", "actual_source": dict(actual_provenance)}
    else:
        rows = [
            _long_tool_row(f"benchmark-long-{index}", turns=turns, payload_words=64)
            for index in range(sessions)
        ]
        fixture = {
            "kind": "synthetic-representative-long-tool-conversation",
            "actual_source": dict(actual_provenance),
        }
    fixture.update(
        {
            "sessions": sessions,
            "synthetic_turns_when_used": turns,
            "repetitions": repetitions,
            "bounded": True,
        }
    )
    return _case(
        "bounded-performance",
        rows,
        packing=_packing(
            max_chunks=max(512, turns * 4),
            max_encoder_tokens=500_000,
            max_system_tokens=100_000,
            max_workspace_tokens=100_000,
            max_target_tokens=32_768,
            max_sequence_tokens=500_000,
        ),
        sampling=_sampling(
            max_sessions=sessions,
            max_decisions_per_session=64,
            max_total_decisions=sessions * 64,
            repetitions=repetitions,
        ),
        allow_unchanged_b=True,
        fixture=fixture,
    )


class CountingTokenizer:
    """Transparent tokenizer proxy whose counters never enter prepared output."""

    def __init__(self, tokenizer: Any) -> None:
        self._tokenizer = tokenizer
        self.calls = 0
        self.input_characters = 0
        self.output_tokens = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self._tokenizer, name)

    def apply_chat_template(self, messages: Any, **kwargs: Any) -> Any:
        self.calls += 1
        tools = kwargs.get("tools")
        self.input_characters += len(_canonical_json(messages))
        if tools is not None:
            self.input_characters += len(_canonical_json(tools))
        result = self._tokenizer.apply_chat_template(messages, **kwargs)
        ids = result.input_ids if hasattr(result, "input_ids") else result
        self.output_tokens += len(ids)
        return result


@dataclass
class _RssSampler:
    baseline: int = 0
    peak: int = 0

    def __post_init__(self) -> None:
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        try:
            import psutil

            process = psutil.Process(os.getpid())
        except ImportError:
            return
        self.baseline = process.memory_info().rss
        self.peak = self.baseline

        def sample() -> None:
            while not self._stop.wait(0.005):
                self.peak = max(self.peak, process.memory_info().rss)

        self._thread = threading.Thread(target=sample, name="history-prepare-rss", daemon=True)
        self._thread.start()

    def finish(self) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=1.0)


def _worker(args: argparse.Namespace) -> int:
    source_root = Path(args.source_root).resolve()
    sys.path.insert(0, str(source_root / "python"))
    from transformers import AutoTokenizer

    from history_memory.policy import RuntimeConfig
    from history_memory.preparation import PackingConfig, SamplingConfig, prepare_paired_corpus

    payload = _json_load(Path(args.case_json))
    inner = AutoTokenizer.from_pretrained(args.tokenizer_path, local_files_only=True)
    tokenizer = CountingTokenizer(inner)
    sampler = _RssSampler()
    sampler.start()
    started = time.perf_counter()
    manifest = prepare_paired_corpus(
        payload["rows"],
        tokenizer,
        args.output_dir,
        packing=PackingConfig(**{**payload["packing"], "ratios": tuple(payload["packing"]["ratios"])}),
        sampling=SamplingConfig(**payload["sampling"]),
        policy_config=RuntimeConfig(**payload["policy"]),
        kv_bytes_per_token=payload["kv_bytes_per_token"],
        source_audit=payload["source_audit"],
        allow_unchanged_b=payload["allow_unchanged_b"],
    )
    elapsed = time.perf_counter() - started
    sampler.finish()
    disk_manifest = _json_load(Path(args.output_dir) / "manifest.json")
    if manifest != disk_manifest:
        raise AssertionError("prepare_paired_corpus return value differs from manifest.json")
    result = {
        "source_root": str(source_root),
        "manifest": manifest,
        "metrics": {
            "wall_seconds": elapsed,
            "tokenizer_calls": tokenizer.calls,
            "tokenizer_input_characters": tokenizer.input_characters,
            "tokenizer_output_tokens": tokenizer.output_tokens,
            "rss_baseline_bytes": sampler.baseline or None,
            "rss_peak_bytes": sampler.peak or None,
            "rss_peak_delta_bytes": max(0, sampler.peak - sampler.baseline) if sampler.peak else None,
        },
    }
    _json_dump(Path(args.worker_result), result)
    return 0


def _run_checked(command: Sequence[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    environment = dict(os.environ)
    environment.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    completed = subprocess.run(
        list(command),
        cwd=cwd,
        env=environment,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode:
        raise RuntimeError(
            f"Subprocess failed ({completed.returncode}): {command!r}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    return completed


def _export_baseline(destination: Path) -> str:
    resolved = subprocess.run(
        ["git", "rev-parse", f"{BASELINE_REVISION}^{{commit}}"],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()
    if resolved != BASELINE_REVISION:
        raise AssertionError(f"Baseline resolved to unexpected commit: {resolved}")
    archive = destination.parent / "baseline.zip"
    subprocess.run(
        ["git", "archive", "--format=zip", "-o", str(archive), resolved],
        cwd=REPO_ROOT,
        check=True,
    )
    destination.mkdir(parents=True)
    with zipfile.ZipFile(archive) as handle:
        handle.extractall(destination)
    return resolved


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _artifact_summary(root: Path) -> dict[str, Any]:
    records = _read_jsonl(root / "paired_decisions.jsonl")
    sessions = _read_jsonl(root / "sessions.jsonl")
    return {
        "sha256": {
            filename: hashlib.sha256((root / filename).read_bytes()).hexdigest()
            for filename in ARTIFACT_FILES
        },
        "bytes": {filename: (root / filename).stat().st_size for filename in ARTIFACT_FILES},
        "sessions": len(sessions),
        "exposures": len(records),
        "base_decisions": len({record["decision_id"] for record in records}),
    }


def _compare_artifacts(baseline: Path, current: Path) -> dict[str, Any]:
    mismatches = [
        filename
        for filename in ARTIFACT_FILES
        if (baseline / filename).read_bytes() != (current / filename).read_bytes()
    ]
    return {
        "exact": not mismatches,
        "mismatched_files": mismatches,
        "compared_files": list(ARTIFACT_FILES),
        "compared_semantics": [
            "selected decisions and order",
            "session IDs and remapping",
            "targets and target hashes",
            "C/B memory views",
            "per-ratio costs",
            "B policy metadata",
            "skip counts and statistics",
            "full manifest including file integrity",
        ],
    }


def _coverage(case: Mapping[str, Any], output: Path) -> dict[str, Any]:
    manifest = _json_load(output / "manifest.json")
    records = _read_jsonl(output / "paired_decisions.jsonl")
    name = case["name"]
    if name == "stateful-selection":
        policies = [record["arms"]["B"]["policy"] for record in records]
        return {
            "unselected_intermediate_decisions": manifest["counts"].get(
                "decisions_advanced_but_not_selected", 0
            )
            > 0,
            "retrieval": any(policy.get("retrieved_event_ids") for policy in policies),
            "retention": any(policy.get("retained_event_ids") for policy in policies),
            "lease_expiry": any(policy.get("expired_lease_event_ids") for policy in policies),
        }
    if name == "identical-c-b":
        return {
            "allow_unchanged_b": manifest["allow_unchanged_b"] is True,
            "all_views_identical": all(
                record["arms"]["C"]["view"] == record["arms"]["B"]["view"]
                for record in records
            ),
        }
    if name == "qa-mixing":
        qa_base = {
            record["decision_id"]
            for record in records
            if record["source"].startswith("qa:")
        }
        return {
            "qa_base_decisions": len(qa_base),
            "actual_qa_base_decision_fraction": manifest["statistics"][
                "actual_qa_base_decision_fraction"
            ],
            "target_fraction": case["sampling"]["qa_target_fraction"],
        }
    if name == "tight-presented-cap":
        long_id = case["fixture"]["long_session_id"]
        zero_id = case["fixture"]["later_zero_gist_session_id"]
        counts_by_decision: Counter[str] = Counter(record["decision_id"] for record in records)
        full_exposures = len(case["packing"]["ratios"]) * case["sampling"]["repetitions"]
        return {
            "presented_cap": case["sampling"]["max_presented_tokens_per_arm"],
            "skipped_exposures": manifest["counts"].get(
                "decision_pairs_skipped_presented_token_cap", 0
            ),
            "partial_decision": any(0 < count < full_exposures for count in counts_by_decision.values()),
            "long_session_precedes_zero_session": next(
                session["session_id"] for session in _read_jsonl(output / "sessions.jsonl")
            )
            == long_id,
            "later_zero_gist_pair_written": any(
                record["session_key"]
                == json.dumps(
                    ["benchmark:synthetic", zero_id], ensure_ascii=False, separators=(",", ":")
                )
                and record["arms"]["C"]["costs"]["gist_tokens"] == 0
                and record["arms"]["B"]["costs"]["gist_tokens"] == 0
                for record in records
            ),
        }
    return {"records_written": len(records)}


def _calibrate_presented_cap(
    case: dict[str, Any],
    *,
    baseline_root: Path,
    tokenizer_path: Path,
    work: Path,
    python: Path,
) -> int:
    calibration = json.loads(json.dumps(case, ensure_ascii=False))
    calibration["name"] = "tight-presented-cap-calibration"
    calibration["sampling"]["max_presented_tokens_per_arm"] = 100_000_000
    result = _run_one(
        calibration,
        label="calibration-baseline",
        source_root=baseline_root,
        tokenizer_path=tokenizer_path,
        work=work / "calibration",
        python=python,
    )
    records = _read_jsonl(Path(result["output_dir"]) / "paired_decisions.jsonl")
    long_key = json.dumps(
        ["benchmark:synthetic", case["fixture"]["long_session_id"]],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    cumulative = {"C": 0, "B": 0}
    for record in records:
        for arm in ("C", "B"):
            cumulative[arm] += record["arms"][arm]["costs"]["presented_encoder_tokens"]
        if record["session_key"] == long_key and max(
            record["arms"][arm]["costs"]["presented_encoder_tokens"] for arm in ("C", "B")
        ):
            return max(cumulative.values())
    raise AssertionError("Tight-cap calibration found no nonzero presented-token exposure")


def _run_one(
    case: Mapping[str, Any],
    *,
    label: str,
    source_root: Path,
    tokenizer_path: Path,
    work: Path,
    python: Path,
) -> dict[str, Any]:
    work.mkdir(parents=True, exist_ok=True)
    case_json = work / "case.json"
    output_dir = work / "prepared"
    result_json = work / "worker_result.json"
    _json_dump(case_json, case)
    command = [
        str(python),
        str(Path(__file__).resolve()),
        "--worker",
        "--source-root",
        str(source_root),
        "--case-json",
        str(case_json),
        "--output-dir",
        str(output_dir),
        "--worker-result",
        str(result_json),
        "--tokenizer-path",
        str(tokenizer_path),
    ]
    _run_checked(command, cwd=source_root)
    result = _json_load(result_json)
    result.update(
        {
            "label": label,
            "output_dir": str(output_dir),
            "artifacts": _artifact_summary(output_dir),
        }
    )
    return result


def _run_pair(
    case: dict[str, Any],
    *,
    baseline_root: Path,
    current_root: Path,
    tokenizer_path: Path,
    work: Path,
    python: Path,
) -> dict[str, Any]:
    if case.get("fixture", {}).get("auto_presented_cap"):
        cap = _calibrate_presented_cap(
            case,
            baseline_root=baseline_root,
            tokenizer_path=tokenizer_path,
            work=work,
            python=python,
        )
        case["sampling"]["max_presented_tokens_per_arm"] = cap
        case["fixture"]["calibrated_presented_cap"] = cap
    baseline = _run_one(
        case,
        label="baseline",
        source_root=baseline_root,
        tokenizer_path=tokenizer_path,
        work=work / "baseline",
        python=python,
    )
    current = _run_one(
        case,
        label="current",
        source_root=current_root,
        tokenizer_path=tokenizer_path,
        work=work / "current",
        python=python,
    )
    comparison = _compare_artifacts(
        Path(baseline["output_dir"]), Path(current["output_dir"])
    )
    if baseline["manifest"] != current["manifest"]:
        comparison["exact"] = False
        comparison["returned_manifest_mismatch"] = True
    coverage = _coverage(case, Path(current["output_dir"]))
    before = baseline["metrics"]
    after = current["metrics"]
    performance = {
        "wall_speedup": before["wall_seconds"] / after["wall_seconds"]
        if after["wall_seconds"]
        else None,
        "tokenizer_call_reduction_fraction": 1.0
        - after["tokenizer_calls"] / before["tokenizer_calls"]
        if before["tokenizer_calls"]
        else None,
        "tokenizer_input_character_reduction_fraction": 1.0
        - after["tokenizer_input_characters"] / before["tokenizer_input_characters"]
        if before["tokenizer_input_characters"]
        else None,
    }
    return {
        "name": case["name"],
        "fixture": case["fixture"],
        "config": {
            "packing": case["packing"],
            "sampling": case["sampling"],
            "policy": case["policy"],
        },
        "baseline": baseline,
        "current": current,
        "comparison": comparison,
        "coverage": coverage,
        "performance": performance,
    }


def _driver(args: argparse.Namespace) -> int:
    tokenizer_path = (
        Path(args.tokenizer_path).resolve()
        if args.tokenizer_path
        else _tokenizer_snapshot(DEFAULT_TOKENIZER_CACHE)
    )
    if tokenizer_path is None or not (tokenizer_path / "tokenizer.json").is_file():
        raise FileNotFoundError(
            "A local Qwen3-4B-Instruct-2507 tokenizer snapshot is required; "
            "pass --tokenizer-path"
        )
    python = Path(args.python).resolve() if args.python else Path(sys.executable).resolve()
    if not python.is_file():
        raise FileNotFoundError(f"Python executable does not exist: {python}")

    owned_temp: tempfile.TemporaryDirectory[str] | None = None
    if args.work_dir:
        work = Path(args.work_dir).resolve()
        work.mkdir(parents=True, exist_ok=True)
    else:
        owned_temp = tempfile.TemporaryDirectory(prefix="history-prepare-differential-")
        work = Path(owned_temp.name)
    output_json = (
        Path(args.output_json).resolve()
        if args.output_json
        else REPO_ROOT
        / "outputs"
        / "history_prepare_benchmark"
        / f"result-{time.strftime('%Y%m%d-%H%M%S')}.json"
    )

    baseline_root = work / "baseline-source"
    resolved = _export_baseline(baseline_root)
    actual_row, actual_provenance = _load_actual_trace(
        Path(args.traces_root).resolve() if args.traces_root else DEFAULT_TRACES_ROOT,
        scan_rows=args.real_scan_rows,
    )
    cases: list[dict[str, Any]] = []
    if args.suite in {"equivalence", "all"}:
        cases.extend(_equivalence_cases(actual_row, actual_provenance))
    if args.suite in {"benchmark", "all"}:
        cases.append(
            _performance_case(
                actual_row,
                actual_provenance,
                sessions=args.benchmark_sessions,
                turns=args.benchmark_turns,
                repetitions=args.benchmark_repetitions,
                force_synthetic=args.force_synthetic,
            )
        )

    results = []
    for case in cases:
        result = _run_pair(
            case,
            baseline_root=baseline_root,
            current_root=REPO_ROOT,
            tokenizer_path=tokenizer_path,
            work=work / "cases" / case["name"],
            python=python,
        )
        results.append(result)
    payload = {
        "schema_version": "history-prepare-benchmark-v1",
        "baseline_revision": resolved,
        "baseline_method": "git archive read-only export; separate Python worker",
        "current_method": "current working tree; separate Python worker",
        "python": str(python),
        "tokenizer_path": str(tokenizer_path),
        "actual_source_fixture": actual_provenance,
        "suite": args.suite,
        "all_exact": all(result["comparison"]["exact"] for result in results),
        "cases": results,
    }
    _json_dump(output_json, payload)
    print(_canonical_json({"output_json": str(output_json), "all_exact": payload["all_exact"]}))
    if owned_temp is not None:
        owned_temp.cleanup()
    return 0 if payload["all_exact"] else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("equivalence", "benchmark", "all"), default="all")
    parser.add_argument("--python", help="Python executable used for both isolated workers")
    parser.add_argument("--tokenizer-path", help="Local tokenizer snapshot; network is never used")
    parser.add_argument("--traces-root", help="Optional local agent-llm-traces checkout")
    parser.add_argument("--real-scan-rows", type=int, default=96)
    parser.add_argument("--benchmark-sessions", type=int, default=4)
    parser.add_argument("--benchmark-turns", type=int, default=8)
    parser.add_argument("--benchmark-repetitions", type=int, default=1)
    parser.add_argument(
        "--force-synthetic",
        action="store_true",
        help="Use scalable synthetic long-tool rows even when a real trace is available",
    )
    parser.add_argument("--work-dir", help="Keep isolated sources and artifacts under this directory")
    parser.add_argument("--output-json", help="Machine-readable result path (default: ignored outputs/)")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--source-root", help=argparse.SUPPRESS)
    parser.add_argument("--case-json", help=argparse.SUPPRESS)
    parser.add_argument("--output-dir", help=argparse.SUPPRESS)
    parser.add_argument("--worker-result", help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.real_scan_rows < 1:
        parser.error("--real-scan-rows must be positive")
    if args.benchmark_sessions < 1 or args.benchmark_turns < 2 or args.benchmark_repetitions < 1:
        parser.error("benchmark sessions/repetitions must be positive and turns must be >= 2")
    if args.worker:
        required = ("source_root", "case_json", "output_dir", "worker_result", "tokenizer_path")
        missing = [name for name in required if not getattr(args, name)]
        if missing:
            parser.error(f"worker mode missing: {', '.join(missing)}")
        return _worker(args)
    return _driver(args)


if __name__ == "__main__":
    raise SystemExit(main())
