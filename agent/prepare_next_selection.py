#!/usr/bin/env python3
"""Build a session-disjoint checkpoint-selection dev corpus on CPU.

The source rows stay on their frozen ``train`` split while the existing history
and tool packers run.  Only the serialized copies are relabelled with the
``checkpoint_selection_dev`` purpose.  Selection is session-level: every source
session is checked against every H0/H1/H2/H3/T0/T1 training corpus before any
decision from it can be selected.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import sys
import tarfile
from collections import Counter, defaultdict
from dataclasses import dataclass, fields, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from history_memory.dataset import event_store_session_id, iter_decision_metadata
from history_memory.preparation import tokenizer_identity
from next_compression.common import (
    SCHEMA,
    TRAINING_PROFILE,
    deserialize_memory,
    sha256_file,
    validate_record,
)
from next_compression.history import HistoryPreparationConfig, iter_history_records
from next_compression.tools import (
    TOOL_VARIANTS,
    ToolPreparationConfig,
    iter_tool_records,
)


PURPOSE = "checkpoint_selection_dev"
NORMALIZED_SCHEMA = "next-compression-normalized-sources-v1"
VARIANTS = ("H0", "H1", "H2", "H3", "T0", "T1")
HISTORY_VARIANTS = ("H0", "H1", "H2", "H3")
HISTORY_PACKED_VARIANTS = ("H1", "H2", "H3")
UNIFORM_EVAL_PROFILE = "decision-mean-uniform-ce-v1"


@dataclass(frozen=True)
class SourceSpec:
    key: str
    path: Path
    manifest_entry: Mapping[str, Any]


@dataclass(frozen=True)
class TrainingInventory:
    manifests: Mapping[str, Mapping[str, Any]]
    sessions_by_variant: Mapping[str, frozenset[str]]
    manifest_sha256: Mapping[str, str]
    records_sha256: Mapping[str, str]

    @property
    def all_sessions(self) -> frozenset[str]:
        return frozenset().union(*self.sessions_by_variant.values())


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _record_session_key(line: bytes, *, path: Path, line_number: int) -> str:
    """Read the early top-level session_key without parsing large token arrays."""

    prefix = line[:16_384].decode("utf-8")
    marker = '"session_key":'
    start = prefix.find(marker)
    if start < 0:
        raise ValueError(f"Missing session_key at {path}:{line_number}")
    value, _ = json.JSONDecoder().raw_decode(prefix, start + len(marker))
    if not isinstance(value, str) or not value:
        raise ValueError(f"Invalid session_key at {path}:{line_number}")
    return value


def _scan_training_records(
    path: Path, info: Mapping[str, Any]
) -> tuple[frozenset[str], str]:
    expected_size = info.get("bytes")
    expected_hash = info.get("sha256")
    expected_count = info.get("count")
    if type(expected_size) is not int or expected_size < 0:
        raise ValueError(f"Invalid training records byte count: {path}")
    if not isinstance(expected_hash, str) or len(expected_hash) != 64:
        raise ValueError(f"Invalid training records SHA256: {path}")
    if type(expected_count) is not int or expected_count < 1:
        raise ValueError(f"Invalid training records count: {path}")

    digest = hashlib.sha256()
    sessions: set[str] = set()
    byte_count = 0
    line_count = 0
    with path.open("rb") as handle:
        for line_count, line in enumerate(handle, start=1):
            digest.update(line)
            byte_count += len(line)
            sessions.add(
                _record_session_key(line, path=path, line_number=line_count)
            )
    actual_hash = digest.hexdigest()
    if byte_count != expected_size or actual_hash != expected_hash:
        raise ValueError(f"Training records integrity mismatch: {path}")
    if line_count != expected_count:
        raise ValueError(
            f"Training records count mismatch: {path}: {line_count} != {expected_count}"
        )
    return frozenset(sessions), actual_hash


def load_training_inventory(root: str | Path) -> TrainingInventory:
    root = Path(root).resolve()
    manifests: dict[str, Mapping[str, Any]] = {}
    sessions: dict[str, frozenset[str]] = {}
    manifest_hashes: dict[str, str] = {}
    records_hashes: dict[str, str] = {}
    for variant in VARIANTS:
        manifest_path = root / variant / "manifest.json"
        manifest = _load_json(manifest_path)
        if manifest.get("schema") != SCHEMA:
            raise ValueError(f"Unexpected training schema for {variant}")
        if manifest.get("training_profile") != TRAINING_PROFILE:
            raise ValueError(f"Unexpected training profile for {variant}")
        if manifest.get("variant") != variant:
            raise ValueError(f"Training manifest variant mismatch: {variant}")
        records_info = manifest.get("records")
        if not isinstance(records_info, Mapping):
            raise ValueError(f"Training manifest has no records object: {variant}")
        records_path = (manifest_path.parent / str(records_info.get("path"))).resolve()
        if records_path.parent != manifest_path.parent.resolve():
            raise ValueError(f"Training records escape their variant directory: {variant}")
        variant_sessions, records_hash = _scan_training_records(
            records_path, records_info
        )
        manifests[variant] = manifest
        sessions[variant] = variant_sessions
        manifest_hashes[variant] = sha256_file(manifest_path)
        records_hashes[variant] = records_hash
        print(
            json.dumps(
                {
                    "event": "training_corpus_verified",
                    "variant": variant,
                    "records": records_info["count"],
                    "sessions": len(variant_sessions),
                    "manifest_sha256": manifest_hashes[variant],
                }
            ),
            flush=True,
        )
    return TrainingInventory(
        manifests=manifests,
        sessions_by_variant=sessions,
        manifest_sha256=manifest_hashes,
        records_sha256=records_hashes,
    )


def assert_session_disjoint(
    selected_session_ids: Iterable[str],
    sessions_by_variant: Mapping[str, Iterable[str]],
) -> None:
    selected = set(selected_session_ids)
    overlaps = {
        variant: sorted(selected.intersection(training_sessions))
        for variant, training_sessions in sessions_by_variant.items()
    }
    overlaps = {variant: values for variant, values in overlaps.items() if values}
    if overlaps:
        summary = ", ".join(
            f"{variant}={len(values)}" for variant, values in sorted(overlaps.items())
        )
        examples = {
            variant: values[:3] for variant, values in sorted(overlaps.items())
        }
        raise ValueError(
            f"Checkpoint-selection sessions overlap trained corpora ({summary}): "
            f"{examples}"
        )


def _load_normalized_manifest(path: Path) -> dict[str, Any]:
    manifest = _load_json(path)
    if manifest.get("schema") != NORMALIZED_SCHEMA:
        raise ValueError(f"Normalized manifest must use {NORMALIZED_SCHEMA!r}")
    if not isinstance(manifest.get("sources"), Mapping) or not manifest["sources"]:
        raise ValueError("Normalized manifest has no source files")
    return manifest


def _source_specs(
    manifest_path: Path,
    manifest: Mapping[str, Any],
    selected: Sequence[str] | None = None,
) -> tuple[SourceSpec, ...]:
    entries = manifest["sources"]
    keys = tuple(entries) if selected is None else tuple(selected)
    unknown = [key for key in keys if key not in entries]
    if unknown:
        raise ValueError(f"Unknown normalized sources: {unknown}")
    root = manifest_path.parent.resolve()
    specs = []
    for key in keys:
        entry = entries[key]
        if not isinstance(entry, Mapping):
            raise ValueError(f"Source entry {key!r} must be an object")
        relative = entry.get("path")
        if not isinstance(relative, str) or not relative:
            raise ValueError(f"Source entry {key!r} has no path")
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"Source path escapes manifest directory: {key}") from exc
        if not path.is_file():
            raise FileNotFoundError(path)
        expected_size = entry.get("bytes")
        expected_hash = entry.get("sha256")
        if type(expected_size) is not int or expected_size < 0:
            raise ValueError(f"Invalid normalized source byte count: {key}")
        if not isinstance(expected_hash, str) or len(expected_hash) != 64:
            raise ValueError(f"Invalid normalized source SHA256: {key}")
        if path.stat().st_size != expected_size or sha256_file(path) != expected_hash:
            raise ValueError(f"Normalized source integrity mismatch: {key}")
        specs.append(SourceSpec(key, path, entry))
        print(
            json.dumps(
                {
                    "event": "normalized_source_verified",
                    "source": key,
                    "bytes": expected_size,
                    "sha256": expected_hash,
                }
            ),
            flush=True,
        )
    return tuple(specs)


def _iter_rows(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected a source object at {path}:{line_number}")
            if row.get("split") != "train":
                raise ValueError(
                    f"Source row must remain train for packing at {path}:{line_number}"
                )
            yield row


def _round_robin_rows(specs: Sequence[SourceSpec]) -> Iterator[tuple[str, dict[str, Any]]]:
    streams = {spec.key: _iter_rows(spec.path) for spec in specs}
    active = [spec.key for spec in specs]
    try:
        while active:
            following = []
            for key in active:
                try:
                    row = next(streams[key])
                except StopIteration:
                    continue
                following.append(key)
                yield key, row
            active = following
    finally:
        for stream in streams.values():
            stream.close()


def _row_identity(row: Mapping[str, Any]) -> str:
    source = row.get("source")
    session_id = row.get("session_id")
    if not isinstance(source, str) or not source:
        raise ValueError("Normalized row has no source identity")
    if not isinstance(session_id, str) or not session_id:
        raise ValueError("Normalized row has no session identity")
    return event_store_session_id(source, session_id)


def _decision_info(row: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    metadata = tuple(iter_decision_metadata(row))
    result: dict[str, dict[str, Any]] = {}
    for item in metadata:
        message = row["messages"][item.source_message_index]
        tool_calls = copy.deepcopy(message.get("tool_calls") or [])
        kind = (
            "tool_call"
            if tool_calls
            else "terminal_stop"
            if item.decision_index == len(metadata) - 1
            else "non_tool_response"
        )
        gold_action: dict[str, Any] = {"role": "assistant"}
        if "content" in message:
            gold_action["content"] = copy.deepcopy(message["content"])
        if tool_calls:
            gold_action["tool_calls"] = tool_calls
        result[item.decision_id] = {
            "decision_type": kind,
            "decision_index": item.decision_index,
            "source_message_index": item.source_message_index,
            "task_id": item.task_id,
            "template_id": item.template_id,
            "source_session_id": item.session_id,
            "gold_action": gold_action,
            "gold_tool_calls": tool_calls,
        }
    return result


def _exact_dataclass_config(
    raw: Mapping[str, Any], config_type: type[Any], **overrides: Any
) -> Any:
    """Construct from every serialized field; never silently fill a limit."""

    names = {field.name for field in fields(config_type)}
    missing = sorted(
        name
        for name in names
        if name not in raw
        and name not in overrides
        and name not in {"h0_decision_ids", "h1_decision_ids"}
    )
    if missing:
        raise ValueError(f"Training preparation config omits fields: {missing}")
    values = {
        name: copy.deepcopy(raw[name])
        for name in names
        if name in raw and name not in {"h0_decision_ids", "h1_decision_ids"}
    }
    for name in ("variants", "ratios"):
        if name in values:
            values[name] = tuple(values[name])
    values.update(overrides)
    return config_type(**values)


def _training_preparation_config(
    inventory: TrainingInventory, variants: Sequence[str]
) -> dict[str, Any]:
    configs = []
    for variant in variants:
        preparation = inventory.manifests[variant].get("preparation")
        if not isinstance(preparation, Mapping) or not isinstance(
            preparation.get("config"), Mapping
        ):
            raise ValueError(f"Training manifest omits preparation.config: {variant}")
        configs.append(copy.deepcopy(dict(preparation["config"])))
    first = configs[0]
    if any(config != first for config in configs[1:]):
        raise ValueError(f"Training preparation configs differ across {variants}")
    return first


def _group_records(
    records: Iterable[tuple[str, Mapping[str, Any]]]
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for variant, record in records:
        grouped[str(record["decision_id"])][variant].append(dict(record))
    return {key: dict(value) for key, value in grouped.items()}


def _decorate_record(
    record: Mapping[str, Any],
    *,
    variant: str,
    info: Mapping[str, Any],
) -> dict[str, Any]:
    result = copy.deepcopy(dict(record))
    result["split"] = PURPOSE
    metadata = dict(result.get("metadata") or {})
    metadata.update(
        {
            "purpose": PURPOSE,
            "variant": variant,
            "decision_type": info["decision_type"],
            "decision_index": info["decision_index"],
            "source_message_index": info["source_message_index"],
            "task_id": info["task_id"],
            "template_id": info["template_id"],
            "source_session_id": info["source_session_id"],
            "gold_action": copy.deepcopy(info["gold_action"]),
            "evaluation_target_weighting": "uniform",
            "view_target_independence": "existing-packer-prefix-only-view",
        }
    )
    if info["gold_tool_calls"]:
        metadata["gold_tool_calls"] = copy.deepcopy(info["gold_tool_calls"])
    else:
        metadata.pop("gold_tool_calls", None)
    if variant == "H3":
        original = metadata.pop("target_weighting", None)
        if original is not None:
            metadata["training_packer_target_weighting"] = original
        result["target_weights"] = [1.0] * len(result["target_ids"])
    else:
        result.pop("target_weights", None)
    result["metadata"] = metadata
    _validate_dev_record(result)
    return result


def _validate_dev_record(record: Mapping[str, Any]) -> None:
    if record.get("split") != PURPOSE:
        raise ValueError(f"Selection record split must be {PURPOSE!r}")
    if record.get("metadata", {}).get("purpose") != PURPOSE:
        raise ValueError("Selection record metadata must carry its purpose")
    training_copy = copy.deepcopy(dict(record))
    training_copy["split"] = "train"
    validate_record(training_copy)


def _take_one_kind(
    available: Mapping[str, str], counts: Counter[str], quotas: Mapping[str, int]
) -> str | None:
    choices = [kind for kind in quotas if kind in available and counts[kind] < quotas[kind]]
    if not choices:
        return None
    return max(
        choices,
        key=lambda kind: (
            (quotas[kind] - counts[kind]) / quotas[kind],
            quotas[kind] - counts[kind],
            -tuple(quotas).index(kind),
        ),
    )


def select_history(
    specs: Sequence[SourceSpec],
    tokenizer: Any,
    config: HistoryPreparationConfig,
    training: TrainingInventory,
    quotas: Mapping[str, int],
    max_sessions_scanned: int,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    records = {variant: [] for variant in HISTORY_VARIANTS}
    counts: Counter[str] = Counter()
    selected_sessions: list[str] = []
    audit: Counter[str] = Counter()
    expected = sum(quotas.values())
    for source_key, row in _round_robin_rows(specs):
        if audit["sessions.scanned"] >= max_sessions_scanned:
            break
        audit["sessions.scanned"] += 1
        audit[f"sessions.scanned.source.{source_key}"] += 1
        session_key = _row_identity(row)
        if session_key in training.all_sessions:
            audit["sessions.skipped_training_overlap"] += 1
            continue
        info = _decision_info(row)
        candidate_ids: dict[str, str] = {}
        for decision_id, item in info.items():
            kind = item["decision_type"]
            if kind in quotas and counts[kind] < quotas[kind] and kind not in candidate_ids:
                candidate_ids[kind] = decision_id
        if not candidate_ids:
            audit["sessions.skipped_no_needed_decision_type"] += 1
            continue
        explicit_ids = frozenset(candidate_ids.values())
        row_config = replace(config, h1_decision_ids=explicit_ids)
        stream, pack_audit = iter_history_records((row,), tokenizer, row_config)
        grouped = _group_records(stream)
        audit.update({f"packer.{key}": value for key, value in pack_audit.items()})
        available: dict[str, str] = {}
        for kind, decision_id in candidate_ids.items():
            variants = grouped.get(decision_id, {})
            if set(variants) != set(HISTORY_PACKED_VARIANTS):
                continue
            if any(
                sorted(record["ratio"] for record in variants[variant]) != [8, 12]
                for variant in HISTORY_PACKED_VARIANTS
            ):
                continue
            available[kind] = decision_id
        chosen_kind = _take_one_kind(available, counts, quotas)
        if chosen_kind is None:
            audit["sessions.skipped_no_jointly_packable_needed_decision"] += 1
            continue
        decision_id = available[chosen_kind]
        selected = grouped[decision_id]
        decorated: dict[str, list[dict[str, Any]]] = {}
        for variant in HISTORY_PACKED_VARIANTS:
            decorated[variant] = [
                _decorate_record(record, variant=variant, info=info[decision_id])
                for record in selected[variant]
            ]
        decorated["H0"] = []
        for record in decorated["H1"]:
            cloned = copy.deepcopy(record)
            cloned["metadata"]["variant"] = "H0"
            cloned["metadata"]["view_policy"] = "static-recent-tool-one"
            cloned["metadata"]["selection_input_pair"] = "H0-H1-static-identical-v1"
            _validate_dev_record(cloned)
            decorated["H0"].append(cloned)
        for variant in HISTORY_VARIANTS:
            records[variant].extend(decorated[variant])
        counts[chosen_kind] += 1
        selected_sessions.append(session_key)
        audit["decisions.selected"] += 1
        audit[f"decisions.selected.type.{chosen_kind}"] += 1
        audit[f"decisions.selected.source.{row['source']}"] += 1
        print(
            json.dumps(
                {
                    "event": "history_selection_progress",
                    "sessions_scanned": audit["sessions.scanned"],
                    "decisions": sum(counts.values()),
                    "type_counts": dict(counts),
                }
            ),
            flush=True,
        )
        if sum(counts.values()) == expected:
            break
    if dict(counts) != dict(quotas):
        raise RuntimeError(
            "History selection could not fill quotas within bounded scan: "
            f"filled={dict(counts)}, required={dict(quotas)}, "
            f"max_sessions_scanned={max_sessions_scanned}"
        )
    assert_session_disjoint(selected_sessions, training.sessions_by_variant)
    return records, {
        "audit": dict(sorted(audit.items())),
        "selected_session_ids": selected_sessions,
        "type_counts": dict(counts),
        "quotas": dict(quotas),
    }


def _tool_decision_type(record: Mapping[str, Any]) -> str:
    metadata = record["metadata"]
    if metadata["target_kind"] == "covered_tool_call":
        return "tool_call"
    return (
        "terminal_stop"
        if metadata.get("source_grounded_terminal_stop")
        else "non_tool_response"
    )


def select_tools(
    specs: Sequence[SourceSpec],
    tokenizer: Any,
    config: ToolPreparationConfig,
    training: TrainingInventory,
    quotas: Mapping[str, int],
    max_sessions_scanned: int,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    records = {variant: [] for variant in TOOL_VARIANTS}
    counts: Counter[str] = Counter()
    selected_sessions: list[str] = []
    audit: Counter[str] = Counter()
    expected = sum(quotas.values())
    for source_key, row in _round_robin_rows(specs):
        if audit["sessions.scanned"] >= max_sessions_scanned:
            break
        audit["sessions.scanned"] += 1
        audit[f"sessions.scanned.source.{source_key}"] += 1
        session_key = _row_identity(row)
        if session_key in training.all_sessions:
            audit["sessions.skipped_training_overlap"] += 1
            continue
        info = _decision_info(row)
        stream, pack_audit = iter_tool_records((row,), tokenizer, config)
        grouped = _group_records(stream)
        audit.update({f"packer.{key}": value for key, value in pack_audit.items()})
        available: dict[str, str] = {}
        for decision_id, variants in grouped.items():
            if set(variants) != set(TOOL_VARIANTS):
                continue
            if any(
                sorted(record["ratio"] for record in variants[variant]) != [8, 12]
                for variant in TOOL_VARIANTS
            ):
                continue
            kind = _tool_decision_type(variants["T0"][0])
            if kind in quotas and counts[kind] < quotas[kind] and kind not in available:
                available[kind] = decision_id
        chosen_kind = _take_one_kind(available, counts, quotas)
        if chosen_kind is None:
            audit["sessions.skipped_no_jointly_packable_needed_decision"] += 1
            continue
        decision_id = available[chosen_kind]
        if decision_id not in info:
            raise AssertionError(f"Packer returned an unknown decision: {decision_id}")
        for variant in TOOL_VARIANTS:
            records[variant].extend(
                _decorate_record(
                    record, variant=variant, info=info[decision_id]
                )
                for record in grouped[decision_id][variant]
            )
        counts[chosen_kind] += 1
        selected_sessions.append(session_key)
        audit["decisions.selected"] += 1
        audit[f"decisions.selected.type.{chosen_kind}"] += 1
        audit[f"decisions.selected.source.{row['source']}"] += 1
        print(
            json.dumps(
                {
                    "event": "tool_selection_progress",
                    "sessions_scanned": audit["sessions.scanned"],
                    "decisions": sum(counts.values()),
                    "type_counts": dict(counts),
                }
            ),
            flush=True,
        )
        if sum(counts.values()) == expected:
            break
    if dict(counts) != dict(quotas):
        raise RuntimeError(
            "Tool selection could not fill quotas within bounded scan: "
            f"filled={dict(counts)}, required={dict(quotas)}, "
            f"max_sessions_scanned={max_sessions_scanned}"
        )
    assert_session_disjoint(selected_sessions, training.sessions_by_variant)
    return records, {
        "audit": dict(sorted(audit.items())),
        "selected_session_ids": selected_sessions,
        "type_counts": dict(counts),
        "quotas": dict(quotas),
    }


def validate_selection_records(
    records: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    variants: Sequence[str],
    expected_decisions: int,
) -> None:
    decision_maps: dict[str, dict[str, list[Mapping[str, Any]]]] = {}
    for variant in variants:
        grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for record in records[variant]:
            _validate_dev_record(record)
            grouped[str(record["decision_id"])].append(record)
        if len(grouped) != expected_decisions:
            raise ValueError(
                f"{variant} has {len(grouped)} decisions, expected {expected_decisions}"
            )
        for decision_id, group in grouped.items():
            if sorted(record["ratio"] for record in group) != [8, 12]:
                raise ValueError(f"{variant}/{decision_id} does not contain ratios 8 and 12")
            for record in group:
                effective_weights = record.get("target_weights") or [1.0] * len(
                    record["target_ids"]
                )
                if len(effective_weights) != len(record["target_ids"]) or any(
                    float(weight) != 1.0 for weight in effective_weights
                ):
                    raise ValueError(f"Non-uniform evaluation weights: {variant}/{decision_id}")
                if variant == "H3" and "target_weights" not in record:
                    raise ValueError("H3 dev records must explicitly override training weights")
        decision_maps[variant] = dict(grouped)
    expected_ids = set(decision_maps[variants[0]])
    if any(set(decision_maps[variant]) != expected_ids for variant in variants[1:]):
        raise ValueError(f"Decision IDs are not shared across {variants}")
    for decision_id in expected_ids:
        by_variant = {
            variant: {record["ratio"]: record for record in decision_maps[variant][decision_id]}
            for variant in variants
        }
        for ratio in (8, 12):
            targets = {
                tuple(by_variant[variant][ratio]["target_ids"]) for variant in variants
            }
            if len(targets) != 1:
                raise ValueError(f"Targets differ across variants for {decision_id}/{ratio}")
            if set(HISTORY_VARIANTS) <= set(variants):
                if (
                    by_variant["H0"][ratio]["memory"]
                    != by_variant["H1"][ratio]["memory"]
                ):
                    raise ValueError(f"H0/H1 inputs differ for {decision_id}/{ratio}")
                if (
                    by_variant["H2"][ratio]["memory"]
                    != by_variant["H3"][ratio]["memory"]
                ):
                    raise ValueError(f"H2/H3 inputs differ for {decision_id}/{ratio}")


class DevCorpusWriter:
    """Small immutable writer using the training schema with an explicit purpose."""

    def __init__(self, path: Path, variant: str) -> None:
        self.path = path
        self.variant = variant
        self.path.mkdir(parents=True)
        self.records_path = self.path / "records.jsonl"
        self.handle = self.records_path.open("wb")
        self.count = 0
        self.ratio_counts: Counter[str] = Counter()
        self.source_counts: Counter[str] = Counter()
        self.counters: Counter[str] = Counter()

    def write(self, record: Mapping[str, Any]) -> None:
        _validate_dev_record(record)
        payload = (
            json.dumps(record, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
            + "\n"
        ).encode(encoding="utf-8")
        self.handle.write(payload)
        self.count += 1
        self.ratio_counts[str(record["ratio"])] += 1
        self.source_counts[str(record["source"])] += 1
        memory = deserialize_memory(record["memory"])
        costs = memory.costs(record["ratio"])
        for key in ("presented_encoder_tokens", "gist_tokens", "resident_kv_tokens"):
            self.counters[key] += costs[key]
        self.counters["supervised_tokens"] += len(record["target_ids"])
        self.counters["gist_bearing_records"] += int(bool(memory.chunks))

    def finish(
        self,
        *,
        tokenizer: Mapping[str, Any],
        training_manifest: Mapping[str, Any],
        preparation: Mapping[str, Any],
        source_files: Mapping[str, Any],
        provenance: Mapping[str, Any],
        audit: Mapping[str, Any],
    ) -> dict[str, Any]:
        self.handle.close()
        if self.count < 1 or set(self.ratio_counts) != {"8", "12"}:
            raise ValueError(f"Empty or ratio-incomplete dev corpus: {self.variant}")
        manifest = {
            "schema": SCHEMA,
            "training_profile": TRAINING_PROFILE,
            "purpose": PURPOSE,
            "split": PURPOSE,
            "variant": self.variant,
            "compression_domain": "history" if self.variant.startswith("H") else "tool",
            "render_profile": preparation["render_profile"],
            "loss_profile": training_manifest["loss_profile"],
            "evaluation_loss_profile": UNIFORM_EVAL_PROFILE,
            "ratios": [8, 12],
            "tokenizer": dict(tokenizer),
            "preparation": dict(preparation),
            "source_files": dict(source_files),
            "records": {
                "path": self.records_path.name,
                "sha256": sha256_file(self.records_path),
                "bytes": self.records_path.stat().st_size,
                "count": self.count,
            },
            "ratio_counts": dict(sorted(self.ratio_counts.items())),
            "source_counts": dict(sorted(self.source_counts.items())),
            "counters": dict(sorted(self.counters.items())),
            "audit": dict(audit),
            "provenance": dict(provenance),
        }
        (self.path / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        return manifest


def _source_files_for_manifest(
    specs: Sequence[SourceSpec], scan_audit: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        spec.key: {
            "path": spec.path.name,
            "sha256": spec.manifest_entry["sha256"],
            "bytes": spec.manifest_entry["bytes"],
            "sessions_available": spec.manifest_entry.get("sessions"),
            "sessions_scanned": scan_audit.get(f"sessions.scanned.source.{spec.key}", 0),
        }
        for spec in specs
    }


def _provenance(
    training: TrainingInventory,
    *,
    normalized_manifest_path: Path,
    selected_session_ids: Sequence[str],
) -> dict[str, Any]:
    assert_session_disjoint(selected_session_ids, training.sessions_by_variant)
    return {
        "selection": {
            "profile": "bounded-session-round-robin-type-quota-v1",
            "source_split_seen_by_packers": "train",
            "serialized_output_purpose": PURPOSE,
            "dev_not_independent_official_holdout": True,
            "target_used_for_memory_view": False,
            "normalized_manifest_sha256": sha256_file(normalized_manifest_path),
        },
        "training_session_disjointness": {
            "status": "verified",
            "identity": "history_memory.dataset.event_store_session_id(source, session_id)",
            "variants": list(VARIANTS),
            "training_manifest_sha256": dict(training.manifest_sha256),
            "training_records_sha256": dict(training.records_sha256),
            "training_session_count": {
                variant: len(training.sessions_by_variant[variant])
                for variant in VARIANTS
            },
            "training_session_union_count": len(training.all_sessions),
            "selected_session_ids": list(selected_session_ids),
            "selected_session_count": len(selected_session_ids),
        },
    }


def verify_existing_output(
    output_dir: str | Path, training_dir: str | Path
) -> dict[str, Any]:
    """Re-read a completed package and independently re-check its contracts."""

    output = Path(output_dir).resolve()
    training = load_training_inventory(training_dir)
    records: dict[str, list[dict[str, Any]]] = {}
    manifest_hashes: dict[str, str] = {}
    records_hashes: dict[str, str] = {}
    domain_sessions: dict[str, set[str]] = {
        "history": set(),
        "tool": set(),
    }
    for variant in VARIANTS:
        manifest_path = output / variant / "manifest.json"
        manifest = _load_json(manifest_path)
        expected_top = {
            "schema": SCHEMA,
            "training_profile": TRAINING_PROFILE,
            "purpose": PURPOSE,
            "split": PURPOSE,
            "variant": variant,
            "evaluation_loss_profile": UNIFORM_EVAL_PROFILE,
            "ratios": [8, 12],
        }
        for field, expected in expected_top.items():
            if manifest.get(field) != expected:
                raise ValueError(
                    f"Selection manifest {variant} has {field}={manifest.get(field)!r}; "
                    f"expected {expected!r}"
                )
        preparation = manifest.get("preparation")
        if not isinstance(preparation, Mapping) or not isinstance(
            preparation.get("config"), Mapping
        ):
            raise ValueError(f"Selection manifest omits complete config: {variant}")
        config_type = (
            HistoryPreparationConfig if variant in HISTORY_VARIANTS else ToolPreparationConfig
        )
        config_overrides: dict[str, Any] = {}
        if config_type is HistoryPreparationConfig:
            config_overrides = {
                "variants": HISTORY_PACKED_VARIANTS,
                "h0_decision_ids": None,
                "h1_decision_ids": None,
            }
        _exact_dataclass_config(preparation["config"], config_type, **config_overrides)
        provenance = manifest.get("provenance", {}).get(
            "training_session_disjointness", {}
        )
        if provenance.get("status") != "verified":
            raise ValueError(f"Selection manifest lacks verified disjointness: {variant}")
        if provenance.get("variants") != list(VARIANTS):
            raise ValueError(f"Selection manifest disjointness variants differ: {variant}")
        if provenance.get("training_manifest_sha256") != dict(
            training.manifest_sha256
        ):
            raise ValueError(f"Training manifest SHA map differs: {variant}")
        if provenance.get("training_records_sha256") != dict(
            training.records_sha256
        ):
            raise ValueError(f"Training records SHA map differs: {variant}")

        record_info = manifest.get("records")
        if not isinstance(record_info, Mapping):
            raise ValueError(f"Selection manifest omits records: {variant}")
        record_path = (manifest_path.parent / str(record_info.get("path"))).resolve()
        if record_path.parent != manifest_path.parent.resolve():
            raise ValueError(f"Selection records escape their variant directory: {variant}")
        if record_path.stat().st_size != record_info.get("bytes"):
            raise ValueError(f"Selection records byte count differs: {variant}")
        actual_records_hash = sha256_file(record_path)
        if actual_records_hash != record_info.get("sha256"):
            raise ValueError(f"Selection records SHA differs: {variant}")
        with record_path.open("r", encoding="utf-8") as handle:
            variant_records = [json.loads(line) for line in handle if line.strip()]
        if len(variant_records) != record_info.get("count"):
            raise ValueError(f"Selection records count differs: {variant}")
        for record in variant_records:
            _validate_dev_record(record)
            metadata = record["metadata"]
            gold_action = metadata.get("gold_action")
            if not isinstance(gold_action, Mapping):
                raise ValueError(f"Missing native gold action: {variant}")
            if metadata.get("decision_type") == "tool_call":
                calls = metadata.get("gold_tool_calls")
                if not isinstance(calls, list) or not calls:
                    raise ValueError(f"Missing native gold tool calls: {variant}")
                if gold_action.get("tool_calls") != calls:
                    raise ValueError(f"Gold tool call copies differ: {variant}")
            elif metadata.get("gold_tool_calls"):
                raise ValueError(f"Non-tool target has gold tool calls: {variant}")
            domain = "history" if variant in HISTORY_VARIANTS else "tool"
            domain_sessions[domain].add(record["session_key"])
        selected_in_manifest = provenance.get("selected_session_ids")
        if set(selected_in_manifest or ()) != {
            record["session_key"] for record in variant_records
        }:
            raise ValueError(f"Selected session provenance differs: {variant}")
        records[variant] = variant_records
        manifest_hashes[variant] = sha256_file(manifest_path)
        records_hashes[variant] = actual_records_hash

    validate_selection_records(
        records, variants=HISTORY_VARIANTS, expected_decisions=32
    )
    validate_selection_records(records, variants=TOOL_VARIANTS, expected_decisions=32)
    assert_session_disjoint(
        domain_sessions["history"] | domain_sessions["tool"],
        training.sessions_by_variant,
    )
    return {
        "schema": "next-compression-checkpoint-selection-verification-v1",
        "purpose": PURPOSE,
        "output_dir": str(output),
        "verified": True,
        "records_per_variant": {
            variant: len(records[variant]) for variant in VARIANTS
        },
        "decisions_per_domain": {"history": 32, "tool": 32},
        "selected_sessions_per_domain": {
            domain: len(values) for domain, values in domain_sessions.items()
        },
        "manifest_sha256": manifest_hashes,
        "records_sha256": records_hashes,
        "training_manifest_sha256": dict(training.manifest_sha256),
    }


def package_delivery(
    output_dir: str | Path,
    training_dir: str | Path,
    delivery_dir: str | Path,
) -> dict[str, Any]:
    """Verify and package the fixed small dev corpus with a hash receipt."""

    verification = verify_existing_output(output_dir, training_dir)
    output = Path(output_dir).resolve()
    delivery = Path(delivery_dir).resolve()
    if delivery.exists() and (not delivery.is_dir() or any(delivery.iterdir())):
        raise FileExistsError(f"Selection delivery directory is not empty: {delivery}")
    delivery.mkdir(parents=True, exist_ok=True)
    archive = delivery / "next-compression-selection-dev-v1.tar.gz"
    pending_archive = archive.with_suffix(archive.suffix + f".pending-{os.getpid()}")
    try:
        with tarfile.open(pending_archive, "w:gz") as bundle:
            bundle.add(output, arcname=output.name, recursive=True)
        pending_archive.replace(archive)
        receipt = {
            "schema": "next-compression-selection-delivery-receipt-v1",
            "purpose": PURPOSE,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "boundary": "Toucan-only session-disjoint proxy; not an independent official holdout",
            "source_directory": str(output),
            "archive": {
                "path": archive.name,
                "sha256": sha256_file(archive),
                "bytes": archive.stat().st_size,
                "archive_root": output.name,
            },
            "validation": verification,
            "model_run_performed": False,
        }
        receipt_path = delivery / "manifest-receipt.json"
        pending_receipt = receipt_path.with_suffix(
            receipt_path.suffix + f".pending-{os.getpid()}"
        )
        pending_receipt.write_text(
            json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        pending_receipt.replace(receipt_path)
    except Exception:
        if pending_archive.exists():
            pending_archive.unlink()
        raise
    return {
        **receipt,
        "receipt": {
            "path": str(receipt_path),
            "sha256": sha256_file(receipt_path),
            "bytes": receipt_path.stat().st_size,
        },
    }


def _positive_quota(value: str) -> dict[str, int]:
    result: dict[str, int] = {}
    for item in value.split(","):
        kind, separator, count = item.partition("=")
        if not separator or kind not in {"tool_call", "non_tool_response", "terminal_stop"}:
            raise argparse.ArgumentTypeError(f"Invalid quota item: {item!r}")
        try:
            parsed = int(count)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"Invalid quota count: {item!r}") from exc
        if parsed <= 0:
            raise argparse.ArgumentTypeError("Every decision type quota must be positive")
        if kind in result:
            raise argparse.ArgumentTypeError(f"Repeated quota kind: {kind}")
        result[kind] = parsed
    return result


def arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--normalized-manifest", required=True)
    parser.add_argument("--tool-manifest", required=True)
    parser.add_argument("--training-dir", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--history-sources",
        nargs="+",
        help="Explicit normalized source keys in deterministic read order",
    )
    parser.add_argument(
        "--tool-sources",
        nargs="+",
        help="Explicit tool source keys in deterministic read order",
    )
    parser.add_argument(
        "--history-quotas",
        type=_positive_quota,
        default=_positive_quota(
            "tool_call=12,non_tool_response=10,terminal_stop=10"
        ),
    )
    parser.add_argument(
        "--tool-quotas",
        type=_positive_quota,
        default=_positive_quota(
            "tool_call=16,non_tool_response=8,terminal_stop=8"
        ),
    )
    parser.add_argument("--max-history-sessions-scanned", type=int, default=6_000)
    parser.add_argument("--max-tool-sessions-scanned", type=int, default=6_000)
    parser.add_argument("--history-workers", type=int, default=4)
    parser.add_argument(
        "--verify-existing",
        action="store_true",
        help="Re-read and verify an already completed output package",
    )
    parser.add_argument(
        "--delivery-dir",
        help="Verify and package an existing output into this empty directory",
    )
    result = parser.parse_args(argv)
    for name in (
        "max_history_sessions_scanned",
        "max_tool_sessions_scanned",
        "history_workers",
    ):
        if getattr(result, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    for name in ("history_sources", "tool_sources"):
        values = getattr(result, name)
        if values is not None and len(values) != len(set(values)):
            parser.error(f"--{name.replace('_', '-')} must not contain duplicates")
    if sum(result.history_quotas.values()) != 32:
        parser.error("--history-quotas must total 32 decisions")
    if sum(result.tool_quotas.values()) != 32:
        parser.error("--tool-quotas must total 32 decisions")
    return result


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output_dir).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Preparation output is not empty: {output}")
    pending = output.with_name(output.name + f".pending-{os.getpid()}")
    if pending.exists():
        raise FileExistsError(f"Stale preparation directory exists: {pending}")

    normalized_path = Path(args.normalized_manifest).resolve()
    tool_path = Path(args.tool_manifest).resolve()
    normalized = _load_normalized_manifest(normalized_path)
    tool_manifest = _load_normalized_manifest(tool_path)
    training = load_training_inventory(args.training_dir)
    history_specs = _source_specs(
        normalized_path, normalized, args.history_sources
    )
    tool_specs = _source_specs(tool_path, tool_manifest, args.tool_sources)

    history_raw_config = _training_preparation_config(
        training, ("H1", "H2", "H3")
    )
    tool_raw_config = _training_preparation_config(training, ("T0", "T1"))
    history_config = _exact_dataclass_config(
        history_raw_config,
        HistoryPreparationConfig,
        variants=HISTORY_PACKED_VARIANTS,
        workers=args.history_workers,
        h0_decision_ids=None,
        h1_decision_ids=None,
    )
    tool_config = _exact_dataclass_config(tool_raw_config, ToolPreparationConfig)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    tokenizer_info = tokenizer_identity(tokenizer)
    for variant in VARIANTS:
        expected = training.manifests[variant].get("tokenizer", {}).get("sha256")
        if expected != tokenizer_info["sha256"]:
            raise ValueError(f"Tokenizer differs from {variant} training corpus")

    history_records, history_selection = select_history(
        history_specs,
        tokenizer,
        history_config,
        training,
        args.history_quotas,
        args.max_history_sessions_scanned,
    )
    tool_records, tool_selection = select_tools(
        tool_specs,
        tokenizer,
        tool_config,
        training,
        args.tool_quotas,
        args.max_tool_sessions_scanned,
    )
    validate_selection_records(
        history_records,
        variants=HISTORY_VARIANTS,
        expected_decisions=sum(args.history_quotas.values()),
    )
    validate_selection_records(
        tool_records,
        variants=TOOL_VARIANTS,
        expected_decisions=sum(args.tool_quotas.values()),
    )

    all_selected_sessions = (
        history_selection["selected_session_ids"]
        + tool_selection["selected_session_ids"]
    )
    assert_session_disjoint(all_selected_sessions, training.sessions_by_variant)

    pending.mkdir(parents=True)
    manifests: dict[str, Any] = {}
    try:
        for variant in VARIANTS:
            is_history = variant in HISTORY_VARIANTS
            selected_records = history_records if is_history else tool_records
            selection = history_selection if is_history else tool_selection
            specs = history_specs if is_history else tool_specs
            manifest_path = normalized_path if is_history else tool_path
            raw_config = history_raw_config if is_history else tool_raw_config
            render_profile = (
                training.manifests[variant]["render_profile"]
                if variant != "H0"
                else training.manifests["H1"]["render_profile"]
            )
            preparation = {
                "purpose": PURPOSE,
                "render_profile": render_profile,
                "evaluation_loss_profile": UNIFORM_EVAL_PROFILE,
                "config": copy.deepcopy(raw_config),
                "packer": (
                    "next_compression.history.iter_history_records"
                    if is_history
                    else "next_compression.tools.iter_tool_records"
                ),
                "source_manifest_sha256": sha256_file(manifest_path),
                "selection_profile": "bounded-session-round-robin-type-quota-v1",
                "source_schedule": "deterministic-session-round-robin-v1",
                "selected_sources_in_order": [spec.key for spec in specs],
                "selection_quotas": selection["quotas"],
                "max_sessions_scanned": (
                    args.max_history_sessions_scanned
                    if is_history
                    else args.max_tool_sessions_scanned
                ),
                "source_rows_passed_to_packer_split": "train",
                "serialized_record_split": PURPOSE,
                "target_truncation": False,
                "memory_view_target_independent": True,
                "static_pair": "H0-H1-identical-input"
                if variant in {"H0", "H1"}
                else None,
                "a_pair": "H2-H3-identical-input"
                if variant in {"H2", "H3"}
                else None,
                "h3_training_weights_are_not_evaluation_criterion": variant == "H3",
            }
            writer = DevCorpusWriter(pending / variant, variant)
            for record in selected_records[variant]:
                writer.write(record)
            provenance = _provenance(
                training,
                normalized_manifest_path=manifest_path,
                selected_session_ids=selection["selected_session_ids"],
            )
            manifests[variant] = writer.finish(
                tokenizer=tokenizer_info,
                training_manifest=training.manifests[variant],
                preparation=preparation,
                source_files=_source_files_for_manifest(
                    specs, selection["audit"]
                ),
                provenance=provenance,
                audit={
                    **selection["audit"],
                    "selection.type_counts": selection["type_counts"],
                    "selection.session_disjoint_all_six_training_corpora": True,
                },
            )
        if output.exists():
            output.rmdir()
        pending.rename(output)
    except Exception:
        if pending.exists():
            shutil.rmtree(pending)
        raise
    return {
        "schema": "next-compression-checkpoint-selection-result-v1",
        "purpose": PURPOSE,
        "output_dir": str(output),
        "history_decisions": sum(args.history_quotas.values()),
        "tool_decisions": sum(args.tool_quotas.values()),
        "records_per_variant": {
            variant: manifests[variant]["records"]["count"] for variant in VARIANTS
        },
        "training_manifest_sha256": dict(training.manifest_sha256),
        "verified_session_disjoint": True,
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = arguments(argv)
    if args.verify_existing and args.delivery_dir:
        raise ValueError("Choose either verification or delivery packaging")
    if args.delivery_dir:
        result = package_delivery(
            args.output_dir, args.training_dir, args.delivery_dir
        )
    elif args.verify_existing:
        result = verify_existing_output(args.output_dir, args.training_dir)
    else:
        result = prepare(args)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
