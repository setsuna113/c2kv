"""Normalize native C1 task artifacts into the common paper telemetry schema.

The converter is deliberately offline.  It does not infer unavailable GPU
measurements, replay benchmark tools, or fold external-tool time into model
latency.  One stable ``outer_request_id`` joins a benchmark decision to all of
its native extraction and generation requests.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

from .telemetry import SCHEMA, canonical_json, read_jsonl


C1_CONVERSION_SCHEMA = "c2kv.measurement.c1-conversion.v1"


def _finite_number(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        numeric = float(value)
        if math.isfinite(numeric):
            return numeric
    return None


def _duration_ns(value: Any, seconds: Any = None) -> Optional[int]:
    numeric = _finite_number(value)
    if numeric is not None and numeric >= 0:
        return int(numeric)
    numeric = _finite_number(seconds)
    if numeric is not None and numeric >= 0:
        return int(round(numeric * 1e9))
    return None


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _metrics(measurement: Any) -> dict[str, Any]:
    value = _mapping(measurement)
    nested = value.get("metrics")
    if isinstance(nested, Mapping):
        return dict(nested)
    return value


def _measurement_with_metrics(
    measurement: Any, additions: Mapping[str, Any]
) -> dict[str, Any]:
    result = copy.deepcopy(_mapping(measurement))
    if isinstance(result.get("metrics"), Mapping):
        metrics = dict(result["metrics"])
        for key, value in additions.items():
            metrics.setdefault(key, value)
        result["metrics"] = metrics
    else:
        for key, value in additions.items():
            result.setdefault(key, value)
    return result


def _exact_token_count(bytes_value: Any, bytes_per_token: Any) -> Optional[int]:
    byte_count = _finite_number(bytes_value)
    unit = _finite_number(bytes_per_token)
    if byte_count is None or unit is None or byte_count < 0 or unit <= 0:
        return None
    quotient = byte_count / unit
    rounded = round(quotient)
    return int(rounded) if math.isclose(quotient, rounded, abs_tol=1e-9) else None


def _same_prefix_metrics(trace: Mapping[str, Any]) -> dict[str, Any]:
    """Read the controller's actual Full-render denominator for this prefix."""

    controller = _mapping(trace.get("controller"))
    reference = _mapping(controller.get("same_prefix_full_reference"))
    ratio = _mapping(controller.get("compression_ratio"))
    unit = controller.get("kv_bytes_per_token")
    full_history = reference.get("full_history_tokens")
    if not isinstance(full_history, int) or isinstance(full_history, bool):
        full_history = _exact_token_count(ratio.get("full_history_bytes"), unit)
    common = reference.get("common_live_tokens")
    if not isinstance(common, int) or isinstance(common, bool):
        common = _exact_token_count(ratio.get("common_live_bytes"), unit)
    active_history = _exact_token_count(ratio.get("active_history_bytes"), unit)
    result: dict[str, Any] = {}
    if full_history is not None:
        result["history_full_kv_tokens"] = full_history
    if active_history is not None:
        result["history_active_kv_tokens"] = active_history
    if common is not None and full_history is not None:
        result["whole_full_kv_tokens"] = common + full_history
    if common is not None and active_history is not None:
        result["whole_active_kv_tokens"] = common + active_history
    if result:
        result["canonical_full_source"] = (
            "controller.same_prefix_full_reference"
        )
    return result


def _task_directories(native_root: Path) -> list[Path]:
    root = native_root.resolve()
    if (root / "server" / "steps.jsonl").is_file():
        return [root]
    for candidate in (root / "task_shards", root / "native" / "task_shards"):
        if candidate.is_dir():
            tasks = sorted(
                path for path in candidate.iterdir()
                if path.is_dir() and (path / "server" / "steps.jsonl").is_file()
            )
            if tasks:
                return tasks
    raise FileNotFoundError(
        f"no native C1 task_shards/*/server/steps.jsonl under {root}"
    )


def _read_steps(task_dir: Path) -> list[dict[str, Any]]:
    rows = list(read_jsonl(task_dir / "server" / "steps.jsonl"))
    if not rows:
        raise ValueError(f"native C1 step journal is empty: {task_dir}")
    return rows


def _harness_sources(task_dir: Path, cell_dir: Path) -> list[Path]:
    result = []
    normalized_output = (
        cell_dir / "measurement" / "harness_events.jsonl"
    ).resolve()
    for name in ("harness_events.jsonl", "harness_telemetry.jsonl"):
        for path in sorted(task_dir.rglob(name)):
            if path.resolve() == normalized_output:
                continue
            if path.is_file():
                result.append(path)
    return list(dict.fromkeys(result))


def _read_harness(tasks: Iterable[Path], cell_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen = set()
    for task_dir in tasks:
        for path in _harness_sources(task_dir, cell_dir):
            for row in read_jsonl(path):
                identity = canonical_json(row)
                if identity not in seen:
                    seen.add(identity)
                    rows.append(row)
    return rows


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    materialized = [dict(row) for row in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in materialized:
            handle.write(json.dumps(
                row, ensure_ascii=False, sort_keys=True, allow_nan=False
            ) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    return len(materialized)


def _component_phases(
    record: Mapping[str, Any], outer_request_id: str
) -> list[dict[str, Any]]:
    rows = []
    timing = _mapping(record.get("controller_timing"))
    for name in ("prepare", "reconsider"):
        duration = _duration_ns(
            timing.get(f"{name}_duration_ns"), timing.get(f"{name}_seconds")
        )
        if duration is not None:
            rows.append({
                "schema": SCHEMA,
                "event_type": "phase",
                "request_id": outer_request_id,
                "outer_request_id": outer_request_id,
                "phase": f"c1_controller_{name}",
                "duration_ns": duration,
                "accounting": "inclusive_controller_phase",
            })
    for trace in record.get("generation_trace") or []:
        if not isinstance(trace, Mapping):
            continue
        duration = _duration_ns(
            trace.get("duration_ns"),
            _mapping(_mapping(trace.get("generation")).get("stats")).get(
                "elapsed_sec"
            ),
        )
        if duration is not None:
            rows.append({
                "schema": SCHEMA,
                "event_type": "phase",
                "request_id": outer_request_id,
                "outer_request_id": outer_request_id,
                "phase": (
                    "c1_native_regeneration"
                    if trace.get("phase") == "regeneration"
                    else "c1_native_draft"
                ),
                "duration_ns": duration,
                "attempt_uid": trace.get("attempt_uid"),
                "accounting": "exclusive_native_generation_request",
            })
    checks = [
        check for check in record.get("recovery_checks") or []
        if isinstance(check, Mapping)
    ]
    decisions = checks or [_mapping(record.get("exact_recovery"))]
    for check_index, decision in enumerate(decisions):
        for item in decision.get("measurement_phases") or []:
            if not isinstance(item, Mapping):
                continue
            duration = _duration_ns(item.get("duration_ns"))
            if duration is not None and isinstance(item.get("phase"), str):
                rows.append({
                    "schema": SCHEMA,
                    "event_type": "phase",
                    "request_id": outer_request_id,
                    "outer_request_id": outer_request_id,
                    "phase": item["phase"],
                    "duration_ns": duration,
                    "recovery_check_index": check_index,
                    "parent_phase": "c1_controller_reconsider",
                    "accounting": "nested_nonadditive_component",
                })
        for receipt in decision.get("selection_model_calls") or []:
            if not isinstance(receipt, Mapping):
                continue
            duration = _duration_ns(None, receipt.get("latency_seconds"))
            capability = receipt.get("capability")
            if duration is not None and isinstance(capability, str):
                rows.append({
                    "schema": SCHEMA,
                    "event_type": "phase",
                    "request_id": outer_request_id,
                    "outer_request_id": outer_request_id,
                    "phase": f"c1_{capability}",
                    "duration_ns": duration,
                    "purpose": receipt.get("purpose"),
                    "recovery_check_index": check_index,
                    "parent_phase": "c1_controller_reconsider",
                    "accounting": "nested_nonadditive_model_call",
                })
    return rows


def _native_rows_for_trace(
    trace: Mapping[str, Any], outer_request_id: str
) -> list[dict[str, Any]]:
    stats = _mapping(_mapping(trace.get("generation")).get("stats"))
    telemetry = _mapping(stats.get("native_telemetry"))
    extraction_rows = [
        item for item in telemetry.get("extractions") or []
        if isinstance(item, Mapping)
    ]
    rows: list[dict[str, Any]] = []

    def normalize(item: Mapping[str, Any], default_phase: str) -> dict[str, Any]:
        server_request_id = (
            item.get("server_request_id")
            or item.get("native_request_id")
            or item.get("request_id")
        )
        measurement = item.get("paper_measurement")
        additions = {}
        for field in (
            "extraction_duration_ns",
            "gist_generation_duration_ns",
        ):
            explicit = item.get(field)
            if _finite_number(explicit) is not None:
                additions[field] = int(explicit)
        if isinstance(item.get("cache_hit"), bool):
            additions["cache_hit"] = item["cache_hit"]
        return {
            "schema": SCHEMA,
            "event_type": "server_request",
            "outer_request_id": outer_request_id,
            "request_id": server_request_id,
            "server_request_id": server_request_id,
            "phase": item.get("phase") or default_phase,
            "paper_measurement": _measurement_with_metrics(
                measurement, additions
            ),
            "native_event": copy.deepcopy(dict(item)),
        }

    for item in extraction_rows:
        rows.append(normalize(item, "c1_gist_extraction"))

    generation = telemetry.get("generation")
    if isinstance(generation, Mapping):
        generation_row = normalize(
            generation,
            "recovery_generation"
            if trace.get("phase") == "regeneration"
            else "generation",
        )
        generation_row["native_phase"] = generation_row.get("phase")
        generation_row["phase"] = (
            "recovery_generation"
            if trace.get("phase") == "regeneration"
            else "generation"
        )
    else:
        transport = _mapping(stats.get("sglang_transport"))
        server_request_id = (
            stats.get("native_request_id")
            or transport.get("native_request_id")
            or transport.get("rid")
            or trace.get("attempt_uid")
        )
        generation_row = {
            "schema": SCHEMA,
            "event_type": "server_request",
            "outer_request_id": outer_request_id,
            "request_id": server_request_id,
            "server_request_id": server_request_id,
            "phase": (
                "recovery_generation"
                if trace.get("phase") == "regeneration"
                else "generation"
            ),
            "paper_measurement": copy.deepcopy(
                _mapping(stats.get("paper_measurement"))
            ),
            "native_event": {
                "native_request_id": server_request_id,
                "request_ids": copy.deepcopy(stats.get("native_request_ids")),
            },
        }

    same_prefix = _same_prefix_metrics(trace)
    if same_prefix:
        # Apply these after response/raw-engine coalescing.  The native engine
        # reports packed-input denominators; this controller receipt records
        # the same-prefix original Full render used by the paper comparison.
        generation_row["controller_same_prefix_metrics"] = same_prefix
    if extraction_rows:
        # A generation-level inclusive gist value would overlap the exact
        # extraction rows. Preserve it as provenance but remove it from the
        # additive field consumed by the shared aggregate.
        measurement = generation_row["paper_measurement"]
        metrics = _metrics(measurement)
        inclusive = metrics.get("gist_generation_duration_ns")
        if _finite_number(inclusive) is not None:
            native = _mapping(generation_row.get("native_event"))
            native["inclusive_gist_generation_duration_ns"] = inclusive
            generation_row["native_event"] = native
            if isinstance(measurement.get("metrics"), Mapping):
                normalized = dict(measurement["metrics"])
                normalized.pop("gist_generation_duration_ns", None)
                measurement["metrics"] = normalized
            else:
                measurement.pop("gist_generation_duration_ns", None)
    rows.append(generation_row)
    return rows


def _deduplicate_server_rows(
    rows: Iterable[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: dict[tuple[str, str], dict[str, Any]] = {}

    def merge_measurement(
        target: Mapping[str, Any], update: Mapping[str, Any], key: tuple[str, str]
    ) -> dict[str, Any]:
        merged = copy.deepcopy(dict(target))
        for field, value in update.items():
            if field == "metrics" and isinstance(value, Mapping):
                metrics = _mapping(merged.get("metrics"))
                for metric, metric_value in value.items():
                    if metric in metrics and metrics[metric] != metric_value:
                        raise ValueError(
                            "conflicting native server measurements reuse "
                            f"outer_request_id/server_request_id={key!r}: {metric}"
                        )
                    metrics.setdefault(metric, copy.deepcopy(metric_value))
                merged["metrics"] = metrics
            elif field in merged and merged[field] != value:
                raise ValueError(
                    "conflicting native server measurements reuse "
                    f"outer_request_id/server_request_id={key!r}: {field}"
                )
            else:
                merged.setdefault(field, copy.deepcopy(value))
        return merged

    for raw in rows:
        row = dict(raw)
        outer = row.get("outer_request_id")
        server = row.get("server_request_id") or row.get("request_id")
        if not isinstance(outer, str) or not outer:
            raise ValueError("C1 server event lacks outer_request_id")
        if not isinstance(server, str) or not server:
            raise ValueError("C1 server event lacks server_request_id")
        key = (outer, server)
        previous = seen.get(key)
        if previous is None:
            seen[key] = row
            result.append(row)
            continue
        if (previous.get("phase") and row.get("phase")
                and previous["phase"] != row["phase"]):
            raise ValueError(
                "conflicting native server events reuse "
                f"outer_request_id/server_request_id={key!r}"
            )
        previous["paper_measurement"] = merge_measurement(
            _mapping(previous.get("paper_measurement")),
            _mapping(row.get("paper_measurement")),
            key,
        )
        for provenance in ("native_event", "native_engine_event"):
            if provenance not in row:
                continue
            if provenance in previous and canonical_json(previous[provenance]) != canonical_json(row[provenance]):
                raise ValueError(
                    "conflicting native server events reuse "
                    f"outer_request_id/server_request_id={key!r}"
                )
            previous.setdefault(provenance, copy.deepcopy(row[provenance]))
    return result


def _native_engine_rows(native_root: Path, cell_dir: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """Read the engine ledger without modifying or replacing the raw file."""

    candidates = [
        cell_dir / "native_engine_telemetry.jsonl",
        native_root / "native_engine_telemetry.jsonl",
    ]
    rows: list[dict[str, Any]] = []
    sources = []
    for path in dict.fromkeys(candidate.resolve() for candidate in candidates):
        if not path.is_file():
            continue
        sources.append(str(path))
        for event in read_jsonl(path):
            outer = event.get("outer_request_id")
            server = event.get("server_request_id")
            phase = event.get("phase")
            if event.get("kind") == "generation":
                phase = (
                    "recovery_generation"
                    if isinstance(phase, str) and "regeneration" in phase
                    else "generation"
                )
            rows.append({
                "schema": SCHEMA,
                "event_type": "server_request",
                "outer_request_id": outer,
                "request_id": server,
                "server_request_id": server,
                "phase": phase,
                "native_phase": event.get("phase"),
                "paper_measurement": copy.deepcopy(event),
                "native_engine_event": copy.deepcopy(event),
            })
    return rows, sources


def _apply_controller_denominators(
    rows: Iterable[dict[str, Any]],
) -> None:
    """Replace only denominator fields with the controller's Full receipt.

    Physical engine measurements, including generation-active and resident KV,
    remain untouched.  The prior engine values stay on the normalized row for
    audit and offline reinterpretation.
    """

    for row in rows:
        reference = _mapping(row.get("controller_same_prefix_metrics"))
        if not reference or row.get("phase") not in {
            "generation", "recovery_generation",
        }:
            continue
        measurement = _mapping(row.get("paper_measurement"))
        nested = isinstance(measurement.get("metrics"), Mapping)
        metrics = _metrics(measurement)
        applied = {
            field: reference[field]
            for field in (
                "history_full_kv_tokens",
                "whole_full_kv_tokens",
                "history_active_kv_tokens",
            )
            if field in reference
        }
        if not applied:
            continue
        applied["canonical_full_source"] = reference[
            "canonical_full_source"
        ]
        prior = {
            field: copy.deepcopy(metrics[field])
            for field in applied
            if field in metrics
        }
        metrics.update(copy.deepcopy(applied))
        if nested:
            measurement["metrics"] = metrics
        else:
            measurement.update(metrics)
        row["paper_measurement"] = measurement
        row["denominator_provenance"] = {
            "source": "controller.same_prefix_full_reference",
            "controller_recorded": copy.deepcopy(applied),
            "native_engine_reported": prior,
            "history_active_basis": (
                "controller.compression_ratio.active_history_bytes / "
                "controller.kv_bytes_per_token"
            ),
            "physical_engine_metrics_overridden": False,
        }


def _decision_metrics(
    record: Mapping[str, Any], server_rows: Iterable[Mapping[str, Any]]
) -> dict[str, Any]:
    traces = [
        trace for trace in record.get("generation_trace") or []
        if isinstance(trace, Mapping)
    ]
    decision = _mapping(record.get("exact_recovery"))
    checks = [
        check for check in record.get("recovery_checks") or []
        if isinstance(check, Mapping)
    ] or [decision]
    selections = [_mapping(check.get("selection")) for check in checks]
    gist_duration = 0
    gist_measured = 0
    for row in _deduplicate_server_rows(server_rows):
        value = _metrics(row.get("paper_measurement")).get(
            "gist_generation_duration_ns"
        )
        numeric = _finite_number(value)
        if numeric is not None:
            gist_duration += int(numeric)
            gist_measured += 1
    return {
        "draft_generation_requests": sum(
            trace.get("phase") != "regeneration" for trace in traces
        ),
        "regeneration_requests": sum(
            trace.get("phase") == "regeneration" for trace in traces
        ),
        "generation_requests": len(traces),
        "detector_calls": sum(
            selection.get("selector") in {"risk", "legacy_prefill"}
            for selection in selections
        ),
        "detector_score_available": any(
            selection.get("available") is True
            and _finite_number(selection.get("score")) is not None
            for selection in selections
        ),
        "recovery_committed": decision.get("status") == "recover",
        "gist_generation_duration_ns": (
            gist_duration if gist_measured else None
        ),
        "gist_generation_measured_requests": gist_measured,
    }


def _response_paper_measurement(response: Any) -> Optional[dict[str, Any]]:
    if not isinstance(response, Mapping):
        return None
    metadata = _mapping(response.get("metadata"))
    proxy = _mapping(response.get("c2kv_proxy"))
    for candidate in (
        response.get("paper_measurement"),
        metadata.get("paper_measurement"),
        proxy.get("server_measurement"),
    ):
        if isinstance(candidate, Mapping):
            return copy.deepcopy(dict(candidate))
    return None


def _enrich_prefix_replay(
    path: Path, server_rows: Iterable[Mapping[str, Any]]
) -> dict[str, int]:
    if not path.is_file():
        return {
            "rows": 0, "source_added": 0,
            "target_added": 0, "target_refreshed": 0,
        }
    target_by_outer: dict[str, dict[str, Any]] = {}
    for row in server_rows:
        if row.get("phase") not in {"generation", "recovery_generation"}:
            continue
        outer = row.get("outer_request_id")
        measurement = row.get("paper_measurement")
        if isinstance(outer, str) and isinstance(measurement, Mapping):
            # Rows are in execution order, so a recovery generation replaces
            # its discarded draft as the recorded target prefix.
            target_by_outer[outer] = copy.deepcopy(dict(measurement))

    rows = list(read_jsonl(path))
    source_added = 0
    target_added = 0
    target_refreshed = 0
    for row in rows:
        if row.get("event_type") != "prefix_replay":
            continue
        if not isinstance(row.get("source_paper_measurement"), Mapping):
            source = _response_paper_measurement(row.get("source_response"))
            if source is not None:
                row["source_paper_measurement"] = source
                source_added += 1
        target = target_by_outer.get(str(row.get("request_id")))
        if target is not None:
            if not isinstance(row.get("target_paper_measurement"), Mapping):
                target_added += 1
            elif canonical_json(row["target_paper_measurement"]) != canonical_json(target):
                target_refreshed += 1
            # target_paper_measurement is derived from this normalized target
            # ledger. Refresh it on every offline conversion so denominator
            # corrections propagate without changing the recorded request.
            row["target_paper_measurement"] = target
    _write_jsonl(path, rows)
    return {
        "rows": sum(row.get("event_type") == "prefix_replay" for row in rows),
        "source_added": source_added,
        "target_added": target_added,
        "target_refreshed": target_refreshed,
    }


def convert_run(
    native_root: "str | os.PathLike[str]",
    cell_dir: "str | os.PathLike[str]",
    *,
    benchmark: str,
    arm: str,
    replay: bool = False,
) -> dict[str, Any]:
    """Convert one native C1 run into common proxy/harness/server JSONL.

    ``native_root`` may be a run root containing ``task_shards`` or one task
    directory. Derived files are atomically replaced under ``cell_dir``; raw
    native journals are never modified.
    """

    if not isinstance(benchmark, str) or not benchmark:
        raise ValueError("benchmark must be a nonempty string")
    if not isinstance(arm, str) or not arm:
        raise ValueError("arm must be a nonempty string")
    native_path = Path(native_root).resolve()
    cell_path = Path(cell_dir).resolve()
    tasks = _task_directories(native_path)
    harness_rows = _read_harness(tasks, cell_path)
    harness_decisions = {
        str(row.get("decision_request_id")): row
        for row in harness_rows
        if row.get("event_type") == "decision"
        and row.get("decision_request_id")
    }

    requests: list[dict[str, Any]] = []
    phases: list[dict[str, Any]] = []
    raw_server_rows: list[dict[str, Any]] = []
    prefix_index: list[dict[str, Any]] = []
    outer_ids = set()
    for task_dir in tasks:
        for record in _read_steps(task_dir):
            outer = record.get("outer_request_id")
            if not isinstance(outer, str) or not outer:
                raise ValueError(
                    f"native C1 step lacks outer_request_id: {task_dir}"
                )
            if outer in outer_ids:
                raise ValueError(f"duplicate native C1 outer_request_id: {outer}")
            outer_ids.add(outer)
            decision_server_rows = []
            for trace in record.get("generation_trace") or []:
                if isinstance(trace, Mapping):
                    decision_server_rows.extend(
                        _native_rows_for_trace(trace, outer)
                    )
            raw_server_rows.extend(decision_server_rows)
            harness = harness_decisions.get(outer, {})
            duration = _duration_ns(harness.get("duration_ns"))
            latency_source = "harness_client_request"
            if duration is None:
                duration = _duration_ns(
                    record.get("decision_duration_ns"),
                    record.get("decision_runtime_seconds"),
                )
                latency_source = "controller_decision"
            request = {
                "schema": SCHEMA,
                "event_type": "request",
                "benchmark": benchmark,
                "arm": arm,
                "request_id": outer,
                "outer_request_id": outer,
                "episode_id": record.get("session_id"),
                "decision_key": record.get("decision_key"),
                "start_unix_ns": (
                    harness.get("start_unix_ns")
                    if harness.get("start_unix_ns") is not None
                    else record.get("decision_start_unix_ns")
                ),
                "end_unix_ns": (
                    harness.get("end_unix_ns")
                    if harness.get("end_unix_ns") is not None
                    else record.get("decision_end_unix_ns")
                ),
                "duration_ns": duration,
                "status": record.get("status"),
                "error": record.get("error"),
                "latency_source": latency_source,
                "c1_metrics": _decision_metrics(record, decision_server_rows),
            }
            requests.append(request)
            phases.extend(_component_phases(record, outer))
            target_metrics = next((
                metrics
                for trace in reversed(record.get("generation_trace") or [])
                if isinstance(trace, Mapping)
                for metrics in [_same_prefix_metrics(trace)]
                if metrics
            ), {})
            prefix_index.append({
                "schema": "c2kv.measurement.c1-prefix-index.v1",
                "outer_request_id": outer,
                "benchmark": benchmark,
                "arm": arm,
                "session_id": record.get("session_id"),
                "decision_key": record.get("decision_key"),
                "target_metrics": target_metrics,
            })

    all_engine_rows, engine_sources = _native_engine_rows(native_path, cell_path)
    engine_rows = [
        row for row in all_engine_rows
        if row.get("outer_request_id") in outer_ids
    ]
    unmatched_engine_rows = len(all_engine_rows) - len(engine_rows)
    server_rows = _deduplicate_server_rows([*raw_server_rows, *engine_rows])
    _apply_controller_denominators(server_rows)
    for row in harness_rows:
        row.setdefault("benchmark", benchmark)
        row.setdefault("arm", arm)

    proxy_path = cell_path / "proxy_telemetry.jsonl"
    server_path = cell_path / "server_telemetry.jsonl"
    harness_path = cell_path / "measurement" / "harness_events.jsonl"
    prefix_path = cell_path / "c1_prefix_index.jsonl"
    counts = {
        "proxy": _write_jsonl(proxy_path, [*requests, *phases]),
        "server": _write_jsonl(server_path, server_rows),
        "harness": _write_jsonl(harness_path, harness_rows),
        "prefix_index": _write_jsonl(prefix_path, prefix_index),
    }
    replay_path = cell_path / "prefix_replay.jsonl"
    replay_enrichment = _enrich_prefix_replay(replay_path, server_rows)
    server_measured = sum(bool(_metrics(row.get("paper_measurement"))) for row in server_rows)
    gist_measured = sum(
        _finite_number(_metrics(row.get("paper_measurement")).get(
            "gist_generation_duration_ns"
        )) is not None
        for row in server_rows
    )
    return {
        "schema": C1_CONVERSION_SCHEMA,
        "native_root": str(native_path),
        "cell_dir": str(cell_path),
        "benchmark": benchmark,
        "arm": arm,
        "replay_requested": bool(replay),
        "tasks": len(tasks),
        "decisions": len(requests),
        "server_requests": len(server_rows),
        "coverage": {
            "harness_decisions": sum(
                request["request_id"] in harness_decisions for request in requests
            ),
            "server_measurements": server_measured,
            "gist_generation_measurements": gist_measured,
        },
        "outputs": {
            "proxy": str(proxy_path),
            "server": str(server_path),
            "harness": str(harness_path),
            "prefix_index": str(prefix_path),
            "prefix_replay": str(replay_path) if replay_path.is_file() else None,
        },
        "rows": counts,
        "native_engine_telemetry_sources": engine_sources,
        "native_engine_rows_ignored_without_step": unmatched_engine_rows,
        "prefix_replay_enrichment": replay_enrichment,
        "replay_payload_available": False,
        "replay_note": (
            "Native C1 journals bind target metrics by outer_request_id but do not "
            "store a portable Full replay payload; the common runner supplies Full prefixes."
        ),
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-root", required=True)
    parser.add_argument("--cell-dir", required=True)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--replay", action="store_true")
    args = parser.parse_args(argv)
    result = convert_run(
        args.native_root,
        args.cell_dir,
        benchmark=args.benchmark,
        arm=args.arm,
        replay=args.replay,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
