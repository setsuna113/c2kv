"""Aggregate raw paper telemetry without filling missing measurements."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .telemetry import read_jsonl


def percentile(values: Iterable[float], probability: float) -> Optional[float]:
    """R-7/NumPy-linear quantile, returned as null for an empty sample."""
    data = sorted(float(value) for value in values if value is not None)
    if not data:
        return None
    if len(data) == 1:
        return data[0]
    position = (len(data) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    return data[lower] * (1.0 - fraction) + data[upper] * fraction


def distribution(values: Iterable[float]) -> Dict[str, Any]:
    data = [float(value) for value in values if value is not None]
    return {
        "n": len(data),
        "mean": sum(data) / len(data) if data else None,
        "p50": percentile(data, 0.50),
        "p95": percentile(data, 0.95),
        "p99": percentile(data, 0.99),
        "max": max(data) if data else None,
    }


def _nested(row: Dict[str, Any], *path: str) -> Any:
    value: Any = row
    for part in path:
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def _metrics(measurement: Any) -> Dict[str, Any]:
    if not isinstance(measurement, dict):
        return {}
    value = measurement.get("metrics")
    return value if isinstance(value, dict) else measurement


def _ratio_summary(pairs: List[tuple]) -> Dict[str, Any]:
    values = [active / full for active, full in pairs if full > 0]
    return {
        "coverage": {"measured": len(values)},
        "per_prefix": distribution(values),
        "ratio_of_sums": (sum(active for active, _ in pairs)
                          / sum(full for _, full in pairs)
                          if pairs else None),
    }


MEMORY_FIELDS = (
    "request_peak_resident_kv_tokens", "request_peak_resident_kv_bytes",
    # Line items of the resident total (never subtracted from it): evictable
    # prefix-cache slots at the decision chain's resident peak, and the
    # chain-wide maximum of evictable slots.
    "request_peak_cached_evictable_kv_tokens",
    "request_peak_cached_evictable_kv_bytes",
    "cached_evictable_kv_peak_tokens", "cached_evictable_kv_peak_bytes",
    "generation_active_kv_tokens", "generation_active_kv_bytes",
    "whole_full_kv_tokens", "whole_active_kv_tokens",
    "history_full_kv_tokens", "history_active_kv_tokens",
    "temporary_extraction_recovery_peak_kv_tokens",
    "temporary_extraction_recovery_peak_kv_bytes",
    "temporary_extraction_recovery_peak_storage_bytes",
    "torch_peak_allocated_bytes", "torch_peak_reserved_bytes",
    "nvml_process_peak_used_bytes", "nvml_process_peak_delta_bytes",
)

# Reported at the sample where the resident total peaked; they follow the
# resident peak across a decision chain instead of taking their own maximum.
AT_RESIDENT_PEAK_FIELDS = (
    "request_peak_cached_evictable_kv_tokens",
    "request_peak_cached_evictable_kv_bytes",
)

PEAK_FIELDS = {
    field for field in MEMORY_FIELDS
    if "peak" in field and field not in AT_RESIDENT_PEAK_FIELDS
}


def _merge_server_measurement(target: Dict[str, Any], update: Dict[str, Any],
                              phase: Optional[str]) -> None:
    generation_phase = phase in {
        "generation", "hiagent_retrieval_generation", "recovery_generation",
    }
    generation_fields = {
        "generation_active_kv_tokens", "generation_active_kv_bytes",
        "whole_full_kv_tokens", "whole_active_kv_tokens",
        "history_full_kv_tokens", "history_active_kv_tokens",
        "full_history_reprefill", "selection_query_tokens_observed",
        "canonical_full_source",
    }
    additive_fields = {"denominator_tokenization_duration_ns"}
    previous_resident = target.get("request_peak_resident_kv_bytes")
    update_resident = update.get("request_peak_resident_kv_bytes")
    takes_resident_peak = (
        isinstance(update_resident, (int, float))
        and not isinstance(update_resident, bool)
        and (not isinstance(previous_resident, (int, float))
             or isinstance(previous_resident, bool)
             or update_resident > previous_resident)
    )
    for key, value in update.items():
        if (key in additive_fields and isinstance(value, (int, float))
                and not isinstance(value, bool)):
            previous = target.get(key)
            target[key] = (
                previous if isinstance(previous, (int, float))
                and not isinstance(previous, bool) else 0
            ) + value
        elif key in additive_fields:
            # Auxiliary server rows can explicitly carry null before the
            # generation request reports this measurement-only overhead.
            # A null is absence, not an additive observation.
            continue
        elif key in PEAK_FIELDS and isinstance(value, (int, float)):
            previous = target.get(key)
            target[key] = (max(previous, value)
                           if isinstance(previous, (int, float)) else value)
        elif key in PEAK_FIELDS:
            continue
        elif key in AT_RESIDENT_PEAK_FIELDS:
            if takes_resident_peak or key not in target:
                target[key] = value
        elif key in generation_fields:
            if generation_phase or key not in target:
                target[key] = value
        elif key.startswith("baseline") or key.startswith("torch_start") \
                or key.startswith("nvml_process_start"):
            target.setdefault(key, value)
        else:
            target[key] = value


def _server_measurement(row: Dict[str, Any]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key in ("server_measurement", "measurement", "paper_measurement", "fields"):
        value = row.get(key)
        if isinstance(value, dict):
            metrics = value.get("metrics")
            if isinstance(metrics, dict):
                result.update(metrics)
            token_counts = value.get("token_counts")
            if isinstance(token_counts, dict):
                aliases = {
                    "full_equivalent_history": "history_full_kv_tokens",
                    "active_history_kv": "history_active_kv_tokens",
                }
                for source, target in aliases.items():
                    if source in token_counts and target not in result:
                        result[target] = token_counts[source]
            result.update(value)
    direct_metrics = row.get("metrics")
    if isinstance(direct_metrics, dict):
        result.update(direct_metrics)
    direct_tokens = row.get("token_counts")
    if isinstance(direct_tokens, dict):
        for source, target in {
            "full_equivalent_history": "history_full_kv_tokens",
            "active_history_kv": "history_active_kv_tokens",
        }.items():
            if source in direct_tokens and target not in result:
                result[target] = direct_tokens[source]
    for field in MEMORY_FIELDS:
        if field in row:
            result[field] = row[field]
    for source, target in {
        "nvml_process_start_bytes": "nvml_process_start_used_bytes",
        "nvml_process_peak_bytes": "nvml_process_peak_used_bytes",
        "nvml_process_end_bytes": "nvml_process_end_used_bytes",
    }.items():
        if source in result and target not in result:
            result[target] = result[source]
    return result


def aggregate(proxy_rows: List[Dict[str, Any]], harness_rows: List[Dict[str, Any]],
              replay_rows: Optional[List[Dict[str, Any]]] = None,
              server_rows: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    requests = [row for row in proxy_rows if row.get("event_type") == "request"]
    phases = [row for row in proxy_rows if row.get("event_type") == "phase"]
    server_rows = server_rows or []

    def add_embedded_phases(row: Dict[str, Any], prefix: str) -> None:
        value = row.get("paper_measurement") or row.get("server_measurement") or row
        items = value.get("phases") if isinstance(value, dict) else None
        for item in items or []:
            if isinstance(item, dict) and item.get("duration_ns") is not None:
                phases.append({
                    **item,
                    "phase": f"{prefix}:{item.get('name') or item.get('phase') or 'unknown'}",
                })

    if not server_rows:
        for row in requests:
            add_embedded_phases(row, "server")
    for row in server_rows:
        if row.get("event_type") in ("phase", "server_phase"):
            phases.append({**row, "phase": f"server:{row.get('phase') or 'unknown'}"})
        else:
            add_embedded_phases(row, "server")
    server_by_request: Dict[str, Dict[str, Any]] = {}
    for row in server_rows:
        request_id = row.get("outer_request_id") or row.get("request_id")
        if request_id:
            _merge_server_measurement(
                server_by_request.setdefault(str(request_id), {}),
                _server_measurement(row), row.get("phase"))
    request_by_id = {row.get("request_id"): row for row in requests if row.get("request_id")}
    actions = [row for row in harness_rows if row.get("event_type") == "tool_action"]
    episodes = [row for row in harness_rows if row.get("event_type") == "episode_end"]

    def server_value(row: Dict[str, Any], field: str) -> Any:
        value = server_by_request.get(str(row.get("request_id")), {}).get(field)
        if value is None:
            value = _nested(row, "server_measurement", field)
        return value

    def algorithm_duration_ns(row: Dict[str, Any]) -> Optional[float]:
        duration = row.get("duration_ns")
        if not isinstance(duration, (int, float)) or isinstance(duration, bool):
            return None
        overhead = server_value(row, "denominator_tokenization_duration_ns")
        if not isinstance(overhead, (int, float)) or isinstance(overhead, bool):
            overhead = 0
        return max(0, duration - overhead)

    associated_chain_ms = []
    commit_wall_ms = []
    joined = 0
    actions_by_request: Dict[str, List[Dict[str, Any]]] = {}
    for action in actions:
        request_id = action.get("decision_request_id")
        if request_id:
            actions_by_request.setdefault(str(request_id), []).append(action)
    for request_id, committed in actions_by_request.items():
        request = request_by_id.get(request_id)
        adjusted_duration = algorithm_duration_ns(request) if request else None
        if adjusted_duration is not None:
            # Allocate one complete model-side decision chain equally across
            # the actions it committed. Repeating the allocation once per
            # action makes the mean exactly total model-side duration divided
            # by total committed actions, without adding external-tool time.
            allocation_ms = adjusted_duration / len(committed) / 1e6
            associated_chain_ms.extend([allocation_ms] * len(committed))
        for action in committed:
            if (request and request.get("start_unix_ns") is not None
                    and action.get("end_unix_ns") is not None):
                commit_wall_ms.append(
                    (action["end_unix_ns"] - request["start_unix_ns"]) / 1e6)
            if request:
                joined += 1

    phase_names = sorted({str(row.get("phase")) for row in phases})
    phase_summary = {
        name: distribution(row.get("duration_ns") / 1e6 for row in phases
                           if str(row.get("phase")) == name and row.get("duration_ns") is not None)
        for name in phase_names
    }
    memory = {}
    for field in MEMORY_FIELDS:
        values = []
        for row in requests:
            request_id = row.get("request_id")
            value = server_value(row, field)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                values.append(float(value))
        memory[field] = {
            "coverage": {"measured": len(values), "requests": len(requests)},
            **distribution(values),
        }
    # The decision chain that holds the cell's resident peak, with its own
    # line items, so the report can pair "how much of that peak was evictable
    # cache" with the peak itself instead of mixing maxima across chains.
    resident_peak_chain = None
    for row in requests:
        resident = server_value(row, "request_peak_resident_kv_bytes")
        if not isinstance(resident, (int, float)) or isinstance(resident, bool):
            continue
        if resident_peak_chain is None or resident > resident_peak_chain["request_peak_resident_kv_bytes"]:
            resident_peak_chain = {
                "request_id": row.get("request_id"),
                "request_peak_resident_kv_bytes": resident,
                "request_peak_cached_evictable_kv_bytes": server_value(
                    row, "request_peak_cached_evictable_kv_bytes"),
                "cached_evictable_kv_peak_bytes": server_value(
                    row, "cached_evictable_kv_peak_bytes"),
                "generation_active_kv_bytes": server_value(
                    row, "generation_active_kv_bytes"),
            }
    memory["resident_peak_chain"] = resident_peak_chain

    ratios = {}
    for scope in ("whole", "history"):
        full_key = f"{scope}_full_kv_tokens"
        active_key = f"{scope}_active_kv_tokens"
        values = []
        for row in requests:
            request_id = str(row.get("request_id"))
            merged = server_by_request.get(request_id, {})
            full = merged.get(full_key, _nested(row, "server_measurement", full_key))
            active = merged.get(active_key, _nested(row, "server_measurement", active_key))
            if isinstance(full, int) and isinstance(active, int) and full > 0:
                values.append(active / full)
        measured_pairs = []
        for row in requests:
            merged = server_by_request.get(str(row.get("request_id")), {})
            full = merged.get(full_key, _nested(row, "server_measurement", full_key))
            active = merged.get(active_key, _nested(row, "server_measurement", active_key))
            if isinstance(full, int) and isinstance(active, int) and full > 0:
                measured_pairs.append((active, full))
        ratios[scope] = {
            "definition": f"sum({active_key}) / sum({full_key})",
            "coverage": {"measured": len(values), "requests": len(requests)},
            "per_request": distribution(values),
            "ratio_of_sums": (sum(pair[0] for pair in measured_pairs)
                              / sum(pair[1] for pair in measured_pairs)
                              if measured_pairs else None),
        }

    replay_rows = replay_rows or []
    replay = [row for row in replay_rows if row.get("event_type") == "prefix_replay"]
    common_prefix_ratios = {}
    for scope in ("whole", "history"):
        pairs = []
        full_key = f"{scope}_full_kv_tokens"
        active_key = f"{scope}_active_kv_tokens"
        for row in replay:
            source = _metrics(row.get("source_paper_measurement"))
            target = _metrics(row.get("target_paper_measurement"))
            full = source.get(full_key)
            active = target.get(active_key)
            if (isinstance(full, int) and not isinstance(full, bool) and full > 0
                    and isinstance(active, int) and not isinstance(active, bool)
                    and active >= 0):
                pairs.append((active, full))
        common_prefix_ratios[scope] = {
            "definition": (
                f"target {active_key} / canonical Full source {full_key}, "
                "joined by recorded prefix"
            ),
            "prefixes": len(replay),
            **_ratio_summary(pairs),
        }
    observed_model_side_ns = sum(
        row["duration_ns"] for row in requests
        if isinstance(row.get("duration_ns"), (int, float))
        and not isinstance(row.get("duration_ns"), bool))
    denominator_tokenization_ns = sum(
        value for row in requests
        for value in [server_value(row, "denominator_tokenization_duration_ns")]
        if isinstance(value, (int, float)) and not isinstance(value, bool))
    total_model_side_ns = sum(
        value for row in requests for value in [algorithm_duration_ns(row)]
        if value is not None)
    committed_actions = len(actions)
    zero_action_requests = [
        row for row in requests
        if not actions_by_request.get(str(row.get("request_id")))
    ]
    return {
        "schema": "c2kv.measurement.summary.v1",
        "counts": {
            "requests": len(requests), "phases": len(phases),
            "tool_actions": len(actions), "episodes": len(episodes),
            "joined_actions": joined, "prefix_replays": len(replay),
            "server_events": len(server_rows),
        },
        "latency_ms": {
            "request": distribution(row.get("duration_ns") / 1e6 for row in requests
                                    if row.get("duration_ns") is not None),
            "request_algorithm": distribution(
                value / 1e6 for row in requests
                for value in [algorithm_duration_ns(row)] if value is not None),
            "measurement_denominator_tokenization": distribution(
                value / 1e6 for row in requests
                for value in [server_value(
                    row, "denominator_tokenization_duration_ns")]
                if isinstance(value, (int, float)) and not isinstance(value, bool)),
            "phase": phase_summary,
            "complete_model_side_per_committed_action": {
                "definition": (
                    "sum(duration_ns of every proxy request, including failed/empty "
                    "decisions and retries inside each request, minus measurement-only "
                    "denominator tokenization) / committed tool actions"
                ),
                "observed_model_side_ns": observed_model_side_ns,
                "measurement_denominator_tokenization_ns": denominator_tokenization_ns,
                "total_model_side_ns": total_model_side_ns,
                "committed_actions": committed_actions,
                "mean_ms": (total_model_side_ns / committed_actions / 1e6
                            if committed_actions else None),
            },
            "action_associated_model_side_allocation": {
                "policy": (
                    "for a decision that commits k actions, divide its complete "
                    "model-side duration by k and emit that allocation k times"
                ),
                **distribution(associated_chain_ms),
            },
            "zero_action_model_side": {
                "requests": len(zero_action_requests),
                **distribution(
                    value / 1e6 for row in zero_action_requests
                    for value in [algorithm_duration_ns(row)] if value is not None),
            },
            "decision_start_to_action_commit_wall": distribution(commit_wall_ms),
            "external_tool": distribution(row.get("duration_ns") / 1e6 for row in actions
                                          if row.get("duration_ns") is not None),
            "episode_wall": distribution(row.get("duration_ns") / 1e6 for row in episodes
                                         if row.get("duration_ns") is not None),
            "prefix_replay": distribution(row.get("duration_ns") / 1e6 for row in replay
                                          if row.get("duration_ns") is not None),
            "prefix_replay_algorithm": distribution(
                value / 1e6 for row in replay
                for value in [algorithm_duration_ns(row)] if value is not None),
        },
        "memory": memory,
        "token_ratios": ratios,
        "common_prefix_token_ratios": common_prefix_ratios,
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proxy", required=True)
    parser.add_argument("--harness")
    parser.add_argument("--replay")
    parser.add_argument("--server")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    result = aggregate(
        list(read_jsonl(args.proxy)),
        list(read_jsonl(args.harness)) if args.harness else [],
        list(read_jsonl(args.replay)) if args.replay else None,
        list(read_jsonl(args.server)) if args.server else None,
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
