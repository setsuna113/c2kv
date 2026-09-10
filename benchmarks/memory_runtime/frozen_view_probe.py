"""Prepare and execute the bounded same-prefix frozen-view diagnostic.

Preparation is CPU/tokenizer-only.  It reconstructs six immutable backend
payloads from the two declared capacity-protect prefixes and records the exact
gist materialization ledger.  Execution materializes those gists once, verifies
the frozen payloads, and submits the predeclared 24 single-generation cells.
Generated responses are recorded but never executed, scored, or fed back.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import signal
import sys
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from urllib import request as urlrequest
from urllib.error import HTTPError, URLError


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent))

import proxy
from arms import get_arm
from backends import get_backend
from memory_runtime.adapter import RuntimeAdapter
from memory_runtime.attempt_journal import AttemptJournal, summarize_attempt_journal
from memory_runtime.extraction_telemetry import ExtractionBudget
from memory_runtime.generation_budget import GenerationBudget


DESIGN_SCHEMA = "a-frozen-view-probe-design-v1"
PREPARED_SCHEMA = "a-frozen-view-prepared-v1"
RECEIPT_SCHEMA = "a-frozen-view-live-receipt-v1"
DEFAULT_CONFIG = HERE / "configs/frozen_view_dev1.json"
PREPARED_NAME = "prepared.json"
CELLS_NAME = "cells.jsonl"
MATERIALIZATIONS_NAME = "materializations.jsonl"
TRANSPORT_NAME = "transport.jsonl"
ATTEMPTS_NAME = "attempts.jsonl"

EXPECTED_VIEWS = {
    "A": {"name": "full_original", "mode": None, "arm": "full"},
    "B": {"name": "full_shared", "mode": "full_exact_shared", "arm": "full",
          "max_retrieved_events": 1},
    "C": {"name": "capacity_protect", "mode": "capacity_protect", "arm": "c2kv4",
          "max_retrieved_events": 2},
    "D": {"name": "no_gist", "mode": "capacity_exact_no_gist", "arm": "full",
          "max_retrieved_events": 1},
}
EXPECTED_BLOCKS = [
    {"block": 1, "negative_order": ["A", "C"], "main_order": ["A", "B", "D", "C"]},
    {"block": 2, "negative_order": ["C", "A"], "main_order": ["B", "C", "A", "D"]},
    {"block": 3, "negative_order": ["A", "C"], "main_order": ["C", "D", "B", "A"]},
    {"block": 4, "negative_order": ["C", "A"], "main_order": ["D", "A", "C", "B"]},
]


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _append_jsonl(path: Path, value: Any) -> None:
    encoded = (json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_APPEND | os.O_CREAT | os.O_WRONLY | getattr(os, "O_BINARY", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        remaining = memoryview(encoded)
        while remaining:
            written = os.write(descriptor, remaining)
            if written <= 0:
                raise OSError("JSONL append made no progress")
            remaining = remaining[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _require_equal(value: Any, expected: Any, label: str) -> None:
    if value != expected:
        raise ValueError(f"Frozen design differs at {label}")


def validate_design(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Validate the complete fixed design and return its 24-cell schedule."""
    _require_equal(config.get("schema"), DESIGN_SCHEMA, "schema")
    _require_equal(config.get("design"), "frozen-view-dev1", "design")
    _require_equal(config.get("source_log"),
                   "capacity_protect/logs/proxy_c2kv4_38273.jsonl", "source_log")
    _require_equal(config.get("extraction_response_source"),
                   "../telemetry_probe_v1/transport.jsonl", "extraction_response_source")
    _require_equal(config.get("task_id"), "multi_turn_base_1", "task_id")
    _require_equal(config.get("source_prefixes"), {
        "negative": {"source_line": 10, "user_turn": 2, "step": 1,
                     "compression_activated": False},
        "main": {"source_line": 11, "user_turn": 2, "step": 2,
                 "compression_activated": True},
    }, "source_prefixes")
    _require_equal(config.get("views"), EXPECTED_VIEWS, "views")
    _require_equal(config.get("negative_views"), ["A", "C"], "negative_views")
    _require_equal(config.get("blocks"), EXPECTED_BLOCKS, "blocks")
    _require_equal(config.get("sampling"),
                   {"temperature": 0.001, "seed": 0, "max_tokens": 4096}, "sampling")
    fixed = {
        "query_projection": "base",
        "doc_packing": "turn",
        "max_doc_length": 512,
        "max_doc_num": 12,
        "bytes_per_kv_token": 147456,
        "history_budget_bytes": 113246208,
        "workspace_budget_bytes": 113246208,
        "lease_decisions": 3,
        "maximum_generation_attempts": 24,
        "maximum_extraction_attempts": 24,
        "maximum_wall_seconds": 900,
        "generation_attempts_per_cell": 1,
        "regeneration": False,
        "tool_execution": False,
        "scorer_access": False,
        "response_feedback": False,
        "transport_retries": 0,
        "cache_miss_retries": 0,
        "automatic_reruns": 0,
        "unused_budget_transfer_or_expansion": False,
    }
    for key, expected in fixed.items():
        _require_equal(config.get(key), expected, key)

    schedule = []
    for block in config["blocks"]:
        for phase, order_key in (("negative", "negative_order"), ("main", "main_order")):
            for position, view in enumerate(block[order_key], 1):
                schedule.append({
                    "cell_id": f"block{block['block']}_{phase}_{position}_{view}",
                    "block": block["block"], "phase": phase,
                    "position": position, "view": view,
                })
    if len(schedule) != config["maximum_generation_attempts"]:
        raise ValueError("Frozen schedule does not contain exactly 24 cells")
    main_counts = {label: sum(
        cell["phase"] == "main" and cell["view"] == label for cell in schedule
    ) for label in EXPECTED_VIEWS}
    negative_counts = {label: sum(
        cell["phase"] == "negative" and cell["view"] == label for cell in schedule
    ) for label in ("A", "C")}
    if main_counts != {label: 4 for label in EXPECTED_VIEWS} or negative_counts != {
        "A": 4, "C": 4
    }:
        raise ValueError("Frozen schedule is not the declared balanced four-block design")
    return schedule


def _load_source_bundle(source_root: Path) -> dict[str, Any]:
    standalone = source_root / "source_bundle.json"
    if standalone.is_file():
        return {"path": str(standalone.resolve()), **_read_object(standalone)}
    pilot_path = source_root / "pilot.json"
    pilot = _read_object(pilot_path)
    bundle = pilot.get("source_bundle")
    if not isinstance(bundle, dict):
        raise ValueError("Source root has neither source_bundle.json nor pilot.json source_bundle")
    return {**copy.deepcopy(bundle), "loaded_from": f"{pilot_path.resolve()}#source_bundle"}


def _load_source_pair(source_root: Path, config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    path = source_root / config["source_log"]
    if not path.is_file():
        raise ValueError(f"Missing frozen capacity-protect request log: {path}")
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            rows.append((number, json.loads(line)))
    selected = {}
    for phase, declared in config["source_prefixes"].items():
        matches = [(number, row) for number, row in rows
                   if number == declared["source_line"]]
        if len(matches) != 1:
            raise ValueError(f"Frozen source line is missing or duplicated: {phase}")
        number, row = matches[0]
        context = row.get("eval_context") or {}
        gate = ((row.get("memory_runtime") or {}).get("capacity_gate") or {}).get(
            "compression_activated"
        )
        if (row.get("status") != "ok" or context.get("task_id") != config["task_id"]
                or context.get("user_turn") != declared["user_turn"]
                or context.get("step") != declared["step"]
                or context.get("attempt") != 0
                or gate is not declared["compression_activated"]):
            raise ValueError(f"Frozen source identity or capacity gate differs: {phase}")
        runtime = row.get("memory_runtime") or {}
        if runtime.get("history_budget_bytes") != config["history_budget_bytes"]:
            raise ValueError(f"Frozen source history budget differs: {phase}")
        request_view = row.get("request_view")
        if (not isinstance(request_view, dict)
                or not isinstance(request_view.get("messages"), list)
                or not isinstance(request_view.get("model"), str)):
            raise ValueError(f"Frozen source lacks a complete native request view: {phase}")
        selected[phase] = {
            "source_path": config["source_log"], "source_line": number,
            "user_turn": declared["user_turn"], "step": declared["step"], "row": row,
        }
    first_active = next((number for number, row in rows
                         if (row.get("eval_context") or {}).get("task_id") == config["task_id"]
                         and (row.get("eval_context") or {}).get("attempt") == 0
                         and ((row.get("memory_runtime") or {}).get("capacity_gate") or {}).get(
                             "compression_activated") is True), None)
    if first_active != config["source_prefixes"]["main"]["source_line"]:
        raise ValueError("The declared main prefix is no longer the first capacity activation")
    return selected


def _build_extraction_manifest(
    source_root: Path, main_row: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    runtime = main_row.get("memory_runtime") or {}
    refs = runtime.get("block_refs")
    forwarded = main_row.get("forwarded_request_views")
    if (not isinstance(refs, list) or not refs or not isinstance(forwarded, list)
            or len(forwarded) != 1 or not isinstance(forwarded[0], dict)):
        raise ValueError("Main source lacks one complete recorded capacity wire")
    messages = forwarded[0].get("messages")
    if not isinstance(messages, list):
        raise ValueError("Recorded capacity wire lacks messages")
    carriers = {}
    for message in messages:
        key = message.get("c2kv_key_hash") if isinstance(message, dict) else None
        if key:
            if key in carriers:
                raise ValueError(f"Recorded capacity wire repeats gist key {key}")
            carriers[key] = message
    ref_keys = [item.get("key_hash") for item in refs]
    if (any(not isinstance(key, str) or not key for key in ref_keys)
            or set(ref_keys) != set(carriers) or len(ref_keys) != len(carriers)):
        raise ValueError("Recorded block_refs and forwarded gist carriers differ")

    ledger = (source_root / config["extraction_response_source"]).resolve()
    if not ledger.is_file():
        raise ValueError(f"Missing declared extraction response ledger: {ledger}")
    responses: dict[str, dict[str, Any]] = {}
    response_lines: dict[str, int] = {}
    for number, line in enumerate(ledger.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        response = row.get("response") if row.get("path") == "/v1/c2kv/extract" else None
        key = response.get("key_hash") if isinstance(response, dict) else None
        if key not in set(ref_keys):
            continue
        record = {field: response.get(field) for field in (
            "key_hash", "gist_len", "original_seq_len"
        )}
        if (row.get("status") != "completed" or response.get("success") is not True
                or any(type(record[field]) is not int or record[field] <= 0
                       for field in ("gist_len", "original_seq_len"))):
            raise ValueError(f"Invalid extraction ledger record for {key}")
        previous = responses.get(key)
        if previous is not None and previous != record:
            raise ValueError(f"Conflicting extraction ledger records for {key}")
        responses[key] = record
        response_lines.setdefault(key, number)
    if set(responses) != set(ref_keys):
        raise ValueError("Extraction telemetry does not cover every retained gist key")

    # Turn-packed history extraction calls proxy._extract(role, text, ratio,
    # timeout) without tools.  The generation payload still carries the full
    # request tool schemas; only this materialization ledger uses empty tools.
    extraction_tools: list[dict[str, Any]] = []
    tools_hash = proxy._digest(extraction_tools)
    items = []
    for ref in refs:
        key = ref["key_hash"]
        carrier = carriers[key]
        content = carrier.get("content")
        role = carrier.get("role")
        ratio = carrier.get("c2kv_ratio")
        expected = responses[key]
        if (not isinstance(content, str) or not isinstance(role, str)
                or type(ratio) is not int or ratio <= 0
                or expected["gist_len"] != ref.get("gist_tokens")):
            raise ValueError(f"Incomplete or inconsistent gist materialization record for {key}")
        items.append({
            "source_indices": copy.deepcopy(ref.get("source_indices")),
            "role": role, "content": content, "ratio": ratio,
            "cache_key": {
                "role": role, "content_hash": proxy._content_key(role, content),
                "ratio": ratio, "tools_hash": tools_hash,
            },
            "expected_response": expected,
            "ledger_line": response_lines[key],
        })
    return {
        "count": len(items), "tools": extraction_tools, "items": items,
        "tools_scope": "turn-packed history extraction omits request tools",
        "provenance": {"path": str(ledger.resolve()), "sha256": _file_digest(ledger)},
    }


class _ReplayExtractor:
    def __init__(self, manifest: dict[str, Any]):
        self._records = {
            (item["cache_key"]["role"], item["cache_key"]["content_hash"],
             item["cache_key"]["ratio"], item["cache_key"]["tools_hash"]):
            copy.deepcopy(item["expected_response"])
            for item in manifest["items"]
        }
        self.seen: list[tuple[Any, ...]] = []

    def __call__(self, role: str, content: str, ratio: int, timeout: int = 600,
                 tools: list[dict[str, Any]] | None = None, force: bool = False):
        del timeout
        if force:
            raise AssertionError("CPU preparation cannot force a live extraction")
        key = (role, proxy._content_key(role, content), ratio, proxy._digest(tools or []))
        if key not in self._records:
            raise AssertionError("CPU preparation requested an undeclared extraction")
        self.seen.append(key)
        return copy.deepcopy(self._records[key])


def _runtime_config(config: dict[str, Any], view: str, run_id: str) -> dict[str, Any]:
    declared = config["views"][view]
    return {
        "mode": declared["mode"], "run_id": run_id,
        "bytes_per_kv_token": config["bytes_per_kv_token"],
        "history_budget_bytes": config["history_budget_bytes"],
        "workspace_budget_bytes": config["workspace_budget_bytes"],
        "lease_decisions": config["lease_decisions"],
        "max_retrieved_events": declared["max_retrieved_events"],
    }


def _base_payload(row: dict[str, Any], sampling: dict[str, Any]) -> dict[str, Any]:
    native = row["request_view"]
    payload = {key: copy.deepcopy(native[key]) for key in ("model", "messages", "tools")
               if key in native}
    payload.update(copy.deepcopy(sampling))
    return payload


def _prepare_backend_payload(backend, payload, messages, arm):
    staged = copy.deepcopy(payload)
    staged["messages"] = copy.deepcopy(messages)
    staged["c2kv_use_gist_projection"] = False
    return backend.prepare_chat(staged, arm, None, context={
        "conversation_id": "frozen-view", "history_kv": None, "kv_reuse": None,
    })


def _prepare_one(
    config: dict[str, Any], phase: str, view: str, source: dict[str, Any], counter,
    backend, replay_extract: _ReplayExtractor,
) -> dict[str, Any]:
    row = source["row"]
    native = row["request_view"]
    messages = native["messages"]
    tools = native.get("tools")
    full, full_counts = proxy._assemble(messages, get_arm("full"))
    context = copy.deepcopy(row["eval_context"])
    context["run_id"] = f"{config['design']}-{phase}-{view}"
    counts = full_counts
    out_messages = full
    runtime_metadata = None
    mode = config["views"][view]["mode"]
    if mode is not None:
        runtime = RuntimeAdapter(_runtime_config(config, view, context["run_id"]), counter)
        if mode in {"full_exact_shared", "capacity_exact_no_gist"}:
            def forbidden(*args, **kwargs):
                raise AssertionError("Full-shared/NoGist CPU preparation cannot extract")
            out_messages, counts, _prepared = runtime.prepare_exact(
                messages, full, full_counts, context, tools,
                render_compressed=forbidden,
            )
        else:
            out_messages, counts = runtime.apply(
                messages, full, full_counts, context, tools,
                render_compressed=lambda source_messages: proxy._assemble(
                    source_messages, get_arm("c2kv4")
                ),
            )
        runtime_metadata = counts["memory_runtime"]

    expected_active = config["source_prefixes"][phase]["compression_activated"]
    if runtime_metadata is not None:
        if mode == "full_exact_shared":
            observed_active = runtime_metadata["auxiliary_gate"]["auxiliary_activated"]
        else:
            observed_active = runtime_metadata["capacity_gate"]["compression_activated"]
        if observed_active is not expected_active:
            raise ValueError(f"Prepared {phase}:{view} capacity gate differs")

    arm = get_arm(config["views"][view]["arm"])
    payload = _prepare_backend_payload(
        backend, _base_payload(row, config["sampling"]), out_messages, arm
    )
    raw_messages = [message for message in payload["messages"]
                    if not message.get("c2kv_key_hash")]
    raw_prompt_tokens = counter(raw_messages, payload.get("tools"))
    evidence = {"selected_event_ids": [], "packet": None, "selection_bytes": 0}
    if runtime_metadata is not None:
        index = runtime_metadata.get("evidence_out_index")
        packet = copy.deepcopy(out_messages[index]) if type(index) is int else None
        evidence = {
            "selected_event_ids": copy.deepcopy(runtime_metadata.get("selected_event_ids") or []),
            "packet": packet,
            "selection_bytes": runtime_metadata.get(
                "auxiliary_selection_bytes", runtime_metadata.get("evidence_bytes")
            ),
        }
    return {
        "phase": phase, "view": view, "view_name": config["views"][view]["name"],
        "source": {key: source[key] for key in (
            "source_path", "source_line", "user_turn", "step"
        )},
        "context": context, "payload": payload, "wire_digest": _digest(payload),
        "counts": copy.deepcopy(counts), "memory_runtime": copy.deepcopy(runtime_metadata),
        "evidence": evidence,
        "server_expectations": {
            "raw_prompt_tokens": raw_prompt_tokens,
            "bytes_per_kv_token": config["bytes_per_kv_token"],
            "c2kv_query_proj_effective": config["query_projection"],
            "c2kv_query_proj_decode_verified": True,
            "c2kv_tools_dump": "full",
            "c2kv_gist_seen": any(
                message.get("c2kv_key_hash") for message in payload["messages"]
            ),
        },
    }


def _validate_prepared_views(
    views: dict[str, dict[str, Any]], main_recorded_wire: list[dict[str, Any]]
) -> None:
    expected_keys = {"negative:A", "negative:C", "main:A", "main:B", "main:C", "main:D"}
    if set(views) != expected_keys:
        raise ValueError("Prepared artifact does not contain exactly six views")
    if views["negative:A"]["payload"] != views["negative:C"]["payload"]:
        raise ValueError("Negative Full and capacity routes are not complete wire identity")
    if views["main:C"]["payload"]["messages"] != main_recorded_wire:
        raise ValueError("Fresh capacity preparation differs from the recorded initial wire")
    full_payload = views["main:A"]["payload"]
    shared_payload = copy.deepcopy(views["main:B"]["payload"])
    shared_meta = views["main:B"]["memory_runtime"]
    shared_index = shared_meta.get("evidence_out_index")
    if type(shared_index) is not int or not 0 <= shared_index < len(shared_payload["messages"]):
        raise ValueError("Full-shared evidence index is invalid")
    shared_payload["messages"].pop(shared_index)
    if shared_payload != full_payload:
        raise ValueError("Full-shared minus its evidence packet does not preserve Full payload")
    evidence = [views[f"main:{label}"]["evidence"] for label in ("B", "C", "D")]
    if any(item["packet"] is None or not item["selected_event_ids"] for item in evidence):
        raise ValueError("One main protected view lacks initial evidence")
    if any(item != evidence[0] for item in evidence[1:]):
        raise ValueError("Main B/C/D do not use one exact common initial evidence packet")
    for label in ("B", "C", "D"):
        metadata = views[f"main:{label}"]["memory_runtime"]
        if metadata.get("retrieved_event_ids") or metadata.get("retained_event_ids"):
            raise ValueError("Prepared view did not start from empty acquisition state")
    if any(message.get("c2kv_key_hash") for message in views["main:B"]["payload"]["messages"]):
        raise ValueError("Full-shared unexpectedly contains gist carriers")
    if any(message.get("c2kv_key_hash") for message in views["main:D"]["payload"]["messages"]):
        raise ValueError("NoGist unexpectedly contains gist carriers")
    if not any(message.get("c2kv_key_hash")
               for message in views["main:C"]["payload"]["messages"]):
        raise ValueError("Capacity view contains no gist carriers")
    bytes_per_token = views["main:A"]["server_expectations"]["bytes_per_kv_token"]
    full_tokens = views["main:A"]["server_expectations"]["raw_prompt_tokens"]
    shared_tokens = views["main:B"]["server_expectations"]["raw_prompt_tokens"]
    if (shared_meta.get("budget_applies") is not False
            or (shared_meta.get("auxiliary_gate") or {}).get("full_raw_prompt_tokens")
            != full_tokens
            or (shared_tokens - full_tokens) * bytes_per_token
            != shared_meta.get("evidence_bytes")
            or shared_meta.get("gist_tokens") != 0
            or views["main:B"]["counts"].get("compressed_records") not in (None, [])):
        raise ValueError("Full-shared Full/E geometry is inconsistent")
    no_gist = views["main:D"]["memory_runtime"]
    if (set(no_gist.get("raw_history_event_ids") or [])
            & set(no_gist.get("selected_event_ids") or [])):
        raise ValueError("NoGist duplicates raw history and evidence events")
    if (no_gist.get("active_history_bytes") > no_gist.get("history_budget_bytes")
            or no_gist.get("gist_tokens") != 0
            or views["main:D"]["counts"].get("compressed_records") not in (None, [])):
        raise ValueError("NoGist R/E geometry exceeds B or retains gist")


def build_prepared(
    config: dict[str, Any], source_root: Path, checkpoint: str, *,
    counter=None, backend=None,
) -> dict[str, Any]:
    """Build the in-memory frozen artifact without generation or live extraction."""
    cpu_started = time.monotonic()
    schedule = validate_design(config)
    source_root = source_root.resolve()
    sources = _load_source_pair(source_root, config)
    manifest = _build_extraction_manifest(
        source_root, sources["main"]["row"], config
    )
    if manifest["count"] > config["maximum_extraction_attempts"]:
        raise ValueError("Required gist materializations exceed the frozen extraction cap")
    if counter is None:
        prototype = RuntimeAdapter.from_config(
            str(HERE / "configs/full_exact_shared.json"), checkpoint
        )
        counter = prototype._token_counter
    if backend is None:
        def forbidden(*args, **kwargs):
            raise AssertionError("CPU preparation cannot make network requests")
        backend = get_backend("sglang", forbidden)
    replay = _ReplayExtractor(manifest)
    globals_before = {name: getattr(proxy, name) for name in (
        "DOC_PACKING", "QUERY_PROJECTION", "MAX_DOC_LENGTH", "MAX_DOC_NUM", "_extract"
    )}
    try:
        proxy.DOC_PACKING = config["doc_packing"]
        proxy.QUERY_PROJECTION = config["query_projection"]
        proxy.MAX_DOC_LENGTH = config["max_doc_length"]
        proxy.MAX_DOC_NUM = config["max_doc_num"]
        proxy._extract = replay
        views = {}
        for phase, labels in (("negative", ("A", "C")),
                              ("main", ("A", "B", "C", "D"))):
            for label in labels:
                views[f"{phase}:{label}"] = _prepare_one(
                    config, phase, label, sources[phase], counter, backend, replay
                )
    finally:
        for name, value in globals_before.items():
            setattr(proxy, name, value)
    expected_replay = [
        (item["cache_key"]["role"], item["cache_key"]["content_hash"],
         item["cache_key"]["ratio"], item["cache_key"]["tools_hash"])
        for item in manifest["items"]
    ]
    if replay.seen != expected_replay:
        raise ValueError("CPU capacity reconstruction did not consume the exact gist manifest once")
    recorded_views = sources["main"]["row"]["forwarded_request_views"]
    _validate_prepared_views(views, recorded_views[0]["messages"])
    artifact = {
        "schema": PREPARED_SCHEMA, "status": "prepared", "design": copy.deepcopy(config),
        "source_root": str(source_root), "source_bundle": _load_source_bundle(source_root),
        "checkpoint": checkpoint, "schedule": schedule, "views": views,
        "extraction_manifest": manifest,
        "cpu_preflight": {
            "generation_requests": 0, "model_requests": 0, "network_requests": 0,
            "replayed_extractions": len(replay.seen),
            "wall_seconds": time.monotonic() - cpu_started,
        },
        "validations": {
            "six_views_prepared": True, "negative_complete_payload_identity": True,
            "main_common_evidence_exact": True, "fresh_empty_acquisition_state": True,
            "capacity_view_matches_recorded_initial_wire": True,
            "no_reconsider_or_regeneration": True,
        },
    }
    artifact["freeze_digest"] = _digest({
        "design": artifact["design"], "schedule": artifact["schedule"],
        "views": artifact["views"], "extraction_manifest": artifact["extraction_manifest"],
    })
    return artifact


def _prepared_output(path: Path) -> Path:
    return path if path.suffix.lower() == ".json" else path / PREPARED_NAME


def prepare_probe(config_path: Path, source_root: Path, checkpoint: str, out: Path) -> Path:
    target = _prepared_output(out)
    if out.exists() or target.exists():
        raise FileExistsError("Output exists; automatic rerun or overwrite is prohibited")
    config = _read_object(config_path)
    artifact = build_prepared(config, source_root, checkpoint)
    if out.suffix.lower() == ".json":
        target.parent.mkdir(parents=True, exist_ok=True)
    else:
        target.parent.mkdir(parents=True, exist_ok=False)
    _save(target, artifact)
    return target


def _validate_prepared(artifact: dict[str, Any]) -> None:
    if artifact.get("schema") != PREPARED_SCHEMA or artifact.get("status") != "prepared":
        raise ValueError("Prepared artifact has invalid schema or status")
    validate_design(artifact.get("design") or {})
    expected = _digest({
        "design": artifact["design"], "schedule": artifact["schedule"],
        "views": artifact["views"], "extraction_manifest": artifact["extraction_manifest"],
    })
    if artifact.get("freeze_digest") != expected:
        raise ValueError("Prepared freeze digest differs")
    expected_schedule = validate_design(artifact["design"])
    if artifact.get("schedule") != expected_schedule:
        raise ValueError("Prepared schedule differs from design")
    views = artifact.get("views") or {}
    for key in ("negative:A", "negative:C", "main:A", "main:B", "main:C", "main:D"):
        view = views.get(key) or {}
        if view.get("wire_digest") != _digest(view.get("payload")):
            raise ValueError(f"Prepared payload digest differs: {key}")
    if views["negative:A"]["payload"] != views["negative:C"]["payload"]:
        raise ValueError("Prepared negative payloads no longer match exactly")
    manifest = artifact.get("extraction_manifest") or {}
    items = manifest.get("items")
    limit = artifact["design"]["maximum_extraction_attempts"]
    if not isinstance(items, list) or len(items) != manifest.get("count") or len(items) > limit:
        raise ValueError("Prepared extraction manifest count is invalid")
    for item in items:
        expected_response = item.get("expected_response") or {}
        if (not isinstance(item.get("content"), str)
                or type(item.get("ratio")) is not int or item["ratio"] <= 0
                or any(type(expected_response.get(field)) is not int
                       or expected_response[field] <= 0
                       for field in ("gist_len", "original_seq_len"))
                or not isinstance(expected_response.get("key_hash"), str)):
            raise ValueError("Prepared extraction manifest has an incomplete record")


class LiveAttemptError(RuntimeError):
    def __init__(self, message: str, attempt_index: int | None = None):
        super().__init__(message)
        self.attempt_index = attempt_index


class LiveWallTimeout(TimeoutError):
    """Raised by the Linux process-wide timer for the fixed live wall cap."""


class _AbsoluteWallTimer:
    def __init__(self, deadline: float):
        self.deadline = deadline
        self._armed = False
        self._old_handler = None

    def arm(self) -> None:
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise LiveWallTimeout("Frozen 900-second live wall budget exhausted")
        if not hasattr(signal, "setitimer"):
            return
        self._old_handler = signal.getsignal(signal.SIGALRM)

        def expired(_signum, _frame):
            raise LiveWallTimeout("Frozen 900-second live wall budget exhausted")

        signal.signal(signal.SIGALRM, expired)
        signal.setitimer(signal.ITIMER_REAL, remaining)
        self._armed = True

    def cancel(self) -> None:
        if not self._armed:
            return
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, self._old_handler)
        self._armed = False


class _LiveTransport:
    def __init__(self, upstream: str, deadline: float, journal: AttemptJournal,
                 transport_path: Path, generation_limit: int, extraction_limit: int,
                 opener=None):
        self.upstream = upstream.rstrip("/")
        self.deadline = deadline
        self.journal = journal
        self.transport_path = transport_path
        self.generation = GenerationBudget(generation_limit)
        self.extraction = ExtractionBudget(extraction_limit)
        self.opener = opener or urlrequest.build_opener(urlrequest.ProxyHandler({}))
        self.request_id = "setup"
        self.eval_context: dict[str, Any] = {}
        self.last_attempt_index: int | None = None

    def remaining(self) -> float:
        value = self.deadline - time.monotonic()
        if value <= 0:
            raise LiveWallTimeout("Frozen 900-second live wall budget exhausted")
        return value

    @contextmanager
    def scope(self, request_id: str, eval_context: dict[str, Any]):
        old = self.request_id, self.eval_context
        self.request_id, self.eval_context = request_id, copy.deepcopy(eval_context)
        try:
            yield
        finally:
            self.request_id, self.eval_context = old

    def post(self, path: str, payload: dict[str, Any], timeout: int | float,
             retries: int = 0) -> dict[str, Any]:
        if retries != 0:
            raise ValueError("Frozen transport prohibits retries")
        self.last_attempt_index = None
        allowed = min(float(timeout), self.remaining())
        if path == "/v1/chat/completions":
            kind, budget = "generation", self.generation
        elif path == "/v1/c2kv/extract":
            kind, budget = "extraction", self.extraction
        else:
            raise ValueError(f"Route is outside the frozen probe: {path}")
        attempt_index = budget.reserve()
        self.last_attempt_index = attempt_index
        handle = self.journal.start(
            kind, attempt_index, self.request_id, self.eval_context
        )
        started = time.monotonic()
        record = {
            "path": path, "kind": kind, "attempt_index": attempt_index,
            "request_id": self.request_id, "status": "started",
        }
        request = urlrequest.Request(
            f"{self.upstream}{path}", data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST",
        )
        try:
            try:
                with self.opener.open(request, timeout=allowed) as response:
                    data = json.loads(response.read().decode("utf-8"))
                if not isinstance(data, dict):
                    raise ValueError("Upstream response must be a JSON object")
            except BaseException as error:
                record.update(status="failed", error_type=type(error).__name__)
                detail = str(error)[:1000]
                if isinstance(error, HTTPError):
                    try:
                        detail = error.read().decode("utf-8", errors="replace")[:1000]
                    except OSError:
                        pass
                try:
                    self.journal.finish(handle, "failed")
                except BaseException as journal_error:
                    record["journal_finish_error_type"] = type(journal_error).__name__
                    raise
                raise LiveAttemptError(
                    f"Frozen {kind} transport failed: {type(error).__name__} {detail}",
                    attempt_index,
                ) from error
            try:
                usage = data.get("usage") if kind == "generation" else None
                self.journal.finish(handle, "completed", usage=usage)
            except BaseException as error:
                record.update(status="journal_finish_failed", error_type=type(error).__name__)
                raise
            record["status"] = "completed"
            if kind == "generation":
                record["usage"] = copy.deepcopy(usage)
            else:
                record["response"] = {key: copy.deepcopy(data.get(key)) for key in (
                    "key_hash", "gist_len", "original_seq_len", "success", "error"
                )}
            return data
        finally:
            record["wall_seconds"] = time.monotonic() - started
            _append_jsonl(self.transport_path, record)


def _verify_materialization(item: dict[str, Any], result: dict[str, Any]) -> None:
    expected = item["expected_response"]
    observed = {field: result.get(field) for field in (
        "key_hash", "gist_len", "original_seq_len"
    )}
    if result.get("success", True) is not True or observed != expected:
        raise ValueError("Live gist materialization differs from the frozen manifest")


def _verify_normalized(view: dict[str, Any], normalized: dict[str, Any]) -> None:
    expected = view["server_expectations"]
    usage = normalized.get("usage") or {}
    cost = normalized.get("cost") or {}
    observed = {
        "raw_prompt_tokens": usage.get("prompt_tokens"),
        "bytes_per_kv_token": cost.get("bytes_per_kv_token"),
        "c2kv_query_proj_effective": cost.get("c2kv_query_proj_effective"),
        "c2kv_query_proj_decode_verified": cost.get("c2kv_query_proj_decode_verified"),
        "c2kv_tools_dump": cost.get("c2kv_tools_dump"),
        "c2kv_gist_seen": cost.get("c2kv_gist_seen"),
    }
    if observed != expected:
        raise ValueError(f"Live backend geometry/provenance differs: {observed!r}")


def execute_prepared(
    artifact: dict[str, Any], prepared_path: Path, upstream: str, out: Path, *, opener=None
) -> Path:
    """Execute one new output directory from an already frozen artifact."""
    if out.exists():
        raise FileExistsError("Output exists; automatic rerun or resume is prohibited")
    started = time.monotonic()
    out.mkdir(parents=True)
    receipt_path = out / "receipt.json"
    cells_path = out / CELLS_NAME
    materializations_path = out / MATERIALIZATIONS_NAME
    attempts_path = out / ATTEMPTS_NAME
    config = artifact.get("design") or {}
    receipt = {
        "schema": RECEIPT_SCHEMA, "status": "validating_frozen_input",
        "design": copy.deepcopy(config), "source_bundle": copy.deepcopy(
            artifact.get("source_bundle")
        ),
        "prepared_path": str(prepared_path.resolve()),
        "prepared_file_sha256": _file_digest(prepared_path),
        "freeze_digest": artifact.get("freeze_digest"), "upstream": upstream,
        "cells_path": str(cells_path), "materializations_path": str(materializations_path),
        "attempt_journal_path": str(attempts_path),
        "counts": {"materializations_completed": 0, "cells_completed": 0},
    }
    _save(receipt_path, receipt)
    transport = None
    current_cell = None
    wall_timer = _AbsoluteWallTimer(started + 900)
    try:
        wall_timer.arm()
        _validate_prepared(artifact)
        deadline = started + config["maximum_wall_seconds"]
        transport = _LiveTransport(
            upstream, deadline, AttemptJournal(attempts_path), out / TRANSPORT_NAME,
            config["maximum_generation_attempts"], config["maximum_extraction_attempts"],
            opener=opener,
        )
        backend = get_backend("sglang", transport.post)
        receipt["status"] = "materializing_gists"
        _save(receipt_path, receipt)
        manifest = artifact["extraction_manifest"]
        for index, item in enumerate(manifest["items"], 1):
            request_id = f"materialize-{index:02d}-{uuid.uuid4().hex}"
            row = {
                "materialization_index": index, "request_id": request_id,
                "source_indices": copy.deepcopy(item["source_indices"]),
                "expected_response": copy.deepcopy(item["expected_response"]),
                "status": "started",
            }
            try:
                with transport.scope(request_id, {
                    "benchmark": "frozen-view-dev1", "run_id": out.name,
                    "task_id": config["task_id"], "attempt_id": "materialization",
                    "decision_id": str(index),
                }):
                    result = backend.extract(
                        item["content"], item["role"], item["ratio"],
                        tools=manifest["tools"],
                    )
                _verify_materialization(item, result)
                row.update(status="completed", response={key: copy.deepcopy(result.get(key))
                           for key in ("key_hash", "gist_len", "original_seq_len", "success")})
            except BaseException as error:
                row.update(status="failed", error_type=type(error).__name__, error=str(error))
                _append_jsonl(materializations_path, row)
                raise
            _append_jsonl(materializations_path, row)
            receipt["counts"]["materializations_completed"] = index
            _save(receipt_path, receipt)

        receipt["status"] = "verifying_frozen_wires"
        _save(receipt_path, receipt)
        _validate_prepared(artifact)
        transport.remaining()
        receipt["status"] = "running"
        _save(receipt_path, receipt)
        for cell in artifact["schedule"]:
            current_cell = cell
            transport.last_attempt_index = None
            view = artifact["views"][f"{cell['phase']}:{cell['view']}"]
            payload = copy.deepcopy(view["payload"])
            if _digest(payload) != view["wire_digest"]:
                raise ValueError("Cell payload differs before transport")
            request_id = f"{cell['cell_id']}-{uuid.uuid4().hex}"
            row = {
                "cell_id": cell["cell_id"], "block": cell["block"],
                "phase": cell["phase"], "view": cell["view"], "status": "started",
                "wire_digest": view["wire_digest"], "request_id": request_id,
                "generation_attempt_index": None, "response": None, "normalized": None,
            }
            response = None
            try:
                with transport.scope(request_id, {
                    "benchmark": "frozen-view-dev1", "run_id": out.name,
                    "task_id": config["task_id"], "attempt_id": cell["cell_id"],
                    "decision_id": cell["cell_id"],
                    "user_turn": view["context"].get("user_turn"),
                    "step": view["context"].get("step"),
                }):
                    response = transport.post(
                        "/v1/chat/completions", payload,
                        config["maximum_wall_seconds"], retries=0
                    )
                row["generation_attempt_index"] = transport.last_attempt_index
                normalized = backend.normalize_response(response)
                _verify_normalized(view, normalized)
                row.update(status="completed", response=response, normalized=normalized,
                           initial_native_action=copy.deepcopy(normalized.get("tool_calls")))
            except BaseException as error:
                row.update(
                    status="transport_or_backend_failure",
                    generation_attempt_index=(getattr(error, "attempt_index", None)
                                              or (transport.last_attempt_index if transport else None)),
                    response=response, error_type=type(error).__name__, error=str(error),
                )
                _append_jsonl(cells_path, row)
                raise
            _append_jsonl(cells_path, row)
            receipt["counts"]["cells_completed"] += 1
            receipt["current_cell"] = cell["cell_id"]
            _save(receipt_path, receipt)
        if (transport.generation.consumed != config["maximum_generation_attempts"]
                or receipt["counts"]["cells_completed"] != len(artifact["schedule"])
                or transport.extraction.consumed != manifest["count"]):
            raise ValueError("Successful run attempt counts differ from the frozen design")
        transport.remaining()
        receipt.update(status="completed", current_cell=None)
    except BaseException as error:
        receipt.update(status="failed", current_cell=(current_cell or {}).get("cell_id"),
                       error_type=type(error).__name__, error=str(error))
        raise
    finally:
        wall_timer.cancel()
        receipt["wall_seconds"] = time.monotonic() - started
        if transport is not None:
            receipt["counts"].update(
                generation_attempts=transport.generation.consumed,
                extraction_attempts=transport.extraction.consumed,
            )
        if attempts_path.exists():
            try:
                receipt["attempt_journal"] = summarize_attempt_journal(attempts_path)
            except BaseException as error:
                receipt["attempt_journal_error"] = {
                    "error_type": type(error).__name__, "error": str(error)
                }
        _save(receipt_path, receipt)
    return receipt_path


def _prepared_input(path: Path) -> Path:
    return path / PREPARED_NAME if path.is_dir() else path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--checkpoint")
    parser.add_argument("--prepared", type=Path)
    parser.add_argument("--upstream")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.prepare:
        if args.source_root is None or not args.checkpoint or args.prepared or args.upstream:
            parser.error("--prepare requires --source-root and --checkpoint only")
        target = prepare_probe(args.config, args.source_root, args.checkpoint, args.out)
        artifact = _read_object(target)
        print(json.dumps({
            "status": artifact["status"], "prepared_path": str(target),
            "freeze_digest": artifact["freeze_digest"],
            "views": len(artifact["views"]),
            "required_extractions": artifact["extraction_manifest"]["count"],
            "scheduled_generations": len(artifact["schedule"]),
        }))
        return 0
    if args.prepared is None or not args.upstream or args.source_root or args.checkpoint:
        parser.error("--execute requires --prepared and --upstream only")
    prepared_path = _prepared_input(args.prepared)
    artifact = _read_object(prepared_path)
    receipt_path = execute_prepared(artifact, prepared_path, args.upstream, args.out)
    receipt = _read_object(receipt_path)
    print(json.dumps({
        "status": receipt["status"], "receipt_path": str(receipt_path),
        "counts": receipt["counts"], "wall_seconds": receipt["wall_seconds"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
