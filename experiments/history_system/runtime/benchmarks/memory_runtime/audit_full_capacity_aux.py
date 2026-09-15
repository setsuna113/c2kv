"""CPU audit of Full plus capacity-matched auxiliary evidence.

The audit replays only recorded capacity-protect source prefixes. It performs
no generation, extraction, scorer access, or network request.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import proxy
from arms import get_arm
from memory_runtime.adapter import RuntimeAdapter


EXPECTED_REQUESTS = 24
EXPECTED_BELOW_B = 19
EXPECTED_ACTIVATIONS = 5


def _nonnegative_int(value: Any, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def _recorded_rows(root: Path):
    paths = sorted((root / "capacity_protect" / "logs").glob("proxy_*.jsonl"))
    if not paths:
        raise ValueError("No capacity_protect proxy logs")
    for path in paths:
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            request = json.loads(line)
            if not isinstance(request, dict):
                raise ValueError(f"{path}:{number}: request row must be an object")
            yield path, number, request


def _single_wire(request: dict) -> list[dict]:
    views = request.get("forwarded_request_views")
    if (not isinstance(views, list) or len(views) != 1
            or not isinstance(views[0], dict)
            or not isinstance(views[0].get("messages"), list)):
        raise ValueError("expected exactly one captured forwarded message view")
    return views[0]["messages"]


def _packet(view: list[dict], index: Any, label: str) -> dict:
    if type(index) is not int or not 0 <= index < len(view):
        raise ValueError(f"{label} evidence_out_index is invalid")
    packet = view[index]
    if not isinstance(packet, dict) or not isinstance(packet.get("content"), str):
        raise ValueError(f"{label} evidence packet is invalid")
    return packet


def _inspect_row(
    root: Path,
    path: Path,
    number: int,
    request: dict,
    runtime: RuntimeAdapter,
    config: dict,
) -> dict:
    native = request["request_view"]
    source = native["messages"]
    tools = native.get("tools")
    context = request["eval_context"]
    old_meta = request["memory_runtime"]
    old_gate = old_meta["capacity_gate"]
    old_wire = _single_wire(request)
    if not isinstance(source, list) or any(not isinstance(item, dict) for item in source):
        raise ValueError("request_view.messages is not a message list")
    if not isinstance(context, dict):
        raise ValueError("eval_context is not an object")

    full, full_counts = proxy._assemble(source, get_arm("full"))
    replay_context = copy.deepcopy(context)
    replay_context["run_id"] = runtime.run_id
    full_plus_e, updated = runtime.apply(
        source, full, full_counts, replay_context, tools)
    meta = updated["memory_runtime"]
    gate = meta["auxiliary_gate"]

    old_activated = old_gate.get("compression_activated")
    activated = gate.get("auxiliary_activated")
    if type(old_activated) is not bool or type(activated) is not bool:
        raise ValueError("capacity activation metadata is not boolean")
    if activated != old_activated:
        raise ValueError("auxiliary activation differs from the recorded capacity gate")
    for key in (
        "full_raw_prompt_tokens",
        "full_raw_history_tokens",
        "full_history_bytes",
        "history_budget_bytes",
    ):
        if gate.get(key) != old_gate.get(key):
            raise ValueError(f"auxiliary gate differs from recorded capacity gate: {key}")

    bytes_per_token = _nonnegative_int(
        config.get("bytes_per_kv_token"), "bytes_per_kv_token")
    if bytes_per_token == 0:
        raise ValueError("bytes_per_kv_token must be positive")
    if (request.get("bytes_per_kv_token") != bytes_per_token
            or old_meta.get("bytes_per_kv_token") != bytes_per_token
            or old_meta.get("backend_bytes_per_kv_token") != bytes_per_token
            or meta.get("bytes_per_kv_token") != bytes_per_token):
        raise ValueError("recorded, backend, and replay KV byte geometry differ")

    old_server_tokens = _nonnegative_int(
        request.get("usage", {}).get("prompt_tokens"), "recorded prompt_tokens")
    old_wire_tokens = runtime._token_counter(
        [message for message in old_wire if not message.get("c2kv_key_hash")], tools)
    if (old_wire_tokens != old_server_tokens
            or old_meta.get("total_raw_prompt_tokens") != old_server_tokens
            or old_meta.get("backend_raw_prompt_tokens") != old_server_tokens
            or old_meta.get("raw_prompt_tokens_verified_by_backend") is not True
            or old_meta.get("byte_geometry_verified_by_backend") is not True):
        raise ValueError("raw token or KV geometry does not match the recorded server")

    full_tokens = runtime._token_counter(full, tools)
    full_plus_e_tokens = runtime._token_counter(full_plus_e, tools)
    evidence_tokens = full_plus_e_tokens - full_tokens
    if evidence_tokens < 0:
        raise ValueError("auxiliary insertion produced a negative token delta")
    full_bytes = full_tokens * bytes_per_token
    evidence_bytes = evidence_tokens * bytes_per_token
    full_plus_e_bytes = full_plus_e_tokens * bytes_per_token
    if (full_tokens != gate["full_raw_prompt_tokens"]
            or evidence_bytes != meta.get("evidence_bytes")
            or full_bytes + evidence_bytes != full_plus_e_bytes):
        raise ValueError("Full/E/Full+E token-byte accounting is inconsistent")
    if meta.get("budget_applies") is not False:
        raise ValueError("history budget must be a gate and must not crop Full")
    if (any(message.get("c2kv_key_hash") for message in full_plus_e)
            or updated.get("compressed_records") not in (None, [])
            or meta.get("retrieved_event_ids") != []
            or meta.get("retained_event_ids") != []):
        raise ValueError("full_capacity_aux retained gist or recovery-only state")
    if (meta.get("history_budget_bytes") != config["history_budget_bytes"]
            or meta.get("workspace_budget_bytes") != config["workspace_budget_bytes"]):
        raise ValueError("replay budgets differ from full_capacity_aux config")

    selected = meta.get("selected_event_ids")
    old_selected = old_meta.get("selected_event_ids")
    packet_text_exact = None
    packet_exact = None
    selection_exact = None
    if not activated:
        if (old_wire != full or full_plus_e != full
                or meta.get("evidence_out_index") is not None
                or meta.get("evidence_bytes") != 0
                or evidence_tokens != 0):
            raise ValueError("below-B replay is not exact Full with no auxiliary evidence")
        if meta.get("workspace_budget_applies") is not False:
            raise ValueError("workspace budget must be inactive below B")
    else:
        old_packet = _packet(old_wire, old_meta.get("evidence_out_index"), "recorded")
        packet_index = meta.get("evidence_out_index")
        packet = _packet(full_plus_e, packet_index, "replayed")
        packet_exact = packet == old_packet
        packet_text_exact = packet["content"] == old_packet["content"]
        selection_exact = selected == old_selected
        if not packet_exact or not packet_text_exact or not selection_exact:
            raise ValueError("auxiliary evidence differs from recorded capacity protection")
        if (len(full_plus_e) != len(full) + 1
                or full_plus_e[:packet_index] + full_plus_e[packet_index + 1:] != full):
            raise ValueError("activated auxiliary replay changed Full message order or content")
        if (meta.get("workspace_budget_applies") is not True
                or evidence_bytes > config["workspace_budget_bytes"]):
            raise ValueError("actual auxiliary insertion exceeds or bypasses W")
        if meta.get("auxiliary_selection_bytes") != old_meta.get("evidence_bytes"):
            raise ValueError("same-common auxiliary selection cost differs from capacity protection")
        if gate["full_history_bytes"] <= config["history_budget_bytes"]:
            raise ValueError("auxiliary evidence activated at or below B")

    if (activated is False
            and gate["full_history_bytes"] > config["history_budget_bytes"]):
        raise ValueError("auxiliary evidence stayed off above B")
    source_ref = {
        "path": path.resolve().relative_to(root.resolve()).as_posix(),
        "line": number,
    }
    return {
        "source": source_ref,
        "context": context,
        "auxiliary_activated": activated,
        "gate": {
            "full_raw_history_tokens": gate["full_raw_history_tokens"],
            "full_history_bytes": gate["full_history_bytes"],
            "history_budget_bytes": gate["history_budget_bytes"],
            "relation": "above" if activated else "at_or_below",
        },
        "selected_event_ids": selected if activated else [],
        "tokens": {
            "full": full_tokens,
            "evidence_insertion_delta": evidence_tokens,
            "full_plus_evidence": full_plus_e_tokens,
        },
        "bytes": {
            "full": full_bytes,
            "evidence_insertion_delta": evidence_bytes,
            "full_plus_evidence": full_plus_e_bytes,
            "auxiliary_selection_same_common": (
                meta.get("auxiliary_selection_bytes") if activated else 0),
        },
        "checks": {
            "gate_matches_recorded_capacity": True,
            "full_order_and_content_preserved": True,
            "history_budget_is_gate_only": True,
            "no_gist_or_recovery_retention": True,
            "actual_evidence_within_workspace_budget": True,
            "raw_token_and_kv_geometry_match_recorded_server": True,
            "selected_event_ids_and_order_match_recorded": selection_exact,
            "evidence_packet_exact": packet_exact,
            "evidence_text_exact": packet_text_exact,
        },
    }


def audit(root: Path, tokenizer: str) -> dict:
    root = root.resolve()
    config_path = HERE / "configs/full_capacity_aux.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    runtime = RuntimeAdapter.from_config(str(config_path), tokenizer)

    def forbidden(*args, **kwargs):
        raise AssertionError("CPU auxiliary audit cannot call generation or extraction")

    original_post, original_extract = proxy._post_json, proxy._extract
    proxy._post_json = forbidden
    proxy._extract = forbidden
    rows = []
    try:
        for path, number, request in _recorded_rows(root):
            try:
                rows.append(_inspect_row(
                    root, path, number, request, runtime, config))
            except (AssertionError, KeyError, TypeError, ValueError) as error:
                relative = path.resolve().relative_to(root).as_posix()
                raise ValueError(f"{relative}:{number}: {error}") from error
    finally:
        proxy._post_json, proxy._extract = original_post, original_extract

    below = [row for row in rows if not row["auxiliary_activated"]]
    activated = [row for row in rows if row["auxiliary_activated"]]
    if (len(rows), len(below), len(activated)) != (
            EXPECTED_REQUESTS, EXPECTED_BELOW_B, EXPECTED_ACTIVATIONS):
        raise ValueError(
            "expected 24 recorded capacity prefixes with 19 below B and 5 above B; "
            f"got {len(rows)}, {len(below)}, and {len(activated)}")

    byte_keys = (
        "full",
        "evidence_insertion_delta",
        "full_plus_evidence",
        "auxiliary_selection_same_common",
    )
    token_keys = ("full", "evidence_insertion_delta", "full_plus_evidence")
    result = {
        "schema": "a-runtime-full-capacity-aux-cpu-audit-v1",
        "status": "passed",
        "analysis": "recorded_capacity_prefix_cpu_replay",
        "additional_chat_requests": 0,
        "additional_extraction_requests": 0,
        "network_requests": 0,
        "scorer_records_read": 0,
        "history_budget_bytes": config["history_budget_bytes"],
        "workspace_budget_bytes": config["workspace_budget_bytes"],
        "bytes_per_kv_token": config["bytes_per_kv_token"],
        "request_count": len(rows),
        "below_B_exact_full_without_evidence_count": len(below),
        "above_B_capacity_matched_evidence_count": len(activated),
        "full_preserved_count": len(rows),
        "raw_token_and_kv_geometry_match_recorded_server_count": len(rows),
        "token_totals": {
            key: sum(row["tokens"][key] for row in rows) for key in token_keys
        },
        "byte_totals": {
            key: sum(row["bytes"][key] for row in rows) for key in byte_keys
        },
        "scope": (
            "CPU replay of the 24 recorded capacity-protect source prefixes; "
            "no generation, extraction, scorer input, or network access"
        ),
        "rows": rows,
    }
    source_bundle = HERE.parents[1] / "tmp/a_memory_runtime_20260907/source_bundle.json"
    if source_bundle.exists():
        result["source_bundle"] = json.loads(source_bundle.read_text(encoding="utf-8"))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.root, args.tokenizer)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items()
                      if key not in {"rows", "source_bundle"}}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
