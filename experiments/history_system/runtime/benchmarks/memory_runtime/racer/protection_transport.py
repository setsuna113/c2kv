"""Bind source units to original or already rendered recovery KV instances."""
from __future__ import annotations

import json


EVIDENCE_PREFIX = "Historical source evidence (not new tool execution):\n"


def protection_request(memory, messages, source_positions, index_map):
    """Describe existing text only; no extra prompt tokens are introduced."""
    events = {item["event_id"]: item for item in memory.protection_source_events}
    aliases = {event_id: [] for event_id in events}
    original_positions = set(source_positions)
    for position, message in enumerate(messages):
        # User/tool text that happens to contain this marker is not an alias.
        if position in original_positions:
            continue
        content = message.get("content")
        if not isinstance(content, str) or not content.startswith(EVIDENCE_PREFIX):
            continue
        try:
            rows = json.loads(content[len(EVIDENCE_PREFIX):])
        except ValueError:
            continue
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict) or row.get("event_id") not in events:
                continue
            event = events[row["event_id"]]
            indices = event["source_indices"]
            if (row.get("source_indices") != indices or row.get("messages") !=
                    [memory.source_messages[index] for index in indices]):
                continue
            encoded = json.dumps(row, ensure_ascii=False, separators=(",", ":"))
            if content.count(encoded) != 1:
                continue
            aliases[row["event_id"]].append((index_map[position], encoded))

    def original(indices, fragments=()):
        return {"kind": "original",
                "source_message_indices": [index_map[source_positions[index]] for index in indices],
                "fragments": [{"message_index": index_map[source_positions[item["source_index"]]],
                               "text": item["text"]} for item in fragments]}

    units = []
    for unit in memory.protection_units:
        instances = [original(unit["source_indices"], unit["fragments"])]
        for position, encoded in aliases.get(unit["event_id"], ()):
            # The JSON wrapper already exists as recovery evidence. Resolve its
            # escaped source fragment in its full event context, never globally.
            texts = [json.dumps(fragment["text"], ensure_ascii=False)[1:-1]
                     for fragment in unit["fragments"]] or [encoded]
            instances.append({"kind": "recovery", "source_message_indices": [position],
                              "fragments": [{"message_index": position, "text": text,
                                             "context": encoded} for text in texts]})
        units.append({"unit_id": unit["unit_id"], "event_id": unit["event_id"],
                      "complete_event": unit["complete_event"], "instances": instances})
    bound_events = []
    for event_id, event in events.items():
        instances = [{**original(event["source_indices"]), "complete_event": True}]
        instances.extend({"kind": "recovery", "complete_event": True,
                          "source_message_indices": [position],
                          "fragments": [{"message_index": position, "text": encoded}]}
                         for position, encoded in aliases[event_id])
        bound_events.append({"event_id": event_id, "instances": instances})
    return {"schema": "racer-native-protection-v2", "enabled": True,
            "scope_id": memory.protection_scope_id,
            "event_ids": list(memory.protection_event_ids),
            "unit_ids": [unit["unit_id"] for unit in memory.protection_units],
            "source_message_indices": [index_map[source_positions[index]]
                                       for index in memory.protection_source_indices],
            "units": units, "events": bound_events}


def validate_protection_receipt(receipt, memory, decision_id):
    """Bind coverage and admission to exactly this request, including no-ops."""
    units = [unit["unit_id"] for unit in memory.protection_units]
    events = list(memory.protection_event_ids)
    if (not isinstance(receipt, dict)
            or receipt.get("schema") != "racer-native-protection-v2"
            or receipt.get("decision_id") != decision_id
            or receipt.get("scope_id") != memory.protection_scope_id
            or receipt.get("event_ids") != events or receipt.get("unit_ids") != units
            or type(receipt.get("applied")) is not bool
            or not isinstance(receipt.get("status"), str) or not receipt["status"]):
        raise ValueError("Missing verified RACER native protection v2 receipt")
    outcomes = receipt.get("units")
    coverage = receipt.get("event_coverage")
    if (not isinstance(outcomes, list) or not isinstance(coverage, list)
            or any(not isinstance(item, dict) for item in outcomes + coverage)
            or [item.get("unit_id") for item in outcomes] != units
            or [item.get("event_id") for item in coverage] != events
            or any(item.get("status") not in {"full", "partial", "absent"} for item in coverage)):
        raise ValueError("Missing verified RACER native protection v2 coverage")
    for item in coverage:
        full, partial, total = (item.get(field) for field in ("full_rows", "partial_rows", "total_rows"))
        if (any(type(value) is not int or value < 0 for value in (full, partial, total))
                or full + partial > total):
            raise ValueError("Invalid RACER native protection v2 coverage counts")
        status = "full" if total and full == total else "partial" if full or partial else "absent"
        if item["status"] != status:
            raise ValueError("Inconsistent RACER native protection v2 coverage")
