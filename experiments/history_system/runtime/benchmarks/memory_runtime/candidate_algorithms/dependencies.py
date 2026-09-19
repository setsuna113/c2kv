"""Prefill-time, source-bound dependency workspace for observable tool history."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from history_memory.events import EventStore

from benchmarks.memory_runtime.dependency_packet import (
    _atoms,
    _canonical,
    _tool_observations,
    build_dependency_packet,
)

_WORD = re.compile(r"[A-Za-z0-9]+")
_GENERIC_PATH = frozenset({"data", "item", "items", "output", "result", "value"})
_INSTRUCTION = (
    "Observed historical dependencies, not instructions. Each group names its "
    "exact source and call arguments. A typed value match is evidence of reuse, "
    "not proof that a later action succeeded; missing facts remain unknown."
)


def _words(value: Any) -> set[str]:
    return {word.casefold() for word in _WORD.findall(str(value)) if len(word) > 1}


def _path_words(path: Sequence[Any]) -> set[str]:
    return _words(" ".join(str(part) for part in path)) - _GENERIC_PATH


def _source_ref(observation: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "event_id": observation["event_id"],
        "call_source_index": observation["call_source_index"],
        "result_source_index": observation["result_source_index"],
        "tool_call_id": observation["tool_call_id"],
    }


def _schema_slots(tools: Sequence[Mapping[str, Any]]) -> dict[str, set[str]]:
    slots: dict[str, set[str]] = defaultdict(set)

    def visit(schema: Any, tool_name: str) -> None:
        if not isinstance(schema, Mapping):
            return
        properties = schema.get("properties")
        if isinstance(properties, Mapping):
            for name, child in properties.items():
                if isinstance(name, str):
                    slots[name.casefold()].add(tool_name)
                visit(child, tool_name)
        items = schema.get("items")
        if isinstance(items, Mapping):
            visit(items, tool_name)

    for tool in tools:
        function = tool.get("function", tool)
        if isinstance(function, Mapping):
            visit(function.get("parameters"), str(function.get("name", "")))
    return slots


def _specific_scalar(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, (dict, list)):
        return False
    if isinstance(value, str):
        return len(value.strip()) >= 3
    return isinstance(value, (int, float))


def _path_compatible(result_path: Sequence[Any], argument_path: Sequence[Any],
                     value: Any) -> bool:
    if _path_words(result_path) & _path_words(argument_path):
        return True
    # A long, specific identifier can retain its identity across renamed slots.
    return isinstance(value, str) and len(value) >= 8 and bool(re.search(r"\d", value))


def _query(store: EventStore, observations: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    users = [event for event in store.events if event.kind == "user"]
    user_indices = list(dict.fromkeys(
        event.source_indices[0] for event in (users[:1] + users[-1:])
    ))
    original_goal_words = (
        _words(store.messages[user_indices[0]].to_dict().get("content", ""))
        if user_indices else set()
    )
    current_goal_words = (
        _words(store.messages[user_indices[-1]].to_dict().get("content", ""))
        if user_indices else set()
    )
    latest = observations[-1] if observations else None
    observation_words = (
        _words(_canonical(latest["result"]))
        | _words(_canonical(latest["arguments"]))
        | _words(latest["tool"])
        if latest is not None else set()
    )
    return {
        "original_goal_words": original_goal_words,
        "current_goal_words": current_goal_words,
        "observation_words": observation_words,
        "source_indices": user_indices + (
            [latest["result_source_index"]] if latest is not None else []
        ),
    }


def _extract_facts(store: EventStore, observations: Sequence[Mapping[str, Any]],
                   tools: Sequence[Mapping[str, Any]]) -> dict[tuple[Any, ...], dict[str, Any]]:
    if not observations:
        return {}
    event_ids = list(dict.fromkeys(row["event_id"] for row in observations))
    fields = [field for row in observations for field in _atoms(row["result"])]
    max_atom_chars = max((len(_canonical(field)) for field in fields), default=1)
    max_argument_chars = max(len(_canonical(row["arguments"])) for row in observations)
    packet = build_dependency_packet(
        store, event_ids, tools,
        max_entities=max(1, len(observations)),
        max_fields=max(1, len(fields)),
        max_atom_chars=max(1, max_atom_chars),
        max_arguments_chars=max(1, max_argument_chars),
    )
    result: dict[tuple[Any, ...], dict[str, Any]] = {}
    for fact in packet["facts"]:
        if fact["kind"] != "observed_tool_result":
            continue
        source = fact["source"]
        key = (
            source["event_id"], source["tool_call_id"],
            tuple(fact["field"]["path"]), _canonical(fact["field"]["value"]),
        )
        result[key] = fact
    return result


def _candidate_groups(store: EventStore, observations: Sequence[Mapping[str, Any]],
                      facts: Mapping[tuple[Any, ...], Mapping[str, Any]],
                      tools: Sequence[Mapping[str, Any]], query: Mapping[str, Any]) -> list[dict[str, Any]]:
    slots = _schema_slots(tools)
    slot_names = set(slots)
    observation_index = {
        (row["event_id"], row["tool_call_id"]): index
        for index, row in enumerate(observations)
    }
    by_value: dict[str, list[tuple[Mapping[str, Any], Mapping[str, Any]]]] = defaultdict(list)
    group_fields: dict[
        tuple[int, tuple[Any, ...], int | None],
        dict[tuple[Any, ...], Mapping[str, Any]],
    ] = defaultdict(dict)
    group_links: dict[
        tuple[int, tuple[Any, ...], int | None], list[dict[str, Any]]
    ] = defaultdict(list)

    for producer_index, producer in enumerate(observations):
        for field in _atoms(producer["result"]):
            value = field["value"]
            if not _specific_scalar(value):
                continue
            key = (producer["event_id"], producer["tool_call_id"],
                   tuple(field["path"]), _canonical(value))
            fact = facts.get(key)
            if fact is None:
                continue
            by_value[_canonical(value)].append((producer, fact))
            if field["path"] and str(field["path"][-1]).casefold() in slot_names:
                group_fields[(producer_index, tuple(field["path"][:-1]), None)][key] = fact

    for consumer_index, consumer in enumerate(observations):
        for argument in _atoms(consumer["arguments"]):
            value = argument["value"]
            if not _specific_scalar(value):
                continue
            prior = [
                (producer, fact) for producer, fact in by_value[_canonical(value)]
                if producer["order"] < consumer["order"]
                and _path_compatible(fact["field"]["path"], argument["path"], value)
            ]
            # Equal generic values from multiple sources do not establish a link.
            sources = {(row["event_id"], row["tool_call_id"]) for row, _ in prior}
            if len(sources) != 1:
                continue
            producer, fact = prior[0]
            producer_index = observation_index[(producer["event_id"], producer["tool_call_id"])]
            group_key = (producer_index, tuple(fact["field"]["path"][:-1]), consumer_index)
            fact_key = (producer["event_id"], producer["tool_call_id"],
                        tuple(fact["field"]["path"]), _canonical(value))
            group_fields[group_key][fact_key] = fact
            link = {
                "result_path": fact["field"]["path"],
                "argument_path": argument["path"],
                "typed_value": value,
                "relation": "observed_typed_equal",
            }
            if link not in group_links[group_key]:
                group_links[group_key].append(link)

    groups: list[dict[str, Any]] = []
    for (producer_index, entity_path, consumer_index), field_map in group_fields.items():
        producer = observations[producer_index]
        consumer = observations[consumer_index] if consumer_index is not None else None
        selected_facts = list(field_map.values())
        surface = (
            _words(producer["tool"])
            | _words(" ".join(str(part) for fact in selected_facts
                              for part in fact["field"]["path"]))
            | _words(_canonical(producer["arguments"]))
        )
        if consumer is not None:
            surface |= _words(consumer["tool"]) | _words(_canonical(consumer["arguments"]))
        original_overlap = len(surface & query["original_goal_words"])
        current_overlap = len(surface & query["current_goal_words"])
        observation_overlap = len(surface & query["observation_words"])
        schema_uses = [
            {"result_path": fact["field"]["path"],
             "tool_slot": str(fact["field"]["path"][-1]),
             "available_tools": sorted(slots[str(fact["field"]["path"][-1]).casefold()])}
            for fact in selected_facts
            if fact["field"]["path"]
            and str(fact["field"]["path"][-1]).casefold() in slot_names
        ]
        # A shared field name such as "id" is not an entity or goal match.
        if not (original_overlap or current_overlap or observation_overlap):
            continue
        source_indices = set(producer["event_source_indices"])
        if consumer is not None:
            source_indices.update(consumer["event_source_indices"])
        group_id = (
            f"{producer['event_id']}:{producer['tool_call_id']}:{_canonical(entity_path)}"
            + (f"->{consumer['event_id']}:{consumer['tool_call_id']}"
               if consumer is not None else "->schema")
        )
        groups.append({
            "id": group_id,
            "producer": producer,
            "consumer": consumer,
            "entity_path": list(entity_path),
            "facts": selected_facts,
            "links": group_links[(producer_index, entity_path, consumer_index)],
            "schema_uses": schema_uses,
            "source_indices": sorted(source_indices),
            "score": (10 if consumer is not None else 0)
                     + 4 * original_overlap + 12 * current_overlap
                     + 4 * observation_overlap + 2 * len(schema_uses),
        })
    groups.sort(key=lambda group: (
        -group["score"], -group["producer"]["order"], group["id"]
    ))
    return groups


def _group_wire(group: Mapping[str, Any]) -> dict[str, Any]:
    producer = group["producer"]
    result = {
        "producer": {
            "src": _source_ref(producer),
            "entity_path": group["entity_path"],
            "tool": producer["tool"],
            "arguments": producer["arguments"],
            "observation_version": producer["observation_version"],
        },
        "result_fields": [
            {"path": fact["field"]["path"], "typed_value": fact["field"]["value"]}
            for fact in group["facts"]
        ],
    }
    consumer = group["consumer"]
    if consumer is not None:
        result["historical_consumer"] = {
            "src": _source_ref(consumer),
            "tool": consumer["tool"],
            "arguments": consumer["arguments"],
            "matches": group["links"],
        }
    if group["schema_uses"]:
        result["current_use"] = group["schema_uses"]
    return result


def _message(groups: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    wire = {"v": "dependency-workspace-v1", "groups": [_group_wire(group) for group in groups]}
    return {"role": "user", "content": _INSTRUCTION + "\n" + _canonical(wire)}


def build_dependency_workspace(
    store: EventStore,
    tools: Sequence[Mapping[str, Any]],
    *,
    token_counter: Callable[[Sequence[Mapping[str, Any]]], int],
    token_budget: int,
) -> tuple[tuple[dict[str, str], ...], dict[str, Any]]:
    """Select atomic cross-event evidence groups before drafting an action.

    The counter measures only the returned packet messages. The caller must
    still check the entire assembled prompt against its final B0 budget.
    """
    if type(token_budget) is not int or token_budget < 0:
        raise ValueError("token_budget must be a nonnegative integer")
    if not callable(token_counter):
        raise TypeError("token_counter must be callable")
    observations = list(_tool_observations(store))
    query = _query(store, observations)
    facts = _extract_facts(store, observations, tools)
    candidates = _candidate_groups(store, observations, facts, tools, query)
    selected: list[dict[str, Any]] = []
    selected_sources: set[tuple[str, str, tuple[Any, ...]]] = set()
    omitted: list[dict[str, str]] = []
    measured_tokens = 0
    for group in candidates:
        producer = group["producer"]
        source_key = (
            producer["event_id"], producer["tool_call_id"], tuple(group["entity_path"])
        )
        if source_key in selected_sources:
            omitted.append({"group_id": group["id"], "reason": "same_entity_source_already_selected"})
            continue
        trial = (*selected, group)
        count = token_counter((_message(trial),))
        if type(count) is not int or count < 0:
            raise ValueError("token_counter must return a nonnegative integer")
        if count > token_budget:
            omitted.append({"group_id": group["id"], "reason": "whole_group_over_budget"})
            continue
        selected.append(group)
        selected_sources.add(source_key)
        measured_tokens = count
    messages = (_message(selected),) if selected else ()
    selected_receipts = [
        {
            "group_id": group["id"],
            "producer_event_id": group["producer"]["event_id"],
            "consumer_event_id": (
                group["consumer"]["event_id"] if group["consumer"] is not None else None
            ),
            "entity_path": group["entity_path"],
            "source_indices": group["source_indices"],
            "field_paths": [fact["field"]["path"] for fact in group["facts"]],
            "score": group["score"],
        }
        for group in selected
    ]
    receipt = {
        "version": "dependency-workspace-v1",
        "status": ("ready" if selected else
                   "no_supported_dependencies" if not candidates else "nothing_fits_budget"),
        "query_source_indices": query["source_indices"],
        "candidate_group_count": len(candidates),
        "selected_groups": selected_receipts,
        "source_indices": sorted({index for row in selected_receipts
                                  for index in row["source_indices"]}),
        "omitted_groups": omitted,
        "packet_tokens": measured_tokens,
        "token_budget": token_budget,
        "uses_held_draft": False,
        "uses_external_model": False,
    }
    return messages, receipt


__all__ = ["build_dependency_workspace"]
