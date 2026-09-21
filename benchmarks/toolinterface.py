"""Split a recognized tool into executable JSON and descriptive prose.

Only string-valued prose annotations at known schema positions are extracted.
Literal values and unknown extensions remain in the executable interface.
"""
from __future__ import annotations

import ast
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


_ANNOTATIONS = frozenset({"title", "description", "$comment"})
_SCHEMA_MAPS = frozenset({"properties", "patternProperties", "$defs", "definitions",
                          "dependentSchemas"})
_SCHEMA_VALUES = frozenset({"items", "additionalItems", "additionalProperties",
                            "unevaluatedProperties", "propertyNames", "not", "if", "then",
                            "else", "contains", "unevaluatedItems", "contentSchema"})
_SCHEMA_ARRAYS = frozenset({"allOf", "anyOf", "oneOf", "prefixItems"})
_TOOL_SCHEMA_KEYS = frozenset({"parameters", "input_schema", "schema", "output_schema"})


def compact_schema(value: Any) -> Any:
    """Walk only JSON Schema positions; never interpret literal data as schema."""
    if not isinstance(value, Mapping):
        return value
    out = {}
    for key, child in value.items():
        if key in _ANNOTATIONS and isinstance(child, str) and child.strip():
            continue
        if key in _SCHEMA_MAPS and isinstance(child, Mapping):
            out[key] = {name: compact_schema(schema) for name, schema in child.items()}
        elif key in _SCHEMA_VALUES:
            out[key] = compact_schema(child)
        elif key in _SCHEMA_ARRAYS and isinstance(child, list):
            out[key] = [compact_schema(schema) for schema in child]
        elif key == "dependencies" and isinstance(child, Mapping):
            out[key] = {name: compact_schema(dependency) if isinstance(dependency, Mapping)
                        else dependency for name, dependency in child.items()}
        else:
            out[key] = child
    return out


def _compact_tool_fields(tool: Mapping[str, Any]) -> dict[str, Any]:
    out = {}
    for key, value in tool.items():
        if key in _ANNOTATIONS and isinstance(value, str) and value.strip():
            continue
        out[key] = compact_schema(value) if key in _TOOL_SCHEMA_KEYS else value
    return out


def compact_tool(value: Any) -> dict[str, Any] | None:
    """Return None when the producer's document shape is not recognizable."""
    if not isinstance(value, Mapping):
        return None
    function = value.get("function")
    if isinstance(function, Mapping):
        if not isinstance(function.get("name"), str) or not function["name"]:
            return None
        return {key: _compact_tool_fields(function) if key == "function" else child
                for key, child in value.items()
                if not (key in _ANNOTATIONS and isinstance(child, str) and child.strip())}
    if isinstance(value.get("name"), str) and value["name"]:
        return _compact_tool_fields(value)
    # AppWorld uses api_name and a non-JSON-Schema parameter list. Preserve it
    # in full because its list entries have producer-specific execution meaning.
    return None


@dataclass(frozen=True)
class InterfaceCopy:
    text: str
    fallback: bool
    reason: str = "compact"


def _schema_prose(value: Any, path: tuple[Any, ...]) -> list[dict[str, Any]]:
    if not isinstance(value, Mapping):
        return []
    result = []
    for key, child in value.items():
        child_path = (*path, key)
        if key in _ANNOTATIONS and isinstance(child, str) and child.strip():
            result.append({"path": list(child_path), "text": child})
        elif key in _SCHEMA_MAPS and isinstance(child, Mapping):
            for name, schema in child.items():
                result.extend(_schema_prose(schema, (*child_path, name)))
        elif key in _SCHEMA_VALUES:
            result.extend(_schema_prose(child, child_path))
        elif key in _SCHEMA_ARRAYS and isinstance(child, list):
            for index, schema in enumerate(child):
                result.extend(_schema_prose(schema, (*child_path, index)))
        elif key == "dependencies" and isinstance(child, Mapping):
            for name, schema in child.items():
                if isinstance(schema, Mapping):
                    result.extend(_schema_prose(schema, (*child_path, name)))
    return result


def tool_prose(value: Any) -> list[dict[str, Any]]:
    """Return only prose removed by compact_tool, with source paths."""
    if compact_tool(value) is None:
        return []
    result = []
    for key, child in value.items():
        if key in _ANNOTATIONS and isinstance(child, str) and child.strip():
            result.append({"path": [key], "text": child})
        elif key == "function" and isinstance(child, Mapping):
            result.extend(_tool_field_prose(child, (key,)))
        elif "function" not in value and key in _TOOL_SCHEMA_KEYS:
            result.extend(_schema_prose(child, (key,)))
    return result


def _tool_field_prose(value: Mapping[str, Any], path: tuple[Any, ...]) -> list[dict[str, Any]]:
    result = []
    for key, child in value.items():
        if key in _ANNOTATIONS and isinstance(child, str) and child.strip():
            result.append({"path": [*path, key], "text": child})
        elif key in _TOOL_SCHEMA_KEYS:
            result.extend(_schema_prose(child, (*path, key)))
    return result


def description_document(value: Any, index: int) -> dict[str, Any] | None:
    """T0 document with mapping metadata and prose, without a second schema."""
    interface = compact_tool(value)
    if interface is None:
        return None
    annotations = []
    for annotation in tool_prose(value):
        parent = interface
        address = []
        for component in annotation["path"][:-1]:
            if isinstance(parent, Mapping):
                keys = list(parent)
                if component not in parent:
                    raise ValueError("prose parent missing from compact interface")
                address.append(keys.index(component))
            elif isinstance(parent, list):
                address.append(component)
            parent = parent[component]
        annotations.append({"address": address, "field": annotation["path"][-1],
                            "text": annotation["text"]})
    if not annotations:
        return None
    return {"type": "tool_description", "tool_index": index,
            "annotations": annotations}


def json_string_value_spans(text: str, paths: set[tuple[Any, ...]]) -> list[dict[str, Any]]:
    """Locate JSON string value tokens exactly, including original whitespace."""
    decoder = json.JSONDecoder()
    found: list[dict[str, Any]] = []

    def skip(position: int) -> int:
        while position < len(text) and text[position].isspace():
            position += 1
        return position

    def walk(position: int, path: tuple[Any, ...]) -> int:
        position = skip(position)
        if text[position] == "{":
            position = skip(position + 1)
            if text[position] == "}":
                return position + 1
            while True:
                key, end = decoder.raw_decode(text, position)
                if not isinstance(key, str):
                    raise ValueError("JSON object key is not a string")
                position = skip(end)
                if text[position] != ":":
                    raise ValueError("invalid JSON object")
                position = walk(position + 1, (*path, key))
                position = skip(position)
                if text[position] == "}":
                    return position + 1
                if text[position] != ",":
                    raise ValueError("invalid JSON object")
                position = skip(position + 1)
        if text[position] == "[":
            position = skip(position + 1)
            if text[position] == "]":
                return position + 1
            index = 0
            while True:
                position = walk(position, (*path, index))
                position = skip(position)
                if text[position] == "]":
                    return position + 1
                if text[position] != ",":
                    raise ValueError("invalid JSON array")
                index += 1
                position = skip(position + 1)
        value, end = decoder.raw_decode(text, position)
        if path in paths and isinstance(value, str):
            found.append({"path": list(path), "start": position, "end": end,
                          "text": text[position:end]})
        return end

    if skip(walk(0, ())) != len(text):
        raise ValueError("trailing JSON content")
    return found


def interface_copy(value: Any, *, source_text: str | None = None) -> InterfaceCopy:
    """Use exact source bytes for an opaque visible span; JSON for known tools."""
    candidate = value
    if source_text is not None:
        try:
            candidate = json.loads(source_text)
        except (TypeError, ValueError):
            try:
                candidate = ast.literal_eval(source_text)
            except (SyntaxError, ValueError, TypeError, MemoryError, RecursionError):
                return InterfaceCopy(source_text, True, "opaque_full")
    compact = compact_tool(candidate)
    if compact is None:
        return InterfaceCopy(source_text if source_text is not None else
                             json.dumps(value, ensure_ascii=False, separators=(",", ":"),
                                        allow_nan=False), True, "opaque_full")
    try:
        text = json.dumps(compact, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError):
        return InterfaceCopy(source_text if source_text is not None else
                             json.dumps(value, ensure_ascii=False, separators=(",", ":"),
                                        allow_nan=False), True, "opaque_full")
    return InterfaceCopy(text, False)
