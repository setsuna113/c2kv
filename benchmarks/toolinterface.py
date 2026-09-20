"""Extract a compact executable interface without changing tool definitions.

Only JSON Schema annotation keywords are discarded. Unknown schema keywords,
extension values, literal examples in enum/default/const, and property names
are kept verbatim. Opaque documents fall back to their full source text.
"""
from __future__ import annotations

import ast
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


_ANNOTATIONS = frozenset({"title", "description", "examples", "$comment"})
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
        if key in _ANNOTATIONS:
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
        if key in _ANNOTATIONS:
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
                for key, child in value.items() if key not in _ANNOTATIONS}
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
