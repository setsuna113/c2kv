"""Exact declarations for text-history budget task failures.

Only typed proxy/API codes are accepted.  Message prose, HTTP status alone,
and generic transport errors never establish a method-budget outcome.
"""
from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any


TEXT_HISTORY_BUDGET_FAILURE_CODES = frozenset({
    "acon_history_budget_exceeded",
    "hiagent_history_budget_exceeded",
    "hiagent_retrieval_budget_exceeded",
})

_TYPED_CODE = re.compile(
    r"[\"']code[\"']\s*:\s*[\"'](" +
    "|".join(sorted(TEXT_HISTORY_BUDGET_FAILURE_CODES)) +
    r")[\"']"
)


def proxy_text_budget_failure_code(row: Any) -> str | None:
    """Return an exact proxy declaration, rejecting contradictory metadata."""
    if not isinstance(row, Mapping):
        return None
    status = row.get("status")
    error_kind = row.get("error_kind")
    if (status in TEXT_HISTORY_BUDGET_FAILURE_CODES
            and error_kind in (None, status)):
        return str(status)
    return None


def typed_text_budget_failure_code(value: Any) -> str | None:
    """Extract one exact API ``error.code`` from structured data or its repr."""
    if isinstance(value, Mapping):
        code = value.get("code")
        if code in TEXT_HISTORY_BUDGET_FAILURE_CODES:
            return str(code)
        error = value.get("error")
        if isinstance(error, Mapping):
            code = error.get("code")
            if code in TEXT_HISTORY_BUDGET_FAILURE_CODES:
                return str(code)
        return None
    if not isinstance(value, str):
        return None
    match = _TYPED_CODE.search(value)
    return match.group(1) if match else None
