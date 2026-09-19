"""Distinguish terminal benchmark failures from incomplete BFCL execution.

The original result and traceback must reach the official scorer unchanged.
A transport status alone never establishes a terminal model/method failure.
"""
from __future__ import annotations

import json
import re
from collections.abc import Mapping


def completion_kind(row: Mapping) -> str:
    if "result" not in row:
        return "incomplete"
    failure = row.get("traceback")
    if failure is None:
        return "model_output"
    text = failure if isinstance(failure, str) else json.dumps(failure, ensure_ascii=False)
    # Only the server's explicit typed method failure is terminal. Generic
    # runner failures and transport errors remain incomplete.
    if re.search(r"[\"']code[\"']\s*:\s*[\"']c2kv_capacity_infeasible[\"']", text):
        return "capacity_infeasible"
    # SGLang's explicit context admission error, also recognized by prefix replay.
    # BFCL embeds the OpenAI exception repr, which can escape the apostrophe.
    if re.search(r"The input \(\d+ tokens\) is longer than the model\\*'s context length \(\d+ tokens\)", text):
        return "context_overflow"
    # These are actor-selected invalid retrievals, not backend availability errors.
    if any(marker in text for marker in (
        "HiAgent requested nonexistent completed subgoals",
        "HiAgent requested an already revealed trajectory without advancing",
    )):
        return "hiagent_invalid_retrieval"
    return "incomplete"


def bfcl_row_is_terminal(row: Mapping) -> bool:
    return completion_kind(row) != "incomplete"


def terminal_failure_kind(row: Mapping) -> str | None:
    kind = completion_kind(row)
    return kind if kind not in {"incomplete", "model_output"} else None
