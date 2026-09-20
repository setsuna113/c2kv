"""Frozen ToolSandbox scenario cohorts shared by paper and direct adapters."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


THREE_DISTRACTION_TOOLS_129 = "three_distraction_tools_129"
SUITE_PATH = Path(__file__).with_name("toolsandbox_suites") / f"{THREE_DISTRACTION_TOOLS_129}.json"


def load_named_suite(name: str) -> dict:
    if name != THREE_DISTRACTION_TOOLS_129:
        raise ValueError(f"Unknown ToolSandbox suite: {name!r}")
    data = json.loads(SUITE_PATH.read_text(encoding="utf-8"))
    ids = data.get("scenario_ids")
    if (data.get("schema") != "toolsandbox-frozen-cohort-v1"
            or data.get("suite") != name
            or data.get("source_checkout_head") != "165848b9a78cead7ca7fe7c89c688b58e6501219"
            or data.get("filter_suffix") != "_3_distraction_tools"
            or type(ids) is not list or len(ids) != 129
            or data.get("scenario_count") != len(ids)
            or any(type(item) is not str or not item.endswith("_3_distraction_tools")
                   for item in ids)
            or ids != sorted(set(ids))):
        raise ValueError("Frozen ToolSandbox cohort identity is invalid")
    digest = hashlib.sha256(("\n".join(ids) + "\n").encode("utf-8")).hexdigest()
    if digest != data.get("scenario_ids_sha256"):
        raise ValueError("Frozen ToolSandbox cohort checksum differs")
    return data


def selected_scenarios(suite: str | None, explicit=None, *, require_paper_suite=False):
    """Explicit IDs override paper selection; None denotes full/test CLI mode."""
    if suite not in (None, "", "full", THREE_DISTRACTION_TOOLS_129):
        raise ValueError(f"Unknown ToolSandbox suite: {suite!r}")
    if explicit:
        if (type(explicit) is not list or
                any(type(item) is not str or not item for item in explicit) or
                len(explicit) != len(set(explicit))):
            raise ValueError("toolsandbox_scenarios must contain unique scenario names")
        return list(explicit)
    if suite == THREE_DISTRACTION_TOOLS_129:
        return list(load_named_suite(suite)["scenario_ids"])
    if suite in (None, "", "full") and (not require_paper_suite or suite == "full"):
        return None
    raise ValueError(f"Unknown or missing ToolSandbox paper suite: {suite!r}")
