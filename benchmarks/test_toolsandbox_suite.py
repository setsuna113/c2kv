"""The named ToolSandbox cohort is a frozen official resolver result."""

import json

import pytest

from benchmarks.toolsandbox_suite import (
    THREE_DISTRACTION_TOOLS_129, SUITE_PATH, load_named_suite, selected_scenarios,
)


def test_frozen_cohort_has_exact_official_identity():
    cohort = load_named_suite(THREE_DISTRACTION_TOOLS_129)
    assert cohort["resolved_count"] == 1032
    assert cohort["scenario_count"] == 129
    assert len(cohort["scenario_ids"]) == 129
    assert "wifi_off_3_distraction_tools" in cohort["scenario_ids"]
    assert all(name.endswith("_3_distraction_tools") for name in cohort["scenario_ids"])
    assert selected_scenarios(THREE_DISTRACTION_TOOLS_129) == cohort["scenario_ids"]


def test_frozen_cohort_rejects_corrupted_ids(tmp_path, monkeypatch):
    import benchmarks.toolsandbox_suite as suites

    cohort = json.loads(SUITE_PATH.read_text(encoding="utf-8"))
    cohort["scenario_ids"][-1] = "zzzz_3_distraction_tools"
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(cohort), encoding="utf-8")
    monkeypatch.setattr(suites, "SUITE_PATH", path)
    with pytest.raises(ValueError, match="checksum"):
        suites.load_named_suite(THREE_DISTRACTION_TOOLS_129)


def test_explicit_and_full_paper_modes_remain_available():
    assert selected_scenarios("full", require_paper_suite=True) is None
    assert selected_scenarios(THREE_DISTRACTION_TOOLS_129, ["get_wifi"],
                              require_paper_suite=True) == ["get_wifi"]
    with pytest.raises(ValueError, match="unique"):
        selected_scenarios("full", ["get_wifi", "get_wifi"], require_paper_suite=True)
