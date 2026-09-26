"""CPU contracts for source-verified known-support packaging."""
from __future__ import annotations

import copy
import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
RUNTIME = HERE / "runtime"
sys.path.insert(0, str(RUNTIME / "python"))
sys.path.insert(0, str(RUNTIME))
sys.path.insert(0, str(HERE))

from benchmarks.memory_runtime.recovery.evidence_units import (  # noqa: E402
    SpanProvenance,
    _sha256,
    _span_container_text,
    build_catalog,
)
from benchmarks.memory_runtime.tests.test_event_native_recovery import request  # noqa: E402
from benchmarks.memory_runtime.tests.test_gp_recovery import (  # noqa: E402
    UnitTokenizer,
    make_controller,
)
from recoverability_support import (  # noqa: E402
    SupportAnnotationError,
    commit_support,
    plan_support,
)


def controller(*, maximum=4, **updates):
    return make_controller(
        selection_protocol="evidence_sets_v1",
        U="tokens_1024",
        K=maximum,
        selector_max_units=maximum,
        B="source",
        P="quoted",
        **updates,
    )


def annotation(prepared, event_id, quote, *, reason="fixture source supports the fixture label"):
    unit = next(unit for unit in build_catalog(prepared._store, UnitTokenizer(), "tokens_1024")
                if unit.event_id == event_id)
    for original in unit.provenance:
        container = _span_container_text(prepared._store, original)
        start = container.find(quote, original.char_start, original.char_end)
        if start >= 0:
            span = SpanProvenance(original.event_id, original.source_index,
                original.container_path, original.field_path, start, start + len(quote),
                _sha256(quote))
            return {**span.to_dict(), "quote": quote, "support_reason": reason,
                    "reviewed": True}
    raise AssertionError(f"fixture quote {quote!r} is absent from {event_id}")


@pytest.mark.parametrize("mutation", ["foreign", "future", "edited_quote"])
def test_foreign_future_and_edited_source_are_rejected(mutation):
    recovery = controller()
    prepared = recovery.prepare(request(), ratio=4, max_new_tokens=32)
    item = annotation(prepared, "task-1:m2", "violet")
    if mutation == "foreign":
        item["event_id"] = "other-session:m2"
    elif mutation == "future":
        item["event_id"] = "task-1:m99"
        item["source_index"] = len(prepared._store.messages) + 1
    else:
        item["quote"] = "violet-edited"
    with pytest.raises(SupportAnnotationError):
        plan_support(recovery, prepared, [item])


def test_entire_archive_can_supply_support_outside_held_candidates_and_commit_it():
    recovery = controller()
    prepared = recovery.prepare(request(), ratio=4, max_new_tokens=32)
    archive = build_catalog(prepared._store, UnitTokenizer(), "tokens_1024")
    held = next(unit for unit in archive if unit.event_id == "task-1:m4")
    prepared._t02_selection = ([{"unit_id": held.unit_id, "unit": held}], [()], {})
    rationale = "fixture-only rationale must never enter actor evidence"

    selected, receipt = plan_support(
        recovery, prepared, [annotation(prepared, "task-1:m2", "violet", reason=rationale)]
    )

    assert receipt["status"] == "admitted"
    assert [unit.event_id for unit in selected] == ["task-1:m2"]
    assert receipt["held_candidates"]["present"] is True
    assert receipt["held_candidates"]["all_selected_present"] is False
    assert rationale not in json.dumps(receipt)
    assert all(rationale not in unit.text for unit in selected)
    result = commit_support(recovery, prepared, selected)
    assert result["regenerate"] is True
    assert result["decision"]["appended_unit_ids"] == receipt["selected_unit_ids"]
    assert result["decision"]["selection"]["selector_or_model_called"] is False
    assert result["decision"]["selection"]["reason"] == "reviewed_known_support"
    with pytest.raises(ValueError, match="already been committed"):
        commit_support(recovery, prepared, selected)


def test_raw_visible_support_is_not_duplicated_and_cancelled_source_is_distinct():
    recovery = controller()
    prepared = recovery.prepare(request(), ratio=4, max_new_tokens=32)
    selected, receipt = plan_support(
        recovery, prepared, [annotation(prepared, "task-1:m6", "other-b")]
    )
    assert selected == []
    assert receipt["status"] == "already_exact_visible"
    assert receipt["candidate_ids"] == []

    other = controller()
    cancelled = other.prepare(request("d2"), ratio=4, max_new_tokens=32)
    cancelled.metadata["revision_cancelled_event_ids"] = ["task-1:m2"]
    selected, receipt = plan_support(
        other, cancelled, [annotation(cancelled, "task-1:m2", "violet")]
    )
    assert selected == []
    assert receipt["status"] == "source_not_eligible"


def test_frozen_max_units_reports_static_nonrepresentability():
    recovery = controller(maximum=1)
    prepared = recovery.prepare(request(), ratio=4, max_new_tokens=32)
    selected, receipt = plan_support(recovery, prepared, [
        annotation(prepared, "task-1:m2", "violet"),
        annotation(prepared, "task-1:m4", "other-a"),
    ])
    assert selected == []
    assert receipt["status"] == "support_not_representable_in_static_units"
    assert receipt["catalog"]["max_units"] == 1
    assert receipt["catalog"]["search_complete"] is True


def test_combined_real_b0_charges_the_quoted_wrapper_and_rejects_the_pack():
    recovery = controller(maximum=2)
    prepared = recovery.prepare(request(), ratio=4, max_new_tokens=32)
    annotations = [
        annotation(prepared, "task-1:m2", "violet"),
        annotation(prepared, "task-1:m4", "other-a"),
    ]
    archive = {unit.event_id: unit for unit in
               build_catalog(prepared._store, UnitTokenizer(), "tokens_1024")}
    units = [archive["task-1:m2"], archive["task-1:m4"]]
    budget = 1700
    recovery._packer.policy_config = replace(
        recovery._packer.policy_config,
        history_budget_bytes=budget,
        workspace_budget_bytes=budget,
    )
    assert prepared.metadata["actual_history_bytes"] + sum(unit.token_count for unit in units) < budget
    assert all(recovery._append_measure(prepared, [unit])[0] is not None for unit in units)
    assert recovery._append_measure(prepared, units)[0] is None

    selected, receipt = plan_support(recovery, prepared, annotations)

    assert {unit.event_id for unit in selected} == {"task-1:m2", "task-1:m4"}
    assert receipt["status"] == "selected_support_set_not_admitted_under_b0"
    assert receipt["admission"]["status"] == "abstained"
    assert receipt["admission"]["all_wrappers_charged"] is True
    assert receipt["admission"]["b0_rechecked_after_all_changes"] is True


def test_configured_static_fallback_is_used_only_after_parent_fails_real_b0():
    payload = request()
    payload["messages"][3]["content"] = json.dumps({"value": "violet " * 300})
    recovery = controller(fallback_unit="tokens_256")
    prepared = recovery.prepare(payload, ratio=4, max_new_tokens=32)
    recovery._packer.policy_config = replace(
        recovery._packer.policy_config,
        history_budget_bytes=2500,
        workspace_budget_bytes=2500,
    )

    selected, receipt = plan_support(
        recovery, prepared, [annotation(prepared, "task-1:m2", "violet")]
    )

    assert receipt["status"] == "admitted"
    assert [unit.source_type for unit in selected] == ["tokens_256"]
    assert receipt["catalog"]["fallback_parent_ids"]
    assert receipt["admission"]["all_wrappers_charged"] is True
