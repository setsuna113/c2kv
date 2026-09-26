"""CPU contracts for round-two evidence selectors and hybrid fusion."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "benchmarks"), str(ROOT / "python")]

from memory_runtime.recovery.selection import (  # noqa: E402
    prepare_selection_dependencies,
    select_candidates,
    train_candidate_scorer,
)


@dataclass(frozen=True)
class Unit:
    unit_id: str
    text: str
    source_indices: tuple[int, ...]
    source_type: str = "event"
    event_id: str | None = None
    association_id: str | None = None
    metadata: dict = field(default_factory=dict)


class Backend:
    def __init__(self, *, reply='{"selected_ids":[]}', vectors=None):
        self.reply = reply
        self.vectors = vectors
        self.calls = []

    def chat(self, *, messages, purpose, config):
        self.calls.append(("chat", purpose, list(messages)))
        return self.reply

    def embed(self, *, texts, purpose, config):
        self.calls.append(("embed", purpose, list(texts)))
        return self.vectors


def config(**updates):
    result = {
        "Q": "lexical",
        "K": 1,
        "D": "candidate_rule",
        "candidate_limit": 8,
        "selection_threshold": 0.0,
        "backend": None,
        "candidate_scorer": None,
    }
    result.update(updates)
    return result


def test_explicit_fixed_selector_does_not_require_legacy_llm_count_backend():
    gp = config(K="llm", selector="fixed", selector_max_units=1)
    assert prepare_selection_dependencies(gp) is None
    units = [Unit("useful", "invoice ZX", (0,)), Unit("unrelated", "other text", (1,))]
    chosen, receipt = select_candidates(units, goal="invoice ZX", draft_text="",
                                        draft_tool_calls=[], config=gp)
    assert [unit.unit_id for unit in chosen] == ["useful"]
    assert receipt["backend"]["required_capabilities"] == []


def select(catalog, gp, backend=None, *, goal="find invoice ZX"):
    return select_candidates(
        catalog,
        goal=goal,
        draft_text="lookup invoice",
        draft_tool_calls=[],
        config=gp,
        backends=backend,
    )


def training_rows(label_kind, labels):
    candidates = ["invoice ZX result", "unrelated weather"]
    return [
        {
            "goal": "find invoice ZX",
            "draft_text": "lookup invoice",
            "candidate": candidate,
            "label": label,
            "label_kind": label_kind,
            "provenance": {"task_id": f"train-{index}"},
        }
        for index, (candidate, label) in enumerate(zip(candidates, labels))
    ]


def test_llm_selector_is_independent_of_detector_and_enforces_one_id():
    units = [
        Unit("top", "invoice ZX result", (1,), event_id="e1"),
        Unit("other", "weather", (2,), event_id="e2"),
    ]
    backend = Backend(reply='{"selected_ids":["top"]}')
    gp = config(
        selector="llm",
        selector_min_units=1,
        selector_max_units=1,
        selector_catalog="units",
    )

    assert prepare_selection_dependencies(gp, backend) is backend
    selected, receipt = select(units, gp, backend)

    assert selected == [units[0]]
    assert receipt["controller_mode"] == "candidate_rule"
    assert receipt["selector_mode"] == "llm"
    assert receipt["selector_fallback"] == {"applied": False, "reason": None}
    assert backend.calls[0][1] == "recovery_evidence_selection"


@pytest.mark.parametrize(
    "reply",
    [
        '{"selected_ids":["not-in-catalog"]}',
        '{"selected_ids":[]}',
        '{"selected_ids":["top","other"]}',
        "not json",
    ],
)
def test_invalid_llm_output_uses_frozen_lexical_top1_fallback(reply):
    units = [
        Unit("other", "weather", (2,), event_id="e2"),
        Unit("top", "invoice ZX result", (1,), event_id="e1"),
    ]
    gp = config(selector="llm", selector_min_units=1, selector_max_units=1)
    selected, receipt = select(units, gp, Backend(reply=reply))

    assert selected == [units[1]]
    assert receipt["selected_ids"] == ["top"]
    assert receipt["selector_fallback"] == {
        "applied": True,
        "reason": "invalid_or_malformed_selector_output",
        "policy": "lexical_top_1",
    }


def test_retrieved_field_catalog_bounds_sources_then_fields_before_llm():
    fields = []
    for source in range(10):
        for field_index in range(4):
            text = "invoice ZX" if source == 0 and field_index == 0 else "unrelated"
            fields.append(
                Unit(
                    f"e{source}:f{field_index}",
                    text,
                    (source,),
                    source_type="field",
                    event_id=f"e{source}",
                    metadata={"field_name": f"f{field_index}"},
                )
            )
    backend = Backend(reply='{"selected_ids":["illegal"]}')
    gp = config(
        selector="llm",
        selector_min_units=1,
        selector_max_units=4,
        selector_catalog="retrieved_fields",
        candidate_limit=8,
        field_candidate_limit=32,
    )
    selected, receipt = select(fields, gp, backend)

    payload = json.loads(backend.calls[0][2][1]["content"])
    assert len(receipt["retrieved_source_ids"]) == 8
    assert len(receipt["bounded_catalog_ids"]) == 32
    assert len(payload["catalog"]) == 32
    assert {row["unit_id"] for row in payload["catalog"]} == set(
        receipt["bounded_catalog_ids"]
    )
    assert [item.unit_id for item in selected] == ["e0:f0"]
    assert receipt["selector_fallback"]["policy"] == "lexical_top_1"


def test_retrieved_field_catalog_rejects_non_field_units():
    gp = config(selector="fixed", selector_catalog="retrieved_fields")
    with pytest.raises(ValueError, match="requires field evidence units"):
        select([Unit("e1", "invoice", (1,), event_id="e1")], gp)


def test_retrieved_field_catalog_caps_the_source_retrieval_stage_at_eight():
    gp = config(
        selector="llm",
        selector_catalog="retrieved_fields",
        candidate_limit=9,
    )
    with pytest.raises(ValueError, match="at most eight"):
        prepare_selection_dependencies(gp, Backend())


def test_candidate_relevance_scorer_only_ranks_and_keeps_a_nonempty_choice():
    scorer = train_candidate_scorer(
        training_rows("candidate_relevance", [1.0, 0.0]),
        label_kind="candidate_relevance",
        dataset_provenance={"dataset": "round2-cpu"},
    )
    gp = config(
        selector="supervised",
        selector_min_units=1,
        selector_max_units=1,
        selection_threshold=2.0,
        candidate_scorer=scorer,
    )
    selected, receipt = select(
        [Unit("only", "unrelated weather", (1,), event_id="e1")], gp
    )

    assert [item.unit_id for item in selected] == ["only"]
    assert receipt["effective_threshold"] is None
    assert receipt["candidate_scorer"]["label_kind"] == "candidate_relevance"
    assert receipt["candidate_scorer"]["selection_semantics"] == "ranking_only"
    assert receipt["candidate_scorer"]["empty_set_allowed"] is False


def test_supervised_ranking_inherits_integer_k_when_selector_max_is_omitted():
    scorer = train_candidate_scorer(
        training_rows("candidate_relevance", [1.0, 0.0]),
        label_kind="candidate_relevance",
        dataset_provenance={"dataset": "round2-cpu"},
    )
    gp = config(
        K=4,
        selector="supervised",
        candidate_scorer=scorer,
    )
    units = [
        Unit(f"u{index}", f"invoice ZX result {index}", (index,), event_id=f"e{index}")
        for index in range(4)
    ]
    selected, receipt = select(units, gp)

    assert len(selected) == 4
    assert receipt["selector_max_units"] == 4


def test_intervention_value_scorer_can_select_the_empty_set():
    scorer = train_candidate_scorer(
        training_rows("intervention_value", [-1.0, -2.0]),
        label_kind="intervention_value",
        dataset_provenance={"dataset": "round2-cpu"},
    )
    gp = config(
        selector="supervised",
        selector_min_units=1,
        selector_max_units=4,
        candidate_scorer=scorer,
    )
    selected, receipt = select(
        [Unit("bad", "unrelated weather", (1,), event_id="e1")], gp
    )

    assert selected == []
    assert receipt["reason"] == "selector_abstained"
    assert receipt["candidate_scorer"]["label_kind"] == "intervention_value"
    assert receipt["candidate_scorer"]["selection_semantics"] == (
        "positive_intervention_value"
    )
    assert receipt["candidate_scorer"]["empty_set_allowed"] is True


def test_hybrid_rrf_uses_fixed_rank_fusion_and_requires_embeddings():
    units = [
        Unit("lexical", "invoice ZX", (1,), event_id="e1"),
        Unit("semantic", "weather", (2,), event_id="e2"),
    ]
    gp = config(
        Q="hybrid",
        selector="fixed",
        selector_max_units=1,
        hybrid_fusion="rrf",
        rrf_k=60,
    )
    with pytest.raises(ValueError, match="embed"):
        prepare_selection_dependencies(gp)

    backend = Backend(vectors=[[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    _, receipt = select(units, gp, backend)
    components = receipt["ranked_candidates"][0]["components"]
    assert components["hybrid_fusion"] == "rrf"
    assert components["rrf_k"] == 60
    assert {components["lexical_rank"], components["semantic_rank"]} <= {1, 2}
    assert backend.calls[0][1] == "hybrid_query_and_catalog"


def test_legacy_selection_has_no_new_selector_behavior():
    units = [
        Unit("top", "invoice ZX", (1,), event_id="e1"),
        Unit("other", "weather", (2,), event_id="e2"),
    ]
    selected, receipt = select(units, config(K=1))
    assert selected == [units[0]]
    assert receipt["selector_mode"] is None
    assert "selector_fallback" not in receipt
