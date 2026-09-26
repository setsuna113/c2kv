"""Behavioral seams for versioned archive supply and ID-set recovery."""
from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest

from history_memory.events import EventStore
from benchmarks.memory_runtime.attempt_journal import AttemptJournal
from benchmarks.memory_runtime.event_native_step import EventNativeDecisionRunner
from benchmarks.memory_runtime.recovery.evidence_units import build_catalog
from benchmarks.memory_runtime.recovery.experiment_config import parse_gp_config
from benchmarks.memory_runtime.recovery.selection import select_candidates
from benchmarks.memory_runtime.recovery.set_protocol import (
    PROPOSAL_PROTOCOL, build_legal_sets, build_proposal_catalog, context_from_prepared,
    uncovered_units, choose_parameter_source, supported_parameters,
)
from benchmarks.memory_runtime.recovery.set_retrieval import retrieve_archive, supply_candidates
from benchmarks.memory_runtime.recovery.set_selectors import select_evidence_set
from benchmarks.memory_runtime.tests.test_gp_recovery import make_controller, UnitTokenizer
from benchmarks.memory_runtime.tests.test_event_native_recovery import request, draft_call
from benchmarks.memory_runtime.tests.test_event_native_step import Generator


def config(**updates):
    return parse_gp_config({"selection_protocol": "evidence_sets_v1", "D": "candidate_rule",
        "Q": "lexical", "U": "tokens_1024", "candidate_limit": 8,
        "selector_max_units": 4, "R": 1, **updates})


class LocalModels:
    def __init__(self, action=0):
        self.action, self.payload, self.calls = action, None, []

    def embed(self, *, texts, purpose, config):
        self.calls.append((purpose, list(texts)))
        return [[1.0, float("violet" in text), float("other-a" in text)] for text in texts]

    def embed_retrieval_queries(self, *, task, draft, overflow_policy):
        texts = [task, draft, task + "\n" + draft]
        return self.embed(texts=texts, purpose="query", config={}), {
            "schema": "test-retrieval-query-limit-v1",
            "overflow_policy": overflow_policy,
        }

    def choose_action(self, *, context_payload, allowed_actions):
        self.payload = context_payload
        return self.action

    def rerank(self, *, query, documents):
        return [0.8] * len(documents)

    def rerank_retrieval_candidates(
        self, *, task, draft, documents, overflow_policy
    ):
        return self.rerank(query=task + "\n" + draft, documents=documents)


def test_legacy_serialization_stays_unchanged_and_invalid_field_contract_fails_early():
    assert "selection_protocol" not in parse_gp_config({})
    with pytest.raises(ValueError, match="requires U=field"):
        parse_gp_config({"U": "tokens_1024", "selector_catalog": "retrieved_fields"})
    assert parse_gp_config({"selector_min_units": 0})["selector_min_units"] == 0


def test_proposal_selectors_require_versioned_h0_config():
    common = {
        "selection_protocol": "evidence_sets_v1",
        "D": "candidate_rule",
        "set_selector": "risk_source_proposal",
        "selector_artifact": {"fixture": True},
    }
    with pytest.raises(ValueError, match="proposal_protocol"):
        parse_gp_config(common)
    with pytest.raises(ValueError, match="H0"):
        parse_gp_config({
            **common, "G": "record_bound", "proposal_protocol": PROPOSAL_PROTOCOL
        })
    parsed = parse_gp_config({
        **common, "G": "current", "proposal_protocol": PROPOSAL_PROTOCOL
    })
    assert parsed["proposal_protocol"] == PROPOSAL_PROTOCOL


def test_explicit_empty_llm_selection_does_not_fallback():
    controller = make_controller()
    prepared = controller.prepare(request(), ratio=4, max_new_tokens=32)
    catalog = build_catalog(prepared._store, UnitTokenizer(), "tokens_1024")
    backend = SimpleNamespace(chat=lambda **kwargs: '{"selected_ids":[]}')
    selected, receipt = select_candidates(catalog, goal="violet", draft_text="lookup violet",
        draft_tool_calls=[draft_call()], config=parse_gp_config({"selector": "llm", "selector_min_units": 0}),
        backends=backend)
    assert selected == []
    assert receipt["selector_fallback"]["applied"] is False


def test_legal_action_catalog_has_39_sets_and_checks_whole_set_budget():
    store = EventStore.from_messages("s", [{"role": "user", "content": str(i)} for i in range(8)])
    units = build_catalog(store, UnitTokenizer(), "tokens_1024")
    candidates = [{"unit": unit} for unit in units]
    actions, rejected = build_legal_sets(candidates, lambda units: (True, {}))
    assert len(actions) == 39 and actions[0] == () and not rejected
    assert len(actions[-1]) == 4
    limited, rejected = build_legal_sets(candidates, lambda units: (len(units) <= 1, {"reason": "sum_budget"}))
    assert len(limited) == 9 and len(rejected) == 30


def test_exact_partial_coverage_is_subtracted_but_gist_is_not():
    store = EventStore.from_messages("s", [{"role": "user", "content": "x" * 1400}])
    tokenizer = UnitTokenizer()
    coarse = build_catalog(store, tokenizer, "tokens_1024")[0]
    fine = build_catalog(store, tokenizer, "tokens_256")[0]
    assert uncovered_units(coarse, store, tokenizer, (), ()) == [coarse]
    pieces = uncovered_units(coarse, store, tokenizer, (), [fine])
    assert pieces and sum(piece.token_count for piece in pieces) < coarse.token_count
    assert all(piece.provenance[0].char_start >= fine.provenance[-1].char_end for piece in pieces)
    assert uncovered_units(coarse, store, tokenizer, (0,), ()) == []


def test_three_routes_reach_selector_supply_without_k_truncation():
    controller = make_controller()
    prepared = controller.prepare(request(), ratio=4, max_new_tokens=32)
    context = context_from_prepared(prepared, [draft_call()], "lookup violet other-a")
    models = LocalModels()
    gp = config(Q="archive_rrf", K=1)
    rows, receipt = supply_candidates(prepared, UnitTokenizer(), gp, context,
        lambda units: (True, {}), models)
    assert len(rows) >= 2
    assert len({row["event_id"] for row in rows}) >= 2
    assert set(receipt["retrieval"]["routes"]) == {"task_lexical", "draft_lexical", "semantic"}
    assert receipt["retrieval"]["semantic_query_limits"]["overflow_policy"] == "error"
    assert models.calls[0][0] == "query" and models.calls[1][0] == "document"
    assert len(models.calls[1][1]) == receipt["n_archive_units"]
    assert all("score" in row for row in rows)


def test_query_overflow_policy_is_explicit_and_validated():
    assert config().get("semantic_query_overflow_policy", "error") == "error"
    assert config(
        semantic_query_overflow_policy="task_head_tail_preserve_draft_v1"
    )["semantic_query_overflow_policy"] == "task_head_tail_preserve_draft_v1"
    with pytest.raises(ValueError, match="semantic_query_overflow_policy"):
        config(semantic_query_overflow_policy="silent_truncate")


def test_infeasible_first_candidate_does_not_end_supply():
    controller = make_controller()
    prepared = controller.prepare(request(), ratio=4, max_new_tokens=32)
    context = context_from_prepared(prepared, [draft_call()], "lookup violet other-a")
    gp = config()
    ranked, _ = retrieve_archive(build_catalog(prepared._store, UnitTokenizer(), gp["U"]), context, gp)
    eligible = set(prepared.metadata["eligible_extraction"]["eligible_event_ids"])
    raw = set(prepared.memory.raw_source_indices)
    first = next(row["unit_id"] for row in ranked if row["unit"].event_id in eligible
        and not set(row["unit"].source_indices) <= raw)
    probed = []
    def admission(units):
        probed.extend(unit.unit_id for unit in units)
        return all(unit.unit_id != first for unit in units), {"policy": "synthetic_tight_budget"}
    rows, receipt = supply_candidates(prepared, UnitTokenizer(), gp, context, admission)
    assert first in probed
    assert rows and all(row["unit_id"] != first for row in rows)
    assert any(row["unit_id"] == first and row["reason"] == "candidate_not_admitted" for row in receipt["rejections"])


def test_candidate_supply_accepts_openai_text_message_with_null_tool_calls():
    payload = request("turn-1/step-0")
    payload["messages"].insert(-1, {
        "role": "assistant",
        "content": "The violet file was read successfully.",
        "tool_calls": None,
    })
    payload["messages"][-1] = {"role": "user", "content": "Read violet file1 again."}
    controller = make_controller()
    prepared = controller.prepare(payload, ratio=4, max_new_tokens=32)
    context = context_from_prepared(prepared, [draft_call()], "cat violet file1")

    rows, receipt = supply_candidates(
        prepared, UnitTokenizer(), config(), context, lambda units: (True, {})
    )

    assert rows
    assert receipt["n_presented_to_selector"] == len(rows)


def test_static_256_fallback_has_original_source_ranges():
    controller = make_controller()
    payload = request()
    payload["messages"][3]["content"] = json.dumps({"value": "violet " * 300})
    prepared = controller.prepare(payload, ratio=4, max_new_tokens=32)
    context = context_from_prepared(prepared, [draft_call()], "lookup violet")
    rows, receipt = supply_candidates(prepared, UnitTokenizer(), config(fallback_unit="tokens_256"), context,
        lambda units: (all(unit.token_count <= 256 for unit in units), {}))
    assert receipt["fallback_attempts"] > 0
    assert any(row["fallback_parent_id"] and row["source_type"] == "tokens_256" for row in rows)
    assert all(row["token_count"] <= 256 and row["provenance"] for row in rows)


def test_empty_set_preserves_actor_input_cache_and_final_response(tmp_path):
    outputs = "<tool_call>" + json.dumps({"name": "lookup", "arguments": {"id": "item-17"}}) + "</tool_call>"
    states = []
    for name, switches in (("legacy", {"D": "detector"}),
                            ("sets", config(set_selector="local_llm")),
                            ("proposal_veto", config())):
        controller = make_controller(**switches)
        if name == "sets":
            controller.backends = LocalModels(action=0)
        elif name == "proposal_veto":
            controller.gp.update(
                set_selector="risk_source_proposal",
                proposal_protocol=PROPOSAL_PROTOCOL,
                selector_threshold=0.5,
            )
            controller.trained_selector = _RiskProposalSelector(0.5)
        else:
            controller._detector_gate = lambda prepared: {"triggered": False, "reason": "test_no_intervention"}
        journal = tmp_path / (name + ".jsonl")
        generator = Generator(journal, [outputs])
        runner = EventNativeDecisionRunner(controller, generator, UnitTokenizer(), ratio=4,
            max_new_tokens=32, max_generation_calls=96, journal=AttemptJournal(journal))
        record = runner.run(request())
        states.append((generator.inputs, record["response"], record["session_cache_after"]))
        assert record["generation_attempts"] == 1
        assert not controller._recovery_counts
        assert not controller._leases["task-1"]
        if name in {"sets", "proposal_veto"}:
            check = record["recovery_checks"][0]
            assert check["selection"]["action_id"] == 0
            assert check["candidate_supply"]["n_presented_to_selector"] >= 2
            assert check["candidate_supply"]["actually_appended_ids"] == []
            if name == "sets":
                assert controller.backends.payload["candidates"]
    assert all(state == states[0] for state in states[1:])


def test_c0_append_and_reranker_use_same_legal_actions_and_b0(tmp_path):
    controller = make_controller(**config(export_selection_state=True))
    prepared = controller.prepare(request(), ratio=4, max_new_tokens=32)
    original = prepared.memory
    result = controller.reconsider(prepared, [draft_call()], draft_text="lookup violet other-a")
    assert result["regenerate"]
    assert result["memory"].chunks == original.chunks
    assert result["memory"].view.raw_event_ids == original.view.raw_event_ids
    decision = result["decision"]
    assert len(decision["selected_unit_ids"]) == 1
    assert decision["candidate_supply"]["selected_ids"] == decision["appended_unit_ids"]
    assert decision["selection_state"]["schema"] == "t02-state-candidate-v1"
    assert decision["selection_state"]["allowed_actions"][0]["candidate_ids"] == []
    assert result["metadata"]["actual_history_bytes"] <= controller.policy_config.history_budget_bytes


def test_reservation_restores_original_total_cap_and_is_recorded_before_draft():
    controller = make_controller(**config(recovery_reserve_tokens=512))
    original = controller._packer.policy_config
    prepared = controller.prepare(request(), ratio=4, max_new_tokens=32)
    assert controller._packer.policy_config is original
    reserve = prepared.metadata["gp_reservation"]
    assert reserve["reserved_bytes"] == 512 * controller.kv_bytes_per_token
    assert prepared.metadata["history_budget_bytes"] == original.history_budget_bytes
    assert prepared.metadata["actual_history_bytes"] <= reserve["first_draft_history_budget_bytes"]


def test_generation_limit_keeps_selection_telemetry_without_retrieval():
    controller = make_controller(**config())
    controller.required_task_generation_limit = 1
    prepared = controller.prepare(request(), ratio=4, max_new_tokens=32)
    result = controller.reconsider(prepared, [draft_call()], draft_text="lookup violet")
    supply = result["decision"]["candidate_supply"]
    assert supply["stage"] == "not_evaluated" and supply["n_archive_units"] is None
    assert supply["n_presented_to_selector"] == 0
    assert supply["selected_ids"] == supply["actually_appended_ids"] == []
    assert supply["rejection_reason"] == "shared_task_generation_limit"


def test_parameter_source_rule_ignores_visible_parameters_and_uses_typed_values():
    context = {"draft_tool_calls": [draft_call()], "parse_ok": True,
        "goal": "Continue", "raw_visible": []}
    candidates = [{"unit_id": "a", "event_id": "old", "text": '{"id":"item-17"}', "token_count": 16},
                  {"unit_id": "b", "event_id": "other", "text": '{"id":"item-18"}', "token_count": 16}]
    actions = [(), ("a",), ("b",), ("a", "b")]
    assert choose_parameter_source(context, candidates, actions)[0] == ("a",)
    context["raw_visible"] = [{"role": "tool", "content": '{"id":"item-17"}'}]
    assert choose_parameter_source(context, candidates, actions)[0] == ()


def _proposal_context(*, stop=False):
    return {
        "goal": "Continue",
        "last_action_observation": [],
        "raw_visible": [],
        "raw_source_ids": ["current"],
        "draft_logprobs": [-0.1],
        "draft_text": "" if stop else "lookup item-17",
        "draft_tool_calls": [] if stop else [draft_call()],
        "parse_ok": True,
        "is_stop": stop,
    }


def _proposal_candidates(*, source_first=False):
    first = '{"id":"item-17"}' if source_first else "unrelated"
    return [
        {"unit_id": "a", "event_id": "ea", "text": first, "token_count": 8},
        {"unit_id": "b", "event_id": "eb", "text": '{"id":"item-17"}', "token_count": 8},
    ]


def test_proposal_catalog_deduplicates_actions_but_keeps_origin_aliases():
    candidates = _proposal_candidates(source_first=True)
    proposals, receipt = build_proposal_catalog(
        _proposal_context(), candidates, [(), ("a",), ("b",), ("a", "b")]
    )
    assert proposals == ((), ("a",))
    assert receipt["protocol"] == PROPOSAL_PROTOCOL
    assert receipt["proposals"]["Slex"]["selected_ids"] == ["a"]
    assert receipt["proposals"]["Ssrc"]["selected_ids"] == ["a"]
    assert receipt["origin_aliases"][1] == {
        "selected_ids": ["a"], "origins": ["Slex", "Ssrc"],
        "origin_selected_ids": {"Slex": ["a"], "Ssrc": ["a"]},
    }
    assert receipt["distinct"] is False


def test_proposal_catalog_keeps_multisource_ssrc_and_discards_illegal_actions():
    context = _proposal_context()
    context["draft_tool_calls"] = [{
        "function": {
            "name": "lookup",
            "arguments": {"id": "item-17", "target": "item-18"},
        }
    }]
    candidates = [
        {"unit_id": "a", "event_id": "ea", "text": '{"id":"item-17"}', "token_count": 8},
        {"unit_id": "b", "event_id": "eb", "text": '{"target":"item-18"}', "token_count": 8},
    ]
    proposals, receipt = build_proposal_catalog(
        context, candidates,
        [(), ("missing",), ("a", "a"), ("a",), ("b",),
         ("a", "b"), ("b", "a")],
    )
    assert proposals == ((), ("a",), ("a", "b"))
    assert receipt["proposals"]["Ssrc"]["selected_ids"] == ["a", "b"]
    assert receipt["proposals"]["Ssrc"]["unsupported_parameter_coverage_count"] == 2
    assert all("missing" not in action for action in receipt["deduplicated_actions"])
    assert {row["reason"] for row in receipt["discarded_actions"]} == {
        "unknown_candidate_id", "repeated_candidate_id", "duplicate_candidate_set"
    }
    assert receipt["distinct"] is True


def test_proposal_telemetry_is_present_when_empty_is_the_only_legal_action():
    chosen, receipt = select_evidence_set(
        _proposal_context(stop=True), [], [()],
        config(
            set_selector="risk_source_proposal",
            selector_artifact={"fixture": True},
            proposal_protocol=PROPOSAL_PROTOCOL,
        ),
        models=LocalModels(), tokenizer=UnitTokenizer(),
        trained=_RiskProposalSelector(0.9),
    )
    assert chosen == ()
    assert receipt["reason"] == "no_nonempty_proposal"
    assert receipt["proposal_ids"] == {"Slex": [], "Ssrc": []}
    assert receipt["proposal_available"] == {"Slex": False, "Ssrc": False}
    assert receipt["proposal_catalog"]["deduplicated_actions"] == [[]]


class _RiskProposalSelector:
    kind = "risk"
    artifact = {"model_kind": "c1_risk_logistic"}

    def __init__(self, score, *, available=True):
        self.score = score
        self.available = available

    def predict_risk(self, context):
        del context
        return SimpleNamespace(
            available=self.available,
            score=self.score if self.available else None,
            reason=None if self.available else "prefill unavailable",
            missing_fields=() if self.available else ("context.prefill_hidden",),
        )


def test_risk_source_proposal_stop_uses_slex_fallback_and_records_it():
    chosen, receipt = select_evidence_set(
        _proposal_context(stop=True),
        _proposal_candidates(),
        [(), ("a",), ("b",), ("a", "b")],
        config(
            set_selector="risk_source_proposal",
            selector_artifact={"fixture": True},
            proposal_protocol=PROPOSAL_PROTOCOL,
            selector_threshold=0.5,
        ),
        models=LocalModels(), tokenizer=UnitTokenizer(),
        trained=_RiskProposalSelector(0.9),
    )
    assert chosen == ("a",)
    assert receipt["chosen_origin"] == "Slex"
    assert receipt["proposal_fallback"] == {
        "applied": True, "from": "Ssrc", "to": "Slex",
        "reason": "source_proposal_empty",
    }
    assert receipt["reason"] == "risk_triggered_lexical_fallback"


def test_risk_source_proposal_c1_is_the_final_veto_and_unavailable_abstains():
    kwargs = dict(
        context=_proposal_context(), candidates=_proposal_candidates(),
        actions=[(), ("a",), ("b",), ("a", "b")],
        config=config(
            set_selector="risk_source_proposal",
            selector_artifact={"fixture": True},
            proposal_protocol=PROPOSAL_PROTOCOL,
            selector_threshold=0.5,
        ),
        models=LocalModels(), tokenizer=UnitTokenizer(),
    )
    chosen, receipt = select_evidence_set(
        **kwargs, trained=_RiskProposalSelector(0.5)
    )
    assert chosen == () and receipt["reason"] == "risk_not_above_threshold"
    chosen, receipt = select_evidence_set(
        **kwargs, trained=_RiskProposalSelector(0.9, available=False)
    )
    assert chosen == () and receipt["available"] is False
    assert receipt["reason"] == "prefill unavailable"


class _GainProposalSelector:
    kind = "gain_turn"

    def __init__(self, selected=("b",)):
        self.artifact = {
            "model_kind": "c4_gain_turn", "proposal_protocol": PROPOSAL_PROTOCOL
        }
        self.selected = selected
        self.seen_actions = None

    def select(self, context, candidates, legal_actions, *, tokenizer, delta):
        del context, candidates, tokenizer, delta
        self.seen_actions = tuple(tuple(action) for action in legal_actions)
        return SimpleNamespace(
            selected_ids=self.selected,
            score=0.8,
            available=True,
            reason="selected_positive_gain",
            scores=((self.seen_actions[1], 0.2), (self.seen_actions[2], 0.8)),
            missing_fields=(),
        )


def test_gain_proposal_selector_scores_only_empty_slex_ssrc_not_full_catalog():
    trained = _GainProposalSelector()
    candidates = _proposal_candidates()
    chosen, receipt = select_evidence_set(
        _proposal_context(), candidates,
        [(), ("a",), ("b",), ("a", "b")],
        config(
            set_selector="gain_turn_proposals",
            selector_artifact=trained.artifact,
            proposal_protocol=PROPOSAL_PROTOCOL,
        ),
        models=LocalModels(), tokenizer=UnitTokenizer(), trained=trained,
    )
    assert trained.seen_actions == ((), ("a",), ("b",))
    assert chosen == ("b",) and receipt["chosen_origin"] == "Ssrc"
    assert [row["selected_ids"] for row in receipt["scored_actions"]] == [["a"], ["b"]]


def test_gain_proposal_selector_rejects_action_outside_proposal_catalog():
    trained = _GainProposalSelector(selected=("a", "b"))
    with pytest.raises(ValueError, match="outside the proposal catalog"):
        select_evidence_set(
            _proposal_context(), _proposal_candidates(),
            [(), ("a",), ("b",), ("a", "b")],
            config(
                set_selector="gain_turn_proposals",
                selector_artifact=trained.artifact,
                proposal_protocol=PROPOSAL_PROTOCOL,
            ),
            models=LocalModels(), tokenizer=UnitTokenizer(), trained=trained,
        )


@pytest.mark.parametrize(
    "selector,expected",
    [("candidate_rule", ("a",)), ("parameter_source", ("b",))],
)
def test_original_set_selector_modes_keep_their_action_semantics(selector, expected):
    chosen, _ = select_evidence_set(
        _proposal_context(), _proposal_candidates(),
        [(), ("a",), ("b",), ("a", "b")],
        config(set_selector=selector), models=LocalModels(), tokenizer=UnitTokenizer(),
    )
    assert chosen == expected


def test_typed_parameter_support_parses_json_after_ordinary_task_text():
    parameters = {("number", 17), ("string", "17"), ("string", "target")}
    text = 'Find target\n[{"role":"tool","content":"{\\"id\\":\\"17\\"}"}]'
    assert supported_parameters(parameters, text) == {("string", "target"), ("string", "17")}
    assert supported_parameters({("string", "17")}, 'prefix {"id":17}') == set()


def test_k_one_limits_legal_actions_without_truncating_presented_pool():
    controller = make_controller(**config(set_selector="local_llm", K=1))
    controller.backends = LocalModels(action=0)
    prepared = controller.prepare(request(), ratio=4, max_new_tokens=32)
    controller.reconsider(prepared, [draft_call()], draft_text="lookup violet other-a")
    payload = controller.backends.payload
    assert len(payload["candidates"]) >= 2
    assert all(len(action["selected_ids"]) <= 1 for action in payload["allowed_sets"])


def test_reranker_calibration_uses_shared_set_budget_and_configured_threshold():
    context = {"goal": "query", "last_action_observation": [], "is_stop": False,
        "parse_ok": True, "draft_tool_calls": [draft_call()], "draft_text": "lookup"}
    candidates = [{"unit_id": "a", "text": "evidence a"}, {"unit_id": "b", "text": "evidence b"}]
    predictions = {"a": 0.7, "b": 0.9}
    calibrated = SimpleNamespace(model=SimpleNamespace(predict_candidate=lambda q, row:
        SimpleNamespace(available=True, score=predictions[row["unit_id"]])))
    chosen, receipt = select_evidence_set(context, candidates, [(), ("a",), ("b",)],
        config(set_selector="reranker", selector_threshold=0.8), models=LocalModels(),
        tokenizer=UnitTokenizer(), trained=calibrated)
    assert chosen == ("b",)
    assert receipt["score_semantics"] == "sum_calibrated_relevance_minus_threshold"


def test_empty_set_preserves_sglang_adapter_requests_handles_and_next_decision(tmp_path):
    # Exercise the actual adapter with a deterministic in-memory HTTP server.
    # This covers handle/session state, not physical device KV equivalence.
    from history_memory.test_sglang_generator import _Opener, _Response, _generator, _native_response

    output = "violet other-a"
    class ActorTransport(_Opener):
        def open(self, request, *, timeout):
            if request.get_method() == "GET":
                return super().open(request, timeout=timeout)
            payload = json.loads(request.data)
            self.requests.append(payload)
            self.generate_calls += 1
            response = _native_response(payload, cache_hit=self.generate_calls > 1)
            response.update(output_ids=list(map(ord, output)), text=output,
                token_logprobs=[-0.25] * len(output))
            return _Response(response)

    observed = []
    for name, switches in (("legacy", {"D": "detector"}), ("sets", config(set_selector="local_llm"))):
        controller = make_controller(**switches)
        if name == "sets":
            controller.backends = LocalModels(action=0)
        else:
            controller._detector_gate = lambda prepared: {"triggered": False, "reason": "test_no_intervention"}
        transport = ActorTransport()
        generator = _generator(transport, model_context=100_000, max_new_tokens=32,
            max_extraction_calls=100, encoding_scope="current")
        runner = EventNativeDecisionRunner(controller, generator, UnitTokenizer(), ratio=4,
            max_new_tokens=32, max_generation_calls=96,
            journal=AttemptJournal(tmp_path / (name + ".jsonl")))
        responses = [runner.run(request(key))["response"] for key in ("d1", "d2")]
        actor_inputs = [{key: value for key, value in row.items() if key not in {"rid", "generation_id"}}
                        for row in transport.requests if "encoder_chunks" in row]
        observed.append((actor_inputs, responses, generator.session_cache_info(), copy.deepcopy(generator._session_cache)))
        assert transport.generate_calls == 2
        assert generator._session_cache.generation == 2
        assert generator._session_cache.handles
    assert observed[0] == observed[1]
