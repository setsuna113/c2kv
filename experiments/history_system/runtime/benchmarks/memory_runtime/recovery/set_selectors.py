"""Mutually exclusive C0--C5 implementations of one evidence-set interface."""
from __future__ import annotations

import math

from .set_protocol import build_proposal_catalog, choose_parameter_source
from .set_retrieval import enrich_candidates


def public_candidates(candidates):
    return [{key: value for key, value in row.items() if key != "unit"} for row in candidates]


def _proposal_origins(catalog, action):
    selected = list(action)
    return next((list(row["origins"]) for row in catalog["origin_aliases"]
                 if row["selected_ids"] == selected), [])


def select_evidence_set(context, candidates, actions, config, *, models, tokenizer, trained=None):
    selector = config.get("set_selector", "candidate_rule")
    receipt = {"schema": "recovery-set-selector-v1", "selector": selector,
        "n_presented_to_selector": len(candidates), "legal_action_count": len(actions),
        "selected_ids": [], "action_id": 0, "available": True, "reason": "empty_set",
        "score": 0.0, "score_semantics": None}
    proposal_selector = selector in {
        "risk_source_proposal", "gain_turn_proposals", "gain_task_proposals"
    }
    if len(actions) <= 1 and not proposal_selector:
        receipt["reason"] = "no_feasible_evidence_sets"
        return (), receipt
    chosen, score = (), 0.0
    if selector == "candidate_rule":
        chosen = next((tuple(action) for action in actions if len(action) == 1), ())
        receipt["score_semantics"] = "first_feasible_ranked_single"
    elif selector == "parameter_source":
        chosen, score = choose_parameter_source(context, candidates, actions)
        receipt["score_semantics"] = "unsupported_typed_parameter_coverage"
    elif selector == "reranker":
        enrich_candidates(
            context,
            candidates,
            models,
            rerank=True,
            semantic=trained is not None,
            overflow_policy=config.get("semantic_query_overflow_policy", "error"),
        )
        scores = {row["unit_id"]: row["reranker_score"] for row in candidates}
        if trained is not None:
            for row in candidates:
                prediction = trained.model.predict_candidate(context, row)
                if not prediction.available or prediction.score is None:
                    receipt.update(available=False, reason=prediction.reason,
                        missing_fields=list(prediction.missing_fields), score_semantics="t01_calibrated_relevance")
                    return (), receipt
                scores[row["unit_id"]] = prediction.score
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in scores.values()):
            raise ValueError("reranker scores must be finite values between zero and one")
        threshold = config.get("selector_threshold", 0.5)
        action_scores = [sum(scores[identifier] - threshold for identifier in action) for action in actions]
        best = min(range(len(actions)), key=lambda index: (-action_scores[index], len(actions[index]), index))
        chosen, score = tuple(actions[best]), action_scores[best]
        receipt.update(score_semantics="sum_calibrated_relevance_minus_threshold" if trained is not None else "sum_relevance_minus_threshold", threshold=threshold,
            candidate_scores=scores, action_scores=action_scores)
    elif selector == "local_llm":
        payload = {"q_t": {key: value for key, value in context.items()
                    if key not in {"prefill_hidden", "draft_logprobs"}},
            "d_t": {"kind": "STOP" if context["is_stop"] else "CALLS" if context["parse_ok"] else "MALFORMED",
                    "text": context["draft_text"], "tool_calls": context["draft_tool_calls"]},
            "candidates": public_candidates(candidates),
            "allowed_sets": [{"action_id": index, "selected_ids": list(action)} for index, action in enumerate(actions)]}
        action_id = models.choose_action(context_payload=payload, allowed_actions=actions)
        if type(action_id) is not int or not 0 <= action_id < len(actions):
            raise ValueError("local selector returned an action outside the legal action catalog")
        chosen = tuple(actions[action_id])
        receipt["score_semantics"] = "constrained_action_id"
    elif proposal_selector:
        if trained is None:
            raise ValueError("trained selector artifact was not loaded")
        proposal_actions, proposal_catalog = build_proposal_catalog(
            context, candidates, actions
        )
        receipt.update(
            proposal_catalog=proposal_catalog,
            proposal_protocol=proposal_catalog["protocol"],
            proposal_ids={
                origin: proposal_catalog["proposals"][origin]["selected_ids"]
                for origin in ("Slex", "Ssrc")
            },
            proposal_available={
                origin: proposal_catalog["proposals"][origin]["available"]
                for origin in ("Slex", "Ssrc")
            },
            proposal_distinct=proposal_catalog["distinct"],
            proposal_fallback=proposal_catalog["fallback"],
            scored_actions=[],
            chosen_origin="empty",
            chosen_origin_aliases=["empty"],
        )
        if len(proposal_actions) <= 1:
            receipt.update(reason="no_nonempty_proposal", score_semantics="proposal_catalog_empty")
        elif selector == "risk_source_proposal":
            prediction = trained.predict_risk(context)
            threshold = config.get("selector_threshold", 0.5)
            receipt.update(
                available=prediction.available,
                reason=prediction.reason,
                score_semantics="current_turn_failure_risk_final_veto",
                threshold=threshold,
                missing_fields=list(prediction.missing_fields),
            )
            score = prediction.score
            if prediction.available and score is not None:
                if score <= threshold:
                    receipt["reason"] = "risk_not_above_threshold"
                else:
                    ssrc = tuple(proposal_catalog["proposals"]["Ssrc"].get(
                        "canonical_selected_ids", ()
                    ))
                    slex = tuple(proposal_catalog["proposals"]["Slex"].get(
                        "canonical_selected_ids", ()
                    ))
                    if ssrc:
                        chosen = ssrc
                        receipt["chosen_origin"] = "Ssrc"
                        receipt["reason"] = "risk_triggered_source_proposal"
                    elif slex:
                        chosen = slex
                        receipt["chosen_origin"] = "Slex"
                        receipt["reason"] = "risk_triggered_lexical_fallback"
                    else:
                        receipt["reason"] = "no_nonempty_proposal"
                    receipt["chosen_origin_aliases"] = _proposal_origins(
                        proposal_catalog, chosen
                    ) or [receipt["chosen_origin"]]
        else:
            from .set_models import validate_proposal_artifact

            validate_proposal_artifact(trained.artifact)
            enrich_candidates(
                context,
                candidates,
                models,
                rerank=True,
                semantic=True,
                overflow_policy=config.get("semantic_query_overflow_policy", "error"),
            )
            prediction = trained.select(
                context,
                candidates,
                legal_actions=proposal_actions,
                tokenizer=tokenizer,
                delta=config.get("gain_delta", 0.0),
            )
            chosen, score = tuple(prediction.selected_ids), prediction.score
            receipt.update(
                available=prediction.available,
                reason=prediction.reason,
                missing_fields=list(prediction.missing_fields),
                score_semantics="estimated_" + ("gain_turn" if selector == "gain_turn_proposals" else "gain_task"),
                delta=config.get("gain_delta", 0.0),
                scored_actions=[
                    {
                        "selected_ids": list(action),
                        "score": action_score,
                        "origins": _proposal_origins(proposal_catalog, action),
                    }
                    for action, action_score in prediction.scores
                ],
            )
            origins = _proposal_origins(proposal_catalog, chosen)
            receipt["chosen_origin_aliases"] = origins or ["empty"]
            receipt["chosen_origin"] = (origins or ["empty"])[0]
        if chosen not in proposal_actions:
            raise ValueError("proposal selector returned an action outside the proposal catalog")
    elif selector in {"risk", "gain_turn", "gain_task"}:
        if trained is None:
            raise ValueError("trained selector artifact was not loaded")
        if selector != "risk":
            enrich_candidates(
                context,
                candidates,
                models,
                rerank=True,
                semantic=True,
                overflow_policy=config.get("semantic_query_overflow_policy", "error"),
            )
        prediction = trained.select(context, candidates, legal_actions=actions,
            **({"threshold": config.get("selector_threshold", 0.5)} if selector == "risk"
               else {"tokenizer": tokenizer, "delta": config.get("gain_delta", 0.0)}))
        chosen, score = tuple(prediction.selected_ids), prediction.score
        receipt.update(available=prediction.available, reason=prediction.reason,
            score_semantics="current_turn_failure_risk" if selector == "risk" else "estimated_" + selector)
    else:
        raise ValueError(f"Unknown set selector: {selector}")
    chosen = tuple(chosen)
    if chosen not in actions:
        raise ValueError("selector returned a set outside the admissible action catalog")
    receipt.update(selected_ids=list(chosen), action_id=actions.index(chosen), score=score)
    if receipt["available"] and selector not in {
        "risk_source_proposal", "gain_turn_proposals", "gain_task_proposals"
    }:
        receipt["reason"] = "selected_legal_evidence_set" if chosen else "selector_empty_set"
    return chosen, receipt
