"""Opt-in BFCL bridge for bounded recoverability diagnostics.

This adapter subclasses the live exact-state T02 adapter without changing its
ordinary branch path.  Selected snapshots can be pinned before A0/A1/A2 run,
reviewed from the restored observed prefix, and released explicitly after the
two diagnostic continuations have finished.
"""
from __future__ import annotations

import copy
import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import recoverability
import recoverability_runtime
import t02
import t02_bfcl


DIAGNOSTIC_OUTCOME_SCHEMA = "recoverability-bfcl-official-outcome-v1"
PREFIX_ARTIFACT_SCHEMA = "recoverability-bfcl-prefix-validity-v1"
ATTEMPT_ARTIFACT_SCHEMA = "recoverability-bfcl-attempt-v1"
MAX_RETAINED_STATES = 12
MAX_DIAGNOSTIC_ATTEMPTS = 24


class RecoverabilityBFCLAdapter(t02_bfcl.ExactBFCLBranchAdapter):
    """Retain exact BFCL snapshots for diagnostic-only continuations."""

    def __init__(self, *, frozen_policy: Mapping[str, Any], artifact_dir: str | Path,
                 family_bindings: Mapping[str, str] | None = None) -> None:
        super().__init__(frozen_policy=frozen_policy, artifact_dir=artifact_dir,
                         family_bindings=family_bindings)
        self._diagnostic_retained: set[str] = set()
        self._diagnostic_deferred_releases: set[str] = set()
        self._diagnostic_support: dict[str, dict[str, Any]] = {}
        self._diagnostic_attempt_keys: set[tuple[str, str]] = set()
        self._last_restore: dict[str, Any] | None = None
        self.diagnostic_branch_attempts = 0
        self.diagnostic_generation_calls = 0
        self.diagnostic_wall_seconds = 0.0
        self._load_attempt_receipts()

    def capabilities(self) -> dict[str, Any]:
        return {
            **super().capabilities(),
            "recoverability_diagnostic": {
                "known_support": True,
                "full_history": True,
            },
        }

    def retain_for_diagnostic(self, state_ids: Sequence[str]) -> None:
        """Pin at most twelve already-live snapshots before ordinary release."""
        requested = list(state_ids)
        if any(not isinstance(state_id, str) or not state_id for state_id in requested):
            raise ValueError("Diagnostic state ids must be nonempty strings")
        if len(set(requested)) != len(requested):
            raise ValueError("Diagnostic state ids must be unique")
        retained = self._diagnostic_retained | set(requested)
        if len(retained) > MAX_RETAINED_STATES:
            raise ValueError("Recoverability diagnostic retains at most 12 states")
        missing = set(requested) - set(self._states)
        if missing:
            raise ValueError(f"Diagnostic states lack live snapshots: {sorted(missing)}")
        self._diagnostic_retained = retained

    def capture_state(self, state: Mapping[str, Any], *,
                      frozen_policy: Mapping[str, Any]) -> dict[str, Any]:
        self._last_restore = None
        return super().capture_state(state, frozen_policy=frozen_policy)

    def restore_state(self, snapshot: Mapping[str, Any], *,
                      frozen_policy: Mapping[str, Any]) -> dict[str, Any]:
        receipt = super().restore_state(snapshot, frozen_policy=frozen_policy)
        self._last_restore = t02._json_copy(receipt, label="diagnostic restore receipt")
        return receipt

    def run_branch(self, state: Mapping[str, Any], branch: Mapping[str, Any], *,
                   frozen_policy: Mapping[str, Any],
                   restore_receipt: Mapping[str, Any]) -> dict[str, Any]:
        try:
            return super().run_branch(
                state, branch, frozen_policy=frozen_policy,
                restore_receipt=restore_receipt,
            )
        finally:
            # An ordinary T02 branch consumes the restored environment.  Its
            # receipt cannot certify a later diagnostic without a fresh restore.
            self._last_restore = None

    def _release_state(self, state_id: str) -> None:
        if state_id in self._diagnostic_retained:
            self._diagnostic_deferred_releases.add(state_id)
            return
        super()._release_state(state_id)

    def release_diagnostic(self, state_id: str) -> None:
        """Release one pinned snapshot after all intended diagnostics finish."""
        if state_id not in self._diagnostic_retained:
            raise ValueError(f"State is not retained for diagnostic use: {state_id}")
        self._diagnostic_retained.remove(state_id)
        self._diagnostic_deferred_releases.discard(state_id)
        self._diagnostic_support.pop(state_id, None)
        if self._active is not None and self._active["state"]["state_id"] == state_id:
            self._active = None
            self._last_restore = None
        if state_id in self._states:
            super()._release_state(state_id)

    def prepare_support(self, state: Mapping[str, Any],
                        annotations: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        registered = self._require_active_state(state)
        del registered
        exact = t02._json_copy(list(annotations), label="support annotations")
        prior = self._diagnostic_support.get(state["state_id"])
        if prior is not None and prior["annotations"] != exact:
            raise ValueError("Support annotations changed for the retained snapshot")
        prepared = recoverability_runtime.prepare_support(self._active["actor"], exact)
        value = t02._json_copy(prepared, label="prepared diagnostic support")
        self._diagnostic_support[state["state_id"]] = {
            "annotations": exact,
            "prepared": value,
        }
        return copy.deepcopy(value)

    def run_diagnostic(self, state: Mapping[str, Any], branch: Mapping[str, Any], *,
                       frozen_policy: Mapping[str, Any],
                       restore_receipt: Mapping[str, Any]) -> dict[str, Any]:
        registered = self._require_active_state(state)
        if t02_bfcl._json_copy(frozen_policy) != self.frozen_policy:
            raise ValueError("Frozen continuation policy changed before diagnostic branch")
        restore = t02._check_restore(restore_receipt, snapshot=registered["public"])
        if self._last_restore != restore:
            raise ValueError("Diagnostic branch was not preceded by this exact live restore")

        branch_id = branch.get("branch_id")
        if branch_id not in {"known_support", "full_history"}:
            raise ValueError(f"Unsupported recoverability branch: {branch_id}")
        candidate_ids = branch.get("candidate_ids")
        if not isinstance(candidate_ids, list) or any(
            not isinstance(item, str) or not item for item in candidate_ids
        ):
            raise ValueError("Diagnostic candidate_ids must be a list of nonempty strings")
        support = self._diagnostic_support.get(state["state_id"])
        if branch_id == "known_support":
            if support is None or support["prepared"].get("receipt", {}).get("status") != "admitted":
                raise ValueError("Known-support branch lacks an admitted reviewed support set")
            if candidate_ids != support["prepared"].get("candidate_ids"):
                raise ValueError("Known-support candidate ids changed after review")
            if branch.get("support_receipt") != support["prepared"].get("receipt"):
                raise ValueError("Known-support source/B0 receipt changed after review")
            if branch.get("continuation_memory_policy") != "frozen_T02_policy":
                raise ValueError("Known-support continuation must use the frozen T02 policy")
        elif candidate_ids:
            raise ValueError("Full-history diagnostic does not accept support candidate ids")
        elif branch.get("continuation_memory_policy") != recoverability.FULL_POLICY:
            raise ValueError("Full-history continuation policy changed")

        attempt_key = (registered["public"]["snapshot_id"], branch_id)
        if attempt_key in self._diagnostic_attempt_keys:
            raise ValueError("The same snapshot diagnostic branch cannot be retried automatically")
        if self.diagnostic_branch_attempts >= MAX_DIAGNOSTIC_ATTEMPTS:
            raise ValueError("Recoverability diagnostic branch budget exhausted")
        self._diagnostic_attempt_keys.add(attempt_key)
        self.diagnostic_branch_attempts += 1

        attempt_path = self._attempt_path(state["state_id"], branch_id)
        attempt_receipt = {
            "schema": ATTEMPT_ARTIFACT_SCHEMA,
            "diagnostic_only": True,
            "state_id": state["state_id"],
            "branch_id": branch_id,
            "snapshot_id": registered["public"]["snapshot_id"],
            "component_digests": copy.deepcopy(registered["public"]["component_digests"]),
            "restore_receipt_sha256": t02._digest(restore),
            "frozen_policy_sha256": self.policy_sha256,
            "candidate_ids": list(candidate_ids),
            "execution_receipt": recoverability._branch_receipt(branch),
            "execution_status": "attempted",
            "cost_finalized": False,
        }
        t02_bfcl._write_json(attempt_path, attempt_receipt)

        environment, actor = registered["environment"], registered["actor"]
        intervention_turn = environment.turn_index
        before_calls = getattr(actor, "total_generation_calls", None)
        if type(before_calls) is not int:
            raise TypeError("T02 actor must expose cumulative total_generation_calls")
        started = time.perf_counter()
        failure: BaseException | None = None
        cost: dict[str, Any]
        try:
            if branch_id == "known_support":
                response = recoverability_runtime.submit_support(
                    actor,
                    support["annotations"],
                    expected_candidate_ids=list(candidate_ids),
                )
            else:
                response = recoverability_runtime.submit_full_history(actor)
            environment.commit_response(response)
            while not environment.finished:
                response = actor.generate(environment.next_payload())
                environment.commit_response(response)
            outcomes = environment.official_outcomes(intervention_turn)
        except BaseException as error:
            failure = error
            raise
        finally:
            elapsed = time.perf_counter() - started
            after_calls = getattr(actor, "total_generation_calls", None)
            if type(after_calls) is not int or after_calls < before_calls:
                raise RuntimeError("T02 actor cumulative generation count regressed")
            branch_calls = after_calls - before_calls
            self.diagnostic_generation_calls += branch_calls
            self.diagnostic_wall_seconds += elapsed
            self._last_restore = None
            cost = {
                "measurement_scope": "diagnostic_branch_execution",
                "generation_calls": branch_calls,
                "wall_seconds": elapsed,
                "diagnostic_branch_attempts_total": self.diagnostic_branch_attempts,
                "diagnostic_generation_calls_total": self.diagnostic_generation_calls,
                "diagnostic_wall_seconds_total": self.diagnostic_wall_seconds,
                "parent_t02_branch_generation_calls": self.branch_generation_calls,
            }
            attempt_receipt.update(
                execution_status="failed" if failure is not None else "complete",
                cost_finalized=True,
                cost=cost,
            )
            if failure is not None:
                attempt_receipt["error"] = {
                    "type": type(failure).__name__,
                    "message": str(failure),
                }
            t02_bfcl._write_json(attempt_path, attempt_receipt)
        artifact = {
            "schema": DIAGNOSTIC_OUTCOME_SCHEMA,
            "diagnostic_only": True,
            "state_id": state["state_id"],
            "branch_id": branch_id,
            "candidate_ids": list(candidate_ids),
            "snapshot_id": registered["public"]["snapshot_id"],
            "component_digests": copy.deepcopy(registered["public"]["component_digests"]),
            "frozen_policy_sha256": self.policy_sha256,
            "official_checker": {
                "turn_prefix": "bfcl_eval.eval_checker.multi_turn_eval.multi_turn_checker.multi_turn_checker",
                "task": "bfcl_eval.eval_checker.eval_runner._evaluate_single_multi_turn_entry",
            },
            "outcomes": outcomes,
            "model_result": t02_bfcl._json_copy(environment.all_model_response),
            "cost": cost,
        }
        artifact_path = (
            self.artifact_dir / t02_bfcl._safe_name(state["state_id"])
            / "recoverability" / f"{branch_id}.json"
        )
        artifact_sha256 = t02_bfcl._write_json(artifact_path, artifact)
        return {
            "schema": recoverability.BRANCH_SCHEMA,
            "state_id": state["state_id"],
            "branch_id": branch_id,
            "snapshot_id": registered["public"]["snapshot_id"],
            "restore_receipt_sha256": t02._digest(restore),
            "frozen_policy_sha256": self.policy_sha256,
            "execution_status": "complete",
            "execution_receipt": recoverability._branch_receipt(branch),
            "official_outcome": {
                "source": "official",
                "scorer": "BFCL_v4_multi_turn_checker",
                "artifact": str(artifact_path),
                "artifact_sha256": artifact_sha256,
                "turn_success": outcomes["turn_success"],
                "task_success": outcomes["task_success"],
            },
            "cost": cost,
        }

    def review_packet_for_state(self, state: Mapping[str, Any]) -> dict[str, Any]:
        """Export observed review data and an official prior-prefix receipt."""
        snapshot = self.capture_state(state, frozen_policy=self.frozen_policy)
        self.restore_state(snapshot, frozen_policy=self.frozen_policy)
        registered = self._require_active_state(state)
        environment, actor = registered["environment"], registered["actor"]
        try:
            packet = recoverability_runtime.review_packet(actor, state)
            current_responses = environment._current_turn_response
            if not isinstance(current_responses, list):
                raise RuntimeError("BFCL current-turn response state is invalid")
            current_turn_step = len(current_responses)
            previous_success = environment.previous_turn_valid()
            prefix_artifact = {
                "schema": PREFIX_ARTIFACT_SCHEMA,
                "source": "official",
                "state_id": state["state_id"],
                "all_previous_turns_success": previous_success,
                "current_turn_step": current_turn_step,
                "decision_key": state["decision_key"],
                "snapshot_id": snapshot["snapshot_id"],
                "component_digests": copy.deepcopy(snapshot["component_digests"]),
                "environment_turn_index": environment._turn_index,
                "environment_global_step": environment._global_step,
                "environment_tool_step_count": environment._turn_step_count,
                "current_turn_response_empty": not current_responses,
                "official_checker": (
                    "bfcl_eval.eval_checker.multi_turn_eval.multi_turn_checker."
                    "multi_turn_checker"
                ),
                "model_input_scope": "held_decision_observed_prefix_only",
                "future_ground_truth_entered_actor_input": False,
            }
            artifact_path = (
                self.artifact_dir / t02_bfcl._safe_name(state["state_id"])
                / "recoverability" / "prefix-validity.json"
            )
            artifact_sha256 = t02_bfcl._write_json(artifact_path, prefix_artifact)
            prefix = {
                key: copy.deepcopy(prefix_artifact[key])
                for key in (
                    "source", "all_previous_turns_success", "current_turn_step",
                    "decision_key", "snapshot_id", "component_digests",
                )
            }
            prefix.update(
                artifact=str(artifact_path), artifact_sha256=artifact_sha256,
                official_checker=prefix_artifact["official_checker"],
                current_turn_response_empty=prefix_artifact["current_turn_response_empty"],
            )
            value = t02._json_copy(packet, label="recoverability review packet")
            value["snapshot_id"] = snapshot["snapshot_id"]
            value["component_digests"] = copy.deepcopy(snapshot["component_digests"])
            value["prefix_validity"] = prefix
            return value
        finally:
            # previous_turn_valid may fill a checker cache; exact restore removes
            # that review-only mutation before execution.
            self.restore_state(snapshot, frozen_policy=self.frozen_policy)

    def diagnostic_cost_summary(self) -> dict[str, Any]:
        return {
            "max_retained_states": MAX_RETAINED_STATES,
            "retained_states": len(self._diagnostic_retained),
            "max_branch_attempts": MAX_DIAGNOSTIC_ATTEMPTS,
            "branch_attempts": self.diagnostic_branch_attempts,
            "generation_calls": self.diagnostic_generation_calls,
            "wall_seconds": self.diagnostic_wall_seconds,
            "parent_t02_branch_generation_calls": self.branch_generation_calls,
            "unfinished_attempt_receipts": sum(
                1 for path in self._attempt_receipt_paths()
                if json.loads(path.read_text(encoding="utf-8")).get("cost_finalized") is not True
            ),
        }

    def _attempt_path(self, state_id: str, branch_id: str) -> Path:
        return (
            self.artifact_dir / t02_bfcl._safe_name(state_id)
            / "recoverability" / "attempts" / f"{branch_id}.json"
        )

    def _attempt_receipt_paths(self) -> list[Path]:
        if not self.artifact_dir.exists():
            return []
        return sorted(self.artifact_dir.glob("*/recoverability/attempts/*.json"))

    def _load_attempt_receipts(self) -> None:
        for path in self._attempt_receipt_paths():
            value = json.loads(path.read_text(encoding="utf-8"))
            if value.get("schema") != ATTEMPT_ARTIFACT_SCHEMA:
                raise ValueError(f"Invalid recoverability attempt receipt: {path}")
            key = (value.get("snapshot_id"), value.get("branch_id"))
            if not all(isinstance(item, str) and item for item in key):
                raise ValueError(f"Incomplete recoverability attempt identity: {path}")
            if key in self._diagnostic_attempt_keys:
                raise ValueError(f"Duplicate recoverability attempt receipt: {path}")
            self._diagnostic_attempt_keys.add(key)
            self.diagnostic_branch_attempts += 1
            cost = value.get("cost", {})
            calls, seconds = cost.get("generation_calls", 0), cost.get("wall_seconds", 0.0)
            if type(calls) is not int or calls < 0 or type(seconds) not in (int, float) or seconds < 0:
                raise ValueError(f"Invalid recoverability attempt cost: {path}")
            self.diagnostic_generation_calls += calls
            self.diagnostic_wall_seconds += float(seconds)

    def _require_active_state(self, state: Mapping[str, Any]) -> dict[str, Any]:
        state_id = state.get("state_id")
        if self._active is None or self._active["state"]["state_id"] != state_id:
            raise ValueError("Diagnostic operation does not target the active live BFCL state")
        if state_id not in self._diagnostic_retained:
            raise ValueError("Diagnostic operation requires an explicitly retained state")
        return self._active

    def close(self) -> None:
        self._diagnostic_retained.clear()
        self._diagnostic_deferred_releases.clear()
        self._diagnostic_support.clear()
        self._last_restore = None
        super().close()
