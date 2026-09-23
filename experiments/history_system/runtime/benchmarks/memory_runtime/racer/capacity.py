"""Typed, source-bound rejection of a persistent RACER admission."""
from __future__ import annotations

import copy

from ..always_compress import CapacityInfeasible


CAPACITY_ERROR_CODE = "RACER_CAPACITY_INFEASIBLE"


class HistoryCapacityInfeasible(CapacityInfeasible):
    """A measured capacity failure, with explicit transaction rollback status."""

    def __init__(self, message, receipt):
        super().__init__(message)
        self.receipt = copy.deepcopy(receipt)

    @classmethod
    def from_response(cls, response, *, decision_id, phase):
        error = response.get("error") if isinstance(response, dict) else None
        if not isinstance(error, dict) or error.get("code") != CAPACITY_ERROR_CODE:
            return None
        receipt = error.get("capacity")
        if (not isinstance(receipt, dict)
                or receipt.get("schema") != "racer-capacity-infeasible-v1"
                or receipt.get("decision_id") != decision_id
                or receipt.get("stage") != phase
                or type(receipt.get("rollback_safe")) is not bool
                or any(type(receipt.get(key)) is not int or receipt[key] < 0
                       for key in ("required_tokens", "capacity_tokens"))
                or receipt["required_tokens"] <= receipt["capacity_tokens"]):
            raise ValueError("Invalid RACER capacity rejection receipt")
        return cls(str(error.get("message") or CAPACITY_ERROR_CODE), receipt)

    def can_retain_draft(self, decision_id):
        return (self.receipt["stage"] == "regeneration"
                and self.receipt["decision_id"] == decision_id
                and self.receipt["rollback_safe"] is True)
