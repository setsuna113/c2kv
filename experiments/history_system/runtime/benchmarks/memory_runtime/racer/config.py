"""Explicit history-backend identity, independent from tool and recovery policy."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

BACKENDS = ("c2kv", "commitkv", "h2o", "snapkv", "streamingllm")


@dataclass(frozen=True)
class BackendConfig:
    backend: str
    policy: str
    history_budget_tokens: int
    backend_config: dict = field(default_factory=dict)
    detector_calibration: str = "not_used"
    allocation: str = "backend_native_persistent"

    @classmethod
    def parse(cls, value: Mapping):
        if not isinstance(value, Mapping) or value.get("schema") != "racer-backend-v1":
            raise ValueError("RACER requires schema racer-backend-v1")
        unknown = set(value) - {"schema", *cls.__dataclass_fields__}
        if unknown:
            raise ValueError(f"Unknown RACER backend fields: {sorted(unknown)}")
        from ..candidate_algorithms import ALL_VARIANTS, REPAIR_VARIANTS
        if value.get("backend") not in BACKENDS:
            raise ValueError("Unknown RACER history backend")
        if value.get("policy") not in ("off", "t02", *ALL_VARIANTS):
            raise ValueError("Unknown RACER policy; policy identities are not aliases")
        budget = value.get("history_budget_tokens")
        if type(budget) is not int or budget <= 0:
            raise ValueError("history_budget_tokens must be a positive integer")
        result = cls(**{key: value[key] for key in cls.__dataclass_fields__ if key in value})
        expected = ("not_used" if result.policy in ("off", *REPAIR_VARIANTS) else
                    "reference" if result.backend == "c2kv" else
                    "frozen_c2kv_unvalidated_transfer")
        if result.detector_calibration != expected:
            raise ValueError(f"Detector calibration must be {expected!r}")
        allocation = "c2kv_s0" if result.backend == "c2kv" else "backend_native_persistent"
        if result.allocation != allocation:
            raise ValueError(f"Backend allocation must be {allocation!r}")
        return result

    @property
    def method(self):
        return "snapkv_persistent" if self.backend == "snapkv" else self.backend

    def history_spec(self, target_tokens=None):
        defaults = {"method": self.method, "backend": "reference_attention",
                    "persistent_session": True, "retention_ratio": None,
                    "target_tokens": self.history_budget_tokens,
                    "recent_window": 64, "kernel_size": 5,
                    "pooling": "avgpool", "h2o_recent_fraction": 0.5}
        supplied = dict(self.backend_config)
        unknown = set(supplied) - set(defaults)
        if unknown:
            raise ValueError(f"Unknown history backend settings: {sorted(unknown)}")
        defaults.update(supplied)
        if (defaults["method"] != self.method or not defaults["persistent_session"]
                or defaults["backend"] not in {"reference_attention", "physical_eviction"}
                or defaults["retention_ratio"] is not None):
            raise ValueError("RACER requires its named persistent backend and an absolute budget")
        if self.backend == "commitkv" and defaults["backend"] != "reference_attention":
            raise ValueError("CommitKV requires reference_attention")
        defaults["target_tokens"] = self.history_budget_tokens if target_tokens is None else target_tokens
        return defaults

    def receipt(self):
        from dataclasses import asdict
        return {"schema": "racer-backend-v1", **asdict(self),
                "identity": f"racer:{self.backend}:{self.policy}:b{self.history_budget_tokens}",
                "quality_validated": False}
