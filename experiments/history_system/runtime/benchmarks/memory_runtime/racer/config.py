"""Explicit history-backend identity, independent from tool and recovery policy."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

BACKENDS = ("c2kv", "commitkv", "agentkv", "h2o", "snapkv", "pyramidkv", "streamingllm")

# Match the registered persistent proxy arms. A different serving route changes
# the KV algorithm even when its method name and token budget stay the same.
PERSISTENT_METHODS = {
    "commitkv": ("commitkv", "reference_attention"),
    "agentkv": ("agentkv", "reference_attention"),
    "h2o": ("h2o", "physical_eviction"),
    "snapkv": ("snapkv_persistent", "physical_eviction"),
    "pyramidkv": ("pyramidkv", "reference_attention"),
    "streamingllm": ("streamingllm", "physical_eviction"),
}
PERSISTENT_SELECTORS = {"recent_window": 64, "kernel_size": 5,
                        "pooling": "avgpool", "h2o_recent_fraction": 0.5}


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
        if result.backend != "c2kv":
            result.history_spec()
        return result

    @property
    def method(self):
        return PERSISTENT_METHODS[self.backend][0] if self.backend != "c2kv" else "c2kv"

    def history_spec(self, target_tokens=None):
        if self.backend == "c2kv":
            raise ValueError("C2KV does not use a persistent history-KV spec")
        method, backend = PERSISTENT_METHODS[self.backend]
        defaults = {"method": method, "backend": backend,
                    "persistent_session": True, "retention_ratio": None,
                    "target_tokens": self.history_budget_tokens,
                    **PERSISTENT_SELECTORS}
        if not isinstance(self.backend_config, Mapping):
            raise ValueError("RACER backend_config must be a mapping")
        supplied = dict(self.backend_config)
        unknown = set(supplied) - set(defaults)
        if unknown:
            raise ValueError(f"Unknown history backend settings: {sorted(unknown)}")
        for key, value in supplied.items():
            if type(value) is not type(defaults[key]) or value != defaults[key]:
                raise ValueError(f"RACER {self.backend} {key} differs from the registered proxy arm")
        if target_tokens is not None and (type(target_tokens) is not int or target_tokens <= 0):
            raise ValueError("RACER effective history target must be a positive integer")
        defaults["target_tokens"] = self.history_budget_tokens if target_tokens is None else target_tokens
        return defaults

    def receipt(self):
        from dataclasses import asdict
        return {"schema": "racer-backend-v1", **asdict(self),
                "identity": f"racer:{self.backend}:{self.policy}:b{self.history_budget_tokens}",
                "quality_validated": False}
