"""Device-neutral absolute-budget interface for registered history-KV arms.

The budget changes capacity, not the selector, backend, or session protocol.
Text summarizers and learned gist ratios have different contracts and do not
implement this interface.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

if __package__:
    from .arms import Arm, history_kv_spec
else:
    from arms import Arm, history_kv_spec


@dataclass(frozen=True)
class HistoryKVBudget:
    target_tokens: int

    def __post_init__(self):
        if type(self.target_tokens) is not int or self.target_tokens < 1:
            raise ValueError("History-KV budget must be a positive integer")

    def apply(self, arm: Arm) -> Arm:
        """Resolve an independent arm without mutating the registry."""
        spec = history_kv_spec(arm)
        if spec is None:
            raise ValueError(f"arm {arm.name!r} does not support a history-KV budget")
        resolved = replace(arm, history_kv={
            **spec, "target_tokens": self.target_tokens, "retention_ratio": None,
        })
        resolved.validate()
        return resolved

    def variant_name(self, arm_name: str) -> str:
        """Experiment identity; the underlying arm name remains unchanged."""
        return f"{arm_name}_b{self.target_tokens}"

    def cli_args(self) -> list[str]:
        return ["--history-kv-target-tokens", str(self.target_tokens)]


def parse_history_kv_budget(value: str) -> tuple[str, int]:
    """Parse an explicit registry arm and capacity, e.g. ``commitkv=768``."""
    arm_name, separator, tokens = value.partition("=")
    if not separator or not arm_name or not tokens.isascii() or not tokens.isdecimal():
        raise ValueError("History-KV budget must be ARM=POSITIVE_TOKENS")
    return arm_name, HistoryKVBudget(int(tokens)).target_tokens
