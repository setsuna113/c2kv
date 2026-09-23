"""Device-neutral budget interface for registered history-KV arms.

The budget changes capacity, not the selector, backend, or session protocol.
Text summarizers and learned gist ratios have different contracts and do not
implement this interface.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
import re

if __package__:
    from .arms import Arm, history_kv_spec
else:
    from arms import Arm, history_kv_spec


@dataclass(frozen=True)
class HistoryKVBudget:
    target_tokens: int | None = None
    retention_ratio: float | None = None

    def __post_init__(self):
        if (self.target_tokens is None) == (self.retention_ratio is None):
            raise ValueError("History-KV budget requires exactly one of target_tokens "
                             "(positive integer) or retention_ratio")
        if self.target_tokens is not None:
            if type(self.target_tokens) is not int or self.target_tokens < 1:
                raise ValueError("History-KV budget must be a positive integer")
        elif (isinstance(self.retention_ratio, bool)
              or not isinstance(self.retention_ratio, (int, float))
              or not 0 < self.retention_ratio <= 1
              or not math.isfinite(float(self.retention_ratio))):
            raise ValueError("History-KV retention_ratio must be finite and in (0, 1]")
        else:
            object.__setattr__(self, "retention_ratio", float(self.retention_ratio))

    def apply(self, arm: Arm) -> Arm:
        """Resolve an independent arm without mutating the registry."""
        spec = history_kv_spec(arm)
        if spec is None:
            raise ValueError(f"arm {arm.name!r} does not support a history-KV budget")
        resolved = replace(arm, history_kv={
            **spec, "target_tokens": self.target_tokens,
            "retention_ratio": self.retention_ratio,
        })
        resolved.validate()
        return resolved

    def variant_name(self, arm_name: str) -> str:
        """Experiment identity; the underlying arm name remains unchanged."""
        # Registry names can encode their default retention (e.g. _r25).
        # A budget variant must not keep that stale number in its identity.
        base = (re.sub(r"_r[0-9]+(?=_persistent$|$)", "", arm_name)
                if arm_name.startswith("history_kv_") else arm_name)
        if self.target_tokens is not None:
            return f"{base}_b{self.target_tokens}"
        ratio = repr(float(self.retention_ratio)).replace(".", "p")
        return f"{base}_r{ratio}"

    def cli_args(self) -> list[str]:
        if self.target_tokens is not None:
            return ["--history-kv-target-tokens", str(self.target_tokens)]
        return ["--history-kv-retention-ratio", str(self.retention_ratio)]


def parse_history_kv_budget(value: str) -> tuple[str, int]:
    """Parse an explicit registry arm and capacity, e.g. ``commitkv=768``."""
    arm_name, separator, tokens = value.partition("=")
    if not separator or not arm_name or not tokens.isascii() or not tokens.isdecimal():
        raise ValueError("History-KV budget must be ARM=POSITIVE_TOKENS")
    return arm_name, HistoryKVBudget(int(tokens)).target_tokens


def parse_history_kv_retention(value: str) -> tuple[str, float]:
    """Parse an explicit registry arm and retained fraction, e.g. ``commitkv=0.5``."""
    arm_name, separator, ratio_text = value.partition("=")
    if (not separator or not arm_name or not ratio_text or not ratio_text.isascii()
            or ratio_text.strip() != ratio_text):
        raise ValueError("History-KV retention must be ARM=RATIO in (0, 1]")
    try:
        ratio = float(ratio_text)
    except ValueError as error:
        raise ValueError("History-KV retention must be ARM=RATIO in (0, 1]") from error
    return arm_name, HistoryKVBudget(retention_ratio=ratio).retention_ratio
