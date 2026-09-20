"""Explicit capacity variants for native C2KV allocation and recovery."""
from dataclasses import dataclass

from .arms import Arm


@dataclass(frozen=True)
class NativeHistoryBudget:
    target_tokens: int

    def __post_init__(self):
        if type(self.target_tokens) is not int or self.target_tokens < 1:
            raise ValueError("Native history budget must be a positive integer")

    def validate_arm(self, arm: Arm) -> None:
        controller = arm.native_controller or ""
        if controller not in {"c1_t02", "c1_recovery_off"} and not controller.startswith("candidate_"):
            raise ValueError(f"arm {arm.name!r} does not support a native history budget")

    def variant_name(self, arm_name: str) -> str:
        return f"{arm_name}_b{self.target_tokens}"

    def cli_args(self) -> list[str]:
        return ["--history-budget-tokens", str(self.target_tokens)]


def parse_native_history_budget(value: str) -> tuple[str, int]:
    arm, separator, tokens = value.partition("=")
    if not separator or not arm or not tokens.isascii() or not tokens.isdecimal():
        raise ValueError("Native history budget must be ARM=POSITIVE_TOKENS")
    return arm, NativeHistoryBudget(int(tokens)).target_tokens
