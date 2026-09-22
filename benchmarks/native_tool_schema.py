"""Explicit tool-prologue schema variants for native C2KV cells.

The native serving API renders the actor's tool prologue under one declared
schema mode (``benchmarks.memory_runtime.tokenization.TOOL_SCHEMA_MODES``):
``sglang-full`` is the release default and matches the Full SGLang actor;
``raw`` passes the client tool JSON through unchanged (the pre-2026-09-21
native prologue).  A variant cell only ever adds an explicit non-default
mode; the historical cells keep their identity and commands.
"""
from dataclasses import dataclass

from .arms import Arm

NATIVE_TOOL_SCHEMAS = ("sglang-full", "raw")
DEFAULT_NATIVE_TOOL_SCHEMA = "sglang-full"


@dataclass(frozen=True)
class NativeToolSchema:
    schema: str

    def __post_init__(self):
        if self.schema not in NATIVE_TOOL_SCHEMAS:
            raise ValueError(f"Native tool schema must be one of {NATIVE_TOOL_SCHEMAS}")

    @property
    def is_default(self) -> bool:
        return self.schema == DEFAULT_NATIVE_TOOL_SCHEMA

    def validate_arm(self, arm: Arm) -> None:
        if not arm.native_controller:
            raise ValueError(
                f"arm {arm.name!r} is not a native controller arm; "
                "its tool prologue is rendered by the engine (--c2kv-tools-dump)")

    def cell_suffix(self) -> str:
        return "" if self.is_default else f"__toolschema-{self.schema}"

    def cli_args(self) -> list[str]:
        return ["--tool-schema", self.schema]


def parse_native_tool_schema(value: str) -> tuple[str, str]:
    arm, separator, schema = value.partition("=")
    if not separator or not arm or not schema:
        raise ValueError("Native tool schema must be ARM=SCHEMA")
    return arm, NativeToolSchema(schema).schema
