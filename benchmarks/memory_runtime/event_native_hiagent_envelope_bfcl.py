"""Official BFCL worker for the native HiAgent action-envelope control.

The control keeps native full history and the Subgoal action protocol while
removing both summary rewriting and Trajectory Retrieval.  It therefore does
not claim to reproduce the complete HiAgent system.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator, Mapping, Sequence

from . import event_native_hiagent_bfcl as _base


RUN_SCHEMA = "a-event-native-hiagent-envelope-bfcl-run-v1"
VALIDATION_SCHEMA = "a-event-native-hiagent-envelope-bfcl-official-validation-v1"
ARM = "hiagent_envelope_only_native"
BACKEND = _base.BACKEND
CAPABILITY_FEATURES: tuple[str, ...] = ()
WORKER_MODULE = "benchmarks.memory_runtime.event_native_hiagent_envelope_bfcl"
SHARED_CALL_CAP_PER_TASK = _base.SHARED_CALL_CAP_PER_TASK
CLI_DESCRIPTION = __doc__
_BASE_RUN_BENCHMARKS = _base.run_benchmarks


@contextmanager
def _control_identity() -> Iterator[None]:
    names = (
        "RUN_SCHEMA", "VALIDATION_SCHEMA", "ARM", "CAPABILITY_FEATURES",
        "WORKER_MODULE", "CLI_DESCRIPTION",
    )
    saved = {name: getattr(_base, name) for name in names}
    try:
        for name in names:
            setattr(_base, name, globals()[name])
        yield
    finally:
        for name, value in saved.items():
            setattr(_base, name, value)


def build_run_argv(contract: Mapping[str, Any]) -> list[str]:
    with _control_identity():
        return _base.build_run_argv(contract)


def validate_official_artifacts(contract: Mapping[str, Any]) -> dict[str, Any]:
    with _control_identity():
        return _base.validate_official_artifacts(contract)


def run_benchmarks(argv: Sequence[str]) -> None:
    _BASE_RUN_BENCHMARKS(argv)


def worker(contract_path) -> None:
    with _control_identity():
        original = _base.run_benchmarks
        _base.run_benchmarks = run_benchmarks
        try:
            _base.worker(contract_path)
        finally:
            _base.run_benchmarks = original


def main(argv: list[str] | None = None) -> None:
    with _control_identity():
        _base.main(argv)


if __name__ == "__main__":
    main()
