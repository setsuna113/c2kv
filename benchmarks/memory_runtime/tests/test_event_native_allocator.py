"""Seam tests for process-local NPU allocator generation measurements."""

from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from benchmarks.memory_runtime.event_native_allocator import (
    MEASUREMENT_CONTRACT,
    NpuAllocatorMeasuredGenerator,
)


class Device:
    type = "npu"
    index = 0

    def __str__(self):
        return "npu:0"


class FakeNpu:
    def __init__(self, calls, *, fail_on=None):
        self.calls = calls
        self.fail_on = fail_on
        self.allocated = iter((100, 140))
        self.reserved = iter((200, 250))

    def _call(self, name, device):
        self.calls.append((name, device))
        if self.fail_on == name:
            raise RuntimeError(f"{name} unavailable")

    def synchronize(self, device):
        self._call("synchronize", device)

    def reset_peak_memory_stats(self, device):
        self._call("reset_peak_memory_stats", device)

    def memory_allocated(self, device):
        self._call("memory_allocated", device)
        return next(self.allocated)

    def memory_reserved(self, device):
        self._call("memory_reserved", device)
        return next(self.reserved)

    def max_memory_allocated(self, device):
        self._call("max_memory_allocated", device)
        return 180

    def max_memory_reserved(self, device):
        self._call("max_memory_reserved", device)
        return 280


class Generator:
    cache_trace_schema = "event-native-cache-trace-v1"
    session_cache_policy = "last-final-view-v1"

    def __init__(self, result=None, error=None, *, device=None, calls=None):
        weight = SimpleNamespace(device=Device() if device is None else device)
        self.runtime = SimpleNamespace(
            base_model=SimpleNamespace(
                model=SimpleNamespace(embed_tokens=SimpleNamespace(weight=weight))
            )
        )
        self.result = result or SimpleNamespace(stats={"existing": "kept"})
        self.error = error
        self.calls = []
        self.timeline = calls
        self.last_generation_trace = {"attempt_uid": "partial-attempt"}

    def generate(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        if self.timeline is not None:
            self.timeline.append(("generate", None))
        if self.error is not None:
            raise self.error
        return self.result

    def decision_scope(self, *, session_id=None):
        return nullcontext(session_id)

    def session_cache_info(self):
        return {"session_id": "delegated"}

    def close_session(self):
        return "closed"


def context(uid="attempt-7"):
    return {"attempt_uid": uid, "session_id": "s", "decision_key": "d"}


def test_success_preserves_result_inputs_kwargs_and_known_numeric_peak():
    timeline = []
    generator = Generator(calls=timeline)
    npu = FakeNpu(timeline)
    wrapper = NpuAllocatorMeasuredGenerator(generator, npu=npu)
    memory = object()
    trace_context = context()

    result = wrapper.generate(
        memory, ratio=8, max_new_tokens=64, trace_context=trace_context
    )

    assert result is generator.result
    assert generator.calls == [
        ((memory,), {
            "ratio": 8,
            "max_new_tokens": 64,
            "trace_context": trace_context,
        })
    ]
    assert generator.calls[0][1]["trace_context"] is trace_context
    assert result.stats["existing"] == "kept"
    assert result.stats["torch_allocator_peak_allocated_bytes"] == 180
    measurement = result.stats["allocator_measurement"]
    assert measurement == {
        "schema": "c2kv-npu-allocator-measurement-v1",
        "backend": "torch_npu.npu",
        "device": "npu:0",
        "attempt_uid": "attempt-7",
        "window": (
            "generator.generate only; excludes model load, controller and "
            "decision_scope exit"
        ),
        "allocated_bytes_before": 100,
        "reserved_bytes_before": 200,
        "allocated_bytes_after": 140,
        "reserved_bytes_after": 250,
        "peak_allocated_bytes": 180,
        "peak_reserved_bytes": 280,
        "status": "ok",
    }
    assert [name for name, _ in timeline] == [
        "synchronize",
        "reset_peak_memory_stats",
        "memory_allocated",
        "memory_reserved",
        "generate",
        "synchronize",
        "memory_allocated",
        "memory_reserved",
        "max_memory_allocated",
        "max_memory_reserved",
    ]


def test_generation_failure_is_not_retried_or_replaced_by_observation():
    failure = RuntimeError("model failed after writing its own partial trace")
    generator = Generator(error=failure)
    npu = FakeNpu([])
    wrapper = NpuAllocatorMeasuredGenerator(generator, npu=npu)

    with pytest.raises(RuntimeError) as captured:
        wrapper.generate(object(), trace_context=context("failed-attempt"))

    assert captured.value is failure
    assert len(generator.calls) == 1
    assert generator.last_generation_trace == {"attempt_uid": "partial-attempt"}
    assert [name for name, _ in npu.calls].count("synchronize") == 1


@pytest.mark.parametrize("fail_on", ["reset_peak_memory_stats", "max_memory_allocated"])
def test_observation_failure_is_unknown_while_model_result_stays_successful(fail_on):
    generator = Generator()
    wrapper = NpuAllocatorMeasuredGenerator(
        generator, npu=FakeNpu([], fail_on=fail_on)
    )

    result = wrapper.generate(object(), trace_context=context("observed-attempt"))

    assert result is generator.result and len(generator.calls) == 1
    assert result.stats["torch_allocator_peak_allocated_bytes"] is None
    measurement = result.stats["allocator_measurement"]
    assert measurement["status"] == "failed"
    assert measurement["attempt_uid"] == "observed-attempt"
    assert measurement["peak_allocated_bytes"] is None
    assert measurement["peak_reserved_bytes"] is None
    assert measurement["error"] == {
        "type": "RuntimeError",
        "message": f"{fail_on} unavailable",
    }


def test_missing_attempt_uid_and_non_npu_device_reject_before_generation():
    npu = FakeNpu([])
    generator = Generator()
    wrapper = NpuAllocatorMeasuredGenerator(generator, npu=npu)
    with pytest.raises(ValueError, match="nonempty attempt_uid"):
        wrapper.generate(object())
    with pytest.raises(ValueError, match="nonempty string"):
        wrapper.generate(object(), trace_context=context(""))
    assert generator.calls == [] and npu.calls == []

    cpu_generator = Generator(device=SimpleNamespace(type="cpu"))
    cpu_wrapper = NpuAllocatorMeasuredGenerator(cpu_generator, npu=npu)
    with pytest.raises(ValueError, match="requires an NPU model device"):
        cpu_wrapper.generate(object(), trace_context=context())
    assert cpu_generator.calls == [] and npu.calls == []


def test_reentrant_generate_on_the_same_wrapper_is_rejected():
    generator = Generator()
    wrapper = NpuAllocatorMeasuredGenerator(generator, npu=FakeNpu([]))
    original = generator.generate
    reentrant_error = None

    def generate(*args, **kwargs):
        nonlocal reentrant_error
        try:
            wrapper.generate(*args, **kwargs)
        except RuntimeError as error:
            reentrant_error = error
        return original(*args, **kwargs)

    generator.generate = generate
    result = wrapper.generate(object(), trace_context=context())

    assert result is generator.result and len(generator.calls) == 1
    assert reentrant_error is not None
    assert "reentrant or concurrent" in str(reentrant_error)


def test_generator_interfaces_and_measurement_contract_are_delegated_or_exposed():
    generator = Generator()
    wrapper = NpuAllocatorMeasuredGenerator(generator, npu=FakeNpu([]))

    assert wrapper.cache_trace_schema == generator.cache_trace_schema
    assert wrapper.session_cache_policy == generator.session_cache_policy
    assert wrapper.runtime is generator.runtime
    assert wrapper.last_generation_trace is generator.last_generation_trace
    with wrapper.decision_scope(session_id="session-a") as session_id:
        assert session_id == "session-a"
    assert wrapper.session_cache_info() == {"session_id": "delegated"}
    assert wrapper.close_session() == "closed"
    assert wrapper.measurement_contract is MEASUREMENT_CONTRACT
    assert wrapper.measurement_contract["counter_scope"] == (
        "this process's NPU caching allocator only"
    )
    assert "resident model weights and cache allocations" in (
        wrapper.measurement_contract["peak_semantics"]
    )
    assert wrapper.measurement_contract["execution_requirement"] == (
        "one serial generator with no competing allocator peak-stat resets"
    )
