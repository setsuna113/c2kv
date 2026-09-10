"""Profiler-range contract for opt-in cache-memory annotations."""

from __future__ import annotations

import json

import pytest


torch = pytest.importorskip("torch")

from history_memory.cache_trace import (
    CACHE_MEMORY_OPERATION_KINDS,
    CACHE_MEMORY_RANGE_PREFIX,
    CacheTrace,
)
from history_memory.inference import EventNativeGenerator


def _bare_generator() -> EventNativeGenerator:
    """Build only the two fields used by the annotation seam, without a model."""

    generator = object.__new__(EventNativeGenerator)
    generator._cache_trace = CacheTrace({"attempt_uid": "annotation-attempt"})
    generator._cache_memory_annotations_active = False
    return generator


def _cpu_profiler():
    if torch.profiler.ProfilerActivity.CPU not in torch.profiler.supported_activities():
        pytest.skip("the installed torch profiler does not support CPU activities")
    return torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU],
        profile_memory=True,
    )


def test_enabled_annotation_is_a_chrome_range_around_a_real_allocation(tmp_path):
    generator = _bare_generator()
    with _cpu_profiler() as profiler:
        with generator.cache_memory_annotations():
            with generator._cache_op("extract"):
                allocation = torch.empty(16, dtype=torch.float32)
                assert allocation.numel() == 16
                del allocation

    op = generator._cache_trace.data["ops"][0]
    expected = CACHE_MEMORY_RANGE_PREFIX + op["op_id"]
    assert op["memory_profile_annotation"] == expected
    trace_path = tmp_path / "cache-memory-annotation.json"
    profiler.export_chrome_trace(str(trace_path))
    chrome = json.loads(trace_path.read_text(encoding="utf-8"))
    markers = [
        event
        for event in chrome["traceEvents"]
        if event.get("name") == expected and event.get("ph") == "X"
    ]
    assert len(markers) == 1
    assert any(
        event.get("name") == "[memory]" and event.get("tid") == markers[0]["tid"]
        for event in chrome["traceEvents"]
    )


def test_default_is_off_and_only_physical_operation_kinds_are_marked(tmp_path):
    generator = _bare_generator()
    with _cpu_profiler() as profiler:
        with generator._cache_op("extract"):
            allocation = torch.empty(4, dtype=torch.float32)
            del allocation
        with generator.cache_memory_annotations():
            with generator._cache_op("release"):
                pass

    extract, release = generator._cache_trace.data["ops"]
    assert "memory_profile_annotation" not in extract
    assert "memory_profile_annotation" not in release
    trace_path = tmp_path / "cache-memory-default-off.json"
    profiler.export_chrome_trace(str(trace_path))
    chrome = json.loads(trace_path.read_text(encoding="utf-8"))
    assert not any(
        event.get("name", "").startswith(CACHE_MEMORY_RANGE_PREFIX)
        for event in chrome["traceEvents"]
    )
    assert "extract" in CACHE_MEMORY_OPERATION_KINDS
    assert "release" not in CACHE_MEMORY_OPERATION_KINDS


def test_annotation_context_restores_off_after_failure_and_rejects_reentry():
    generator = _bare_generator()
    with pytest.raises(RuntimeError, match="operation failure"):
        with generator.cache_memory_annotations():
            with pytest.raises(RuntimeError, match="not reentrant"):
                with generator.cache_memory_annotations():
                    pass
            with generator._cache_op("extract"):
                raise RuntimeError("operation failure")

    assert generator._cache_memory_annotations_active is False
    op = generator._cache_trace.data["ops"][0]
    assert op["status"] == "failed"
    assert op["memory_profile_annotation"].startswith(CACHE_MEMORY_RANGE_PREFIX)
    with generator._cache_op("extract"):
        pass
    assert "memory_profile_annotation" not in generator._cache_trace.data["ops"][1]
