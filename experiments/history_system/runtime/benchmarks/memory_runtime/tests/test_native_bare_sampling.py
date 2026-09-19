"""ACE native bare preserves source sampling without loosening C1 defaults."""
import pytest

from history_memory.sglang_generator import SGLangEventNativeGenerator, SGLangEventNativeError
from benchmarks.memory_runtime.event_native_server import _sampling_params_for_benchmark


def generator(**kwargs):
    return SGLangEventNativeGenerator(
        "http://127.0.0.1:1234", expected_model_path="/checkpoint", model_context=4096,
        max_new_tokens=1000, max_generation_calls=4, max_extraction_calls=20,
        timeout_seconds=30, eos_token_ids=[2], eos_source="test", **kwargs)


def test_ace_source_sampling_is_explicit_and_greedy_default_is_unchanged():
    source = _sampling_params_for_benchmark("acebench", "ac_gist_static")
    assert source == {"temperature": 0.001, "top_p": 1.0}
    actual = generator(sampling_params=source, sampling_profile="acebench-agent-v1")
    assert actual.sampling_params == source
    assert _sampling_params_for_benchmark("acebench") == {"temperature": 0.0, "seed": 0}
    with pytest.raises(ValueError, match="greedy"):
        generator(sampling_params=source)
    with pytest.raises(ValueError, match="no explicit seed"):
        generator(sampling_params=dict(source, seed=42), sampling_profile="acebench-agent-v1")
    with pytest.raises(ValueError, match="temperature"):
        generator(sampling_params={"temperature": 0, "top_p": 1}, sampling_profile="acebench-agent-v1")


def test_ace_profile_refuses_an_old_engine_before_generation(monkeypatch):
    actual = generator(sampling_params={"temperature": 0.001, "top_p": 1},
                       sampling_profile="acebench-agent-v1")
    capability = {"model_binding": {"model_path": "/checkpoint"}, "kv_bytes_per_token": 1}
    response = {"model_path": "/checkpoint", "c2kv_native_packed": capability}
    monkeypatch.setattr(actual, "_read_json", lambda *_args, **_kwargs: (response, 200))
    with pytest.raises(SGLangEventNativeError, match="sampling profile"):
        actual._ensure_model_info()
    capability["sampling_profiles"] = ["greedy-v1", "acebench-agent-v1"]
    actual._ensure_model_info()


def test_hybrid_budget_guard_does_not_apply_to_exact_bare_controllers():
    """The bare ``ac_gist_static`` route uses EventNativeExactController, which declares no
    S0 ``common_raw_prompt_tokens``; the hybrid pre-generation check must report itself as
    not applicable instead of failing every decision (first CUDA run of c2kv_native_r4)."""
    from types import SimpleNamespace
    from benchmarks.memory_runtime.budget_guard import history_budget_receipt

    memory = SimpleNamespace(costs=lambda ratio: {"resident_kv_tokens": 120, "gist_tokens": 10})
    exact_like = SimpleNamespace(policy_config=SimpleNamespace(history_budget_bytes=1000, workspace_budget_bytes=1000),
                                 kv_bytes_per_token=1)
    receipt = history_budget_receipt(memory, {"actual_history_bytes": 20}, exact_like, ratio=4, phase="first_draft")
    assert receipt["status"] == "not_applicable" and receipt["errors"] == []
    # the S0 contract is still enforced when it is declared
    s0_meta = {"common_raw_prompt_tokens": 100, "actual_history_bytes": 20}
    assert history_budget_receipt(memory, s0_meta, exact_like, ratio=4, phase="first_draft")["status"] == "passed"
    bad = dict(s0_meta, actual_history_bytes=99)
    assert history_budget_receipt(memory, bad, exact_like, ratio=4, phase="first_draft")["status"] == "rejected"
    with pytest.raises(ValueError, match="explicit native history budget contract"):
        history_budget_receipt(memory, s0_meta, SimpleNamespace(), ratio=4, phase="first_draft")
