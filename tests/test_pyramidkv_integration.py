from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from controller_runtime.benchmarks.arms import get_arm, history_kv_spec  # noqa: E402
from generality import design  # noqa: E402
from generality.historykv_cell import ARM_OF, target_tokens_for_cell  # noqa: E402


def test_pyramidkv_is_a_first_class_generality_backend():
    assert "pyramidkv" in design.BACKENDS
    assert design.cells() == 72
    for working_point in ("K0", "K2"):
        assert ("pyramidkv", working_point, "recovery_off_same_initial") in ARM_OF
        assert ("pyramidkv", working_point, "compression_full_budget") in ARM_OF


def test_pyramidkv_arm_uses_server_side_history_kv_contract():
    arm_name = ARM_OF[("pyramidkv", "K0", "recovery_off_same_initial")]
    # The generation names are resolved on the NPU checkout; this local
    # controller registry still validates the canonical PyramidKV spec.
    arm = get_arm("history_kv_pyramidkv_r312")
    spec = history_kv_spec(arm)
    assert spec["method"] == "pyramidkv"
    assert arm.history_kv["method"] == "pyramidkv"
    assert arm_name == "gen_pyramidkv_k0"


def test_absolute_budget_arms_are_registered_for_all_history_backends():
    for method in ("h2o", "snapkv_persistent", "pyramidkv"):
        for suffix in ("k0", "k2", "b0", "b2"):
            arm = get_arm(f"gen_{method}_{suffix}")
            spec = history_kv_spec(arm)
            assert spec["method"] == method
            assert spec["backend"] == ("reference_attention" if method == "pyramidkv" else "physical_eviction")
            assert spec["persistent_session"] is True
            assert spec["target_tokens"] == 1  # launch override is explicit
            assert spec["retention_ratio"] is None


def test_session_tracer_does_not_alias_pyramidkv_to_snapkv():
    # Inspect the mapping through the source so this test stays dependency
    # free: constructing SessionTracerTask would load the NPU model.
    source = (ROOT / "generality" / "session_tracer_cell.py").read_text(encoding="utf-8")
    assert '"pyramidkv": "pyramidkv"' in source
    assert 'if self.method == "pyramidkv"' in source
    assert '"reference_attention"' in source


def test_off_driver_uses_resolved_k_or_b_token_budget():
    base = {"cell_id": "x", "condition": "recovery_off_same_initial",
            "budget_tokens": {"K": 768, "R": 1280, "B": 2048}}
    assert target_tokens_for_cell(base) == 768
    base["condition"] = "compression_full_budget"
    assert target_tokens_for_cell(base) == 2048


def test_every_off_launcher_name_resolves_in_registry():
    for (backend, _, _), arm_name in ARM_OF.items():
        spec = history_kv_spec(get_arm(arm_name))
        assert spec["method"] == ("snapkv_persistent" if backend == "snapkv" else backend)
