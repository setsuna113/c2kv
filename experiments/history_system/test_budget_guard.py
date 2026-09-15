"""Check the final assembled representation, including derived raw content."""
from types import SimpleNamespace

from benchmarks.memory_runtime.budget_guard import history_budget_receipt


class Memory:
    def __init__(self, resident, gist=20):
        self.resident, self.gist = resident, gist

    def costs(self, ratio):
        return {"resident_kv_tokens": self.resident, "gist_tokens": self.gist}


def test_budget_includes_bridge_state_and_lease_tokens_in_assembled_input():
    controller = SimpleNamespace(kv_bytes_per_token=4,
        policy_config=SimpleNamespace(history_budget_bytes=240, workspace_budget_bytes=240))
    base = history_budget_receipt(Memory(150), {"common_raw_prompt_tokens":100, "actual_history_bytes":200},
                                  controller, ratio=4, phase="draft")
    assert base["status"] == "passed" and base["active_raw_and_derived_history_bytes"] == 120
    derived = history_budget_receipt(Memory(170), {"common_raw_prompt_tokens":100, "actual_history_bytes":280},
                                     controller, ratio=4, phase="regeneration")
    assert derived["status"] == "rejected" and derived["active_history_bytes"] == 280
    assert "active_history_exceeds_budget" in derived["errors"]


def test_stale_accounting_is_rejected_before_generation():
    controller = SimpleNamespace(kv_bytes_per_token=4,
        policy_config=SimpleNamespace(history_budget_bytes=400, workspace_budget_bytes=400))
    receipt = history_budget_receipt(Memory(170), {"common_raw_prompt_tokens":100, "actual_history_bytes":200},
                                     controller, ratio=4, phase="draft")
    assert receipt["status"] == "rejected"
    assert receipt["errors"] == ["assembled_history_differs_from_controller_accounting"]
