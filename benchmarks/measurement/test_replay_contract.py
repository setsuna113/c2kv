import pytest

from benchmarks.measurement.replay import replay_prefixes


@pytest.mark.parametrize("arm", ["agentkv", "commitkv"])
def test_full_teacher_forced_replay_rejects_exact_output_arms_before_io(arm, tmp_path):
    with pytest.raises(ValueError, match="unchanged prior assistant output"):
        replay_prefixes(
            tmp_path / "missing-prefixes.jsonl", "http://127.0.0.1:1", tmp_path / "out.jsonl",
            source_run_id="bfcl_base__full", target_run_id=f"bfcl_base__{arm}",
        )
