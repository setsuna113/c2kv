import copy
import json

import pytest

from benchmarks.paper.runner import DEFAULT_CONFIG, prepare


def test_algorithm_switch_requires_fresh_output(tmp_path):
    config = json.loads(DEFAULT_CONFIG.read_text())
    assert config["c1"]["detector"] == "d3_hybrid"
    previous = copy.deepcopy(config)
    previous["c1"]["detector"] = "t02_risk"
    output = tmp_path / "old"
    source = tmp_path / "sglang"
    prepare(previous, output, source)
    before = (output / "config.resolved.json").read_bytes()
    with pytest.raises(RuntimeError, match="different config"):
        prepare(config, output, source)
    assert (output / "config.resolved.json").read_bytes() == before
    plan, _ = prepare(config, tmp_path / "new", source)
    assert any(row["arm"] == "c2kv_c1_t02_r8" for row in plan)
