import pytest
from benchmarks.model_identity import QWEN3_4B, require_qwen3_4b


def test_served_alias_does_not_substitute_for_loaded_architecture():
    with pytest.raises(ValueError):
        require_qwen3_4b({"model_family": "qwen3-4b", "model_path": "Qwen3-4B"})
    info = {"model_type": "qwen3", "model_path": "/checkpoint-1000",
            "model_dimensions": dict(QWEN3_4B)}
    assert require_qwen3_4b(info)["model_path"] == "/checkpoint-1000"
    info["model_dimensions"]["hidden_size"] = 4096
    with pytest.raises(ValueError, match="hidden_size"):
        require_qwen3_4b(info)
