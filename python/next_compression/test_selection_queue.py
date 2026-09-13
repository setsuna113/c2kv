"""Check the finite queue through real gist export, with generation stubbed."""
import importlib.util
import json
from pathlib import Path
import shutil

from next_compression.common import sha256_file
from next_compression.test_export import _trained_checkpoint
from next_compression.test_selection import rows


def test_queue_exports_ranked_candidate_and_final(tmp_path, monkeypatch):
    agent_root = Path(__file__).resolve().parents[2] / "agent"
    monkeypatch.syspath_prepend(str(agent_root))
    spec = importlib.util.spec_from_file_location("select_next_for_test", agent_root / "select_next_checkpoints.py")
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    trained = tmp_path / "trained"
    first = _trained_checkpoint(trained / "H3")
    final = first.with_name("checkpoint-2")
    shutil.copytree(first, final)
    dev = tmp_path / "dev" / "H3"
    dev.mkdir(parents=True)
    (dev / "manifest.json").write_text(json.dumps({"purpose": "checkpoint_selection_dev"}))
    import launch_next_compression
    monkeypatch.setattr(launch_next_compression, "check_devices", lambda devices: None)

    def fake_evaluate(command, **kwargs):
        checkpoint = Path(command[command.index("--checkpoint") + 1])
        output = Path(command[command.index("--output") + 1])
        result = {"status": "completed", "checkpoint": str(checkpoint.resolve()), "variant": "H3",
                  "config_sha256": sha256_file(checkpoint / "config.json"),
                  "eval_manifest_sha256": sha256_file(dev / "manifest.json"),
                  "protocol": {"ratios": [8, 12], "max_new_tokens": 512,
                               "max_decisions_per_ratio": 2, "decode_strategy": "incremental", "sampling": "greedy"},
                  "records": rows(checkpoint.name == "checkpoint-1")}
        output.write_text(json.dumps(result), encoding="utf-8")

    monkeypatch.setattr(entry.subprocess, "run", fake_evaluate)
    output = tmp_path / "selection"
    assert entry.main(["--checkpoint-root", str(trained), "--dev-root", str(dev.parent),
                       "--output-root", str(output), "--variants", "H3", "--decisions", "2", "--run"]) == 0
    receipt = json.loads((output / "return/RETURN.json").read_text())
    assert {item["package"] for item in receipt["candidates"]} == {"H3/checkpoint-1", "H3/checkpoint-2"}
    assert all((output / "return" / item["package"] / "c2kv-gist.safetensors").is_file()
               for item in receipt["candidates"])
    assert not list((output / "return").rglob("optimizer.pt"))
