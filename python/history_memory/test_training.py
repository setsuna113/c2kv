"""Real optimizer tests for resume, accumulation and distributed row coverage."""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from history_memory.runtime import HistoryMemoryModel
from history_memory.training import epoch_batches, seed_everything, train

spec = importlib.util.spec_from_file_location("history_training_entry", Path(__file__).resolve().parents[2] / "agent/train_history_memory.py")
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)


def execute(path, *, resume=None, stop=-1, batch=2, accumulation=1, bf16=False):
    options = ["--cpu_smoke", "--arm", "C", "--output_dir", str(path), "--num_train_epochs", "2",
               "--per_device_batch_size", str(batch), "--gradient_accumulation_steps", str(accumulation),
               "--max_steps", "4", "--save_steps", "100", "--stop_after_steps", str(stop)]
    if resume:
        options += ["--resume_from_checkpoint", str(resume)]
    args = entry.arguments(options)
    args.bf16 = bf16
    torch.set_num_threads(1)
    seed_everything(args.seed)
    model, tokenizer = entry.build_model(args, torch.device("cpu"))
    if bf16:
        model.to(dtype=torch.bfloat16)
    wrapper = HistoryMemoryModel(model)
    state = train(wrapper, tokenizer, entry.smoke_corpus(), args, device=torch.device("cpu"), corpus_identity="test-corpus")
    return {key: value.detach().clone() for key, value in model.state_dict().items()}, state


@pytest.mark.parametrize("bf16", [False, True])
def test_interrupted_resume_matches_uninterrupted_weights_and_history(tmp_path, bf16):
    expected, expected_state = execute(tmp_path / "continuous", bf16=bf16)
    _, stopped = execute(tmp_path / "resumed", stop=2, bf16=bf16)
    assert stopped["global_step"] == 2 and not stopped["completed"]
    actual, state = execute(tmp_path / "resumed", resume=tmp_path / "resumed/checkpoint-2", bf16=bf16)
    assert state["completed"] and state["global_step"] == 4
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    histories = []
    for directory in ("continuous", "resumed"):
        histories.append([json.loads(line) for line in (tmp_path / directory / "train_history.jsonl").read_text().splitlines()])
    for first, second in zip(*histories):
        assert (first["step"], first["loss"], first["learning_rate"]) == (second["step"], second["loss"], second["learning_rate"])
    assert state["counters"] == expected_state["counters"]
    if bf16:
        saved = torch.load(tmp_path / "resumed/checkpoint-4/optimizer.pt", weights_only=False)
        assert all(value.dtype == torch.float32 for value in saved["master_parameters"].values())
        assert any(not torch.equal(value, value.bfloat16().float()) for value in saved["master_parameters"].values())


def test_accumulation_matches_same_global_decision_batch(tmp_path):
    batched, _ = execute(tmp_path / "batched", batch=2, accumulation=1)
    accumulated, _ = execute(tmp_path / "accumulated", batch=1, accumulation=2)
    for key in batched:
        torch.testing.assert_close(accumulated[key], batched[key], rtol=2e-5, atol=2e-7)


def test_distributed_sampler_has_no_padding_repetition_and_preserves_group_order():
    groups = ["a"] * 5 + ["b"] * 4 + ["c"] * 4
    first, omitted = epoch_batches(groups, seed=42, epoch=0, batch_size=2, world_size=2, rank=0)
    second, _ = epoch_batches(groups, seed=42, epoch=0, batch_size=2, world_size=2, rank=1)
    assert omitted == 1 and [len(batch) for batch in first] == [len(batch) for batch in second]
    all_indices = [index for rank in (first, second) for batch in rank for index in batch]
    assert len(all_indices) == len(set(all_indices)) == 12
    rebuilt = [index for pair in zip(first, second) for batch in pair for index in batch]
    for group in set(groups):
        indices = [index for index in rebuilt if groups[index] == group]
        assert indices == sorted(indices)


def test_resume_rejects_changed_objective_geometry(tmp_path):
    execute(tmp_path / "run", stop=2)
    with pytest.raises(ValueError, match="Resume requires"):
        execute(tmp_path / "run", resume=tmp_path / "run/checkpoint-2", batch=1)
