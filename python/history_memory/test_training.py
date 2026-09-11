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


def test_pretrained_export_keeps_native_eval_metadata(tmp_path):
    """Exercise from_pretrained's config copy using a real tiny base checkpoint."""
    from transformers import Qwen3Config, Qwen3ForCausalLM

    base = tmp_path / "base"
    Qwen3ForCausalLM(Qwen3Config(
        vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        max_position_embeddings=8192, eos_token_id=2,
    )).save_pretrained(base)
    entry.tiny_tokenizer().save_pretrained(base)
    args = entry.arguments([
        "--model_name_or_path", str(base), "--data_path", "unused",
        "--output_dir", str(tmp_path / "train"), "--arm", "B", "--seed", "17",
        "--device", "cpu", "--no-bf16", "--no-gradient_checkpointing",
        "--attn_impl", "eager",
    ])
    model, tokenizer = entry.build_model(args, torch.device("cpu"))
    exported = tmp_path / "exported"
    model.save_pretrained(exported)
    tokenizer.save_pretrained(exported)
    config = json.loads((exported / "config.json").read_text())
    assert config["history_memory_training_profile"] == "history-event-base-query-v1"
    assert config["history_memory_packing_version"] == "history-event-v1"
    assert config["history_memory_raw_layout"] == "event-native-evidence-v1"
    assert config["history_memory_normal_query"] == "base"
    assert config["history_memory_arm"] == "B"
    assert config["history_memory_seed"] == 17
    assert config["history_memory_trainable_dtype"] == "float32"
    assert config["history_memory_supported_ratios"] == [4, 8]


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
