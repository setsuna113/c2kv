"""A two-process CPU DDP update must match a single-process global batch."""
import importlib.util
import os
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
import torch.distributed as dist
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from history_memory.runtime import HistoryMemoryModel
from history_memory.training import seed_everything, train


def run_training(output, *, rank=0, world_size=1):
    spec = importlib.util.spec_from_file_location("b_training_entry_ddp", Path(__file__).resolve().parents[2] / "agent/train_history_memory.py")
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    args = entry.arguments(["--cpu_smoke", "--arm", "C", "--output_dir", str(output),
        "--per_device_batch_size", "1" if world_size == 2 else "4",
        "--gradient_accumulation_steps", "2" if world_size == 2 else "1",
        "--num_train_epochs", "1", "--max_steps", "2", "--save_steps", "100"])
    torch.set_num_threads(1)
    seed_everything(42)
    model, tokenizer = entry.build_model(args, torch.device("cpu"))
    train(HistoryMemoryModel(model), tokenizer, entry.smoke_corpus(), args,
          rank=rank, world_size=world_size, device=torch.device("cpu"), corpus_identity="tiny-ddp")


def distributed_worker(rank, rendezvous, output):
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    try:
        run_training(output, rank=rank, world_size=2)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(os.name == "nt" or not dist.is_available() or not dist.is_gloo_available(),
                   reason="Run on Linux: the tested Windows torch 2.9 CPU wheel has no usable Gloo device")
def test_two_process_ddp_matches_single_global_batch(tmp_path):
    from safetensors.torch import load_file
    run_training(tmp_path / "single")
    torch.multiprocessing.spawn(distributed_worker,
        args=((tmp_path / "rendezvous").as_uri(), str(tmp_path / "ddp")), nprocs=2, join=True)
    expected = load_file(tmp_path / "single/checkpoint-2/model.safetensors")
    actual = load_file(tmp_path / "ddp/checkpoint-2/model.safetensors")
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=2e-5, atol=2e-7)
    assert (tmp_path / "ddp/checkpoint-2/rng-rank-1.pt").exists()
