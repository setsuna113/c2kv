"""Deterministic DDP training and resumable, optimizer-boundary checkpoints."""
from __future__ import annotations

import contextlib
import json
import math
import os
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

TRAINING_PROFILE = "history-event-base-query-v1"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def epoch_batches(group_ids, *, seed, epoch, batch_size, world_size, rank):
    """Shuffle sessions, preserving decision chronology within each session.

    No example is repeated to pad a distributed sampler. At most world_size-1
    final examples are omitted per epoch, with the count reported by callers.
    Every rank has the same number of forward/backward calls.
    """
    if batch_size < 1 or world_size < 1 or not 0 <= rank < world_size:
        raise ValueError("Invalid distributed batch geometry")
    groups = defaultdict(list)
    for index, group in enumerate(group_ids):
        groups[group].append(index)
    keys = list(groups)
    random.Random(seed + epoch).shuffle(keys)
    order = [index for key in keys for index in groups[key]]
    usable = len(order) - len(order) % world_size
    batches = []
    for start in range(0, usable, batch_size * world_size):
        global_batch = order[start:min(usable, start + batch_size * world_size)]
        local_size = len(global_batch) // world_size
        batches.append(global_batch[rank * local_size:(rank + 1) * local_size])
    return batches, len(order) - usable


def rng_state():
    state = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def barrier():
    if dist.is_initialized():
        dist.barrier()


def save_checkpoint(output_dir, wrapper, tokenizer, optimizer, scheduler, state, contract, *, rank, world_size):
    """Write a checkpoint atomically only after an entire optimizer update."""
    destination = Path(output_dir) / f"checkpoint-{state['global_step']}"
    pending = destination.with_name(destination.name + ".pending")
    if rank == 0:
        if destination.exists() or pending.exists():
            raise FileExistsError(f"Checkpoint already exists: {destination} (or its .pending directory)")
        pending.mkdir(parents=True)
        wrapper.base_model.save_pretrained(pending, safe_serialization=True)
        tokenizer.save_pretrained(pending)
        master_parameters = {name: parameter.detach().cpu() for name, parameter in wrapper.base_model.named_parameters()
                             if parameter.requires_grad}
        torch.save({"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                    "master_parameters": master_parameters}, pending / "optimizer.pt")
        (pending / "trainer_state.json").write_text(json.dumps({**state, "contract": contract, "world_size": world_size,
            "parameter_version": wrapper.parameter_version, "training_profile": contract["profile"]}, indent=2) + "\n", encoding="utf-8")
    barrier()
    torch.save(rng_state(), pending / f"rng-rank-{rank}.pt")
    barrier()
    if rank == 0:
        pending.rename(destination)
        pointer = {"checkpoint": destination.name, "global_step": state["global_step"], "completed": state.get("completed", False)}
        (Path(output_dir) / "latest.json").write_text(json.dumps(pointer, indent=2) + "\n", encoding="utf-8")
    barrier()
    return destination


def read_resume(checkpoint, contract, *, rank, world_size, optimizer, scheduler, wrapper):
    path = Path(checkpoint)
    state = json.loads((path / "trainer_state.json").read_text(encoding="utf-8"))
    if state["world_size"] != world_size or state["contract"] != contract:
        raise ValueError("Resume requires the same corpus, arm, seed, batch geometry, optimizer and planned schedule")
    if state["training_profile"] != contract["profile"]:
        raise ValueError("Checkpoint has a different training/serving profile")
    # Files are local checkpoints produced by this entry point, not external pickle data.
    saved = torch.load(path / "optimizer.pt", map_location="cpu", weights_only=False)
    expected_names = {name for name, parameter in wrapper.base_model.named_parameters() if parameter.requires_grad}
    if set(saved["master_parameters"]) != expected_names:
        raise ValueError("Checkpoint trainable parameter set differs from the current runtime")
    with torch.no_grad():
        for name, parameter in wrapper.base_model.named_parameters():
            if parameter.requires_grad:
                parameter.copy_(saved["master_parameters"][name])
    optimizer.load_state_dict(saved["optimizer"])
    scheduler.load_state_dict(saved["scheduler"])
    wrapper.restore_parameter_version(state["parameter_version"])
    restore_rng(torch.load(path / f"rng-rank-{rank}.pt", map_location="cpu", weights_only=False))
    return state


def train(wrapper, tokenizer, corpus, args, *, rank=0, world_size=1, device=None, corpus_identity=""):
    """Normalize each global optimizer update by its actual decision weight.

    Runtime loss is a weighted mean within one microbatch. Multiplication by
    its weight and division by the all-rank update weight makes accumulation,
    uneven final batches and DDP gradient averaging equivalent to one batch.
    """
    if not len(corpus):
        raise ValueError("Prepared training corpus is empty")
    device = device or next(wrapper.parameters()).device
    first_batches, omitted = epoch_batches(corpus.group_ids, seed=args.seed, epoch=0,
        batch_size=args.per_device_batch_size, world_size=world_size, rank=rank)
    if not first_batches:
        raise ValueError("Corpus must contain at least one decision per rank")
    updates_per_epoch = math.ceil(len(first_batches) / args.gradient_accumulation_steps)
    planned_steps = updates_per_epoch * args.num_train_epochs
    if args.max_steps > 0:
        planned_steps = min(planned_steps, args.max_steps)
    warmup_steps = math.ceil(planned_steps * args.warmup_ratio)
    trainable = [parameter for parameter in wrapper.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=args.weight_decay)

    def lr_factor(step):
        if warmup_steps and step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, planned_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    profile = getattr(
        getattr(wrapper.base_model, "config", None),
        "history_memory_training_profile",
        TRAINING_PROFILE,
    )
    if not isinstance(profile, str) or not profile:
        raise ValueError("Model config must declare a nonempty training profile")
    contract = {name: getattr(args, name) for name in ("arm", "seed", "per_device_batch_size", "gradient_accumulation_steps",
        "num_train_epochs", "learning_rate", "weight_decay", "warmup_ratio", "max_grad_norm", "bf16", "attn_impl")}
    contract.update(corpus_identity=corpus_identity, planned_steps=planned_steps, profile=profile)
    extensions = getattr(args, "training_contract", None)
    if extensions is not None:
        if not isinstance(extensions, dict):
            raise TypeError("args.training_contract must be a dictionary")
        overlap = sorted(contract.keys() & extensions.keys())
        if overlap:
            raise ValueError(
                f"Training contract extensions overlap built-in fields: {overlap}"
            )
        # Check this before the first update rather than discovering an
        # unserializable provenance value while writing a checkpoint.
        json.dumps(extensions, allow_nan=False)
        contract.update(extensions)
    state = {"global_step": 0, "epoch": 0, "next_microbatch": 0, "counters": {}, "completed": False}
    if args.resume_from_checkpoint:
        state = read_resume(args.resume_from_checkpoint, contract, rank=rank, world_size=world_size,
            optimizer=optimizer, scheduler=scheduler, wrapper=wrapper)
    elif Path(args.output_dir).exists() and any(Path(args.output_dir).iterdir()):
        raise FileExistsError("Nonempty output_dir requires --resume_from_checkpoint")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    ddp = wrapper
    if world_size > 1:
        ddp = torch.nn.parallel.DistributedDataParallel(wrapper,
            device_ids=[device.index] if device.type == "cuda" else None, broadcast_buffers=False,
            find_unused_parameters=False)
    wrapper.train()
    wandb_run = None
    if rank == 0 and args.wandb_mode != "disabled":
        import wandb
        run_id = args.wandb_run_id or state.get("wandb_run_id")
        wandb_run = wandb.init(project=args.wandb_project, entity=args.wandb_entity,
            name=getattr(args, "wandb_name", None) or f"history_{args.arm}_s{args.seed}",
            config=contract, mode=args.wandb_mode,
            id=run_id, resume="allow" if run_id else None,
            dir=args.output_dir)
        state["wandb_run_id"] = wandb_run.id
    if rank == 0:
        print(json.dumps({"event": "training_start", **contract, "world_size": world_size,
            "trainable_parameters": sum(p.numel() for p in trainable), "decisions": len(corpus),
            "omitted_tail_decisions_per_epoch": omitted}), flush=True)
    last_saved = state["global_step"] if args.resume_from_checkpoint else -1
    for epoch in range(state["epoch"], args.num_train_epochs):
        batches, _ = epoch_batches(corpus.group_ids, seed=args.seed, epoch=epoch,
            batch_size=args.per_device_batch_size, world_size=world_size, rank=rank)
        cursor = state["next_microbatch"] if epoch == state["epoch"] else 0
        while cursor < len(batches) and state["global_step"] < planned_steps:
            stop = min(cursor + args.gradient_accumulation_steps, len(batches))
            window = [[corpus[index] for index in indices] for indices in batches[cursor:stop]]
            local_weight = sum(record.weight for batch in window for record in batch)
            denominator = torch.tensor(local_weight, dtype=torch.float64, device=device)
            if world_size > 1:
                dist.all_reduce(denominator)
            optimizer.zero_grad(set_to_none=True)
            loss_sum = torch.zeros((), dtype=torch.float64, device=device)
            statistics = defaultdict(int)
            started = time.monotonic()
            for micro_index, batch in enumerate(window):
                synchronization = ddp.no_sync() if world_size > 1 and micro_index < len(window) - 1 else contextlib.nullcontext()
                with synchronization:
                    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=args.bf16):
                        result = ddp(batch)
                        weight = sum(record.weight for record in batch)
                        scaled = result["loss"] * (weight * world_size / denominator.item())
                    scaled.backward()
                loss_sum += result["loss"].detach().double() * weight
                for key, value in result["stats"].items():
                    statistics[key] += value
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm, error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            wrapper.advance_parameter_version()
            state["global_step"] += 1
            cursor = stop
            state["epoch"] = epoch if cursor < len(batches) else epoch + 1
            state["next_microbatch"] = cursor if cursor < len(batches) else 0
            keys = sorted(statistics)
            counts = torch.tensor([statistics[key] for key in keys], dtype=torch.int64, device=device)
            if world_size > 1:
                dist.all_reduce(loss_sum)
                dist.all_reduce(counts)
            for key, value in zip(keys, counts.tolist()):
                state["counters"][key] = state["counters"].get(key, 0) + value
            record = {"step": state["global_step"], "epoch": epoch, "loss": (loss_sum / denominator).item(),
                "learning_rate": scheduler.get_last_lr()[0], "grad_norm": float(grad_norm),
                "step_seconds": time.monotonic() - started, **dict(zip(keys, counts.tolist())),
                "cumulative": dict(state["counters"])}
            if device.type == "cuda":
                record["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
            if rank == 0:
                with (Path(args.output_dir) / "train_history.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record) + "\n")
                if state["global_step"] % args.logging_steps == 0:
                    print(json.dumps(record), flush=True)
                if wandb_run is not None:
                    wandb_run.log(record, step=state["global_step"])
            state["completed"] = state["global_step"] >= planned_steps
            stopping = args.stop_after_steps > 0 and state["global_step"] >= args.stop_after_steps
            if state["global_step"] % args.save_steps == 0 or state["completed"] or stopping:
                save_checkpoint(args.output_dir, wrapper, tokenizer, optimizer, scheduler, state, contract,
                    rank=rank, world_size=world_size)
                last_saved = state["global_step"]
            if stopping or state["completed"]:
                break
        if state["completed"] or (args.stop_after_steps > 0 and state["global_step"] >= args.stop_after_steps):
            break
    if state["global_step"] != last_saved:
        save_checkpoint(args.output_dir, wrapper, tokenizer, optimizer, scheduler, state, contract,
            rank=rank, world_size=world_size)
    if wandb_run is not None:
        wandb_run.finish()
    return state
