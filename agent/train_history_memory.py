#!/usr/bin/env python3
"""Train matched C/B history compression from the original frozen Qwen base."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_path")
    parser.add_argument("--model_name_or_path")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--arm", choices=["C", "B"], required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--per_device_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--num_train_epochs", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--stop_after_steps", type=int, default=-1,
                        help="Controlled interruption after an update; keeps the planned LR schedule for resume tests")
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--resume_from_checkpoint")
    parser.add_argument("--attn_impl", choices=["eager", "sdpa", "flex_attention"], default="sdpa")
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--cpu_smoke", action="store_true", help="Use synthetic short decisions and a random tiny Qwen on CPU")
    parser.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wandb_mode", choices=["disabled", "offline", "online"], default="offline")
    parser.add_argument("--wandb_entity", default="liuyc1025-university-of-cambridge")
    parser.add_argument("--wandb_project", default="c2kv-history-b")
    parser.add_argument("--wandb_run_id")
    # Packing is fixed in the prepared manifest; optional assertions catch a
    # stale launcher rather than silently preparing different training arms.
    parser.add_argument("--ratios")
    for flag in ("max_chunk_tokens", "chunk_overlap", "max_system_tokens", "max_workspace_tokens", "max_target_tokens", "max_chunks"):
        parser.add_argument("--" + flag, type=int)
    result = parser.parse_args(argv)
    for flag in ("per_device_batch_size", "gradient_accumulation_steps", "num_train_epochs", "save_steps", "logging_steps"):
        if getattr(result, flag) < 1:
            parser.error(f"--{flag} must be positive")
    if not 0 <= result.warmup_ratio <= 1 or result.learning_rate <= 0 or result.weight_decay < 0 or result.max_grad_norm <= 0:
        parser.error("Invalid optimizer settings")
    if result.cpu_smoke:
        result.device, result.bf16, result.attn_impl, result.wandb_mode = "cpu", False, "eager", "disabled"
        result.gradient_checkpointing = False
    elif not result.data_path or not result.model_name_or_path:
        parser.error("--data_path and --model_name_or_path are required outside --cpu_smoke")
    return result


def tiny_tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast
    backend = Tokenizer(models.WordLevel({"[UNK]": 1, "[PAD]": 0, "[EOS]": 2, **{f"t{i}": i for i in range(3, 128)}}, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]", eos_token="[EOS]")


def smoke_corpus():
    from history_memory.runtime import PreparedDecision
    from history_memory.packing import EncoderChunk, MemoryView, PackedMemory

    class Corpus:
        group_ids = ["smoke-a"] * 4 + ["smoke-b"] * 4

        def __len__(self):
            return len(self.group_ids)

        def __getitem__(self, index):
            chunk = EncoderChunk("event", 0, (0,), 0, 16, tuple(range(4 + index // 4, 20 + index // 4)))
            memory = PackedMemory(MemoryView(("event",), ("current",)), (20, 21), (22, 23 + index), (1,), (chunk,))
            ratio = 4 if index % 4 < 2 else 8
            return PreparedDecision(memory, (40 + index, 2), ratio, 1.0, f"smoke-{index}")

    return Corpus()


def build_model(args, device):
    import torch
    from transformers import AutoTokenizer
    from models.qwen3.configuration_qwen3 import Qwen3Config
    from models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
    from history_memory.training import TRAINING_PROFILE
    from history_memory.packing import PACKING_VERSION, RAW_LAYOUT_PROFILE
    from history_memory.evidence import EVIDENCE_VERSION

    load_path = args.resume_from_checkpoint or args.model_name_or_path
    if args.cpu_smoke and not args.resume_from_checkpoint:
        config = Qwen3Config(vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2, head_dim=8, max_position_embeddings=8192,
            pad_token_id=None, eos_token_id=2, attention_dropout=0.0,
            gist_type="dynamic-interleave", gist_param="qkv", gist_token_id=2, gist_residual_type="embed-mean")
        config._attn_implementation = "eager"
        model = Qwen3ForCausalLM(config)
        tokenizer = tiny_tokenizer()
    else:
        tokenizer = AutoTokenizer.from_pretrained(load_path, local_files_only=True)
        config = Qwen3Config.from_pretrained(load_path, local_files_only=True)
        if args.resume_from_checkpoint:
            if getattr(config, "history_memory_training_profile", None) != TRAINING_PROFILE:
                raise ValueError("Resume must use a checkpoint from this B/C training entry")
        else:
            raw_config = json.loads((Path(load_path) / "config.json").read_text(encoding="utf-8"))
            if raw_config.get("model_type") != "qwen3" or raw_config.get("architectures") != ["Qwen3ForCausalLM"]:
                raise ValueError("This entry supports the original Qwen3 causal language model")
            if "gist_param" in raw_config or "history_memory_training_profile" in raw_config:
                raise ValueError("Fresh C/B training requires the original base checkpoint; use --resume_from_checkpoint for continuation")
        config.gist_type = "dynamic-interleave"
        config.gist_param = "qkv"
        config.gist_residual_type = "embed-mean"
        config.gist_token_id = tokenizer.eos_token_id
        config._attn_implementation = args.attn_impl
        model = Qwen3ForCausalLM.from_pretrained(load_path, config=config,
            dtype=torch.bfloat16 if args.bf16 else torch.float32, local_files_only=True)
    if not args.resume_from_checkpoint:
        # Explicit initialization also makes fresh random tiny tests follow the
        # exact base-to-gist route used for the real pretrained model.
        with torch.no_grad():
            model.model.gist_embed_tokens.weight.copy_(model.model.embed_tokens.weight[config.gist_token_id:config.gist_token_id + 1])
            for layer in model.model.layers:
                for projection in ("q", "k", "v"):
                    source = getattr(layer.self_attn, f"{projection}_proj")
                    target = getattr(layer.self_attn, f"gist_{projection}_proj")
                    target.load_state_dict(source.state_dict())
    config.history_memory_training_profile = TRAINING_PROFILE
    config.history_memory_packing_version = PACKING_VERSION
    config.history_memory_raw_layout = RAW_LAYOUT_PROFILE
    config.history_memory_evidence_version = EVIDENCE_VERSION
    config.history_memory_normal_query = "base"
    config.history_memory_arm = args.arm
    config.history_memory_seed = args.seed
    config.history_memory_supported_ratios = [4, 8]
    config.history_memory_trainable_dtype = "float32"
    model.to(device)
    if args.gradient_checkpointing:
        from models import gist_utils
        gist_utils.GIST_GRADIENT_CHECKPOINTING = True
        # This module imported the flag by value; set its actual binding too.
        from models.qwen3 import modeling_qwen3
        modeling_qwen3.GIST_GRADIENT_CHECKPOINTING = True
        os.environ["C2KV_GIST_CHECKPOINT_USE_REENTRANT"] = "false"
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return model, tokenizer


def main(argv=None):
    args = arguments(argv)
    import torch
    import torch.distributed as dist
    from history_memory.runtime import HistoryMemoryModel
    from history_memory.training import seed_everything, train

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; --cpu_smoke is the bounded local test")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        if args.bf16 and not torch.cuda.is_bf16_supported():
            raise RuntimeError("The selected CUDA device does not support BF16")
    else:
        device = torch.device("cpu")
        torch.set_num_threads(1)
    if world_size > 1:
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    try:
        seed_everything(args.seed)
        model, tokenizer = build_model(args, device)
        if args.cpu_smoke:
            corpus, corpus_identity = smoke_corpus(), "synthetic-tiny-history-v1"
        else:
            from history_memory.preparation import PreparedCorpus
            corpus = PreparedCorpus(args.data_path, tokenizer, args.arm)
            path = Path(args.data_path)
            manifest_path = path / "manifest.json" if path.is_dir() else path
            manifest_bytes = manifest_path.read_bytes()
            corpus_identity = hashlib.sha256(manifest_bytes).hexdigest()
            manifest = json.loads(manifest_bytes)
            packing = manifest.get("packing", manifest.get("packing_config", {}))
            for key in ("max_chunk_tokens", "chunk_overlap", "max_system_tokens", "max_workspace_tokens", "max_target_tokens", "max_chunks"):
                value = getattr(args, key)
                if value is not None and packing.get(key) != value:
                    raise ValueError(f"--{key} differs from prepared manifest: {value} vs {packing.get(key)}")
            if args.ratios is not None:
                expected = [int(value) for value in args.ratios.split(",")]
                actual = packing.get("ratios", manifest.get("ratios"))
                if expected != actual:
                    raise ValueError(f"--ratios differs from prepared manifest: {expected} vs {actual}")
            model.config.history_memory_supported_ratios = packing["ratios"]
            model.config.history_memory_packing = packing
            model.config.history_memory_policy = manifest["policy"]
            model.config.history_memory_corpus_identity = corpus_identity
        wrapper = HistoryMemoryModel(model)
        state = train(wrapper, tokenizer, corpus, args, rank=rank, world_size=world_size,
                      device=device, corpus_identity=corpus_identity)
        if rank == 0:
            print(json.dumps({"event": "training_finished", "global_step": state["global_step"],
                              "completed": state["completed"], "output_dir": args.output_dir}), flush=True)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
