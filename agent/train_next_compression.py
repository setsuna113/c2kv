#!/usr/bin/env python3
"""Train one pretokenized next-compression variant on one or more H100 GPUs."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from next_compression.common import TRAINING_PROFILE, VARIANTS


_GIST_PROJECTIONS = (".gist_q_proj.", ".gist_k_proj.", ".gist_v_proj.")


def _is_gist_parameter(name: str) -> bool:
    return name.startswith("model.gist_embed_tokens.") or any(
        marker in name for marker in _GIST_PROJECTIONS
    )


def _checkpoint_gist_tensors(path: str | Path):
    """Read only the saved gist tensors, retaining their on-disk precision."""

    from safetensors import safe_open

    path = Path(path)
    index_path = path / "model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict):
            raise ValueError("Warm-start safetensors index has no weight map")
        locations = {
            name: path / filename
            for name, filename in weight_map.items()
            if _is_gist_parameter(name)
        }
    else:
        single = path / "model.safetensors"
        if not single.is_file():
            raise ValueError("Warm start requires safetensors model weights")
        with safe_open(single, framework="pt", device="cpu") as handle:
            names = tuple(handle.keys())
        locations = {
            name: single for name in names if _is_gist_parameter(name)
        }
    tensors = {}
    for filename in sorted(set(locations.values())):
        with safe_open(filename, framework="pt", device="cpu") as handle:
            for name, location in locations.items():
                if location == filename:
                    tensors[name] = handle.get_tensor(name)
    return tensors


def _restore_warm_start_gist_precision(model, checkpoint: str | Path) -> None:
    """Undo global BF16 load casting for saved FP32 compressor tensors."""

    import torch

    saved = _checkpoint_gist_tensors(checkpoint)
    live = {
        name: parameter
        for name, parameter in model.named_parameters()
        if _is_gist_parameter(name)
    }
    if set(saved) != set(live):
        raise ValueError("Warm-start checkpoint has a different gist parameter set")
    with torch.no_grad():
        for name, parameter in live.items():
            value = saved[name]
            if value.shape != parameter.shape:
                raise ValueError(f"Warm-start gist shape differs for {name}")
            parameter.data = value.to(device=parameter.device, dtype=torch.float32)


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_path")
    parser.add_argument("--model_name_or_path")
    parser.add_argument("--warm_start_checkpoint")
    parser.add_argument("--resume_from_checkpoint")
    parser.add_argument("--initialization_id")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument(
        "--ratios",
        default="8,12",
        help="Assertion against the prepared corpus; this round requires 8,12",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--per_device_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--num_train_epochs", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument(
        "--stop_after_steps",
        type=int,
        default=-1,
        help="Controlled interruption after an update; preserves the planned resume schedule",
    )
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument(
        "--attn_impl",
        choices=["eager", "sdpa", "flex_attention"],
        default="sdpa",
    )
    parser.add_argument(
        "--bf16", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument(
        "--cpu_smoke",
        action="store_true",
        help="Use a random tiny Qwen; consume --data_path when supplied, otherwise synthetic records",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--wandb_mode", choices=["disabled", "offline", "online"], default="offline"
    )
    parser.add_argument(
        "--wandb_entity", default="liuyc1025-university-of-cambridge"
    )
    parser.add_argument("--wandb_project", default="c2kv-next-compression")
    parser.add_argument("--wandb_run_id")
    result = parser.parse_args(argv)

    for flag in (
        "per_device_batch_size",
        "gradient_accumulation_steps",
        "num_train_epochs",
        "save_steps",
        "logging_steps",
    ):
        if getattr(result, flag) < 1:
            parser.error(f"--{flag} must be positive")
    if (
        not 0 <= result.warmup_ratio <= 1
        or result.learning_rate <= 0
        or result.weight_decay < 0
        or result.max_grad_norm <= 0
    ):
        parser.error("Invalid optimizer settings")
    if result.resume_from_checkpoint and result.warm_start_checkpoint:
        parser.error("--resume_from_checkpoint and --warm_start_checkpoint are distinct and mutually exclusive")
    if result.warm_start_checkpoint and result.model_name_or_path:
        parser.error("A warm start takes its model only from --warm_start_checkpoint")
    if result.cpu_smoke:
        result.device = "cpu"
        result.bf16 = False
        result.attn_impl = "eager"
        result.wandb_mode = "disabled"
        result.gradient_checkpointing = False
    else:
        if not result.data_path:
            parser.error("--data_path is required outside --cpu_smoke")
        if not result.resume_from_checkpoint and not result.warm_start_checkpoint:
            if not result.model_name_or_path:
                parser.error("Fresh training requires --model_name_or_path")
    try:
        parsed_ratios = [int(value) for value in result.ratios.split(",")]
    except ValueError:
        parser.error("--ratios must be a comma-separated integer list")
    if parsed_ratios != [8, 12]:
        parser.error("This training entry is frozen to --ratios 8,12")

    # The shared training loop retains its old ``arm`` API.  The distinct
    # training profile and config metadata keep these variants out of B/C.
    result.arm = result.variant
    result.training_contract = None
    result.wandb_name = f"next_compression_{result.variant}_s{result.seed}"
    return result


def tiny_tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocabulary = {
        "[PAD]": 0,
        "[UNK]": 1,
        "[EOS]": 2,
        **{f"t{index}": index for index in range(3, 128)},
    }
    backend = Tokenizer(models.WordLevel(vocabulary, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        eos_token="[EOS]",
    )
    tokenizer.chat_template = (
        "{% for message in messages %}{{ message['role'] }}: "
        "{{ message['content'] }}\n{% endfor %}"
        "{% if add_generation_prompt %}assistant: {% endif %}"
    )
    return tokenizer


def smoke_corpus(variant: str):
    from history_memory.packing import EncoderChunk, MemoryView, PackedMemory
    from next_compression.common import PreparedDecision

    class Corpus:
        group_ids = ["smoke-a"] * 4 + ["smoke-b"] * 4
        manifest = {
            "variant": variant,
            "compression_domain": "history" if variant.startswith("H") else "tool",
            "ratios": [8, 12],
            "preparation": {
                "render_profile": "synthetic-next-compression-v1",
                "loss_profile": (
                    "decision-mean-critical-token-weighted-ce-v1"
                    if variant == "H3"
                    else "decision-mean-complete-ce-v1"
                ),
            },
        }

        def __len__(self):
            return len(self.group_ids)

        def __getitem__(self, index):
            event_id = f"event-{index // 4}"
            token_start = 4 + index // 4
            chunk = EncoderChunk(
                event_id,
                0,
                (0,),
                0,
                24,
                tuple(range(token_start, token_start + 24)),
            )
            memory = PackedMemory(
                MemoryView((event_id,), ("current",)),
                (30, 31),
                (32, 33 + index),
                (1,),
                (chunk,),
            )
            target_weights = (1.0, 4.0) if variant == "H3" else None
            return PreparedDecision(
                memory,
                (50 + index, 2),
                8 if index % 2 == 0 else 12,
                1.0,
                f"smoke-{index}",
                target_weights,
            )

    return Corpus()


def _source_kind(args) -> str:
    if args.resume_from_checkpoint:
        return "resume"
    if args.warm_start_checkpoint:
        return "warm-start"
    return "fresh-base"


def _validate_gist_checkpoint(config, *, purpose: str) -> None:
    expected = {
        "gist_type": "dynamic-interleave",
        "gist_param": "qkv",
        "gist_residual_type": "embed-mean",
    }
    for field, value in expected.items():
        if getattr(config, field, None) != value:
            raise ValueError(f"{purpose} checkpoint has incompatible {field}")
    if getattr(config, "history_memory_normal_query", None) != "base":
        raise ValueError(f"{purpose} checkpoint does not declare normal query=base")


def _checkpoint_metadata_identity(path: str | Path, kind: str) -> str:
    """Derive a stable fallback ID without hashing multi-GB weights per rank."""

    path = Path(path)
    digest = hashlib.sha256()
    found = False
    for descriptor in (
        path / "config.json",
        path / "C2KV_SOURCE_REVISION.json",
        path / "model.safetensors.index.json",
        path / "trainer_state.json",
    ):
        if descriptor.is_file():
            found = True
            digest.update(descriptor.name.encode("utf-8"))
            digest.update(descriptor.read_bytes())
    for weight in sorted(path.glob("*.safetensors")):
        found = True
        digest.update(weight.name.encode("utf-8"))
        digest.update(str(weight.stat().st_size).encode("ascii"))
    if not found:
        raise ValueError(f"Initialization path has no checkpoint metadata: {path}")
    return f"{kind}-metadata-sha256:{digest.hexdigest()}"


def build_model(args, device):
    import torch
    from transformers import AutoTokenizer
    from models.qwen3.configuration_qwen3 import Qwen3Config
    from models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

    kind = _source_kind(args)
    load_path = (
        args.resume_from_checkpoint
        or args.warm_start_checkpoint
        or args.model_name_or_path
    )
    if kind != "resume" and args.initialization_id is None and load_path is not None:
        args.initialization_id = _checkpoint_metadata_identity(load_path, kind)
    if args.cpu_smoke and kind == "fresh-base":
        config = Qwen3Config(
            vocab_size=128,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            max_position_embeddings=8192,
            pad_token_id=None,
            eos_token_id=2,
            attention_dropout=0.0,
            gist_type="dynamic-interleave",
            gist_param="qkv",
            gist_token_id=2,
            gist_residual_type="embed-mean",
        )
        config._attn_implementation = "eager"
        model = Qwen3ForCausalLM(config)
        tokenizer = tiny_tokenizer()
    else:
        tokenizer = AutoTokenizer.from_pretrained(load_path, local_files_only=True)
        config = Qwen3Config.from_pretrained(load_path, local_files_only=True)
        if kind == "fresh-base":
            raw_config = json.loads(
                (Path(load_path) / "config.json").read_text(encoding="utf-8")
            )
            if (
                raw_config.get("model_type") != "qwen3"
                or raw_config.get("architectures") != ["Qwen3ForCausalLM"]
            ):
                raise ValueError("Fresh training requires the original Qwen3 causal language model")
            if "gist_param" in raw_config or "history_memory_training_profile" in raw_config:
                raise ValueError("Fresh training cannot silently treat a gist checkpoint as the base")
            config.gist_type = "dynamic-interleave"
            config.gist_param = "qkv"
            config.gist_residual_type = "embed-mean"
            config.gist_token_id = tokenizer.eos_token_id
        else:
            _validate_gist_checkpoint(config, purpose=kind)
            if kind == "resume":
                if getattr(config, "history_memory_training_profile", None) != TRAINING_PROFILE:
                    raise ValueError("Resume requires this next-compression training profile")
                if getattr(config, "history_memory_variant", None) != args.variant:
                    raise ValueError("Resume variant differs from the checkpoint")
        config._attn_implementation = args.attn_impl
        model = Qwen3ForCausalLM.from_pretrained(
            load_path,
            config=config,
            dtype=torch.bfloat16 if args.bf16 else torch.float32,
            local_files_only=True,
        )
        if kind == "warm-start":
            _restore_warm_start_gist_precision(model, load_path)

    if kind == "fresh-base":
        # This is the only path that creates gist weights.  Warm start and
        # optimizer resume retain every saved gist tensor exactly as loaded.
        with torch.no_grad():
            model.model.gist_embed_tokens.weight.copy_(
                model.model.embed_tokens.weight[
                    config.gist_token_id : config.gist_token_id + 1
                ]
            )
            for layer in model.model.layers:
                for projection in ("q", "k", "v"):
                    source = getattr(layer.self_attn, f"{projection}_proj")
                    target = getattr(layer.self_attn, f"gist_{projection}_proj")
                    target.load_state_dict(source.state_dict())

    live_config = model.config
    initialization_group = "history" if args.variant.startswith("H") else "tool"
    if kind == "resume":
        saved_id = getattr(live_config, "history_memory_initialization_id", None)
        if not isinstance(saved_id, str) or not saved_id:
            raise ValueError("Resume checkpoint is missing its initialization identity")
        if args.initialization_id is not None and args.initialization_id != saved_id:
            raise ValueError("--initialization_id differs from the resume checkpoint")
        args.initialization_id = saved_id
        saved_group = getattr(live_config, "history_memory_initialization_group", None)
        if saved_group != initialization_group:
            raise ValueError("Resume checkpoint has a different initialization group")
        initialization_kind = getattr(
            live_config, "history_memory_initialization_kind", None
        )
        if initialization_kind not in {"fresh-base", "warm-start"}:
            raise ValueError("Resume checkpoint has an invalid initialization kind")
    else:
        if args.initialization_id is None:
            args.initialization_id = f"cpu-smoke-tiny-seed-{args.seed}"
        initialization_kind = kind

    live_config.history_memory_training_profile = TRAINING_PROFILE
    live_config.history_memory_normal_query = "base"
    live_config.history_memory_variant = args.variant
    live_config.history_memory_compression_domain = initialization_group
    live_config.history_memory_supported_ratios = [8, 12]
    live_config.history_memory_trainable_dtype = "float32"
    live_config.history_memory_seed = args.seed
    live_config.history_memory_initialization_id = args.initialization_id
    live_config.history_memory_initialization_group = initialization_group
    live_config.history_memory_initialization_kind = initialization_kind
    model.to(device)
    if args.gradient_checkpointing:
        from models import gist_utils
        from models.qwen3 import modeling_qwen3

        gist_utils.GIST_GRADIENT_CHECKPOINTING = True
        modeling_qwen3.GIST_GRADIENT_CHECKPOINTING = True
        os.environ["C2KV_GIST_CHECKPOINT_USE_REENTRANT"] = "false"
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    return model, tokenizer


def _manifest_profile(manifest: dict, field: str) -> str:
    value = manifest.get(field)
    if value is None and isinstance(manifest.get("preparation"), dict):
        value = manifest["preparation"].get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Prepared manifest must declare {field}")
    return value


def load_corpus(args, tokenizer):
    if args.cpu_smoke and not args.data_path:
        corpus = smoke_corpus(args.variant)
        identity = "synthetic-next-compression-v1"
    else:
        from next_compression.common import SerializedCorpus

        corpus = SerializedCorpus(
            args.data_path, tokenizer=tokenizer, expected_variant=args.variant
        )
        identity = corpus.identity
    manifest = corpus.manifest
    domain = "history" if args.variant.startswith("H") else "tool"
    if manifest.get("compression_domain") != domain:
        raise ValueError("Prepared compression domain differs from the variant")
    if manifest.get("ratios") != [8, 12]:
        raise ValueError("Next-compression training is frozen to ratios 8 and 12")
    render_profile = _manifest_profile(manifest, "render_profile")
    loss_profile = _manifest_profile(manifest, "loss_profile")
    expected_loss_profile = (
        "decision-mean-critical-token-weighted-ce-v1"
        if args.variant == "H3"
        else "decision-mean-complete-ce-v1"
    )
    if loss_profile != expected_loss_profile:
        raise ValueError(
            f"Prepared loss profile differs from {args.variant}: "
            f"{loss_profile!r} vs {expected_loss_profile!r}"
        )

    weighted_records = sum(
        corpus[index].target_weights is not None for index in range(len(corpus))
    )
    if args.variant == "H3" and weighted_records != len(corpus):
        raise ValueError("Every H3 record must carry prepared per-token target weights")
    if args.variant != "H3" and weighted_records:
        raise ValueError("Prepared per-token target weights are exclusive to H3")
    return corpus, identity, render_profile, loss_profile


def configure_training_metadata(
    model,
    args,
    *,
    corpus_identity: str,
    render_profile: str,
    loss_profile: str,
) -> None:
    metadata = {
        "history_memory_training_profile": TRAINING_PROFILE,
        "history_memory_variant": args.variant,
        "history_memory_compression_domain": (
            "history" if args.variant.startswith("H") else "tool"
        ),
        "history_memory_supported_ratios": [8, 12],
        "history_memory_corpus_identity": corpus_identity,
        "history_memory_render_profile": render_profile,
        "history_memory_loss_profile": loss_profile,
        "history_memory_normal_query": "base",
    }
    if args.resume_from_checkpoint:
        for field, expected in metadata.items():
            if getattr(model.config, field, None) != expected:
                raise ValueError(f"Resume checkpoint metadata differs for {field}")
    for field, value in metadata.items():
        setattr(model.config, field, value)

    args.training_contract = {
        "initialization_id": args.initialization_id,
        "initialization_group": model.config.history_memory_initialization_group,
        "initialization_kind": model.config.history_memory_initialization_kind,
        "dataset_sha256": corpus_identity,
        "variant": args.variant,
        "compression_domain": metadata["history_memory_compression_domain"],
        "ratios": [8, 12],
        "render_profile": render_profile,
        "loss_profile": loss_profile,
        "normal_query": "base",
    }


def main(argv=None):
    args = arguments(argv)
    import torch
    import torch.distributed as dist
    from history_memory.runtime import HistoryMemoryModel
    from history_memory.training import seed_everything, train

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size < 1 or not 0 <= rank < world_size or local_rank < 0:
        raise ValueError("Invalid torchrun rank environment")
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
        # Identical seeds before construction keep model initialization equal
        # across ranks; epoch_batches handles rank-specific data partitioning.
        seed_everything(args.seed)
        model, tokenizer = build_model(args, device)
        corpus, identity, render_profile, loss_profile = load_corpus(args, tokenizer)
        configure_training_metadata(
            model,
            args,
            corpus_identity=identity,
            render_profile=render_profile,
            loss_profile=loss_profile,
        )
        wrapper = HistoryMemoryModel(model)
        state = train(
            wrapper,
            tokenizer,
            corpus,
            args,
            rank=rank,
            world_size=world_size,
            device=device,
            corpus_identity=identity,
        )
        if rank == 0:
            print(
                json.dumps(
                    {
                        "event": "training_finished",
                        "variant": args.variant,
                        "global_step": state["global_step"],
                        "completed": state["completed"],
                        "output_dir": args.output_dir,
                    }
                ),
                flush=True,
            )
        return state
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
