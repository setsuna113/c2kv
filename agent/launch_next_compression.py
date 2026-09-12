#!/usr/bin/env python3
"""Plan or run a fixed six-arm queue on an explicit set of available GPUs."""
from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import io
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from next_compression.common import SerializedCorpus, VARIANTS, sha256_file


BASE_REPOSITORY = "Qwen/Qwen3-4B-Instruct-2507"
BASE_REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"


def check_base_receipt(model_path):
    path = Path(model_path) / "C2KV_SOURCE_REVISION.json"
    if not path.is_file():
        raise ValueError(
            f"Fresh base is missing {path.name}; run bootstrap_h100.sh without --skip-model"
        )
    receipt = json.loads(path.read_text(encoding="utf-8"))
    expected = {"repository": BASE_REPOSITORY, "revision": BASE_REVISION}
    if receipt != expected:
        raise ValueError(f"Fresh base source receipt differs: {receipt!r} vs {expected!r}")
    return receipt


def device_groups(devices, per_run=1, per_device_batch_size=2):
    if not devices or len(set(devices)) != len(devices) or any(not str(x).isdigit() for x in devices):
        raise ValueError("Supply distinct physical GPU indices, e.g. --devices 0,2,4")
    if per_run < 1 or len(devices) % per_run:
        raise ValueError("The explicit GPU list must divide into whole per-run groups")
    if per_device_batch_size < 1 or 32 % (per_device_batch_size * per_run):
        raise ValueError(
            "Effective batch 32 requires per-device batch times GPUs-per-run "
            "to be a positive divisor of 32"
        )
    return [devices[start:start + per_run] for start in range(0, len(devices), per_run)]


def check_devices(devices):
    query = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,name,memory.total", "--format=csv,noheader,nounits"],
                           check=True, text=True, capture_output=True)
    inventory = {row[0].strip(): dict(uuid=row[1].strip(), name=row[2].strip(), memory_mib=int(row[3].strip()))
                 for row in csv.reader(io.StringIO(query.stdout))}
    missing = set(devices) - set(inventory)
    if missing:
        raise ValueError(f"Unknown physical GPU indices: {sorted(missing)}")
    wrong_model = [
        index for index in devices if "H100" not in inventory[index]["name"].upper()
    ]
    if wrong_model:
        raise RuntimeError(
            f"Selected devices are not H100 GPUs: {wrong_model}; choose H100 indices"
        )
    applications = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
                                  check=True, text=True, capture_output=True)
    used = {row[0].strip() for row in csv.reader(io.StringIO(applications.stdout)) if row}
    busy = [index for index in devices if inventory[index]["uuid"] in used]
    if busy:
        raise RuntimeError(f"Selected GPUs already have compute processes: {busy}; choose unused devices")
    return {index: inventory[index] for index in devices}


def build_command(args, variant, group):
    script = Path(__file__).resolve().with_name("train_next_compression.py")
    if len(group) == 1:
        command = [args.python, str(script)]
    else:
        command = [args.python, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node", str(len(group)), str(script)]
    command += ["--variant", variant, "--data_path", str(Path(args.data_root) / variant),
                "--output_dir", str(Path(args.output_root) / variant),
                "--seed", str(args.seed), "--per_device_batch_size", str(args.per_device_batch_size),
                "--gradient_accumulation_steps", str(32 // (args.per_device_batch_size * len(group))),
                "--num_train_epochs", str(args.epochs), "--save_steps", str(args.save_steps),
                "--max_steps", str(args.max_steps), "--ratios", "8,12", "--wandb_mode", args.wandb_mode]
    if args.stop_after_steps > 0:
        command += ["--stop_after_steps", str(args.stop_after_steps)]
    if variant in args.resumes:
        command += ["--resume_from_checkpoint", args.resumes[variant]]
    elif args.warm_start_history and variant.startswith("H"):
        command += ["--warm_start_checkpoint", args.warm_start_history]
    elif args.warm_start_tool and variant.startswith("T"):
        command += ["--warm_start_checkpoint", args.warm_start_tool]
    else:
        command += ["--model_name_or_path", args.model]
    return command


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--devices", required=True, help="Explicit physical indices; unlisted GPUs remain untouched")
    parser.add_argument("--gpus-per-run", type=int, default=1)
    parser.add_argument(
        "--per-device-batch-size",
        type=int,
        default=2,
        help="Microbatch per GPU; accumulation preserves effective batch 32",
    )
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help="Optional optimizer-update cap; -1 runs every update in --epochs",
    )
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--stop-after-steps", type=int, default=-1, help="Bounded real-data smoke; saves a resumable checkpoint")
    parser.add_argument("--wandb-mode", choices=["offline", "online", "disabled"], default="offline")
    parser.add_argument("--warm-start-history")
    parser.add_argument("--warm-start-tool")
    parser.add_argument("--resume", action="append", default=[], metavar="VARIANT=CHECKPOINT")
    parser.add_argument("--run", action="store_true", help="Without this flag only print the exact commands")
    args = parser.parse_args(argv)
    devices = args.devices.split(",")
    groups = device_groups(
        devices, args.gpus_per_run, args.per_device_batch_size
    )
    variants = args.variants.split(",")
    if not variants or len(set(variants)) != len(variants) or set(variants) - set(VARIANTS):
        parser.error("Select distinct known variants")
    args.resumes = {}
    for value in args.resume:
        if "=" not in value:
            parser.error("--resume must be VARIANT=CHECKPOINT")
        variant, checkpoint = value.split("=", 1)
        if not variant or not checkpoint:
            parser.error("--resume must be VARIANT=CHECKPOINT")
        if variant in args.resumes:
            parser.error(f"Duplicate resume mapping for {variant}")
        args.resumes[variant] = checkpoint
    if set(args.resumes) - set(variants):
        parser.error("Resume mappings must refer to selected variants")
    fresh_variants = [
        variant
        for variant in variants
        if variant not in args.resumes
        and not (variant.startswith("H") and args.warm_start_history)
        and not (variant.startswith("T") and args.warm_start_tool)
    ]
    if fresh_variants:
        check_base_receipt(args.model)
    manifests = {}
    for variant_index, variant in enumerate(variants):
        path = Path(args.data_root) / variant / "manifest.json"
        corpus = SerializedCorpus(path, expected_variant=variant)
        manifest = corpus.manifest
        expected_domain = "history" if variant.startswith("H") else "tool"
        expected_loss = (
            "decision-mean-critical-token-weighted-ce-v1"
            if variant == "H3"
            else "decision-mean-complete-ce-v1"
        )
        if manifest.get("compression_domain") != expected_domain:
            raise ValueError(f"Wrong prepared compression domain for {variant}")
        if manifest.get("loss_profile") != expected_loss:
            raise ValueError(f"Wrong prepared loss profile for {variant}")
        if not isinstance(manifest.get("render_profile"), str) or not manifest["render_profile"]:
            raise ValueError(f"Prepared corpus has no render profile for {variant}")
        world_size = len(groups[variant_index % len(groups)])
        omitted_per_epoch = len(corpus) % world_size
        usable_records = len(corpus) - omitted_per_epoch
        uncapped_steps = args.epochs * math.ceil(usable_records / 32)
        planned_steps = (
            min(uncapped_steps, args.max_steps)
            if args.max_steps > 0
            else uncapped_steps
        )
        manifests[variant] = dict(
            path=str(path.resolve()),
            sha256=sha256_file(path),
            records=len(corpus),
            world_size=world_size,
            omitted_tail_records_per_epoch=omitted_per_epoch,
            uncapped_steps=uncapped_steps,
            planned_steps=planned_steps,
        )
    plan = dict(schema="next-compression-h100-queue-v1", variants=variants, groups=groups, effective_batch=32,
                corpus_manifests=manifests, automatic_retries=0, max_steps=args.max_steps, epochs=args.epochs,
                command_examples={variant: build_command(args, variant, groups[index % len(groups)])
                                  for index, variant in enumerate(variants)})
    if not args.run:
        print(json.dumps(plan, indent=2))
        for variant, command in plan["command_examples"].items():
            print(f"# {variant}\n{shlex.join(command)}")
        return 0
    output = Path(args.output_root)
    output.mkdir(parents=True, exist_ok=True)
    receipt = output / ("queue-" + str(time.time_ns()) + ".json")
    plan.update(status="running", hardware=check_devices(devices), started_at=time.time(), outcomes=[])
    receipt.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    pending = deque(variants)

    def worker(group, variant):
        # Recheck immediately before each arm starts. No process is killed or retried.
        check_devices(group)
        command = build_command(args, variant, group)
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(group), TOKENIZERS_PARALLELISM="false")
        log = output / (variant + "-" + str(time.time_ns()) + ".log")
        started = time.time()
        with log.open("w", encoding="utf-8") as handle:
            result = subprocess.run(command, env=environment, stdout=handle, stderr=subprocess.STDOUT)
        return dict(variant=variant, devices=group, returncode=result.returncode, log=str(log),
                    command=command, elapsed_seconds=time.time() - started)

    # A finite queue: failed arms are recorded once, and every other arm runs at most once.
    with ThreadPoolExecutor(max_workers=len(groups)) as pool:
        futures = {}
        for group in groups:
            if pending:
                variant = pending.popleft()
                futures[pool.submit(worker, group, variant)] = (group, variant)
        while futures:
            future = next(as_completed(futures))
            group, variant = futures.pop(future)
            try:
                outcome = future.result()
            except Exception as error:
                outcome = dict(
                    variant=variant,
                    devices=group,
                    returncode=1,
                    error=str(error),
                )
            plan["outcomes"].append(outcome)
            receipt.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
            if pending:
                variant = pending.popleft()
                futures[pool.submit(worker, group, variant)] = (group, variant)
    plan.update(status="completed" if all(x["returncode"] == 0 for x in plan["outcomes"]) else "failed",
                finished_at=time.time())
    receipt.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": plan["status"], "receipt": str(receipt)}))
    return 0 if plan["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
