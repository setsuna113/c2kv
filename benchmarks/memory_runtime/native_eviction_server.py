"""Finite native history-boundary eviction service for the B500 system study."""
from __future__ import annotations

import argparse
import json
import signal
import time
from pathlib import Path

from .event_native import load_generator, inspect_checkpoint, validate_inference_byte_profile
from .event_native_api import EventNativeAPI, make_server
from .event_native_server import save_json


class EvictionAPI(EventNativeAPI):
    def __init__(self, runner, *, method, **kwargs):
        # Reuse the native transport validation, with an explicit new method
        # identity installed before the endpoint can accept any request.
        super().__init__(runner, view_mode="full_original", **kwargs)
        self.view_mode = "native_" + method + "_boundary_reselect"
        self.route_contract = {
            "view_mode": self.view_mode,
            "baseline_identity": self.view_mode,
            "recovery_enabled": False,
            "max_generations_per_decision": 1,
            "legacy_1088_equivalent": False,
            "history_reselection": "full_raw_history_each_decision",
            "persistent_eviction": False,
        }

    def health(self):
        result = super().health()
        result["session_cache_policy"] = "history-boundary-reselect-v1"
        result["decode_strategy"] = "incremental"
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--method", choices=("snapkv_style", "h2o_style"), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--max-wall-seconds", type=float, required=True)
    parser.add_argument("--history-budget-bytes", type=int, default=113246208)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--max-decisions", type=int, default=96)
    parser.add_argument("--device", default="npu:0")
    args = parser.parse_args()
    if args.max_wall_seconds <= 0 or args.max_decisions <= 0:
        raise ValueError("positive finite limits required")
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    deadline = started + args.max_wall_seconds
    manifest = {"schema": "a-event-native-server-v1", "status": "loading",
                "run_id": args.run_id, "method": args.method}
    save_json(args.out / "loading.json", manifest)
    runner = server = api = None
    stopped = False

    def stop(_signum, _frame):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        import torch
        if args.device.startswith("npu"):
            import torch_npu  # noqa: F401
        from transformers import AutoTokenizer
        from .native_eviction_runtime import NativeEvictionRunner
        torch.set_num_threads(4)
        torch.manual_seed(0)
        profile = inspect_checkpoint(args.checkpoint)
        validate_inference_byte_profile(profile, "bfloat16")
        tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
        generator, profile = load_generator(args.checkpoint, device=args.device,
            dtype="bfloat16", decode_strategy="incremental", prefill_chunk_size=256)
        runner = NativeEvictionRunner(generator, tokenizer, method=args.method,
            history_budget_bytes=args.history_budget_bytes,
            max_new_tokens=args.max_new_tokens, max_generation_calls=args.max_decisions,
            max_sequence_tokens=40960, prefill_chunk_size=256)
        policy = {"schema": "a-event-native-runtime-policy-v1", "source": "explicit_eval_policy",
                  "effective_policy": {"history_budget_bytes": args.history_budget_bytes,
                    "workspace_budget_bytes": args.history_budget_bytes,
                    "common_current_input": "native_S0_raw_source_cutoff",
                    "maintenance_prefill": "full_history_reselect_counted_separately"}}
        api = EvictionAPI(runner, method=args.method, run_id=args.run_id,
            model_name="b500-" + args.method, max_new_tokens=args.max_new_tokens,
            allowed_task_ids=[args.task_id], max_decisions=args.max_decisions,
            deadline_monotonic=deadline, steps_path=args.out / "steps.jsonl",
            runtime_policy_contract=policy)
        server = make_server(api, host="127.0.0.1", port=args.port)
        server.timeout = 1
        manifest = {**manifest, **api.health(), "schema": "a-event-native-server-v1",
                    "status": "ready", "checkpoint": profile,
                    "max_generation_calls": args.max_decisions,
                    "sampling": {"mode": "greedy", "temperature": 0, "seed": 0}}
        save_json(args.out / "ready.json", manifest)
        while not stopped and time.monotonic() < deadline and not api.health()["terminal"]:
            server.handle_request()
        manifest["status"] = "stopped"
    except BaseException as error:
        manifest.update(status="failed", error={"type": type(error).__name__, "message": str(error)})
        raise
    finally:
        if server is not None:
            server.server_close()
        if api is not None:
            manifest["api_health"] = api.health()
        if runner is not None:
            runner.close()
        manifest.update(wall_seconds=time.monotonic() - started, wall_seconds_final=True)
        save_json(args.out / "final.json", manifest)


if __name__ == "__main__":
    main()
