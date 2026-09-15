"""CPU integration seams for native Text server construction and cleanup."""
from __future__ import annotations

import copy
import importlib.machinery
import json
import sys
from contextlib import contextmanager
from types import ModuleType, SimpleNamespace

import pytest

from benchmarks.memory_runtime import event_native_api as api_module
from benchmarks.memory_runtime import event_native_server as server
from benchmarks.memory_runtime.event_native_always import NATIVE_TEXT_S0_MODE
from benchmarks.memory_runtime.event_native_summary_transport import (
    ATTEMPTS_PER_TASK,
    EventNativeSummaryTransport,
)
from benchmarks.memory_runtime.tests.test_event_native_policy import (
    Tokenizer,
    contracts,
)
from benchmarks.memory_runtime.text_summary import SummaryRenderer


class Generator:
    session_cache_policy = "last-final-view-v1"

    def __init__(self, runtime, kv_bytes):
        self.runtime = runtime
        self.kv_bytes = kv_bytes
        self.closed = 0

    def kv_bytes_per_token(self):
        return self.kv_bytes

    def session_cache_info(self):
        return {
            "policy": self.session_cache_policy,
            "closed": self.closed,
            "cpu_memo_present": False,
            "device_raw_snapshot_present": False,
        }

    def close_session(self):
        self.closed += 1

    @contextmanager
    def decision_scope(self, *, session_id=None):
        yield

    def generate(self, *args, **kwargs):
        raise AssertionError("CPU server wiring test must not generate")


class Runner:
    generation_calls = 0

    def __init__(self, controller, generator, tokenizer, **kwargs):
        self.controller = controller
        self.generator = generator


class API:
    def __init__(self, runner, **kwargs):
        self.runner = runner

    def health(self):
        return {
            "terminal": True,
            "terminal_reason": "cpu_wiring_complete",
            "decisions_reserved": 0,
            "generation_calls_reserved": 0,
        }


class LoopbackServer:
    server_address = ("127.0.0.1", 31001)
    timeout = None

    def handle_request(self):
        raise AssertionError("terminal CPU wiring API must not receive a request")

    def server_close(self):
        pass


def _profile():
    packing, policy = contracts()
    policy = {**policy, "lease_decisions": 0, "max_retrieved_events": 2}
    return {
        "declared_supported_ratios": packing["ratios"],
        "packing_contract": packing,
        "policy_contract": policy,
        "model_geometry": {
            "max_position_embeddings": packing["max_sequence_tokens"]
        },
    }


def _args(tmp_path, *, allocator=False):
    config = tmp_path / "s0.json"
    config.write_text(json.dumps({"cpu": "server wiring only"}), encoding="utf-8")
    argv = [
        "--checkpoint", str(tmp_path / "checkpoint"),
        "--out", str(tmp_path / "server"),
        "--run-id", "native-text-server-test",
        "--view-mode", NATIVE_TEXT_S0_MODE,
        "--ratio", "4",
        "--max-new-tokens", "16",
        "--decode-strategy", "incremental",
        "--prefill-chunk-size", "64",
        "--task-ids", "multi_turn_base_7",
        "--max-decisions", "1",
        "--max-generation-calls", "1",
        "--s0-config", str(config),
        "--max-wall-seconds", "30",
    ]
    if allocator:
        argv.extend(["--device", "npu:0", "--npu-allocator-metrics"])
    return server.parser().parse_args(argv)


def _wire(monkeypatch, trained, *, timeline, wrappers=None):
    runtime = object()
    actor = Generator(runtime, trained["policy_contract"]["kv_bytes_per_token"])
    auxiliaries = []
    controllers = []
    load_kwargs = []

    class AuxiliaryGenerator(Generator):
        def __init__(self, supplied_runtime, **kwargs):
            timeline.append("auxiliary_generator")
            assert supplied_runtime is runtime
            assert kwargs == {
                "decode_strategy": "incremental",
                "prefill_chunk_size": 64,
            }
            super().__init__(supplied_runtime, actor.kv_bytes)
            auxiliaries.append(self)

    def load_generator(*args, **kwargs):
        timeline.append("actor_generator")
        load_kwargs.append(copy.deepcopy(kwargs))
        return actor, copy.deepcopy(trained)

    def controller_factory(tokenizer, **kwargs):
        timeline.append("controller")
        controllers.append(kwargs)
        return SimpleNamespace(policy=copy.deepcopy(kwargs["policy"]))

    monkeypatch.setattr(server, "inspect_checkpoint", lambda _: copy.deepcopy(trained))
    monkeypatch.setattr(
        server,
        "validate_inference_byte_profile",
        lambda *args: actor.kv_bytes,
    )
    monkeypatch.setattr(server, "load_generator", load_generator)
    monkeypatch.setattr(server, "build_event_native_controller", controller_factory)
    monkeypatch.setattr(server, "EventNativeDecisionRunner", Runner)
    monkeypatch.setattr(server, "_saved_cost_summary", lambda _out: {"actor_only": True})
    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained",
        lambda *args, **kwargs: Tokenizer(),
    )
    inference = ModuleType("history_memory.inference")
    inference.__spec__ = importlib.machinery.ModuleSpec(
        "history_memory.inference", loader=None)
    inference.EventNativeGenerator = AuxiliaryGenerator
    monkeypatch.setitem(sys.modules, "history_memory.inference", inference)
    torch = ModuleType("torch")
    torch.__spec__ = importlib.machinery.ModuleSpec("torch", loader=None)
    torch.set_num_threads = lambda _threads: None
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(api_module, "EventNativeAPI", API)
    monkeypatch.setattr(api_module, "make_server", lambda *args, **kwargs: LoopbackServer())

    if wrappers is not None:
        class Wrapper:
            measurement_contract = {"schema": "cpu-fake-allocator-measurement-v1"}

            def __init__(self, generator):
                self.generator = generator
                wrappers.append(generator)

            def __getattr__(self, name):
                return getattr(self.generator, name)

        monkeypatch.setattr(
            "benchmarks.memory_runtime.event_native_allocator.NpuAllocatorMeasuredGenerator",
            Wrapper,
        )

    return actor, auxiliaries, controllers, load_kwargs


def test_text_route_loads_actor_before_distinct_shared_runtime_summary_generator(
    tmp_path, monkeypatch
):
    trained = _profile()
    timeline = []
    actor, auxiliaries, controllers, load_kwargs = _wire(
        monkeypatch, trained, timeline=timeline
    )
    args = _args(tmp_path)

    server._serve(args)

    assert timeline == ["actor_generator", "auxiliary_generator", "controller"]
    assert len(auxiliaries) == len(controllers) == 1
    auxiliary = auxiliaries[0]
    assert auxiliary is not actor and auxiliary.runtime is actor.runtime
    assert "max_extraction_calls" not in load_kwargs[0]
    controller = controllers[0]
    assert controller["run_id"] == args.run_id
    assert controller["view_mode"] == NATIVE_TEXT_S0_MODE
    assert controller["compression_policy"] is None
    renderer = controller["summary_renderer"]
    assert isinstance(renderer, SummaryRenderer)
    assert isinstance(renderer.generate, EventNativeSummaryTransport)
    assert renderer.generate.generator is auxiliary
    assert renderer.generate.deadline_monotonic > 0

    startup = json.loads((args.out / "startup.json").read_text(encoding="utf-8"))
    ready = json.loads((args.out / "ready.json").read_text(encoding="utf-8"))
    final = json.loads((args.out / "final.json").read_text(encoding="utf-8"))
    contract = startup["auxiliary_summary_contract"]
    assert contract == ready["auxiliary_summary_contract"]
    assert contract["attempts_per_task"] == ATTEMPTS_PER_TASK == 1152
    assert contract["max_extraction_calls"] is None
    assert contract["attempt_journal"].endswith("summary_attempts.jsonl")
    assert contract["private_trace"].endswith("summary_calls.jsonl")
    assert final["journal_summary"] is None
    assert final["auxiliary_summary_journal_summary"] is None
    assert final["auxiliary_summary_trace_exists"] is False
    assert final["auxiliary_summary_cache_before_close"]["closed"] == 0
    assert final["auxiliary_summary_cache_after_close"]["closed"] == 1
    assert final["session_cache_before_close"]["closed"] == 0
    assert final["session_cache_after_close"]["closed"] == 1
    assert actor.closed == auxiliary.closed == 1


def test_text_route_wraps_actor_and_summary_allocator_measurement_separately(
    tmp_path, monkeypatch
):
    trained = _profile()
    wrappers = []
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace())
    actor, auxiliaries, _, _ = _wire(
        monkeypatch, trained, timeline=[], wrappers=wrappers
    )
    args = _args(tmp_path, allocator=True)

    server._serve(args)

    assert wrappers == [actor, auxiliaries[0]]
    final = json.loads((args.out / "final.json").read_text(encoding="utf-8"))
    expected = {"schema": "cpu-fake-allocator-measurement-v1"}
    assert final["allocator_measurement"] == expected
    assert final["auxiliary_summary_contract"]["allocator_measurement"] == expected
    assert final["journal_summary"] is None
    assert final["auxiliary_summary_journal_summary"] is None


def test_text_route_rejects_actor_extraction_cap_before_checkpoint_or_output(
    tmp_path, monkeypatch
):
    args = _args(tmp_path)
    args.max_extraction_calls = 1
    monkeypatch.setattr(
        server,
        "inspect_checkpoint",
        lambda *_: pytest.fail("checkpoint must not load"),
    )

    with pytest.raises(ValueError, match="does not accept max-extraction-calls"):
        server._serve(args)

    assert not args.out.exists()


def test_existing_route_keeps_controller_before_generator_load(tmp_path, monkeypatch):
    trained = _profile()
    timeline = []
    _wire(monkeypatch, trained, timeline=timeline)
    args = _args(tmp_path)
    args.view_mode = "capacity_protect"
    args.s0_config = None

    server._serve(args)

    assert timeline == ["controller", "actor_generator"]
