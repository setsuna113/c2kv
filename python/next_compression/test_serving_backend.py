"""Production routing, stop propagation, and concurrent request accounting."""

from concurrent.futures import ThreadPoolExecutor
import importlib.util
import sys
import threading
from types import ModuleType

import pytest

from next_compression import live
from next_compression.test_live import REPOSITORY_ROOT, _payload, _service


def _entry():
    spec = importlib.util.spec_from_file_location(
        "serve_backend_test", REPOSITORY_ROOT / "agent" / "serve_next_checkpoint.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _args():
    return ["--checkpoint", "checkpoint", "--training-manifest", "manifest.json",
            "--ratio", "8", "--output-dir", "out", "--max-requests", "10"]


def test_cli_requires_sglang_url_without_implicit_reference_fallback():
    entry = _entry()
    with pytest.raises(SystemExit):
        entry.parse_args(_args())
    args = entry.parse_args(_args() + ["--sglang-url", "http://127.0.0.1:34010"])
    assert args.backend == "sglang"
    assert entry.parse_args(_args() + ["--backend", "native"]).backend == "native"
    with pytest.raises(SystemExit):
        entry.parse_args(_args() + ["--backend", "native", "--device", "npu"])


def test_sglang_loader_keeps_training_binding_and_publishes_backend(monkeypatch):
    reference, generator, tokenizer = _service("T0", "ready")
    calls = []
    module = ModuleType("next_compression.sglang")

    def load(checkpoint, **kwargs):
        calls.append((checkpoint, kwargs))
        return generator, tokenizer, {**reference.profile, "serving_engine": {"verified": True}}

    module.load_sglang_checkpoint = load
    monkeypatch.setitem(sys.modules, "next_compression.sglang", module)
    binding_calls = []

    def binding(path, **kwargs):
        binding_calls.append((path, kwargs))
        return reference.binding

    monkeypatch.setattr(live, "load_training_binding", binding)
    service = live.LiveNextCompressionService.from_checkpoint(
        "checkpoint", "training-manifest.json", device="cuda", dtype="bfloat16",
        sglang_url="http://127.0.0.1:34010", ratio=8, mode="compressed",
        model="candidate", max_new_tokens=1024, max_requests=2,
    )
    assert calls[0][1]["upstream"] == "http://127.0.0.1:34010"
    assert binding_calls[0][1]["checkpoint_profile"]["generation_backend"] == "sglang"
    health = service.health()
    assert health["generation_backend"] == "sglang"
    assert health["serving_engine"] == {"verified": True}
    assert health["max_new_tokens"] == 1024
    assert health["capabilities"]["serialized_generation"] is False


def test_sglang_requests_overlap_and_keep_atomic_finite_budget():
    reference, generator, tokenizer = _service("T0", "done")
    service = live.LiveNextCompressionService(
        generator, tokenizer, {**reference.profile, "generation_backend": "sglang"},
        reference.binding, ratio=8, mode="full", model="c2kv-next",
        max_new_tokens=1024, max_requests=2,
    )
    barrier = threading.Barrier(2)
    original_generate = generator.generate
    options = []

    def generate(memory, **kwargs):
        options.append(kwargs)
        barrier.wait(timeout=10)
        return original_generate(memory, **kwargs)

    generator.generate = generate
    with ThreadPoolExecutor(max_workers=2) as pool:
        responses = list(pool.map(lambda _: service.complete(_payload(stop=["END"])), range(2)))
    assert len({response["id"] for response in responses}) == 2
    assert all(item["stop_strings"] == ("END",) for item in options)
    assert all("token_prefix_stop" not in item for item in options)
    assert service.health()["counters"] == {
        "requests_started": 2, "requests_completed": 2, "requests_failed": 0,
        "requests_remaining": 0, "completion_tokens_emitted": 8,
    }
    with pytest.raises(live.LiveRequestError, match="Finite server request cap") as failure:
        service.complete(_payload())
    assert failure.value.status == 429


def test_reference_backend_stays_serial_and_is_labelled():
    service, _, _ = _service("T0", "done")
    assert service.health()["capabilities"]["serialized_generation"] is True
    response = service.complete(_payload())
    assert response["x_c2kv"]["generation_backend"] == "native"


def test_reference_stop_matches_inside_a_token_and_across_token_boundaries():
    class Tokenizer:
        def decode(self, ids, **kwargs):
            return "".join({1: "answer EN", 2: "D trailing", 3: "answer END trailing"}[i] for i in ids)

    callback = live._stop_callback(Tokenizer(), ("END",))
    assert callback((1,)) is False
    assert callback((1, 2)) is True
    assert callback((3,)) is True
    assert live._strip_stop(Tokenizer().decode((3,)), ("END",)) == "answer "
