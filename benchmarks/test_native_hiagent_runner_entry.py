"""Native HiAgent CLI-to-proxy wiring, without processes or model calls."""
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run
from capabilities import preflight


def test_official_handler_sampling_matches_explicit_bridge_config(tmp_path):
    path = tmp_path / "sampling.json"
    path.write_text(json.dumps({
        "schema": "a-native-hiagent-policy-sampling-v1", "temperature": 0.001,
        "seed": 42, "max_completion_tokens": 4096,
    }), encoding="utf-8")
    args = SimpleNamespace(backend="event_native_hiagent",
        native_hiagent_policy_sampling=str(path), bfcl_temperature=None,
        bfcl_seed=None, bfcl_generation_max_tokens=4096)
    run.align_native_hiagent_sampling(args)
    assert (args.bfcl_temperature, args.bfcl_seed, args.bfcl_generation_max_tokens) == (0.001, 42, 4096)
    args.bfcl_temperature = 0
    with pytest.raises(ValueError, match="conflicts"):
        run.align_native_hiagent_sampling(args)


def test_real_b500_metadata_resolves_without_loading_model_weights(tmp_path):
    checkpoint = (
        Path(__file__).resolve().parents[1]
        / "tmp/a_memory_runtime_20260912/native_hiagent_preparation_v1/b500_metadata"
    )
    config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    state = json.loads((checkpoint / "trainer_state.json").read_text(encoding="utf-8"))
    args = run.build_parser().parse_args([
        "--benchmark", "bfcl", "--arm", "hiagent_full_native",
        "--backend", "event_native_hiagent", "--upstream", "http://127.0.0.1:26000",
        "--out", str(tmp_path), "--checkpoint", str(checkpoint),
    ])

    profile = run.resolve_run_profile(args)

    assert profile["profile_kind"] == "event_native_checkpoint"
    assert profile["training_arm"] == config["history_memory_arm"] == "B"
    assert profile["parameter_version"] == state["parameter_version"] == 500
    assert profile["corpus_identity"] == state["contract"]["corpus_identity"]
    assert profile["declared_supported_ratios"] == config["history_memory_supported_ratios"]
    assert profile["model_geometry"]["max_position_embeddings"] == config["max_position_embeddings"]
    assert profile["packing_contract"] == config["history_memory_packing"]
    assert profile["policy_contract"] == config["history_memory_policy"]
    assert len(profile["profile_fingerprint"]) == 64
    assert (args.doc_packing, args.max_doc_length, args.max_doc_num, args.query_projection) == (
        "turn", 512, 12, None,
    )


def test_native_profile_rejects_legacy_profile_override(tmp_path):
    checkpoint = (
        Path(__file__).resolve().parents[1]
        / "tmp/a_memory_runtime_20260912/native_hiagent_preparation_v1/b500_metadata"
    )
    args = run.build_parser().parse_args([
        "--benchmark", "bfcl", "--arm", "hiagent_full_native",
        "--backend", "event_native_hiagent", "--upstream", "http://127.0.0.1:26000",
        "--out", str(tmp_path), "--checkpoint", str(checkpoint),
        "--reference-profile", "checkpoint-1088",
    ])
    with pytest.raises(run.ProfileError, match="legacy checkpoint/reference profiles"):
        run.resolve_run_profile(args)


def test_sampling_file_reaches_proxy_process(monkeypatch, tmp_path):
    import urllib.request

    seen = {}
    process = SimpleNamespace(poll=lambda: None)

    def spawn(command, **kwargs):
        seen.update(command=command, kwargs=kwargs)
        return process

    monkeypatch.setattr(run, "_require_available_proxy_port", lambda port: None)
    monkeypatch.setattr(run.subprocess, "Popen", spawn)
    monkeypatch.setattr(run.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(urllib.request, "build_opener", lambda *args:
                        SimpleNamespace(open=lambda *a, **k: io.BytesIO(b"{}")))
    config = str(tmp_path / "explicit-sampling.json")
    args = run.build_parser().parse_args([
        "--benchmark", "bfcl", "--arm", "hiagent_full_native",
        "--backend", "event_native_hiagent", "--upstream", "http://127.0.0.1:26000",
        "--out", str(tmp_path), "--native-hiagent-policy-sampling", config,
    ])
    child, _ = run.start_proxy(
        args.upstream, args.arm, 26001, tmp_path, backend=args.backend,
        native_hiagent_policy_sampling=args.native_hiagent_policy_sampling,
        no_upstream_retries=True, capture_request_views=True,
        max_generation_attempts_per_task=96,
    )
    assert child is process
    command = seen["command"]
    assert command[command.index("--native-hiagent-policy-sampling") + 1] == config
    assert command[command.index("--backend") + 1] == "event_native_hiagent"
    assert seen["kwargs"]["env"]["TORCH_DEVICE_BACKEND_AUTOLOAD"] == "0"


def test_preflight_rejects_incomplete_native_contract():
    options = {
        "native_hiagent_policy_sampling": "explicit.json",
        "max_generation_attempts_per_task": 96,
        "no_upstream_retries": True,
        "capture_request_views": True,
    }

    def admitted(opts, benchmark="bfcl"):
        result = preflight(benchmark, "hiagent_full_native", "event_native_hiagent",
                           options=opts, features=["hiagent_trajectory_retrieval_v1"])
        return next(item.satisfied for item in result.requirements
                    if item.code == "native_hiagent_explicit_contract")

    assert admitted(options)
    assert not admitted({**options, "native_hiagent_policy_sampling": ""})
    assert not admitted({**options, "max_generation_attempts_per_task": 97})
    assert not admitted({**options, "no_upstream_retries": False})
    assert not admitted(options, "tau2")
