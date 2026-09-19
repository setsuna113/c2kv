"""BFCL provider imports must not take precedence over the SGL environment."""
import json
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "generality")]
from generality import calibrate


def test_bfcl_site_is_appended_after_existing_environment(tmp_path, monkeypatch):
    bench_site = tmp_path / "bench-site"
    bench_site.mkdir()
    monkeypatch.setattr(calibrate, "DEFAULT_BFCL_SITE", bench_site)
    imported = []
    monkeypatch.setattr(calibrate.importlib, "import_module",
                        lambda name: imported.append(name))
    before = sys.path[:]
    try:
        result = calibrate.configure_bfcl_site(None)
        assert sys.path[:len(before)] == before
        assert sys.path.index(str(bench_site.resolve())) >= len(before)
        assert result == {"source": "npu_default", "path": str(bench_site.resolve())}
        assert imported == ["bfcl_eval.eval_checker.eval_runner"]
    finally:
        sys.path[:] = before


def test_missing_bfcl_site_fails_before_a_model_request(tmp_path, monkeypatch):
    missing = tmp_path / "missing"
    with pytest.raises(FileNotFoundError, match="Explicit BFCL site-packages"):
        calibrate.configure_bfcl_site(str(missing))
    monkeypatch.setattr(calibrate, "DEFAULT_BFCL_SITE", missing)
    monkeypatch.setattr(calibrate.importlib, "import_module", lambda name: object())
    assert calibrate.configure_bfcl_site(None) == {
        "source": "current_environment", "path": None}

    def missing_provider(name):
        raise ModuleNotFoundError("No module named 'cohere'", name="cohere")

    monkeypatch.setattr(calibrate.importlib, "import_module", missing_provider)
    with pytest.raises(RuntimeError, match="cohere"):
        calibrate.configure_bfcl_site(None)


def test_protocol_records_bfcl_provider_source(tmp_path, monkeypatch):
    bench_site = tmp_path / "bench-site"
    bench_site.mkdir()
    monkeypatch.setattr(calibrate, "DEFAULT_BFCL_SITE", bench_site)
    monkeypatch.setattr(calibrate.importlib, "import_module", lambda name: object())
    monkeypatch.setattr(calibrate, "load_rows", lambda *args: [{"state_id": "state-1"}])
    monkeypatch.setattr(calibrate, "replay_one", lambda *args: {
        "risk": {"score": 0.5}, "label_status": "known", "label": 1})
    fake_models = types.ModuleType("benchmarks.memory_runtime.recovery.set_models")
    fake_models.C1RiskArtifact = lambda artifact: object()
    fake_bfcl = types.ModuleType("t02_bfcl")
    fake_bfcl.OfficialBFCLBindings = lambda: object()
    monkeypatch.setitem(sys.modules, fake_models.__name__, fake_models)
    monkeypatch.setitem(sys.modules, fake_bfcl.__name__, fake_bfcl)
    before = sys.path[:]
    try:
        out = tmp_path / "output"
        assert calibrate.main(["--backend", "h2o", "--wp", "K0",
                               "--engine-url", "http://unused", "--out", str(out),
                               "--smoke"]) == 2
        protocol = json.loads((out / "protocol.json").read_text())
        assert protocol["bfcl_site_packages"] == str(bench_site.resolve())
        assert protocol["bfcl_site_source"] == "npu_default"
    finally:
        sys.path[:] = before
