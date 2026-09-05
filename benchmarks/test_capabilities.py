"""Focused compatibility-preflight tests; no external harness is launched."""
from __future__ import annotations

from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import capabilities


def _env(tmp_path: Path, **extra: str) -> dict[str, str]:
    return {"HOME": str(tmp_path), **extra}


def _tau2(tmp_path: Path) -> Path:
    path = tmp_path / "tau2"
    path.mkdir()
    return path


def _acon_qa_tree(tmp_path: Path) -> Path:
    root = tmp_path / "acon"
    marker = root / "src" / "productive_agents" / "llm.py"
    marker.parent.mkdir(parents=True)
    marker.write_text("base_url = os.environ.get('ACON_OPENAI_BASE_URL')\n",
                      encoding="utf-8")
    data = root / "experiments" / "smolagents" / "data" / "nq_multi_8"
    data.mkdir(parents=True)
    (root / "experiments" / "smolagents" / "run.py").write_text("# runner\n",
                                                                       encoding="utf-8")
    (data / "test.jsonl").write_text('{"id":"one"}\n', encoding="utf-8")
    return root


def _codes(result: capabilities.PreflightResult) -> set[str]:
    return {item.code for item in result.errors}


def test_cacheblend_overrides_profile_projection_but_requires_server_capability(tmp_path):
    tau2 = _tau2(tmp_path)
    common = {
        "options": {"runner_python": sys.executable},
        "environ": _env(tmp_path, TAU2_DIR=str(tau2)),
    }
    bad = capabilities.preflight(
        "tau2", "cacheblend_r16", "sglang",
        profile={"query_projection": "gist"}, **common,
    )
    assert _codes(bad) == {"cacheblend_server_capability"}
    assert bad.as_dict()["effective"] == {
        "query_projection": "base",
        "query_projection_source": "cacheblend_arm_override",
    }
    assert "cacheblend_query_projection_overridden" in {item.code for item in bad.warnings}

    good = capabilities.preflight(
        "tau2", "cacheblend_r16", "sglang",
        profile={"serving": {"query_projection": "gist"}, "server_features": [
            capabilities.CACHEBLEND_SERVER_FEATURE]}, **common,
    )
    assert good.ok
    assert good.variants == [{
        "name": "cacheblend_port_v1", "status": "partial",
        "detail": "turn-doc chunks and per-request materialisation; not the upstream artifact runtime",
    }]


def test_acebench_history_arms_are_rejected_until_normalizer_feature(tmp_path):
    result = capabilities.preflight(
        "acebench", "c2kv", "sglang",
        options={"runner_python": sys.executable}, environ=_env(tmp_path),
    )
    assert "acebench_role_history_normalizer" in _codes(result)

    full = capabilities.preflight(
        "acebench", "full", "sglang",
        options={"runner_python": sys.executable}, environ=_env(tmp_path),
    )
    assert "acebench_role_history_normalizer" not in _codes(full)


def test_acon_qa_requires_patch_data_and_explicit_retriever_attestation(tmp_path):
    acon = _acon_qa_tree(tmp_path)
    args = {"acon_dir": acon, "runner_python": sys.executable,
            "bench_python": sys.executable}
    blocked = capabilities.preflight("acon_qa", "c2kv", "sglang",
                                     options=args, environ=_env(tmp_path))
    assert "acon_qa_retriever" in _codes(blocked)

    allowed = capabilities.preflight(
        "acon_qa", "c2kv", "sglang", options=args,
        features=[capabilities.ACON_QA_RETRIEVER_FEATURE], environ=_env(tmp_path),
    )
    assert allowed.ok


def test_text_method_variants_are_warnings_not_static_success_claims(tmp_path):
    tau2 = _tau2(tmp_path)
    result = capabilities.preflight(
        "tau2", "hiagent", "sglang", options={"runner_python": sys.executable},
        environ=_env(tmp_path, TAU2_DIR=str(tau2)),
    )
    assert result.ok
    assert {item.code for item in result.warnings} == {
        "hiagent_protocol_compliance_unverified",
        "hiagent_trajectory_retrieval_unavailable",
    }
    assert result.variants[0]["status"] == "partial"
