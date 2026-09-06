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
    feedback = marker.parent / "env" / "smolagents" / "env.py"
    feedback.parent.mkdir(parents=True)
    feedback.write_text("def _record_execution_error(self, action, error):\n    pass\n",
                        encoding="utf-8")
    data = root / "experiments" / "smolagents" / "data" / "nq_multi_8"
    data.mkdir(parents=True)
    (root / "experiments" / "smolagents" / "run.py").write_text("# runner\n",
                                                                       encoding="utf-8")
    (data / "test.jsonl").write_text('{"id":"one"}\n', encoding="utf-8")
    return root


def _codes(result: capabilities.PreflightResult) -> set[str]:
    return {item.code for item in result.errors}


def test_toolsandbox_requires_empty_tool_call_normalization_for_both_roles(tmp_path):
    root = tmp_path / "toolsandbox"
    roles = root / "tool_sandbox" / "roles"
    roles.mkdir(parents=True)
    for role, endpoint in (("agent", "OPENAI_BASE_URL"), ("user", "TOOLSANDBOX_USER_BASE_URL")):
        (roles / f"openai_api_{role}.py").write_text(endpoint + "\n")
    options = {"runner_python": sys.executable, "toolsandbox_dir": str(root)}
    old = capabilities.preflight("toolsandbox", "full", options=options)
    assert _codes(old) == {"toolsandbox_agent_empty_tool_calls_patch", "toolsandbox_user_empty_tool_calls_patch"}
    for role in ("agent", "user"):
        path = roles / f"openai_api_{role}.py"
        path.write_text(path.read_text() + "if not openai_response_message.tool_calls:\n")
    assert capabilities.preflight("toolsandbox", "full", options=options).ok


def _acebench_tree(tmp_path: Path, *, role_history: bool = False) -> Path:
    root = tmp_path / "acebench"
    (root / "model_inference" / "multi_step").mkdir(parents=True)
    (root / "model_inference" / "multi_turn").mkdir(parents=True)
    (root / "generate.py").write_text("# generate\n", encoding="utf-8")
    (root / "eval_main.py").write_text("# eval\n", encoding="utf-8")
    (root / "model_inference" / "inference_map.py").write_text(
        "ACEBENCH_API_MODELS = ''\n", encoding="utf-8")
    if role_history:
        (root / "model_inference" / "role_history.py").write_text(
            "def agent_messages():\n    return []\n", encoding="utf-8")
        for test in ("multi_step", "multi_turn"):
            (root / "model_inference" / test / "APIModel_agent.py").write_text(
                "agent_messages()\n", encoding="utf-8")
    return root


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


def test_compression_ratio_outside_profile_is_an_explicit_warning(tmp_path):
    tau2 = _tau2(tmp_path)
    result = capabilities.preflight(
        "tau2", "c2kv", "sglang",
        options={"runner_python": sys.executable},
        profile={"serving": {"compression_ratios": [4]}},
        environ={"HOME": str(tmp_path), "TAU2_DIR": str(tau2)},
    )
    assert result.ok
    assert "compression_ratio_out_of_training_profile" in {
        item.code for item in result.warnings
    }


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


def test_acebench_declared_role_history_feature_requires_patched_files(tmp_path):
    root = _acebench_tree(tmp_path)
    options = {"runner_python": sys.executable, "acebench_dir": root,
               "capability_features": capabilities.ACE_ROLE_HISTORY_FEATURE}
    stale = capabilities.preflight("acebench", "c2kv", "sglang",
                                   options=options, environ=_env(tmp_path))
    assert {"acebench_role_history_helper", "acebench_role_history_multi_step_agent",
            "acebench_role_history_multi_turn_agent"} <= _codes(stale)

    ready = _acebench_tree(tmp_path / "ready", role_history=True)
    options["acebench_dir"] = ready
    assert capabilities.preflight("acebench", "c2kv", "sglang",
                                  options=options, environ=_env(tmp_path)).ok


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


def test_acon_qa_rejects_silent_execution_errors_even_with_retriever_ready(tmp_path):
    acon = _acon_qa_tree(tmp_path)
    env_file = acon / "src" / "productive_agents" / "env" / "smolagents" / "env.py"
    env_file.write_text("# Legacy execution errors leave observation empty.\n",
                        encoding="utf-8")
    result = capabilities.preflight(
        "acon_qa", "c2kv", "sglang",
        options={"acon_dir": acon, "runner_python": sys.executable},
        features=[capabilities.ACON_QA_RETRIEVER_FEATURE], environ=_env(tmp_path),
    )
    assert _codes(result) == {"acon_qa_error_feedback_patch"}


def test_text_method_variants_are_warnings_not_static_success_claims(tmp_path):
    tau2 = _tau2(tmp_path)
    result = capabilities.preflight(
        "tau2", "hiagent", "sglang", options={"runner_python": sys.executable},
        environ=_env(tmp_path, TAU2_DIR=str(tau2)),
    )
    assert result.ok
    assert {item.code for item in result.warnings} == {
        "hiagent_protocol_compliance_unverified",
    }
    assert result.variants[0]["status"] == "partial"

    full = capabilities.preflight(
        "tau2", "hiagent_full", "sglang", options={"runner_python": sys.executable},
        environ=_env(tmp_path, TAU2_DIR=str(tau2)),
    )
    assert "arm_capability:hiagent_trajectory_retrieval_v1" in _codes(full)

    acon = capabilities.preflight(
        "tau2", "acon_obs_ut_co", "sglang",
        options={"runner_python": sys.executable},
        environ=_env(tmp_path, TAU2_DIR=str(tau2)),
    )
    assert "acon_offline_guideline_optimizer_not_reproduced" in {
        item.code for item in acon.warnings
    }
    assert acon.variants == [{
        "name": "acon_obs_ut_co", "status": "partial",
        "detail": "fixed ut_co guideline for obs compression; offline optimizer is absent",
    }]
