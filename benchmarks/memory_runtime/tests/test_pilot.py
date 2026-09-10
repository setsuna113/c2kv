import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "benchmarks"))
from memory_runtime import pilot


def startup_receipt():
    corpus, _ = pilot.load_corpus()
    variants, run_id = pilot.load_variants(ROOT / "benchmarks/memory_runtime/configs")
    return dict(
        status="failed", run_id=run_id, requests_attempted=4, responses_received=4,
        request_receipts=[dict(variant=v.spec.name, case_id=c["case_id"]) for v in variants[:2] for c in corpus["cases"]],
        variants=[dict(name=v.spec.name, runtime=dict(v.config) if v.config else None) for v in variants],
        error=dict(message="proxy did not come up on port 12345"),
    )


def invoke(tmp_path, prior):
    path = tmp_path / "prior.json"
    path.write_text(json.dumps(prior), encoding="utf-8")
    return pilot.run_protocol(upstream="http://127.0.0.1:1", tokenizer="local",
                              config_dir=ROOT / "benchmarks/memory_runtime/configs",
                              out_dir=tmp_path / "next", proxy_python=sys.executable,
                              completed_receipt=path)


@pytest.mark.parametrize("change", ["ambiguous_request", "changed_budget", "failed_model"])
def test_continuation_refuses_model_retry_or_changed_protocol(tmp_path, monkeypatch, change):
    prior = startup_receipt()
    if change == "ambiguous_request":
        prior["requests_attempted"] = 5
    elif change == "changed_budget":
        prior["variants"][2]["runtime"]["history_budget_bytes"] += 1
    else:
        prior["error"]["message"] = "HTTP 502"
    monkeypatch.setattr(pilot, "_launch_proxy", lambda *a: pytest.fail("must reject before launch"))
    with pytest.raises(pilot.PilotFailure, match="startup-only failure"):
        invoke(tmp_path, prior)


def test_continuation_sends_only_the_twelve_unattempted_requests(tmp_path, monkeypatch):
    active = {}
    sent = []

    class Process:
        pid = 1
        returncode = 0
        def poll(self):
            return 0

    def launch(upstream, variant, port, log_dir, proxy_python, tokenizer):
        active["variant"] = variant
        return Process(), log_dir / (variant.spec.name + ".jsonl")

    def post(endpoint, payload):
        v = active["variant"]
        sent.append(v.spec.name)
        meta = dict(version=pilot.RUNTIME_VERSION, mode=v.spec.name,
                    **payload["c2kv_eval_context"], bytes_per_kv_token=v.config["bytes_per_kv_token"],
                    byte_geometry_verified_by_backend=True, active_history_bytes=0,
                    evidence_bytes=0, gist_tokens=0, budget_applies=v.spec.name != "full_shared",
                    evicted_gist_keys=[], protected_event_ids=[], retrieved_event_ids=[], retained_event_ids=[])
        response = dict(choices=[dict(message=dict(content="ok"), finish_reason="stop")],
                        c2kv_proxy=dict(memory_runtime=meta))
        return 200, response, json.dumps(response), 0.0

    monkeypatch.setattr(pilot, "_launch_proxy", launch)
    monkeypatch.setattr(pilot, "_post_chat", post)
    receipt = invoke(tmp_path, startup_receipt())
    assert receipt["status"] == "completed"
    assert receipt["requests_attempted"] == 12
    assert receipt["aggregate_planned_requests"] == 16
    assert len(sent) == 12 and "full" not in sent and "full_native" not in sent
