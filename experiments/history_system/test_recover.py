import json

import recover


def test_terminal_wall_is_settled_once(tmp_path, monkeypatch):
    state_path = tmp_path / "search_state.json"
    state_path.write_text(json.dumps({
        "candidates":[{"candidate_id":"sample", "state":"running", "stage_wall_reserved_seconds":21600}],
        "completed_model_stage_seconds_carried":{"completed_model_stage_seconds":100},
    }))
    monkeypatch.setattr(recover, "HERE", tmp_path)
    monkeypatch.setattr(recover, "REPO", tmp_path)
    receipt = {"status":"passed_terminal_full_hash_recovery", "stage_state":"completed",
               "stage_status":"completed_fixed_manifest", "stage_wall_seconds":25,
               "files_verified":3}
    recover.settle("sample", tmp_path / "sample", receipt)
    recover.settle("sample", tmp_path / "sample", receipt)
    state = json.loads(state_path.read_text())
    assert state["completed_model_stage_seconds"] == 125
    assert len(state["settled_model_stages"]) == 1
    assert state["candidates"][0]["state"] == "completed"
    assert "stage_wall_reserved_seconds" not in state["candidates"][0]


def test_failed_peer_wall_is_settled_without_relabelling_quality(tmp_path, monkeypatch):
    path = tmp_path / "search_state.json"
    path.write_text(json.dumps({"candidates":[], "peer_completion":{"methods":{"raw":{"state":"running"}}},
        "completed_model_stage_seconds_carried":{"completed_model_stage_seconds":100}}))
    monkeypatch.setattr(recover, "HERE", tmp_path)
    monkeypatch.setattr(recover, "REPO", tmp_path)
    receipt = {"status":"passed_terminal_full_hash_recovery", "stage_state":"failed",
               "stage_status":"stage_wall_exhausted", "stage_wall_seconds":25, "files_verified":3}
    recover.settle("peer_raw_base10", tmp_path / "peer_raw_base10", receipt, peer_method="raw")
    recover.settle("peer_raw_base10", tmp_path / "peer_raw_base10", receipt, peer_method="raw")
    state = json.loads(path.read_text())
    assert state["completed_model_stage_seconds"] == 125
    assert state["peer_completion"]["methods"]["raw"]["state"] == "failed"
    assert "quality" not in state["peer_completion"]["methods"]["raw"]
