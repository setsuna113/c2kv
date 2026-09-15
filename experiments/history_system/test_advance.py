import json
import sys

import pytest

import advance


def test_running_peer_reserves_lane_without_recovery(tmp_path, monkeypatch):
    monkeypatch.setattr(advance, "HERE", tmp_path)
    monkeypatch.setattr(advance, "REPO", tmp_path)
    advance.save(tmp_path / "search_state.json", {"peer_completion":{"methods":{"raw":{}}}})
    out = tmp_path / "outputs/history_system_search/r001/peer_raw_base10"
    (out / "returned").mkdir(parents=True)
    advance.save(out / "launch.json", {"physical_device":4})
    monkeypatch.setattr(advance, "observe_peer", lambda method: {"pid_alive":True,
        "cells":[{"official_verified":True}, {"official_verified":False}]})
    monkeypatch.setattr(advance, "recover", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("live run recovered")))
    updates, occupied = [], set()
    advance.advance_peers(updates, occupied)
    assert occupied == {4}
    assert updates[0]["completed_tasks"] == 1


def test_terminal_peer_combines_returned_base_and_all_reused_long_roots(tmp_path, monkeypatch):
    monkeypatch.setattr(advance, "HERE", tmp_path)
    monkeypatch.setattr(advance, "REPO", tmp_path)
    advance.save(tmp_path / "search_state.json", {"peer_completion":{"methods":{"raw":{}}}})
    out = tmp_path / "outputs/history_system_search/r001/peer_raw_base10"
    (out / "returned").mkdir(parents=True)
    advance.save(out / "launch.json", {"physical_device":4})
    advance.save(out / "returned/recovery.validation.json", {})
    monkeypatch.setattr(advance, "recover", lambda *a, **kw: {"status":"passed_terminal_full_hash_recovery"})
    monkeypatch.setattr(advance, "load_config", lambda path: {"reused_long_result_roots":["long1", "long2"]})
    monkeypatch.setattr(advance, "build_index_design", lambda config: {"fixed_denominator":20})
    calls = []
    def collect(root, **kwargs):
        calls.append(kwargs)
        return {"quality":{"overall":{"unknown":1}}}
    monkeypatch.setattr(advance, "collect_candidate", collect)
    monkeypatch.setattr(advance, "render_markdown", lambda result: "test")
    advance.advance_peers([], set())
    advance.advance_peers([], set())
    assert len(calls) == 1
    assert calls[0]["results_root"] == [out / "returned", tmp_path / "long1", tmp_path / "long2"]
    assert calls[0]["task_manifest_path"] == tmp_path / "configs/r001.tasks.json"
    assert json.loads((out / "index.design.json").read_text())["fixed_denominator"] == 20


def test_native_queue_launch_precedes_terminal_archive_recovery(tmp_path, monkeypatch):
    monkeypatch.setattr(advance, "HERE", tmp_path)
    monkeypatch.setattr(advance, "REPO", tmp_path)
    monkeypatch.setattr(sys, "argv", ["advance.py", "--dispatch-next"])
    (tmp_path / "configs").mkdir()
    advance.save(tmp_path / "configs/r001.candidates.json", {"dispatch_enabled":True})
    base = tmp_path / "outputs/history_system_search/r001"
    for name in ("done", "next"):
        (base / name).mkdir(parents=True)
    advance.save(base / "done/launch.json", {"physical_device":4})
    advance.save(base / "done/analysis.json", {})
    advance.save(base / "next/upload.json", {})
    advance.save(base / "next/freeze.json", {})
    state = {"candidates":[{"candidate_id":"done", "output":str((base / "done").relative_to(tmp_path))}],
             "execution_queue":[{"candidate_id":"next", "physical_device":4, "port_base":28880}],
             "peer_completion":{"methods":{}, "execution_queue":[]}}
    advance.save(tmp_path / "search_state.json", state)
    monkeypatch.setattr(advance, "refresh_registry", lambda: advance.read(tmp_path / "search_state.json"))
    monkeypatch.setattr(advance, "observe", lambda *a: {"pid_alive":False,
        "stage":{"state":"completed", "completed_task_cells":20}})
    calls = []
    def launch(*args):
        calls.append("launch")
        return {"pid":1234}
    def recover(candidate):
        calls.append("recover")
        assert calls[0] == "launch", "free device was held behind archive recovery"
        return {"status":"passed_terminal_full_hash_recovery"}
    monkeypatch.setattr(advance, "launch", launch)
    monkeypatch.setattr(advance, "recover", recover)
    advance.main()
    assert calls == ["launch", "recover"]


def test_dispatch_disabled_rejects_before_refresh_or_remote_actions(
    tmp_path, monkeypatch, capsys
):
    reason = "fixed current algorithm; historical queue retained"
    monkeypatch.setattr(advance, "HERE", tmp_path)
    monkeypatch.setattr(sys, "argv", ["advance.py", "--dispatch-next"])
    (tmp_path / "configs").mkdir()
    advance.save(
        tmp_path / "configs/r001.candidates.json",
        {"dispatch_enabled": False, "dispatch_disabled_reason": reason},
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("dispatch guard allowed refresh or remote work")

    for name in (
        "refresh_registry",
        "observe",
        "observe_peer",
        "recover",
        "launch",
        "launch_peer",
        "advance_peers",
    ):
        monkeypatch.setattr(advance, name, forbidden)

    with pytest.raises(SystemExit) as captured:
        advance.main()
    assert captured.value.code == 2
    assert reason in capsys.readouterr().err
