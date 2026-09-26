from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

import evidence_expansion_dispatch as dispatch


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _contract(package: Path, *, combination: bool, rows: list[dict]) -> Path:
    name = "combination_contract.json" if combination else "expansion_contract.json"
    path = package / name
    _write(path, {"shards": rows})
    for row in rows:
        shard_id = row["shard_id"]
        _write(package / "shards" / shard_id / "lanes" / shard_id / "lane.json", {
            "name": shard_id,
            "physical_device": row.get("preferred_device", row.get("device")),
            "engine_port": 42000 + len(row["shard_id"]),
            "task_ports": [43000 + len(row["shard_id"])],
        })
    return path


@pytest.mark.parametrize("combination", [False, True])
def test_package_uses_frozen_verifiers_and_normalizes_schedule(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, combination: bool) -> None:
    package = tmp_path / "package"
    rows = ([{"shard_id": "C0_part0", "device": 0}]
            if not combination else
            [{"shard_id": "h1_C0_part0", "preferred_device": 4, "wave": 2}])
    _contract(package, combination=combination, rows=rows)
    calls = []
    fake = SimpleNamespace(
        verify_package=lambda value: calls.append(("package", value)),
        verify_authorization=lambda value, auth: calls.append(("auth", value, auth)))
    monkeypatch.setattr(dispatch, "_load", lambda _path: fake)
    authorization = tmp_path / "authorization.json"
    _write(authorization, {})
    module, name, normalized = dispatch._package(package, authorization)
    assert module is fake
    assert name == ("evidence_eval_combinations.py" if combination
                    else "evidence_eval_expansion.py")
    assert normalized[0]["device"] == (4 if combination else 0)
    assert normalized[0]["wave"] == (2 if combination else 0)
    assert calls == [("package", package), ("auth", package, authorization)]


def test_package_rejects_any_existing_shard_run_state(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    package = tmp_path / "package"
    _contract(package, combination=False,
              rows=[{"shard_id": "C0_part0", "device": 0}])
    (package / "shards/C0_part0/lanes/C0_part0/run").mkdir()
    fake = SimpleNamespace(verify_package=lambda *_: None,
                           verify_authorization=lambda *_: None)
    monkeypatch.setattr(dispatch, "_load", lambda _path: fake)
    with pytest.raises(FileExistsError, match="already has run state"):
        dispatch._package(package, tmp_path / "authorization.json")


def test_dispatch_continues_after_failure_and_respects_per_device_wave(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    package = tmp_path / "package"
    rows = [
        {"shard_id": "first_d0", "device": 0, "wave": 0,
         "shard_package": str(package / "shards/first_d0")},
        {"shard_id": "first_d1", "device": 1, "wave": 0,
         "shard_package": str(package / "shards/first_d1")},
        {"shard_id": "second_d0", "device": 0, "wave": 1,
         "shard_package": str(package / "shards/second_d0")},
    ]
    for row in rows:
        _write(package / "shards" / row["shard_id"] / "lanes"
               / row["shard_id"] / "lane.json", {
                   "name": row["shard_id"], "physical_device": row["device"],
                   "engine_port": 42000 + row["device"],
                   "task_ports": [43000 + row["device"]]})
    authorization = tmp_path / "authorization.json"
    _write(authorization, {})
    module = SimpleNamespace()
    monkeypatch.setattr(dispatch, "_package",
                        lambda *_: (module, "frozen_runner.py", rows))
    free_calls = {"second_d0": 0}

    def lane(_module, _package, shard_id):
        def free(_lane):
            if shard_id == "second_d0" and free_calls[shard_id] == 0:
                free_calls[shard_id] += 1
                raise RuntimeError("Lane ports are occupied: [43000]")
        return SimpleNamespace(_assert_lane_free=free), {"name": shard_id}

    monkeypatch.setattr(dispatch, "_lane", lane)
    monkeypatch.setattr(dispatch.trained, "ascend_environment",
                        lambda: os.environ.__setitem__("ASCEND_FIXTURE", "loaded"))
    monkeypatch.setattr(dispatch.trained, "enable_strict_sampling",
                        lambda: os.environ.__setitem__("C2KV_STRICT_NONFINITE_SAMPLING", "1"))
    monkeypatch.setattr(dispatch.time, "sleep", lambda _seconds: None)
    events = []

    class Process:
        next_pid = 100

        def __init__(self, command, **kwargs):
            self.shard = command[command.index("--shard") + 1]
            self.pid = Process.next_pid
            Process.next_pid += 1
            self.polls = 0
            events.append(("start", self.shard, kwargs["env"]["ASCEND_FIXTURE"]))

        def poll(self):
            self.polls += 1
            if self.shard == "first_d1" and self.polls == 1:
                return None
            return 1 if self.shard == "first_d0" else 0

    monkeypatch.setattr(dispatch.subprocess, "Popen", Process)
    assert dispatch.dispatch(package, authorization, poll_seconds=0) == 2
    state = dispatch._read(package / "dispatch.json")
    statuses = {row["shard_id"]: row["status"] for row in state["shards"]}
    assert statuses == {"first_d0": "failed_no_retry", "first_d1": "completed",
                        "second_d0": "completed"}
    assert state["status"] == "partial_failed_no_retry"
    assert [event[1] for event in events] == ["first_d0", "first_d1", "second_d0"]
    assert free_calls["second_d0"] == 1
    assert not (package / "dispatch.json.tmp").exists()


def test_six_wave_zero_shards_start_before_polling(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    package = tmp_path / "package"
    package.mkdir()
    rows = [{"shard_id": f"s{device}", "device": device, "wave": 0,
             "shard_package": str(package / "shards" / f"s{device}")}
            for device in (0, 1, 2, 3, 4, 6)]
    monkeypatch.setattr(dispatch, "_package",
                        lambda *_: (SimpleNamespace(), "frozen.py", rows))
    monkeypatch.setattr(dispatch, "_lane", lambda _m, _p, shard: (
        SimpleNamespace(_assert_lane_free=lambda _lane: None), {"name": shard}))
    monkeypatch.setattr(dispatch.trained, "ascend_environment", lambda: None)
    monkeypatch.setattr(dispatch.trained, "enable_strict_sampling", lambda: None)
    events = []

    class Process:
        def __init__(self, command, **_kwargs):
            self.pid = 200 + len(events)
            self.shard = command[command.index("--shard") + 1]
            events.append(("start", self.shard))

        def poll(self):
            events.append(("poll", self.shard))
            return 0

    monkeypatch.setattr(dispatch.subprocess, "Popen", Process)
    assert dispatch.dispatch(package, tmp_path / "authorization.json", poll_seconds=0) == 0
    assert [kind for kind, _ in events[:6]] == ["start"] * 6


def test_existing_dispatch_state_refuses_before_environment_or_validation(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    package = tmp_path / "package"
    package.mkdir()
    _write(package / "dispatch.json", {"status": "failed_no_retry"})
    monkeypatch.setattr(dispatch, "_package",
                        lambda *_: pytest.fail("must not validate a second dispatch"))
    monkeypatch.setattr(dispatch.trained, "ascend_environment",
                        lambda: pytest.fail("must not reload environment"))
    with pytest.raises(FileExistsError, match="overwrite existing dispatch state"):
        dispatch.dispatch(package, tmp_path / "authorization.json", poll_seconds=0)


def test_only_known_occupancy_errors_are_waitable() -> None:
    assert dispatch._occupied(RuntimeError("Physical NPU 0 acquired by PIDs [9]"))
    assert dispatch._occupied(RuntimeError("Lane ports are occupied: [43000]"))
    assert not dispatch._occupied(RuntimeError("npu-smi failed"))


def test_real_frozen_combination_package_and_authorization_load_in_isolation(
        tmp_path: Path) -> None:
    import evidence_eval_combinations as combinations
    from test_evidence_eval_combinations import _authorization, _source, _spec

    package = tmp_path / "package"
    combinations.prepare(package, _spec(
        tmp_path, [_source(tmp_path, "C0"), _source(tmp_path, "C4_turn")]))
    authorization = tmp_path / "authorization.json"
    _write(authorization, _authorization(package))
    module, module_name, rows = dispatch._package(package, authorization)
    assert Path(module.__file__).resolve() == (
        package / "evidence_eval_combinations.py").resolve()
    assert module_name == "evidence_eval_combinations.py"
    assert len(rows) == 6
