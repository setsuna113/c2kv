"""C1 v2 native servers pass the paper identity checks on non-BFCL benchmarks.

CPU only. The fixture is the box4 tau2 b512 ready manifest that 4dfab51
rejected, reduced to the fields the checks read (see its provenance block).
"""
import copy
import hashlib
import json
from pathlib import Path

import pytest

from benchmarks.paper import c1_appworld, native_extra

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "benchmarks" / "fixtures" / "c1v2_tau2_ready_b512_20260923.json"
ARM = "c2kv_c1_v2_verified_r8"
CONFIG = {"native_arm": ARM, "native_history_budget_tokens": 512}


def recorded():
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return copy.deepcopy(data["manifest"]), copy.deepcopy(data["controller"])


def bind(tmp_path, manifest, controller, benchmark="tau2", task="0"):
    """Bind the recorded manifest to a local controller file, as the server does."""
    controller_path = (tmp_path / "controller.json").resolve()
    data = json.dumps(controller, sort_keys=True).encode("utf-8")
    controller_path.write_bytes(data)
    manifest["s0_controller_contract"] = {
        "source": str(controller_path), "sha256": hashlib.sha256(data).hexdigest(),
        "config": controller}
    if benchmark != "tau2":
        manifest["benchmark"], manifest["source_profile"] = (
            ("acon_appworld", "openai-single-task-v1") if benchmark == "appworld"
            else native_extra.BENCHMARKS[benchmark])
    manifest["allowed_task_ids"] = [task]
    ready_path = tmp_path / "ready.json"
    ready_path.write_text(json.dumps(manifest), encoding="utf-8")
    return ready_path, controller_path


def test_recorded_c1v2_tau2_ready_manifest_passes(tmp_path):
    manifest, controller = recorded()
    ready, controller_path = bind(tmp_path, manifest, controller)
    assert native_extra.validate_ready_manifest(CONFIG, "tau2", "0", ready, controller_path)


def _leaves(value, path=()):
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _leaves(item, (*path, key))
    else:
        yield path


def _mutated(value):
    if isinstance(value, bool):
        return not value
    if isinstance(value, (int, float)):
        return value + 1
    return str(value) + "-changed"


def _set(document, path, value):
    for key in path[:-1]:
        document = document[key]
    document[path[-1]] = value


MANIFEST_FIELDS = sorted(
    [("candidate_algorithm", *path) for path in _leaves(recorded()[0]["candidate_algorithm"])]
    + [("route_contract", key) for key in
       ("baseline_identity", "recovery_enabled", "max_generations_per_decision")]
    + [("runtime_policy_contract", "effective_policy", key) for key in
       ("history_budget_bytes", "workspace_budget_bytes")]
    + [(key,) for key in ("schema", "status", "benchmark", "source_profile", "model_name",
                          "view_mode", "ratio", "generation_backend")])
CONTROLLER_FIELDS = sorted(
    ("candidate_algorithm", *path) for path in _leaves(recorded()[1]["candidate_algorithm"])
    if path[:1] != ("risk_artifact",) or path == ("risk_artifact", "model_kind"))


@pytest.mark.parametrize("path", MANIFEST_FIELDS, ids=".".join)
def test_changing_any_published_identity_field_is_rejected(tmp_path, path):
    manifest, controller = recorded()
    document = manifest
    for key in path:
        document = document[key]
    _set(manifest, path, _mutated(document))
    ready, controller_path = bind(tmp_path, manifest, controller)
    with pytest.raises(RuntimeError):
        native_extra.validate_ready_manifest(CONFIG, "tau2", "0", ready, controller_path)


@pytest.mark.parametrize("change", ["missing", "extra"])
def test_loaded_c1v2_contract_must_match_exactly(tmp_path, change):
    manifest, controller = recorded()
    if change == "missing":
        del manifest["candidate_algorithm"]["recovery_rounds_per_decision"]
    else:
        manifest["candidate_algorithm"]["unexpected"] = True
    ready, controller_path = bind(tmp_path, manifest, controller)
    with pytest.raises(RuntimeError, match="candidate controller identity differs"):
        native_extra.validate_ready_manifest(CONFIG, "tau2", "0", ready, controller_path)


@pytest.mark.parametrize("path", CONTROLLER_FIELDS, ids=".".join)
def test_changing_a_c1v2_controller_field_is_rejected(tmp_path, path):
    manifest, controller = recorded()
    document = controller
    for key in path:
        document = document[key]
    _set(controller, path, _mutated(document))
    ready, controller_path = bind(tmp_path, manifest, controller)
    with pytest.raises(RuntimeError, match="candidate controller identity differs"):
        native_extra.validate_ready_manifest(CONFIG, "tau2", "0", ready, controller_path)


@pytest.mark.parametrize("benchmark,task", [("toolsandbox", "wifi_off"),
                                            ("acebench_agent", "agent_multi_turn_1")])
def test_c1v2_identity_on_other_native_extra_benchmarks(tmp_path, benchmark, task):
    manifest, controller = recorded()
    ready, controller_path = bind(tmp_path, copy.deepcopy(manifest), controller, benchmark, task)
    native_extra.validate_ready_manifest(CONFIG, benchmark, task, ready, controller_path)
    manifest["route_contract"]["baseline_identity"] = "c2kv-paper-candidates-v1:c1_v2_verified"
    ready, controller_path = bind(tmp_path, manifest, controller, benchmark, task)
    with pytest.raises(RuntimeError, match="candidate controller identity differs"):
        native_extra.validate_ready_manifest(CONFIG, benchmark, task, ready, controller_path)


class Reached(Exception):
    pass


@pytest.mark.parametrize("valid", [True, False])
def test_appworld_run_task_checks_the_c1v2_identity_before_the_harness(tmp_path, monkeypatch, valid):
    manifest, controller = recorded()
    if not valid:
        manifest["candidate_algorithm"]["initial_view"]["terminal_rescue"] = "none"
    ready_path, controller_path = bind(tmp_path, manifest, controller, "appworld", "task_1")
    ready = dict(json.loads(ready_path.read_text(encoding="utf-8")),
                 base_url="http://127.0.0.1:39123/v1")   # not kept in the reduced fixture
    stopped = []
    monkeypatch.setattr(c1_appworld, "server_command", lambda *args: ["python", "--model-name", "m"])
    monkeypatch.setattr(c1_appworld, "_delivery_runner", lambda _delivery: type(
        "Runner", (), {"_stop_server": staticmethod(lambda *args: stopped.append(args))})())
    monkeypatch.setattr(c1_appworld.subprocess, "Popen", lambda *args, **kwargs: object())
    monkeypatch.setattr(c1_appworld, "_wait_ready", lambda *args: ready)

    def harness(*_args):
        raise Reached()

    monkeypatch.setattr(c1_appworld, "_run_official_harness", harness)
    expected = Reached if valid else RuntimeError
    with pytest.raises(expected):
        c1_appworld.run_task(CONFIG, "task_1", tmp_path / "native", tmp_path, controller_path)
    assert stopped   # the server is always stopped
