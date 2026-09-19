import csv
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "generality" / "cellplan.py"


@pytest.fixture
def cellplan(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("cellplan_under_test", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "GENERATION_ROOT", tmp_path)
    monkeypatch.setattr(module, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(module, "CONFIG", tmp_path / "config")
    monkeypatch.setattr(module, "BACKENDS", ("c2kv",))
    monkeypatch.setattr(module, "WORKING_POINTS", ("K0",))
    monkeypatch.setattr(module, "CONDITIONS", ("compression_full_budget",))
    monkeypatch.setattr(module, "BENCH_KEYS", ("bfcl_base",))
    monkeypatch.setattr(module, "EXTRA_BENCH_KEYS", ("toolsandbox", "acebench"))
    monkeypatch.setattr(module, "bfcl_task_ids", lambda category: ["multi_turn_base_1"])
    monkeypatch.setattr(module, "appworld_task_ids", lambda: pytest.fail("AppWorld was not selected"))
    monkeypatch.setattr(module, "serving_provenance", lambda: {
        "checkout": str(tmp_path / "src" / "sglang-gen"),
        "commit": "a" * 40, "worktree_dirty": True,
        "engine_flags_extra": ["--max-running-requests", "4"],
        "note": "test source",
    })
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "rmax_measurement.json").write_text(json.dumps({
        "recovery_allowance_bytes": 147456,
        "packet_resident_tokens_after_page_rounding": 1,
    }))
    return module


def _cell_path(cellplan, bench="bfcl_base"):
    return (cellplan.RESULTS / "closed_loop" / bench / "c2kv" / "K0" /
            "compression_full_budget" / "cell.json")


def test_import_and_default_plan_do_not_require_optional_manifests(cellplan):
    assert cellplan.main([]) == 0
    assert _cell_path(cellplan).exists()
    assert not _cell_path(cellplan, "toolsandbox").exists()
    assert cellplan.main([]) == 0


def test_empty_bench_list_is_rejected_before_writes(cellplan):
    with pytest.raises(SystemExit):
        cellplan.main(["--benches"])
    assert not (cellplan.GENERATION_ROOT / "matrix.csv").exists()


def test_incremental_panel_preserves_calibrated_cell_and_counts(cellplan, tmp_path, monkeypatch):
    cellplan.main([])
    cell_path = _cell_path(cellplan)
    existing = json.loads(cell_path.read_text())
    existing["threshold"] = 0.3
    existing["threshold_status"] = "calibrated"
    cell_path.write_text(json.dumps(existing))
    done = cell_path.parent / "tasks" / "multi_turn_base_1" / "done.json"
    done.parent.mkdir(parents=True)
    done.write_text('{"status": "completed"}')
    ts_path = tmp_path / "ts.json"
    ts_path.write_text(json.dumps({"full": ["scenario_a", "scenario_b"], "test": []}))
    monkeypatch.setattr(cellplan, "TS_SCENARIO_NAMES", str(ts_path))

    assert cellplan.main(["--benches", "toolsandbox"]) == 0
    assert json.loads(cell_path.read_text()) == existing
    assert done.read_text() == '{"status": "completed"}'
    resolved = json.loads((tmp_path / "config" / "resolved_config.json").read_text())
    assert resolved["n_cells"] == 2
    assert resolved["n_task_executions"] == 3
    with (tmp_path / "matrix.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert {row["benchmark"] for row in rows} == {"bfcl_base", "toolsandbox"}
    assert next(row for row in rows if row["benchmark"] == "bfcl_base")[
        "threshold_status"] == "calibrated"


def test_conflict_preflight_leaves_all_outputs_unchanged(cellplan, tmp_path, monkeypatch):
    cellplan.main([])
    original_matrix = (tmp_path / "matrix.csv").read_bytes()
    original_resolved = (tmp_path / "config" / "resolved_config.json").read_bytes()
    cell_path = _cell_path(cellplan)
    existing = json.loads(cell_path.read_text())
    existing["budget_bytes"]["B"] += 1
    cell_path.write_text(json.dumps(existing))
    ts_path = tmp_path / "ts.json"
    ts_path.write_text(json.dumps({"full": ["scenario_a"], "test": []}))
    monkeypatch.setattr(cellplan, "TS_SCENARIO_NAMES", str(ts_path))

    with pytest.raises(RuntimeError, match="existing cell contract differs"):
        cellplan.main(["--benches", "bfcl_base", "toolsandbox"])
    assert (tmp_path / "matrix.csv").read_bytes() == original_matrix
    assert (tmp_path / "config" / "resolved_config.json").read_bytes() == original_resolved
    assert not _cell_path(cellplan, "toolsandbox").exists()


def test_changed_serving_revision_cannot_relabel_old_matrix(cellplan, tmp_path, monkeypatch):
    cellplan.main([])
    original = (tmp_path / "matrix.csv").read_bytes()
    monkeypatch.setattr(cellplan, "serving_provenance", lambda: {
        "checkout": str(tmp_path / "src" / "sglang-gen"),
        "commit": "b" * 40, "worktree_dirty": True,
        "engine_flags_extra": ["--max-running-requests", "4"],
        "note": "test source",
    })
    with pytest.raises(RuntimeError, match="serving provenance differs"):
        cellplan.main([])
    assert (tmp_path / "matrix.csv").read_bytes() == original


def test_serving_provenance_reads_checkout_head_and_dirty_state(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("cellplan_provenance_test", MODULE_PATH)
    cellplan = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cellplan)
    monkeypatch.setattr(cellplan, "GENERATION_ROOT", tmp_path)
    checkout = tmp_path / "src" / "sglang-gen"
    checkout.mkdir(parents=True)
    def git(*args):
        return subprocess.run(["git", "-C", str(checkout), *args],
                              capture_output=True, text=True, check=True).stdout.strip()
    git("init", "-q")
    git("config", "user.name", "Cellplan Test")
    git("config", "user.email", "cellplan-test@example.invalid")
    (checkout / "source.py").write_text("VALUE = 1\n")
    git("add", "source.py")
    git("commit", "-qm", "Test source")
    clean = cellplan.serving_provenance()
    assert clean["commit"] == git("rev-parse", "HEAD")
    assert clean["worktree_dirty"] is False
    (checkout / "source.py").write_text("VALUE = 2\n")
    assert cellplan.serving_provenance()["worktree_dirty"] is True
