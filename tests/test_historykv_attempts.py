import json
from generality.historykv_cell import (
    bfcl_row_healthy, cached_terminal, new_attempt_root, write_receipt,
)


def test_retry_preserves_existing_outputs_and_receipt(tmp_path):
    old = tmp_path / "appworld"
    old.mkdir()
    (old / "evidence.txt").write_text("original")
    write_receipt(tmp_path, {"status": "failed", "error": "connection"})
    assert cached_terminal(tmp_path) is None
    first = new_attempt_root(tmp_path, "appworld")
    second = new_attempt_root(tmp_path, "appworld")
    assert first != second
    assert (old / "evidence.txt").read_text() == "original"
    write_receipt(tmp_path, {"status": "infra_error"})
    saved = json.loads((tmp_path / "status_history.jsonl").read_text())
    assert json.loads(saved["previous_raw"])["error"] == "connection"


def test_typed_method_failure_is_terminal_but_transport_is_not(tmp_path):
    result = tmp_path / "result"
    result.mkdir()
    path = result / "result.json"
    for failure, expected in [
        ('HTTP 422 {"code":"c2kv_capacity_infeasible"}', True),
        ("HTTP 502 connection refused", False),
        (None, True),
    ]:
        path.write_text(json.dumps({"id": "task_1", "result": [], "traceback": failure}))
        assert bfcl_row_healthy(tmp_path, "task_1") is expected


def test_production_holds_cannot_be_overridden():
    from generality.scheduler_npu import cells_ready
    for backend in ("c2kv", "h2o", "snapkv", "pyramidkv"):
        for benchmark in ("bfcl_base", "bfcl_long_context"):
            assert not cells_ready({"backend": backend, "benchmark_key": benchmark,
                                    "condition": "compression_full_budget"})
    assert not cells_ready({"backend": "pyramidkv", "benchmark_key": "appworld",
                            "condition": "compression_full_budget"})
