from pathlib import Path
from unittest.mock import patch

from controller_runtime.benchmarks.memory_runtime import event_native_appworld


def test_wrapper_launches_tracked_canonical_worker():
    root = Path(__file__).resolve().parents[1]
    expected = root / "generality" / "event_native_appworld.py"
    assert expected.is_file()

    with patch.object(event_native_appworld.runpy, "run_path") as run_path:
        event_native_appworld.main()

    run_path.assert_called_once_with(str(expected), run_name="__main__")
