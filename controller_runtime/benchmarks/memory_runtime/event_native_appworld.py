"""Launch the canonical AppWorld worker from source or deployed layout."""
from pathlib import Path
import runpy


def main():
    root = Path(__file__).resolve().parents[3]
    # Repository: root/generality/. Deployment: src/generality/ itself.
    source = root / "generality" / "event_native_appworld.py"
    if not source.is_file():
        source = root / "event_native_appworld.py"
    if not source.is_file():
        raise FileNotFoundError(f"canonical AppWorld worker missing beneath {root}")
    runpy.run_path(str(source), run_name="__main__")


if __name__ == "__main__":
    main()
