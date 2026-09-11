#!/usr/bin/env python3
"""Run manifest-bound native BFCL evaluation for B/C checkpoints."""
from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

from history_memory.checkpoint_eval import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
