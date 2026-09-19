"""Compatibility entrypoint for complete-cell BFCL offline rescoring.

Usage: python -m generality.rescore --cell-dir CELL --out NEW_DIR
       [--score --benchmark-dir BFCL_CHECKOUT]
"""
from __future__ import annotations

try:
    from .bfcl_rescore import main
except ImportError:
    from bfcl_rescore import main


if __name__ == "__main__":
    raise SystemExit(main())
