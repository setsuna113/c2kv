"""Compatibility entry for the NPU generality scheduler."""
from __future__ import annotations

if __package__:
    from . import scheduler_npu as _implementation
else:
    import scheduler_npu as _implementation

main = _implementation.main


def __getattr__(name: str):
    return getattr(_implementation, name)


if __name__ == "__main__":
    raise SystemExit(main())