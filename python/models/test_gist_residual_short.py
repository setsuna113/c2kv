"""Regression coverage for embed-mean residuals on a sub-ratio document."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

_PYTHON_DIR = Path(__file__).resolve().parents[1]
if str(_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(_PYTHON_DIR))

from models.gist_utils import GistConfigMixin, get_apply_gist_residual_func  # noqa: E402


@pytest.mark.parametrize("residual_type", ["mean", "embed-mean"])
@pytest.mark.parametrize("supply_true_lens", [False, True])
def test_short_document_uses_the_partial_block_mean(residual_type, supply_true_lens):
    """A document shorter than its ratio has one gist row: its real-token mean."""
    ratio = 4
    tokens = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 6.0], [5.0, 10.0]],
            [[2.0, 4.0], [4.0, 8.0], [8.0, 16.0]],
        ],
        requires_grad=True,
    )
    gist = torch.tensor(
        [[[10.0, 20.0]], [[30.0, 40.0]]], requires_grad=True
    )
    config = GistConfigMixin(
        gist_type=f"interleave-{ratio}", gist_residual_type=residual_type
    )
    residual = get_apply_gist_residual_func(config, layer_idx=0)

    kwargs = {}
    if supply_true_lens:
        kwargs["gist_token_true_lens"] = torch.full((tokens.shape[0],), tokens.shape[1])
    actual = residual(tokens, gist, **kwargs)
    expected = tokens.mean(dim=1, keepdim=True) + gist
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    actual.sum().backward()
    torch.testing.assert_close(
        tokens.grad,
        torch.full_like(tokens, 1.0 / tokens.shape[1]),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(gist.grad, torch.ones_like(gist), rtol=0, atol=0)
