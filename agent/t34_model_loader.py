"""t34 shared model loader (SERVER: torch + the NPU model).

Three t34 drivers used to load checkpoints with a plain
``transformers.AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)``
and never moved the model off the CPU.  On the NPU box that meant (a) the
smoke ran on 100+ CPU threads (the retrieval-head needle detection sat at
11,000 % CPU for 10 minutes on 18 generations, 2026-09-06) and (b) for the
gist checkpoints the HF Qwen3 class silently DROPPED every gist parameter
(``gist_embed_tokens`` / ``gist_q_proj`` reported UNEXPECTED), so a
``--query-proj gist`` capture or a CacheBlend probe on ``fixed_joint`` would
have measured the base model.

This module routes every such load through the harness's own loader
(``eval_agent_tool_definition_c2kv._load_model``: the repo's
``python/models/qwen3`` class with the gist parameters for ``mode="c2kv"``,
the plain HF class for ``mode="full"``) on the requested device, with the
frozen r2 recipe's dtype/attention settings.  ``mode="auto"`` picks ``full``
when the checkpoint config carries no gist fields and ``c2kv`` otherwise.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional, Tuple

_HERE = Path(__file__).resolve().parent
for _p in (_HERE.parent / "python", _HERE.parent / "python" / "inference", _HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

GIST_CONFIG_KEYS = ("gist_param", "gist_type", "num_gist_tokens", "gist_residual_type",
                    "gist_overlap", "compression_ratio")


def checkpoint_mode(model_path: str, requested: str = "auto") -> str:
    """``full`` for a plain checkpoint, ``c2kv`` for a gist checkpoint; a
    non-``auto`` request is returned as given."""
    if requested in ("full", "c2kv"):
        return requested
    if requested != "auto":
        raise ValueError(f"mode must be auto|full|c2kv, got {requested!r}")
    cfg_path = Path(model_path) / "config.json"
    if not cfg_path.exists():
        return "full"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    return "c2kv" if any(k in cfg for k in GIST_CONFIG_KEYS) else "full"


def harness_namespace(model_path: str, *, mode: str, attn_impl: str = "eager",
                      tokenizer_path: Optional[str] = None,
                      base_model: Optional[str] = None, ratio: int = 8,
                      dataset_path: str = "") -> argparse.Namespace:
    """The eval-args namespace ``t33_svip_gamma._build_eval_args`` builds
    (frozen r2 recipe), with model/tokenizer/mode overridden."""
    import t33_svip_gamma as SV
    cli = argparse.Namespace(attn_impl=attn_impl, ratio=ratio, dataset_path=dataset_path,
                             tokenizer_path=tokenizer_path or model_path, model_path=model_path)
    ns = SV._build_eval_args(cli)
    ns.mode = mode
    ns.base_model = base_model
    return ns


def load_model_and_tokenizer(model_path: str, *, device_type: str = "npu",
                             attn_impl: str = "eager", mode: str = "auto",
                             tokenizer_path: Optional[str] = None,
                             base_model: Optional[str] = None,
                             ratio: int = 8) -> Tuple[Any, Any, str]:  # pragma: no cover - torch
    """(model, tokenizer, resolved_mode) on ``device_type`` via the harness loader."""
    from eval_agent_history_c2kv import _load_tokenizer, _resolve_model_checkpoint
    from eval_agent_tool_definition_c2kv import _load_model, _setup_device

    resolved = checkpoint_mode(model_path, mode)
    ns = harness_namespace(model_path, mode=resolved, attn_impl=attn_impl,
                           tokenizer_path=tokenizer_path, base_model=base_model, ratio=ratio)
    device = _setup_device(device_type)
    ns.model = _resolve_model_checkpoint(ns.model)
    tokenizer = _load_tokenizer(ns)
    model = _load_model(ns, tokenizer, device)
    model.eval()
    return model, tokenizer, resolved
