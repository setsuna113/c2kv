# -*- coding: utf-8 -*-
"""Server-side pre-launch gate for the D downstream extension (runbook 4a).

Mechanical form of two checks that previously lived only as runbook prose
("green on the server" sentences get skipped under time pressure):

1. The REAL serving tokenizer's chat template keeps the block-prefix
   property AND injects no system header into a mid-conversation fragment.
   The in-loop relative-prefix assert in d_kv_intervene._downstream_rows
   cannot see a prologue COMMON to both templating calls (a Qwen2.5-style
   default system header would prefix both sides identically), so this test
   additionally asserts the absolute property: a user-only fragment renders
   with no system-role text at all.
2. The post-generation cache-length contract (pre + prompt + generated - 1)
   holds on the SERVED stack — the CPU suite pins it against transformers on
   this machine (test_downstream_crop_restores_pregen_cache), the NPU run
   must confirm it once before smoke.ok — plus one full K=1 continuation
   pass whose in-loop position/length tripwires assert internally.

Both tests SKIP unless the env vars point at real artifacts:

  C2KV_REAL_TOKENIZER_DIR  tokenizer dir (both tests)
  C2KV_SERVED_MODEL_DIR    served c2kv checkpoint dir (test 2 only)
  C2KV_GATE_DEVICE         device for test 2 (default "npu")

Green output of
  pytest agent/test_d_downstream_server_gate.py -v
is a prerequisite to writing smoke.ok (runbook 4a pre-launch checklist).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _sub in ("python", "python/inference", "agent"):
    _path = str(_REPO_ROOT / _sub)
    if _path not in sys.path:
        sys.path.insert(0, _path)

REAL_TOKENIZER_DIR = os.environ.get("C2KV_REAL_TOKENIZER_DIR")
SERVED_MODEL_DIR = os.environ.get("C2KV_SERVED_MODEL_DIR")

needs_tokenizer = pytest.mark.skipif(
    not REAL_TOKENIZER_DIR, reason="C2KV_REAL_TOKENIZER_DIR unset (server-only gate)"
)
needs_model = pytest.mark.skipif(
    not (REAL_TOKENIZER_DIR and SERVED_MODEL_DIR),
    reason="C2KV_REAL_TOKENIZER_DIR / C2KV_SERVED_MODEL_DIR unset (server-only gate)",
)


def _harness():
    import eval_agent_history_c2kv as HH  # noqa: PLC0415

    return HH


def _driver():
    import d_kv_intervene as D  # noqa: PLC0415

    return D


def _two_span_session():
    """Same shape as the torch tests' fixture: the later span's snapshot
    extends the earlier one exactly, the way real trace snapshots do.  No
    message content contains the word 'system', so the absolute no-header
    check below can string-match safely."""
    from train.train_data_multiturn import CompressHistoryExample  # noqa: PLC0415

    history = []
    for index, text in enumerate([
        "the wind moved sand across the wide flat plain",
        "grains hopped forward and struck the bed again",
        "a small mound grew behind the sheltering stone",
        "the leeward slope reached its resting angle",
        "an avalanche carried grains down to the base",
    ]):
        history.append({"role": "user" if index % 2 == 0 else "assistant", "content": text})

    conv0 = [
        {"role": "user", "content": "tell me about the dune field"},
        {"role": "assistant", "content": "sand collects where the wind slows"},
        {"role": "user", "content": "which way does the ridge travel"},
    ]
    conv1 = conv0 + [
        {"role": "assistant", "content": "downwind along the resultant direction"},
        {"role": "user", "content": "ripples formed on the windward face"},
    ]

    def _mk(qid, conv, answer):
        return CompressHistoryExample(
            qid=qid,
            history_messages=[dict(m) for m in history],
            current_messages=[dict(conv[-1])],
            answer=answer,
            system_prompt="you describe landforms",
            tools=[],
            original_messages=[dict(m) for m in conv],
        )

    return (
        _mk("dune:5", conv0, "downwind along the resultant direction"),
        _mk("dune:7", conv1, "small wavelengths ride the larger form"),
    )


def _real_tokenizer():
    from transformers import AutoTokenizer  # noqa: PLC0415

    return AutoTokenizer.from_pretrained(REAL_TOKENIZER_DIR, trust_remote_code=True)


@needs_tokenizer
def test_real_template_block_prefix_and_no_system_injection():
    HH = _harness()
    D = _driver()
    tokenizer = _real_tokenizer()
    prev, mid = _two_span_session()

    block, skip = D._continuation_block(prev, mid)
    assert skip is None and block

    # (a) relative property: the ids of the whole block are prefixed by the
    # ids of its first message alone — what the in-loop assert gates.
    ids_full = HH._chat_template_ids(tokenizer, block)
    ids_first = HH._chat_template_ids(tokenizer, block[:1])
    assert ids_full[: len(ids_first)] == ids_first

    # (b) absolute property: a prologue common to BOTH calls passes (a), so
    # additionally require that templating a user-only mid-conversation
    # fragment injects no system-role text at all (fixture content carries
    # no 'system' substring, so the string match is safe).
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": "plain continuation turn"}], tokenize=False
    )
    assert "system" not in rendered.lower(), (
        f"chat template injected a system header into a user-only fragment: {rendered!r}"
    )
    decoded = tokenizer.decode(ids_full, skip_special_tokens=False)
    assert "system" not in decoded.lower(), (
        "chat template injected a system header into the continuation block"
    )


def _gate_args(device_type: str):
    HH = _harness()
    argv = [
        "prog",
        "--model", str(SERVED_MODEL_DIR),
        "--tokenizer", str(REAL_TOKENIZER_DIR),
        "--device_type", device_type,
        # the frozen D recipe (HISTORY dialect 768/16)
        "--max_doc_length", "768",
        "--max_doc_num", "16",
        "--min_doc_num", "1",
        "--max_history_tokens", "12288",
        "--max_system_length", "4096",
        "--max_prompt_tokens", "1536",
        "--max_new_tokens", "16",
        "--override_ratio", "8",
        "--system_attn_impl", "eager",
        "--gist_attn_impl", "eager",
        "--generate_attn_impl", "eager",
    ]
    saved = sys.argv
    try:
        sys.argv = argv
        return HH.parse_args()
    finally:
        sys.argv = saved


@needs_model
def test_served_stack_generate_cache_contract_and_k1_pass():
    import argparse  # noqa: PLC0415

    HH = _harness()
    D = _driver()
    device_type = os.environ.get("C2KV_GATE_DEVICE", "npu")
    hargs = _gate_args(device_type)
    device = HH._setup_device(device_type)
    tokenizer = HH._load_tokenizer(hargs)
    hargs.model = HH._resolve_model_checkpoint(str(SERVED_MODEL_DIR))
    model = HH._load_model(hargs, tokenizer, device)
    prev, mid = _two_span_session()

    # (a) the -1 contract on the served stack (transformers pin + NPU path)
    prefix, skip = HH._build_c2kv_prefix(model, tokenizer, prev, hargs)
    assert skip is None, skip
    expected_phys = prefix["cache"].get_seq_length()
    metrics = HH._generate_with_prefix(model, tokenizer, prev, prefix, hargs, "c2kv")
    assert prefix["cache"].get_seq_length() == (
        expected_phys + metrics["prompt_tokens"] + metrics["generated_tokens"] - 1
    ), "post-generation cache-length contract (pre + prompt + generated - 1) violated"
    D._crop_cache(prefix["cache"], expected_phys)

    # (b) one full K=1 continuation pass: the in-loop position invariant and
    # length tripwires assert internally; a scored row must come out.
    prefix2, skip = HH._build_c2kv_prefix(model, tokenizer, prev, hargs)
    assert skip is None, skip
    ds_args = argparse.Namespace(
        downstream_turns=1,
        downstream_max_cache_tokens=28672,
        max_prompt_tokens=hargs.max_prompt_tokens,
        max_new_tokens=hargs.max_new_tokens,
        device_type=device_type,
    )
    rows = D._downstream_rows(
        model, tokenizer, prefix2, prev, [mid], hargs, ds_args, "c2kv", {}
    )
    assert [row["d_turn_offset"] for row in rows] == [1]
    assert not rows[0].get("skipped"), rows[0].get("skip_reason")
    assert rows[0]["target"] == mid.answer
