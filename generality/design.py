"""Frozen design constants for the generality-across-compressors experiment.

Source: output/handoffs/kv_compressor_generality_npu_20260918.md (2026-09-18).
All byte/token numbers here come from that handoff or from live engine probes
recorded in resolved_config.json; nothing in this file may be tuned by test
results.
"""
from __future__ import annotations

SCHEMA = "c2kv-generality-design-v1"

# --- fixed geometry (handoff section 3) -------------------------------------
KV_BYTES_PER_TOKEN = 147456          # verified live: /model_info c2kv_native_packed
K0_BYTES = 113246208                 # 108 MiB == 768 kv-token equivalents
K2_BYTES = 226492416                 # 216 MiB == 1536 kv-token equivalents
WORKING_POINTS = {
    "K0": {"history_allowance_bytes": K0_BYTES, "kv_token_equivalents": 768},
    "K2": {"history_allowance_bytes": K2_BYTES, "kv_token_equivalents": 1536},
}

# Engine page size on NPU (launch flag --page-size 128); actual slot usage after
# page rounding is recorded per cell, never silently equated with K/B.
NPU_PAGE_SIZE = 128

# Evidence unit rule (T02 evidence_sets_v1): U = tokens_1024, single candidate,
# R1 (at most one recovery per decision), retrieval_limit 24, candidate_limit 8.
EVIDENCE_UNIT = "tokens_1024"
UNIT_TOKEN_CAP = 1024

# --- conditions (handoff section 3) ------------------------------------------
CONDITIONS = (
    "compression_full_budget",     # bare backend, whole B, no Tracer machinery
    "tracer_history",              # initial allocation K, bounded append <= B, T02 weights + cell threshold
    "recovery_off_same_initial",   # same allocator and K as tracer, detector/recovery disabled
)

BACKENDS = ("c2kv", "h2o", "snapkv", "pyramidkv")
C2KV_RATIO = 4                     # fixed for every condition in this section

BENCHMARKS = {
    "bfcl_base": {"benchmark": "bfcl", "category": "multi_turn_base", "full_n": 200},
    "bfcl_long_context": {"benchmark": "bfcl", "category": "multi_turn_long_context", "full_n": 200},
    "appworld": {"benchmark": "acon_appworld", "split": "test_normal", "full_n": 168},
    "tau2": {"benchmark": "tau2", "task_set": "airline", "split": "base", "full_n": 50},
}

# Detector-development groups excluded from the BFCL held-out main table
# (handoff section 2; same-numbered base/long_context/miss_* belong to one group).
BFCL_TRAIN_SOURCE_GROUPS = [13, 16, 37, 39, 51, 55, 56, 66, 81, 90, 101, 118, 126, 134, 142, 181, 182]
BFCL_CALIBRATION_GROUPS = [3, 25, 43, 59, 68, 75, 129, 186, 188]

# --- caps and sampling (delivery runtime contract) ---------------------------
MAX_WORKSPACE_TOKENS = 36864
MAX_SEQUENCE_TOKENS = 40960
BFCL_MAX_COMPLETION_TOKENS = 4096
APPWORLD_MAX_COMPLETION_TOKENS = 2048
APPWORLD_MAX_ITER = 50
APPWORLD_SAMPLING = {
    "temperature": 0.0, "top_p": 1.0, "presence_penalty": 0.5,
    "seed": 42, "enable_thinking": False,
}
TASK_TIMEOUT_SECONDS = 10800

# Fixed T02 risk model weights: sha-pinned artifact, never refit.
T02_RISK_ARTIFACT_SHA256 = "18a11f73aa1f7d4b0add86eed66ae9e5e129ea4bdfbe0dfad23faf4f7d2fb4ab"
THRESHOLD_GRID = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
FALLBACK_THRESHOLD = 0.5
TAU2_T02_THRESHOLDS = {"c2kv": 0.6, "h2o": 0.3,
                       "snapkv": 0.3, "pyramidkv": 0.3}

# NPU assets (verified 2026-09-18 on ascend03)
ENGINE_MODEL_PATH = (
    "/home/liuyancheng/c2kv-b-final-20260912/checkpoints/b_history/"
    "arm-C/seed-42/checkpoint-1000"
)
EMBEDDING_MODEL_PATH = (
    "/home/liuyancheng/c2kv-evidence-sets-20260916/models/Qwen3-Embedding-0.6B"
)
T02_LABELS_PATH = "/home/liuyancheng/c2kv-evidence-sets-20260916/prepared_v8/run/labels.json"
T02_TASKS_PATH = "/home/liuyancheng/c2kv-evidence-sets-20260916/prepared_v8/configs/tasks.json"
T02_TRAINED_ARTIFACT_PATH = (
    "/home/liuyancheng/c2kv-evidence-sets-20260916/eval_trained_v5/C1/"
    "canonical_full20/lanes/C1/trained_artifact.json"
)
BFCL_DIR = "/home/liuyancheng/benchmarks/gorilla/berkeley-function-call-leaderboard"
ACON_DIR = "/home/liuyancheng/baselines/acon"
APPWORLD_DATA_ROOT = (
    "/home/liuyancheng/c2kv-a-runtime-20260907/system_search/delivery_20260914/"
    "benchmark_sources/appworld/data"
)
APPWORLD_PYTHON = (
    "/home/liuyancheng/c2kv-integration-followup-20260905/deps/venv-appworld/bin/python"
)
BFCL_PYTHON = "/home/liuyancheng/envs/bench/bin/python"
SGLANG_PYTHON = "/home/liuyancheng/envs/sgl/bin/python"

GENERATION_ROOT = "/home/liuyancheng/c2kv-generality-20260918"
SGEN = f"{GENERATION_ROOT}/src/sglang-gen"      # serving fork (sglang-paper@d2ca37175)
CONTROLLER = f"{GENERATION_ROOT}/src/generality/controller_runtime"
PAPER_HARNESS = f"{GENERATION_ROOT}/src/paper_harness"


def r_max_rule() -> dict:
    """R_max derivation rule; the concrete bytes are resolved in Phase 3 with
    the real tokenizer/formatter and NPU page rounding (see budget.py)."""
    return {
        "rule": "byte upper bound of ONE tokens_1024 evidence packet: "
                "page_round(unit_token_cap + wrapper_tokens) * kv_bytes_per_token",
        "unit_token_cap": UNIT_TOKEN_CAP,
        "page_size": NPU_PAGE_SIZE,
        "kv_bytes_per_token": KV_BYTES_PER_TOKEN,
        "applies_identically_to": ["all backends", "both working points", "both off controls"],
    }


def cells() -> int:
    return len(BACKENDS) * len(WORKING_POINTS) * len(CONDITIONS) * len(BENCHMARKS)
