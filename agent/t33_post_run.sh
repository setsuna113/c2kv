#!/usr/bin/env bash
set -uo pipefail
# t33 post-run pipeline: determinism gate -> feature extraction (both arms)
# -> scoring -> probes -> diff01 -> beta/gamma gate -> svip summary.
# Run on the server from ~/c2kv-t33 after the capture run completes.

REPO="${1:-$HOME/c2kv-t33}"
OUT="${T33_OUT:-/home/liuyancheng/c2kv/outputs_lyc/t33}"
RES="${T33_RES:-${REPO}/results/t33}"
CAP="${OUT}/capture"
TOKENIZER=/home/liuyancheng/c2kv/models/Qwen3-4B-Instruct-2507
FROZEN_FULL="${REPO}/results/bdf_pilot/d_r2/battery_full.jsonl"
FROZEN_C2KV="${REPO}/results/bdf_pilot/d_r2/battery_c2kv.jsonl"
MANIFEST="${REPO}/configs/bdf_pilot/d_cw_manifest_r2.json"
PY=""

cd "${REPO}"
source /usr/local/Ascend/ascend-toolkit/set_env.sh >/dev/null 2>&1 || true
for cand in "$HOME/envs/c2kv/bin/python" python3; do
  if command -v "$cand" >/dev/null 2>&1; then PY="$cand"; break; fi
done
export PYTHONPATH="${REPO}/python:${REPO}/python/inference:${REPO}/agent"
export PATH="$HOME/envs/c2kv/bin:$PATH"
mkdir -p "${RES}"

echo "== [1/7] determinism gate =="
# The old loop verified with a garbled frozen path (${FROZEN_FULL/c2kv/full}
# rewrote the repo dir, not the battery file), swallowed failures with
# `|| true`, and never gated the topup batteries.  A gate FAIL now stops the
# pipeline: downstream numbers do not inherit the manifest labels.
GATE_FAIL=0
gate_one() { # frozen rerun out
  "${PY}" agent/t33_verify_rerun.py --frozen "$1" --rerun "$2" --out "$3" || GATE_FAIL=1
}
gate_one "${FROZEN_FULL}" "${OUT}/battery_full.jsonl" "${RES}/gate_full.json"
gate_one "${FROZEN_C2KV}" "${OUT}/battery_c2kv.jsonl" "${RES}/gate_c2kv.json"
for arm in full c2kv; do
  for suffix in _topup _topup2; do
    tb="${OUT}/battery_${arm}${suffix}.jsonl"
    [ -f "${tb}" ] || continue
    if [ "${arm}" = full ]; then
      gate_one "${FROZEN_FULL}" "${tb}" "${RES}/gate_${arm}${suffix}.json"
    else
      gate_one "${FROZEN_C2KV}" "${tb}" "${RES}/gate_${arm}${suffix}.json"
    fi
  done
done
if [ "${GATE_FAIL}" != "0" ]; then
  echo "!! determinism gate FAILED — stopping (rerun is not the frozen battery)"
  exit 1
fi

echo "== [2/7] feature extraction =="
"${PY}" agent/t33_extract_features.py --capture_dir "${CAP}" --arm full \
  --tokenizer "${TOKENIZER}" --out "${RES}/features_full.jsonl"
"${PY}" agent/t33_extract_features.py --capture_dir "${CAP}" --arm c2kv \
  --tokenizer "${TOKENIZER}" --out "${RES}/features_c2kv.jsonl"

echo "== [3/7] scoring =="
"${PY}" agent/t33_score.py \
  --features_c2kv "${RES}/features_c2kv.jsonl" \
  --features_full "${RES}/features_full.jsonl" \
  --battery_full "${FROZEN_FULL}" --battery_c2kv "${FROZEN_C2KV}" \
  --manifest "${MANIFEST}" \
  --rows_c2kv "${OUT}/battery_c2kv.jsonl" \
  --out_dir "${RES}"

echo "== [4/7] probes =="
for arm in c2kv full; do
  "${PY}" agent/t33_fit_probes.py \
    --capture_dir "${CAP}" --arm "${arm}" \
    --battery_full "${FROZEN_FULL}" --battery_c2kv "${FROZEN_C2KV}" \
    --manifest "${MANIFEST}" \
    --features "${RES}/features_${arm}.jsonl" \
    --out "${RES}/probes_${arm}.json" || true
done

echo "== [5/7] diff-01 deferral =="
"${PY}" agent/t33_diff01.py \
  --features_c2kv "${RES}/features_c2kv.jsonl" \
  --features_full "${RES}/features_full.jsonl" \
  --battery_full "${FROZEN_FULL}" --battery_c2kv "${FROZEN_C2KV}" \
  --manifest "${MANIFEST}" \
  --rows_c2kv "${OUT}/battery_c2kv.jsonl" \
  --out "${RES}/diff01.json" || true

echo "== [6/7] beta/gamma gate + parameter-bearing denominator =="
"${PY}" agent/t33_beta_gamma.py --docs "${CAP}/c2kv/p0.docs.jsonl" \
  --out "${RES}/beta_gamma.json" || true
"${PY}" agent/t33_denominator.py \
  --c2kv "${FROZEN_C2KV}" --manifest "${MANIFEST}" \
  --out "${RES}/denominator.json" || true

echo "== [7/7] svip summary =="
"${PY}" - "${OUT}/svip/gamma.jsonl" "${RES}/svip_summary.json" <<'PYEOF'
import json, sys, os
src, dst = sys.argv[1], sys.argv[2]
rows = []
if os.path.exists(src):
    with open(src) as fh:
        rows = [json.loads(l) for l in fh if l.strip()]
ok = [r for r in rows if r.get("gamma_seq") is not None]
if ok:
    import statistics as st
    gs = sorted(r["gamma_seq"] for r in ok)
    pg = [r["p_gamma_le_136"] for r in ok if r.get("p_gamma_le_136") is not None]
    def pct(p):
        return gs[min(len(gs)-1, int(len(gs)*p))]
    out = {
        "n_scored": len(ok), "n_skipped": len(rows) - len(ok),
        "gamma_seq_median": round(st.median(gs), 4),
        "gamma_seq_p10": round(pct(0.10), 4), "gamma_seq_p90": round(pct(0.90), 4),
        "frac_gamma_le_1_36": round(sum(1 for g in gs if g <= 1.36) / len(gs), 4),
        # per-position certificate fraction: the sequence aggregate above says
        # "0% of rows", the per-position mean says ~17% of POSITIONS satisfy
        # gamma<=2c+1 — both are reported; the old summary quoted only the
        # sequence aggregate as "0%"
        "mean_p_gamma_le_136_per_position": round(sum(pg) / len(pg), 4) if pg else None,
        "n_rows_p_gamma": len(pg),
        "note": ("gamma = H_qp/H_q = 1 + KL/H_q >= 1 ALWAYS (mechanically large "
                 "when H_q is small; Pinsker bound direction unaffected but the "
                 "ratio is not a distance); sequence aggregate and per-position "
                 "fractions are different estimands, both shown; diagnostic "
                 "only per prereg"),
    }
else:
    out = {"n_scored": 0}
with open(dst, "w") as fh:
    json.dump(out, fh, indent=1)
print(json.dumps(out))
PYEOF

echo "== done; results in ${RES} =="
ls -la "${RES}"
