#!/usr/bin/env bash
set -uo pipefail
# t33 post-run pipeline: determinism gate -> feature extraction (both arms)
# -> scoring -> probes -> diff01 -> beta/gamma gate -> svip summary.
# Run on the server from ~/c2kv-t33 after the capture run completes.

REPO="${1:-$HOME/c2kv-t33}"
OUT="${T33_OUT:-/home/liuyancheng/c2kv/outputs_lyc/t33}"
RES="${REPO}/results/t33"
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
for arm in full c2kv; do
  "${PY}" agent/t33_verify_rerun.py \
    --frozen "${FROZEN_FULL/c2kv/full}" \
    --rerun "${OUT}/battery_${arm}.jsonl" \
    --out "${RES}/gate_${arm}.json" || true
done
# correct the full/c2kv frozen pairing explicitly
"${PY}" agent/t33_verify_rerun.py --frozen "${FROZEN_FULL}" --rerun "${OUT}/battery_full.jsonl" --out "${RES}/gate_full.json" || true
"${PY}" agent/t33_verify_rerun.py --frozen "${FROZEN_C2KV}" --rerun "${OUT}/battery_c2kv.jsonl" --out "${RES}/gate_c2kv.json" || true

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

echo "== [6/7] beta/gamma gate =="
"${PY}" agent/t33_beta_gamma.py --docs "${CAP}/c2kv/p0.docs.jsonl" \
  --out "${RES}/beta_gamma.json" || true

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
    def pct(p):
        return gs[min(len(gs)-1, int(len(gs)*p))]
    out = {
        "n_scored": len(ok), "n_skipped": len(rows) - len(ok),
        "gamma_seq_median": round(st.median(gs), 4),
        "gamma_seq_p10": round(pct(0.10), 4), "gamma_seq_p90": round(pct(0.90), 4),
        "frac_gamma_le_1_36": round(sum(1 for g in gs if g <= 1.36) / len(gs), 4),
        "note": "gamma = H_qp/H_q on frozen emitted text under same-checkpoint c2kv/full prefixes; diagnostic only per prereg",
    }
else:
    out = {"n_scored": 0}
with open(dst, "w") as fh:
    json.dump(out, fh, indent=1)
print(json.dumps(out))
PYEOF

echo "== done; results in ${RES} =="
ls -la "${RES}"
