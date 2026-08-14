#!/usr/bin/env bash
# Round-2 archive freeze: snapshot the round-1 agent-eval outputs read-only.
#
# Copies all files matching *agent*.jsonl and *.summary.json directly under
# SRC_DIR (non-recursive, cp -p, source never modified) into DST_DIR, then
# writes a manifest.tsv (relpath<TAB>sha256<TAB>bytes<TAB>mtime_iso) and
# manifest.sha256 (sha256sum of manifest.tsv, verifiable with sha256sum -c)
# into DST_DIR, and prints per-pattern counts.
#
# Refuses to write into an existing non-empty DST_DIR unless --force is given.
#
# Usage: bash agent/forensics_freeze_archive.sh SRC_DIR DST_DIR [--force]
set -euo pipefail

usage() {
  echo "usage: $0 SRC_DIR DST_DIR [--force]" >&2
  exit 2
}

[[ $# -ge 2 ]] || usage
SRC_DIR="$1"
DST_DIR="$2"
shift 2
FORCE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --force) FORCE=1 ;;
    *) usage ;;
  esac
  shift
done

[[ -d "${SRC_DIR}" ]] || { echo "error: SRC_DIR not found: ${SRC_DIR}" >&2; exit 1; }

if [[ -d "${DST_DIR}" ]] && [[ -n "$(ls -A "${DST_DIR}")" ]]; then
  if [[ "${FORCE}" != "1" ]]; then
    echo "error: refusing to overwrite non-empty DST_DIR=${DST_DIR} (pass --force)" >&2
    exit 1
  fi
fi
mkdir -p "${DST_DIR}"

# Copy phase: read-only w.r.t. the source (cp -p preserves mtime/perms).
n_jsonl=0
n_summary=0
while IFS= read -r -d '' src; do
  cp -p "${src}" "${DST_DIR}/"
  case "$(basename "${src}")" in
    *.summary.json) n_summary=$((n_summary + 1)) ;;
    *) n_jsonl=$((n_jsonl + 1)) ;;
  esac
done < <(find "${SRC_DIR}" -maxdepth 1 -type f \
  \( -name '*agent*.jsonl' -o -name '*.summary.json' \) -print0 | sort -z)

# Manifest phase: hash the copies (catches copy corruption), sorted by name.
manifest="${DST_DIR}/manifest.tsv"
: > "${manifest}"
total_bytes=0
while IFS= read -r -d '' dst; do
  rel="$(basename "${dst}")"
  sha="$(sha256sum "${dst}" | awk '{print $1}')"
  bytes="$(stat -c %s "${dst}")"
  mtime="$(date -u -d "@$(stat -c %Y "${dst}")" +%Y-%m-%dT%H:%M:%SZ)"
  printf '%s\t%s\t%s\t%s\n' "${rel}" "${sha}" "${bytes}" "${mtime}" >> "${manifest}"
  total_bytes=$((total_bytes + bytes))
done < <(find "${DST_DIR}" -maxdepth 1 -type f \
  \( -name '*agent*.jsonl' -o -name '*.summary.json' \) -print0 | sort -z)

# Verify-with: (cd DST_DIR && sha256sum -c manifest.sha256 && awk-based row check).
(cd "${DST_DIR}" && sha256sum manifest.tsv > manifest.sha256)

echo "freeze: ${n_jsonl} *agent*.jsonl, ${n_summary} *.summary.json, ${total_bytes} bytes"
echo "freeze: manifest ${manifest} ($((n_jsonl + n_summary)) entries)"
echo "freeze: manifest sha256 $(cut -d' ' -f1 "${DST_DIR}/manifest.sha256")"
