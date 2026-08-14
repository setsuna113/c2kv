#!/usr/bin/env bash
# Round-2 archive freeze: snapshot the round-1 agent-eval outputs read-only.
#
# Scope (matches round-1 PR#1 侦察b description): the top-level entries of
# SRC_DIR named agent_* or ablation_* — the agent_history / agent_tooldef /
# ablation_0724~0810 evaluation series — copied RECURSIVELY (per-mode
# .parts/*.summary.json splits and run logs included; they are the input of the
# mode×qid intersection gate and the T1/T2 pairing). Source is never modified.
#
# Two usage modes:
#   bash agent/forensics_freeze_archive.sh SRC_DIR DST_DIR
#       Full freeze: copy SRC_DIR/{agent_*,ablation_*} into DST_DIR (cp -a),
#       then write the manifest. Requires read access to SRC_DIR.
#   bash agent/forensics_freeze_archive.sh --manifest-only DST_DIR
#       Manifest phase only, for a DST_DIR already populated by other means
#       (on the NPU server the copy is executed once as root via
#       `tar c agent_* ablation_* | tar x -C DST_DIR` because the source
#       outputs/ is owned by another user; manifest then runs as the
#       unprivileged analyst account).
#
# Manifest: manifest.tsv (relpath<TAB>sha256<TAB>bytes<TAB>mtime_iso) sorted by
# relpath, plus manifest.sha256 (sha256sum of manifest.tsv). Both manifest
# files are excluded from the hashed set. Full-freeze mode refuses to write
# into an existing non-empty DST_DIR unless --force is given.
set -euo pipefail

usage() {
  echo "usage: $0 SRC_DIR DST_DIR [--force] | $0 --manifest-only DST_DIR" >&2
  exit 2
}

write_manifest() {
  local dst_dir="$1"
  local manifest="${dst_dir}/manifest.tsv"
  : > "${manifest}"
  local total_bytes=0 n_files=0
  while IFS= read -r -d '' f; do
    rel="${f#"${dst_dir}/"}"
    sha="$(sha256sum "${f}" | awk '{print $1}')"
    bytes="$(stat -c %s "${f}")"
    mtime="$(date -u -d "@$(stat -c %Y "${f}")" +%Y-%m-%dT%H:%M:%SZ)"
    printf '%s\t%s\t%s\t%s\n' "${rel}" "${sha}" "${bytes}" "${mtime}" >> "${manifest}"
    total_bytes=$((total_bytes + bytes))
    n_files=$((n_files + 1))
  done < <(find "${dst_dir}" -type f ! -name 'manifest.tsv' ! -name 'manifest.sha256' -print0 | sort -z)
  (cd "${dst_dir}" && sha256sum manifest.tsv > manifest.sha256)
  echo "freeze: manifest ${manifest} (${n_files} files, ${total_bytes} bytes)"
  echo "freeze: manifest sha256 $(cut -d' ' -f1 "${dst_dir}/manifest.sha256")"
}

[[ $# -ge 2 ]] || usage

if [[ "$1" == "--manifest-only" ]]; then
  DST_DIR="$2"
  [[ -d "${DST_DIR}" ]] || { echo "error: DST_DIR not found: ${DST_DIR}" >&2; exit 1; }
  write_manifest "${DST_DIR}"
  exit 0
fi

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

# Copy phase: read-only w.r.t. the source (cp -a preserves mtime/perms).
( cd "${SRC_DIR}" && cp -a agent_* ablation_* "${DST_DIR}/" )

write_manifest "${DST_DIR}"
