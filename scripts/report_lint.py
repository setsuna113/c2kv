#!/usr/bin/env python
"""Lint weekly-report prose for agent-draft residue that the author deletes by hand.

Rules are derived from the revision trails of weeks 6, 8 and 10 (see the research-kit
skill weekly-report/references/deletion-evidence.md). BLOCK rules match material the
author removed every time it appeared; WARN rules match material removed most times.

Output is coordinates only (path:line level rule-id). Matched text is never echoed, so
banned prose cannot ride back into an agent's context through this tool.

Usage
  python scripts/report_lint.py PATH [PATH ...]   lint files or directories
  python scripts/report_lint.py --staged          lint staged reports/weekly *.md / *.tex
  python scripts/report_lint.py --all             lint every tracked reports/weekly *.md / *.tex
  python scripts/report_lint.py --hook            Claude Code PostToolUse hook (JSON on stdin)

Options
  --no-warn   hide WARN findings
  --strict    treat WARN as BLOCK

Exit codes
  CLI   0 clean or warn-only, 1 any BLOCK (or WARN with --strict)
  --hook  0 clean or warn-only (warnings returned as additionalContext), 2 any BLOCK

Suppress one line by putting `lint: allow` in a comment on that line:
  <!-- lint: allow -->   in Markdown        % lint: allow   in LaTeX
Stdlib only; runs on any Python 3.8+.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

REPORT_DIR = "reports/weekly/"
EXTS = {".md", ".tex"}
SKIP_PARTS = ("/tmp/", "/_archive", "/rendered_", "/build/", "/figures/", "/data/", "/__pycache__/")
ALLOW_MARK = re.compile(r"lint:\s*allow", re.IGNORECASE)

BLOCK = "BLOCK"
WARN = "WARN"


@dataclass(frozen=True)
class Rule:
    id: str
    level: str
    pattern: "re.Pattern[str]"
    kinds: Tuple[str, ...]  # "md", "tex", or both
    scope: str  # "line" or "heading"
    note: str


def _r(pat: str, flags: int = 0) -> "re.Pattern[str]":
    return re.compile(pat, flags)


BLOCK_HEADINGS = (
    r"executive (?:answer|summary)|decision first|results at a glance|how to read|key takeaways|tl;?dr|"
    r"interpretation and limits|limitations?|interpretive boundary|verification and provenance|provenance|"
    r"artifact index|claim boundary|claim ceiling|what this run cannot verify"
)

RULES: List[Rule] = [
    # ---------------- BLOCK ----------------
    Rule("status-token", BLOCK,
         _r(r"\b(?:EXPLORATORY(?:\\?_[A-Z0-9]+)+|[A-Z][A-Z0-9]*(?:\\?_[A-Z0-9]+){2,})\b"),
         ("md", "tex"), "line", "SCREAMING_SNAKE claim label / status token in prose"),
    Rule("heading-block", BLOCK, _r(r"^(?:" + BLOCK_HEADINGS + r")\b", re.IGNORECASE),
         ("md", "tex"), "heading", "executive / decision / limits / provenance heading"),
    Rule("verdict-paragraph", BLOCK,
         _r(r"^\s*(?:\*\*|\\textbf\{)?\s*(?:decision|verdict|promotion decision|disposition)\s*(?:\*\*|\})?\s*:",
            re.IGNORECASE),
         ("md", "tex"), "line", "'Decision:' verdict paragraph"),
    Rule("no-promote-verdict", BLOCK,
         _r(r"\b(?:do not promote|no promotion is (?:authori[sz]ed|granted)|is not promoted|not promoted this round)\b",
            re.IGNORECASE),
         ("md", "tex"), "line", "promotion verdict sentence"),
    Rule("hash-in-prose", BLOCK, _r(r"\b[0-9a-f]{40}\b|\b[0-9a-f]{64}\b"),
         ("md", "tex"), "line", "40/64-hex hash in prose"),
    Rule("audit-vocab-hard", BLOCK,
         _r(r"\b(?:ledgers?|dispositions?|closeouts?|tombstones?|attestations?|pre-?registered|pre-?registration|"
            r"claim[ _-]label|status token)\b", re.IGNORECASE),
         ("md", "tex"), "line", "audit vocabulary"),
    Rule("em-dash", BLOCK, _r("\u2014"), ("md", "tex"), "line", "em dash"),
    Rule("tex-em-dash", BLOCK, _r(r"(?<!-)---(?!-)"), ("tex",), "line", "LaTeX --- em dash"),
    Rule("abstract-toc", BLOCK, _r(r"\\tableofcontents\b|\\begin\{abstract\}"),
         ("tex",), "line", "abstract or table of contents"),
    # ---------------- WARN ----------------
    Rule("fence-sentence", WARN,
         _r(r"\b(?:does|do|did) not (?:establish|support|licen[cs]e|imply|constitute|demonstrate|prove)\b|"
            r"\bis not (?:evidence|proof) (?:of|that|for)\b|"
            r"\bnot an? (?:biomarker|mechanism|clinical)\b|"
            r"\b(?:should not|must not|cannot|can not) be (?:read|interpreted|treated|taken|rewritten|subtracted|compared)\b",
            re.IGNORECASE),
         ("md", "tex"), "line", "claim fence after a result"),
    Rule("x-not-y", WARN,
         _r(r"\bwarning, not\b|, not (?:evidence|proof|a null|no effect|[\"\u201c]no effect)|\bnot killed\b|\bnot falsified\b",
            re.IGNORECASE),
         ("md", "tex"), "line", "'X, not Y' refusal of a misreading"),
    Rule("graded-negative", WARN,
         _r(r"\b(?:not (?:reportably )?estimable|unsupported|inconclusive|unresolved)\b", re.IGNORECASE),
         ("md", "tex"), "line", "graded-negative vocabulary"),
    Rule("audit-vocab-soft", WARN,
         _r(r"\b(?:receipts?|manifests?|promotions?|promoted?|witness(?:es)?|estimands?|gates?|contracts?|"
            r"pre-?specified|pre-?declared)\b", re.IGNORECASE),
         ("md", "tex"), "line", "governance vocabulary"),
    Rule("internal-id", WARN,
         _r(r"\b[a-z0-9]+_v\d+_\d{8}(?:T\d{6}Z)?(?:_[0-9a-f]{6,})?\b|\bPR ?#\d+\b|\bair\d{3}\b|(?:@|\bat )[0-9a-f]{7,12}\b"),
         ("md", "tex"), "line", "attempt id / PR number / hostname / short SHA"),
    Rule("file-mode", WARN, _r(r"\b0[67]00\b"), ("md", "tex"), "line", "filesystem permission mode"),
    Rule("code-in-heading", WARN, _r(r"\b[A-Z]\d{1,2}[A-Za-z]?\b|\\?_v\d+\b"),
         ("md", "tex"), "heading", "machine code in a heading"),
    Rule("subtitle", WARN, _r(r"^#\s*Week\s*\d+\s*(?:Report)?\s*[-:\u2013\u2014]\s*\S", re.IGNORECASE),
         ("md",), "line", "subtitle after the report title"),
    Rule("tex-two-line-title", WARN, _r(r"\\title\{[^}]*\\\\"), ("tex",), "line", "two-line title"),
    Rule("author-person", WARN, _r(r"\\author\{(?!\s*\}|\s*Beyond Accuracy)"), ("tex",), "line",
         "author is not the project name or empty"),
    Rule("paradox", WARN, _r(r"\bparadox\b", re.IGNORECASE), ("md", "tex"), "line",
         "mechanistic 'resolves the paradox' explanation"),
    Rule("en-dash", WARN, _r("\u2013"), ("md",), "line", "en dash (author uses hyphen)"),
    # week 11 (2026-09-02): intervals belong on forest plots only; prose must not re-read a figure or table
    Rule("ci-in-table", WARN, _r(r"^\s*\|.*\[[+-]?\d\.\d{2,},\s*[+-]?\d\.\d{2,}\]"),
         ("md",), "line", "confidence interval inside a table cell"),
    Rule("figure-readout", WARN,
         _r(r"^\s*(?:The|This) (?:figure|table|panel|map|diagram) (?:reads|shows|is read|groups)\b", re.IGNORECASE),
         ("md", "tex"), "line", "paragraph that re-reads a figure or table"),
    Rule("heading-next-week", WARN, _r(r"^next (?:week|steps)\b", re.IGNORECASE),
         ("md", "tex"), "heading", "Next week section (author deleted it in week 11)"),
]

MD_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+(.*?)\s*#*\s*$")
MD_BOLD_LINE = re.compile(r"^\s*\*\*([^*]+)\*\*\s*$")
TEX_HEADING = re.compile(r"\\(?:part|chapter|section|subsection|subsubsection|paragraph)\*?\{([^}]*)\}")
FENCE = re.compile(r"^\s*(?:```|~~~)")


@dataclass(frozen=True)
class Finding:
    path: str
    line: int
    level: str
    rule: str


def kind_of(path: Path) -> Optional[str]:
    suf = path.suffix.lower()
    if suf == ".md":
        return "md"
    if suf == ".tex":
        return "tex"
    return None


def heading_text(line: str, kind: str) -> Optional[str]:
    if kind == "md":
        m = MD_HEADING.match(line)
        if m:
            return m.group(1)
        m = MD_BOLD_LINE.match(line)
        if m:
            return m.group(1)
        return None
    m = TEX_HEADING.search(line)
    return m.group(1) if m else None


def lint_text(text: str, kind: str, rel: str, strict: bool) -> List[Finding]:
    out: List[Finding] = []
    in_fence = False
    for no, raw in enumerate(text.splitlines(), start=1):
        line = raw.rstrip("\r")
        if kind == "md" and FENCE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        stripped = line.lstrip()
        if kind == "tex" and stripped.startswith("%"):
            continue
        if kind == "md" and stripped.startswith("<!--") and stripped.endswith("-->"):
            continue
        if ALLOW_MARK.search(line):
            continue
        head = heading_text(line, kind)
        for rule in RULES:
            if kind not in rule.kinds:
                continue
            target = head if rule.scope == "heading" else line
            if target is None:
                continue
            if rule.pattern.search(target):
                level = BLOCK if (strict and rule.level == WARN) else rule.level
                out.append(Finding(rel, no, level, rule.id))
    return out


def lint_file(path: Path, root: Path, strict: bool) -> List[Finding]:
    kind = kind_of(path)
    if kind is None:
        return []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    try:
        rel = path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        rel = path.as_posix()
    return lint_text(text, kind, rel, strict)


def is_report_path(p: Path) -> bool:
    s = p.as_posix()
    if REPORT_DIR not in s and not s.startswith(REPORT_DIR):
        return False
    if any(part in s for part in SKIP_PARTS):
        return False
    return p.suffix.lower() in EXTS


def expand(paths: Iterable[str]) -> List[Path]:
    files: List[Path] = []
    for p in paths:
        pp = Path(p)
        if pp.is_dir():
            for f in sorted(pp.rglob("*")):
                if f.is_file() and f.suffix.lower() in EXTS and not any(x in f.as_posix() for x in SKIP_PARTS):
                    files.append(f)
        elif pp.is_file():
            files.append(pp)
    return files


def git_files(args: List[str]) -> List[Path]:
    try:
        out = subprocess.run(["git"] + args, check=True, capture_output=True, text=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return []
    return [Path(l.strip()) for l in out.splitlines() if l.strip() and is_report_path(Path(l.strip()))]


def format_findings(findings: List[Finding]) -> str:
    lines = [f"{f.path}:{f.line}  {f.level}  {f.rule}" for f in findings]
    nb = sum(1 for f in findings if f.level == BLOCK)
    nw = len(findings) - nb
    files = len({f.path for f in findings})
    lines.append(f"report_lint: {nb} block, {nw} warn in {files} file(s)")
    return "\n".join(lines)


def run_cli(files: List[Path], root: Path, strict: bool, show_warn: bool) -> int:
    findings: List[Finding] = []
    for f in files:
        findings.extend(lint_file(f, root, strict))
    if not show_warn:
        findings = [x for x in findings if x.level == BLOCK]
    if not findings:
        print(f"report_lint: clean ({len(files)} file(s))")
        return 0
    print(format_findings(findings))
    return 1 if any(x.level == BLOCK for x in findings) else 0


def run_hook(root: Path, strict: bool) -> int:
    try:
        payload = json.load(sys.stdin)
    except Exception:  # malformed or empty stdin: never break the tool call
        return 0
    ti = payload.get("tool_input") or {}
    tr = payload.get("tool_response") or {}
    fp = ti.get("file_path") or tr.get("filePath") or ""
    if not fp:
        return 0
    p = Path(str(fp).replace("\\", "/"))
    if not is_report_path(p) or not p.exists():
        return 0
    findings = lint_file(p, root, strict)
    blocks = [x for x in findings if x.level == BLOCK]
    if blocks:
        sys.stderr.write(
            "report_lint BLOCK: the weekly-report style forbids the following (see research-kit skill weekly-report). "
            "Remove them before continuing.\n" + format_findings(findings) + "\n")
        return 2
    if findings:
        msg = "report_lint WARN (decide each; add 'lint: allow' on the line to keep):\n" + format_findings(findings)
        print(json.dumps({"hookSpecificOutput": {"hookEventName": "PostToolUse", "additionalContext": msg}}))
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("paths", nargs="*", help="files or directories to lint")
    ap.add_argument("--staged", action="store_true", help="lint staged reports/weekly files")
    ap.add_argument("--all", action="store_true", help="lint every tracked reports/weekly file")
    ap.add_argument("--hook", action="store_true", help="Claude Code PostToolUse hook mode (stdin JSON)")
    ap.add_argument("--no-warn", action="store_true", help="hide WARN findings")
    ap.add_argument("--strict", action="store_true", help="treat WARN as BLOCK")
    a = ap.parse_args(argv)

    root = Path(os.environ.get("CLAUDE_PROJECT_DIR") or os.getcwd())
    if a.hook:
        return run_hook(root, a.strict)
    if a.staged:
        files = git_files(["diff", "--cached", "--name-only", "--diff-filter=ACMR"])
    elif a.all:
        files = git_files(["ls-files", "reports/weekly"])
    else:
        files = expand(a.paths)
    if not files:
        print("report_lint: nothing to lint")
        return 0
    return run_cli(files, root, a.strict, not a.no_warn)


if __name__ == "__main__":
    sys.exit(main())
