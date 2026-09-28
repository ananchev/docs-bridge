#!/usr/bin/env python3
"""archive-keeplist.py - build a reviewable keep-list from a raw document archive.

Standalone (stdlib only, never imports docs_bridge), like doxy2md.py / javadoc2md.py.
Runs on the Nextcloud host where the archive lives.

The problem it solves: a raw work archive is mostly ballast (ISOs, videos, zips, CAD,
duplicates) and its spreadsheets are data dumps whose chunk yield dwarfs everything
else. docs-bridge's config include/exclude are fnmatch globs on the doc_id, so they
cannot express "xlsx under N bytes" - the cut has to happen at copy time. This tool
makes that cut auditable: it emits one row per file with a KEEP/DROP decision and the
reason, you review and hand-edit, and it then emits the rsync input.

    # 1. manifest (on the host holding the archive)
    cd /nvme/containers/nextcloud/data/ananchev/files
    find SKF HxGN -type f -printf '%s\t%TY-%Tm\t%p\n' > /tmp/corpus-manifest.tsv

    # 2. decisions
    python3 archive-keeplist.py /tmp/corpus-manifest.tsv -o /tmp/keeplist.tsv --stats

    # 3. review; flip column 1 by hand where you disagree

    # 4. copy what survived
    python3 archive-keeplist.py --emit-paths /tmp/keeplist.tsv > /tmp/keep-paths.txt
    rsync -a --files-from=/tmp/keep-paths.txt ./ vhost2:/data/docs/work/

Re-running regenerates decisions from scratch; --merge <old-keeplist> carries manual
overrides forward so a re-run never silently discards review work.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter

# --- policy ------------------------------------------------------------------
# Everything below is meant to be edited. It encodes one archive's shape, not a law.

# Formats docling can parse today, or after the LibreOffice pre-pass (legacy OLE).
DOC_EXTS = {"pdf", "docx", "pptx", "doc", "ppt", "rtf", "odt", "odp", "md", "txt"}
SHEET_EXTS = {"xlsx", "xls", "xlsm", "ods"}

# Spreadsheets above this are data dumps (migration exports, asset registries), not
# prose: in the SKF/HxGN archive the xlsx mean was 77x the median and the tail alone
# projected ~773k chunks. See notes/work-archive-ingest-plan.md section 3.
SHEET_MAX_BYTES = 2 * 1024 * 1024

# Path segments that mark superseded content. Matched case-insensitively as /seg/.
JUNK_SEGMENTS = (
    "obsolete", "old", "archive", "archiv", "backup",
    "superseded", "deprecated", "trash", "temp", "tmp",
)

# Substrings marking personal-admin or non-work noise. Archive-specific - edit freely.
NOISE_SUBSTRINGS = (
    "/Expenses/",
    "/GoodNotes/",
    "/SKF Visual Studio/",   # Windows + Visual Studio ISOs, 5.73 GB
    "/GPM2 Website/",        # scraped WordPress site, 2039 .htm
    "timesheet",
)

REASONS = (
    "ext",             # not a document/spreadsheet format
    "junk-path",       # lives under obsolete/old/archive/...
    "sheet-too-big",   # spreadsheet over SHEET_MAX_BYTES
    "dup",             # same basename + same size seen already
    "noise",           # matches NOISE_SUBSTRINGS
    "empty",           # zero bytes
)

HEADER = ("decision", "reason", "bytes", "month", "path")


def ext_of(path: str) -> str:
    return os.path.splitext(path)[1].lower().lstrip(".")


def read_manifest(fh):
    """Yield (size, month, path) from a `find -printf '%s\\t%TY-%Tm\\t%p\\n'` manifest."""
    for lineno, line in enumerate(fh, 1):
        parts = line.rstrip("\n").split("\t")
        if len(parts) != 3:
            print(f"!! skipping malformed line {lineno}", file=sys.stderr)
            continue
        size, month, path = parts
        try:
            yield int(size), month, path
        except ValueError:
            print(f"!! skipping non-numeric size on line {lineno}", file=sys.stderr)


def decide(rows):
    """Apply the funnel. Returns [(decision, reason, size, month, path)].

    Order matters only for which reason gets reported; a file dropped for several
    reasons reports the first that fired. Duplicate detection is last so the surviving
    copy is the first in sorted path order (stable across runs)."""
    seen: set[tuple[int, str]] = set()
    out = []
    for size, month, path in sorted(rows, key=lambda r: r[2]):
        e = ext_of(path)
        lower = path.lower()
        reason = None
        if size == 0:
            reason = "empty"
        elif e not in DOC_EXTS and e not in SHEET_EXTS:
            reason = "ext"
        elif any(f"/{seg}/" in lower for seg in JUNK_SEGMENTS):
            reason = "junk-path"
        elif e in SHEET_EXTS and size > SHEET_MAX_BYTES:
            reason = "sheet-too-big"
        elif any(s in path for s in NOISE_SUBSTRINGS):
            reason = "noise"
        else:
            key = (size, os.path.basename(path).lower())
            if key in seen:
                reason = "dup"
            else:
                seen.add(key)
        out.append(("DROP" if reason else "KEEP", reason or "-", size, month, path))
    return out


def load_overrides(path: str) -> dict[str, str]:
    """Read an earlier keep-list, returning {path: decision} for rows whose decision
    differs from what this tool would produce - i.e. the human edits worth preserving."""
    prior: dict[str, str] = {}
    with open(path, encoding="utf-8") as fh:
        rows = [ln.rstrip("\n").split("\t") for ln in fh]
    body = [r for r in rows if len(r) == 5 and r[0] in ("KEEP", "DROP")]
    regen = {r[4]: r[0] for r in decide([(int(r[2]), r[3], r[4]) for r in body])}
    for r in body:
        if regen.get(r[4]) != r[0]:
            prior[r[4]] = r[0]
    return prior


def write_keeplist(rows, out, overrides):
    n_over = 0
    out.write("\t".join(HEADER) + "\n")
    for decision, reason, size, month, path in rows:
        if path in overrides and overrides[path] != decision:
            decision, reason, n_over = overrides[path], "manual", n_over + 1
        out.write(f"{decision}\t{reason}\t{size}\t{month}\t{path}\n")
    return n_over


def print_stats(rows, out=sys.stderr):
    kept = [r for r in rows if r[0] == "KEEP"]
    gb = lambda rs: sum(r[2] for r in rs) / 1073741824
    print(f"\n{'=' * 62}\n keep-list summary\n{'=' * 62}", file=out)
    print(f"  input   {len(rows):7,d} files  {gb(rows):8.2f} GB", file=out)
    print(f"  KEEP    {len(kept):7,d} files  {gb(kept):8.2f} GB", file=out)
    print(f"  DROP    {len(rows) - len(kept):7,d} files  {gb(rows) - gb(kept):8.2f} GB", file=out)

    print("\n  dropped by reason:", file=out)
    by_reason = Counter(r[1] for r in rows if r[0] == "DROP")
    bytes_by_reason = Counter()
    for r in rows:
        if r[0] == "DROP":
            bytes_by_reason[r[1]] += r[2]
    for reason, count in by_reason.most_common():
        print(f"    {reason:16} {count:7,d} files  {bytes_by_reason[reason] / 1073741824:8.2f} GB", file=out)

    print("\n  kept by extension:", file=out)
    by_ext = Counter(ext_of(r[4]) for r in kept)
    bytes_by_ext = Counter()
    for r in kept:
        bytes_by_ext[ext_of(r[4])] += r[2]
    for e, count in by_ext.most_common(12):
        print(f"    {e:16} {count:7,d} files  {bytes_by_ext[e] / 1073741824:8.2f} GB", file=out)

    print("\n  kept by top-level library:", file=out)
    by_lib = Counter()
    bytes_by_lib = Counter()
    for r in kept:
        lib = "/".join(r[4].split("/")[:2])
        by_lib[lib] += 1
        bytes_by_lib[lib] += r[2]
    for lib, count in sorted(by_lib.items(), key=lambda kv: -bytes_by_lib[kv[0]])[:20]:
        print(f"    {count:6,d} files  {bytes_by_lib[lib] / 1073741824:7.2f} GB  {lib}", file=out)
    print(file=out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("manifest", help="find-manifest TSV, or the keep-list with --emit-paths")
    ap.add_argument("-o", "--out", help="write the keep-list here (default: stdout)")
    ap.add_argument("--stats", action="store_true", help="print the funnel summary to stderr")
    ap.add_argument("--merge", metavar="KEEPLIST",
                    help="carry manual KEEP/DROP overrides forward from an earlier keep-list")
    ap.add_argument("--emit-paths", action="store_true",
                    help="read a keep-list and print the KEEP paths (rsync --files-from input)")
    ap.add_argument("--explain", metavar="SUBSTRING",
                    help="print the decision for every path containing SUBSTRING, then exit")
    args = ap.parse_args()

    if args.emit_paths:
        with open(args.manifest, encoding="utf-8") as fh:
            for line in fh:
                parts = line.rstrip("\n").split("\t")
                if len(parts) == 5 and parts[0] == "KEEP":
                    print(parts[4])
        return 0

    with open(args.manifest, encoding="utf-8", errors="replace") as fh:
        rows = decide(list(read_manifest(fh)))

    if args.explain:
        needle = args.explain.lower()
        for decision, reason, size, month, path in rows:
            if needle in path.lower():
                print(f"{decision}\t{reason}\t{size:>12,}\t{month}\t{path}")
        return 0

    overrides = load_overrides(args.merge) if args.merge else {}
    if args.out:
        with open(args.out, "w", encoding="utf-8") as out:
            n = write_keeplist(rows, out, overrides)
    else:
        n = write_keeplist(rows, sys.stdout, overrides)
    if overrides:
        print(f">> carried {n} manual override(s) forward from {args.merge}", file=sys.stderr)
    if args.stats:
        print_stats(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
