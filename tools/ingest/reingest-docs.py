#!/usr/bin/env python3
"""reingest-docs.py — force named docs to be re-ingested on the next `sync`.

Hash-delta means an unchanged file is never re-parsed: the manifest holds the doc's
content_hash, the file still hashes the same, so `sync` classifies it *unchanged* and
moves on. That is exactly right in normal operation — and exactly wrong when the
manifest row is a lie, i.e. the doc was recorded in Pass 1 but its chunks never reached
Qdrant (see verify-corpus.py for how that happens).

This script removes the manifest rows for the named docs, so the next `sync` sees them
as NEW and parses + embeds them from scratch. It touches nothing else: the source files
are not modified, other docs are not reclassified.

Stdlib only. Needs WRITE access to the manifest (it is root-owned under the Ansible
layout, so run under sudo). Dry-run by default, like ingest-prune.sh — pass --apply.

    sudo ./reingest-docs.py --subject mysubject --docs-from /tmp/broken.txt
    sudo ./reingest-docs.py --subject mysubject --docs-from /tmp/broken.txt --apply

Then re-run the ingest (./run-ingest.sh <subject>) and confirm with verify-corpus.py.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import sys
import time
import urllib.error
import urllib.request

DEFAULT_MANIFEST = "/data/docs-bridge-payload/state/manifest.sqlite"
DEFAULT_QDRANT = "http://127.0.0.1:6333"


def qdrant_post(base: str, path: str, body: dict) -> dict:
    req = urllib.request.Request(
        f"{base}{path}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.load(resp)["result"]


def load_doc_ids(args: argparse.Namespace) -> list[str]:
    doc_ids: list[str] = list(args.doc or [])
    if args.docs_from:
        with open(args.docs_from, encoding="utf-8") as fh:
            doc_ids += [line.rstrip("\n") for line in fh if line.strip()]
    # Preserve order, drop repeats.
    seen: set[str] = set()
    return [d for d in doc_ids if not (d in seen or seen.add(d))]


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Drop manifest rows so the named docs are re-ingested next sync."
    )
    ap.add_argument("--subject", required=True, help="manifest subject the docs belong to")
    ap.add_argument("--doc", action="append", metavar="DOC_ID",
                    help="a doc_id to reset (repeatable)")
    ap.add_argument("--docs-from", metavar="FILE",
                    help="file of doc_ids, one per line (verify-corpus.py --write-broken)")
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    ap.add_argument("--apply", action="store_true",
                    help="actually delete (default is a dry run)")
    ap.add_argument(
        "--purge-points",
        action="store_true",
        help="also delete the docs' existing points from Qdrant first. Point ids are a "
        "deterministic UUIDv5 of chunk_id, so a re-ingest overwrites the points it "
        "regenerates — but if the doc now chunks into FEWER pieces than before, the "
        "surplus old points would linger. Use this when a doc had partial or stale "
        "points; unnecessary when it had none.",
    )
    ap.add_argument("--collection", help="Qdrant collection for --purge-points "
                                        "(default: same name as --subject)")
    ap.add_argument("--qdrant", default=DEFAULT_QDRANT)
    ap.add_argument("--no-backup", action="store_true",
                    help="skip the manifest backup copy (not recommended)")
    args = ap.parse_args()

    doc_ids = load_doc_ids(args)
    if not doc_ids:
        print("!! no doc_ids given (use --doc and/or --docs-from)", file=sys.stderr)
        return 2

    if not os.path.exists(args.manifest):
        print(f"!! manifest not found: {args.manifest}", file=sys.stderr)
        return 2

    mode = "APPLY" if args.apply else "DRY RUN"
    print(f">> {mode}: reset {len(doc_ids)} doc(s) in subject {args.subject!r}")
    print(f"   manifest: {args.manifest}")

    conn = sqlite3.connect(args.manifest)
    conn.row_factory = sqlite3.Row
    try:
        found, absent, chunks = [], [], 0
        for doc_id in doc_ids:
            row = conn.execute(
                "SELECT chunk_count FROM docs WHERE doc_id = ? AND subject = ?",
                (doc_id, args.subject),
            ).fetchone()
            if row is None:
                absent.append(doc_id)
            else:
                found.append(doc_id)
                chunks += row["chunk_count"]

        # staged_chunks rows for these docs would be drained by the next sync before it
        # even classifies; clear them so the re-parse is the only source of their chunks.
        staged = 0
        for doc_id in found:
            staged += conn.execute(
                "SELECT count(*) FROM staged_chunks WHERE doc_id = ? AND subject = ?",
                (doc_id, args.subject),
            ).fetchone()[0]

        print(f"   in manifest    : {len(found)} doc(s), {chunks} recorded chunks")
        print(f"   not in manifest: {len(absent)} (already look NEW to sync — nothing to do)")
        print(f"   staged rows to clear: {staged}")
        for doc_id in absent:
            print(f"      absent: {doc_id}")

        if not found:
            print(">> nothing to reset.")
            return 0

        if not args.apply:
            print()
            print("   would delete these manifest rows:")
            for doc_id in found:
                print(f"      {doc_id}")
            print()
            print(">> DRY RUN — nothing changed. Re-run with --apply.")
            return 0

        if args.purge_points:
            collection = args.collection or args.subject
            print(f"   purging points from collection {collection!r} ...")
            for doc_id in found:
                qdrant_post(
                    args.qdrant,
                    f"/collections/{collection}/points/delete?wait=true",
                    {"filter": {"must": [{"key": "doc_id", "match": {"value": doc_id}}]}},
                )
            print(f"   purged points for {len(found)} doc(s)")

        if not args.no_backup:
            stamp = time.strftime("%Y%m%dT%H%M%S")
            backup = f"{args.manifest}.bak-{stamp}"
            shutil.copy2(args.manifest, backup)
            print(f"   backup: {backup}")

        with conn:
            for doc_id in found:
                conn.execute(
                    "DELETE FROM staged_chunks WHERE doc_id = ? AND subject = ?",
                    (doc_id, args.subject),
                )
                conn.execute(
                    "DELETE FROM docs WHERE doc_id = ? AND subject = ?",
                    (doc_id, args.subject),
                )
        print(f">> deleted {len(found)} manifest row(s). The next sync will treat them as NEW.")
        print("   Run the ingest, then verify with verify-corpus.py.")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
