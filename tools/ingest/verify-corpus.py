#!/usr/bin/env python3
"""verify-corpus.py — reconcile the ingest manifest against what Qdrant actually holds.

WHY THIS EXISTS. The worker records each doc in the manifest during Pass 1 (parse),
*before* its chunks are embedded in Pass 2. If Pass 2 dies and a later run clears the
staged chunks, the manifest still says "this doc is done, hash X" — so hash-delta
classifies it as *unchanged* forever and the doc is never re-embedded. The doc is
silently absent from the index and nothing in the normal `sync` output says so.

This script is the detector: for every doc in the manifest it compares the recorded
chunk_count against the real number of points carrying that doc_id, and names the docs
that drifted. Run it after any interrupted ingest, and on a schedule if you want the
condition to be loud rather than silent.

Stdlib only (sqlite3 + urllib) — no venv, runs on the runtime host as any user that can
read the manifest and reach the Qdrant REST port.

    ./verify-corpus.py --subject mysubject
    ./verify-corpus.py --subject mysubject --write-broken /tmp/broken.txt

Exit status: 0 = corpus matches the manifest, 1 = drift found, 2 = could not check.
Feed --write-broken straight into reingest-docs.py --docs-from to repair the drift.
"""

from __future__ import annotations

import argparse
import collections
import json
import sqlite3
import sys
import urllib.error
import urllib.request

DEFAULT_MANIFEST = "/data/docs-bridge-payload/state/manifest.sqlite"
DEFAULT_QDRANT = "http://127.0.0.1:6333"
SCROLL_PAGE = 10_000


def qdrant_post(base: str, path: str, body: dict) -> dict:
    req = urllib.request.Request(
        f"{base}{path}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.load(resp)["result"]


def point_counts(base: str, collection: str) -> collections.Counter:
    """Every doc_id in the collection -> how many points carry it.

    One full scroll (payload-only, no vectors) is far cheaper than a per-doc count
    request when there are thousands of docs: ~12 requests for a 110k-point collection
    instead of 14k.
    """
    counts: collections.Counter = collections.Counter()
    offset = None
    while True:
        body = {"limit": SCROLL_PAGE, "with_payload": ["doc_id"], "with_vector": False}
        if offset is not None:
            body["offset"] = offset
        page = qdrant_post(base, f"/collections/{collection}/points/scroll", body)
        for point in page["points"]:
            doc_id = (point.get("payload") or {}).get("doc_id")
            if doc_id is not None:
                counts[doc_id] += 1
        offset = page.get("next_page_offset")
        if offset is None:
            return counts


def manifest_rows(path: str, subject: str) -> list[tuple[str, int]]:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        return list(
            conn.execute(
                "SELECT doc_id, chunk_count FROM docs WHERE subject = ? ORDER BY doc_id",
                (subject,),
            )
        )
    finally:
        conn.close()


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Reconcile the ingest manifest against the Qdrant collection."
    )
    ap.add_argument("--subject", default="mysubject", help="manifest subject to check")
    ap.add_argument(
        "--collection",
        help="Qdrant collection (default: same name as --subject, which is the "
        "usual 1:1 mapping in config.yaml)",
    )
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    ap.add_argument("--qdrant", default=DEFAULT_QDRANT)
    ap.add_argument(
        "--write-broken",
        metavar="FILE",
        help="write the doc_ids that need re-ingesting (missing + partial), one per "
        "line, for reingest-docs.py --docs-from",
    )
    ap.add_argument(
        "--quiet",
        action="store_true",
        help="only print the summary and any drift, not the per-doc detail",
    )
    args = ap.parse_args()

    collection = args.collection or args.subject

    try:
        rows = manifest_rows(args.manifest, args.subject)
    except sqlite3.Error as exc:
        print(f"!! cannot read manifest {args.manifest}: {exc}", file=sys.stderr)
        return 2
    if not rows:
        print(f"!! no manifest rows for subject {args.subject!r} in {args.manifest}",
              file=sys.stderr)
        return 2

    try:
        counts = point_counts(args.qdrant, collection)
    except (urllib.error.URLError, urllib.error.HTTPError, KeyError) as exc:
        print(f"!! cannot read Qdrant collection {collection!r}: {exc}", file=sys.stderr)
        return 2

    # A doc with chunk_count 0 legitimately has no points (e.g. a Doxygen namespace
    # stub that converts to a bare heading). That is not drift — count it apart so it
    # never masquerades as a missing doc.
    empty = [(d, n) for d, n in rows if n == 0]
    missing = [(d, n) for d, n in rows if n > 0 and counts.get(d, 0) == 0]
    partial = [(d, n, counts[d]) for d, n in rows if n > 0 and 0 < counts.get(d, 0) < n]
    extra = [(d, n, counts[d]) for d, n in rows if counts.get(d, 0) > n]
    known = {d for d, _ in rows}
    orphan = [(d, c) for d, c in counts.items() if d not in known]

    total_manifest = sum(n for _, n in rows)
    total_points = sum(counts.values())

    print(f"subject    : {args.subject}  ->  collection {collection}")
    print(f"manifest   : {len(rows)} docs, {total_manifest} chunks recorded")
    print(f"qdrant     : {len(counts)} docs, {total_points} points present")
    print(f"shortfall  : {total_manifest - total_points} chunks")
    print()
    print(f"  in sync            : {len(rows) - len(empty) - len(missing) - len(partial) - len(extra)}")
    print(f"  empty (0 chunks)   : {len(empty)}   (expected to have no points)")
    print(f"  MISSING (0 points) : {len(missing)}   ({sum(n for _, n in missing)} chunks)")
    print(f"  PARTIAL            : {len(partial)}")
    print(f"  more than manifest : {len(extra)}")
    print(f"  ORPHAN (not in manifest): {len(orphan)}   ({sum(c for _, c in orphan)} points)")

    if missing and not args.quiet:
        print()
        print("MISSING — manifest says ingested, Qdrant has nothing. A plain `sync`")
        print("will SKIP these (the hash still matches); clear their manifest rows")
        print("with reingest-docs.py to force a re-parse:")
        for doc_id, n in sorted(missing, key=lambda r: -r[1]):
            print(f"   {n:6d}  {doc_id}")
    if partial and not args.quiet:
        print()
        print("PARTIAL — some chunks landed, some did not:")
        for doc_id, n, got in sorted(partial, key=lambda r: -(r[1] - r[2])):
            print(f"   manifest={n:6d} qdrant={got:6d}  {doc_id}")
    if extra and not args.quiet:
        print()
        print("MORE THAN MANIFEST — stale points from an earlier chunking of the same")
        print("doc. reingest-docs.py --purge-points clears them before the re-ingest:")
        for doc_id, n, got in extra:
            print(f"   manifest={n:6d} qdrant={got:6d}  {doc_id}")
    if orphan and not args.quiet:
        print()
        print("ORPHAN — points with no manifest row (deleted doc whose points were")
        print("never removed, or a manifest that was reset without dropping the")
        print("collection):")
        for doc_id, got in sorted(orphan, key=lambda r: -r[1]):
            print(f"   {got:6d}  {doc_id}")

    if args.write_broken:
        broken = [d for d, _ in missing] + [d for d, _, _ in partial] + [d for d, _, _ in extra]
        with open(args.write_broken, "w", encoding="utf-8") as fh:
            for doc_id in broken:
                fh.write(doc_id + "\n")
        print()
        print(f">> wrote {len(broken)} doc_id(s) to {args.write_broken}")

    drift = bool(missing or partial or extra or orphan)
    print()
    print("RESULT: drift found — the index does not match the manifest." if drift
          else "RESULT: clean — every manifest doc is fully present in Qdrant.")
    return 1 if drift else 0


if __name__ == "__main__":
    sys.exit(main())
