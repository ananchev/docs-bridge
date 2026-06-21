"""The sync orchestrator: hash-delta detection + the two-pass run for one subject.

Flow (design §8):
  scan disk -> diff against manifest -> classify each doc as
    new (hash unseen) | changed (hash differs) | deleted (gone from disk) | unchanged
  PASS 1 (parse):  parse new+changed docs with Docling, stage chunks to SQLite,
                   then release Docling.
  PASS 2 (embed):  load BGE-M3, drain staged chunks in batches, upsert to Qdrant;
                   delete Qdrant points for changed (before re-insert) and deleted
                   docs; update the manifest.

Keeping Docling and BGE-M3 from being co-resident (the 8GB Pi budget) is the whole
point of staging to disk between the passes.
"""

from __future__ import annotations

import gc
import logging
from pathlib import Path

from . import qdrant_io
from .config import Config
from .embed import get_embedder
from .manifest import Manifest
from .models import DocState, Subject, SyncStats
from .parse import Parser, file_hash, scan

log = logging.getLogger(__name__)


def _classify(
    on_disk: dict[str, Path],
    hashes: dict[str, str],
    known: dict[str, DocState],
) -> tuple[list[str], list[str], list[str]]:
    new, changed = [], []
    for doc_id in on_disk:
        if doc_id not in known:
            new.append(doc_id)
        elif hashes[doc_id] != known[doc_id].content_hash:
            changed.append(doc_id)
    deleted = [doc_id for doc_id in known if doc_id not in on_disk]
    return new, changed, deleted


def sync_subject(
    cfg: Config, subject: Subject, manifest: Manifest, client
) -> SyncStats:
    stats = SyncStats(subject=subject.name)

    on_disk = scan(subject, cfg)
    known = manifest.docs_for_subject(subject.name)
    hashes = {doc_id: file_hash(p) for doc_id, p in on_disk.items()}

    new, changed, deleted = _classify(on_disk, hashes, known)
    stats.new = len(new)
    stats.changed = len(changed)
    stats.deleted = len(deleted)
    stats.unchanged = len(on_disk) - len(new) - len(changed)

    log.info("%s", stats)
    if stats.is_noop:
        return stats  # nothing to parse, embed, or delete

    qdrant_io.ensure_collection(client, cfg, subject.collection)

    # --- PASS 1: parse new + changed -> stage to SQLite ----------------------
    manifest.clear_staged(subject.name)
    to_parse = new + changed
    if to_parse:
        parser = Parser(cfg)
        for doc_id in to_parse:
            path = on_disk[doc_id]
            log.debug("parsing %s", doc_id)
            chunks = parser.parse(subject, doc_id, path, hashes[doc_id])
            manifest.stage_chunks(chunks)
            # Provisionally record the doc; chunk_count is finalised after embed.
            manifest.upsert_doc(
                DocState(
                    doc_id=doc_id,
                    subject=subject.name,
                    source_path=str(path),
                    content_hash=hashes[doc_id],
                    last_updated=chunks[0].last_updated if chunks else "",
                    chunk_count=len(chunks),
                )
            )
        del parser
        gc.collect()  # ensure Docling is gone before BGE-M3 loads (two-pass)

    # --- PASS 2: delete stale points, then embed staged ----------------------
    # Changed docs: drop old points before re-inserting. Deleted docs: drop + forget.
    for doc_id in changed:
        qdrant_io.delete_doc(client, subject.collection, doc_id)
    for doc_id in deleted:
        qdrant_io.delete_doc(client, subject.collection, doc_id)
        manifest.delete_doc(doc_id)

    staged = manifest.count_staged(subject.name)
    if staged:
        embedder = get_embedder(cfg)
        for batch in manifest.iter_staged_batches(subject.name, cfg.ingest.batch_size):
            dense, sparse = embedder.encode([c.text for c in batch])
            qdrant_io.upsert(client, subject.collection, batch, dense, sparse)
            stats.chunks_embedded += len(batch)
            log.debug("embedded %d / %d", stats.chunks_embedded, staged)
        del embedder
        gc.collect()

    manifest.clear_staged(subject.name)
    return stats


def sync(cfg: Config, subject_names: list[str]) -> list[SyncStats]:
    client = qdrant_io.connect(cfg)
    results: list[SyncStats] = []
    with Manifest(cfg.manifest_path) as manifest:
        for name in subject_names:
            results.append(sync_subject(cfg, cfg.subject(name), manifest, client))
    return results
