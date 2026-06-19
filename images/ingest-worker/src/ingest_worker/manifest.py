"""SQLite manifest + staging.

Two roles (design §8):
  1. `docs`  - the persistent manifest: one row per document, carrying the
     content_hash that drives hash-delta change detection across runs.
  2. `staged_chunks` - scratch space written by the parse pass and drained by the
     embed pass. Staging to disk is what keeps Docling and BGE-M3 from being
     co-resident: the parser is fully released before the embedder loads.

Shrink-safety (a doc losing chunks on edit) is handled at upsert time by deleting
all of a doc's Qdrant points by `doc_id` filter before re-inserting, so we do not
need to track the per-chunk id set here.
"""

from __future__ import annotations

import os
import sqlite3
from collections.abc import Iterator

from .models import Chunk, DocState

_SCHEMA = """
CREATE TABLE IF NOT EXISTS docs (
    doc_id       TEXT PRIMARY KEY,
    subject      TEXT NOT NULL,
    source_path  TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    last_updated TEXT NOT NULL,
    chunk_count  INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_docs_subject ON docs(subject);

CREATE TABLE IF NOT EXISTS staged_chunks (
    chunk_id     TEXT PRIMARY KEY,
    doc_id       TEXT NOT NULL,
    subject      TEXT NOT NULL,
    source_path  TEXT NOT NULL,
    chunk_index  INTEGER NOT NULL,
    section_path TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    last_updated TEXT NOT NULL,
    text         TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_staged_subject ON staged_chunks(subject);
"""


class Manifest:
    def __init__(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(_SCHEMA)
        self.db.commit()

    def close(self) -> None:
        self.db.close()

    def __enter__(self) -> "Manifest":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    # --- manifest (persistent) -------------------------------------------------

    def docs_for_subject(self, subject: str) -> dict[str, DocState]:
        rows = self.db.execute(
            "SELECT doc_id, subject, source_path, content_hash, last_updated, "
            "chunk_count FROM docs WHERE subject = ?",
            (subject,),
        ).fetchall()
        return {
            r["doc_id"]: DocState(
                doc_id=r["doc_id"],
                subject=r["subject"],
                source_path=r["source_path"],
                content_hash=r["content_hash"],
                last_updated=r["last_updated"],
                chunk_count=r["chunk_count"],
            )
            for r in rows
        }

    def upsert_doc(self, doc: DocState) -> None:
        self.db.execute(
            "INSERT INTO docs (doc_id, subject, source_path, content_hash, "
            "last_updated, chunk_count) VALUES (?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(doc_id) DO UPDATE SET "
            "source_path=excluded.source_path, content_hash=excluded.content_hash, "
            "last_updated=excluded.last_updated, chunk_count=excluded.chunk_count",
            (
                doc.doc_id,
                doc.subject,
                doc.source_path,
                doc.content_hash,
                doc.last_updated,
                doc.chunk_count,
            ),
        )
        self.db.commit()

    def delete_doc(self, doc_id: str) -> None:
        self.db.execute("DELETE FROM docs WHERE doc_id = ?", (doc_id,))
        self.db.commit()

    # --- staging (scratch, per run) -------------------------------------------

    def clear_staged(self, subject: str) -> None:
        self.db.execute("DELETE FROM staged_chunks WHERE subject = ?", (subject,))
        self.db.commit()

    def stage_chunks(self, chunks: list[Chunk]) -> None:
        self.db.executemany(
            "INSERT OR REPLACE INTO staged_chunks (chunk_id, doc_id, subject, "
            "source_path, chunk_index, section_path, content_hash, last_updated, "
            "text) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    c.chunk_id,
                    c.doc_id,
                    c.subject,
                    c.source_path,
                    c.chunk_index,
                    c.section_path,
                    c.content_hash,
                    c.last_updated,
                    c.text,
                )
                for c in chunks
            ],
        )
        self.db.commit()

    def count_staged(self, subject: str) -> int:
        row = self.db.execute(
            "SELECT COUNT(*) AS n FROM staged_chunks WHERE subject = ?", (subject,)
        ).fetchone()
        return int(row["n"])

    def iter_staged_batches(
        self, subject: str, batch_size: int
    ) -> Iterator[list[Chunk]]:
        """Yield staged chunks in batches, ordered so a doc's chunks stay together."""
        cur = self.db.execute(
            "SELECT * FROM staged_chunks WHERE subject = ? "
            "ORDER BY doc_id, chunk_index",
            (subject,),
        )
        batch: list[Chunk] = []
        for r in cur:
            batch.append(
                Chunk(
                    doc_id=r["doc_id"],
                    subject=r["subject"],
                    source_path=r["source_path"],
                    chunk_index=r["chunk_index"],
                    text=r["text"],
                    section_path=r["section_path"],
                    content_hash=r["content_hash"],
                    last_updated=r["last_updated"],
                )
            )
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch
