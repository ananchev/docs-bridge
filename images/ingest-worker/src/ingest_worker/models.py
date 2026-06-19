"""Plain data structures shared across the two passes.

These mirror the metadata contract in design §8: every chunk carries
doc_id, source_path, subject, section_path, content_hash, last_updated.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Subject:
    """One independent corpus: a source dir mapped to a Qdrant collection."""

    name: str
    dir: str
    collection: str


@dataclass
class Chunk:
    """A structure-aware chunk staged in pass 1, embedded in pass 2.

    `content_hash` is the *document's* hash (not the chunk's): it is what the
    hash-delta logic compares against the manifest, and what ties every chunk of
    a doc together for shrink-safe deletes.
    """

    doc_id: str
    subject: str
    source_path: str
    chunk_index: int
    text: str
    section_path: str
    content_hash: str
    last_updated: str

    @property
    def chunk_id(self) -> str:
        # Stable, human-readable id. design §8: "{doc_id}:{chunk_index}".
        return f"{self.doc_id}:{self.chunk_index}"


@dataclass
class DocState:
    """A document as the manifest knows it (one row in `docs`)."""

    doc_id: str
    subject: str
    source_path: str
    content_hash: str
    last_updated: str
    chunk_count: int


@dataclass
class SyncStats:
    """Per-subject outcome. Powers the idempotency check (design §15):
    a no-change re-sync must report 0 new / 0 changed / 0 deleted."""

    subject: str
    new: int = 0
    changed: int = 0
    deleted: int = 0
    unchanged: int = 0
    chunks_embedded: int = 0

    @property
    def is_noop(self) -> bool:
        return self.new == 0 and self.changed == 0 and self.deleted == 0

    def __str__(self) -> str:
        return (
            f"[{self.subject}] {self.new} new / {self.changed} changed / "
            f"{self.deleted} deleted / {self.unchanged} unchanged "
            f"({self.chunks_embedded} chunks embedded)"
        )
