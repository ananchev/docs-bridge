"""Qdrant access for the ingest side: collection creation, idempotent upserts,
and doc-scoped deletes.

Collection layout (must match the docs-bridge query side):
  - named DENSE vector "dense"  (cosine, size = embedding_dim)
  - named SPARSE vector "sparse"
  - payload index on `doc_id` (keyword) so doc-scoped deletes are cheap, and on
    `subject` for query-time filtering.

Point ids are deterministic UUIDv5 of the human "{doc_id}:{chunk_index}" string,
so re-ingesting a doc overwrites its points rather than duplicating them
(idempotent upserts, design §8). The readable chunk_id is also kept in the payload.
"""

from __future__ import annotations

import logging
import time
import uuid

from qdrant_client import QdrantClient
from qdrant_client import models as qm

from .config import Config
from .embed import SparseVec
from .models import Chunk

log = logging.getLogger(__name__)

DENSE = "dense"
SPARSE = "sparse"
_NS = uuid.UUID("6f9619ff-8b86-d011-b42d-00c04fc964ff")  # fixed namespace for ids

# Qdrant can drop the connection mid-upsert (e.g. it restarts under memory pressure),
# surfacing as ResponseHandlingException "Server disconnected without sending a
# response." Upserts are idempotent (deterministic point ids), so we retry with
# capped backoff: a transient disconnect recovers on a fresh httpx connection, and
# the backoff rides out a brief Qdrant restart instead of killing a multi-hour run.
_UPSERT_MAX_ATTEMPTS = 8
_UPSERT_BACKOFF_CAP = 60.0


def point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(_NS, chunk_id))


def connect(cfg: Config) -> QdrantClient:
    return QdrantClient(host=cfg.qdrant.host, port=cfg.qdrant.port)


def ensure_collection(client: QdrantClient, cfg: Config, collection: str) -> None:
    if client.collection_exists(collection):
        return
    log.info("creating collection %s", collection)

    quant = None
    if cfg.qdrant.quantization == "scalar":
        quant = qm.ScalarQuantization(
            scalar=qm.ScalarQuantizationConfig(
                type=qm.ScalarType.INT8, always_ram=True
            )
        )

    client.create_collection(
        collection_name=collection,
        vectors_config={
            DENSE: qm.VectorParams(
                size=cfg.embedding_dim,
                distance=qm.Distance.COSINE,
                on_disk=cfg.qdrant.on_disk_vectors,
            )
        },
        sparse_vectors_config={SPARSE: qm.SparseVectorParams()},
        quantization_config=quant,
    )
    client.create_payload_index(collection, "doc_id", qm.PayloadSchemaType.KEYWORD)
    client.create_payload_index(collection, "subject", qm.PayloadSchemaType.KEYWORD)


def delete_doc(client: QdrantClient, collection: str, doc_id: str) -> None:
    """Remove every point belonging to a doc. Used for changed (before re-insert)
    and deleted docs; also makes shrinking docs shrink-safe without tracking the
    per-chunk id set."""
    client.delete(
        collection_name=collection,
        points_selector=qm.FilterSelector(
            filter=qm.Filter(
                must=[qm.FieldCondition(key="doc_id", match=qm.MatchValue(value=doc_id))]
            )
        ),
    )


def upsert(
    client: QdrantClient,
    collection: str,
    chunks: list[Chunk],
    dense: list[list[float]],
    sparse: list[SparseVec],
) -> None:
    points = [
        qm.PointStruct(
            id=point_id(c.chunk_id),
            vector={
                DENSE: d,
                SPARSE: qm.SparseVector(indices=s.indices, values=s.values),
            },
            payload={
                "chunk_id": c.chunk_id,
                "doc_id": c.doc_id,
                "subject": c.subject,
                "source_path": c.source_path,
                "section_path": c.section_path,
                "chunk_index": c.chunk_index,
                "content_hash": c.content_hash,
                "last_updated": c.last_updated,
                "text": c.text,
            },
        )
        for c, d, s in zip(chunks, dense, sparse)
    ]
    delay = 2.0
    for attempt in range(1, _UPSERT_MAX_ATTEMPTS + 1):
        try:
            client.upsert(collection_name=collection, points=points)
            return
        except Exception as exc:  # transport-level: disconnect / timeout / 5xx
            if attempt == _UPSERT_MAX_ATTEMPTS:
                log.error(
                    "upsert to %s failed after %d attempts: %r",
                    collection, attempt, exc,
                )
                raise
            log.warning(
                "upsert to %s failed (attempt %d/%d): %r; retrying in %.0fs",
                collection, attempt, _UPSERT_MAX_ATTEMPTS, exc, delay,
            )
            time.sleep(delay)
            delay = min(delay * 2, _UPSERT_BACKOFF_CAP)
