"""Config loading. The whole Pi->M2 portability model is config-only (design §6):
host-variable values live in containers-at-home group_vars, get templated into
this file, and mounted at /config/config.yaml. Nothing host-specific is baked
into the image.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import yaml

from .models import Subject

DEFAULT_CONFIG_PATH = os.environ.get("DOCS_BRIDGE_CONFIG", "/config/config.yaml")

# BGE-M3 dense dimension. Overridable in config for a different embedding model,
# but the full-fidelity stack pins BAAI/bge-m3 (design §1).
DEFAULT_EMBED_DIM = 1024

SUPPORTED_SUFFIXES = {".pdf", ".html", ".htm", ".md", ".docx", ".pptx"}


@dataclass
class ChunkCfg:
    target_tokens: int = 400
    overlap: int = 60
    strategy: str = "structure_aware"


@dataclass
class QdrantCfg:
    host: str = "qdrant"
    port: int = 6333
    on_disk_vectors: bool = False  # pi5: false (tiny corpus); macmini: true
    quantization: str = "none"     # "none" | "scalar"


@dataclass
class ParseCfg:
    ocr: bool = False              # OCR is slow on CPU + needless for digital PDFs
    table_structure: bool = True   # keep: tables carry real content


@dataclass
class IngestCfg:
    batch_size: int = 8            # pi5: 8 (peak RAM < 8GB); macmini: 64
    two_pass: bool = True          # release Docling before loading BGE-M3


@dataclass
class RerankCfg:
    """docs-bridge server only. BGE-reranker-v2-m3 as ONNX/INT8 (built on the M2,
    baked into the server image — the 8GB Pi can't quantize, only run)."""
    enabled: bool = True
    model_dir: str = "/opt/bge-reranker-onnx"
    int8: bool = True
    max_length: int = 512          # cross-encoder truncates (query, passage) pairs
    top_n: int = 30                # rerank this many fused candidates, then cut to k
    # Total candidate budget when a search spans MULTIPLE pools (subject=list|"all").
    # Candidates are gathered across the named collections and capped at this many
    # before rerank, so latency stays bounded regardless of pool count (rerank is the
    # cost). Bigger than top_n so cross-pool hits aren't squeezed out; single-pool
    # search still uses top_n unchanged.
    multi_top_n: int = 60


@dataclass
class ServerCfg:
    """docs-bridge server only (ignored by the ingest-worker)."""
    host: str = "0.0.0.0"
    port: int = 8080
    default_k: int = 6             # design §9: search(..., k=6)
    prefetch_limit: int = 50       # per branch (dense / sparse) before RRF fusion
    # MCP server `instructions` (returned in the initialize handshake). Clients (e.g.
    # LibreChat) inject this into the model's context to steer how the tools' results
    # are used — citation + answer-language policy. Config-driven so it's tunable via
    # the mounted config.yaml (re-render + restart) without rebuilding the image.
    instructions: str = ""


@dataclass
class Config:
    embedding_model: str
    embedding_dim: int
    chunk: ChunkCfg
    parse: ParseCfg
    qdrant: QdrantCfg
    ingest: IngestCfg
    subjects: list[Subject]
    manifest_path: str
    suffixes: set[str] = field(default_factory=lambda: set(SUPPORTED_SUFFIXES))
    # Global file filters (globs matched against the doc_id = path relative to the
    # subject dir, posix). Apply to EVERY subject and are unioned with each subject's
    # own include/exclude. include = whitelist (any include pattern at either level
    # means a file must match one); exclude = blacklist (exclude wins). See scan().
    include: list[str] = field(default_factory=list)
    exclude: list[str] = field(default_factory=list)
    # Embedding backend (design: ONNX/INT8 validated 2026-06-21 as the default —
    # 4.5x faster than FlagEmbedding/torch, retrieval quality on par). The INT8 model
    # is baked into the image at onnx_model_dir; "flagembedding" stays as a fallback.
    embedding_backend: str = "onnx"        # "onnx" | "flagembedding"
    embedding_int8: bool = True            # onnx only: model.int8.onnx vs model.onnx
    onnx_model_dir: str = "/opt/bge-m3-onnx"
    rerank: RerankCfg = field(default_factory=RerankCfg)    # server only
    server: ServerCfg = field(default_factory=ServerCfg)    # server only

    def subject(self, name: str) -> Subject:
        for s in self.subjects:
            if s.name == name:
                return s
        known = ", ".join(s.name for s in self.subjects) or "<none>"
        raise KeyError(f"unknown subject {name!r}; configured subjects: {known}")


def load(path: str | None = None) -> Config:
    path = path or DEFAULT_CONFIG_PATH
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    chunk = ChunkCfg(**(raw.get("chunk") or {}))
    parse = ParseCfg(**(raw.get("parse") or {}))
    qdrant = QdrantCfg(**(raw.get("qdrant") or {}))
    ingest = IngestCfg(**(raw.get("ingest") or {}))
    rerank = RerankCfg(**(raw.get("rerank") or {}))
    server = ServerCfg(**(raw.get("server") or {}))

    subjects = [
        Subject(
            name=s["name"],
            dir=s["dir"],
            collection=s["collection"],
            description=(s.get("description") or "").strip(),
            include=tuple(s.get("include") or ()),
            exclude=tuple(s.get("exclude") or ()),
        )
        for s in (raw.get("subjects") or [])
    ]
    if not subjects:
        raise ValueError(f"no subjects configured in {path}")

    suffixes = raw.get("suffixes")
    suffixes = {s.lower() for s in suffixes} if suffixes else set(SUPPORTED_SUFFIXES)

    return Config(
        embedding_model=raw.get("embedding_model", "BAAI/bge-m3"),
        embedding_dim=int(raw.get("embedding_dim", DEFAULT_EMBED_DIM)),
        chunk=chunk,
        parse=parse,
        qdrant=qdrant,
        ingest=ingest,
        subjects=subjects,
        manifest_path=raw.get("manifest_path", "/data/state/manifest.sqlite"),
        suffixes=suffixes,
        include=list(raw.get("include") or []),
        exclude=list(raw.get("exclude") or []),
        embedding_backend=raw.get("embedding_backend", "onnx"),
        embedding_int8=bool(raw.get("embedding_int8", True)),
        onnx_model_dir=raw.get("onnx_model_dir", "/opt/bge-m3-onnx"),
        rerank=rerank,
        server=server,
    )
