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

    subjects = [
        Subject(name=s["name"], dir=s["dir"], collection=s["collection"])
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
    )
