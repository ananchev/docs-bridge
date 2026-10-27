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
from .ort_session import check_precision

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
class EmbedCfg:
    """Where vectors come from -- worker and server alike. REQUIRED (plan §8 3b: no
    silent defaults); _embed() checks it.
      mode: service   the embedder service (embed_service.py) at `url` -- EC2. The
                      service owns the model, the runtime and the precision; this
                      process holds no embedder at all.
      mode: local     OnnxEmbedder in-process from `model_dir` on `device` at
                      `precision` -- the CPU INT8 line (runtime-cpu images)."""
    mode: str
    url: str | None = None
    device: str | None = None
    precision: str | None = None
    model_dir: str | None = None


@dataclass
class RerankCfg:
    """docs-bridge server only. BGE-reranker-v2-m3 as ONNX, from the models image.

    device / precision / bucket are REQUIRED (plan §8 3b: no silent defaults) -- a
    config without them refuses to start, naming the key; _rerank() checks them.
      device     cpu | cuda    (cuda = CUDA provider only, never a CPU fallback)
      precision  int8 | fp32 | fp16   (int8 is CPU only, fp16 CUDA only)
      bucket     0 = one batch; N = length-sorted sub-batches of N (rerank.py)"""
    device: str
    precision: str
    bucket: int
    enabled: bool = True
    model_dir: str = "/opt/bge-reranker-onnx"
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
    embed: EmbedCfg
    suffixes: set[str] = field(default_factory=lambda: set(SUPPORTED_SUFFIXES))
    # Global file filters (globs matched against the doc_id = path relative to the
    # subject dir, posix). Apply to EVERY subject and are unioned with each subject's
    # own include/exclude. include = whitelist (any include pattern at either level
    # means a file must match one); exclude = blacklist (exclude wins). See scan().
    include: list[str] = field(default_factory=list)
    exclude: list[str] = field(default_factory=list)
    rerank: RerankCfg = field(default_factory=RerankCfg)    # server only
    server: ServerCfg = field(default_factory=ServerCfg)    # server only

    def subject(self, name: str) -> Subject:
        for s in self.subjects:
            if s.name == name:
                return s
        known = ", ".join(s.name for s in self.subjects) or "<none>"
        raise KeyError(f"unknown subject {name!r}; configured subjects: {known}")


def _check_unique(subjects: list[Subject], path: str) -> None:
    """One name and one collection per subject. Two subjects MAY share a dir (each
    picks its files with include/exclude), but never a collection: a collection is
    the unit stamped with one embedder fingerprint (manifest `collections`), and a
    doc-scoped delete in a shared collection would hit the other subject's points."""
    for attr in ("name", "collection"):
        seen: dict[str, str] = {}
        for s in subjects:
            v = getattr(s, attr)
            if v in seen:
                raise ValueError(
                    f"subjects {seen[v]!r} and {s.name!r} share {attr} {v!r} in {path}; "
                    f"each subject needs its own {attr}"
                )
            seen[v] = s.name


RERANK_REQUIRED = ("device", "precision", "bucket")
EMBED_REQUIRED = {"service": ("url",), "local": ("device", "precision", "model_dir")}
# Top-level keys the `embed:` section replaced; refused by name, never ignored.
EMBED_REPLACED = ("embedding_backend", "embedding_int8", "onnx_model_dir")


def _embed(raw: dict, top: dict, path: str) -> EmbedCfg:
    """EmbedCfg from the `embed:` section, refusing anything missing or impossible."""
    old = [k for k in EMBED_REPLACED if k in top]
    if old:
        raise ValueError(f"{', '.join(old)} in {path} replaced by the `embed:` section "
                         f"(mode: service + url, or mode: local + device, precision, "
                         f"model_dir)")
    mode = raw.get("mode")
    if mode not in EMBED_REQUIRED:
        got = "not set" if mode is None else repr(mode)
        raise ValueError(f"embed.mode {got} in {path}: service (the embedder service "
                         f"at embed.url) or local (in-process) -- required, no default")
    missing = [k for k in EMBED_REQUIRED[mode] if raw.get(k) is None]
    if missing:
        raise ValueError(f"embed.{', embed.'.join(missing)} not set in {path} -- "
                         f"required for embed.mode {mode}, no default")
    stray = [k for k in raw if k != "mode" and k not in EMBED_REQUIRED[mode]]
    if stray:
        raise ValueError(f"embed.{', embed.'.join(stray)} in {path} means nothing with "
                         f"embed.mode {mode} -- the "
                         + ("service owns its model, device and precision"
                            if mode == "service" else "local embedder has no url"))
    cfg = EmbedCfg(**raw)
    if mode == "local":
        check_precision("embed", cfg.device, cfg.precision, path)
    return cfg


def _rerank(raw: dict, path: str) -> RerankCfg:
    """RerankCfg from the `rerank:` section, refusing anything missing or impossible."""
    if "int8" in raw:
        raise ValueError(f"rerank.int8 in {path} was replaced by rerank.device + "
                         f"rerank.precision (int8 on CPU: device: cpu, precision: int8)")
    missing = [k for k in RERANK_REQUIRED if raw.get(k) is None]
    if missing:
        raise ValueError(f"rerank.{', rerank.'.join(missing)} not set in {path} -- "
                         f"required, no default (device: cpu|cuda, precision: "
                         f"int8|fp32|fp16, bucket: 0 or a sub-batch size)")
    cfg = RerankCfg(**raw)
    check_precision("rerank", cfg.device, cfg.precision, path)
    if not isinstance(cfg.bucket, int) or isinstance(cfg.bucket, bool) or cfg.bucket < 0:
        raise ValueError(f"rerank.bucket {cfg.bucket!r} in {path}: 0 or a positive int")
    return cfg


def load(path: str | None = None) -> Config:
    path = path or DEFAULT_CONFIG_PATH
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    chunk = ChunkCfg(**(raw.get("chunk") or {}))
    parse = ParseCfg(**(raw.get("parse") or {}))
    qdrant = QdrantCfg(**(raw.get("qdrant") or {}))
    ingest = IngestCfg(**(raw.get("ingest") or {}))
    embed = _embed(raw.get("embed") or {}, raw, path)
    rerank = _rerank(raw.get("rerank") or {}, path)
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
    _check_unique(subjects, path)

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
        embed=embed,
        rerank=rerank,
        server=server,
    )
