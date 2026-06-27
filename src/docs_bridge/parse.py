"""Pass 1 - parse.

Docling converts each source file (PDF/HTML/DOCX/...) and a structure-aware
chunker splits it into chunks that carry their heading path. This module owns the
Docling import so the converter is created and released entirely within the parse
pass (the embedder never sees it).

API note: pinned to docling==2.15.1. `HybridChunker` lives at
`docling.chunking.HybridChunker`; chunk heading metadata is `chunk.meta.headings`.
Re-verify these two if you bump Docling.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone
from fnmatch import fnmatch
from pathlib import Path

from .config import Config
from .models import Chunk, Subject

log = logging.getLogger(__name__)


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def doc_id_for(subject_dir: Path, path: Path) -> str:
    """Stable id = path relative to the subject dir (posix). Survives re-runs and
    is readable in citations; renaming a file is a delete + add, which is correct."""
    return path.relative_to(subject_dir).as_posix()


def _included(doc_id: str, include: tuple[str, ...], exclude: tuple[str, ...]) -> bool:
    """Apply the whitelist/blacklist globs to a doc_id (posix path relative to the
    subject dir). include is a whitelist — if any pattern is present a file must
    match one; exclude is a blacklist and wins over include. Patterns use fnmatch,
    so `*` spans `/`."""
    if include and not any(fnmatch(doc_id, pat) for pat in include):
        return False
    if any(fnmatch(doc_id, pat) for pat in exclude):
        return False
    return True


def scan(subject: Subject, cfg: Config) -> dict[str, Path]:
    """Map doc_id -> path for every supported, non-filtered file under the subject
    dir. Global (cfg) and per-subject include/exclude globs are unioned."""
    root = Path(subject.dir)
    if not root.is_dir():
        log.warning("subject %s: dir %s does not exist", subject.name, root)
        return {}
    include = tuple(cfg.include) + subject.include
    exclude = tuple(cfg.exclude) + subject.exclude
    found: dict[str, Path] = {}
    for p in sorted(root.rglob("*")):
        if not (p.is_file() and p.suffix.lower() in cfg.suffixes):
            continue
        doc_id = doc_id_for(root, p)
        if not _included(doc_id, include, exclude):
            continue
        found[doc_id] = p
    return found


class Parser:
    """Holds the Docling converter + chunker for the duration of the parse pass."""

    def __init__(self, cfg: Config) -> None:
        from docling.chunking import HybridChunker
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption

        self.cfg = cfg
        # OCR (EasyOCR) runs on every page by default and is the dominant cost on
        # CPU — unnecessary for digital PDFs. Default off; enable per deployment for
        # scanned corpora. Table-structure detection stays on (tables carry content).
        pdf_opts = PdfPipelineOptions()
        pdf_opts.do_ocr = cfg.parse.ocr
        pdf_opts.do_table_structure = cfg.parse.table_structure
        self._convert = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_opts)}
        ).convert
        # Structure-aware chunking with the configured token budget. HybridChunker
        # merges undersized peers and splits oversized blocks to ~max_tokens.
        self._chunker = HybridChunker(max_tokens=cfg.chunk.target_tokens)

    def parse(
        self, subject: Subject, doc_id: str, path: Path, content_hash: str
    ) -> list[Chunk]:
        result = self._convert(str(path))
        doc = result.document
        last_updated = datetime.fromtimestamp(
            path.stat().st_mtime, tz=timezone.utc
        ).isoformat()

        chunks: list[Chunk] = []
        for i, ch in enumerate(self._chunker.chunk(doc)):
            text = (ch.text or "").strip()
            if not text:
                continue
            headings = getattr(ch.meta, "headings", None) or []
            chunks.append(
                Chunk(
                    doc_id=doc_id,
                    subject=subject.name,
                    source_path=str(path),
                    chunk_index=i,
                    text=text,
                    section_path=" > ".join(headings),
                    content_hash=content_hash,
                    last_updated=last_updated,
                )
            )
        return chunks
