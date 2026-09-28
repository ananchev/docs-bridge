"""Pass 1 - parse.

Docling converts each source file (PDF/HTML/DOCX/...) and a structure-aware
chunker splits it into chunks that carry their heading path. This module owns the
Docling import so the converter is created and released entirely within the parse
pass (the embedder never sees it).

API notes, re-verified on docling==2.130.0 / docling-core==2.99.0 (2026-09-28);
re-check all of them on the next bump:
  * `HybridChunker` lives at `docling.chunking.HybridChunker`; chunk heading
    metadata is `chunk.meta.headings`.
  * `HybridChunker(max_tokens=...)` still works but is deprecated: the budget now
    rides on an explicit tokenizer object. We pass the SAME tokenizer the old
    default used (sentence-transformers/all-MiniLM-L6-v2), so a 400-token chunk
    means what it meant when `teamcenter` was built.
  * PPTX speaker notes are extracted, but into `ContentLayer.NOTES`, which the
    chunker skips — so a notes-only deck yields ZERO chunks unless we move them
    into the body (`_promote_speaker_notes`). Reviewer comments share that layer
    inside `COMMENT_SECTION` groups and are deliberately left out.
  * Authored image alt-text (`descr` in DOCX/PPTX) is still ignored by docling:
    the `PictureItem` arrives with no caption and serializes to nothing
    (`_attach_alt_text`).
"""

from __future__ import annotations

import hashlib
import io
import logging
import re
import zipfile
from datetime import datetime, timezone
from fnmatch import fnmatch
from pathlib import Path
from xml.etree import ElementTree

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


# The tokenizer HybridChunker counted chunk budgets with before docling-core made it
# explicit. Kept deliberately: chunk sizes stay comparable with the existing corpus.
CHUNK_TOKENIZER = "sentence-transformers/all-MiniLM-L6-v2"

# Alt-text that says nothing: Office placeholders ("Picture 3", "image1.png") and
# Office's own auto-captions, which describe pixels, not meaning.
_JUNK_ALT = re.compile(
    r"^\s*((picture|image|grafik|bild|graphic|chart|diagram)\s*\d*"
    r"|.*\.(png|jpe?g|gif|bmp|emf|wmf|tiff?|svg))\s*$"
    r"|automatically generated",
    re.IGNORECASE,
)
_NS_R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_NS_A = "http://schemas.openxmlformats.org/drawingml/2006/main"
_NS_PKG_REL = "http://schemas.openxmlformats.org/package/2006/relationships"


def _pixel_key(img) -> str:
    """Identity of an image by its decoded pixels, so an OOXML media part and the
    docling PictureItem built from it match regardless of re-encoding."""
    rgb = img.convert("RGB")
    return hashlib.sha1(f"{rgb.size}".encode() + rgb.tobytes()).hexdigest()


def _ooxml_alt_texts(path: Path) -> dict[str, str]:
    """Pixel-key -> authored alt-text for every captioned image in a DOCX/PPTX.

    Matching by pixels rather than by position means a picture docling skipped or
    reordered can never inherit a neighbour's description. Images whose bytes PIL
    cannot decode (EMF/WMF) are skipped — docling cannot render those either."""
    from PIL import Image

    out: dict[str, str] = {}
    try:
        zf = zipfile.ZipFile(path)
    except (zipfile.BadZipFile, OSError):
        return out
    with zf:
        names = set(zf.namelist())
        parts = [n for n in names if re.match(r"(word/document|ppt/slides/slide\d+)\.xml$", n)]
        for part in parts:
            rels_name = f"{part.rsplit('/', 1)[0]}/_rels/{part.rsplit('/', 1)[1]}.rels"
            if rels_name not in names:
                continue
            rels = {
                r.get("Id"): r.get("Target")
                for r in ElementTree.fromstring(zf.read(rels_name))
                if r.tag == f"{{{_NS_PKG_REL}}}Relationship"
            }
            root = ElementTree.fromstring(zf.read(part))
            parent = {c: p for p in root.iter() for c in p}
            for el in root.iter():
                alt = (el.get("descr") or "").strip()
                if not alt or _JUNK_ALT.search(alt):
                    continue
                # climb from docPr / cNvPr to the element that holds the a:blip
                node, blip = el, None
                for _ in range(4):
                    node = parent.get(node)
                    if node is None:
                        break
                    blip = node.find(f".//{{{_NS_A}}}blip")
                    if blip is not None:
                        break
                rid = blip.get(f"{{{_NS_R}}}embed") if blip is not None else None
                target = rels.get(rid or "")
                if not target:
                    continue
                media = (Path(part).parent / target).as_posix()
                media = re.sub(r"[^/]+/\.\./", "", media)
                if media not in names:
                    continue
                try:
                    key = _pixel_key(Image.open(io.BytesIO(zf.read(media))))
                except Exception:  # noqa: BLE001 — undecodable vector formats
                    continue
                out.setdefault(key, alt)
    return out


def _promote_speaker_notes(doc) -> int:
    """Move PPTX speaker notes from the NOTES layer into the body, so the chunker
    indexes them under their slide. Comments (inside COMMENT_SECTION groups) stay
    where they are. Returns how many notes were promoted."""
    from docling_core.types.doc import ContentLayer, DocItemLabel, GroupLabel

    promoted = 0
    for item, _ in doc.iterate_items(included_content_layers={ContentLayer.NOTES}):
        if getattr(item, "label", None) != DocItemLabel.TEXT:
            continue
        par = item.parent.resolve(doc) if item.parent else None
        if getattr(par, "label", None) == GroupLabel.COMMENT_SECTION:
            continue
        item.content_layer = ContentLayer.BODY
        item.text = f"Speaker notes: {item.text}"
        promoted += 1
    return promoted


def _attach_alt_text(doc, path: Path) -> int:
    """Give every picture that has authored alt-text a caption carrying it, so the
    chunk that holds the picture carries its description. Returns how many."""
    from docling_core.types.doc import DocItemLabel, PictureItem

    if path.suffix.lower() not in (".docx", ".pptx"):
        return 0
    alts = _ooxml_alt_texts(path)
    if not alts:
        return 0
    attached = 0
    for item, _ in doc.iterate_items():
        if not isinstance(item, PictureItem) or item.captions or item.image is None:
            continue
        pil = item.image.pil_image
        alt = alts.get(_pixel_key(pil)) if pil is not None else None
        if not alt:
            continue
        cap = doc.add_text(label=DocItemLabel.CAPTION, text=f"Figure: {alt}", parent=item)
        item.captions.append(cap.get_ref())
        attached += 1
    return attached


class Parser:
    """Holds the Docling converter + chunker for the duration of the parse pass."""

    def __init__(self, cfg: Config) -> None:
        from docling.chunking import HybridChunker
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import (
            ConvertPipelineOptions,
            PdfPipelineOptions,
            RapidOcrOptions,
        )
        from docling.document_converter import (
            DocumentConverter,
            PdfFormatOption,
            PowerpointFormatOption,
            WordFormatOption,
        )
        from docling_core.transforms.chunker.tokenizer.huggingface import (
            HuggingFaceTokenizer,
        )

        self.cfg = cfg
        # OCR does NOT run on every page by default: with force_full_page_ocr=False
        # docling OCRs a page in full only when bitmaps cover >75% of it, otherwise
        # only the detected bitmap regions, and not at all when there are none. So
        # on digital PDFs it costs little more than their figures. It stays a
        # per-deployment switch (parse.ocr); when on, the engine is RapidOCR on
        # onnxruntime — the runtime this stack already ships — not torch EasyOCR.
        # Table-structure detection stays on (tables carry content).
        pdf_opts = PdfPipelineOptions()
        pdf_opts.do_ocr = cfg.parse.ocr
        pdf_opts.ocr_options = RapidOcrOptions(backend="onnxruntime")
        pdf_opts.do_table_structure = cfg.parse.table_structure
        # Office formats get their own registered options. Without an entry here
        # every future enrichment flag (picture description, Tier 3) silently
        # no-ops on DOCX/PPTX (docling-jobkit #145) — the PDF-only map was that bug.
        office_opts = ConvertPipelineOptions()
        self._convert = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_opts),
                InputFormat.DOCX: WordFormatOption(pipeline_options=office_opts),
                InputFormat.PPTX: PowerpointFormatOption(pipeline_options=office_opts),
            }
        ).convert
        # Structure-aware chunking with the configured token budget. HybridChunker
        # merges undersized peers and splits oversized blocks to ~max_tokens.
        self._chunker = HybridChunker(
            tokenizer=HuggingFaceTokenizer.from_pretrained(
                model_name=CHUNK_TOKENIZER, max_tokens=cfg.chunk.target_tokens
            )
        )

    def parse(
        self, subject: Subject, doc_id: str, path: Path, content_hash: str
    ) -> list[Chunk]:
        result = self._convert(str(path))
        doc = result.document
        notes = _promote_speaker_notes(doc)
        alts = _attach_alt_text(doc, path)
        if notes or alts:
            log.debug("%s: %d speaker notes, %d alt-texts", doc_id, notes, alts)
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
