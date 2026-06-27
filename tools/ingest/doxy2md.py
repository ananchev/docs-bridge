"""Doxygen-HTML -> clean Markdown preprocessor (standalone; runs OUTSIDE the product).

A corpus *preprocessor*, not part of the docs_bridge package — it never imports
docs_bridge. It turns a published Doxygen HTML tree into clean Markdown the existing
ingest already accepts (`.md` is in config.SUPPORTED_SUFFIXES), so no change to
parse/ingest is needed.

Output mirrors the input tree (same relative path + basename, .html -> .md). That
keeps the nextcloud->inbound copy step's structure mirroring intact, so the
nextcloud tags that ride on that structure are preserved.

Per page: `#` = the class/group title, `##` = each documented member. Docling's
HybridChunker turns those headings into the chunk `section_path`, so citations
read e.g. "File utilities > readBuffer()".

    python tools/doxy2md.py <src_html_dir> <dst_md_dir>

Only class/group/struct/union/interface/namespace pages are emitted; doxygen's
index/nav pages and the duplicate `*-members.html` listings are dropped.
"""
from __future__ import annotations
import re, sys
from pathlib import Path
from bs4 import BeautifulSoup

# --- which doxygen pages carry real content (everything else is nav/index/dupe)
CONTENT_PREFIXES = ("class_", "struct_", "union_", "interface_",
                    "namespace_", "group__")
SKIP_EXACT = {  # generated index/nav pages -> never useful as corpus docs
    "annotated.html", "classes.html", "hierarchy.html", "files.html",
    "namespaces.html", "modules.html", "deprecated.html", "index.html",
    "pages.html", "globals.html", "todo.html", "bug.html",
}

def is_content_page(name: str) -> bool:
    if not name.endswith(".html"):
        return False
    if name in SKIP_EXACT or name.startswith(("functions", "globals", "dir_")):
        return False
    if name.endswith("-members.html"):       # duplicate member-list pages
        return False
    return name.startswith(CONTENT_PREFIXES)

# --- rendering helpers -------------------------------------------------------
def clean(s: str) -> str:
    return re.sub(r"\s+", " ", s.replace(" ", " ")).strip()

def render_fragment(div) -> str:
    """doxygen code example: <div class=fragment><div class=line>..</div>..
    Each line carries a <span class=lineno> gutter -> drop it so the fenced
    block holds only code, not doxygen's line numbers."""
    lines = []
    for ln in div.select("div.line"):
        for gutter in ln.select("span.lineno"):
            gutter.decompose()
        lines.append(ln.get_text().replace(" ", " ").rstrip())
    if not lines:
        lines = [div.get_text()]
    return "```\n" + "\n".join(lines).rstrip() + "\n```"

def render_params(tbl) -> str:
    """doxygen params: <table class=params> rows of (name, [dir], desc)."""
    out = []
    for tr in tbl.select("tr"):
        cells = [clean(td.get_text()) for td in tr.select("td")]
        cells = [c for c in cells if c]
        if not cells:
            continue
        name, desc = cells[0], " ".join(cells[1:])
        out.append(f"- **{name}** — {desc}" if desc else f"- **{name}**")
    return "\n".join(out)

def render_memdoc(div) -> str:
    """Detailed doc body of one member: prose + params + returns + examples."""
    parts = []
    for el in div.children:
        if getattr(el, "name", None) is None:
            continue
        cls = el.get("class", [])
        if "fragment" in cls:
            parts.append(render_fragment(el))
        elif el.name == "dl" and "params" in cls:   # <dl class=params> wrapping a
            tbl = el.select_one("table.params")     # <table class=params> -> bullets
            if tbl:
                parts.append("**Parameters**\n" + render_params(tbl))
        elif el.name == "table" and "params" in cls:
            parts.append("**Parameters**\n" + render_params(el))
        elif el.name == "dl":                       # return values, notes, etc.
            label = clean(el.select_one("dt").get_text()) if el.select_one("dt") else ""
            body = clean(" ".join(dd.get_text() for dd in el.select("dd")))
            parts.append(f"**{label}** {body}".strip())
        elif el.name in ("p", "div"):
            t = clean(el.get_text())
            if t:
                parts.append(t)
    return "\n\n".join(p for p in parts if p.strip())

# --- one page ----------------------------------------------------------------
def convert(html: str) -> str | None:
    soup = BeautifulSoup(html, "lxml")
    title_el = soup.select_one("div.headertitle div.title") or soup.select_one("div.title")
    title = clean(title_el.get_text()) if title_el else clean(
        (soup.title.get_text() if soup.title else "").split(":", 1)[-1])
    contents = soup.select_one("div.contents")
    if not contents or not title:
        return None

    # Cross-refs: doxygen links a symbol to its doc via <a class="el" href=..>.
    # The mangled href is useless to a RAG model, but the visible symbol name is
    # the meaningful token -> keep it, dropped of its link, wrapped as a code span
    # so the model reads it as an API symbol and it survives chunk boundaries.
    # Skip links inside code examples (div.fragment) so we don't inject backticks
    # into a fenced block.
    for a in contents.select("a.el"):
        if a.find_parent("div", class_="fragment"):
            continue
        text = a.get_text().strip()
        if text:
            a.replace_with(soup.new_string(f"`{text}`"))

    md = [f"# {title}", ""]

    # class/group-level detailed description (the prose before the members)
    textblock = contents.select_one("div.textblock")
    if textblock:
        t = clean(textblock.get_text())
        if t:
            md += [t, ""]

    # each documented member: <h2 class=memtitle> + <div class=memitem>
    for memitem in contents.select("div.memitem"):
        h2 = memitem.find_previous_sibling("h2", class_="memtitle")
        name = clean(h2.get_text()).lstrip("◆ ").strip() if h2 else ""
        proto = memitem.select_one("div.memproto")
        memdoc = memitem.select_one("div.memdoc")
        sig = clean(proto.get_text()) if proto else ""
        if name:
            md.append(f"## {name}")
        # Drop a signature that's just the bare member name (Tcl free functions
        # have no typed prototype) — it only echoes the heading. Keep real
        # signatures that carry arguments (e.g. Python methods).
        if sig and sig != name.rstrip("()").strip():
            md += ["", "```", sig, "```", ""]
        if memdoc:
            body = render_memdoc(memdoc)
            if body:
                md += [body, ""]
    return "\n".join(md).rstrip() + "\n"

# --- tree walk ---------------------------------------------------------------
def main(src: Path, dst: Path) -> None:
    kept = skipped = 0
    for p in sorted(src.rglob("*.html")):
        if not is_content_page(p.name):
            skipped += 1
            continue
        md = convert(p.read_text(encoding="utf-8", errors="replace"))
        if not md:
            skipped += 1
            continue
        out = dst / p.relative_to(src).with_suffix(".md")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md, encoding="utf-8")
        kept += 1
    print(f"converted {kept} content pages, skipped {skipped} non-content/empty",
          file=sys.stderr)

if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
