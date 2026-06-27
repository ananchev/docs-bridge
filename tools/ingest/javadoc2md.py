"""Javadoc-HTML -> clean Markdown preprocessor (standalone; runs OUTSIDE the product).

A corpus *preprocessor*, not part of the docs_bridge package — it never imports
docs_bridge. Sibling of doxy2md.py: it turns a published **JDK Javadoc** HTML tree
into clean Markdown the existing ingest already accepts (`.md` is in
config.SUPPORTED_SUFFIXES), so no change to parse/ingest is needed.

Output mirrors the input tree (same relative path + basename, .html -> .md), so the
copy step's structure mirroring (and the FQN-rooted doc ids it produces) stay intact.

Per page:
  - `# <fully-qualified class name>` = the class/interface/enum title. Using the FQN
    (not just the simple name) matters here: large multi-module sets reuse simple class
    names, so the FQN is what disambiguates a chunk during retrieval and reads cleanly
    in a citation, e.g. "com.example.app.services.core.DataService".
  - `## <member>` = each documented constructor / method / field. Docling's
    HybridChunker turns those headings into the chunk `section_path`, so a citation
    reads e.g. "...DataService > getRecords".

    python tools/ingest/javadoc2md.py <src_html_dir> <dst_md_dir>

Only the two content page types are emitted (selected by the JDK `<body class>`):
`class-declaration-page` (classes/interfaces/enums/records) and
`package-declaration-page` (package-summary). Every nav/index/use/tree/search/help
page is dropped.
"""
from __future__ import annotations
import re, sys
from pathlib import Path
from bs4 import BeautifulSoup

# --- which javadoc pages carry real content (selected by <body class>) -------
KEEP_BODY = {"class-declaration-page", "package-declaration-page"}

# --- rendering helpers -------------------------------------------------------
def clean(s: str) -> str:
    return re.sub(r"\s+", " ", s.replace("\xa0", " ")).strip()

def block_text(el) -> str:
    """Visible prose of a <div class="block"> (or similar), links flattened to
    their anchor text by get_text() — the symbol name is the meaningful token."""
    return clean(el.get_text(" ")) if el else ""

def render_notes(dl) -> list[str]:
    """A javadoc <dl class="notes"> holds Parameters/Returns/Throws/etc. as
    alternating <dt>label</dt> <dd>..</dd>* runs. Keep the developer-relevant
    labels; drop navigational ones (Specified by / Overrides / See Also)."""
    DROP = {"specified by:", "overrides:", "see also:", "since:",
            "all implemented interfaces:", "all known implementing classes:",
            "all known subinterfaces:", "all superinterfaces:",
            "enclosing class:", "enclosing interface:", "functional interface:"}
    out, label = [], None
    bullets: list[str] = []

    def flush():
        nonlocal bullets, label
        if label and bullets:
            if label in ("parameters", "throws", "type parameters"):
                cap = {"parameters": "Parameters", "throws": "Throws",
                       "type parameters": "Type Parameters"}[label]
                out.append(f"**{cap}**")
                out.extend(bullets)
                out.append("")                       # blank line between groups
            elif label == "returns":
                out.append(f"**Returns** {' '.join(bullets)}".strip())
                out.append("")
        bullets = []

    for child in dl.find_all(["dt", "dd"], recursive=False):
        if child.name == "dt":
            flush()
            label = clean(child.get_text()).rstrip(":").lower()
        elif label and label not in DROP:
            txt = clean(child.get_text(" "))
            if not txt:
                continue
            if label in ("parameters", "throws", "type parameters"):
                # dd looks like "name - description" (name was a <code>); the
                # description is often empty -> dd text is just "name -".
                m = re.match(r"^(.*?)\s+-\s+(.*)$", txt, re.S)
                name, desc = (m.group(1), m.group(2)) if m else (txt.rstrip(" -"), "")
                bullets.append(f"- `{name.strip()}` — {desc.strip()}"
                               if desc.strip() else f"- `{name.strip()}`")
            else:  # returns and any other kept single-value label
                bullets.append(txt)
    flush()
    return out

def render_member(sec) -> list[str]:
    """One <section class="detail">: heading + signature + description + notes."""
    h3 = sec.select_one("h3")
    name = clean(h3.get_text()) if h3 else ""
    if not name:
        return []
    md = [f"## {name}", ""]
    sig = sec.select_one("div.member-signature")
    if sig:
        # no separator: javadoc uses &nbsp; where a space is wanted, so plain
        # concatenation keeps generics/params tight ("Map<String,String[]>").
        md += ["```java", clean(sig.get_text()), "```", ""]
    desc = sec.select_one(":scope > div.block")
    body = block_text(desc)
    if body:
        md += [body, ""]
    for dl in sec.select(":scope > dl.notes"):
        md += render_notes(dl)
    if md and md[-1] != "":
        md.append("")
    return md

# --- one page ----------------------------------------------------------------
def convert(html: str) -> str | None:
    soup = BeautifulSoup(html, "lxml")
    body = soup.find("body")
    if not body or not (set(body.get("class", [])) & KEEP_BODY):
        return None

    h1 = soup.select_one("main h1.title") or soup.select_one("h1.title")
    if not h1:
        return None

    if "package-declaration-page" in body.get("class", []):
        return convert_package(soup, h1)
    return convert_class(soup, h1)

def convert_class(soup, h1) -> str | None:
    # FQN = package (from the sub-title) + simple name (from the h1, kind stripped)
    simple = re.sub(r"^(Class|Interface|Enum Class|Record Class|Annotation Interface)\s+",
                    "", clean(h1.get_text()))
    pkg_a = soup.select_one(".header .sub-title a") or soup.select_one(".sub-title a")
    pkg = clean(pkg_a.get_text()) if pkg_a else ""
    fqn = f"{pkg}.{simple}" if pkg else simple

    md = [f"# {fqn}", ""]
    sig = soup.select_one("section.class-description div.type-signature")
    if sig:
        md += ["```java", clean(sig.get_text()), "```", ""]
    desc = soup.select_one("section.class-description > div.block")
    body = block_text(desc)
    if body:
        md += [body, ""]

    for sec in soup.select("main section.detail"):
        md += render_member(sec)

    out = "\n".join(md).rstrip() + "\n"
    # An empty stub (no description, no members) is not worth a corpus doc.
    return out if len(out.splitlines()) > 3 else None

def convert_package(soup, h1) -> str | None:
    name = re.sub(r"^Package\s+", "", clean(h1.get_text()))
    md = [f"# Package {name}", ""]
    desc = soup.select_one("main div.block")
    body = block_text(desc)
    if body:
        md += [body, ""]
    # Contained packages/types: pair each linked name with its description cell.
    entries = []
    for first in soup.select("div.summary-table div.col-first"):
        a = first.select_one("a")
        if not a:
            continue
        nm = clean(a.get_text())
        last = first.find_next_sibling("div", class_="col-last")
        d = clean(last.get_text()) if last else ""
        entries.append(f"- `{nm}` — {d}" if d else f"- `{nm}`")
    if entries:
        md += ["**Contents**"] + entries + [""]
    out = "\n".join(md).rstrip() + "\n"
    return out if len(out.splitlines()) > 1 else None

# --- tree walk ---------------------------------------------------------------
def main(src: Path, dst: Path) -> None:
    # Dedup by FQN within this run: in many multi-module sets every module re-bundles
    # the same shared base/runtime packages as byte-identical pages. When the whole set
    # is converted in one run (e.g. tagging the top-level parent folder), keep only the
    # first occurrence of each FQN -> the corpus holds one copy of each class instead of
    # one per module. The output stays UNDER `dst` either way, so a folder-tag prune
    # that protects the tagged subtree is unaffected. (A single-module run has no
    # internal dups, so this is a no-op for the scoped flow.)
    kept = skipped = deduped = 0
    seen: set[str] = set()
    for p in sorted(src.rglob("*.html")):
        md = None
        try:
            md = convert(p.read_text(encoding="utf-8", errors="replace"))
        except Exception as e:                      # never let one bad page abort
            print(f"  ! {p}: {e}", file=sys.stderr)
        if not md:
            skipped += 1
            continue
        key = md.splitlines()[0]                    # the `# <FQN>` / `# Package x` line
        if key in seen:
            deduped += 1
            continue
        seen.add(key)
        out = dst / p.relative_to(src).with_suffix(".md")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md, encoding="utf-8")
        kept += 1
    print(f"converted {kept} content pages, skipped {skipped} non-content/empty, "
          f"deduped {deduped} repeated FQNs", file=sys.stderr)

if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
