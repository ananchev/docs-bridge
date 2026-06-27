"""Javadoc-HTML -> clean Markdown preprocessor (standalone; runs OUTSIDE the product).

A corpus *preprocessor*, not part of the docs_bridge package — it never imports
docs_bridge. Sibling of doxy2md.py: it turns a published Javadoc HTML tree into clean
Markdown the existing ingest already accepts (`.md` is in config.SUPPORTED_SUFFIXES),
so no change to parse/ingest is needed.

Handles BOTH Javadoc HTML generations, auto-detected per page:
  * MODERN  (JDK 11+ "new" doclet): `<body class="class-declaration-page">`,
            `h1.title`, `section.class-description div.type-signature`,
            `main section.detail` / `div.member-signature`, `dl.notes`.
  * LEGACY  (JDK 8/9/10 doclet, e.g. Teamcenter TcDoclet): `<body>` (no class),
            `h2.title`, `div.subTitle` package, `div.description > pre`,
            `div.details` with `<h4>` members + `<pre class="methodSignature">` + `<dl>`.

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

    python javadoc2md.py <src_html_dir> <dst_md_dir>

Only the two content page types are emitted (class/interface/enum/record pages and
package-summary pages). Every nav/index/use/tree/search/help page is dropped.
"""
from __future__ import annotations
import re, sys
from pathlib import Path
from bs4 import BeautifulSoup

# --- which MODERN javadoc pages carry real content (selected by <body class>) ----
KEEP_BODY = {"class-declaration-page", "package-declaration-page"}

# --- LEGACY title kinds (text of <h2 class="title">) -----------------------------
LEGACY_KIND = re.compile(
    r"^(Class|Interface|Enum|Annotation Type|Record|Record Class|Enum Class)\s+(.+)$")

# notes labels to drop (navigational, not developer-relevant). Compared colon-stripped.
DROP_NOTES = {"specified by", "overrides", "see also", "since",
              "all implemented interfaces", "all known implementing classes",
              "all known subinterfaces", "all superinterfaces",
              "enclosing class", "enclosing interface", "functional interface"}

# --- rendering helpers -------------------------------------------------------
def clean(s: str) -> str:
    return re.sub(r"\s+", " ", s.replace("\xa0", " ")).strip()

def block_text(el) -> str:
    """Visible prose of a <div class="block"> (or similar), links flattened to
    their anchor text by get_text() — the symbol name is the meaningful token."""
    return clean(el.get_text(" ")) if el else ""

def render_notes(dl) -> list[str]:
    """A javadoc <dl> (modern: class="notes") holds Parameters/Returns/Throws/etc.
    as alternating <dt>label</dt> <dd>..</dd>* runs. Keep the developer-relevant
    labels; drop navigational ones (Specified by / Overrides / See Also)."""
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
        elif label and label not in DROP_NOTES:
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

# ======================= MODERN (JDK 11+) ====================================
def render_member(sec) -> list[str]:
    """One <section class="detail">: heading + signature + description + notes."""
    h3 = sec.select_one("h3")
    name = clean(h3.get_text()) if h3 else ""
    if not name:
        return []
    md = [f"## {name}", ""]
    sig = sec.select_one("div.member-signature")
    if sig:
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
    return out if len(out.splitlines()) > 3 else None

def convert_package(soup, h1) -> str | None:
    name = re.sub(r"^Package\s+", "", clean(h1.get_text()))
    md = [f"# Package {name}", ""]
    desc = soup.select_one("main div.block")
    body = block_text(desc)
    if body:
        md += [body, ""]
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

# ======================= LEGACY (JDK 8/9/10, TcDoclet) =======================
def render_member_legacy(h4) -> list[str]:
    """One legacy member: <h4>name</h4> + <pre> signature + <div class="block">
    + <dl> notes, all siblings of the h4 inside its <li class="blockList">."""
    name = clean(h4.get_text())
    if not name:
        return []
    md = [f"## {name}", ""]
    block = None
    notes = []
    sig = None
    for sib in h4.find_next_siblings():
        nm = getattr(sib, "name", None)
        if nm == "h4":
            break                                    # next member
        if nm == "pre" and sig is None:
            sig = sib
        elif nm == "div" and "block" in (sib.get("class") or []) and block is None:
            block = sib
        elif nm == "dl":
            notes.append(sib)
    if sig:
        md += ["```java", clean(sig.get_text()), "```", ""]
    body = block_text(block)
    if body:
        md += [body, ""]
    for dl in notes:
        md += render_notes(dl)
    if md and md[-1] != "":
        md.append("")
    return md

def convert_class_legacy(soup) -> str | None:
    h2 = soup.select_one("h2.title")
    if not h2:
        return None
    m = LEGACY_KIND.match(clean(h2.get_text()))
    if not m:                                        # not a class/iface/enum page
        return None
    simple = m.group(2)
    pkg_a = soup.select_one("div.subTitle a")
    pkg = clean(pkg_a.get_text()) if pkg_a else ""
    fqn = f"{pkg}.{simple}" if pkg else simple

    md = [f"# {fqn}", ""]
    desc_div = soup.select_one("div.description")
    if desc_div:
        sig = desc_div.find("pre")
        if sig:
            md += ["```java", clean(sig.get_text()), "```", ""]
        body = block_text(desc_div.select_one("div.block"))
        if body:
            md += [body, ""]

    details = soup.select_one("div.details")
    if details:
        for h4 in details.find_all("h4"):
            md += render_member_legacy(h4)

    out = "\n".join(md).rstrip() + "\n"
    return out if len(out.splitlines()) > 3 else None

def convert_package_legacy(soup) -> str | None:
    title_el = soup.select_one("h1.title") or soup.select_one("h2.title")
    if not title_el:
        return None
    m = re.match(r"^Package\s+(.+)$", clean(title_el.get_text()))
    if not m:
        return None
    name = m.group(1)
    md = [f"# Package {name}", ""]
    block = soup.select_one("div.contentContainer div.block") or soup.select_one("div.block")
    body = block_text(block)
    if body:
        md += [body, ""]
    entries = []
    for tbl in soup.select("table.typeSummary, table.packageSummary, table.overviewSummary"):
        for row in tbl.select("tr"):
            a = row.select_one("th.colFirst a, td.colFirst a")
            if not a:
                continue
            nm = clean(a.get_text())
            d_el = row.select_one("td.colLast div.block") or row.select_one("td.colLast")
            d = clean(d_el.get_text()) if d_el else ""
            entries.append(f"- `{nm}` — {d}" if d else f"- `{nm}`")
    if entries:
        md += ["**Contents**"] + entries + [""]
    out = "\n".join(md).rstrip() + "\n"
    return out if len(out.splitlines()) > 1 else None

# --- one page ----------------------------------------------------------------
def convert(html: str) -> str | None:
    soup = BeautifulSoup(html, "lxml")
    body = soup.find("body")
    if not body:
        return None
    classes = set(body.get("class", []))

    # MODERN: selected by the JDK <body class>.
    if classes & KEEP_BODY:
        h1 = soup.select_one("main h1.title") or soup.select_one("h1.title")
        if not h1:
            return None
        if "package-declaration-page" in classes:
            return convert_package(soup, h1)
        return convert_class(soup, h1)

    # LEGACY: <body> has no declaration-page class. Try class page, then package
    # summary. Nav/use/index/tree/help pages match neither -> dropped.
    return convert_class_legacy(soup) or convert_package_legacy(soup)

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
