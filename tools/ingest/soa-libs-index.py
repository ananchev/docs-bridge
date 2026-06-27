#!/usr/bin/env python3
"""Generate soa-libs-index.md — a Markdown reference of the Teamcenter SOA Java
client libraries, for ingestion into docs-bridge so an LLM can advise *what jars to
copy* for a trimmed client.

It reads every jar under a libs dir and, from each MANIFEST.MF, records the OSGi
identity (`Bundle-SymbolicName`) and dependencies (`Require-Bundle`) plus the SOA
service packages the jar provides. From that it computes, per strong-service jar, the
exact transitive `Require-Bundle` closure (resolved to filenames). Third-party jars
(no bundle headers: http client, fcc/fsc, xerces, sso, jackson, jaxb, log4j, …) are
listed as a fixed runtime baseline since OSGi headers never reference them.

Converter contract (sibling of javadoc2md.py / doxy2md.py): given a source dir and a
destination dir, it writes `<dst>/soa-libs-index.md` **only if the source contains
jars** — so when the ingest loop runs every converter over a tagged folder, this one
emits nothing for a javadoc/doxygen/other folder (mutual exclusivity), exactly like
the others. That lets the SOA ingest script run it on any tagged dir; only a libs dir
produces output.

Usage:  soa-libs-index.py <src_dir> <dst_dir>
"""
from __future__ import annotations
import re, sys, zipfile
from pathlib import Path
from collections import defaultdict

def manifest(jar: Path) -> dict[str, str]:
    """Parse MANIFEST.MF, unfolding 72-col continuation lines (newline + space)."""
    try:
        with zipfile.ZipFile(jar) as z:
            raw = z.read("META-INF/MANIFEST.MF").decode("utf-8", "replace")
    except (KeyError, zipfile.BadZipFile):
        return {}
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")
    raw = re.sub(r"\n ", "", raw)                       # unfold continuations
    out = {}
    for line in raw.split("\n"):
        if ":" in line:
            k, _, v = line.partition(":")
            out[k.strip()] = v.strip()
    return out

def service_pkgs(jar: Path) -> list[str]:
    """SOA service packages provided. Classic modules use com/teamcenter/services/...,
    newer ones a vendor prefix: com/<code>0/services/<typing>/<domain>/."""
    pkgs = set()
    try:
        with zipfile.ZipFile(jar) as z:
            for n in z.namelist():
                m = re.match(r"(com/[a-z0-9_]+)/services/(strong|loose)/([^/]+)/", n)
                if m:
                    root = m.group(1).replace("/", ".")
                    pkgs.add(f"{root}.services.{m.group(2)}.{m.group(3)}")
    except zipfile.BadZipFile:
        pass
    return sorted(pkgs)

def req_bundles(mf: dict[str, str]) -> list[str]:
    rb = mf.get("Require-Bundle", "")
    # entries are comma-separated; each may carry ;attr="..." -> keep the name only.
    return [e.split(";")[0].strip() for e in rb.split(",") if e.strip()]

def main(src: Path, dst: Path) -> None:
    jars = sorted(src.rglob("*.jar"))
    if not jars:                                # not a libs dir -> emit nothing
        print(f"no jars under {src} — nothing to do", file=sys.stderr)
        return
    out = dst / "soa-libs-index.md"
    dst.mkdir(parents=True, exist_ok=True)
    sym2file: dict[str, str] = {}
    info: dict[str, dict] = {}
    thirdparty: list[str] = []

    for j in jars:
        mf = manifest(j)
        sym = mf.get("Bundle-SymbolicName", "").split(";")[0].strip()
        rec = {"file": j.name, "sym": sym, "req": req_bundles(mf),
               "spkgs": service_pkgs(j), "title": mf.get("Implementation-Title", "")}
        info[j.name] = rec
        if sym:
            sym2file[sym] = j.name
        else:
            thirdparty.append(j.name)

    # Alias third-party Require-Bundle names (often unversioned, e.g. "httpclient")
    # to their versioned filenames (httpclient-4.5.13.jar) so closures resolve them.
    for fn in thirdparty:
        base = re.sub(r"[-_]\d[\d.]*$", "", fn[:-4])     # strip trailing -version
        sym2file.setdefault(base, fn)
        sym2file.setdefault(fn[:-4], fn)
    # Bundles known to live in the full RAC/server install, not the SOA client kit.
    NOT_IN_KIT = {"com.teamcenter.rac.external", "com.teamcenter.SecurityServices"}

    def closure(fname: str) -> tuple[set[str], set[str]]:
        """Transitive Require-Bundle closure as filenames; + unresolved sym names."""
        seen_files, unresolved, stack = set(), set(), [fname]
        while stack:
            f = stack.pop()
            for sym in info.get(f, {}).get("req", []):
                tgt = sym2file.get(sym)
                if tgt is None:
                    unresolved.add(sym)
                elif tgt not in seen_files:
                    seen_files.add(tgt)
                    stack.append(tgt)
        return seen_files, unresolved

    # strong-service jars = jars that provide at least one com...services.strong.* pkg
    strong = {f: r for f, r in info.items()
              if any((".services.strong." in p) for p in r["spkgs"])}

    # framework baseline = bundles common to EVERY strong service's closure
    closures = {f: closure(f)[0] for f in strong}
    common = set.intersection(*closures.values()) if closures else set()

    # package -> jar (strong)
    pkg2jar: dict[str, str] = {}
    for f, r in strong.items():
        for p in r["spkgs"]:
            if (".services.strong." in p):
                pkg2jar[p] = f

    L = []
    A = L.append
    A("# Teamcenter SOA — Java Strong Client Library Index\n")
    A("Which JARs to copy to build a **trimmed** Teamcenter SOA *strong* Java client. "
      "Generated from the `soa_client` kit `java/libs` manifests (OSGi "
      "`Bundle-SymbolicName` + `Require-Bundle`). The JARs are binary and are never "
      "ingested — this index is their copy/dependency map.\n")
    A("**How to use this**\n")
    A("- *Developing* (not packaging): just put the whole `java/libs` on the classpath "
      "— no curation needed. Trim only when shipping a self-contained client.\n")
    A("- *Trimming*: copy the **Runtime baseline** below, plus the **service jar** for "
      "each service you call and its **extra jars** (its `Require-Bundle` closure minus "
      "the baseline). Then run offline and add anything a `ClassNotFoundException` "
      "names — the OSGi graph covers TC bundles but not reflective/optional loads.\n")
    A(f"- Totals: {len(jars)} jars; {len(strong)} strong-service jars; "
      f"{len(thirdparty)} third-party/runtime jars.\n")
    A("- Note: `com.teamcenter.rac.external` and `com.teamcenter.SecurityServices` "
      "appear in `Require-Bundle` but ship with the full RAC/server install, not this "
      "SOA client kit; a stand-alone SOA client does not need them.\n")

    A("\n## Runtime baseline (copy for ANY strong client)\n")
    A("**Framework bundles** — in every service's dependency closure:\n")
    for f in sorted(common):
        A(f"- `{f}`")
    if not common:
        A("- (none common — see per-service closures)")
    A("\n**Third-party / runtime jars** (no OSGi headers; OSGi `Require-Bundle` never "
      "references these, but a running client needs the transport/binding/logging "
      "stack). Copy the transport + XML/JSON binding + logging ones for any client:\n")
    for f in sorted(thirdparty):
        A(f"- `{f}`")

    A("\n## Services (strong) — jar + extra jars to copy\n")
    A("`extra jars` = the service's transitive `Require-Bundle` closure with the "
      "framework baseline removed (copy these on top of the baseline).\n")
    for f in sorted(strong, key=lambda x: strong[x]["spkgs"][0] if strong[x]["spkgs"] else x):
        r = strong[f]
        cl, unres = closure(f)
        extra = sorted(cl - common - {f})
        doms = ", ".join(p.split(".")[-1] for p in r["spkgs"]
                         if (".services.strong." in p))
        A(f"\n### {f}")
        if r["title"]:
            A(f"_{r['title']}_")
        A(f"- provides: {', '.join('`'+p+'`' for p in r['spkgs'])}")
        A(f"- extra jars: " + (", ".join(f"`{e}`" for e in extra) if extra else "_none beyond baseline_"))
        unres = unres - NOT_IN_KIT
        if unres:
            A(f"- unresolved bundles: {', '.join('`'+u+'`' for u in sorted(unres))}")

    A("\n## Package → jar lookup (strong)\n")
    A("| service package | jar |")
    A("|---|---|")
    for p in sorted(pkg2jar):
        A(f"| `{p}` | `{pkg2jar[p]}` |")

    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {out}  ({len(strong)} strong services, {len(pkg2jar)} packages, "
          f"{len(common)} framework + {len(thirdparty)} third-party baseline jars)")

if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
