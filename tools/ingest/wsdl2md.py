"""WSDL/XSD -> clean Markdown wire-contract preprocessor (standalone; runs OUTSIDE the
product). Sibling of javadoc2md.py / soa-libs-index.py: turns a Teamcenter SOA SOAP
service WSDL (document/literal) into a language-agnostic Markdown description of the
*wire contract* — endpoint, namespaces, operations, soapActions, and request/response
message types resolved from the accompanying XSD. This is the protocol-level companion
to the Javadoc (which gives API semantics): together they let a client be (re)built in
ANY language — generate stubs (gowsdl, wsdl2java, svcutil, zeep, …) or hand-roll SOAP.

Converter contract: given <src_dir> and <dst_dir>, it writes one mirrored `.md` per
`.wsdl` (same relative path, .wsdl -> .md) and emits NOTHING if the source has no WSDL
— so the shared ingest loop can run it over any tagged folder and only a WSDL dir
produces output (mutually exclusive with the javadoc / doxygen / libs converters).

Per WSDL page:
  - `# <Service>` + endpoint + target namespace + prefix table.
  - `## <operation>` with soapAction, request/response element, and faults.
  - `## Types` listing the service XSD's complexTypes (field name : type [occurs]).
Types from shared namespaces (Base/Exceptions) are referenced by QName, not expanded —
they are the SOA data model, already covered by the Javadoc corpus.

    python wsdl2md.py <src_dir> <dst_dir>
"""
from __future__ import annotations
import sys
from pathlib import Path
from lxml import etree

def ln(el) -> str:
    """local name of an element or QName tag."""
    return etree.QName(el).localname

def qn_local(qname: str) -> str:
    """local part of a 'prefix:Local' string."""
    return qname.split(":")[-1] if qname else qname

def load_xsd_types(paths: list[Path]) -> tuple[dict, dict]:
    """Across the given XSD files build localname -> node maps for top-level
    xsd:element and xsd:complexType."""
    elems, types = {}, {}
    for p in paths:
        if not p.exists():
            continue
        try:
            root = etree.parse(str(p)).getroot()
        except etree.XMLSyntaxError:
            continue
        for el in root:
            if not isinstance(el.tag, str):
                continue
            l = ln(el)
            name = el.get("name")
            if not name:
                continue
            if l == "element":
                elems.setdefault(name, el)
            elif l == "complexType":
                types.setdefault(name, el)
    return elems, types

def fields_of(node, types: dict) -> list[tuple[str, str, str]]:
    """(name, type, occurs) for the direct element particles of a complexType (or of an
    xsd:element's inline complexType). Walks sequence/all/choice; one level deep."""
    if node is None:
        return []
    out = []
    for e in node.iter():
        if isinstance(e.tag, str) and ln(e) == "element" and e.get("name"):
            mn, mx = e.get("minOccurs", "1"), e.get("maxOccurs", "1")
            occ = "" if (mn, mx) == ("1", "1") else f" [{mn}..{mx}]"
            typ = e.get("type", "") or "(inline)"
            out.append((e.get("name"), typ, occ))
    return out

def elem_fields(elem_qname: str, elems: dict, types: dict) -> tuple[str, list]:
    """Resolve a message part element QName to its complexType fields."""
    local = qn_local(elem_qname)
    el = elems.get(local)
    if el is None:
        return elem_qname, []
    # inline complexType?
    for c in el:
        if isinstance(c.tag, str) and ln(c) == "complexType":
            return elem_qname, fields_of(c, types)
    # else referenced type
    t = el.get("type")
    if t and qn_local(t) in types:
        return elem_qname, fields_of(types[qn_local(t)], types)
    return elem_qname, []

def convert(wsdl_path: Path) -> str | None:
    try:
        root = etree.parse(str(wsdl_path)).getroot()
    except etree.XMLSyntaxError:
        return None
    if ln(root) != "definitions":
        return None

    nsmap = {k: v for k, v in (root.nsmap or {}).items() if k}
    tns = root.get("targetNamespace", "")

    # messages: name -> part element QName
    messages = {}
    for m in root.iter():
        if isinstance(m.tag, str) and ln(m) == "message":
            part = next((c for c in m if isinstance(c.tag, str) and ln(c) == "part"), None)
            if part is not None:
                messages[m.get("name")] = part.get("element") or part.get("type") or ""

    # portType operations
    ops = []
    for pt in root.iter():
        if isinstance(pt.tag, str) and ln(pt) == "portType":
            for op in pt:
                if not (isinstance(op.tag, str) and ln(op) == "operation"):
                    continue
                rec = {"name": op.get("name"), "in": None, "out": None, "faults": []}
                for c in op:
                    if not isinstance(c.tag, str):
                        continue
                    msg = qn_local(c.get("message", ""))
                    if ln(c) == "input":
                        rec["in"] = msg
                    elif ln(c) == "output":
                        rec["out"] = msg
                    elif ln(c) == "fault":
                        rec["faults"].append(c.get("name") or msg)
                ops.append(rec)
            break

    # binding: operation name -> soapAction
    actions = {}
    for b in root.iter():
        if isinstance(b.tag, str) and ln(b) == "binding":
            for op in b:
                if not (isinstance(op.tag, str) and ln(op) == "operation"):
                    continue
                so = next((c for c in op if isinstance(c.tag, str) and ln(c) == "operation"), None)
                if so is not None:
                    actions[op.get("name")] = so.get("soapAction", "")

    # endpoint
    address = ""
    for a in root.iter():
        if isinstance(a.tag, str) and ln(a) == "address" and a.get("location"):
            address = a.get("location")
            break

    # XSDs referenced from <types> via schemaLocation (+ sibling fallback)
    xsd_locs = set()
    for imp in root.iter():
        if isinstance(imp.tag, str) and ln(imp) == "import" and imp.get("schemaLocation"):
            xsd_locs.add(imp.get("schemaLocation"))
    cand = [wsdl_path.parent / loc for loc in xsd_locs]
    cand.append(wsdl_path.with_suffix(".xsd"))
    sib = wsdl_path.parent / (wsdl_path.stem.replace("Service", "") + ".xsd")
    cand.append(sib)
    elems, types = load_xsd_types(list(dict.fromkeys(cand)))

    # ---- render ----
    svc = wsdl_path.stem
    md = [f"# {svc}", ""]
    md += [f"SOAP service wire contract (document/literal). Target namespace "
           f"`{tns}`." + (f" Endpoint `{address}`." if address else ""), ""]

    # only the SOA-relevant prefixes (teamcenter schemas/services)
    tc_ns = {k: v for k, v in nsmap.items() if "teamcenter.com" in v}
    if tc_ns:
        md += ["**Namespaces**", ""]
        for k in sorted(tc_ns):
            md.append(f"- `{k}` = `{tc_ns[k]}`")
        md.append("")

    def render_msg(label, msgname):
        if not msgname or msgname not in messages:
            return [f"- {label}: _none_"]
        qn, flds = elem_fields(messages[msgname], elems, types)
        lines = [f"- {label}: `{qn}`"]
        for n, t, o in flds:
            lines.append(f"    - `{n}`: `{t}`{o}")
        return lines

    for op in ops:
        md += [f"## {op['name']}", ""]
        if op["name"] in actions:
            md.append(f"- soapAction: `{actions[op['name']]}`")
        md += render_msg("request", op["in"])
        md += render_msg("response", op["out"])
        if op["faults"]:
            md.append("- faults: " + ", ".join(f"`{f}`" for f in op["faults"]))
        md.append("")

    # service-XSD complexTypes (the message structures, expanded one level)
    if types:
        md += ["## Types", ""]
        for name in sorted(types):
            flds = fields_of(types[name], types)
            if not flds:
                continue
            md.append(f"### {name}")
            for n, t, o in flds:
                md.append(f"- `{n}`: `{t}`{o}")
            md.append("")

    out = "\n".join(md).rstrip() + "\n"
    return out if len(out.splitlines()) > 4 else None

def main(src: Path, dst: Path) -> None:
    wsdls = sorted(src.rglob("*.wsdl"))
    if not wsdls:                                   # not a WSDL dir -> emit nothing
        print(f"no .wsdl under {src} — nothing to do", file=sys.stderr)
        return
    kept = skipped = 0
    for p in wsdls:
        try:
            md = convert(p)
        except Exception as e:
            print(f"  ! {p}: {e}", file=sys.stderr)
            md = None
        if not md:
            skipped += 1
            continue
        out = dst / p.relative_to(src).with_suffix(".md")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md, encoding="utf-8")
        kept += 1
    print(f"converted {kept} WSDL(s), skipped {skipped}", file=sys.stderr)

if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
