#!/usr/bin/env bash
# Parse one document ALONE in the real worker image, memory-capped, and report time,
# memory (RSS every 15 s), docling's label counts, and the chunk-level symptoms of the
# 2026-10-15 regression: formula placeholders, and whether `taxonomy -f=` survived
# (default doc: Teamcenter Administration.pdf). Writes nothing but $OUT/diag-*.txt.
#   ./diag-parse.sh "<doc file name under teamcenter-2512>"
# Do not run it while a rebuild is running (they would compete for RAM).
set -uo pipefail
. "$(dirname "$0")/vars.sh"
DOC=${1:-"Teamcenter Administration.pdf"}
if sudo podman ps -q --filter name=rebuild- | grep -q .; then
  bad "a rebuild is running — run this after it finishes"; finish
fi
LOG=$OUT/diag-$(echo "$DOC" | tr -c 'A-Za-z0-9.\n' '_' | cut -c1-60).txt
say "parsing alone, capped at 10g: $DOC"
sudo podman run --rm --network docs-bridge-net --memory 10g --memory-swap 10g \
  -v $P/docs:/data/docs:ro,z -v $P/cache:/data/cache:z -v $NEW_CFG:/config/config.yaml:ro,z \
  --entrypoint python localhost/ingest-worker:$NEW - "/data/docs/teamcenter-2512/$DOC" <<'PY' 2>&1 | tee $LOG
import resource, sys, threading, time
from pathlib import Path
from docs_bridge import config
from docs_bridge.parse import Parser

t0 = time.time(); phase = ["load"]
def rss():
    for line in open("/proc/self/status"):
        if line.startswith("VmRSS"): return int(line.split()[1]) // 1024
def sampler():
    while True:
        print(f"[{time.time()-t0:7.0f}s] {phase[0]:9} rss {rss()} MB", flush=True); time.sleep(15)
threading.Thread(target=sampler, daemon=True).start()

cfg = config.load("/config/config.yaml"); p = Parser(cfg)
phase[0] = "convert"; t1 = time.time()
doc = p._convert(sys.argv[1]).document
print(f"CONVERT done in {time.time()-t1:.0f}s: {len(doc.pages)} pages, {len(doc.tables)} tables, "
      f"{len(doc.pictures)} pictures, {len(doc.texts)} text items; rss {rss()} MB", flush=True)
big = sorted(((t.data.num_rows * t.data.num_cols, t.data.num_rows, t.data.num_cols,
               t.prov[0].page_no if t.prov else None) for t in doc.tables), reverse=True)[:5]
print("largest tables (cells, rows, cols, page):", big, flush=True)
from collections import Counter
labels = Counter(str(x.label) for x in doc.texts)
print("labels:", dict(labels.most_common(10)), flush=True)
phase[0] = "chunk"; t2 = time.time()
chunks = [ch.text for ch in p._chunker.chunk(doc)]
print(f"CHUNK done in {time.time()-t2:.0f}s: {len(chunks)} chunks; "
      f"formula={labels['formula']} code={labels['code']} "
      f"placeholders={sum(x.count('formula-not-decoded') for x in chunks)} "
      f"taxonomy_cmd={'yes' if any('taxonomy -f=' in x for x in chunks) else 'NO'}; "
      f"peak rss {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024} MB", flush=True)
PY
code=${PIPESTATUS[0]}
[ "$code" = 0 ] && ok "parsed within 10g" || bad "exit $code (137 = hit the 10g cap) — the phase it was in is the last line above"
finish "Send me $LOG"
