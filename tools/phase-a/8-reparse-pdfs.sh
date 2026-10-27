#!/usr/bin/env bash
# Step 8: re-parse every PDF with the font-fixed worker image (2026-10-16).
#
# Docling renders each PDF page for its layout model; the first rebuild ran in an image
# with no fonts, so non-embedded Courier code samples rendered as pictures/formulas and
# their text was dropped (~70% of command lines). Markdown/HTML are never rendered and
# keep their chunks. For each subject this resets the PDFs only — manifest rows deleted
# and their Qdrant points purged (tools/ingest/reingest-docs.py, which backs the
# manifest up first) — then a normal sync re-parses exactly those as NEW.
# what's-new (9 PDFs) runs in the foreground; teamcenter-2512 (~110 PDFs) detached.
# Run ./1-build.sh first (it must report the Courier font check OK).
set -uo pipefail
. "$(dirname "$0")/vars.sh"
cd "$SRC"
MAN=$P/state/$NEW_MANIFEST

if sudo podman ps -q --filter name=rebuild- | grep -q .; then
  bad "a rebuild is running — wait for it (./6-check-rebuild.sh)"; finish
fi
systemctl is-active -q docs-bridge-ingest.timer && { bad "the nightly timer is active again — stop it first"; finish; }

say "the worker image must have the fonts"
mono=$(sudo podman run --rm --entrypoint fc-match localhost/ingest-worker:$NEW Courier 2>&1)
echo "   fc-match Courier -> $mono"
grep -qi nimbus <<< "$mono" && ok "fonts present" || { bad "image has no URW fonts — run ./1-build.sh"; finish; }

say "resetting the PDFs (manifest rows + Qdrant points)"
for pair in teamcenter-whats-new:teamcenter_whats_new teamcenter-2512:teamcenter_2512; do
  s=${pair%%:*}; c=${pair##*:}
  sudo python3 -c 'import sqlite3, sys
db = sqlite3.connect(f"file:{sys.argv[1]}?mode=ro", uri=True)
for (d,) in db.execute("SELECT doc_id FROM docs WHERE subject = ? AND lower(doc_id) LIKE ?", (sys.argv[2], "%.pdf")):
    print(d)' $MAN $s > $OUT/8-pdfs-$s.txt
  n=$(wc -l < $OUT/8-pdfs-$s.txt)
  echo "   $s: $n PDFs ($c)"
  sudo python3 tools/ingest/reingest-docs.py --subject $s --docs-from $OUT/8-pdfs-$s.txt \
    --manifest $MAN --purge-points --collection $c --apply > $OUT/8-reset-$s.txt 2>&1 \
    && ok "$(grep -E '^>> deleted' $OUT/8-reset-$s.txt)" || { bad "reset failed — see $OUT/8-reset-$s.txt"; finish; }
done

# With the live server stopped there is room for a 10 GB cap (big PDF parses).
if ! sudo podman ps --format '{{.Names}}' | grep -qx docs-bridge; then
  WORKER_MEM=10g; WORKER_LIMITS="--memory $WORKER_MEM --memory-swap $WORKER_MEM"
fi

say "teamcenter-whats-new (foreground, ~15 min)"
worker localhost/ingest-worker:$NEW sync --subject teamcenter-whats-new > $OUT/8-whats-new.txt 2>&1 \
  && ok "$(grep '^\[teamcenter-whats-new\]' $OUT/8-whats-new.txt | tail -1)" || { bad "see $OUT/8-whats-new.txt"; finish; }

say "teamcenter-2512 PDFs (detached, capped at $WORKER_MEM)"
start_rebuild teamcenter-2512 && ok "running as container rebuild-teamcenter-2512" || { bad "did not start"; finish; }
ram_logger rebuild-teamcenter-2512 $OUT/8-ram.txt
finish "Progress any time: ./6-check-rebuild.sh — when it says done it runs the checks; then ./7-gate.sh"
