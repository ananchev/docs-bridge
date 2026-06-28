#!/usr/bin/env bash
# resume-embed.sh — drain already-staged chunks through embed -> Qdrant (Pass 2 only).
#
# WHEN TO USE: a `sync` run was killed/crashed AFTER parse (Pass 1) but DURING or
# BEFORE embed (Pass 2). Because Pass 1 records each doc in the manifest
# provisionally (sync.py upsert_doc, before embed), a normal re-`sync` classifies
# every doc as "unchanged" -> SyncStats.is_noop -> it returns WITHOUT draining the
# staged chunks. So the staged set never gets embedded by a plain re-run. This
# script runs Pass 2 only: it embeds whatever is in `staged_chunks` and upserts to
# Qdrant. No re-parse. See the resume-gap note in tools/ingest/README.md.
#
# IDEMPOTENT + RE-RUNNABLE: point ids are deterministic (UUIDv5 of chunk_id), so
# any partial points already in Qdrant are overwritten, not duplicated. `clear_staged`
# only runs after a subject fully drains, so if this is killed again, just re-run it.
#
# RESILIENT: each batch's upsert is retried with reconnect + capped backoff, so a
# Qdrant restart/disconnect (the failure that triggered the original crash —
# "Server disconnected without sending a response") pauses-and-retries instead of
# aborting the whole multi-hour drain.
#
# SELF-VERIFYING: prints `staged=N` per subject first. N>0 -> drains (and prints a
# running s/chunk + ETA). N=0 for everything -> staging was wiped; fall back to a
# full re-parse: delete the doc rows so they re-classify as "new", then run sync:
#     sqlite3 /data/docs-bridge-payload/state/manifest.sqlite "DELETE FROM docs;"
#     ./run-ingest.sh teamcenter
#
# Built to run inside screen/tmux (rootful podman -> sudo prompts once):
#     screen -S resume
#     ./resume-embed.sh
#     # detach: Ctrl-a d   reattach: screen -r resume
# Logs to ~/ingest-logs/resume_<startISO>.log AND the console (tee).
set -uo pipefail

DATA_ROOT="/data/docs-bridge-payload"
CONFIG="/data/docker/docs-bridge/config/config.yaml"
NET="docs-bridge-net"
IMAGE="ingest-worker:latest"          # the worker image still carries this tag

start_iso="$(date +%Y%m%dT%H%M%S%z)"  # ISO-8601 basic, no colons (fs-safe)
logdir="$HOME/ingest-logs"; mkdir -p "$logdir"
log="$logdir/resume_${start_iso}.log"

{
  echo "docs-bridge resume-embed (Pass 2 only)"
  echo "started : $(date --iso-8601=seconds)"
  echo "log     : ${log}"
  echo "--------------------------------------------------"
} | tee "$log"

start=$(date +%s)
sudo docker run --rm \
  --network "$NET" \
  -v "${DATA_ROOT}/state:/data/state:z" \
  -v "${DATA_ROOT}/cache:/data/cache:z" \
  -v "${CONFIG}:/config/config.yaml:ro,z" \
  --entrypoint python "$IMAGE" -c '
import logging, time
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
from docs_bridge import config, qdrant_io
from docs_bridge.manifest import Manifest
from docs_bridge.embed import get_embedder

cfg = config.load("/config/config.yaml")
client = qdrant_io.connect(cfg)


def upsert_resilient(client, collection, batch, dense, sparse):
    """Upsert one batch; on disconnect, reconnect and retry with capped backoff.
    Returns the (possibly fresh) client to keep using."""
    delay = 5
    for attempt in range(1, 13):
        try:
            qdrant_io.upsert(client, collection, batch, dense, sparse)
            return client
        except Exception as e:  # ResponseHandlingException et al.
            print(f"    upsert failed (attempt {attempt}): {e!r} -- reconnecting in {delay}s",
                  flush=True)
            time.sleep(delay)
            try:
                client = qdrant_io.connect(cfg)
            except Exception as ce:
                print(f"    reconnect failed: {ce!r}", flush=True)
            delay = min(delay * 2, 120)
    raise RuntimeError("upsert still failing after retries -- Qdrant down? aborting (safe to re-run)")


for subj in cfg.subjects:
    with Manifest(cfg.manifest_path) as m:
        n = m.count_staged(subj.name)
    print(f"[{subj.name}] staged={n}", flush=True)
    if not n:
        continue
    qdrant_io.ensure_collection(client, cfg, subj.collection)
    emb = get_embedder(cfg)  # same backend as sync (onnx/int8 by default)
    done = 0
    t0 = time.time()
    with Manifest(cfg.manifest_path) as m:
        for batch in m.iter_staged_batches(subj.name, cfg.ingest.batch_size):
            dense, sparse = emb.encode([c.text for c in batch])
            client = upsert_resilient(client, subj.collection, batch, dense, sparse)
            done += len(batch)
            dt = time.time() - t0
            rate = dt / done
            eta = rate * (n - done)
            print(f"[{subj.name}] {done}/{n}  {rate:.3f}s/chunk  "
                  f"elapsed={dt/60:.1f}m  eta={eta/60:.1f}m", flush=True)
        m.clear_staged(subj.name)
    print(f"[{subj.name}] DONE {done} chunks in {(time.time()-t0)/60:.1f}m", flush=True)
    cnt = client.count(subj.collection, exact=True).count
    print(f"[{subj.name}] collection points_count now = {cnt}", flush=True)
' 2>&1 | tee -a "$log"
status=${PIPESTATUS[0]}
end=$(date +%s); el=$((end - start))

{
  echo "--------------------------------------------------"
  echo "finished: $(date --iso-8601=seconds)"
  echo "exit    : ${status}"
  printf 'elapsed : %dh%02dm%02ds (%ds)\n' $((el/3600)) $(((el%3600)/60)) $((el%60)) "$el"
} | tee -a "$log"
exit "$status"
