#!/usr/bin/env bash
# Step 4 (runbook B.1, second half): write the rebuild config from the live one and
# check the new subjects see exactly the old corpus. ~2 min. Changes nothing live.
set -uo pipefail
. "$(dirname "$0")/vars.sh"

say "writing $NEW_CFG"
sudo podman run --rm -i -v $CFG:/in.yaml:ro,z --entrypoint python localhost/ingest-worker:$NEW - \
  > $NEW_CFG <<'EOF'
import yaml
c = yaml.safe_load(open("/in.yaml"))
assert [s for s in c["subjects"] if s["name"] == "teamcenter"], "no teamcenter subject in the live config"
# sow, ecosys, polarion: dropped for now (runbook, top)
c["subjects"] = [
    {"name": "teamcenter-2512", "dir": "/data/docs/teamcenter-2512",
     "collection": "teamcenter_2512",
     "description": (
         "Siemens Teamcenter PLM, release 2512 — installation, configuration, "
         "administration, Active Workspace, BMIDE data modelling, server/client APIs, "
         "SOA & ITK development and customization (incl. the SOA API reference), and "
         "the Active Integration Gateway (T4S/T4EA/T4O) for SAP and enterprise-"
         "application integration (AIG runs on and is configured through Teamcenter, "
         "so AIG install/config topics live here). Use for how Teamcenter 2512 works; "
         "for what changed between releases use teamcenter-whats-new.")},
    {"name": "teamcenter-whats-new", "dir": "/data/docs/teamcenter-whats-new",
     "collection": "teamcenter_whats_new",
     "description": (
         "Siemens Teamcenter release notes — the What's New guide of each release "
         "(Teamcenter 14.3, 2312, 2406, 2412, 2506, 2512, 2606) and the Teamcenter "
         "Gateway (T4S, T4EA) What's New for 2512. Use for which release introduced, "
         "changed or removed a feature, and for upgrade planning; for how a feature "
         "works in detail, search that release's own documentation pool "
         "(e.g. teamcenter-2512).")},
]
c["manifest_path"] = "/data/state/manifest-subject-keyed.sqlite"
c.setdefault("ingest", {})["batch_size"] = 32   # batch 64 peaks at 7.6 GB (runbook B.3)
print(yaml.safe_dump(c, sort_keys=False, allow_unicode=True))
EOF
grep -nE '^  - name:|^    dir:|^    collection:|^manifest_path' $NEW_CFG | tee $OUT/4-config.txt
[ -s $NEW_CFG ] && ok "written" || bad "empty config"

say "what each new subject will ingest vs what the old one did"
counts=$(worker --entrypoint python localhost/ingest-worker:$NEW -c '
from docs_bridge import config
from docs_bridge.parse import scan
cfg = config.load("/config/config.yaml")
print(*[len(scan(s, cfg)) for s in cfg.subjects])')
read -r n2512 nnew <<< "$counts"
nold=$(sudo python3 -c 'import sqlite3
print(sqlite3.connect("file:'$P'/state/manifest.sqlite?mode=ro", uri=True).execute(
  "SELECT count(*) FROM docs WHERE subject = ?", ("teamcenter",)).fetchone()[0])')
echo "old teamcenter: $nold docs | new: 2512 $n2512 + whats-new $nnew + 1 dropped duplicate = $((n2512 + nnew + 1))" \
  | tee -a $OUT/4-config.txt
[ "$nnew" = 9 ] && ok "what's-new: 9 docs" || bad "what's-new should have 9 docs"
[ "$((n2512 + nnew + 1))" = "$nold" ] && ok "same corpus as the old teamcenter" \
  || bad "counts differ from the old manifest — send me $OUT/4-config.txt"
finish "Next: ./5-start-rebuild.sh — in the evening, it runs overnight"
