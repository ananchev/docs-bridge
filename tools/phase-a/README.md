# Phase A step scripts (Mac mini rebuild)

The runbook (`notes/phase-a-runbook.md`) explains each step; these scripts ARE the
steps. On vhost2, as yourself (each script asks for sudo when it needs it):

```bash
cd ~/phase-a/src/tools/phase-a
./1-build.sh            # ~10 min  build the new images, check models + versions
./2-rerank.sh           # ~8 min   reranker scores on the new runtime
./3-smoke.sh            # ~15 min  small scratch ingest; identity fix + stamps
#   labels: OPTIONAL — drafts are fine for this gate (runbook step 8)
./4-config.sh           # ~2 min   write the rebuild config, check the file counts
./5-start-rebuild.sh    # evening  pause timer, what's-new now, 2512 overnight (detached)
./5b-resume.sh          # only if 5's overnight run died: resume it (quarantines the PDF that OOMed)
./6-check-rebuild.sh    # any time progress while running; the full checks once it is done
./7-gate.sh             # ~25 min  old vs new on the 45 questions -> VERDICT
./8-reparse-pdfs.sh     # after a worker-image parse fix: re-parse only the PDFs (then 6, 7)
```

`./diag-parse.sh "<file>"` parses one document alone under a 10 GB cap and reports
where its time and memory go — run it when no rebuild is running.

Every script ends with **Step passed** (go to the next) or **Step FAILED** (stop and
send me `~/phase-a/out`). Nothing before `5-start-rebuild.sh` changes the running
stack; that one pauses the nightly timer. The switch-over itself is done with me.

Source sync (workstation, repo root, after any change such as the labels):

```bash
rsync -a --delete --exclude .git ./ vhost2:phase-a/src/
```

Results back (workstation, repo root): `rsync -a --exclude '*.tar' vhost2:phase-a/out/ notes/phase-a/`
