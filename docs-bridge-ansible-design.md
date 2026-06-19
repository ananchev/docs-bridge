# docs-bridge — RAG Stack (Design Sketch)

> Cold-start spec for a separate build session. A multi-subject, self-updating
> documentation RAG stack. Validate on a Raspberry Pi 5 (8GB, 500GB SSD) with a
> <500MB PDF/HTML corpus, then redeploy to a Mac mini M2 (16GB, Asahi Fedora)
> and scale the corpus.

> ⚠️ **Updated — deployment boundary (supersedes the original "self-deploying"
> premise).** docs-bridge is **not** self-deploying. This repo is the *product*
> (container images + config contract + design + the Pi OS provisioning). It is
> **deployed by the existing `containers-at-home` fleet repo** as just-another-app,
> reusing that repo's inventory, vault, module-based deploy pattern, runtime
> install, and backup/restore. Runtime is **Podman** on the RHEL-family hosts (Pi
> Rocky 10, M2 Asahi Fedora) via Podman's **Docker-compatible socket**, so
> `containers-at-home`'s `community.docker.*` modules drive it unchanged. The rest
> of this doc still describes the stack's architecture; read §3–§4 with this
> boundary in mind (deploy mechanism lives in `containers-at-home`, not here).

---

## 1. Goals & constraints

- **Generator is remote** (OpenRouter, ZDR + EU in-region). The stack hosts only
  the **retrieval layer** + ingestion + an MCP server. No local LLM.
- **Full-fidelity stack on the Pi** — real BGE-M3, Docling, BGE-reranker. No
  stand-in models. Whatever runs on the Pi is what ships to the M2.
- **Ansible deploys the containers**, not just the host. Re-running the playbook
  is the deploy/update mechanism on either host.
- **Portable Pi → M2 with config only.** No code or image changes.
- **Multi-subject:** N independent corpora ("subjects"), each its own source dir
  and Qdrant collection, shared models in memory.
- **Self-updating:** incremental, hash-delta ingestion (new / changed / deleted),
  triggered on a schedule.

### The free win: both hosts are arm64
Pi 5 and M2-under-Asahi are both **aarch64 Linux**. Images build once, run on
both — no multi-arch, no cross-build. The only host-level difference is the
package manager (apt vs dnf), which Ansible normalizes.

---

## 2. Topology

```
inventory
├── pi5        (Pi OS, aarch64)      — prototype / validation
└── macmini    (Fedora Asahi, aarch64) — target / scale

each host runs the same compose stack:
  qdrant         — vector store (volume-backed)
  ingest-worker  — two-pass ingestion job (Docling -> manifest -> BGE-M3 -> Qdrant)
  docs-bridge    — FastMCP server: retrieve -> rerank -> (optional) OpenRouter synth
behind: Nginx Proxy Manager -> Cloudflare, bearer-token auth (existing fleet pattern)
```

---

## 3. Container runtime — DECIDED: Podman via Docker-compatible socket

Resolved (supersedes the earlier "Docker+compose vs Podman+Quadlet" options).
The fleet's `containers-at-home` repo deploys every app with **`community.docker.*`
modules** (imperative `docker_container`/`docker_network`, ~55 uses) — **not**
compose, **not** Quadlet. docs-bridge follows that same pattern.

On the RHEL-family hosts (Pi Rocky 10, M2 Asahi Fedora) the runtime is **Podman**,
exposed through its **Docker-API socket**:

- `podman` + `podman-docker` + `systemctl enable --now podman.socket podman-restart.service`
- `podman-docker` symlinks `/run/docker.sock` → `podman.sock`, so `community.docker.*`
  connect unchanged; `podman-restart.service` reproduces `restart_policy: unless-stopped`
  on boot (Podman is daemonless); `:Z`/`:z` SELinux labels are native to Podman.

Net: docs-bridge joins the Docker-based fleet without forking the automation, and
the rest of the fleet stays on Docker untouched. The one rough edge to smoke-test
is **image build over Podman's compat API** (`docker_image_build`) — fallback is a
`podman build` shell task or pre-built/pinned images pulled from a registry
(preferred for Pi→M2 portability).

---

## 4. Repo layout (split across two repos)

**This repo — the product:**
```
docs-bridge/
├── images/
│   ├── ingest-worker/   # Dockerfile + two-pass pipeline (Python)
│   └── docs-bridge/     # Dockerfile + FastMCP server (Python)
├── config.schema.md     # documented config keys (the contract)
├── provisioning/
│   └── os-swap.md        # one-time Pi 5 OS bring-up (Rocky → NVMe)
├── docs-bridge-ansible-design.md
└── README.md
```

**`containers-at-home` — the operator** (existing fleet repo; docs-bridge is added
as just-another-app following its conventions):
```
containers-at-home/ansible/
├── inventories/pi5/hosts + host_vars/pi5     # paths (/data/docker), podman flag
├── inventories/group_vars/all/
│   ├── vault.yml                              # SECRETS (OpenRouter key, bearer token)
│   └── app_versions.yml                       # pinned image tags (qdrant, docs-bridge)
├── roles or task: podman runtime              # podman + podman-docker + sockets
└── applications/docs-bridge.yml               # THE deploy playbook:
      community.docker.docker_network / docker_container, config.yaml.j2 from
      group_vars+vault, :Z volume mounts under /data. Run via run-playbook.sh
      (inherits --ask-vault-pass + group_vars symlink).
```

The earlier `roles/docs_bridge` (compose/Quadlet, self-deploying) is replaced by
the single `applications/docs-bridge.yml` in `containers-at-home`. Ingestion
triggers (`ingest`) and Qdrant snapshot/restore become tasks there too (or a
`systemd timer`, §8).

---

## 5. How Ansible brings the containers up

`roles/docs_bridge/tasks/main.yml` (sketch, Docker path):

```yaml
- name: Render config + env + compose from host vars
  ansible.builtin.template:
    src: "{{ item.src }}"
    dest: "{{ deploy_dir }}/{{ item.dest }}"
  loop:
    - { src: config.yaml.j2,         dest: config.yaml }
    - { src: env.j2,                 dest: .env }
    - { src: docker-compose.yml.j2,  dest: docker-compose.yml }
  notify: recreate stack

- name: Build local images (arm64)
  community.docker.docker_image:
    name: "{{ item }}"
    build: { path: "{{ playbook_dir }}/../images/{{ item }}" }
    source: build
    state: present
  loop: [ingest-worker, docs-bridge]

- name: Pull pinned third-party images
  community.docker.docker_image:
    name: "qdrant/qdrant:{{ qdrant_tag }}"
    source: pull

- name: Deploy stack
  community.docker.docker_compose_v2:
    project_src: "{{ deploy_dir }}"
    state: present          # == compose up -d, idempotent
    # 'recreate stack' handler runs `state: present` with recreate on config diff
```

Re-running `site.yml` = the deploy/update. Config changes trigger the handler;
only affected services are recreated.

---

## 6. Config & portability model

Everything host-variable lives in `group_vars`; the role templates it. This is
the entire Pi→M2 portability mechanism.

`group_vars/all.yml` (shared):
```yaml
embedding_model: BAAI/bge-m3
reranker_model:  BAAI/bge-reranker-v2-m3
chunk: { target_tokens: 400, overlap: 60, strategy: structure_aware }
qdrant_tag: "v1.x.x"            # PIN
ports: { mcp: 8080, qdrant: 6333 }
subjects:
  - { name: teamcenter-aig, dir: /data/docs/aig, collection: aig }
  - { name: power-query,    dir: /data/docs/pq,  collection: pq }
openrouter:
  base_url: https://openrouter.ai/api/v1
  model:    qwen/qwen3.7-plus     # ZDR + EU provider enforced in code/headers
  require_zdr: true
```

`group_vars/pi5.yml`:
```yaml
container_runtime: docker
ingest: { batch_size: 8, two_pass: true }   # keep peak RAM < 8GB
qdrant: { on_disk_vectors: false, quantization: none }  # tiny corpus
mem_limits: { ingest_worker: 4g, docs_bridge: 3g }
```

`group_vars/macmini.yml`:
```yaml
container_runtime: docker        # or podman, if you switch to Quadlet
ingest: { batch_size: 64, two_pass: true }
qdrant: { on_disk_vectors: true, quantization: scalar }  # available when corpus grows
```

Same role + different vars → host-appropriate behavior. Adding a subject = one
list entry + drop files + re-run `ingest.yml`.

---

## 7. Secrets

- `group_vars/all/vault.yml` (ansible-vault encrypted): **OpenRouter API key**,
  **bearer token** for the MCP server.
- Templated into `.env` (mode 0600) and injected as container env. Never in the
  compose file or git plaintext.

---

## 8. Ingestion design (the `ingest-worker` image)

**Two-pass** (keeps BGE-M3 and Docling from being resident together → fits 8GB):

1. **Parse pass** — Docling reads PDFs + HTML → structure-aware chunks +
   metadata (`doc_id`, `source_path`, `subject`, `section_path`, `content_hash`,
   `last_updated`). Write chunks + a per-doc manifest row to **SQLite**.
2. **Embed pass** — load BGE-M3, embed new/changed chunks, upsert to the
   subject's Qdrant collection. Delete chunks for removed/changed docs.

**Hash-delta change detection** (the self-updating core):
- new file (hash unseen) → parse, embed, upsert
- changed file (hash differs) → delete old chunks, re-process
- deleted file (manifest has it, disk doesn't) → delete its chunks
- unchanged → skip (no embed cost)

**Idempotent upserts:** chunk id = `{doc_id}:{chunk_index}`; manifest tracks the
chunk-id set per doc so shrinking docs clean up correctly.

**Trigger:** Ansible deploys a **systemd timer** (or cron) on the host that runs
`docker compose run --rm ingest-worker sync --subject <all>` nightly; plus an
optional inotify watch for near-real-time. (Ad-hoc: `playbooks/ingest.yml`.)

---

## 9. docs-bridge MCP surface (the `docs-bridge` image)

FastMCP / FastAPI, bearer-auth, behind NPM/Cloudflare. Shared BGE-M3 +
BGE-reranker loaded once, used across all subject collections.

Tools (sketch):
- `list_subjects()` → available corpora
- `search(subject, query, k=6)` → hybrid retrieve (dense+sparse) → rerank →
  return chunks with citations (source_path, section, last_updated). **Retrieval
  only — no LLM.** Lets Claude or any client synthesize.
- `answer(subject, query)` *(optional)* → search + OpenRouter synthesis (ZDR/EU
  model from config) → grounded answer with citations + an explicit
  "not in the documentation" abstention path.

Decide in build session whether `answer()` lives here or whether the bridge is
pure retrieval and synthesis happens client-side.

---

## 10. Migration Pi → M2

1. Add `macmini` to `inventory/hosts.yml`, create `group_vars/macmini.yml`.
2. `ansible-playbook playbooks/site.yml -l macmini`
   → installs runtime (dnf branch), templates config, brings the stack up.
3. Get the data, either:
   - **Re-ingest** (M2 is much faster): `ansible-playbook playbooks/ingest.yml -l macmini`, or
   - **Restore snapshot**: `playbooks/snapshot.yml` — snapshot on pi5, copy, restore on macmini.
4. Bump corpus in `macmini.yml` (more subjects, larger batches, enable
   quantization/on-disk if needed).

No image rebuild (same arch), no code change. Inventory + group_vars only.

---

## 11. Pi validation checklist (what "validated" means before moving)

- [ ] `site.yml` runs clean on pi5 from bare OS → stack healthy
- [ ] Two-pass ingest completes on the <500MB corpus within RAM budget (watch
      peak RSS; confirm Docling + BGE-M3 never co-resident)
- [ ] Multi-subject: ≥2 collections, queries isolated per subject
- [ ] Hash-delta: add / edit / delete a PDF → next sync reflects all three;
      deleted doc no longer retrievable
- [ ] Re-running `site.yml` is idempotent (no spurious recreates)
- [ ] MCP reachable through NPM/Cloudflare with bearer auth; rejects without it
- [ ] `search()` returns correctly-cited chunks; `answer()` abstains when the
      answer isn't in the corpus
- [ ] Snapshot → restore round-trips a collection
- [ ] Treat latency as NON-representative — Pi proves correctness, not perf

---

## 12. Open decisions for the build session

1. **Runtime:** Docker+compose (portable default) vs Podman+Quadlet (Fedora-native on M2).
2. **Ingest trigger:** systemd timer vs cron vs long-running scheduler container.
3. **MCP scope:** pure retrieval vs retrieval+synthesis (`answer()`).
4. **Hybrid search:** BGE-M3 dense+sparse in one, or dense + separate BM25 in Qdrant.
5. **Reranker on Pi:** keep BGE-reranker-v2-m3 (slow but full-fidelity) vs smaller
   cross-encoder for the prototype only (note: violates "full stack" goal — prefer keeping it).
6. **OpenRouter provider pinning:** how to enforce ZDR + EU at request time
   (provider allow-list / routing params) in the bridge.

---

## 13. Collections / dependencies

`ansible/requirements.yml`:
```yaml
collections:
  - name: community.docker        # docker_compose_v2, docker_image
  # - name: containers.podman     # if Quadlet path chosen
```

Image bases: arm64 Python slim for both custom images; pin Docling, FastMCP,
fastembed/transformers, qdrant-client, and the model revisions.

---

## 14. LibreChat — self-hosted web chat (M2 target)

The MCP client for human users. LibreChat is *both* the MCP client (calls
docs-bridge tools) and the OpenRouter front-end, so the retrieved in-house
content is synthesised on **your** ZDR/EU model, not Anthropic's. This is the
controlled data path; consuming docs-bridge from claude.ai instead would route
synthesis through Anthropic.

> This resolves the §9 / §12-item-3 fork: with an MCP-client chat, expose
> **`search()`** (retrieval only) and let LibreChat's OpenRouter model do the
> synthesis. Keep `answer()` only if a thin/non-agentic client also needs it.

**Footprint:** heaviest service in the stack — pulls **MongoDB + Meilisearch**.
Too heavy to run comfortably on the Pi's 8GB alongside Qdrant + BGE-M3 +
reranker. **Deploy it on the M2 only.** Gate it behind a `deploy_chat` var so the
role conditionally includes these services (pi5: `false`, macmini: `true`).
Do **not** enable LibreChat's own RAG API — docs-bridge is the retrieval layer;
the duplicate just wastes resources.

### Compose additions (templated, conditional on `deploy_chat`)
```yaml
librechat:
  image: ghcr.io/danny-avila/librechat:{{ librechat_tag }}   # PIN
  env_file: [.env]
  volumes:
    - ./librechat.yaml:/app/librechat.yaml:ro
  depends_on:
    mongodb:      { condition: service_started }
    meilisearch:  { condition: service_started }
    docs-bridge:  { condition: service_healthy }   # wait so MCP loads cleanly
  ports: ["{{ ports.chat }}:3080"]

mongodb:
  image: mongo:{{ mongo_tag }}
  volumes: [mongo-data:/data/db]

meilisearch:
  image: getmeili/meilisearch:{{ meili_tag }}
  environment: { MEILI_MASTER_KEY: "${MEILI_MASTER_KEY}" }
  volumes: [meili-data:/meili_data]
```

### `librechat.yaml` (templated sketch — verify keys against current schema)
```yaml
version: 1.3.x
cache: true

# --- model provider: OpenRouter (your ZDR/EU model) ---
endpoints:
  custom:
    - name: "OpenRouter"
      # NOTE: variable MUST be OPENROUTER_KEY. OPENROUTER_API_KEY hijacks the
      # built-in OpenAI endpoint — not what you want.
      apiKey: "${OPENROUTER_KEY}"
      baseURL: "https://openrouter.ai/api/v1"
      models:
        default: ["{{ openrouter.model }}"]   # e.g. qwen/qwen3.7-plus
        fetch: false
      titleConvo: true
      titleModel: "{{ openrouter.model }}"
      dropParams: ["stop"]

# --- tool source: docs-bridge over remote HTTP MCP ---
mcpServers:
  docs-bridge:
    type: streamable-http          # match the server transport (or 'sse')
    url: "{{ docs_bridge_url }}"    # behind NPM, e.g. https://docs.example.com/mcp
    headers:
      Authorization: "Bearer ${DOCS_BRIDGE_TOKEN}"
```

ZDR + EU enforcement (the §12-item-6 decision) is best set at the **OpenRouter
account/key level**, optionally reinforced with provider-routing params. Don't
rely on the UI to enforce it.

### Secrets to add to `vault.yml` → `.env`
- `OPENROUTER_KEY` — model provider
- `DOCS_BRIDGE_TOKEN` — bearer for the MCP call
- `MEILI_MASTER_KEY`, Mongo creds
- LibreChat's own required secrets: `CREDS_KEY`, `CREDS_IV`,
  `JWT_SECRET`, `JWT_REFRESH_SECRET`

### `group_vars` additions
```yaml
# all.yml
ports: { mcp: 8080, qdrant: 6333, chat: 3080 }
librechat_tag: "vX.Y.Z"   # PIN all three
mongo_tag: "7.x"
meili_tag: "vX.Y"
docs_bridge_url: "https://docs.example.com/mcp"

# pi5.yml
deploy_chat: false
# macmini.yml
deploy_chat: true
```

Bonus: because LibreChat is a general MCP client, the same instance can mount
your whole fleet — docs-bridge, cycling-coach, healthbridge — into one
self-hosted chat surface. Add more entries under `mcpServers`.

---

## 15. Pi validation via curl (headless)

The Pi has **no chat UI** (LibreChat lives on the M2). Validate docs-bridge
retrieval directly over HTTP. Two surfaces:

- **Recommended:** add a thin, bearer-authed **REST mirror** to docs-bridge for
  ops/validation — `GET /healthz`, `GET /v1/subjects`, `POST /v1/search` — that
  wraps the same retrieve→rerank path as the `search()` MCP tool. curl-friendly,
  and useful for monitoring later. Keep it debug/ops-gated.
- **Fallback:** curl the raw MCP JSON-RPC endpoint (initialize → tools/list →
  tools/call). Works but fiddly (session handshake, SSE framing) — see note below.

Validate **localhost first** (isolates the app), then repeat through the public
NPM/Cloudflare URL (validates the auth + proxy path).

### Smoke tests (map to the §11 checklist)
```bash
TOKEN="$DOCS_BRIDGE_TOKEN"
BASE="http://localhost:8080"      # then re-run with BASE="https://docs.example.com"

# [liveness]
curl -fsS "$BASE/healthz"

# [auth] must reject without a token  -> expect 401
curl -s -o /dev/null -w '%{http_code}\n' "$BASE/v1/search"

# [subjects] both collections present
curl -fsS "$BASE/v1/subjects" -H "Authorization: Bearer $TOKEN" | jq

# [retrieval + citations] known fact returns cited chunks from the right source
curl -fsS "$BASE/v1/search" \
  -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d '{"subject":"aig","query":"<a fact you know is in the docs>","k":5}' \
  | jq '.results[] | {score, source_path, section, snippet}'

# [subject isolation] same query, different subject -> different/empty sources
curl -fsS "$BASE/v1/search" \
  -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d '{"subject":"power-query","query":"<the AIG-specific fact>","k":5}' | jq '.results | length'

# [abstention] absent topic -> empty or low-score results
curl -fsS "$BASE/v1/search" \
  -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d '{"subject":"aig","query":"<topic definitely not in the corpus>","k":5}' \
  | jq '[.results[].score] | max'
```

### Hash-delta validation (the self-updating core)
```bash
# baseline
curl -fsS "$BASE/v1/search" -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"subject":"aig","query":"<fact in fileX>","k":3}' | jq '.results | length'

# 1) ADD a new PDF to the subject dir, then run a sync
docker compose run --rm ingest-worker sync --subject aig
#    -> re-query a fact from the new file; expect hits

# 2) EDIT fileX (change a fact), sync again
docker compose run --rm ingest-worker sync --subject aig
#    -> re-query; expect the OLD chunk gone, NEW content returned

# 3) DELETE fileX, sync again
docker compose run --rm ingest-worker sync --subject aig
#    -> re-query <fact in fileX>; expect zero hits (chunks purged)
```

### Idempotency check
```bash
# second sync with no source changes should be a no-op (no re-embed)
docker compose run --rm ingest-worker sync --subject aig --verbose
#    -> logs should report 0 new / 0 changed / 0 deleted
```

### Raw MCP fallback (if you skip the REST mirror)
MCP over HTTP is JSON-RPC with a handshake. Shape only:
```bash
# 1) initialize -> capture the session id from the response headers
# 2) tools/list  -> confirm 'search' is advertised
# 3) tools/call  -> {"name":"search","arguments":{"subject":"aig","query":"...","k":5}}
```
Because of the session + SSE framing, a small Python client (mcp SDK) is easier
than curl here — but the REST mirror above avoids the whole problem for smoke
tests. Reserve the raw path for confirming the actual MCP surface works before
the M2/LibreChat brings it up for real.

### Definition of done on the Pi
All §11 boxes green via curl, hash-delta add/edit/delete all reflected, idempotent
re-sync, auth enforced both at localhost and through NPM. Then promote to the M2
(§10) and bring up LibreChat (§14). Remember: Pi validates **correctness**, not latency.
