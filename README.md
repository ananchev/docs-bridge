# docs-bridge

A multi-subject, self-updating documentation **RAG stack** — retrieval layer +
ingestion + an MCP server. Built and validated on a **Raspberry Pi 5 (Rocky Linux
10, NVMe)**, then redeployed to a **Mac mini M2 (Asahi Fedora)**. See
[`docs-bridge-ansible-design.md`](./docs-bridge-ansible-design.md) for the full
stack design.

## What lives here (the product) vs. where it's deployed (the operator)

This repo is the **product**: the container images, the config contract, the
design, and the one-time host provisioning. **Deployment, secrets, runtime
install, and backup are owned by the separate `containers-at-home` fleet repo** —
docs-bridge is deployed as just-another-app there, using the same module-based
playbook pattern, vault, and backup machinery as every other app.

| Concern | Repo |
|---|---|
| Container images (`ingest-worker`, `docs-bridge`) — Dockerfiles + source | **docs-bridge** |
| Config schema / contract + design docs | **docs-bridge** |
| Pi 5 OS bring-up (flash Rocky to NVMe) — `provisioning/os-swap.md` | **docs-bridge** |
| Inventory, `host_vars`, **Podman runtime** install | containers-at-home |
| Deploy playbook (`applications/docs-bridge.yml`, `docker_container` modules) | containers-at-home |
| Vault secrets, pinned image tags, backup/restore wiring | containers-at-home |

**The boundary = a container image + a documented config contract.** Nothing else
crosses it: containers-at-home never needs the app internals; docs-bridge never
needs the inventory/vault/backup topology.

## Runtime: Podman (via its Docker-compatible socket)

The Pi (Rocky) and the M2 (Asahi Fedora) are both RHEL-family, so they run
**Podman** — but through Podman's **Docker-API socket** (`podman` + `podman-docker`
+ `podman.socket` + `podman-restart.service`). That lets containers-at-home's
existing `community.docker.*` modules drive Podman unchanged, so docs-bridge joins
a Docker fleet without forking the automation. The rest of the fleet stays on
Docker, untouched.

## Status

- **Pi 5 → Rocky Linux 10.2 on NVMe: done & verified** (enforcing SELinux, same
  user/key, `/data` on a 446 GB XFS partition, SD kept as fallback). See
  [`provisioning/os-swap.md`](./provisioning/os-swap.md) for the as-built runbook
  and the gotchas (PARTUUID boot, SELinux relabel, **NVMe power-save/APST fix**).
- **NVMe stability soak in progress** — APST disabled; remote logging to the
  Manjaro log server is set up to catch any recurrence.
- **Next:** build the images here; wire the deploy/runtime/backup in
  containers-at-home.
