# docs-bridge

Ansible-deployed, multi-subject RAG stack. Built and validated on a **Raspberry
Pi 5**, then redeployed unchanged to a **Mac mini M2 (Asahi Fedora)**. See
[`docs-bridge-ansible-design.md`](./docs-bridge-ansible-design.md) for the full
stack design (retrieval layer, ingestion, MCP server, LibreChat front-end).

## Current focus: Pi 5 host setup & provisioning

Everything in this repo right now is about getting the Pi 5 into a clean,
Ansible-manageable state — **before** any docs-bridge containers are deployed.

The Pi 5 is being migrated from its current SD-card Raspberry Pi OS install to
**Rocky Linux 10 on the NVMe SSD**, with the SD card retained as an automatic
boot fallback. Rationale, hardware layout, and the full runbook live in
[`provisioning/`](./provisioning/).

| Doc | What it covers |
|---|---|
| [`provisioning/00-decisions.md`](./provisioning/00-decisions.md) | OS choice (why Rocky 10, not 9 / not Debian 12), disk layout, research findings + sources |
| [`provisioning/01-os-swap-runbook.md`](./provisioning/01-os-swap-runbook.md) | Step-by-step bare-metal swap: image → NVMe, repartition, chroot preseed, boot order, first boot |
| [`provisioning/02-host-prep.md`](./provisioning/02-host-prep.md) | Phase-2 Ansible host prep: Docker CE, data-root relocation, SELinux, mounts, firewall |

### Two-phase model

```
Phase 1  (bare metal, one-time, manual runbook)
  Raspberry Pi OS on SD  ──►  Rocky Linux 10 on NVMe (reachable, same user + SSH key)

Phase 2  (Ansible, idempotent, re-runnable)
  reachable Rocky host   ──►  Docker host with data-root on /data, ready for docs-bridge
```

Phase 1 cannot be done by Ansible (Ansible needs a running, reachable host).
Phase 2 onward — and all docs-bridge deployment — is Ansible-driven.
