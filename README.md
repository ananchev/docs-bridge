# docs-bridge

Ansible-deployed, multi-subject RAG stack. Built and validated on a **Raspberry
Pi 5**, then redeployed unchanged to a **Mac mini M2 (Asahi Fedora)**. See
[`docs-bridge-ansible-design.md`](./docs-bridge-ansible-design.md) for the full
stack design (retrieval layer, ingestion, MCP server, LibreChat front-end).

## Current focus: Pi 5 host setup & provisioning

Everything in this repo right now is about getting the Pi 5 into a clean,
Ansible-manageable state — **before** any docs-bridge containers are deployed.

The Pi 5 is migrated from SD-card Raspberry Pi OS to **Rocky Linux 10 on the NVMe
SSD**, with the SD kept as boot fallback. Two phases:

```
Phase 1  one-time, manual         provisioning/os-swap.md
  RPi OS on SD  ──►  Rocky 10 on NVMe  (reachable, same user + SSH key)

Phase 2  Ansible, idempotent      ansible/
  reachable Rocky host  ──►  Docker host, data-root on /data
```

Phase 1 can't be Ansible (no Rocky host to manage yet). Everything after — host
prep and all docs-bridge deployment — is Ansible (`ansible/playbooks/site.yml`).

**Why Rocky 10:** Pi 5 needs kernel ≥6.6 → rules out Debian 12 and Rocky 9;
Rocky 10 (RHEL 10, k6.12) supports Pi 5 and is dnf/RHEL-family like the M2 target,
so the Pi and M2 share one Ansible path.

```
ansible/
├── inventory/   hosts.yml (pi5 @ 192.168.2.11) + group_vars
├── playbooks/   site.yml → common, container_runtime
└── roles/
    ├── common/            packages, firewall, time, ssh hardening, /data assert
    └── container_runtime/ Docker CE, data-root → /data/docker, SELinux label
```

Run: `cd ansible && ansible-galaxy collection install -r requirements.yml && ansible-playbook playbooks/site.yml`
