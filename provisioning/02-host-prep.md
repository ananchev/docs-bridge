# Pi 5 Host Prep — Phase 2 (Ansible)

Runs once the Pi is reachable on Rocky 10 (end of `01-os-swap-runbook.md`). This
is **idempotent and re-runnable** — it maps onto the design doc's `roles/common`
and `roles/container_runtime` (`docs-bridge-ansible-design.md` §4). Scope here is
**host readiness only**: Docker CE, data-root on `/data`, SELinux, mounts,
firewall. The docs-bridge app stack (`roles/docs_bridge`) comes later.

Because both the Pi (Rocky) and the M2 (Fedora Asahi) are now **RHEL-family /
dnf**, these tasks share one code path — no apt/dnf branching for the Pi.

## Inventory entry

`ansible/inventory/hosts.yml` — connection uses the preserved user + key:

```yaml
all:
  children:
    pi5:
      hosts:
        pi5:
          ansible_host: <pi-ip-or-hostname>
          ansible_user: <$USER from runbook>
          ansible_become: true
```

`ansible/inventory/group_vars/pi5.yml` (host-prep slice; merges with the design
doc's pi5 vars):

```yaml
container_runtime: docker
data_mount: /data
docker_data_root: /data/docker
docker_compose_plugin: true       # docker compose v2 plugin (for community.docker)
firewall_allowed_tcp: [22]        # add 80/443 etc. when NPM lands
```

## Role: common (host baseline)

What it must do on Rocky:

1. **Confirm `/data` is mounted** (fstab entry was added in the runbook). Fail
   fast if not — everything stateful depends on it.
2. **Base packages:** `dnf` install `vim`, `git`, `policycoreutils-python-utils`
   (provides `semanage`, needed for SELinux fcontext), `firewalld`, `chrony`.
3. **User/SSH hygiene:** ensure `$USER` in `wheel`; harden `sshd`
   (`PasswordAuthentication no` once key login is confirmed, `PermitRootLogin no`).
4. **Firewall:** enable `firewalld`, allow `firewall_allowed_tcp`.
5. **Time sync:** `chronyd` enabled.

Sketch:

```yaml
- name: Ensure /data is mounted
  ansible.builtin.command: findmnt -no TARGET {{ data_mount }}
  changed_when: false

- name: Base packages
  ansible.builtin.dnf:
    name: [vim, git, policycoreutils-python-utils, firewalld, chrony]
    state: present

- name: Firewall up + ssh allowed
  ansible.builtin.systemd: { name: firewalld, enabled: true, state: started }
- name: Allow ssh
  ansible.posix.firewalld: { service: ssh, permanent: true, immediate: true, state: enabled }
```

## Role: container_runtime (Docker CE + data-root on /data)

The two Rocky-specific things: install Docker from the **CentOS/RHEL repo**, and
make the relocated **data-root SELinux-clean**.

```yaml
# 1. Docker CE repo + packages (RHEL-family path)
- name: Add Docker CE repo
  ansible.builtin.get_url:
    url: https://download.docker.com/linux/centos/docker-ce.repo
    dest: /etc/yum.repos.d/docker-ce.repo

- name: Install Docker CE + compose plugin
  ansible.builtin.dnf:
    name: [docker-ce, docker-ce-cli, containerd.io, docker-buildx-plugin, docker-compose-plugin]
    state: present

# 2. Relocate data-root to /data BEFORE first start of the daemon
- name: Create data-root dir
  ansible.builtin.file:
    path: "{{ docker_data_root }}"
    state: directory
    mode: "0710"

# 2a. SELinux: label the relocated data-root like the default /var/lib/docker
- name: fcontext for relocated docker data-root
  community.general.sefcontext:
    target: "{{ docker_data_root }}(/.*)?"
    setype: container_var_lib_t
    state: present
- name: Apply the SELinux labels
  ansible.builtin.command: restorecon -RvF {{ docker_data_root }}
  register: relabel
  changed_when: "'Relabeled' in relabel.stdout"

# 3. Point the daemon at the new data-root
- name: daemon.json
  ansible.builtin.copy:
    dest: /etc/docker/daemon.json
    content: |
      {
        "data-root": "{{ docker_data_root }}",
        "log-driver": "json-file",
        "log-opts": { "max-size": "10m", "max-file": "3" }
      }
  notify: restart docker

- name: Enable + start docker
  ansible.builtin.systemd: { name: docker, enabled: true, state: started }

- name: Add {{ ansible_user }} to docker group
  ansible.builtin.user: { name: "{{ ansible_user }}", groups: docker, append: true }
```

> **Why SELinux context matters here:** without `container_var_lib_t` on
> `{{ docker_data_root }}`, the daemon (and containers) are denied access to the
> relocated tree and Docker fails to start or volumes are unreadable. The runbook
> already forced a full relabel via `/.autorelabel` on first boot; this role
> makes the labeling **explicit and idempotent** so re-runs and future dirs stay
> correct.

## Validation (after the role runs)

```bash
docker info | grep -i 'docker root dir'     # → /data/docker
docker run --rm hello-world                 # daemon + runtime sane
findmnt /data                               # data partition mounted
ls -Z /data/docker | head                   # contexts = ...:container_var_lib_t:...
sudo getenforce                             # Enforcing
sudo firewall-cmd --list-services           # ssh (+ later 80/443)
```

**Done when:** Docker runs with `data-root` on `/data/docker`, `hello-world`
passes, SELinux stays Enforcing with correct labels, and the host is reachable
by Ansible with the preserved credentials. The Pi is then ready for
`roles/docs_bridge` and the rest of the design doc.

## Notes / decisions deferred to the app phase

- **`:z`/`:Z` on bind mounts** — the docs-bridge compose templates will need
  SELinux volume labels for any host bind mounts (named volumes are handled
  automatically). Flag for `roles/docs_bridge`.
- **Rootless / Podman** — not used here (Docker chosen). If revisited, Rocky's
  native Podman + Quadlet path is in the design doc §3.
- **Swap / zram, cgroup memory limits** — the design doc sets per-container
  `mem_limits`; confirm cgroup v2 memory accounting is enabled (default on the
  6.12 kernel) when those land.
