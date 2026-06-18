# Provisioning — Decisions & Findings

Recorded so the reasoning isn't lost. Verified against current (mid-2026) sources.

## Hardware

- Raspberry Pi 5 (8 GB).
- Boot/OS today: **Raspberry Pi OS (64-bit), headless, on the microSD card.**
- NVMe HAT + **512 GB NVMe SSD** (currently unused for OS).
- Managed over **SSH + Ansible**. Will run **Docker**.

## Decision 1 — OS and Docker live on the NVMe; SD card stays as fallback

**Chosen:** New OS + Docker on the 512 GB NVMe. The microSD keeps its current
working Raspberry Pi OS, **untouched**, as an automatic boot fallback.

Why, over "OS on SD / SSD for data only":

- Docker does constant small writes; microSD has poor wear-leveling and dies fast
  under that load. NVMe is dramatically faster and more durable.
- The Pi 5 firmware can be told to try NVMe first and **fall back to the SD card**
  if NVMe boot fails — the single thing that makes a remote-ish OS swap safe.
  Wiping the SD to host the OS would throw that fallback away.

## Decision 2 — Rocky Linux 10 (NOT 9, NOT Debian 12)

**Chosen:** **Rocky Linux 10, aarch64 Raspberry Pi image.**

- **Debian 12 "Bookworm" — rejected (won't run).** The Pi 5 shipped after
  Bookworm; its 6.1 kernel has no device-tree for the BCM2712 SoC. Debian's own
  Pi image project states the Pi 5 is only supported on Debian *testing* (Forky)
  or *unstable* (Sid) — not Bookworm, and not even stable Trixie.
  Sources: https://raspi.debian.net/ , https://wiki.debian.org/RaspberryPiImages
- **Rocky Linux 9 — rejected (experimental on Pi 5).** Built on RHEL 9's 5.14
  kernel; no upstream Pi 5 device-tree. Pi 5 support is a community test image
  that "still needs lots of work." Not for a remotely-managed server.
  Source: https://forums.rockylinux.org/t/rocky-linux-9-image-for-rpi5-available-for-testing/13669
- **Rocky Linux 10 — chosen.** RHEL 10 / kernel 6.12 has the Pi 5 device-tree.
  Rocky 10 officially lists **Pi 4 and Pi 5** support with real Raspberry Pi
  images. Support runs into the mid-2030s.
  Sources: https://docs.rockylinux.org/release_notes/10_0/ ,
  https://forums.rockylinux.org/t/image-for-raspberry-pi-5/17355

### Why Rocky fits docs-bridge specifically

The design doc (`docs-bridge-ansible-design.md` §1, §4) calls the package
manager (apt vs dnf) "the only host-level difference" between the Pi and the M2,
and the M2 target is **Fedora Asahi (dnf, RHEL-family)**. Putting Rocky 10 (dnf,
RHEL-family) on the Pi makes **both hosts the same family** → the `common` and
`container_runtime` roles converge instead of branching. Net simplification of
the portability story.

### Rocky-specific gotchas to carry into the runbook

1. **SELinux is enforcing by default.** Docker works, but bind mounts need
   `:z`/`:Z` labels, and the relocated Docker `data-root` must carry the correct
   context (`container_var_lib_t`) or the daemon won't start. Handled via a
   first-boot `/.autorelabel`.
2. **RHEL-family defaults to Podman.** We install **docker-ce** from Docker's
   official CentOS/RHEL repo (per the user's choice). Podman remains the "native"
   option if ever reconsidered.
3. **Headless first boot.** Rocky Pi/cloud images use cloud-init, but we preseed
   more reliably via **chroot** from the running Raspberry Pi OS (both are
   aarch64 — no qemu needed) to guarantee the same user, password, and SSH key
   are in place before first boot.
4. **Filesystem is XFS** (default). Fine for Docker `overlay2` (needs `ftype=1`,
   which mkfs.xfs sets by default). XFS grows but cannot shrink — size the root
   partition right the first time.

## Decision 3 — Disk layout on the NVMe

```
/dev/nvme0n1
├── p1  firmware / boot  (from the Rocky image; FAT/EFI — size as shipped)
├── p2  /        XFS   30 GB        ← OS root
└── p3  /data    XFS   ~rest (~470 GB)
        ├── /data/docker   ← Docker data-root (relocated off the OS partition)
        ├── /data/docs/... ← corpora (matches design doc: /data/docs/<subject>)
        └── container volumes (qdrant, etc.)
```

- 30 GB root is ample for Rocky + packages + logs.
- Single `/data` XFS partition holds everything stateful: Docker images/layers,
  named volumes, and the document corpora. One thing to back up, one thing to
  grow.

## Decision 4 — Boot order (Pi 5 EEPROM)

`BOOT_ORDER=0xf416` — read right-to-left: **6 = NVMe first**, then **1 = SD**,
then **4 = USB**, then **f = restart/loop**. Plus `PCIE_PROBE=1` (SSD HATs are
usually non-HAT+ adapters and need it). Set with `rpi-eeprom-config --edit`.
This lives in the Pi EEPROM and persists regardless of which OS boots.
Source: https://www.jeffgeerling.com/blog/2023/nvme-ssd-boot-raspberry-pi-5/

## Risk posture

- **Physical access to the Pi is available.** Ultimate fallback is "reflash the
  SD / reseat the NVMe and retry" — so a failed swap is a retry, not a brick.
- Firmware-level NVMe boot failure auto-falls-back to the SD (RPi OS still there).
- A "boots-but-hangs" NVMe state would not auto-fall-back, but physical access
  covers it.
