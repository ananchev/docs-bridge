# Pi 5 OS Swap Runbook — Raspberry Pi OS (SD) → Rocky Linux 10 (NVMe)

**Phase 1, one-time, manual.** Outcome: the Pi boots Rocky 10 from the NVMe with
your existing username, password, and SSH key, reachable over SSH at the same
address — ready for Phase-2 Ansible host prep (`02-host-prep.md`).

> Read `00-decisions.md` first for the layout and rationale.

## Conventions / fill these in before starting

| Placeholder | Meaning | How to get it |
|---|---|---|
| `$USER` | your login on the current Pi (must be reused on Rocky) | `whoami` on the Pi |
| `$HOST` | hostname to keep | `hostname` |
| `$NVME` | NVMe device | almost certainly `/dev/nvme0n1` — confirm in Stage 1 |
| `$IMG` | Rocky 10 Pi image file | downloaded in Stage 1 |

Run everything below **on the Pi over SSH**, booted from the SD card as now.
Use `sudo`/root throughout. **Do not** operate on the SD device — only `$NVME`.

⚠️ **Identify the NVMe device explicitly before any `dd`/`parted`.** Writing to
the wrong device destroys the running system. Every destructive command names
`$NVME`; verify it points at the SSD, not the SD card (`mmcblk0`).

---

## Stage 0 — Prep & safety

```bash
# 0.1  Update the Pi 5 bootloader/EEPROM firmware (needed for reliable NVMe boot)
sudo rpi-eeprom-update -a
# reboot if it staged an update, then continue
sudo reboot   # only if 0.1 reported an update

# 0.2  Keep the same IP so SSH/Ansible continuity holds.
#      Set a DHCP reservation on your router for the Pi's MAC, OR note the
#      current static config to replicate later:
ip -4 addr show
cat /etc/dhcpcd.conf 2>/dev/null | sed -n '/static/p'   # if static today

# 0.3  Capture what we must replicate onto Rocky:
whoami; hostname
sudo getent shadow "$USER"        # the password hash line (copy it; reused later)
cat ~/.ssh/authorized_keys        # your SSH public key(s)
# Save these somewhere safe on your workstation.

# 0.4  Confirm the SD stays as fallback: do NOT modify mmcblk0 anywhere below.
lsblk -o NAME,SIZE,TYPE,MOUNTPOINT
```

**VERIFY:** you have, saved off-device: `$USER`, `$HOST`, the shadow hash, and
your `authorized_keys`. The SD device is `mmcblk0`; the NVMe is `/dev/nvme0n1`.

---

## Stage 1 — Write the Rocky 10 image to the NVMe

```bash
# 1.1  Identify the NVMe unambiguously
lsblk -o NAME,SIZE,TYPE,MODEL | grep -i nvme
NVME=/dev/nvme0n1            # adjust only if the above shows otherwise
sudo wipefs -n "$NVME"      # dry-run: shows existing signatures (sanity check)

# 1.2  Download the Rocky 10 Raspberry Pi image to a working dir on the SD.
#      Get the current URL + checksum from https://rockylinux.org/download
#      (Raspberry Pi / aarch64 section). Example shape:
cd /var/tmp
curl -fLO "https://dl.rockylinux.org/pub/rocky/10/images/aarch64/<ROCKY-PI-IMAGE>.img.xz"
curl -fLO "https://dl.rockylinux.org/pub/rocky/10/images/aarch64/<ROCKY-PI-IMAGE>.img.xz.CHECKSUM"
sha256sum -c <ROCKY-PI-IMAGE>.img.xz.CHECKSUM      # MUST pass before flashing
IMG=/var/tmp/<ROCKY-PI-IMAGE>.img.xz

# 1.3  Make sure nothing on the NVMe is mounted, then flash.
sudo umount "${NVME}"* 2>/dev/null
xzcat "$IMG" | sudo dd of="$NVME" bs=16M conv=fsync status=progress

# 1.4  Re-read the partition table
sudo partprobe "$NVME"
lsblk "$NVME"
```

**VERIFY:** `lsblk "$NVME"` now shows the image's partitions (typically a small
FAT/EFI boot partition + a small root). Checksum passed.

> 📌 **Inspect the image's boot mechanism now** — it determines how `root=` is
> set and is the one image-specific unknown. Mount the boot partition and look:
> ```bash
> sudo mkdir -p /mnt/inspect && sudo mount "${NVME}p1" /mnt/inspect
> ls /mnt/inspect
> # Pi-firmware style → cmdline.txt + config.txt
> # extlinux/u-boot   → extlinux/extlinux.conf
> # GRUB/EFI          → EFI/ + grub.cfg
> grep -RiEl 'root=' /mnt/inspect || true
> sudo umount /mnt/inspect
> ```
> Note which scheme it is; you'll confirm the `root=` reference in Stage 3.7.

---

## Stage 2 — Repartition: 30 GB root + rest as /data

The image root is small; grow it to 30 GB and add a data partition after it.
Identify partition numbers from `lsblk "$NVME"` (assume `p2` = root below).

```bash
# 2.1  Grow the root partition to 30 GB (start stays, end moves to 30 GiB region).
#      parted prints the existing layout; note p2's start before resizing.
sudo parted "$NVME" unit GiB print

# Resize p2 to end at ~30 GiB from disk start (keep p2's existing start).
# Example assumes p2 starts well under 1 GiB; adjust the end if start is larger.
sudo parted "$NVME" resizepart 2 30GiB

# 2.2  Grow the XFS filesystem on the (now larger) root partition.
sudo mkdir -p /mnt/rocky
sudo mount "${NVME}p2" /mnt/rocky
sudo xfs_growfs /mnt/rocky          # XFS grows online/offline; fills p2

# 2.3  Create the data partition on the remaining space and format it.
sudo parted "$NVME" mkpart primary xfs 30GiB 100%
sudo partprobe "$NVME"
lsblk "$NVME"                        # confirm new p3 exists
sudo mkfs.xfs -L DATA "${NVME}p3"    # ftype=1 by default → overlay2-ready
```

**VERIFY:** `lsblk "$NVME"` shows p2 ≈ 30 GB (XFS, mounted at /mnt/rocky) and a
new p3 ≈ 470 GB (XFS, labeled DATA). `root`'s PARTUUID is unchanged by resize, so
the image's `root=` reference still points at it.

> If `parted resizepart` complains about the partition being in use, ensure
> `${NVME}p2` is unmounted first; remount for 2.2 after the resize.

---

## Stage 3 — Preseed Rocky via chroot (same user, key, hostname)

Both the running OS and the Rocky root are **aarch64**, so we can chroot directly
— no qemu. Goal: a host that boots and accepts your SSH key. Docker/data-root are
deferred to Phase-2 Ansible.

```bash
# 3.1  Mount the Rocky root + its boot partition, and bind the kernel virtual FS.
sudo mount "${NVME}p2" /mnt/rocky                 # if not already mounted
sudo mount "${NVME}p1" /mnt/rocky/boot            # adjust if boot mountpoint differs
for d in dev dev/pts proc sys run; do sudo mount --rbind /$d /mnt/rocky/$d; done
sudo cp /etc/resolv.conf /mnt/rocky/etc/resolv.conf

# 3.2  Enter the chroot
sudo chroot /mnt/rocky /bin/bash
```

Inside the chroot:

```bash
# 3.3  Hostname
echo "$HOST" > /etc/hostname

# 3.4  Recreate your user with the SAME name, add to wheel (sudo) and the group
#      docker will create later. Skip useradd if the image already has $USER.
id "$USER" 2>/dev/null || useradd -m -G wheel "$USER"

# 3.5  Reuse the SAME password — paste the shadow hash captured in 0.3:
#      (usermod -p takes an already-hashed value)
usermod -p '<PASTE_THE_HASH_FROM_getent_shadow>' "$USER"
#   …or set interactively:  passwd "$USER"

# 3.6  Reuse the SAME SSH key
install -d -m700 -o "$USER" -g "$USER" /home/$USER/.ssh
cat > /home/$USER/.ssh/authorized_keys <<'EOF'
<PASTE YOUR authorized_keys FROM 0.3>
EOF
chmod 600 /home/$USER/.ssh/authorized_keys
chown "$USER:$USER" /home/$USER/.ssh/authorized_keys

# passwordless sudo for wheel is usually preset on Rocky; confirm:
grep -E '^\s*%wheel' /etc/sudoers /etc/sudoers.d/* 2>/dev/null

# 3.7  Enable SSH on first boot
systemctl enable sshd

# 3.8  Mount /data persistently. Use the partition UUID (stable), not pN.
blkid "${NVME}p3"   # note UUID=...   (run OUTSIDE chroot if blkid is absent here)
mkdir -p /data
cat >> /etc/fstab <<EOF
UUID=<DATA_PARTITION_UUID>  /data  xfs  defaults,noatime  0 0
EOF

# 3.9  Avoid cloud-init clobbering our manual user setup on first boot.
#      (Rocky Pi images ship cloud-init; we configured by hand, so disable it.)
touch /etc/cloud/cloud-init.disabled 2>/dev/null || true

# 3.10 Force a full SELinux relabel on first boot (covers /data + our new files)
touch /.autorelabel

exit   # leave chroot
```

```bash
# 3.11 (outside chroot) confirm the root= reference matches reality.
#      Using the scheme you found in Stage 1's inspect step, verify root= points
#      at the root partition's PARTUUID/label. Example for Pi-firmware style:
sudo blkid "${NVME}p2"            # note PARTUUID=...
sudo grep -R 'root=' /mnt/rocky/boot   # ensure it matches the PARTUUID above
```

**VERIFY:** `/etc/hostname`, `$USER` present with your hash + key, `sshd` enabled,
`/data` in `/etc/fstab` by UUID, `/.autorelabel` exists, and `root=` resolves to
`${NVME}p2`.

```bash
# 3.12 Clean unmount
sudo umount -R /mnt/rocky
```

---

## Stage 4 — Set the boot order (NVMe first, SD fallback)

This is a Pi EEPROM change; do it from the **current Raspberry Pi OS** (it has
`rpi-eeprom-config`). It persists across the OS swap.

```bash
sudo -E rpi-eeprom-config --edit
# Set / add:
#   BOOT_ORDER=0xf416     # 6=NVMe first, 1=SD fallback, 4=USB, f=restart-loop
#   PCIE_PROBE=1          # SSD HAT is typically a non-HAT+ adapter
# Save & exit; tool stages the update for next boot.

# Confirm it staged:
sudo rpi-eeprom-config
```

**VERIFY:** `rpi-eeprom-config` output shows `BOOT_ORDER=0xf416` and
`PCIE_PROBE=1`.

---

## Stage 5 — Reboot into Rocky & verify

```bash
sudo reboot
```

First Rocky boot does a full SELinux relabel (from 3.10) and reboots itself once
— give it a few minutes before expecting SSH. Then from your workstation:

```bash
ssh $USER@$HOST            # same key, same password, same host as before

# On the Pi (now Rocky):
cat /etc/rocky-release      # confirm Rocky 10
findmnt /                   # root on /dev/nvme0n1p2 (XFS)
findmnt /data               # /data on /dev/nvme0n1p3 (XFS)
lsblk                       # mmcblk0 present but NOT the root → SD is fallback only
getenforce                  # Enforcing
uname -r                    # 6.12-series kernel
```

**VERIFY:** Rocky 10 boots **from the NVMe**, you log in with the existing key,
`/` is on `nvme0n1p2`, `/data` is mounted, SELinux is Enforcing. The SD card is
present but unused for boot.

➡️ Host is now ready for **Phase 2** (`02-host-prep.md`): Docker CE install,
data-root → `/data/docker`, firewall, and the rest, via Ansible.

---

## Stage 6 — Rollback

- **NVMe failed at firmware level** (no valid boot partition found): the Pi auto-
  falls back to the SD and boots the old Raspberry Pi OS. Debug the NVMe from
  there, re-do the stage that failed.
- **NVMe boots but hangs / unreachable:** power down, and either
  - pull the NVMe so firmware falls back to SD, or
  - temporarily force SD-only boot: `rpi-eeprom-config --edit` → `BOOT_ORDER=0xf41`
    (no NVMe), boot SD, fix, then restore `0xf416`.
- **Start over cleanly:** re-run from Stage 1 (`dd` re-flashes the NVMe). The SD
  install is never touched, so it's always a safe harbour.

---

## Open items to confirm at execution (image-specific)

1. **Exact Rocky 10 Pi image filename + checksum URL** (Stage 1.2) — from
   rockylinux.org/download.
2. **Boot mechanism** — Pi-firmware `cmdline.txt` vs `extlinux.conf` vs GRUB
   (Stage 1 inspect, Stage 3.11). Adjust the `root=` verification accordingly.
3. **Boot partition mountpoint inside the image** (`/boot` vs `/boot/efi`) — set
   the 3.1 mount target to match.
4. **Whether the image preseeds a default user** (e.g. `rocky`) — if so, you may
   keep it disabled/locked and rely solely on `$USER`.
