# Pi 5: RPi OS (SD) → Rocky 10 (NVMe)

One-time bare-metal swap. Can't be Ansible (host isn't Rocky yet). Done from the
running RPi OS over SSH. After this, the Pi is reachable on Rocky and Ansible
takes over (`../ansible/`).

**Layout:** NVMe = 30 GB XFS root + rest as `/data` (Docker data-root, corpora,
volumes). SD kept untouched as boot fallback.
**Why Rocky 10:** Pi 5 needs kernel ≥6.6 — rules out Debian 12 and Rocky 9; Rocky
10 (RHEL 10, k6.12) officially supports Pi 5 and is dnf/RHEL-family like the M2.

⚠️ Every destructive step targets `/dev/nvme0n1` (`$NVME`). Never touch `mmcblk0`
(the SD). Confirm with `lsblk` first.

1. **Prep:** `sudo rpi-eeprom-update -a` (reboot if it updates). Save off-device:
   `getent shadow ananchev`, `~/.ssh/authorized_keys`.
2. **Flash:** download Rocky 10 Pi image + checksum from rockylinux.org/download,
   `sha256sum -c`, then `xzcat IMG | sudo dd of=$NVME bs=16M conv=fsync status=progress`, `partprobe`.
3. **Partition:** `parted $NVME resizepart 2 30GiB`; grow root XFS
   (`mount … && xfs_growfs`); `parted $NVME mkpart primary xfs 30GiB 100%`;
   `mkfs.xfs -L DATA ${NVME}p3`.
4. **Preseed (chroot — both aarch64, no qemu):** mount root+boot, rbind
   dev/proc/sys/run, `chroot`. Inside: set hostname; ensure user `ananchev`
   (`-G wheel`); reuse password hash (`usermod -p '<hash>'`); drop the same
   `authorized_keys`; `systemctl enable sshd`; add `/data` to fstab by UUID;
   `touch /etc/cloud/cloud-init.disabled`; `touch /.autorelabel` (SELinux).
   Confirm `root=` points at `${NVME}p2`.
5. **Boot order:** `sudo rpi-eeprom-config --edit` → `BOOT_ORDER=0xf416`
   (NVMe→SD fallback) + `PCIE_PROBE=1`.
6. **Reboot & verify:** `ssh ananchev@192.168.2.11`; `cat /etc/rocky-release`;
   `findmnt /` (=nvme0n1p2), `findmnt /data`, `getenforce`=Enforcing.

**Rollback:** firmware-level NVMe failure auto-falls-back to SD. Otherwise pull
NVMe or set `BOOT_ORDER=0xf41` (SD only). The SD is never modified — always safe.

> Confirm at execution: exact image filename, boot scheme (cmdline.txt vs
> extlinux vs GRUB) and where `root=` lives, boot-partition mountpoint.
