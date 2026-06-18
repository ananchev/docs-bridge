# Pi 5: RPi OS (SD) → Rocky 10 (NVMe) — as-built

One-time bare-metal swap, done from the running RPi OS over SSH. **Completed &
verified:** Rocky Linux 10.2 boots unattended from the NVMe, SELinux enforcing,
user `ananchev` + SSH key + passwordless sudo, `/data` mounted, SD untouched as
fallback. Image: `Rocky-10-SBC-RaspberryPi.latest.aarch64.raw.xz` (Pi-firmware
boot, **ext4** root — not xfs).

⚠️ Destructive steps target `/dev/nvme0n1` only. Never `mmcblk0` (SD).

## Final partition layout (NVMe)

```
p1  500M  vfat  EFI     Pi firmware boot (config.txt, cmdline.txt, kernels)  -> /boot/efi
p2  512M  swap  SWAP
p3   30G  ext4  RPIROOT  /     (grown from the image's ~2.3G)
p4  446G  xfs   DATA     /data (created by us; Docker data-root + corpora)
```

## Steps that worked

1. **Prep:** `sudo rpi-eeprom-update -a`. Note `getent shadow ananchev` (hash) and
   `~/.ssh/authorized_keys` to replicate.
2. **Flash:** download image + `.CHECKSUM` from
   `dl.rockylinux.org/pub/rocky/10/images/aarch64/`, verify sha256, then
   `sudo wipefs -a /dev/nvme0n1` → `xzcat IMG | sudo dd of=/dev/nvme0n1 bs=16M conv=fsync status=progress` → `partprobe`.
3. **Fix GPT + grow root + add data** (needs `gdisk`, `xfsprogs` on the host):
   - `sudo sgdisk -e /dev/nvme0n1`  (relocate backup GPT to end of disk)
   - `sudo parted -s /dev/nvme0n1 resizepart 3 31GiB`
   - `sudo e2fsck -f -y /dev/nvme0n1p3 && sudo resize2fs /dev/nvme0n1p3`
   - `sudo parted -s /dev/nvme0n1 mkpart p.data 31GiB 100%`
   - `sudo mkfs.xfs -f -L DATA /dev/nvme0n1p4`
4. **Chroot preseed** (both sides aarch64 — no qemu):
   - mount `p3`→`/mnt/r`, `p1`→`/mnt/r/boot/efi`, rbind `dev dev/pts proc sys run`,
     then **`sudo mount --make-rprivate /mnt/r`** (see gotcha #3), copy resolv.conf.
   - in chroot: set `/etc/hostname`; `useradd -m ananchev` + `usermod -aG wheel`;
     `usermod -p '<hash>' ananchev`; drop `authorized_keys`; sudoers NOPASSWD
     drop-in; `systemctl enable sshd`; add `LABEL=DATA /data xfs defaults,noatime 0 0`
     to fstab; `touch /.autorelabel`.
5. **Boot config (the critical fix — see gotcha #1):** on `/boot/efi`:
   - `cmdline.txt`: `root=LABEL=RPIROOT` → `root=PARTUUID=<p3 PARTUUID>`
   - `config.txt`: comment out `auto_initramfs=1`
6. **EEPROM:** `sudo rpi-eeprom-config --apply <file>` with `BOOT_ORDER=0xf416`
   (NVMe-first) + `PCIE_PROBE=1`.
7. **Reboot.** Then `restorecon -Rn /` (expect 0), `setenforce 1`, and remove any
   permissive crutch from cmdline.

## Gotchas we actually hit (read these)

1. **NVMe boot needs `root=PARTUUID` + NO initramfs.** The Pi 5 firmware loads the
   **+16k** kernel (`kernel_2712.img`) but the image ships only `initramfs8` built
   for the **+4k** kernel → wrong/no initramfs → `root=LABEL=` can't be resolved →
   **kernel panic "VFS unable to mount root fs"**. Fix: `nvme`, `ext4`, `mmc` are
   **built into** the Pi kernel, so point the kernel straight at the partition with
   `root=PARTUUID=…` and disable `auto_initramfs` — no initramfs required.
2. **EEPROM applies on the *next* boot.** `rpi-eeprom-config --apply` flashes the
   chip immediately, but `vcgencmd bootloader_config` / `rpi-eeprom-config` keep
   showing the value cached at last boot until you reboot. `0xf416`=NVMe-first,
   `0xf461`=SD-first. **NVMe-first means a NVMe *kernel* panic does NOT fall back
   to SD** (firmware already handed off) — keep **SD-first while testing**, or be
   ready to unplug the NVMe ribbon and boot the SD to fix things.
3. **chroot mount propagation can nuke the host.** Without
   `mount --make-rprivate /mnt/r`, `umount -R /mnt/r` propagated and unmounted the
   *host's* `/dev/pts` → `sudo: unable to allocate pty`. Recover with
   `sudo mount -t devpts devpts /dev/pts`.
4. **SELinux `/.autorelabel` can console-storm.** The boot relabel + rsyslog
   writing each notice to `/var/log/messages` fed back into a flood that never
   reached sshd. Boot **permissive** first (`enforcing=0 loglevel=3` on cmdline),
   confirm `restorecon -Rn /` shows **0** mislabels, `setenforce 1` live, then drop
   the cmdline flags. (The relabel itself completed fine — it was just the noise.)
5. **Lock the image's default account:** `sudo usermod -L rocky` (default password
   `rockylinux` is public).

**Rollback:** SD is never touched. Unplug NVMe ribbon → boots SD; or set
`BOOT_ORDER=0xf461`. Re-flash from step 2 to start over.
