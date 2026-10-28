# Writing the password hash to flash: which flash, and would it be accepted?

**Date:** 2026-10-28
**Status:** analysis of the device and host drivers. One host-side read test is
outstanding (§5).

## The short answer

**Not "the password hash" — there are two different things here, and only
one of them is in flash.**

1. **`/etc/shadow` on the device** is a *file* in the device OS. It is not
   in flash. Its content comes from the `usershadow` item in the
   **user-config store**, which *is* flash-backed. Writing the hash to flash
   means writing the right bytes into that store, in the store's own
   format, with the store's own head/checksum.

2. **The `usershadow` item is the *copy* of `/etc/shadow`.** It is not the
   authoritative password source — the device OS's own `/etc/shadow` is, and
   at boot `save_shadow -r` copies `usershadow` back over it.

So the honest answer to "would writing the hash to flash work?" is:

- **It would very likely be *accepted*** — the user-config store's integrity
  model is a head block with a validity check
  (`devdrv_user_cfg_head_check`, `devdrv_config_check_blk_validity`,
  "The block is invalid."), **not** a signature and **not** a keyed MAC. The
  device's `hsm_flash.ko` does plain MTD partition erase+write. The store is
  not a trust region in the cryptographic sense; it is a config area with a
  consistency check.
- **But it would not, by itself, change the password**, because the device
  OS does not read `/etc/shadow` from there at runtime — it reads the file.
  You would have to either (a) write the store *and* get the device to
  re-read it (`save_shadow -r`, which needs device execution), or (b) write
  the file, which needs the file to be writable.

**And there is a third option that is much better than either**: the same
store is the one `halSetUserConfig` writes. A single call to
`halSetUserConfig(devId, "usershadow", <shadow file>)` — from the
`aicpu_scheduler` process, which the device's own whitelist permits — writes
the store *in the format the store expects*, with whatever head/checksum the
store requires. That is the `usershadow` route from
`USERSSHADOW-ROUTE-20261028.md`, and it is strictly better than hand-crafting
flash bytes, because the store's own writer is the only thing that
guarantees the format is right.

## 1. The two stores, precisely

```text
  device flash (MTD partition, via hsm_flash.ko)
     |
     |  sec_flash_read / sec_flash_write
     v
  +------------------------------------------------------+
  | user-config store  (drv_user_cfg.ko)                 |
  |   depends=ascend_kernel_adapt,hsm_flash              |
  |   items: "usershadow", "sign_auth_enable", ...        |
  |   integrity: head block + block validity check       |
  |              (devdrv_user_cfg_head_check,            |
  |               devdrv_config_check_blk_validity)      |
  |              NOT a signature, NOT a keyed MAC        |
  +------------------------------------------------------+
     |
     |  read at boot:  save_shadow -r
     v
  /etc/shadow  (the device OS's real password file)
```

```text
  /var/asdrv/dev<N>/device-sw-plugin/verify/flag   <-- a VFS FILE
     read by drv_pkicms.ko via filp_open + read
     NOT flash.  /var/asdrv is not in the shipped rootfs and /var is a
     plain root-owned 0755 dir, so it is created at runtime.
     Written by the vendor's dms_custom_device.o (absent in the 310P build).
```

**This distinction matters and is easy to get wrong.** `/var/asdrv/...` is
*not* a flash path — `drv_pkicms.ko` opens it as an ordinary file
(`filp_open` + `read`, with "read file failed. (read_flag=%d;pos=%lld;read_len=%d)").
The thing that is genuinely flash-backed is the **user-config store** in
`drv_user_cfg.ko`, and `usershadow` is an item in *that*.

## 2. Would the device accept a hand-written store?

The evidence says yes, with caveats:

- `drv_user_cfg.ko` contains **no signature verification and no keyed MAC**
  over the store. Its integrity model is:
  - `devdrv_user_cfg_head_check` — a head block with a self-describing
    length/version and a validity check;
  - `devdrv_config_check_blk_validity` — per-block validity, with
    "The block is invalid." as the failure message.
  Both are *consistency* checks, not *authenticity* checks.
- `hsm_flash.ko` does `mtd` partition erase + write, with no per-write
  signing. Its error strings are "flash write failed", "flash raw write
  failed", "erase length invalid" — i.e. it is a plain MTD writer.
- The store's own writer is `devdrv_config_write_item_content` /
  `devdrv_config_sync_data_to_flash` / `devdrv_config_write_flash_head_content`.
  The **head** is written separately from the item content, so a
  hand-crafted store has to reproduce both, with the right lengths.

Compare with the regions that *are* cryptographically protected on this
card: the MCU's elabel and customer-certificate regions (CRC-16 only, per the
previous session's `mcu_elabel_forge.py`) and the esbc/RSA-PSS package
signature. The user-config store is in neither category.

**So the failure mode is not "the OS rejects the modification as
unauthenticated" — it is "the store's format is more intricate than it
looks, and getting it wrong makes the device fall back to a default".**
That is a solvable problem, not a security boundary.

## 3. What the store looks like on the device

The device's `libtsdaemon.so` and `drv_user_cfg.ko` give the item names
and the API:

- `devdrv_get_user_config_static` / `devdrv_set_user_config_static` /
  `devdrv_clear_user_config_static`
- `devdrv_get_user_config_core`, `devdrv_config_get_item_dynamic`,
  `devdrv_config_update_item`, `devdrv_config_set_item`
- items we know exist: `usershadow`, `sign_auth_enable`

And the *device-side* call we want, from `USERSSHADOW-ROUTE-20261028.md`:

```c
  halSetUserConfig(devId, "usershadow", <shadow file bytes>, len)
```

with the device's `drv_user_cfg.ko` whitelist (`whitelist_process_handler`,
`+0x9d7c`) permitting the process whose `current->comm` is
`aicpu_scheduler` and whose uid is 1000.

## 4. So there are three routes to a changed password, ranked

| # | route | what it needs | assessment |
|---|---|---|---|
| **1** | `halSetUserConfig("usershadow", ...)` from a loaded AICPU library | our library loaded by the AICPU scheduler — i.e. the `SandBoxSo=` channel — which is behind the vendor signature | **best integrity, blocked by the signature** |
| **2** | write the `usershadow` item into the user-config store by hand | knowing the store's block/head format, and a way to write the device's flash | plausible; the store is not signed, but hand-crafting it is error-prone |
| **3** | write `/etc/shadow` on the device | a write to the device's filesystem as uid 1000 | the device's `/etc` is root-owned 0755; not writable by uid 1000 |

Route 1 dominates route 2 *if* we can get a library loaded — it uses the
store's own writer, so the format cannot be wrong. And route 2 dominates
route 3 in one respect: the store is writable in a way the filesystem is
not, if we can find a flash-write primitive that reaches the device's
partition from the host.

## 5. The one test that would settle route 2's viability

The host has a flash-write path: `dmsi_set_flash_content` →
`DMS_SET_FLASH_CONTENT` → `dms_basic_info.c:568` → `sec_flash_write(dev_id,
RTC_FLASH_ADDR + offset, buf, size)`, with
`RTC_FLASH_ADDR = 0xDE0000`, `RTC_FLASH_SIZE = 128 KiB`, and a per-call
cap of `FLASH_CONTENT_BUFFER_MAX_SIZE = 1024` bytes.

**Two things to establish:**

1. **Is the device's user-config store inside the host-writable RTC flash
   region?** `RTC_FLASH_ADDR` is the SoC's *RTC* region. The user-config
   store is a separate MTD partition reached by the *device's* `hsm_flash.ko`.
   If they are different partitions, the host's flash write cannot reach the
   store at all, and route 2 is dead without a device-side writer.
2. **Does `sec_flash_write` verify what it wrote?** The host source has no
   read-back in the write path (`dms_basic_info.c:568` writes and logs
   success). If the region is protected, the write should fail or be
   rejected — and a **read-back test on an unused offset** answers both
   questions at once, without touching anything live.

That read-back test is the obvious next experiment: write a 16-byte pattern
to an unused offset in the RTC region, read it back, and compare. If the
round-trip works, the host can write that flash; if it does not, the region
is write-protected and the question is settled.

The previous session's `flash_probe.c` established the *read* side only
(`dsmi_cmd_get_flash_content`); the write side was never exercised.

## 6. What I would not do

I would not hand-craft flash bytes for the `usershadow` item as the *first*
move. The store's head and block layout is intricate enough that a
malformed write is likely to be *silently* rejected (falling back to the
default flag, or an invalid-block message in the device log) rather than
obviously broken, and that makes it a poor debugging target.

The order that I would actually follow:

1. **Read-back test on the host RTC flash** (§5). Cheap, decisive about
   whether the host has any flash-write capability over the region, and
   needed for route 2 regardless.
2. **Establish whether the device's user-config store and the host's RTC
   region are the same partition** — from the device's partition table
   (`/proc/mtd` or the MTD sysfs), which we can get from the device's
   dmesg or from the collected system_info.
3. **If route 2 looks live, use it via the store's own writer rather than by
   hand** — i.e. find whether the device's `devdrv_config_write_item_content`
   is reachable from an ioctl we can drive from the host, rather than
   synthesising flash content ourselves.
4. **Keep route 1 in view**, because it is the one that also gets us
   `/dev/drv_smbus_intf` and the rest of the device OS.

## 7. On the "the OS would reject it" framing

Worth being precise, because the two halves have different answers:

- The **cryptographic** answer: no. The user-config store is not signed and
  not MAC'd. `usershadow` is a config item with a consistency-checked head
  block, written by a plain MTD writer. There is no signature over it that
  a host-written image would fail.

- The **format** answer: possibly yes. If the store's head encodes
  lengths, versions and a checksum, a hand-written image can be
  structurally wrong, and the device may then reject the *item* (not the
  whole store) and fall back. `devdrv_config_check_blk_validity` and
  "The block is invalid." are the messages to watch for in the device log.

- The **plumbing** answer: this is where it probably fails. Even with a
  perfect image, someone still has to write the device's flash, and the
  host's `sec_flash_write` targets a *different* region (§5.1). Which is
  why route 1 — using the store's own writer, from a process the device's
  whitelist already permits — is the route worth wanting.

## 8. Card state

Unchanged by this analysis. Device log level INFO; whitelist, package-load
ini and host AICPU package at their original md5s; 6/6 chips OK, 0 errors.
