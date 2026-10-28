# Getting a shell on the device: the `usershadow` route, and what it needs

**Date:** 2026-10-28
**Status:** mechanism identified and verified against the device binaries and the
live card. The final step (planting the library) is not yet executed.

Short answer to "can we get the SSH password reset": **we do not need to
reset it.** The device's password lives in a flash-backed config item called
`usershadow`, and the kernel driver that guards that item explicitly
whitelists the AICPU scheduler process — which is exactly the process our
AICPU route gets code into.

---

## 1. The password is a config item, not just a file

The device's `/var/save_shadow` (present in the device rootfs, ARM64, stripped)
is the mechanism:

```text
  Usage: save_shadow [OPTION...]
    save_shadow -r  # load /etc/shadow file from flash.
    save_shadow -w  # store /etc/shadow file to flash.
    save_shadow -c  # check /etc/shadow, if not same, store /etc/shadow file to flash.
  /etc/shadow
  open /etc/shadow err
  ERR:shadow file too large:%u
  save shadow failed
  clear flash shadow file
  save shadow success
  shadow file err, use default shadow file, magic:0x%x
  usershadow
  halSetUserConfig
  halGetUserConfig
```

So `/etc/shadow` is cached in flash under the user-config item **`usershadow`**,
and at boot it is written back over `/etc/shadow`. `halSetUserConfig` and
`halGetUserConfig` are the APIs, and the device's `libascend_hal.so` contains
the `usershadow` string.

**Consequence: writing the `usershadow` item with a shadow file whose
`HwHiAiUser` (or `matteius`, or `root`) line has a known hash *is* setting the
device password.** No `passwd` invocation, no `sudo`, no `/etc/shadow` write
permission needed.

## 2. The host cannot do it — confirmed empirically

`halSetUserConfig` is exported by the **host's** `libascend_hal.so` as well, so
it was worth testing directly rather than assuming. A probe
(`/var/local/ascend-work/tools/userconf_probe.c`) against the live card:

```text
halSetUserConfig=0x7fb3294417da halGetUserConfig=0x7fb3294417c0
halGetUserConfig(usershadow) = 65534
halSetUserConfig(kilo_probe_tmp) = 65534 (0xfffe)
```

Both return **0xfffe** — permission denied. The host is refused outright; the
previous session's note that the API was "device-side only" was right in
effect, if imprecise about why.

## 3. Why the device side is different: the whitelist

The gate is in the device's `drv_user_cfg.ko` (the `user_cfg_drv` module,
confirmed loaded on the device by the kernel log). The exported
`whitelist_process_handler` (`drv_user_cfg.ko+0x9d7c`) does this:

```asm
  9d90:  mov   w22, w1              ; w22 = caller's uid
  9db4:  sub   w0, w22, #1
  9db8:  cmp   w0, #0x31            ; uid must be in 1..50
  9dc0:  ccmp  x21, #0, #0x4, ls    ; table must be non-NULL
  9dc8:  ldr   x0, [x21, x6, lsl #3]   ; entry = table[uid-1]
  9dd4:  cmp   w22, w6             ; bound the loop
  9df8:  bl    whitelist_get_cur_process_name
  9e24:  ldr   x20, [x21, x19, lsl #3]  ; the uid's allowed name
  9e2c:  bl    strlen
  9e30:  cmp   w23, w0             ; lengths must match
  9e34:  b.ne  next
  9e3c:  mov   x1, x25             ; current->comm
  9e40:  mov   x0, x20
  9e44:  bl    memcmp              ; must match exactly
  9e48:  cbnz  w0, next
```

`whitelist_get_cur_process_name` reads `current->comm` straight out of the
kernel task struct (`mrs x4, sp_el0; ldr x0, [x4, #1264]` → `current`, then
`->comm`).

**So the rule is: the caller's uid selects exactly one permitted process
*name*, and the running process's `comm` must equal it byte for byte.**

The permitted names are visible in the module's rodata:

```text
  /var/dmp_daemon
  /var/tsdaemon
  /var/aicpu_scheduler          <-- this one
  /var/slogd
  /usr/lib64/device-compat-plugin/0/aicpu_scheduler
  /usr/lib64/device-compat-plugin/1/aicpu_scheduler
  ... (per-vf variants up to /63/)
```

plus a separate root check whose failure message is
`"Check write authority for root failed."` and a per-item one,
`"The config authority error. (cfg_index=%d; cmd=%d)"`.

**`aicpu_scheduler` is whitelisted.** And the AICPU scheduler is forked by
`tsdaemon`, which runs as **uid 1000** — a uid in the permitted 1..50 range.

## 4. The chain, end to end

1. We deliver a package whose `version.info` has `SandBoxSo=libours.so`.
2. The device's `MoveSoToSandBox` renames `libours.so` into
   `/usr/lib64/aicpu_kernels/0/aicpu_kernels_device/sand_box/` — a plain
   `rename()`, with no hash check, no signature check and no path validation
   (verified by disassembly of `libtsdaemon.so+0x1411b0`).
3. `sand_box/` is **second on the `LD_LIBRARY_PATH`** the device uses to launch
   the AICPU scheduler (device log, `launch_strategy_fork.cpp:105`).
4. The scheduler `dlopen`s our library. Its `comm` is `aicpu_scheduler`, and
   its uid is 1000.
5. Our library calls `halSetUserConfig(0, "usershadow", <our shadow file>)`.
   The kernel module's whitelist check passes: uid 1000 → allowed name
   `aicpu_scheduler`, and `current->comm` is exactly `aicpu_scheduler`.
6. The flash now holds our `usershadow`. On the next boot (or `save_shadow -r`),
   `/etc/shadow` is replaced with ours, and the device's SSH accepts our
   password.

**No exploit, no signature bypass, no hash collision.** The only thing that
has to be true is that the device loads our library — which the previous
session's own analysis had already reduced to "make the package install
correctly", and which this session's hardware test showed *already works*.

## 5. What is still missing

The library has to be built and the package rebuilt. Constraints already
identified:

- **Package size gate.** The device compares the received file's size against
  the saved value (53559460) — `Start load package, ... checkCode=0,
  pkgSize=53559460` then `Load package success, ... checkCode=53559460`.
  The previous session solved this: reproduce the tar byte-for-byte in GNU
  format, pick a gzip memLevel whose output lands under target, and pad the
  container to exact length. The tooling is in
  `/var/local/ascend-work/tools/final_pkg.py`.
- **The hash map covers only `.so` files** — all ten keys observed in the
  device log are libraries, and `version.info` is not among them. So
  `version.info` is an unverified control channel. But the library we
  install *is* named in `bin_hash.cfg`, and that file *is* hash-checked, so
  our `libours.so` needs its SHA-256 written into `bin_hash.cfg` (which the
  previous session's `final_pkg.py` already does for the library it patches).
- **The main `aicpu_scheduler` has no seccomp filter.** Only
  `libaicpu_custom_scheduler.so` installs one (31 syscalls;
  `ioctl`/`openat`/`read`/`write`/`mmap`/`mprotect` allowed, `execve`/
  `socket`/`clone`/`ptrace` not). The scheduler that loads `sand_box/` is the
  main one, so `halSetUserConfig` — an ioctl — is available.
- **`halSetUserConfig` on the device side** must be callable from the
  scheduler's process. `libascend_hal.so` is already a `NEEDED` dependency of
  `libaicpu_scheduler.so`, so the symbol is in the process's link map; the
  scheduler can `dlsym` it.

## 6. If that turns out to be blocked, the fallback

If `halSetUserConfig` turns out to be refused for some reason we have not
anticipated, the same code-execution primitive still gives us a login without
a password at all:

- **`authorized_keys` on the device.** The device's `sshd_config` has
  `PubkeyAuthentication yes` and `AuthorizedKeysFile .ssh/authorized_keys`,
  and the device's account `HwHiAiUser` is **uid 1000 gid 1000 with
  `/bin/bash`**, whose home is `/home/HwHiAiUser` — mode **0750, owned by uid
  1000**, i.e. owned by exactly the uid our AICPU code runs as. Writing
  `/home/HwHiAiUser/.ssh/authorized_keys` (creating the `.ssh` dir, mode 0700)
  needs only that directory, which we own. That account's sudo rights are
  limited to `/bin/ls` by the device's `sudoers`, so this yields a shell,
  not root — but a shell on the device is enough to reach the MCU transport
  (device-side SPI/SMBUS) that the earlier session identified as the real
  target.

  Note the device's `HwHiAiUser` is uid **1000** — the same uid the AICPU
  scheduler runs as — and its shell is `/bin/bash` (the *card host*'s
  `HwHiAiUser` is a different account, uid 1002 with `/usr/sbin/nologin`;
  the two must not be confused).

- **read `/etc/shadow` or `/etc/passwd` directly** as uid 1000 — the AICPU
  process can `openat`/`read`, and the hashes are there.

The `authorized_keys` route does not need the password at all: it needs only a
write into a directory our code already owns. It is the simpler of the two,
and `usershadow` is the one that yields root.

## 6a. A note on the two machines

There are two distinct Linux systems in play, and the earlier session's notes
mix them:

| | card host (`192.168.53.187`) | the Ascend device OS |
|---|---|---|
| uid 1000 | `matteius` (bash) | `HwHiAiUser` (bash) |
| `HwHiAiUser` | uid 1002, `nologin` | uid 1000, bash |
| SSH reachable | yes | yes, IPv6 link-local `fe80::2e0:fcff:fefc:d302%endvnic` from the card host |
| `/etc/shadow` | — | root-owned; shadow cached in flash as `usershadow` |

The device's SSH was confirmed reachable during this session: the key exchange
completes and the banner (`Authorized users only. All activities may be
monitored and reported.`) is the device's, but the credential is refused.

The device's shipped `/etc/shadow` has a **real SHA-512 crypt** for
`HwHiAiUser` (not `!`), so that account does have a password — it is simply
not known, and 87 vendor/common defaults were already tried against it
without a match. That is the credential-recovery problem the AICPU route
sidesteps: we do not need to guess it, we can *replace* it (via
`usershadow`) or bypass it entirely (via `authorized_keys`).

## 7. Method notes

The device-side whitelist was found by reading `drv_user_cfg.ko` from the
extracted device rootfs (`/tmp/kilo/devroot/var/drv_user_cfg.ko`) and
following the `usershadow` string in the device's `libascend_hal.so` to
`halSetUserConfig` → the `0x80044301` ioctl → `dev_user_cfg_ioctl` →
`devdrv_config_ioctl` → `whitelist_process_handler`.

The host-side permission denial was confirmed by direct probe rather than
inference (`userconf_probe.c`, above) — worth doing, because the API turned
out to be exported on the host too, which the earlier notes had missed.
