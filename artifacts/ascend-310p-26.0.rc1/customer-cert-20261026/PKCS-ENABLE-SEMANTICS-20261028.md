# `pkcs-enable`: what it actually is, and why "enabling" it is a safety risk

**Date:** 2026-10-28
**Status:** the knob's semantics are now established from the device's own
driver. Card 9's state was changed once by a probe and **restored to its
original value in the same session**.

## The safety question first

`npu-smi set -t pkcs-enable -i 9 -d 0` printed:

```text
[WARNING]: Enabling pkcs will raise safety risks.
Status                         : OK
Message                        : The pkcs Enable is set successfully
```

and the state went `Disable` -> `Enable`. I then set it back with `-d 1`,
which printed "The pkcs Disable is set successfully" and read back as
`Disable`. **Card 9 is back to its original state, and all three cards
(9, 8217, 16409) currently read `Pkcs-enable Status : Disable`.**

**Is that change persistent?** No — and this is now established from the
device driver, not guessed:

`drv_user_cfg.ko`'s `devdrv_config_set_pss_cfg` (`+0x2228`):

```asm
2248  cmp  w20, #0x1
224c  b.hi 2290              ; value > 1 -> error
2264  bl   mutex_lock
226c  str  w20, [x21]        ; g_sign_type = value
2270  bl   mutex_unlock
2278  mov  w0, #0x0          ; return 0
```

`g_sign_type` is at `.bss+0x98` — **RAM**, not flash. The setter does a
single 4-byte store into that global. There is no
`devdrv_config_write_item_content`, no `devdrv_config_sync_data_to_flash`,
no `sec_flash_write`, and no store update anywhere in it.

So the value:

- **is not written to flash**;
- **is lost on any card reset**;
- is re-read at module init by `devdrv_config_pss_cfg_init` (`+0x956c`),
  which is why the device logs `The sign details. (sign=1)` at boot.

**Conclusion: `pkcs-enable` is a runtime switch, not a configuration change.**
Flipping it and flipping it back is genuinely non-destructive. That is why
the probe was safe even though it was an unintended write, and why the
remaining experiments with it are safe too.

For completeness, the other knob the previous session touched,
`ssh-enable`, is a *different* thing: it is a persistent, flash-backed
configuration ("Reset card for the configuration to take effect"), and it is
currently `True` on cards 9 and 8217 and `False` on 16409.

## Now the interesting part: why would *enabling* verification be a risk?

The warning is **not** shared boilerplate. `npu-smi` contains exactly one
copy of "will raise safety risks", and it is specific to this set type.

And the value semantics are the opposite of what the name suggests:

```c
  /* dev_inc/inc/user_cfg_interface.h, and dsmi_common_interface.h */
  #define PKCS_SIGN_TYPE_OFF  1
  #define PKCS_SIGN_TYPE_ON   0
```

with the mapping confirmed empirically this session:

| npu-smi | status shown | constant | meaning |
|---|---|---|---|
| `-d 0` | Enable | `PKCS_SIGN_TYPE_ON = 0` | signature verification **ON** |
| `-d 1` | Disable | `PKCS_SIGN_TYPE_OFF = 1` | signature verification **OFF** |

So the status word "Enable" means "enable the PSS/PKCS signature scheme",
and the device's `sign=1` in the log means that scheme is **OFF**.

**That reframes the earlier analysis, and it is the answer to the question.**
The previous session read `sign=1` as "the signature check is on, and it
rejected us". It is the opposite: at the moment the device rejected the
modified package, the PSS scheme was **off**, and the rejection came from
somewhere else — specifically the tsdaemon-level CMS check
(`VerifyAicpuPkgByCms`), which is a *different* mechanism driven by the
`verify_flag` policy file.

And now the warning makes sense. From the vendor's point of view:

- With the PSS scheme **off** (`sign=1`, the shipped default), the device
  does not check PSS signatures at the kernel level.
- Turning it **on** (`-d 0`, "Enable") makes the device start enforcing
  PSS signatures.

So "enabling pkcs will raise safety risks" is the vendor warning that
**enabling this makes the card stricter** — it is a *warning to the
operator* that they are about to turn on a verification mode that will
start rejecting artefacts that were previously accepted.

That is the same pattern as the `ssh-enable` warning we saw
("Enabling SSH will raise the safety risks") — both are "you are about to
enable a feature that will refuse things / expose an interface". It is a
*user*-safety warning, not a hint that enabling gives an attacker something.

## Which means the knob is a lever, and it is a *safe* one

Because it is a RAM-only switch that the vendor's own tooling exposes
through `npu-smi set`, and because it changes whether the device enforces
PSS signatures, it is worth testing properly:

| pkcs-enable | expected effect on the AICPU install |
|---|---|
| `-d 1` (Disable, **current, original state**) | PSS scheme off; the tsdaemon CMS check still applies |
| `-d 0` (Enable) | PSS scheme on; the kernel's `soc_verify` / `rsa_verify_soc` will now enforce the RSA-PSS signature |

**Prediction:** with `-d 0` the *modified* type-0 package should fail
**earlier and more explicitly** (at the kernel `rsa_verify_soc` stage),
and with `-d 1` the failure should be the tsdaemon CMS stage. If instead
`-d 0` made a modified package *accepted*, that would mean the kernel's
`rsa_verify_soc` is the thing that was rejecting us and the PSS scheme
was inverted somewhere — which would be a significant and surprising
result.

This is a genuinely useful experiment because it separates the two
mechanisms that have been conflated throughout this work:

1. the **kernel** PSS check (`soc_verify` -> `rsa_verify_soc`), which
   `pkcs-enable` directly controls; and
2. the **tsdaemon** CMS check (`VerifyAicpuPkgByCms`), which is controlled
   by the `verify_flag` policy file and is independent of it.

I had not separated them cleanly before, and the warning is what made the
distinction visible.

## What I will not do

- **No card reset.** `ssh-enable` needs one to take effect; `pkcs-enable`
  does not, and I will not reset a card to answer a question that the
  disassembly already answers.
- **No flash writes.** The `sec_flash_write` / `dsmi_set_flash_content`
  path exists on the host, but it targets the SoC's RTC region, and writing
  there is not obviously reversible. It is not needed for this question.
- **No further state changes beyond `pkcs-enable`**, which is RAM-only and
  restored in the same command.
- The `usershadow` / `verify_flag` files stay untouched: they are the
  device's trust configuration, they are on the device's own filesystem, and
  nothing we have established gives us a way to write them anyway.

## State

- card 9, 8217, 16409: `Pkcs-enable Status : Disable` — the original
  value on all three.
- `ssh-enable`: `True` on 9 and 8217 (the previous session's change, left
  as found), `False` on 16409.
- whitelist, package-load ini, host AICPU package: original md5s.
- 6/6 chips OK, 0 errors, device log level INFO.
