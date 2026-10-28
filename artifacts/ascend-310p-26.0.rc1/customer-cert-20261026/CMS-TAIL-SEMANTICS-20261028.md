# The narrow question, answered: what the device does with a package that has no CMS tail

**Date:** 2026-10-28
**Status:** resolved, by disassembly *and* by three hardware experiments that
agree with it.

**The question.** `CmsCbbVerifyPkg::GetCmsInfoOffSetFromPkg` returns `0x2f`
when it cannot find the CMS structure, and `VerifyAicpuPkgByCms` appears to
take that branch and return success (`mov w20, #0x0`). But a package with no
CMS tail demonstrably fails on the device. Which is right, and why?

**The answer.** The disassembly was right about the code and I was wrong about
what the code *reaches*. The `0x2f` path is not the path a no-tail package
takes, because the scanner does not return `0x2f` for "no CMS tail" — it
returns `0x2f` for "the container magic is not where it should be", and the
check that produces it happens **before** the magic is ever examined.

---

## 1. What the scanner actually does

`CmsCbbVerifyPkg::GetPkgCodeLenAndCertType` (`+0x18e3d8`):

```asm
18e4a0  bl   fseek                 ; seek to the end
18e4ac  bl   ftell                 ; x25 = file size
18e4b4  cmp  x0, #3
18e4b8  b.gt 18e658               ; size > 3 -> read the magic
        ...                       ; else "file length less than 4 bytes" -> 0x26

18e658  ldur x0, [x29, #-144]      ; the FILE*
18e65c  bl   rewind                ; back to the start
18e664  mov  x2, #1                ; 1 element
18e668  mov  x1, #4                ; 4 bytes
18e66c  sub  x0, x29, #0x94        ; destination buffer
18e670  stur wzr, [x29, #-148]      ; zeroed
18e674  bl   fread                 ; fread(buf, 4, 1, file)   <- the FIRST 4 bytes
18e67c  cmp  w0, #1
18e680  b.eq 18e7a0               ; 1 element read -> check it
        ...                       ; else "fread file magic number failed" -> 0x26

18e7b4  ldur w1, [x29, #-148]      ; the 4 bytes just read
18e7b8  mov  w0, #0x55aa
18e7bc  movk w0, #0x55aa, lsl #16  ; w0 = 0x55aa55aa
18e7c0  cmp  w1, w0
18e7c4  b.eq 18e7f0                ; magic OK -> parse the CMS block
18e7c8  ldr  w0, [x24]             ; else: the "isCmsFlag" out-param
18e7cc  cbnz w0, 18e8a8            ; isCmsFlag set -> error path
        ...
18e7e8  mov  w20, #0x2f            ; isCmsFlag clear -> 0x2f (no CMS tail)
18e7ec  b    18e594
```

So there are **two** ways to get `0x2f`, and they are not the same thing:

| condition | result |
|---|---|
| the file's **first 4 bytes are not `0x55aa55aa`** and `isCmsFlag` is clear | `0x2f` — "this is not a CMS container" |
| the file's first 4 bytes **are** `0x55aa55aa` but the CMS block then fails to parse | error (0xe / 0x14) |

`0x55aa55aa` is the **esbc container magic**, and it is the first 4 bytes of
the file. The `ascend_aicpu` container starts with them:

```text
  file[0:4]  = aa 55 aa 55      <- 0x55aa55aa, the container magic
  file[4:8]  = 00 00 00 00      <- header length
  ...
  file[0x2018] = 32-byte digest = sha256(the gzip member)
  file[0x205c] = gzip member length
  file[0x2068] = package type (9)
```

## 2. The three experiments

All three were sent into the device's `aicpu/` staging directory as
`1830147_Ascend310P-aicpu_legacy.tar.gz` and checked in the device's own
tsdaemon log. Only the control passed.

| variant | first 4 bytes | device result |
|---|---|---|
| X1 | `00 00 00 00` | "by cms start" only — **fails** |
| X2 (control) | `aa 55 aa 55` (unchanged) | **"Load package success, type=9"** |
| X3 | `1f 8b 08 00` | "by cms start" only — **fails** |

And, from the earlier isolation run:

| variant | result |
|---|---|
| stock tar, CMS tail **stripped** | "by cms start" only — fails |
| stock tar, CMS tail **kept** | "Package verify success" → "Load package success, type=9" |
| our modified tar, tail stripped | fails |
| our modified tar, stock tail copied on | fails |
| our modified tar + magic in the last 4 bytes | fails |
| our modified tar + a stub `cms\0` block | fails |

**Stripping the CMS tail removes the magic with it** — because in the
`ascend_aicpu` container the magic is at the *front* and the CMS block is at
the *back*, and the two are the same file. A "plain tar.gz" has neither, so
its first 4 bytes are `1f 8b 08 00`, which is not the magic — and the
scanner takes the `isCmsFlag` branch, not the `0x2f` branch.

## 3. Why `0x2f` is not reachable for us

`0x2f` requires **two** things at once:

1. the file's first 4 bytes are not `0x55aa55aa`, **and**
2. `isCmsFlag` is clear.

`isCmsFlag` is the out-parameter of `GetPkgCodeLenAndCertType`, set at
`+0x18e7c8` from `[x24]`. It comes from the caller. Tracing back, it is
`GetCmsInfoOffSetFromPkg`'s `bool&` argument — the "is this a mixed
(vendor+community) package" flag — which `VerifyAicpuPkgByCms` passes as the
`isMix` parameter, and which the device sets from the package's cert type.

So on the device the flag is **set**, and a file without the magic therefore
takes the error path at `+0x18e8a8` (which restores the error message
"GetPkgCodeLen and CertType failed" and returns `0xe` = 14), not the `0x2f`
path.

**`0x2f` is the "not a CMS container at all" case, and it is only reachable
for a package the device has already decided is not CMS-verified** — i.e.
after `IsVerifyFlagValid` has already said so. It is not a way in; it is the
way the code says "I did not check".

## 4. The complete picture of the gate

```text
VerifyPkgByCmsEnhance(devId, path, isMix)                 +0x18fa10
  SetVerifyFlag(devId)                                   -> ParseVerifyFlagFromCfgFile
                                                          reads /var/asdrv/dev<N>/device-sw-plugin/verify/flag
                                                          missing file -> returns 1
                                                          "verify_flag=N" -> returns N
  IsVerifyFlagValid()                                    cmp w0,#7 ; cset w0,le   (flag <= 7)
  b.eq -> skip verification entirely, log "verify pkg[%s] no cms just rewrite"
  VerifyAicpuPkgByCms(devId, path, isMix)                +0x18f528
    GetCmsInfoOffSetFromPkg(path, ...)                   +0x18ee38
      GetPkgCodeLenAndCertType(path, ...)                +0x18e3d8
        fread(file, 4, 1)  ->  compare against 0x55aa55aa
          not the magic:
            isCmsFlag set  -> error 0xe
            isCmsFlag clear -> 0x2f  (treated as success, no CMS tail)
          is the magic:    -> parse the CMS block
    0x2f  -> w20 = 0 -> SUCCESS (no "by cms end" log)
    else  -> verify the PKCS#7 signature over the package code
```

Three separate things must hold, and our modified package fails the third:

1. **`verifyFlag <= 7`** so the check runs at all. The device's default is 1
   (`SIGN_FLAG_ONLY_VENDOR`) because the policy file does not exist, and the
   310P build has no writer for it. This is the only *policy* gate, and it
   is a numeric bypass: a flag of 8 or more would skip verification
   entirely.
2. **the container magic `0x55aa55aa` at the start of the file**, so the
   scanner recognises it as a container and parses the CMS block.
3. **a valid PKCS#7 signature over the package code** — which is the
   vendor's signature over *our* content, and cannot be forged.

Requirement 2 is why the plain-tar.gz idea fails: a tar.gz has neither the
container magic nor the CMS block, and the device treats "not a container" as
an error (when `isCmsFlag` is set) rather than as "skip".

## 5. What this rules out, and what it leaves

**Ruled out.** Any hope of delivering an unsigned or modified package
through the type-9 route. The `0x2f` "no CMS tail" path is not a bypass; it
is the code reporting that it declined to check, and it is only reachable for
packages the device has already classified as non-CMS.

**Left, and it is a smaller set than before:**

- **The `verifyFlag` policy gate.** A value of 8+ on the device makes
  `IsVerifyFlagValid` false and skips the whole check, logging
  `"verify pkg[%s] no cms just rewrite"`. The flag is a plain integer in
  `/var/asdrv/dev<N>/device-sw-plugin/verify/flag`, read by tsdaemon on the
  device, and the 310P build compiles out both `dms_custom_host.o` and
  `dms_custom_device.o` — the only writers of it. Getting a value there is
  the one remaining host-side lever, and it is a *file write on the device*,
  not a package.

- **The `SandBoxSo=` channel**, which remains the delivery mechanism for
  code execution — but it lives inside the signed payload, so it needs
  requirement 3 to be met first.

The dependency is now explicit and one-directional:

```text
  get verifyFlag >= 8 onto the device   (needs device-side file write)
        |
        v
  the type-9 package's CMS check is skipped
        |
        v
  a modified package (new version.info, our .so in sand_box/) installs
        |
        v
  MoveSoToSandBox renames our .so into sand_box/  (no validation)
        |
        v
  the AICPU scheduler dlopen()s it
```

Step one is the bottleneck, and it is the same bottleneck the previous
session identified from the other side: **we need to write one file on the
device, and the only file-write primitive we have is the package installer,
which is behind the check we are trying to skip.**

That is a genuine circularity, and it is the honest state of the problem.
The ways out are:

1. a writer for `verify_flag` on the device (the 310P build lacks
   `dms_custom_device.o`, but the *device* module list may contain an
   equivalent ioctl that the previous session did not find — worth one
   check);
2. a device-side code-execution path that is not the package installer —
   which is the `usershadow` route from `USERSSHADOW-ROUTE-20261028.md`, and
   which also needs a foothold;
3. the `drvHdcSendFile`-style channel writing *directly* to
   `/var/asdrv/dev0/device-sw-plugin/verify/flag` — the HDC file channel
   reaches the device's staging directory, and the whitelist is
   root-editable, so whether it can be pointed at `/var/asrv` is the one
   concrete thing left to try. It was refused for every path outside the
   trusted base, but the trusted base is now known to be extensible
   (`save_sub_path`), so this is worth one experiment.

## 6. Method note

The mistake that made this question take a while was reading a disassembly
*excerpt* and inferring a whole branch from it. The `0x2f` comparison at
`+0x18e5b0` looked like a "no signature tail" case because a nearby string
said "Package does not have a signature tail for cms verification" — but that
string belongs to a different function. Reading `GetPkgCodeLenAndCertType` in
full, and then confirming with three experiments that varied only the first
four bytes of the file, resolved it in one pass.

## 7. Card state

- device log level INFO; `/etc/hdcFileTransWhiteList.cfg` and
  `ascend_package_load.ini` restored to their original md5s;
- host AICPU package still stock; 6/6 chips OK; no NPU processes;
- the device was sent several non-stock packages during this experiment, all
  of which the device rejected before installing anything. The only
  packages that installed were byte-identical to the stock legacy package.
