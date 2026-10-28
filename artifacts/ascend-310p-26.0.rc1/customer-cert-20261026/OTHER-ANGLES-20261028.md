# Other angles, explored 2026-10-28 afternoon

**Status:** six angles examined. Two produced real results; one produced a
mechanism that reframes the problem; the rest are closed with evidence.

---

## 1. The `dsmi_cmd_set_custom_sign_*` API — the trust-store writer exists but is a stub

The host's `libdrvdsmi_host.so` exports:

```text
  dsmi_cmd_set_custom_sign_flag     dsmi_cmd_set_custom_sign_cert
  dsmi_cmd_get_custom_sign_flag     dsmi_cmd_get_custom_cert_show_info
  dsmi_cmd_set_sec_revocation       dsmi_set_sec_revocation
```

These are the DMS commands that implement the runtime trust store. The
vendor's `dms/custom/dms_custom_host.c` registers a **forward** command for
them:

```c
  ADD_FEATURE_COMMAND(DMS_CUSTOM_FORWARD_CMD_NAME, DMS_GET_SET_DEVICE_INFO_CMD, ZERO_CMD,
                      "main_cmd=0x35,sub_cmd=0x2", ...)
  ADD_FEATURE_COMMAND(DMS_CUSTOM_FORWARD_CMD_NAME, DMS_GET_SET_DEVICE_INFO_CMD, ZERO_CMD,
                      "main_cmd=0x35,sub_cmd=0x3", ...)
```

and the device side (`dms/custom/dms_custom_device.c`) registers the matching
handlers against the device's `dmp_daemon`:

```c
  "main_cmd=0x35,sub_cmd=0x2", "dmp_daemon", DMS_ACC_ROOT | DMS_ENV_NOT_NORMAL_DOCKER | DMS_VDEV_ALL
  "main_cmd=0x35,sub_cmd=0x3", "dmp_daemon", DMS_ACC_ROOT | ...
```

**So the full mechanism exists in the vendor's source, on both sides, and is
gated out of the 310P build in two places:**

- `dms/custom/dms_custom.mk` compiles `dms_custom_host.o` only for
  `hi1980b` (DKMS) or `ascend910B` (CMake) — the 310P is neither;
- the same file compiles `dms_custom_device.o` (the device-side writer) only
  for `ascend910B`.

**And the device's own `dmp_daemon` is a 26 KB minimal build** with no
custom-sign support at all — its strings contain no `main_cmd=0x35`, no
`sign_flag`, no `asdrv`. (It does have `dm_udp_init`, `dev_sendmsg_to_mcu` and
`/var/dmp/dmpdaemon_service.socket` — noted in §5.)

**Probed the exported symbols directly** (`dmp_sign_probe.c`):

```text
  set_custom_sign_flag = 0x7766e23a5c3e
  set_custom_sign_cert = 0x7766e23a606e
  get_custom_sign_flag = 0x7766e23a5997
  get_custom_cert_info = 0x7766e23b5ce0
  get_custom_sign_flag -> 8
  get_custom_cert_show_info -> 8 size=0
```

Return code **8**, not 0 — the stubs are present but do nothing. This is the
"both sides gated out" situation, confirmed at runtime rather than by reading
the makefile.

**Verdict: closed, and now closed on evidence rather than inference.**

---

## 2. The trust store is *not* the OS CA store — and the device holds a community root

`libtsdaemon.so` contains exactly three cert-file paths and one policy file:

```text
  /var/asdrv/dev<N>/device-sw-plugin/verify/vendor-certificate
  /var/asdrv/dev<N>/device-sw-plugin/verify/user-certificate
  /var/asdrv/dev<N>/device-sw-plugin/verify/community-certificate
  /var/asdrv/dev<N>/device-sw-plugin/verify/flag
```

and **no** reference to `ca-certificates.crt`, `/etc/ssl/certs`, `ca-bundle`,
`X509_STORE_set_default_paths` or `set_default_paths`. The one
`SSL_CERT_FILE` string is OpenSSL's own compiled-in default
(`.../euleros2.8/aarch64/openssl/ssl/cert.pem`) — the OpenSSL build's
default, not the verifier's store.

**Consequence: adding a root to the device's OS trust store would not make a
package verify.** The published Huawei root list (125 roots, including the
`emSign Root CA - G1` whose key was exposed) is therefore not a route.

What *is* in the device's `drv_pkicms.ko` store is five anchors, including
`CN=Huawei hisilicon ROOT CA` (the community root) with `Certificate Sign`
usage — the root a customer CA would chain to. The `user-certificate` slot
exists for exactly that. The route is unchanged: write `flag` and
`user-certificate` into `/var/asdrv/dev<N>/device-sw-plugin/verify/` on the
device.

---

## 3. The package's own `bin_hash.cfg` populates the device's hash map

**This is a mechanism result, and it matters.**

The device log for the successful type-9 install shows:

```text
[tsd_hash_verify.cpp:66][InsertOrUpdateHashValue]
    update hash value key:libtensorflow.so,     value:63eb2f59...
    update hash value key:libtf_kernels.so,     value:8d0eb713...
    update hash value key:libaicpu_kernels.so,  value:74f888c0...
    update hash value key:libcpu_kernels.so,    value:596121f4...
    update hash value key:libpt_kernels.so,     value:ffe08454...
    update hash value key:libaicpu_extend_kernels.so, value:c28b88d2...
    update hash value key:libccl_kernel.so,     value:43112ace...
    update hash value key:libccl_kernel_plf.so, value:63744faf...
```

and the stock `Ascend310P-aicpu_legacy.tar.gz`'s own `aicpu_kernels_device/bin_hash.cfg`
is:

```text
  libtensorflow.so=63eb2f59735075fef94a273f87a118b9f558bbdda802a5bf2a4bf3c53b5414ed
  libtf_kernels.so=8d0eb7134077d85378a3c11c1ed2a6f6a17de4ea2134c0696cfc7cb9e4d61e4a
  libaicpu_kernels.so=74f888c0cb887efc84543be7556d0670efcac445bcc66d445b475b01586fdd5f
  libcpu_kernels.so=596121f4443904b834d22dc88f7555aa1d278c548d7626bf69b102d013b074a3
  libpt_kernels.so=ffe084547d2c703c36e2329b77b0924018ebca728a13eda3fc4b10ef718bfb33
  libaicpu_extend_kernels.so=c28b88d2105415c44308568fde220477e49768aea280797ac91631b5590a59cf
  libccl_kernel.so=43112ace31253108a6c4e2fedfdccf29265abd9f8c7d380a33297509f85bef94
  libccl_kernel_plf.so=63744fafad0043dd67290a3cc28043292ff951eb5944382387ad79c330fd31b2
```

**Identical, value for value.** So the device's hash map is *not* an
independent trust anchor — it is populated *from the package we deliver*.

That reframes the hash map from an obstacle into a **mechanism**: the device
will happily accept a library whose hash is recorded in the package's
`bin_hash.cfg`. The only thing standing between us and a hash-accepted
library is the CMS signature over the package.

And our modified package **does** get that far: it reaches the type-9 worker
and its `bin_hash.cfg` is read — the trace shows

```text
  Start common sink package, originPkg=.../aicpu/1830147_Ascend310P-aicpu_legacy.tar.gz
  update verify flag old:1, new:1
  [CMSCBB_VERIFY] verify package[...] by cms start, isMix[0]
```

and then stops. The stock package reaches "by cms end"; ours does not.
**The wall is exactly one thing: the PKCS#7 signature over the package
content.**

---

## 4. The device's own type-9 routing accepts our name

`libtsdaemon.so` contains the routing key `-aicpu_legacy.tar.gz`
(`+0x3b0498`) and the type-9 worker is selected by the `aicpu/` directory.
Our name `1830147_Ascend310P-aicpu_legacy.tar.gz` contains that suffix, and
the device confirms the routing:

```text
  Begin to load package, type=9, path=/home/HwHiAiUser/hdcd/device0/aicpu/,
                     fileName=1830147_Ascend310P-aicpu_legacy.tar.gz
```

**So the name, the directory, the routing and the hash map are all fine. The
only failing link is the signature.**

---

## 5. `dmp_daemon`'s UDP and MCU paths — noted, not pursued

The device's `dmp_daemon` has `dm_udp_init`, `dev_sendmsg_to_mcu`,
`init_mini2mcu_connect_keeper` and `/var/dmp/dmpdaemon_service.socket`. But
neither `libtsdaemon.so` nor `libtsdclient.so` references
`dmpdaemon_service`, `/var/dmp` or `devmon`, so nothing we can reach
appears to be a client of that socket. Recorded, not pursued.

---

## 6. The path-manager member table, complete

For reference, the full `TsdPathMgr` table, recovered from the constructor at
`libtsdaemon.so+0xd16c8` (which sets `x19 = 0x4b2d50`):

```text
  bin_hash.cfg                             aicpuExtend_bin_hash.cfg
  sea_                                     ascendcpp_bin_hash.cfg
  /usr/utils/rootfs                        /var/aicpu_scheduler
  /var/aicpu_custom_scheduler              /var/queue_schedule
  /var/hccp_service.bin                    /home/CustAiCpuUser
  /home/HwHiAiUser/hs/                     runtime/bin/
  runtime/lib64/                           opp
  omfile/                                  runtime/
  ops                                      /etc/mdc/base-plat/process-manager/bin_hash.cfg
  /home/HwHiAiUser/inuse/                  comop/lib64/
  /home/HwHiAiUser/aicpu_kernels/          /home/HwHiAiUser/device-sw-plugin/
  /usr/lib64/device-sw-plugin/             aicpu_scheduler
  aicpu_custom_scheduler                   queue_schedule
  hccp_service.bin                         /usr/lib64/device-compat-plugin/
  /sys/devices/system/node/node
```

Note `/home/HwHiAiUser/inuse/` — the "in use" directory. The device's process
manager uses it (it is a `TsdPathMgr` member), and the device's
`drv_spi_smbus.ko`/`drv_smbus_intf.ko` live under it. Its ownership has
never been measured, and it is a candidate for the "device-side file write"
that the whole problem now reduces to.

---

## 7. The scheduler's `LD_LIBRARY_PATH`, in full

```text
  /usr/lib64/aicpu_kernels/0/aicpu_kernels_device/          (1st)
  /usr/lib64/aicpu_kernels/0/aicpu_kernels_device/sand_box/ (2nd)
  /home/HwHiAiUser/hs/0/runtime/lib64/                      (3rd)
  /usr/lib64/device-compat-plugin/0/                        (4th)
  /usr/lib64/device-sw-plugin/0/device-sw-plugin/           (5th)
```

All five are created at runtime (none is in the shipped rootfs). The type-9
install decompresses to `/home/HwHiAiUser/device-compat-plugin/<id>_<name>`,
which is *not* on this list — so a type-9 install does not by itself place a
library where the scheduler would load it. The type-0 install is the one that
populates `aicpu_kernels_device/` and `sand_box/`.

---

## Summary of where each angle landed

| angle | outcome |
|---|---|
| host `dsmi_cmd_set_custom_sign_*` | **closed on evidence** — exported but stubbed (returns 8); both the host and device implementations are gated out of the 310P build |
| published Huawei root list / emSign | **closed** — the device's package verifier does not use the OS CA store, and trusts none of the publicly-compromised roots |
| package `bin_hash.cfg` → device hash map | **mechanism confirmed** — the device's hash map comes from the package we deliver, so a hash-accepted library is achievable; only the CMS signature blocks us |
| type-9 name routing | **works** — our name is accepted and routed to the type-9 worker |
| `dmp_daemon` UDP/MCU | **noted, not pursued** — no reachable client of the socket |
| `TsdPathMgr` members | **complete table recovered** — including `/home/HwHiAiUser/inuse/`, an unmeasured candidate for a device-side write |

The problem is now a single wall: **one PKCS#7 signature over the package
content, verified by the device's tsdaemon.** Everything else — routing, hash
acceptance, the sandbox rename, the library search path — is already working
or already under our control.

---

## Card state

- `pkcs-enable`: `Disable` on cards 9, 8217, 16409 (the original value).
- `ssh-enable`: `True` on 9 and 8217 (previous session's change, left as
  found), `False` on 16409.
- whitelist, package-load ini, host AICPU package: original md5s.
- 6/6 chips OK, 0 errors, device log level INFO.
- The device received several of our test packages; all were rejected by the
  CMS check before installing anything. The only packages that installed were
  byte-identical to the stock legacy package.
