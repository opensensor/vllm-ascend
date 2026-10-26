# Software-only routes to the card's trust regions — assessment, 2026-10-26

Context: the MCU's electronic label and customer-certificate regions are protected by a **CRC-16
only** (proven: `mcu_elabel_forge.py` — a modified elabel passes the firmware's own verification,
21/21 items). What is missing is a *transport*. This note assesses the software-only options,
so the physical SPI clip can be a last resort rather than the first one.

## Summary

| route | status | blocker |
|---|---|---|
| **A. Device SSH** | **reachable, credential unknown** | no software vuln found; stock OpenSSH 9.6p1 |
| **B. Host→device HDC file transfer** | driver present on both sides, **client missing** | no host HDC client library; would have to write one |
| **C. SoC customer cert (`--user_cert`)** | **works today** | SoC CMS path only, not the MCU |
| **D. Device-side SPI SMBUS → MCU** | **the real transport**, present on the device | needs shell on the device (route A) |
| **E. Driver amendment (host)** | possible but **cannot add a missing transport** | the MCU link is inside the device |
| **F. Bit-bang / GPIO I2C to the MCU** | **unexplored, plausible** | needs the right GPIO/pin config |

## A. Device SSH — reachable

```
npu-smi set -t ssh-enable -i 9 -d 1
  [WARNING]: Enabling SSH will raise the safety risks.
  Status : OK
  Message: The ssh-enable of the card is set successfully. Reset card for the configuration to take effect.
```
confirmed in the host log: `Set the SSH function for card9 success. enable_flag=1.`

The device is reachable over the **PCIe vNIC**, not the LAN:

```
ssh -6 matteius@'fe80::2e0:fcff:fefc:d302%endvnic'
  -> Authorized users only. All activities may be monitored and reported.   (the device's /etc/issue.net)
  -> Permission denied (publickey,gssapi-keyex,gssapi-with-mic,keyboard-interactive)
```

The card-side vNIC addresses are `fe80::2e0:fcff:fefc:d30{2,3,4,5}` (d302 = the SSH-enabled card;
d306/d307 do not answer). `192.168.50.1` is a *different* path through the router and is blocked by
the VLAN policy — the vNIC link is the intended route.

### The device-side SSH stack (extracted from `Ascend310P.cpio.gz`)

- **OpenSSH 9.6p1**, stock, aarch64, unpatched by the vendor.
- `sshd_config`: `PasswordAuthentication no` but `KbdInteractiveAuthentication yes` and
  `ChallengeResponseAuthentication yes` — so passwords authenticate through **PAM
  keyboard-interactive**.
- PAM stack (`/etc/pam.d/sshd` → `password-auth` → `password-auth-local`):

  ```
  auth  required      pam_env.so
  auth  required      pam_faillock.so preauth audit deny=3 even_deny_root unlock_time=300
  auth  sufficient    pam_unix.so nullok try_first_pass
  -auth sufficient    pam_sss.so use_first_pass
  auth  [default=die] pam_faillock.so authfail audit deny=3 ...
  auth  sufficient    pam_faillock.so authsucc audit ...
  auth  requisite     pam_succeed_if.so uid >= 1000 quiet_success
  auth  required      pam_deny.so
  ```

  **`pam_succeed_if.so uid >= 1000 quiet_success` is notable**: `HwHiAiUser` is uid 1000, so after
  `pam_unix` succeeds it short-circuits to success. `pam_unix.so` is invoked with **`nullok`**, so an
  account with an *empty* shadow hash would authenticate. On this card it is **not** empty:

  ```
  HwHiAiUser : $6$BOrWuUA3AuseKqT.$DCnzXrNod3qQ.FVSQua43zFWRaR/p8M9U6IsrG1gWa9bAwkILcFKVW7rBoVryuTHyOK4/31zPPk43s3CMFo0i1
  root       : $6$UwIDEcxa$ubFDxGJyFZwJIjH4oInU/SD0Kn.Zqe0ramnuiG4zwhHggFfEzkyQNquSS9sngOxGLcUM5k9BEM0WG8u25gVS61
  ```

  SHA-512 crypt, 5000 rounds by default. No empty-hash bypass, no vendor patch, no known CVE in
  9.6p1's SSH implementation. 87 common/vendor-default passwords tested against both hashes — no
  match.

- Other facts: `HwHiAiUser` is restricted to `/bin/ls` via sudo; `tc-config` adds it to `wheel`;
  `pam_wheel.so` gates `su-local`; `bin/passwd` is setuid-root (normal, no escalation);
  `/etc/shadow` is `----------` root-only.

**Verdict: no software vulnerability found in the SSH implementation.** The credential is the only
gate. This is a credential-recovery problem, not a vuln.

## B. Host→device HDC file transfer — the driver is there, the client is not

The host-side HDC driver is **ours**:

```
/srv/ai/src/ascend-310p-host-driver-pre-git-restore-20260919/hdc_host/
  hdcdrv_cmd.h  hdcdrv_cmd_enum.h  hdcdrv_core.c  hdcdrv_core_com.c  hdcdrv_adapter.c ...
```

and it exposes a complete RPC to the device OS over PCIe:

```
HDCDRV_SERVICE_TYPE_FILE_TRANS 3      <-- file transfer
HDCDRV_SERVICE_TYPE_UPGRADE     17
HDCDRV_SERVICE_TYPE_DMP         0
HDCDRV_CMD_SERVER_CREATE 0x10   HDCDRV_CMD_CONNECT 0x13   HDCDRV_CMD_SEND 0x15
HDCDRV_CMD_RECV 0x17            HDCDRV_CMD_CLOSE 0x14
struct hdcdrv_cmd_send { session; src_buf; len; pool_buf; pool_addr; wait_flag; timeout; }
```

The device side runs the agent:

```
/var/hdcd                    (uid 1000, restart: on-forever, per /var/hdcd.yaml)
/usr/lib64/libhdcserver.so
/etc/hdcFileTransWhiteList.cfg
```

and the whitelist is **permissive for the AICPU kernels**:

```
//save_filename save_sub_path name_full_check_flag file_max_size file_max_num root_privilege_flag
_Ascend310P-aicpu_syskernels.tar.gz        *  0  104857600  64  0
_Ascend310P-aicpu_extend_syskernels.tar.gz  *  0  104857600  64  0
```

`save_sub_path = *` (any path), **`name_full_check_flag = 0`** (no filename check), 100 MB, 64 files,
`root_privilege_flag = 0`. The upgrade entries (`OnChipRom.bin`, `HBOOT1_a.bin`, `HBOOT1_b.bin`,
`AS610_HBOOT2_UEFI.fd`, `lpddr_mcu.bin`, ...) *are* name-checked and root-privileged, but the two
AICPU kernel tarballs are not.

**Blocker:** there is no host-side HDC *client* library installed (`find / -name "libhdc*"` → nothing).
Writing one against `hdcdrv_cmd.h` is real work but entirely within our own driver fork — this is the
same "we own the host driver" position that made the AI Core proof possible.

## C. SoC customer certificate — already working

The host log shows the earlier session's `--user_cert` path **succeeded on all six device indices**
today at 15:54:

```
UPGRADE MINOR ... Set user cert, device_index[0], file path is /usr/local/Ascend/CMS/user.xer.
UPGRADE MINOR ... User cert file_size=9948.
UPGRADE MINOR ... Set user cfg, device_index[0], size=0x20.
```

`upgrade-tool --device_index <0..5> --user_cert --path <cert>` installs a SoC customer
certificate; the current one is `/usr/local/Ascend/CMS/user.xer` (9948 bytes).

The host driver implements the same command:

```
dms/custom/dms_custom_device.c
  ADD_FEATURE_COMMAND(DMS_MODULE_CUSTOM_INFO, DMS_GET_SET_DEVICE_INFO_CMD,
      "main_cmd=0x35,sub_cmd=0x3", "dmp_daemon",
      DMS_ACC_ROOT | DMS_ENV_NOT_NORMAL_DOCKER | DMS_VDEV_ALL,
      dms_set_custom_sign_cert_dsmi)
```

**But** `dms_custom.mk` gates that file on `hi1980b` (DKMS) / `ascend910B` (CMake) — never 310P —
so `dms_custom_device.o` is **not compiled** into the 310P module. The `upgrade-tool` path works
because it goes through the SoC's own firmware update channel, not through that file.

`npu-smi set -t custom-op-secverify-cert` reports **"This device does not support setting
custom-op-secverify-cert."** — consistent.

## D. Device-side SPI SMBUS → the actual MCU transport

This is the important one, and it lives **inside the device**:

```
/var/drv_spi_smbus.ko
  description=Hisilicon SPI SMBUS driver
  alias=of:N*T*Cspi_mcuC*
  alias=of:N*T*Cspi_mcu
  spi_mcu_reg_read / spi_mcu_reg_write / spi_smbus_reg_read / spi_smbus_reg_write
  spi_smbus_send_req_crc / spi_smbus_check_msg_crc
```

The SoC drives the MCU's SPI bus directly and speaks a CRC-checked protocol to it. That is the
link the vendor's elabel path uses.

Also on the device:

```
/var/drv_smbus_intf.ko
  smbus_secure_boot_init / smbus_secure_boot_query
  smbus_secure_boot_get_cert / smbus_secure_boot_get_single_cert
  smbus_secure_boot_get_flag
  smbus_secure_boot_query_cert
  "Huawei Certification Authority1"
  "Huawei CSI RSA-PSS Code-Signing Authority"
  "Can not get cert from flash"
```

so the device can also read the **vendor's signing certificates out of flash**.

**Consequence:** the MCU is reachable from the device OS, and the device OS is reachable from the
host over HDC (route B) or SSH (route A). The full chain exists; what is missing is a way in.

## E. Can the host driver be amended to write the elabel?

Partially, and it is worth being precise about why:

- The host driver **is** ours and we have already rebuilt and reloaded it successfully (the AI Core
  proof). So driver changes are feasible.
- But the MCU link is **not** in the host driver. The host's `libascend_hal.so` reaches the MCU over
  **I2C**, and on this host there is no such bus:
  - the HAL selects the adapter by the *name* `"SMBus I801 adapter"` — an **Intel** chipset string;
  - this is an **AMD** host with `SMBus PIIX4` adapters, so the name never matches and the HAL falls
    back to `i2c-0`, the chipset's empty `i2c_designware AMDI0010:00` controller ("controller timed out");
  - the card's MCU is on no host I2C bus at all (the host buses carry only DDR4 SPD EEPROMs).
- The 310P board-comm opcodes `DEV_MON_CMD_READ_LABEL_DATA (0x15)` and
  `DEV_MON_CMD_WRITE_CLEAR_LABEL_DATA (0x14)` are **defined** in `dev_mon_cmd_def.h` but **not
  implemented** in the 310P module; `drv_devmng_host.ko` contains no label/MCU handler at all.
- The generic MCU pass-through (`dms_lpm_pass_through_mcu.o`) is gated to `ascend910`/`ascend910B`,
  and `OBJ_IMU` / ipcdrv do not exist on the 310P.

So amending the host driver can give us a **host-side command** (e.g. implement opcode 0x15), but the
packet still has to travel to the MCU over a link the SoC reaches through the *device's* SPI SMBUS,
not the host's I2C. The host driver cannot conjure that transport.

**The one amendment that would help**: implement the host-side HDC **client** (route B) so the host
can drive the device's own `spi_mbus` — i.e. use the device's transport instead of adding a second
one. That is squarely in our own code and does not need the clip.

## F. Bit-bang / GPIO path to the MCU

Not yet explored, and it is the most promising unexplored item:

```
libascend_hal.so:  dmanage_i2c_gpio_read
libascend_hal.so:  dmanage_reset_i2c_controller
libascend_hal.so:  drvResetDeviceByI2C
libascend_hal.so:  drvGetI2CDevCount
```

The HAL has **GPIO read** and **I2C controller reset** primitives, and its string
`"send I2C cmd reset:0,i2c dev:%s"` shows a software-controlled I2C path. If the SoC can bit-bang
SCL/SDA to the MCU, then a driver change *can* reach the MCU directly — which would be a genuine
software-only route to the elabel and the customer-certificate region.

This is the specific thing to investigate next: find `dmanage_i2c_gpio_read`'s caller and the GPIO
numbers it uses, and determine whether those pins go to the MCU.

## Recommended order

1. **F (bit-bang I2C)** — investigate the GPIO path in `libascend_hal.so`. If the SoC can bit-bang
   to the MCU, we get a software-only route to the CRC-protected regions, and the host driver is ours
   to change.
2. **B (HDC client)** — write a host-side HDC client against our own `hdcdrv_cmd.h` to reach the
   device OS, then drive the device's `spi_mbus` from inside. Larger effort, but no unknowns: the
   device agent, the service types, and the permissive file-transfer whitelist are all known.
3. **A (SSH credential)** — a credential-recovery problem, not a vuln; worth pursuing only if a
   password is obtainable from the vendor or from a default set not yet tried.
4. **Clip** — the known-good route; keep as the fallback rather than the first move.

---

## Follow-up results (same session)

### Route F refined: the GPIO bit-bang path is closed on the host

The host driver *does* register a GPIO command:

```
dms/status/dms_basic_info.c
  ADD_FEATURE_COMMAND(DMS_MODULE_BASIC_INFO, DMS_GET_GET_GPIO_STATUS_CMD,
      ZERO_CMD, NULL, NULL, DMS_SUPPORT_ROOT_ONLY, dms_get_gpio_status)

  dms_get_gpio_status():
      gpio_num = arg->gpio_num;
      if (!gpio_is_valid(gpio_num))   -> "Invalid GPIO. (num=%u)"
      ret = gpio_request(gpio_num, DEVDRV_GPIO_NAME);
      gpio_val = gpio_get_value(gpio_num);
      gpio_free(gpio_num);
```

with `DSMI_MAIN_CMD_GPIO 0x8002` and sub-commands
`DSMI_GPIO_SUB_CMD_GET_VALUE / SET_VALUE / DIRECT_OUTPUT / DIRECT_INPUT` — the last two being
exactly what bit-banging I2C needs.

**Measured on card 9:**

```
dsmi_get_gpio_status(0, gpio 0..15)  ->  ret=65534 (0xfffe, NOT_SUPPORT)   0/16 readable
```

So the command exists but the 310P has no GPIO block behind it. Corroborated in the source: the
only GPIO consumers are in `pcie_host/soc_adapt/res_drv_mdc_v2.c` — the **MDC** platform. There
is also **no exported `set_gpio`** in `libdrvdsmi_host.so`, so even where GPIO existed the host
API was read-only.

The device side has no GPIO or i2c-gpio modules either (`var/*.ko` contains no `gpio*` and no
i2c-gpio).

**Verdict on F: closed on the host.** A software I2C to the MCU would have to be built inside the
device's own kernel, not from the host.

### The real device→MCU transport: `ipc_drv.ko` + `drv_spi_smbus.ko`

The device's module list contains the two pieces of the chain:

```
ipc_drv.ko
  hisi-mailbox / hisi_mailbox_dev / "mailboxes" / "prop \"mailboxes\" error"
  mdev_send_msg / mdev_receive_msg / mbox_msg_send_async
  hisi_mbox_dequeue_task / hisi_mbx_get_mbox_irq_id
  "mailboxes not ready" / "no mailboxes resources."

drv_spi_smbus.ko
  description=Hisilicon SPI SMBUS driver
  alias=of:N*T*Cspi_mcuC*  /  alias=of:N*T*Cspi_mcu
  spi_mcu_reg_read / spi_mcu_reg_write
  spi_smbus_send_req_crc / spi_smbus_check_msg_crc
```

So: the SoC's **control CPU** talks to the MCU over a **mailbox** (`ipc_drv.ko`), and the MCU's
**SPI SMBUS** (`drv_spi_smbus.ko`) is what carries the elabel/cert payloads. `hisi-mailbox` is the
same family of transport the host's `dms_lpm_pass_through_mcu.o` would have used — which is why
`OBJ_IMU`/ipcdrv mattered: on the 310P **the host has no ipcdrv, but the device does**.

**This is the decisive asymmetry.** The host→device link (HDC over PCIe) works. The
device→MCU link (mailbox + SPI SMBUS) works. Only the middle — a way to make the host ask the
device to do something — is missing, and that middle is `hdcdrv` client code in *our* driver fork.

Other modules of interest on the device:
`drv_pkicms.ko` (CMS), `hsm_flash.ko`, `hsm_firmware_update.ko`, `drv_user_cfg.ko`,
`drv_upgrade.ko`, `authenc.ko`, `tzdriver.ko`, `hisi_sec2.ko`, `drv_spi_slave.ko`.

### Route B refined: what an HDC client would need

Everything needed is known and in our own tree:

```
hdc_host/hdcdrv_cmd.h         struct hdcdrv_cmd_send / recv / connect / server_create ...
hdc_host/hdcdrv_cmd_enum.h    HDCDRV_SERVICE_TYPE_FILE_TRANS 3
                              HDCDRV_SERVICE_TYPE_UPGRADE 17
                              HDCDRV_CMD_SERVER_CREATE 0x10, CONNECT 0x13, SEND 0x15, RECV 0x17
hdc_host/hdcdrv_core.c        hdcdrv_ioctl(file, cmd, arg)   -> the /dev/hisi_hdc ioctl entry
```

`/dev/hisi_hdc` on the host is `crw-rw---- HwHiAiUser HwHiAiUser` and `drv_pcie_hdc_host` is
loaded. The device agent `/var/hdcd` is running (uid 1000, `restart: on-forever`).

The file-transfer whitelist is permissive for the AICPU kernel tarballs
(`save_sub_path=*`, `name_full_check_flag=0`, 100 MB, 64 files, no root privilege required), which
means an HDC file transfer can land an arbitrary AICPU kernel payload on the device.

### Revised recommendation

1. **Write the host HDC client** against our own `hdcdrv_cmd.h` (route B). It is the only route that
   needs *no* new mechanism on either side: the host driver already exposes the ioctl, the device
   agent already runs, and the whitelist already permits the AICPU kernel path. Once we can drive the
   device, the device's own `drv_spi_smbus.ko` becomes the MCU transport — using the vendor's own
   link rather than inventing one.
2. **Then, from inside the device**, drive `spi_mcu_reg_read`/`write` (the `spi_mcu` char device)
   to reach the MCU's elabel and customer-certificate regions, whose integrity is CRC-16 only.
3. **Keep the clip as the fallback**, not the first move.
4. **The GPIO/bit-bang idea is closed** — do not spend more time there.

### The SoC↔MCU SPI link, from the device tree

Decompiled from `Ascend310P_dt.img` (esbc container, FDT at offset `0x2900`):

```dts
spi@A44A0000 {
    compatible = "spi_slave_mcu";      /* MCU -> SoC direction */
    reg = <0x00 0xa44a0000 0x00 0x10000>;
    interrupts = <0x00 0x7c 0x04>;
    mode = <0x00>;  id = <0x05>;  status = "ok";
    pinctrl-0 = <0x5a 0x5b 0x5c 0x5d>;
};

spi@A44B0000 {
    compatible = "hisi-spi";
    reg = <0x00 0xa44b0000 0x00 0x10000>;
    interrupts = <0x00 0x7d 0x04>;
    clocks = <0x5e>;  clock-names = "spi_clk";
    num-cs = <0x01>;  id = <0x06>;  status = "ok";
    pinctrl-0 = <0x5f 0x60 0x61 0x62>;

    spi_smbus@0 {
        compatible = "spi_mcu";          /* SoC -> MCU: this is the elabel link */
        reg = <0x00>;
        spi-max-frequency = <0xf4240>;   /* 1 MHz */
        mode = <0x00>;                   /* SPI mode 0 */
    };
};
```

So the SoC owns a **dedicated SPI controller at `0xA44B0000`** with an `spi_mcu` chip select, 1 MHz,
mode 0. `drv_spi_smbus.ko` binds to it via `compatible = "spi_mcu"` and exposes
`spi_mcu_reg_read` / `spi_mcu_reg_write`.

**This closes the last question about the transport.** The elabel and customer-certificate
payloads travel over *this* SPI link, and it is driven from the **device's** kernel
(`drv_spi_smbus.ko`) — not from the host, and not over I2C.

The earlier I2C conclusion therefore needs restating precisely: the *vendor's libdcmi elabel path*
is I2C-based and looks for an Intel SMBus that does not exist on this AMD host, so that particular
host-side path is dead. But the *actual* device→MCU transport is this SPI link, and it is present
and functional on the device.

### Where the device's SPI SMBUS driver can be reached from

`spi_mcu` is a **platform device**, not a `/dev/` char node, so it is reached by opening the
underlying `hisi-spi` device — and the driver stack around it is:

```
ipc_drv.ko         hisi-mailbox / mdev_send_msg / mdev_receive_msg   (SoC <-> MCU mailbox)
drv_spi_smbus.ko   spi_mcu_reg_read / spi_mcu_reg_write / crc checks (payload transport)
drv_smbus_intf.ko  smbus_secure_boot_init / _query / _get_cert       (cert access)
```

All three are on the device and all three are reachable from device-side code. The only thing
needed is the ability to execute code on the device — which is route A (SSH, credential unknown)
or route B (HDC client, ours to write).

### Summary of the chain, end to end

```
host (ours)                device (EulerOS aarch64)              MCU (Cortex-M)
-----------                -------------------------              --------------
npu-smi set -t ssh-enable   sshd  (OpenSSH 9.6p1)                  SPI flash
   |                            |                                    elabel region
   | HDC (PCIe) -- drv_pcie_hdc_host (OURS)                          PPPP region
   |   |                        |                                    trustlist
   |   |  hdcd (device agent)   |                                    trust records
   |   |                        |                                    (all CRC-16 only)
   |   |                        +-- ipc_drv.ko (hisi-mailbox) ---+
   |   |                        +-- drv_spi_smbus.ko (spi_mcu) ---+--> SPI @0xA44B0000
   |   |                                                            |
   +--> we can ADD the host-side HDC client ourselves (route B)      |
                                                                       v
                                                        forge elabel, recompute CRC-16,
                                                        write via spi_mcu_reg_write
```

**Two things are missing and only two:** (1) a host HDC *client* — pure user-space code against
our own `hdcdrv_cmd.h`; and (2) device-side code to drive `spi_mcu` — which the file-transfer
whitelist makes possible without any credentials
(`_Ascend310P-aicpu_syskernels.tar.gz  *  0  104857600  64  0`).

Neither needs the physical clip.

---

## Result: the host→device HDC file transfer works

**No client had to be written.** The host's own `libascend_hal.so` already exports the complete HDC
client API — 43 `drvHdc*` symbols — including `drvHdcSendFile`, `drvHdcSendFileV2`,
`drvHdcPull`, `drvHdcClientCreate`, `drvHdcSessionConnect`, `drvHdcGetTrustedBasePath` and
`drvHdcLoadFileTransWhiteList`. What was missing was not code but configuration and the right
incantation.

### Two clients written

`hdc_client.c` — a minimal client straight against the driver's ioctl
(`_KA_IO('H', HDCDRV_CMD_xxx)` on `/dev/hisi_hdc`, with the `union hdcdrv_cmd` layout from
`hdc_host/hdcdrv_cmd.h`). It proves the raw RPC works:

```
drvHdcGetPeerDevId(node 0) -> peer_devid = 0     (dev 0..5 -> 0,1,0,1,0,1)
HDCDRV_CMD_SERVER_CREATE   -> ok
HDCDRV_CMD_CONNECT         -> ok, session = 0
HDCDRV_CMD_CLOSE           -> ret=0
```

`hdc_push.c` / `hdc_pull.c` — wrappers over the vendor API.

### The two things that had to be right

1. **The host-side whitelist was missing.** `drvHdcLoadFileTransWhiteList()` takes *no argument* and
   opens `/etc/hdcFileTransWhiteList.cfg` itself; without it the host rejects the push with
   *"File not in the white list, quit."* The device's own copy was installed on the host
   (`/etc/hdcFileTransWhiteList.cfg`, mode 0444).

2. **`drvHdcSendFileV2` + the right destination.** `drvHdcSendFile` fails with
   `DRV_ERROR_SOCKET_CLOSE (25)`; `drvHdcSendFileV2` (documented as *"do not perform whitelist
   validation for file names"*) succeeds with **ret=0**.

### What the device accepts

| destination | result |
|---|---|
| `/home/HwHiAiUser/hdcd/device0/aicpu/vfid0` | **accepted (ret=0)** |
| `/home/HwHiAiUser/hdcd/device0/aicpu/vfid1`, `vfid2` | 25 (wrong VF for this dev_id) |
| `/home/HwHiAiUser/hdcd/device1/aicpu/vfid0` | 52 (`DST_PATH_ILLEGAL`) |
| `/home/HwHiAiUser/hdcd/device0/upgrade` | 52 |
| `/home/HwHiAiUser`, `/home/HwHiAiUser/.ssh` | 52 |

So the writable surface is **exactly the bound VF's AICPU kernel tmpfs**
(`/home/HwHiAiUser/hdcd/device0/aicpu/vfid0`), which the device mounts as:

```
mount -t tmpfs -o size=50%,uid=$uid,gid=$gid,nosuid,nodev,mode=0700 tmpfs \
     "/home/${username}/hdcd/device0/aicpu/vfid$count"
```

### The filename is NOT enforced — as the whitelist predicts

`_Ascend310P-aicpu_syskernels.tar.gz * 0 104857600 64 0` — `save_sub_path = *`,
`name_full_check_flag = 0`. Measured, all accepted with ret=0:

```
_Ascend310P-aicpu_syskernels.tar.gz        ret=0
_Ascend310P-aicpu_extend_syskernels.tar.gz ret=0
my_payload.sh                              ret=0
authorized_keys                            ret=0     <-- name check is off
evil.tar.gz                                ret=0
```

And the **genuine 53 MB** `Ascend310P-aicpu_syskernels.tar.gz` transfers with ret=0, so this is
not a small-file artefact.

### Why this is the right target

That directory is exactly where the device expects its **AICPU kernels** — the same tarball the
vendor deploys. `/usr/lib64/aicpu_kernels/` on the device holds:

```
aicpu_kernel.json  aicpu_tf_kernel.json  libaicpu_kernels.so  libcpu_kernels.so  libdvpp_kernels.so
```

and the AICPU kernels are **plain ARM64 ELF shared objects loaded by `dlopen`** with no signature
(the same conclusion reached earlier for the host's CANN copy). So a substituted file here is
**device-executed code on the AI CPU**, reached through a vendor-supported, no-credential path.

This closes the gap that session 5 identified: the host→device link (HDC over PCIe) works, the
file-transfer whitelist permits the AICPU kernel path, and the destination is the AICPU kernel
directory. What remains is to establish that the substituted tarball is actually *consumed* from
that directory — i.e. that the device unpacks and `dlopen`s it — which needs either device-side
code execution or observation of the device's AICPU kernel load.

### Note on the SSH lockout

`pam_faillock` on the device is `deny=3 unlock_time=300`, so repeated wrong-password attempts lock
the account for 5 minutes per card — it recovers on its own, but three cards are available to
rotate between (NPU ID 9, 8217, 16409; `ssh-enable` is now set on 9 and 8217).

`drvHdcPull` still returns 25, so the file cannot be read back yet; that is the outstanding
verification step.

---

## The AICPU kernels container is rebuildable — the substitution chain closes

The HDC push lands in exactly the directory the device expects its AICPU kernels in. So the
question becomes: can a substituted AICPU kernel be produced that the device will accept?

### The AICPU kernels container has no signature

```
/usr/local/Ascend/cann-9.1.0/opp/Ascend310P/aicpu/Ascend310P-aicpu_syskernels.tar.gz
  esbc container: magic 0x55aa55aa, head_len 0x4dc, cert_type 0xffffffff
  payload at 0x2000: "3\xaa::aicpu_kernels" + gzip(tar)
    aicpu_kernels_device/
      libaicpu_kernels.so    8267928
      libcpu_kernels.so     30251592
      libpt_kernels.so      20470976
      libtensorflow.so      105180752
      libtf_kernels.so       3635704
      bin_hash.cfg              414
      version.info             1056
```

`cert_type` is `0xffffffff` and the CMS section start lies past EOF — **there is no certificate and
nothing to forge**. The container's entire integrity mechanism is three plain SHA-256 digests:

```
+0x02C  u8[32]  payload_sha256 = SHA-256(payload)
+0x45C  u32      code_len        = payload length
+0x4BC  u8[32]  head_sha0       = SHA-256(container[0:0x4BC])
+0x14BC u8[32]  head_sha1       = SHA-256(container[0x1000:0x14BC])
```

and inside the payload, `bin_hash.cfg` holds one SHA-256 per library. Verified to be exactly
`sha256(file)`:

```
libtensorflow.so     cfg=63eb2f59735075fe... sha256=63eb2f59735075fe... MATCH
libtf_kernels.so     cfg=8d0eb7134077d853... sha256=8d0eb7134077d853... MATCH
libaicpu_kernels.so  cfg=74f888c0cb887efc... sha256=74f888c0cb887efc... MATCH
libcpu_kernels.so    cfg=596121f4443904b8... sha256=596121f4443904b8... MATCH
libpt_kernels.so     cfg=ffe084547d2c703c... sha256=ffe084547d2c703c... MATCH
```

The libraries themselves are **plain unsigned ARM64 ELF shared objects** (`libaicpu_kernels.so`:
`ELF 64-bit LSB shared object, ARM aarch64, stripped`, exporting `Cast`, `CropAndResize`, `CTCLoss`,
`DataEnqueue`, …). `version.info` carries a `Trustlist=` naming the libraries the device will load.

### The substitution chain, proven end to end

`rebuild_aicpu_kernels.py` substitutes a library and rebuilds the container:

1. replace the member in the tar;
2. recompute `bin_hash.cfg`;
3. re-gzip into the `3\xaa::aicpu_kernels` payload;
4. rewrite `code_len`;
5. recompute `payload_sha256`, `head_sha0`, `head_sha1`.

The test substituted `libaicpu_kernels.so` with a copy carrying an equal-length marker
(`construct null not valid` → `AICPUPROOF-AX-0000000XXX`, same 24 bytes at the same offset, so the
library stays structurally valid):

```
=== original container ===
  payload sha256                             OK
  head sha c0                                OK
  head sha c1                                OK
  bin_hash.cfg (per-library sha256)          OK
  cert_type == 0xffffffff (unsigned)         OK

=== rebuilt container ===
  payload sha256                             OK
  head sha c0                                OK
  head sha c1                                OK
  bin_hash.cfg (per-library sha256)          OK
  cert_type == 0xffffffff (unsigned)         OK

  libaicpu_kernels.so in the rebuilt tar: 8267928 bytes,
    sha256=a98a83664963c6d36feef574ef9b69c6c153fb5fc4585577d665034e7423c6a1
  marker_present=True
```

and independently re-verified with the pre-existing `esbc_info.py`:

```
=== /tmp/kilo/aicpu_proof.tar.gz  (53572732 bytes) ===
  code_len     : 0x331547c (53564540)
  payload sha  : OK
  head sha c0  : OK
  head sha c1  : OK
  cert_type    : 0xffffffff
```

**A substituted AICPU kernel produces a container that passes every integrity check the vendor
defines.** This is the same conclusion the AI Core work reached from the other side — but here it
needs no key, no certificate and no privileged access: the whole trust anchor is a checksum we
recompute.

### The chain, complete

```
host (ours)                          device (EulerOS aarch64)            AI CPU (Cortex-A55)
---------                            -------------------------            ------------------
HDC over PCIe (ours)                 hdcd, service 3 (file trans)
  drvHdcSendFileV2  ret=0               whitelist: *_aicpu_*_syskernels.tar.gz
  dst = .../aicpu/vfid0                  save_sub_path=*, name_full_check=0
                                         -> file lands in the AICPU tmpfs
                                                                       AICPU kernel
rebuild_aicpu_kernels.py                 unpacked to /usr/lib64/aicpu_kernels/   dlopen'd,
  bin_hash.cfg recomputed                   libaicpu_kernels.so, ...           no signature
  container digests recomputed          bin_hash.cfg checked: sha256 only
```

**Everything needed is in our hands**: the transport (our driver, the vendor's own API, the
whitelist), the payload format (public), the integrity check (a checksum), and the destination (the
directory the device reads). The one step not yet observed is the device *consuming* the
substituted tarball — that needs either device-side execution or a load-time observation, and
`drvHdcPull` still returns 25 so the file cannot be read back yet.

Note the credential situation is no longer a blocker for this chain: the HDC path needs no
password at all.

---

## The AICPU package deployment path, end to end

The goal is to place something on the card that we can execute, so that we get device-side code
execution (and from there, SSH and the MCU). The vendor's own mechanism for delivering AICPU
kernels turns out to be exactly this, and it is **not key-protected**.

### What the vendor does, traced through the code

**Host side** — `CANN/x86_64-linux/lib64/libtsdclient.so` (`tsd::PackageWorker`):

```
[TsdClient][deviceId=%u] can not find aicpu packages
[TsdClient][deviceId=%u] aicpu package is not existed, skip send package
[ProcessModeManager][deviceId=%u] aicpu package already exist in device
[ProcessModeManager][deviceId=%u] init failed for send aicpu package
[TsdClient][deviceId=%u] send aicpu package to device failed
[TsdClient] after lock, the checksum of aicpu package[%u] is same as device[%u], skip send
[TsdClient][deviceId_=%u] the checksum of host package[%u] is the same as device[%u], skip send package.
[TsdSha256] Using x86 SSSE3+BMI2 accelerated path.
```

`tsd::ProcessModeManager` sends the AICPU package, the "check code" is a **SHA-256** (`tsd_sha256.cpp`),
and the host **skips the send if the device's checksum already matches**.

**Device side** — `/usr/lib64/libtsdaemon.so` (`tsd::TSDaemon`):

```
sessionId[%u], msgType[%u], host hash code[%s], device hash code[%s], package name[%s], rspCode[%u]
```

The device receives the host's hash code, recomputes its own over the received file, and compares.
**There is no certificate, no vendor key and no signature in this path** — the integrity of the
AICPU package is a SHA-256 that the host supplies and the device recomputes.

The device then decompresses and loads it:

```
tsd::TsdPathMgr::BuildAicpuPackageRootPath
tsd::AicpuProcessPackageWorker::CopyExtendSoToCommonSoPath
tsd::AicpuPackageProcess::CopyExtendSoToCommonSoPath
tsd::BasePackageWorker::DecompressPackage / MoveOriginPackageToDecompressDir
TSDaemon::ProcessOmFileDecompressStatus
```

and the resulting libraries are `dlopen`ed:

```
/usr/lib64/libtsdaemon.so : /usr/lib64/aicpu_kernels/libaicpu_kernels.so
                              /usr/lib64/aicpu_kernels/libcpu_kernels.so
                              /usr/lib64/aicpu_kernels/libtensorflow.so
/usr/lib64/libtdtserver.so : /aicpu_kernels_device/libaicpu_kernels.so
```

The op dispatcher names the kernel shared object explicitly — `aicpu_kernel.json`:

```json
"Acos": { "opInfo": { "engine": "DNN_VM_AICPU", "functionName": "RunCpuKernel",
                      "kernelSo": "libcpu_kernels.so", "opKernelLib": "AICPUKernel" } }
```

### The whole chain, and where each piece stands

| step | status |
|---|---|
| rebuild the AICPU container with our library | **done** — every vendor digest recomputes and validates |
| push it to the card's AICPU kernel directory | **done** — `drvHdcSendFileV2` ret=0 into `.../hdcd/device0/aicpu/vfid0` |
| make the device decompress + load it | **the open step** |

The device decompresses on receipt of the package message, so the remaining question is how to
make it take the package we pushed. The vendor's own sender does this; we can either drive the
same host-side library (`libtsdclient.so` is present in CANN and exports the whole `tsd::`
API) or reproduce its message.

### What we would need our payload to do

`libcpu_kernels.so` and `libaicpu_kernels.so` export ordinary kernel entry points
(`Cast`, `CropAndResize`, `CTCLoss`, `DataEnqueue`, …) loaded by `dlopen` into the device's
EulerOS user space as uid 1000. A substituted library therefore runs as **uid 1000 in the device
OS**, which is exactly `HwHiAiUser`'s privilege level — the account that owns `/dev/hisi_hdc` and
`/dev/davinci_manager` on the *host*.

So the goal the user described — "put something there we could execute that would elevate our
access to have like ssh basically" — is reachable by this route: a library that runs at AICPU
kernel-load time in the device OS, with uid 1000 and access to the device's `/dev` nodes.

Two things to note honestly:

1. **This does not change the boot chain.** The SoC bootloader and the MCU firmware remain
   RSA-signed against a vendor key we do not hold, so the card still boots the vendor's firmware.
   What this route adds is *post-boot code execution on the device OS* — which is a different and
   useful thing, and is the step that would let us drive `spi_mcu` and reach the CRC-only MCU
   regions.
2. **An aarch64 toolchain is needed** to build the payload, and the host has no cross-compiler
   (`aarch64-linux-gnu-gcc` absent). That is a tooling gap to fill, not a blocker in principle.

---

## The vendor's own AICPU load path, driven end to end

Rather than reimplement the protocol, the vendor's own client in CANN can be driven directly.
`CANN/x86_64-linux/lib64/libtsdclient.so` exports the whole supported API, and its header
`x86_64-linux/pkg_inc/aicpu/tsd/tsd_client.h` documents it:

```c
uint32_t TsdOpenAicpuSd(const uint32_t logicDeviceId);
uint32_t TsdOpen(const uint32_t logicDeviceId, const uint32_t rankSize);
uint32_t TsdFileLoad(const uint32_t logicDeviceId,
                     const char *filePath, const uint64_t pathLen,
                     const char *fileName, const uint64_t fileNameLen);
uint32_t TsdClose(const uint32_t logicDeviceId);
```

### What actually happens, from the client's own log

`tsd_load` logs to `/root/ascend/log/run/plog/` (it runs as root). The run:

```
[TsdClient][deviceId=0] begin to init hdc client
VersionVerify: send client version to server
device plugin info rsp, pkgCount:0
[TsdClient][deviceId=0] [sessionId=1] start hccp and computer process success,
    phase1:[1]ms load sink package config to device, phase2:[0]ms load opkernel to device,
    phase3:[0]ms load sink package to device, phase4:[0]ms send open message to device,
    phase5:[558]ms receive open message to device, whole duration:[559]ms
loadfile name:Ascend310P-aicpu_syskernels.tar.gz
begin load file:Ascend310P-aicpu_syskernels.tar.gz
file path is:/usr/local/Ascend/cann-9.1.0/opp/Ascend310P/aicpu
[TsdClient][deviceId=0][pathLen=49] SendFileToDevice enter
[TsdClient][deviceId=0] hdc send file[/usr/local/.../Ascend310P-aicpu_syskernels.tar.gz]
    to [/home/HwHiAiUser/hdcd/device0/307644_Ascend310P-aicpu_syskernels.tar.gz] success
[TsdClient][deviceId=0] [sessionId=1] close hccp and computer process success
```

**Our rebuilt container was accepted and sent to the device**, using the vendor's own client,
with no credential. Note the destination: `hdcd/device0/<pid>_<name>` — the device's `tsdaemon`
then decompresses that file into `aicpu_kernels_device/` and installs it.

### The device answers the capability query

`ProcessModeManager::IsOkToLoadFileToDevice()` consults `IsSupportHeterogeneousInterface()`,
a device-side capability bitmask. Queried directly with `TsdCapabilityGet`:

```
type 0 (PIDQOS)         : ret=0 value=0x0
type 1 (LEVEL)          : ret=0 value=0x13f
type 2 (OM_INNER_DEC)   : ret=0 value=0x1     <-- OM inner decompression: supported
type 3 (BUILTIN_UDF)    : ret=0 value=0x1
type 4 (DRIVER_VERSION) : ret=0 value=0x0
type 5 (ADPROF)         : ret=0 value=0x1
type 6 (MUTIPLE_HCCP)   : ret=0 value=0x1
```

So the device supports the OM inner-decompression capability the AICPU package load needs.

### The remaining gap

The client sent the file and then **waited ~3 minutes for the device's reply and got none** —
no `host hash code` / `device hash code` comparison ever appeared, and the device-side
`ProcessOmFileDecompressStatus` never ran. So the file *arrived* but the device did not *act* on it.

Two candidate reasons, both testable:

1. **The device-side handler expects a different file name.** The vendor client names the file
   `<pid>_<packagename>`, and the device's `hdcd` service (service type 3, `arguments: [3]` in
   `hdcd.yaml`) matches on that name. Our manual `hdc_push` used the bare name, which the
   whitelist permits but the *device-side* matcher may not.
2. **The device-side `hdcd` needs to be in the right state.** The client's own log shows
   `can not find aicpu packages` and `Opening file: was not successful` for the *host-side*
   `aicpu_package_install.info`, which is a separate descriptor the device reads to know what
   to install. That file does not exist on the host either — it is written by the install
   tooling, not by the runtime.

Neither has been excluded yet, so the honest status is: **the package is delivered and
accepted by the transport; the device-side install step has not been observed to run.**

### What was built for this

| file | purpose |
|---|---|
| `rebuild_aicpu_kernels.py` | substitutes a library and rebuilds the container so every vendor digest validates |
| `inject_ctor.py` | injects an ELF `.init_array` constructor into a stripped shared object without relinking it (extends the RX segment, patches `DT_INIT_ARRAY`/`DT_INIT_ARRAYSZ`) |
| `probe_init.s` | an ARM64 constructor that writes a marker file, so a loaded library is provably executed |
| `tsd_load.c` | drives the vendor client (`TsdOpenAicpuSd` / `TsdOpen` / `TsdFileLoad`) |
| `cap_probe.c` | queries the device's TSD capability set |
| `hdc_push.c` / `hdc_pull.c` | raw HDC file transfer against `/dev/hisi_hdc` |
| `hdc_client.c` | a minimal client straight against the driver's ioctl, proving the RPC independently |

The aarch64 toolchain needed to build the payload is present on the host
(`aarch64-linux-gnu-as` / `-ld` / `-objcopy` / `-readelf`); there is no aarch64 gcc, which is why
the constructor is written in assembly.

Card state: stock AICPU container restored, 6 davinci nodes, 6/6 chips OK.

---

## Where the AICPU op actually runs — a locality test that changed the picture

The obvious oracle for "did the device install our library?" is an AICPU op's numeric output.
`aicpu_kernel.json` lists 233 ops with `"kernelSo": "libcpu_kernels.so"`, and the card has
**7 AI CPUs active** (`npu-smi info -t cpu-num-cfg`: `Current AI CPU number: 7`).

Test plan: replace `libcpu_kernels.so` with a copy whose exported `RunCpuKernel` is patched to
`mov x0, #0 ; ret` — every AICPU op would then return nothing. If the op's result changes, the
library in question is the one being executed.

But first, a control: **is the AICPU op even running on the device?** The host has its own copy:

```
/usr/local/Ascend/cann-9.1.0/opp/built-in/op_impl/host_aicpu/libcpu_kernels.so
    RunCpuKernel @ 0x1844630
```

So the same library exists on both sides. I patched the **host** copy's `RunCpuKernel` to the same
no-op and re-ran `torch.acos`:

```
host library patched  ->  acos got: [1.047197699546814, 1.3181161880493164]
                         acos expected: [1.0471975511965979, 1.318116071652818]
                         MATCHES STOCK: True
```

**The result did not change.** So `torch.acos` is *not* using the host's
`libcpu_kernels.so` — and by the same token, the fact that it also did not change when the
*device's* copy was patched to a no-op means **the op did not go through `RunCpuKernel` at all**
in that path, or the device-side library was never installed.

Both patches were reverted immediately and verified:

```
host libcpu_kernels.so   ec28199efe1a5060a73ca951663be6f546e403c7a  (restored)
AICPU container          663e3dcc22b9e8a5d4c7884ebafb87987bb4f0d6  (stock)
acos                    stock-ok: True
```

### What this establishes

The AICPU *op dispatch* on this stack does not go through the `RunCpuKernel` entry point in
either library, so patching it was the wrong oracle. The op is likely dispatched through the
CANN op master (`AICPUKernel` / `libopmaster`) rather than the raw kernel library, or the
AICPU path in this configuration is served by a different mechanism (the `opKernelLib` field in
the JSON points at `AICPUKernel`, not at `RunCpuKernel` directly).

### The state of the AICPU route, stated plainly

| step | status |
|---|---|
| rebuild the container with a substituted library, every vendor digest recomputes and validates | **done** |
| the vendor's own client accepts our container and sends it to the device | **done** (`hdc send file ... success`) |
| the raw HDC push accepts it into the device's AICPU tmpfs | **done** (`drvHdcSendFileV2 ret=0`) |
| the device *installs* it (decompress → `aicpu_kernels_device/` → `/usr/lib64/aicpu_kernels/`) | **not observed** |
| the device *executes* our code | **not observed** |

The blocker is now specific and not a transport problem: **the device-side install step does not
fire.** And the control test proves this is *not* caused by our substitution — the vendor's own
**stock** package behaves identically (sent successfully, then the device never replies; the
client waits ~3 minutes and gives up).

The device-side code path is known from the strings and it is not being entered:

```
Move om package to decompress dir failed, ret=%d
Decompress om pkg failed, ret=%d
Post process om pkg failed, ret=%d
Load om package success, originPkg=%s, decomPkg=%s, verifyName=%s, checkCode=%lu
```

plus the matching pattern it looks for: `/*_Ascend*-aicpu_syskernels.tar.gz`, and the
destination it installs to: `aicpu_kernels_device/` → `/usr/lib64/aicpu_kernels/`.

The remaining unknown is *which* step fails first on the device, and the device's own log
(`[TSDaemon] Checksum aicpu package failed, ret=%u` / `Check aicpu package in os[%u] vfId[%d]
checkCode[...]`) is not visible from the host. **Reading the device's log is now the blocking
requirement**, and that is what the SSH credential would give us — which makes the SSH angle the
practical unblocker rather than a parallel curiosity.

### A side finding worth keeping

The device's certificate-plugin paths are in the same binary:

```
/device-sw-plugin/verify/community-certificate
/device-sw-plugin/verify/user-certificate
/device-sw-plugin/verify/vendor-certificate
/device-sw-plugin/verify/flag
```

So the device OS *does* implement the runtime customer-certificate mechanism that the host-side
`dms_custom` code was 910B-gated away from. Reaching it needs code execution on the device,
which is exactly what this route is for.

---

## The AICPU install handshake, fully traced — and why the tar-slip idea is a dead end

### The tar-slip idea: tested and closed

The device's extraction command (from `libtsdaemon.so`, the device-side TSD daemon) is:

```
mkdir -p <dir> ; tar --no-same-owner -ozxf <file> -C <dir> ; rm -f *.tar.gz
```

with the sibling form `cp '<src>' '<dst>'` for the move step, executed via **popen**
(`Popen execute cmd[%s]`). Since the tarball is ours, a tar-slip (writing outside the extraction
directory) looked attractive.

**It does not work.** GNU tar's default protections apply, and I confirmed all three variants
against the exact options the device uses:

| technique | result |
|---|---|
| absolute member path | `tar: Removing leading '/' from member names` — lands inside the extract dir |
| `../` traversal | `tar: ../escaped.txt: Member name contains '..'` — **refused, rc=2** |
| symlink then write through it | `tar: esc/planted.txt: Cannot open: Not a directory` — even when the symlink target **exists** |

The third was the serious one and it still fails: GNU tar refuses to follow a symlink that was
itself just extracted, so the write cannot be redirected. `--no-same-owner` only affects
ownership, not path handling. **Closed.**

### The real handshake, and why it does not complete

Reading both sides of the protocol (`libtsdclient.so` on the host, `libtsdaemon.so` on the
device — same code, different build) gives the complete sequence:

1. **host** opens the session: `TsdOpenAicpuSd` → 0, `TsdOpen` → 0, sessionId=1.
2. **host** sends the package: `hdc send file[.../Ascend310P-aicpu_syskernels.tar.gz] to
   [/home/HwHiAiUser/hdcd/device0/<pid>_Ascend310P-aicpu_syskernels.tar.gz] success`.
3. **host** must then send `tsd.HDCMessage.omfile_name` — the device logs
   `omfileName[%s], deviceId[%u], vfId[%u], hostPid[%u].`
4. **device** moves and extracts it:
   `Move origin package to decompress path` → `Start decompress package, cmd=%s` →
   `Post process of aicpu process package success` → `Load om package success, originPkg=%s,
   decomPkg=%s, verifyName=%s, checkCode=%lu`.
5. **device** reports back: `Send GetOmFileDecompressStatus success, sessionId[%u] rspCode:%u`.

The device's filename matcher is a regex, and our name satisfies it:

```
^[0-9]+_Ascend([0-9]{3}(P)?)?-aicpu_syskernels\.tar\.gz$
^[0-9]+_Ascend([0-9]{3}(P)?)?-aicpu_extend_syskernels\.tar\.gz$
```

and the device watches, via inotify + epoll:

```
/home/HwHiAiUser/hdcd/device<N>/Ascend310P/aicpu/vfid<N>
```

The **host** is the side that assembles the device-side path and issues the extraction command
(host `libascend_hal.so` carries `/aicpu`, `/home/HwHiAiUser/hdcd/`, `/home/HwHiAiUser/hdcd/device`,
`/var/hdcd`; the device carries `Ascend310P/aicpu/`, `/aicpu/vfid`, `aicpu_kernels_device/`,
`/usr/lib64/aicpu_kernels/`).

**Step 3 is where it stops.** The client sends the file and then waits for
`TSD_OM_PKG_DECOMPRESS_STATUS`; the device never answers, so `TsdFileLoad` blocks until the
3-minute timeout and the client gives up.

### The control result that isolates the cause

This was tested with the **vendor's own stock package** through the identical path:

```
[TsdClient][deviceId=0] hdc send file[.../Ascend310P-aicpu_syskernels.tar.gz]
    to [/home/HwHiAiUser/hdcd/device0/340736_Ascend310P-aicpu_syskernels.tar.gz] success
```

and it behaves **identically** — sent successfully, then the device never replies. It was also
tested with a deliberately broken package (`RunCpuKernel` patched to `mov x0,#0 ; ret`), with the
same result.

**So the device-side install step does not fire for the vendor's own package either.** This is a
property of this card/device configuration, not a consequence of our substitution. That rules out
the whole "our payload is malformed" family of explanations.

### The one thing that would settle it

The device's own log names the failing step precisely —

```
[TSDaemon] Checksum aicpu package failed, ret=%u
[TSDaemon] Check aicpu package in os[%u] vfId[%d] checkCode[%lu]
Load om package success, originPkg=%s, decomPkg=%s, verifyName=%s, checkCode=%lu
```

— but the device's log is **not visible from the host**. The host syslog only carries host-side
messages; the device's journal is a separate filesystem. Reading it requires device-side access,
which is exactly what the SSH credential would provide.

### Host-path rules learned from the driver's own log

The host HAL's `drvHdcSendFileParaCheck` (hdc_file.c:1416) enforces:

```
Calling strstr failed. (dev_id=0; file="..."; dst_path="/home/HwHiAiUser/hdcd/device0/omfile";
                        base_path="/home/HwHiAiUser/hdcd/device0/aicpu")
```

i.e. **`dst_path` must contain the trusted base path `/home/HwHiUser/hdcd/device0/aicpu`**.
Measured acceptance:

| dst_path | result |
|---|---|
| `/home/HwHiAiUser/hdcd/device0/aicpu` | accepted |
| `/home/HwHiAiUser/hdcd/device0/aicpu/vfid0` | accepted |
| `/home/HwHiAiUser/hdcd/device0/Ascend310P/aicpu/vfid0` | rejected (52) |
| `/home/HwHiAiUser/hdcd/device0/omfile` | rejected (52) |
| `/var/hdcd/...` | rejected (52) |
| `/home/HwDmUser/hdcd/device0/upgrade` | rejected (52) |

So `/var/hdcd` is **not** an alternative landing zone on this platform — it is a fallback
*base* string in the HAL, and the whitelist's `save_sub_path = *` means "into the base path",
not "anywhere".

Also established: the filename is **not** checked (`name_full_check_flag = 0`); `_Ascend310P-
aicpu_syskernels.tar.gz`, `my_payload.sh`, `authorized_keys` and `evil.tar.gz` were all accepted
with ret=0.

### Where this leaves the two open questions

1. **Why does the device not reply?** The device log would say (`Checksum aicpu package failed`
   vs `Move origin package to decompress path failed` vs `Decompress om pkg failed`). Needs
   device-side visibility.
2. **Does our library get executed if the install succeeds?** Not testable until (1) is answered;
   the numeric oracle also turned out to be wrong, because the AICPU op dispatch goes through
   `opKernelLib: AICPUKernel` in the CANN op master rather than the `RunCpuKernel` entry point
   in the kernel library.

### A note on the dev_id failures in the log

The `Convert virtual id to physical id failed` / `uda_devid_to_phy_devid` errors in the host
kernel log are **not** a bug in our pushes — they are the *other cards*. `npu-smi info -m` gives the
mapping:

```
NPU ID 9     Chip 0,1  -> Chip Logic ID 0,1
NPU ID 8217  Chip 0,1  -> Chip Logic ID 2,3
NPU ID 16409 Chip 0,1  -> Chip Logic ID 4,5
```

so `dev 2`/`dev 4` address cards 1 and 2, whose HDC sessions are not up in the same way. Card 0
(`dev 0`) is the one that works.

---

## Breakthrough: the `latest` symlink, and what it unlocked

The client log had a warning I had read past twice:

```
[package_process_config.cpp:106][GetHostFilePath] cannot get package path:/usr/local/Ascend/latest///conf/,
    reason:No such file or directory
[TsdClient][logicDeviceId_=0] can not find aicpu packages
```

**`/usr/local/Ascend/latest` did not exist on the host.** The driver install creates
`cann -> cann-9.1.0` and `cann-9.1.0/conf -> x86_64-linux/conf`, but no `latest`. Creating the
symlink changes the client's behaviour materially:

```
$ sudo ln -sfn /usr/local/Ascend/cann-9.1.0 /usr/local/Ascend/latest
```

and the same `TsdFileLoad` call now sends **three** packages instead of one:

```
[SendAICPUPackageSimple] hdc send file[/usr/local/Ascend/latest//opp/Ascend310P/aicpu/Ascend310P-aicpu_legacy.tar.gz]
                              to [/home/HwHiAiUser/hdcd/device0/aicpu/443641_Ascend310P-aicpu_legacy.tar.gz] success
[SendAICPUPackageSimple] hdc send file[/usr/local/Ascend/latest//opp/built-in/op_impl/aicpu/kernel/aicpu_hcomm.tar.gz]
                              to [/home/HwHiAiUser/hdcd/device0/aicpu/443641_aicpu_hcomm.tar.gz] success
[SendFileToDevice]        hdc send file[/usr/local/Ascend/cann-9.1.0/opp/Ascend310P/aicpu/Ascend310P-aicpu_syskernels.tar.gz]
                              to [/home/HwHiAiUser/hdcd/device0/443641_Ascend310P-aicpu_syskernels.tar.gz] success
```

The symlink is safe: `npu-smi info` and a torch op both still work, and it is a symlink to the
already-current CANN.

### The exact host-side path rule (from `drvHdcSendFileParaCheck`)

The check is `strstr(base_path, dst_path)` — **`dst_path` must be a substring of `base_path`**, where
`base_path = <root>/hdcd/device<id><sub_path>` and `sub_path` comes from the whitelist.
Measured:

| dst_path | result |
|---|---|
| `/home/HwHiAiUser/hdcd/device0/aicpu` | accepted |
| `/home/HwHiAiUser/hdcd/device0/aicpu/vfid0` | accepted |
| `/home/HwHiAiUser/hdcd/device0` | rejected (52) |
| `/home/HwHiAiUser/hdcd/device0/omfile` | rejected (52) |
| `/home/HwHiAiUser/hdcd/device0/Ascend310P/aicpu/vfid0` | rejected (52) |
| `/home/HwHiAiUser/hdcd/device0/../` | rejected (52) |
| `/home/HwHiAiUser/hdcd/device0/x/../../../../tmp` | rejected (52) |
| `/var/hdcd/...` | rejected (52) |
| `/home/HwDmUser/hdcd/device0/upgrade` | rejected (52) |

So `/var/hdcd` is **not** an alternative landing zone — it is a fallback *base* string in the HAL,
and the whitelist's `save_sub_path = *` means "into the base path", i.e. `.../hdcd/device0/aicpu`.
Path traversal is rejected because `dst_path` must be a literal substring of the base.

### What the device does with the file

The device-side `libascend_hal.so` saves it as:

```
Invalid len to save file. (len=%u, path=%s)
Format full path failed. (dir=%s, fileName=%s)      <- the device derives dir + fileName
Open failed. (errno=%s, path=%s)
Write failed. (errno=%s, path=%s)
Write file return bad write length. (path=%s, ret=%d, len=%u)
Sync failed. (errno=%s, path=%s)
```

with roots `/home/HwHiAiUser/hdcd/device` and `/home/HwDmUser/hdcd/device`.

And the device's TSD daemon (`libtsdaemon.so`) does the extraction and the install:

```
Move origin package to decompress path      (cp '<src>' '<dst>')
Start decompress package, cmd=%s            (mkdir -p <dir> ; tar --no-same-owner -ozxf <f> -C <dir>)
Post process of aicpu process package success
Load om package success, originPkg=%s, decomPkg=%s, verifyName=%s, checkCode=%lu
```

and it matches filenames against:

```
^[0-9]+_Ascend([0-9]{3}(P)?)?-aicpu_syskernels\.tar\.gz$
^[0-9]+_Ascend([0-9]{3}(P)?)?-aicpu_extend_syskernels\.tar\.gz$
^[0-9]+_Ascend-runtime_device-minios\.tar\.gz$
^[0-9]+_Ascend-device-sw-plugin\.tar\.gz$
```

watching `/home/HwHiAiUser/hdcd/device<N>/Ascend310P/aicpu/vfid<N>` via inotify + epoll.

### The device-side customer-certificate mechanism (new)

The device daemon also contains a **complete CMS/PKCS#7 verifier** (OpenSSL 3.0.12) and the
runtime certificate plugin paths:

```
/device-sw-plugin/verify/community-certificate
/device-sw-plugin/verify/user-certificate
/device-sw-plugin/verify/vendor-certificate
```

with `[CMSCBB_VERIFY]` logging (`cmsLen[%u], iniLen[%u], crlLen[%u]`, `Add config certificate not
pass ret[0x%x]`, `add crl not pass, ret[0x%x]`) and `only support signed type package by cms`.

So the **device** implements the runtime customer-certificate installation that the host-side
`dms_custom` code was 910B-gated away from — and the host's own driver header points at exactly
these paths:

```
CUSTOM_SIGN_CONF_USER_CERT_PATH "/var/asdrv/dev%u/device-sw-plugin/verify/user-certificate"
DSMI_SEC_SUB_CMD_CUST_SIGN_USER_CERT 3
```

The device-side handler is reached through `halVerifyImg` and the package types
`Ascend-device-sw-plugin.tar.gz` / `Ascend-runtime_device-minios.tar.gz`.

**That is the path to the customer certificate**: it is installed by delivering a **CMS-signed**
device-sw-plugin package, and the device verifies it with OpenSSL. The signature is checked
against the vendor's trust root — but note it is the *device's* OpenSSL, and the certificate
chain check is the thing to examine (`Add config certificate not pass ret[0x%x]`).

### Where the install still stops

Even with the `latest` symlink, the three packages are sent successfully and the device still does
not answer the `TSD_OM_PKG_DECOMPRESS_STATUS` request, so `TsdFileLoad` blocks. The host's
`LoadOmFileToDevice` log sequence shows the intended order:

```
Load om file name:%s to device logic device id:%u
send TSD_OM_PKG_DECOMPRESS_STATUS msg failed.
Wait TSD_OM_PKG_DECOMPRESS_STATUS response from device failed.
```

The device-side counterpart is `GetOmFileDecompressStatus` (device asks the host) and
`ProcessOmFileDecompressStatus` (device replies), with the filename carried in
`tsd.HDCMessage.omfile_name`. The public header
`pkg_inc/aicpu/tsd/tsd.h` documents the TSD event ids (0..44) but **not** the OM-package ones —
they are internal, so the ids have to come from the library.

### The remaining question, sharply

The device never replies. Two candidate reasons, both now testable:

1. the device's `hdcd` service for type 3 (file transfer) is not the service that owns the
   AICPU tmpfs on this card, so the file lands but nobody acts on it; or
2. the `omfile_name` message is never sent because the host's package-config step still cannot
   find the descriptor it needs (`aicpu_package_install.info` — which the vendor's own installer
   writes and which does not exist here).

**That second one is the most concrete lead**, and it is *not* on the device — it is a host-side
file the installer creates. The device reads it to learn what to install. The client log's
`cannot get package path:/usr/local/Ascend/latest///conf/` and `Opening file: was not successful`
point straight at it: the config directory is empty of what the client needs.

---

## The install descriptor, the package-name derivation, and why the device stays silent

Picking up the `aicpu_package_install.info` lead paid off — it revealed the whole configuration
surface the client uses.

### The package descriptor format (found: `conf/ascend_package_load.ini`)

The CANN ships it at `cann-9.1.0/conf/ascend_package_load.ini`, and it is a block-structured ini
with five keys per package:

```
name:mc2_server.tar.gz
install_path:2
optional:true
package_path:opp/built-in/op_impl/aicpu/kernel
load_as_per_soc:false
```

`install_path` is a **numeric code**, and `PackageProcessConfig::ParseInstallPath` maps it:

| code | directory |
|---|---|
| 0 | `aicpu_kernels_device/` |
| 1 | `aicpu_extend_syskernels/` |
| 2 | `/usr/lib64/aicpu_kernels/` |
| 3 | `/home/HwHiAiUser/inuse/` |

The shipped file has **no entry for `Ascend310P-aicpu_syskernels.tar.gz`** — only for
`aicpu_hccl`, `mc2_server`, `aicpu_hcomm` and the `cann-*-compat` packages. That is why the client
logs `can not find aicpu packages` and then sends the AICPU syskernels through a *different* code
path (`SendFileToDevice` rather than `LoadSinglePackageToDevice`).

### The package names are derived from the SoC version — and that lookup fails

The client builds per-SOC package names from a **short SoC version** string. Its literals are
`Ascend310P` (our card), `Ascend310`, `Ascend910`, `Ascend310RC`, `Ascend`, plus the short versions
`E39009` / `E39005`. It obtains that from `halGetSocVersion()` and from
`fe::PlatFormInfos::GetPlatformRes`, and on this host **both fail**:

```
[GetShortSocVersion] get soc_version by halGetSocVersion failed
[GetShortSocVersion] get short_soc_version by fe::PlatFormInfos::GetPlatformRes failed.
```

The visible symptom is that the client looks for the wrong filenames:

```
[LoadSinglePackageToDevice] cannot find package:Ascend-aicpu_legacy.tar.gz, optional is true skip
[LoadSinglePackageToDevice] cannot find package:Ascend910-aicpu_legacy.tar.gz, optional is true skip
```

— `Ascend-...` and `Ascend910-...` rather than `Ascend310P-...`. The real file on disk is
`Ascend310P-aicpu_legacy.tar.gz`. So the fallback path names are wrong for this card, and the
packages that *do* get sent (`Ascend310P-aicpu_legacy.tar.gz`, `aicpu_hcomm.tar.gz`,
`Ascend310P-aicpu_syskernels.tar.gz`) go out via `SendAICPUPackageSimple` /
`SendFileToDevice` without the matching `omfile_name` bookkeeping.

**This is the most likely reason the device never replies.** The client is running the *generic*
AICPU path, not the per-SOC path, because it cannot determine the SoC version.

### What the client reads to build these

From the library's literals, the inputs are:

| input | purpose |
|---|---|
| `ASCEND_LATEST_INSTALL_PATH` | the install tree root (default `/usr/local/Ascend/latest/`) |
| `ASCEND_HOME_PATH` | the CANN root |
| `ASCEND_AICPU_PATH` | where the AICPU packages live |
| `ASCEND_RT_VISIBLE_DEVICES` | which cards to talk to |
| `PROCESS_MODE` / `THREAD_MODE` / `AICPU_PROFILING_MODE` | run mode |
| `/opp/` + `/aicpu/` + `/opp/built-in/` | package path prefixes |
| `conf/ascend_package_load.ini` | the package descriptor |
| `aicpu_package_install.info` | the per-device install record (`deviceId=`, `vfId=`) |

`aicpu_package_install.info` is the *runtime* record of what is installed on a given device and
VF — written by the installer, read by the device. It does not exist on this host, and the
device-side equivalents (`/device-sw-plugin/verify/*`) are likewise absent from the extracted
device image.

### The device-side CMS verifier (confirmed, with the OpenSSL version)

The device's `libtsdaemon.so` embeds a full OpenSSL **3.0.12** (24 Oct 2023) with PKCS#7/CMS
support, and the runtime certificate plugin paths:

```
/device-sw-plugin/verify/community-certificate
/device-sw-plugin/verify/user-certificate
/device-sw-plugin/verify/vendor-certificate
```

with `[CMSCBB_VERIFY]` logging (`cmsLen[%u], iniLen[%u], crlLen[%u]`, `Add config certificate not
pass ret[0x%x]`, `add crl not pass, ret[0x%x]`) and the constraint
`only support signed type package by cms`. It is reached through `halVerifyImg` and the package
types `Ascend-device-sw-plugin.tar.gz` / `Ascend-runtime_device-minios.tar.gz`.

OpenSSL 3.0.12 is a 2023 release, so the usual 3.0.x CVEs are patched; the interesting question is
the *policy* in the CMS chain check rather than a parsing bug. The host driver's own header
points at exactly these paths:

```
CUSTOM_SIGN_CONF_USER_CERT_PATH "/var/asdrv/dev%u/device-sw-plugin/verify/user-certificate"
DSMI_SEC_SUB_CMD_CUST_SIGN_USER_CERT 3
```

so the device implements the runtime customer-certificate installation that the host-side
`dms_custom` code was 910B-gated away from — and the certificate chain check
(`Add config certificate not pass ret[0x%x]`) is where a self-signed or otherwise acceptable
chain would have to be judged.

### Next step, and it is a small one

The client cannot determine the SoC version. That is worth chasing directly: `halGetSocVersion`
is in the host's `libascend_hal.so`, and the driver knows the platform. If the SoC version resolves
to `Ascend310P`, the client takes the **per-SOC path** — which is the path that also sends the
`omfile_name` message and waits for the device's `TSD_OM_PKG_DECOMPRESS_STATUS` reply.

Concretely: find why `halGetSocVersion` fails on this host (it is used by plenty of other tools
too, so it is likely a missing driver symbol rather than a card quirk), and see whether
`ASCEND_LATEST_INSTALL_PATH` + a populated `conf/` is enough to make it resolve.

### Card state

Stock everywhere: AICPU container `663e3dcc…`, host AICPU library `ec28199e…`, device images
`e1e75ec2…`, `conf/ascend_package_load.ini` untouched (it was read-only), the temporary install
tree removed. The `/usr/local/Ascend/latest -> cann-9.1.0` symlink is left in place — it is
correct (the driver install should have created it) and `npu-smi info` plus a torch op both work
with it.

---

## The host/device HAL asymmetry — the root cause of the silent device

This is the finding that explains why the device never answers.

### `halGetSocVersion` is a stub on the host, and real on the device

```
host   libascend_hal.so:  halGetSocVersion @ 0x29a42
       mov $0xfffe,%eax ; pop rbp ; ret          <- 6-byte stub, returns NOT_SUPPORT

device libascend_hal.so:  halGetSocVersion @ 0x8a348   REAL implementation
                          drvGetPlatformInfo @ 0x856d8 REAL
```

and the platform side agrees:

```
drvGetPlatformInfo -> 0, info=1   (HOST side)
```

Measured on the card host with the real signatures from `ascend_hal.h`:

```
halGetSocVersion(dev=0) -> 65534 (0xfffe)  buf=""
halGetSocVersion(dev=1) -> 65534 (0xfffe)  buf=""
drvGetPlatformInfo -> 0 info=1 (HOST side)
```

So **the host HAL physically cannot report the SoC version on the 310P**, while the device HAL
can. The AICPU loader runs on the host, derives its per-SOC package names from that version, and
gets nothing — so it falls back to generic names:

```
[LoadSinglePackageToDevice] cannot find package:Ascend-aicpu_legacy.tar.gz, optional is true skip
[LoadSinglePackageToDevice] cannot find package:Ascend910-aicpu_legacy.tar.gz, optional is true skip
```

while the file on disk is `Ascend310P-aicpu_legacy.tar.gz`. The loader therefore never takes the
per-SOC path, never builds the `omfile_name` bookkeeping, and never gets the device's
`TSD_OM_PKG_DECOMPRESS_STATUS` reply.

### The user-config store is device-side too

The device's `/var/save_shadow` stores the device's `/etc/shadow` into the user-config store as the
item `usershadow`, via `halGetUserConfig` / `halSetUserConfig` /
`halClearUserConfig`. The host's HAL exports the same three symbols, but they are **stubs on the
host** as well:

```
halGetUserConfig(dev=0, "usershadow")       -> 65534 (0xfffe)
halGetUserConfig(dev=0, "ccpu_usr_cert_hash") -> 65534 (0xfffe)
halGetUserConfig(dev=0, "sign_type")         -> 65534 (0xfffe)
halGetUserConfig(dev=0, "cpu_num_cfg")       -> 65534 (0xfffe)
```

Consistent with `drvGetPlatformInfo` saying we are on the **host** side: the device's user-config
store (which holds `usershadow`, `ccpu_usr_cert_hash`, `sign_type`, `cpu_num_cfg`, …) lives on the
**device**, and the host HAL cannot read or write it. The device-side HAL is a different build
with real implementations.

### The seven stubs in the host HAL

For reference, the host `libascend_hal.so` has exactly seven functions that return 0xfffe
immediately:

```
drvUnbindHostPid      halBuffPoolGet         halBuffPoolPut
halDrvEventThreadInit halDrvEventThreadUninit
halMbufGetMbufInfo    halQueueGetMaxNum
```

plus `halGetSocVersion` and the user-config trio in the same category.

### What this means for the plan

The AICPU package load is **host-driven and the host half is compiled out on the 310P**. Two ways
forward, both now concrete:

1. **Make the host HAL able to report the SoC version.** `halGetSocVersion` is a stub that returns
   0xfffe; the real implementation exists in the *device* HAL. Since we already rebuild the host
   driver, the cleanest fix is to provide the missing host-side implementation — the SoC version
   is a static property of the card and the host *does* know it (`npu-smi info` reports
   `Product Type: Atlas 300I Duo`, and the driver has `drvGetPhyChipType` as a named entry point
   that the HAL looks up at runtime). If the HAL's table lookup for `drvGetPhyChipType` is the only
   thing missing, supplying it in our driver fork would unblock the per-SOC package path — and
   with it the `omfile_name` message and the device's install.

   Concretely: `halGetSocVersion` calls a function pointer that is supposed to be
   `drvGetPhyChipType`, and **no driver module exports that symbol** — `nm -D` over
   `drv_devmng_host.ko` finds nothing. Adding it to our fork is a small, well-scoped change, and
   it is the same kind of change that already worked for the AI Core proof.

2. **Do it from the device side.** With the AICPU library landing on the device's
   `/usr/lib64/aicpu_kernels/`, the device's own `tsdaemon` (which has a real `halGetSocVersion`)
   would handle it — but getting there still needs the package to land, which is the thing that
   is currently blocked.

Option 1 is the smallest step that could change the outcome, and it is entirely in code we own.

---

## BREAKTHROUGH: the SoC version shim works, and the loader now takes the per-SOC path

Picking up the "modify the host driver" idea — but in the cheapest possible form first.

### What the loader was missing

`GetShortSocVersion` resolves like this:

```
halGetSocVersion(devId, buf, len)                    -> 0xfffe
fe::PlatformInfoManager::Instance()
fe::PlatformInfoManager::InitializePlatformInfo()
fe::PlatformInfoManager::GetPlatformInfos(...)
```

and `halGetSocVersion` in the host's `libascend_hal.so` is a **6-byte stub**:

```
halGetSocVersion:
    mov  %edi,-0x4(%rbp)
    mov  %rsi,-0x10(%rbp)
    mov  %edx,-0x8(%rbp)
    mov  $0xfffe,%eax
    pop  %rbp
    ret
```

The runtime symbol resolves to file offset **0x29a42** — the stub — confirmed at runtime with
`dlsym` + `link_map`. There is a *second* entry point at 0x29a57 that contains the real
implementation (issue dev-manager command **0x4d97**, the SoC version query), but nothing reaches
it. The driver *does* handle 0x4d97 — `drv_devmng_host.ko` contains `cmp $0x4d97,%esi` twice — so
the query exists on the device side of the ioctl; the HAL simply never asks.

(Correction to an earlier note in this session: the *device* HAL's `halGetSocVersion` is also a
stub — `mov w0,#0xfffe; add sp,sp,#0x10; ret` — my first stub-detector matched the wrong byte
pattern for aarch64. The host/device asymmetry I described was wrong; the real story is that
**both** are stubbed and the working implementation is the dead code at 0x29a57 in the host HAL.)

### The shim

`socver_shim.c` — an `LD_PRELOAD` library that defines `halGetSocVersion` and answers with the
card's own platform identity, read from CANN's platform config:

```
x86_64-linux/data/platform_config/Ascend310P3.ini
    [version]
    SoC_version=Ascend310P3
    Short_SoC_version=Ascend310P
```

Before:

```
[GetShortSocVersion] get soc_version by halGetSocVersion failed
[GetShortSocVersion] get short_soc_version by fe::PlatFormInfos::GetPlatformRes failed.
[LoadSinglePackageToDevice] cannot find package:Ascend-aicpu_legacy.tar.gz, optional is true skip
[LoadSinglePackageToDevice] cannot find package:Ascend910-aicpu_legacy.tar.gz, optional is true skip
```

After:

```
[socver_shim] halGetSocVersion(dev=0) -> "Ascend310P3"
[GetShortSocVersion] get short_soc_version:Ascend310P
[SendAICPUPackageSimple] hdc send file[.../Ascend310P/aicpu/Ascend310P-aicpu_legacy.tar.gz]
                              to [/home/HwHiAiUser/hdcd/device0/aicpu/520013_Ascend310P-aicpu_legacy.tar.gz] success
```

**The per-SOC package name is now correct** (`Ascend310P-aicpu_legacy.tar.gz`, not `Ascend-...`),
so the loader takes the per-SOC path.

### A second, separate gap: the package location

`conf/ascend_package_load.ini` lists per-SOC packages with `package_path:opp`:

```
name:Ascend310P-aicpu_legacy.tar.gz
install_path:2
optional:true
package_path:opp
```

but the files actually live at `opp/Ascend310P/aicpu/`. So the loader looks for
`opp/Ascend310P-aicpu_legacy.tar.gz` and does not find it. Symlinks in `opp/` fix that, and the
loader then finds and sends the per-SOC package.

### Where it still stops, and the evidence

The log now runs:

```
[IsOkToLoadFileToDevice] loadfile name:Ascend310P-aicpu_syskernels.tar.gz
[LoadFileToDevice] begin load file:Ascend310P-aicpu_syskernels.tar.gz
[LoadOmFileToDevice] file path is:/usr/local/Ascend/cann-9.1.0/opp/Ascend310P/aicpu
[SendFileToDevice] hdc send file[...] success
```

and then stops — no `Load om file name:`, no `TSD_OM_PKG_DECOMPRESS_STATUS`. The thread state
says exactly where:

```
pid 522819: hdcdrv_recv_peek.part.0
             hdcdrv_ioctl_com
             hdcdrv_ioctl
```

**the client is blocked in `hdcdrv_recv_peek`, waiting for a message from the device.** The
transfer itself is fast — a 53 MB push to the same destination takes **3.2 s** via our own
`drvHdcSendFileV2` — so this is not throughput.

So the sequence is now: the file arrives on the device, and the device never sends the
`TSD_OM_PKG_DECOMPRESS_STATUS` reply that the client is blocked on. The device-side code for that
reply exists — `TSDaemon::GetOmFileDecompressStatus` and
`TSDaemon::ProcessOmFileDecompressStatus` in the device's `libtsdaemon.so` — but the device's
`hdcd` service for type 3 (file transfer, uid 1000) is a *different* agent from `tsdaemon`, and
the AICPU install is driven by `tsdaemon`'s own inotify/epoll watcher on
`/home/HwHiAiUser/hdcd/device<N>/Ascend310P/aicpu/vfid<N>`.

**The most likely remaining cause is the file name/location.** The device's watcher looks for
`^[0-9]+_Ascend([0-9]{3}(P)?)?-aicpu_syskernels\.tar\.gz$` in a directory reached as
`<base>/Ascend310P/aicpu/vfid<N>`. The vendor client sends to
`hdcd/device0/aicpu/<pid>_<name>` — i.e. `hdcd/device0/aicpu/`, **not**
`hdcd/device0/Ascend310P/aicpu/vfid0/`. The client's own destination may not be where the device
watches.

That is a concrete, testable difference, and it is the next thing to try: push to
`hdcd/device0/Ascend310P/aicpu/vfid0/` (which the host-side path check rejected earlier) and see
whether the device reacts.

### Card state

Healthy: 6/6 chips OK, 6 davinci nodes, 0 phy_startup errors, bringup active. The AICPU container
is stock (`663e3dcc…`), device images stock (`e1e75ec2…`).

Changes made on the host (all reversible, and none of them touch the card's firmware):
- `/usr/local/Ascend/latest` → `cann-9.1.0` symlink — the driver install should have created it;
  without it the client cannot find the package tree at all.
- `opp/Ascend310P-aicpu_{legacy,syskernels,extend_syskernels}.tar.gz` symlinks into
  `opp/Ascend310P/aicpu/` — matching what `conf/ascend_package_load.ini` expects.
- `socver_shim.so` in `/var/local/ascend-work/tools/` — loaded only with `LD_PRELOAD`.
