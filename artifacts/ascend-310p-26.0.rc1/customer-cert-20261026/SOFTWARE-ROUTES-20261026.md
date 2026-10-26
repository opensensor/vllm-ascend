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
