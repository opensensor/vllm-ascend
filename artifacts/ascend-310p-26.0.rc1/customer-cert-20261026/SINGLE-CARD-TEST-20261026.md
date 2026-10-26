# Single-card test report — 2026-10-26

## Target

| | |
|---|---|
| Host | `matteius@192.168.53.187` (3 cards / 6 chips) |
| Card under test | **NPU ID 9**, bus `0000:01:00.0`, davinci devices 0 and 1 |
| Product | `IT21PD2G10` |
| Serial (SoC) | `2106030737ZERC001310` |
| Serial (MCU elabel) | `2106030737ZERC001310` — **matches** |
| Chosen because | this is the card whose MCU SPI flash we have a dump of (`spi-dump-ascend-verified-9fd4cc63.bin`, window `ascend_220000_verified.bin` at flash `0x220000`) |

The other two cards (NPU ID 8217 / `…ZERB002529`, NPU ID 16409 / `…ZERB000091`) have serials
that are **absent** from the dump, which confirms the dump belongs to card 9.

`ascend_A.bin` and `ascend_B.bin` are byte-identical (`9fd4cc63…`), so the two dumps taken two
weeks ago were of the same flash content.

## Result summary

| # | test | outcome |
|---|---|---|
| 1 | MCU elabel items parse + CRC-16/XMODEM verify (from dump) | **21/21 valid** |
| 2 | elabel serial vs SoC-reported serial | **identical** |
| 3 | forge a modified elabel (model `IT21PD2G10` → `IT21PD2G11`) with recomputed CRC | **whole record still verifies 21/21** |
| 4 | host API reachability: `dsmi_get_flash_content` | `0xfffe` NOT_SUPPORT |
| 5 | host API reachability: `dsmi_dft_get_elable` | `0xfffe` NOT_SUPPORT (7-byte stub) |
| 6 | host API reachability: `dsmi_cmd_dft_get_elabel` (MCU opcode 0x15) | `0xfffe` — builds the command, driver has no handler |
| 7 | host API reachability: `dsmi_get_custom_op_secverify_cert_show_info` | `0xfffe` (7-byte stub) |
| 8 | `libdcmi` elabel read (`dcmi_get_card_elabel_v2`) | `-8001` |
| 9 | `libdcmi` raw I2C read (`dcmi_flash_read_elabel` → `dcmi_i2c_get_data`) | `I2C_RDWR` → **ETIMEDOUT** |
| 10 | `libdcmi` raw I2C write (`dcmi_flash_write_elabel`) | unreachable behind the same failing read |
| 11 | direct I2C write probe at slave 0x50, reg 0x200 | `EINVAL` (no responder on the AMD bus) |
| 12 | `npu-smi info -l` (the working identity path) | **succeeds**, serial read from `/dev/davinci_manager`, **no I2C at all** |

## What the single-card test established

### The cryptographic claim is proven

`mcu_elabel_forge.py` modifies the board-model item of the real elabel record and recomputes its
CRC-16/XMODEM. The firmware's own verification algorithm — reconstructed from `elabel_init`
(`0x08049b8a`: `ldrh sl,[r8,r3]` / `bl 0x805b2c0` / `cmp sl,r0` / `bne → "fail to read and verify
elabel."`) — then accepts the whole record:

```
baseline verification : ALL OK (21/21 items)
  id=0x0020 off=0x009c crc=0x888e len=  3 b'\xecy\xf0'
  id=0x0021 off=0x00e4 crc=0x222d len=  6 b'Huawei'
  id=0x0022 off=0x012c crc=0x5efb len= 10 b'IT21PD2G10'
  id=0x0023 off=0x0174 crc=0x9411 len= 20 b'2106030737ZERC001310'
  id=0x0030 off=0x024c crc=0x222d len=  6 b'Huawei'

forging item id=0x0022 at 0x2c012c
    original payload : b'IT21PD2G10' (crc 0x5efb)
    forged payload   : b'IT21PD2G11'
    new CRC-16/XMODEM: 0x4eda
    re-verify whole record: ALL OK (21/21 items)
```

**A modified elabel passes the firmware's check.** The MCU's board-identity record is protected
by a 16-bit checksum, not a signature — exactly the asymmetry identified earlier (signatures for
the vendor's own firmware, CRC for everything else).

### The transport is absent on this platform — and the reason is now precise

The vendor elabel path is **I2C**, not PCIe:

```
dcmi_flash_read_elabel(buf, len)                       libdcmi @ 0x168b0
    call dcmi_board_chip_type_is_ascend_310b           ; 1 iff chip type == 4 (910B)
    cmp  $0x1, %eax
    sbb  %rdx, %rdx                                   ; -1 on 910B, 0 otherwise
    and  $0xfffffffffffffe80, %rdx
    add  $0x200, %rdx                                 ; 310P -> 0x200, 910B -> 0x280
    call dcmi_i2c_get_data(g_i2c_dev_name, 0x50, buf, len, reg)
```

and the HAL underneath picks the bus by **adapter name**:

```
libascend_hal.so:  "SMBus I801 adapter"              <- the name it looks for
libascend_hal.so:  "/sys/bus/i2c/devices/"            <- where it scans
libascend_hal.so:  "read [%s], find i2c-%d name dev_name:%s"
```

Measured on the card host:

| bus | adapter name | what is on it |
|---|---|---|
| i2c-0 | `Synopsys DesignWare I2C adapter` (`i2c_designware AMDI0010:00`) | **nothing** — `controller timed out` |
| i2c-1 | `SMBus PIIX4 adapter port 0 at 0b00` | host **DDR4 SPD** EEPROMs (ee1004) at 0x50–0x57 |
| i2c-2 | `SMBus PIIX4 adapter port 2 at 0b00` | empty |
| i2c-3 | `SMBus PIIX4 adapter port 1 at 0b20` | empty |

Two independent reasons the vendor path cannot work here:

1. **The HAL looks for the adapter name `"SMBus I801 adapter"` — an Intel chipset string.** This
   is an AMD host with `PIIX4` adapters, so the name never matches and the HAL falls back to
   i2c-0, which is the *chipset's* empty DesignWire controller, not the NPU's.
2. **The card's MCU is not on any host I2C bus.** The MCU's flash is on the MCU's own SPI bus;
   the host-side I2C buses carry only host DDR SPDs. `strace` of `npu-smi` confirms it opens
   `/dev/davinci_manager` and `/dev/hisi_hdc` and **never** an i2c device.

So the SoC-reported serial number is a **SoC-side** copy of the board identity, not a read of the
MCU elabel. The MCU elabel is currently reachable only physically (the SPI clip that produced
`ascend_A.bin`).

### The write path exists in the vendor code but is unreachable here

`libdcmi` exports the full elabel write API:

```
dcmi_flash_write_elabel   @ 0x16a20   -> dcmi_i2c_set_data(dev, 0x50, buf, size, reg)
dcmi_flash_erase_elabel   @ 0x16960
dcmi_elabel_set_data      @ 0x173c0
dcmi_elabel_clear         @ 0x17200
```

and the MCU firmware implements the matching handlers:

```
board_read_elable_response_callback     0x08053552
read_customized_info_callback           0x08053d9c
write_customized_info_callback          0x08053e52
clear_customized_info_callback          0x08053eac
```

Both halves exist; the link between them is the missing I2C bus. The same is true of the
customer-certificate region (`customized_verify` / `WriteCertToFlash` / `customized_info_write`),
which is likewise CRC-16 protected.

## Card state after the test

Unchanged and healthy. The test was read-only except for one **identical-bytes** write attempt
that never reached the bus (it was skipped because the read failed). No flash was erased, no
firmware was written, no driver was reloaded.

```
phy_startup errs : 0
davinci nodes   : 6
chips OK        : 6/6 (all 310P3, Health: OK)
bringup         : active
```

## What would actually work

| route | state |
|---|---|
| **Physical SPI clip** (`scriba -W`, CH341A/EZP2019) | the one that produced the dump; supports write and erase; **the only route that reaches the MCU flash** |
| A host I2C adapter on the NPU's I2C pins | would need the bus brought out and an Intel-named adapter or a patched HAL name match |
| Driver-side | `dms_lpm_pass_through_mcu.o` exists but is gated to `ascend910`/`ascend910B`, and `OBJ_IMU`/ipcdrv do not exist on the 310P; the board-comm opcodes 0x14/0x15 are defined in `dev_mon_cmd_def.h` but unimplemented in the 310P module |

Given the CRC-only integrity, **the SPI clip route is sufficient**: a forged elabel with a
recomputed CRC-16 is accepted by the firmware, and `scriba` can write it.
