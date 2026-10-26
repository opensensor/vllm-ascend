#!/usr/bin/env python3
"""Read the Ascend card's secure flash via the DMS feature command.

Fixed 2026-10-26: the previous version failed with `ENODEV` because it issued
the feature-command ioctl on the davinci_manager fd *without* first opening
the DMS sub-module.  The davinci_manager interface requires the two-step
protocol that `dms_log.py` already uses:

    1. ioctl(fd, _IO('Z', 0), {char module_name[256]; int device_id;})
         with module_name = "DMS", device_id = -1
    2. ioctl(fd, ('V' << 8) | 0, &dms_ioctl_arg)
         main_cmd = 'V' (DMS_MAIN_CMD_FLASH), sub_cmd = 0 (GET_FLASH_CONTENT)
         input  = struct dms_flash_content_in
                  { int dev_id; u32 type; u8 *buf; u32 size; u32 offset; }
         output = unused

The driver maps this onto sec_flash_read(dev_id, RTC_FLASH_ADDR + offset, ...),
with RTC_FLASH_ADDR = 0xDE0000 and RTC_FLASH_SIZE = 128 KiB, and caps a single
transfer at FLASH_CONTENT_BUFFER_MAX_SIZE = 1024 bytes.

Usage:
    python3 read_sec_flash.py <dev_id> [offset] [size] [outfile]
      dev_id   0..5 (chip id; the driver validates against DEVICE_NUM_MAX)
      offset   byte offset within the 128 KiB RTC flash region (default 0)
      size     bytes to read (default 1024, the driver maximum per call)
      outfile  optional; writes the raw bytes
"""
import ctypes
import os
import sys

DMS_MAIN_CMD_FLASH = ord('V')
DMS_SUBCMD_GET_FLASH_CONTENT = 0
DMS_CONTENT_RTC_CONFIG = 0
NODE = "/dev/davinci_manager"

DAVINCI_INTF_IOC_MAGIC = ord('Z')
DAVINCI_INTF_IOCTL_OPEN = (DAVINCI_INTF_IOC_MAGIC << 8) | 0
DAVINCI_INTF_IOCTL_CLOSE = (DAVINCI_INTF_IOC_MAGIC << 8) | 1
DAVINCI_INTF_MODULE_DMS = "DMS"
DAVINIC_MODULE_NAME_MAX = 256

RTC_FLASH_ADDR = 0xDE0000
RTC_FLASH_SIZE = 128 * 1024
MAX_CHUNK = 1024


class DavinciIntfOpenArg(ctypes.Structure):
    _fields_ = [("module_name", ctypes.c_char * DAVINIC_MODULE_NAME_MAX),
                ("device_id", ctypes.c_int)]


class DmsIoctlArg(ctypes.Structure):
    _fields_ = [
        ("msg_source", ctypes.c_uint32),
        ("main_cmd", ctypes.c_uint32),
        ("sub_cmd", ctypes.c_uint32),
        ("filter", ctypes.c_void_p),
        ("filter_len", ctypes.c_uint32),
        ("input", ctypes.c_void_p),
        ("input_len", ctypes.c_uint32),
        ("output", ctypes.c_void_p),
        ("output_len", ctypes.c_uint32),
    ]


class FlashContentIn(ctypes.Structure):
    _fields_ = [
        ("dev_id", ctypes.c_int),
        ("type", ctypes.c_uint32),
        ("buf", ctypes.c_void_p),
        ("size", ctypes.c_uint32),
        ("offset", ctypes.c_uint32),
    ]


libc = ctypes.CDLL("libc.so.6", use_errno=True)
libc.ioctl.restype = ctypes.c_long
libc.ioctl.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_void_p, ctypes.c_void_p]


def ioctl(fd, cmd, arg):
    ctypes.set_errno(0)
    ret = libc.ioctl(fd, cmd, ctypes.byref(arg) if arg is not None else None, None)
    return ret, ctypes.get_errno()


def read_chunk(fd, dev_id, offset, size):
    buf = ctypes.create_string_buffer(size)
    fin = FlashContentIn()
    fin.dev_id = dev_id
    fin.type = DMS_CONTENT_RTC_CONFIG
    fin.buf = ctypes.cast(buf, ctypes.c_void_p)
    fin.size = size
    fin.offset = offset

    arg = DmsIoctlArg()
    arg.msg_source = 0
    arg.main_cmd = DMS_MAIN_CMD_FLASH
    arg.sub_cmd = DMS_SUBCMD_GET_FLASH_CONTENT
    arg.filter = 0
    arg.filter_len = 0
    arg.input = ctypes.addressof(fin)
    arg.input_len = ctypes.sizeof(fin)
    arg.output = 0
    arg.output_len = 0

    cmd = (DMS_MAIN_CMD_FLASH << 8) | DMS_SUBCMD_GET_FLASH_CONTENT
    ret, err = ioctl(fd, cmd, arg)
    if ret != 0:
        raise OSError(err, f"flash ioctl failed at offset 0x{offset:x}: "
                           f"{os.strerror(err) if err else '-'}")
    return buf.raw[:size]


def main():
    dev_id = int(sys.argv[1])
    offset = int(sys.argv[2], 0) if len(sys.argv) > 2 else 0
    size = int(sys.argv[3], 0) if len(sys.argv) > 3 else MAX_CHUNK
    outfile = sys.argv[4] if len(sys.argv) > 4 else None

    if offset + size > RTC_FLASH_SIZE:
        print(f"refusing: offset+size exceeds the {RTC_FLASH_SIZE}-byte RTC flash region",
              file=sys.stderr)
        return 2

    fd = os.open(NODE, os.O_RDONLY)

    open_arg = DavinciIntfOpenArg()
    open_arg.module_name = DAVINCI_INTF_MODULE_DMS.encode()
    open_arg.device_id = -1
    ret, err = ioctl(fd, DAVINCI_INTF_IOCTL_OPEN, open_arg)
    if ret != 0:
        print(f"OPEN DMS module failed: ret={ret} errno={err} "
              f"({os.strerror(err) if err else '-'})", file=sys.stderr)
        os.close(fd)
        return 1

    data = bytearray()
    try:
        done = 0
        while done < size:
            n = min(MAX_CHUNK, size - done)
            data += read_chunk(fd, dev_id, offset + done, n)
            done += n
    except OSError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        close_arg = DavinciIntfOpenArg()
        close_arg.module_name = DAVINCI_INTF_MODULE_DMS.encode()
        close_arg.device_id = -1
        ioctl(fd, DAVINCI_INTF_IOCTL_CLOSE, close_arg)
        os.close(fd)
        return 1

    close_arg = DavinciIntfOpenArg()
    close_arg.module_name = DAVINCI_INTF_MODULE_DMS.encode()
    close_arg.device_id = -1
    ioctl(fd, DAVINCI_INTF_IOCTL_CLOSE, close_arg)
    os.close(fd)

    data = bytes(data)
    if outfile:
        with open(outfile, "wb") as fh:
            fh.write(data)
        print(f"wrote {len(data)} bytes to {outfile}")
    else:
        for i in range(0, len(data), 16):
            row = data[i:i + 16]
            txt = "".join(chr(b) if 32 <= b < 127 else "." for b in row)
            print(f"  +0x{i:06x}: {row.hex():32s}  {txt}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
