#!/usr/bin/env python3
"""Dump the Ascend host driver's kernel-log ring buffer.

The driver's devdrv_info/devdrv_err messages (boot mode, per-file load
decisions, "Load file check image failed", "Skip file", ...) are written to a
ring buffer by log_save_to_ringbuf(), not to printk, so they never appear in
dmesg.  The only userspace reader is the DMS feature command, reached through
the davinci_manager interface:

  1. open the DMS sub-module:
         ioctl(fd, _IO('Z', 0), {char module_name[256]; int device_id;})
     with module_name = "DMS"  (DAVINCI_INTF_MODULE_DMS)
  2. issue the log command on the same fd:
         ioctl(fd, ('S' << 8) | 0, &dms_ioctl_arg)
     main_cmd = 'S' (DMS_MAIN_CMD_LOG), sub_cmd = 0 (DMS_SUBCMD_GET_LOG_INFO)
     input  = struct LogInfo { void *buffer; u32 length; u32 buf_len; }
              with length == 0x1400000 (DMS_LOG_BUF_SIZE)
     output = u32 *   (actual bytes written)

Usage: python3 dms_log.py [device-node] [grep-pattern]
"""
import ctypes
import os
import re
import sys

DAVINCI_INTF_IOC_MAGIC = ord('Z')
DAVINCI_INTF_IOCTL_OPEN = (DAVINCI_INTF_IOC_MAGIC << 8) | 0
DAVINCI_INTF_IOCTL_CLOSE = (DAVINCI_INTF_IOC_MAGIC << 8) | 1

DMS_MAIN_CMD_LOG = ord('S')
DMS_SUBCMD_GET_LOG_INFO = 0
DMS_LOG_CMD = (DMS_MAIN_CMD_LOG << 8) | DMS_SUBCMD_GET_LOG_INFO
DMS_LOG_BUF_SIZE = 0x1400000
DAVINCI_INTF_MODULE_DMS = "DMS"
DAVINIC_MODULE_NAME_MAX = 256


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


class LogInfo(ctypes.Structure):
    _fields_ = [("buffer", ctypes.c_void_p), ("length", ctypes.c_uint32),
                ("buf_len", ctypes.c_uint32)]


libc = ctypes.CDLL("libc.so.6", use_errno=True)
libc.ioctl.restype = ctypes.c_long
libc.ioctl.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_void_p, ctypes.c_void_p]


def ioctl(fd, cmd, arg):
    ctypes.set_errno(0)
    ret = libc.ioctl(fd, cmd, ctypes.byref(arg) if arg is not None else None, None)
    err = ctypes.get_errno()
    return ret, err


def main():
    node = sys.argv[1] if len(sys.argv) > 1 else "/dev/davinci_manager"
    pattern = sys.argv[2] if len(sys.argv) > 2 else None

    fd = os.open(node, os.O_RDONLY)

    # 1. open the DMS sub-module
    open_arg = DavinciIntfOpenArg()
    open_arg.module_name = DAVINCI_INTF_MODULE_DMS.encode()
    open_arg.device_id = -1
    ret, err = ioctl(fd, DAVINCI_INTF_IOCTL_OPEN, open_arg)
    if ret != 0:
        print(f"OPEN DMS module failed: ret={ret} errno={err} "
              f"({os.strerror(err) if err else '-'})", file=sys.stderr)
        os.close(fd)
        return 1

    # 2. ask for the log
    buf = ctypes.create_string_buffer(DMS_LOG_BUF_SIZE)
    out_len = ctypes.c_uint32(0)
    info = LogInfo(ctypes.addressof(buf), DMS_LOG_BUF_SIZE, 0)

    arg = DmsIoctlArg()
    arg.msg_source = 0
    arg.main_cmd = DMS_MAIN_CMD_LOG
    arg.sub_cmd = DMS_SUBCMD_GET_LOG_INFO
    arg.filter = 0
    arg.filter_len = 0
    arg.input = ctypes.addressof(info)
    arg.input_len = ctypes.sizeof(info)
    arg.output = ctypes.addressof(out_len)
    arg.output_len = ctypes.sizeof(out_len)

    ret, err = ioctl(fd, DMS_LOG_CMD, arg)
    if ret != 0:
        print(f"LOG command failed: ret={ret} errno={err} "
              f"({os.strerror(err) if err else '-'})", file=sys.stderr)
        os.close(fd)
        return 1

    # 3. close the sub-module
    close_arg = DavinciIntfOpenArg()
    close_arg.module_name = DAVINCI_INTF_MODULE_DMS.encode()
    close_arg.device_id = -1
    ioctl(fd, DAVINCI_INTF_IOCTL_CLOSE, close_arg)
    os.close(fd)

    n = info.buf_len or out_len.value
    text = buf.raw[:n].decode("utf-8", errors="replace")
    lines = text.splitlines()
    if pattern:
        rx = re.compile(pattern, re.I)
        lines = [l for l in lines if rx.search(l)]
    for line in lines:
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
