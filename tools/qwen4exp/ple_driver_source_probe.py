# SPDX-License-Identifier: Apache-2.0
"""Qualify private host-page registration; never submit DMA or touch MMIO."""

import argparse
import ctypes
import errno
import fcntl
import json
import mmap
import os
import resource
from contextlib import ExitStack
from dataclasses import asdict

from tools.qwen4exp import ple_driver as driver


def locked_kib():
    with open("/proc/self/status") as status:
        return int(next(line.split()[1] for line in status if line.startswith("VmLck:")))


def expect_errno(operation, expected):
    try:
        operation()
    except OSError as exc:
        if exc.errno != expected:
            raise
    else:
        raise AssertionError(f"expected errno {expected}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", type=int, nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    libc = ctypes.CDLL(None, use_errno=True)
    libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]
    libc.mmap.restype = ctypes.c_void_p
    libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    libc.munmap.restype = ctypes.c_int
    page = os.sysconf("SC_PAGE_SIZE")
    size = 2 * page
    before = locked_kib()
    limits = resource.getrlimit(resource.RLIMIT_MEMLOCK)
    records = []
    with ExitStack() as stack:
        fd = os.memfd_create("ple-private-source-probe", os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING)
        stack.callback(os.close, fd)
        data = bytes(i % 251 for i in range(size))
        assert os.write(fd, data) == size
        fcntl.fcntl(
            fd, fcntl.F_ADD_SEALS, fcntl.F_SEAL_WRITE | fcntl.F_SEAL_GROW | fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_SEAL
        )
        address = libc.mmap(None, size, mmap.PROT_READ, mmap.MAP_SHARED, fd, 0)
        if address == ctypes.c_void_p(-1).value:
            raise OSError(ctypes.get_errno(), "private read-only mmap failed")
        stack.callback(libc.munmap, address, size)
        source = address + page - 10
        leases = stack.enter_context(driver.reserve_devices(args.devices))
        # RLIMIT_MEMLOCK must be enforced even on an otherwise valid source.
        resource.setrlimit(resource.RLIMIT_MEMLOCK, (0, limits[1]))
        try:
            expect_errno(lambda: leases[0].register_source(source, 320), errno.ENOMEM)
        finally:
            resource.setrlimit(resource.RLIMIT_MEMLOCK, limits)
        assert locked_kib() == before
        for round_id in range(3):
            for lease in leases:
                info = lease.register_source(source, 320)
                assert lease.query_source() == info
                assert info.page_count == 2 and info.segment_count > 0
                records.append({"round": round_id, "device": lease.info.device_id, **asdict(info)})
            assert locked_kib() == before + len(leases) * 2 * page // 1024
            assert ctypes.string_at(address, size) == data
            if round_id == 2:
                # The lease FD's last-close path releases the registered source.
                break
            for lease in leases:
                info = lease.source_info
                lease.release_source()
                stale = driver._source_request(info.generation, region_id=info.region_id)
                expect_errno(
                    lambda owner=lease, request=stale: fcntl.ioctl(owner._fd, driver.SOURCE_QUERY, request, True),
                    errno.ESTALE,
                )
            assert locked_kib() == before
    assert locked_kib() == before
    result = {
        "passed": True,
        "dma_submitted": False,
        "source": "shared sealed read-only memfd",
        "registrations": records,
        "memlock_recovered": True,
        "memlock_limit_enforced": True,
    }
    with open(args.output, "w") as output:
        json.dump(result, output, indent=2)
        output.write("\n")
    print(json.dumps({"passed": True, "registrations": len(records), "dma_submitted": False}))


if __name__ == "__main__":
    main()
