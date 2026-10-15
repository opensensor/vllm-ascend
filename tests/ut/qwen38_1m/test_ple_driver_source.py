# SPDX-License-Identifier: Apache-2.0
"""The startup client must never confuse source registration with submission."""

import errno
from unittest.mock import patch

import pytest

from tools.qwen4exp import ple_driver as driver


def lease():
    return driver.DeviceDMALease(
        83, driver.LeaseInfo(0, 20, 7, driver.CAP_HOST_ROUTING_ISOLATED | driver.CAP_SOURCE_REGISTRATION, 64, 64)
    )


def reply(_fd, command, buffer, mutate):
    assert mutate
    fields = list(driver.SOURCE_STRUCT.unpack(buffer))
    assert fields[0] == 1 and fields[2] == 7 and not fields[1] and not any(fields[6:])
    if command == driver.SOURCE_REGISTER:
        assert fields[3:6] == [4090, 320, 0]
    else:
        assert fields[3:6] == [0, 0, 3]
    if command != driver.SOURCE_RELEASE:
        buffer[:] = driver.SOURCE_STRUCT.pack(1, 0, 7, 4090, 320, 3, 2, 1, 0, 0)
    return 0


def test_source_abi_lifecycle():
    assert driver.SOURCE_STRUCT.size == 64
    assert (driver.SOURCE_REGISTER, driver.SOURCE_QUERY, driver.SOURCE_RELEASE) == (0xC04044D3, 0xC04044D4, 0x404044D5)
    with patch.object(driver.fcntl, "ioctl", side_effect=reply), patch.object(driver.os, "close") as close:
        owner = lease()
        info = owner.register_source(4090, 320)
        assert info == driver.SourceInfo(7, 4090, 320, 3, 2, 1)
        assert owner.query_source() == info
        with pytest.raises(RuntimeError, match="already"):
            owner.register_source(4090, 320)
        owner.release_source()
        owner.release_source()
        assert owner.source_info is None
        close.assert_not_called()
        owner.register_source(4090, 320)
        owner.close()
        close.assert_called_once_with(83)
        assert owner.source_info is None
        with pytest.raises(RuntimeError, match="closed"):
            owner.register_source(4090, 320)
        with pytest.raises(RuntimeError, match="no live"):
            owner.query_source()


@pytest.mark.parametrize(
    "address,length",
    [
        (0, 320),
        (-1, 320),
        (True, 320),
        ("4096", 320),
        (4096, False),
        (4096, 0),
        (4096, -1),
        (4096, "320"),
        (2**64 - 100, 320),
        (4096, driver.MAX_SOURCE_BYTES + 1),
    ],
)
def test_bad_ranges_never_reach_kernel(address, length):
    with patch.object(driver.fcntl, "ioctl") as ioctl:
        with pytest.raises(ValueError):
            lease().register_source(address, length)
        ioctl.assert_not_called()


def test_old_driver_fails_without_ioctl():
    owner = driver.DeviceDMALease(83, driver.LeaseInfo(0, 20, 7, driver.CAP_HOST_ROUTING_ISOLATED, 64, 64))
    with patch.object(driver.fcntl, "ioctl") as ioctl:
        with pytest.raises(RuntimeError, match="lacks source"):
            owner.register_source(4090, 320)
        ioctl.assert_not_called()


@pytest.mark.parametrize(
    "field,value", [(0, 2), (1, 1), (2, 8), (3, 4096), (4, 321), (5, 0), (6, 1), (7, 0), (7, 3), (8, 1)]
)
def test_invalid_registration_reply_closes_lease(field, value):
    def broken(fd, cmd, buf, mutate):
        reply(fd, cmd, buf, mutate)
        fields = list(driver.SOURCE_STRUCT.unpack(buf))
        fields[field] = value
        buf[:] = driver.SOURCE_STRUCT.pack(*fields)

    with patch.object(driver.fcntl, "ioctl", side_effect=broken), patch.object(driver.os, "close") as close:
        owner = lease()
        with pytest.raises(RuntimeError, match="source reply"):
            owner.register_source(4090, 320)
        close.assert_called_once_with(83)
        assert owner.source_info is None


def test_ioctl_failure_retains_known_ownership():
    with patch.object(driver.fcntl, "ioctl", side_effect=reply) as ioctl, patch.object(driver.os, "close") as close:
        owner = lease()
        info = owner.register_source(4090, 320)
        ioctl.side_effect = OSError(errno.EBUSY, "busy")
        with pytest.raises(OSError):
            owner.release_source()
        assert owner.source_info == info
        close.assert_not_called()
        ioctl.side_effect = reply
        owner.release_source()
        assert owner.source_info is None


def test_query_detects_changed_handle():
    with patch.object(driver.fcntl, "ioctl", side_effect=reply) as ioctl:
        owner = lease()
        info = owner.register_source(4090, 320)

        def changed(fd, cmd, buf, mutate):
            reply(fd, cmd, buf, mutate)
            fields = list(driver.SOURCE_STRUCT.unpack(buf))
            fields[5] += 1
            buf[:] = driver.SOURCE_STRUCT.pack(*fields)

        ioctl.side_effect = changed
        with pytest.raises(RuntimeError, match="source changed"):
            owner.query_source()
        assert owner.source_info == info
