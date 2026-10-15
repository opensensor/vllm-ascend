# SPDX-License-Identifier: Apache-2.0
import errno
import os
from unittest.mock import patch

import pytest

from tools.qwen4exp import ple_driver as driver


def complete_reply(_fd, command, buffer, mutate):
    assert mutate
    if command in (driver.ACQUIRE, driver.QUERY):
        device = driver.LEASE_STRUCT.unpack(buffer)[1]
        buffer[:] = driver.LEASE_STRUCT.pack(1, device, 0, 20, 7, 0, 64, 64, 0, 0, 0)
    return 0


def test_context_lifetime_and_request_abi():
    assert driver.LEASE_STRUCT.size == 64
    # Compare with _IOWR/_IOW values compiled from the kernel UAPI header.
    assert (driver.ACQUIRE, driver.QUERY, driver.RELEASE) == (0xC04044D0, 0xC04044D1, 0x404044D2)
    with (
        patch.object(os, "open", return_value=83) as opened,
        patch.object(os, "close") as closed,
        patch.object(driver.fcntl, "ioctl", side_effect=complete_reply) as ioctl,
    ):
        with driver.DeviceDMALease.acquire(2) as lease:
            assert lease.info.device_id == 2
            assert lease.info.generation == 7
            assert lease.info.capabilities == 0
            assert lease.query() == lease.info
            closed.assert_not_called()
        closed.assert_called_once_with(83)
        assert ioctl.call_count == 2
        assert opened.call_args.args[1] & os.O_CLOEXEC
        assert opened.call_args.args[1] & os.O_NOFOLLOW
        lease.close()
        closed.assert_called_once()
        with pytest.raises(RuntimeError, match="closed"):
            lease.query()


@pytest.mark.parametrize("bad", [-1, 2**32, True, "0", None])
def test_invalid_device_does_not_open_driver(bad):
    with patch.object(os, "open") as opened:
        with pytest.raises(ValueError):
            driver.DeviceDMALease.acquire(bad)
        opened.assert_not_called()


def test_busy_admission_closes_fd_without_fallback():
    with (
        patch.object(os, "open", return_value=83),
        patch.object(os, "close") as closed,
        patch.object(driver.fcntl, "ioctl", side_effect=OSError(errno.EBUSY, "busy")),
    ):
        with pytest.raises(OSError) as exc:
            driver.DeviceDMALease.acquire(0)
        assert exc.value.errno == errno.EBUSY
        closed.assert_called_once_with(83)


@pytest.mark.parametrize("field,value", [(0, 2), (1, 9), (2, 1), (4, 0), (6, 0), (7, 0), (10, 1)])
def test_invalid_reply_releases_open_file(field, value):
    def broken_reply(fd, command, buffer, mutate):
        complete_reply(fd, command, buffer, mutate)
        values = list(driver.LEASE_STRUCT.unpack(buffer))
        values[field] = value
        buffer[:] = driver.LEASE_STRUCT.pack(*values)

    with (
        patch.object(os, "open", return_value=83),
        patch.object(os, "close") as closed,
        patch.object(driver.fcntl, "ioctl", side_effect=broken_reply),
    ):
        with pytest.raises(RuntimeError):
            driver.DeviceDMALease.acquire(0)
        closed.assert_called_once_with(83)


def test_release_failure_preserves_ownership_for_retry():
    with (
        patch.object(os, "open", return_value=83),
        patch.object(os, "close") as closed,
        patch.object(driver.fcntl, "ioctl", side_effect=complete_reply) as ioctl,
    ):
        lease = driver.DeviceDMALease.acquire(0)
        ioctl.side_effect = OSError(errno.EBUSY, "undrained")
        with pytest.raises(OSError):
            lease.release()
        closed.assert_not_called()
        ioctl.side_effect = complete_reply
        lease.release()
        sent = driver.LEASE_STRUCT.unpack(ioctl.call_args.args[2])
        assert (sent[1], sent[3], sent[4]) == (0, 20, 7)
        closed.assert_called_once_with(83)
        lease.release()
        closed.assert_called_once()


def test_reset_invalidates_query_without_reacquisition():
    with (
        patch.object(os, "open", return_value=83),
        patch.object(os, "close") as closed,
        patch.object(driver.fcntl, "ioctl", side_effect=complete_reply) as ioctl,
    ):
        with driver.DeviceDMALease.acquire(0) as lease:
            ioctl.side_effect = OSError(errno.ENODEV, "device reset")
            with pytest.raises(OSError) as exc:
                lease.query()
            assert exc.value.errno == errno.ENODEV
            assert ioctl.call_count == 2
        closed.assert_called_once_with(83)


def test_no_reservations_by_default():
    with patch.object(os, "open") as opened:
        with driver.reserve_devices([]) as leases:
            assert leases == ()
        opened.assert_not_called()


@pytest.mark.parametrize("devices", [[1, 1], [-1], [True], [2**32]])
def test_group_validates_before_acquiring(devices):
    with patch.object(driver.DeviceDMALease, "acquire") as acquire:
        with pytest.raises(ValueError), driver.reserve_devices(devices):
            pytest.fail("invalid group admitted")
        acquire.assert_not_called()


def isolated_reply(fd, command, buffer, mutate):
    complete_reply(fd, command, buffer, mutate)
    fields = list(driver.LEASE_STRUCT.unpack(buffer))
    fields[5] = driver.CAP_HOST_ROUTING_ISOLATED
    buffer[:] = driver.LEASE_STRUCT.pack(*fields)


@pytest.mark.parametrize("failure", [None, "body", "second", "capability"])
def test_group_lifetime_and_partial_rollback(failure):
    def reply(fd, command, buffer, mutate):
        if failure == "second" and fd == 84:
            raise OSError(errno.EBUSY, "busy")
        (complete_reply if failure == "capability" else isolated_reply)(fd, command, buffer, mutate)

    with (
        patch.object(os, "open", side_effect=[83, 84]),
        patch.object(os, "close") as closed,
        patch.object(driver.fcntl, "ioctl", side_effect=reply),
    ):

        def run():
            with driver.reserve_devices([2, 7]) as leases:
                assert [lease.info.device_id for lease in leases] == [2, 7]
                closed.assert_not_called()
                if failure == "body":
                    raise RuntimeError("server startup failed")

        if failure:
            with pytest.raises((RuntimeError, OSError)):
                run()
        else:
            run()
        assert [call.args[0] for call in closed.call_args_list] == ([83] if failure == "capability" else [84, 83])
