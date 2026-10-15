# SPDX-License-Identifier: Apache-2.0
import subprocess
from unittest.mock import Mock

import pytest

from tools.qwen4exp import thermal_guard as guard


def sample(temperature=82, health="OK"):
    return f"""
| NPU Name | Health | Power(W) Temp(C) Hugepages-Usage(page) |
| 16409 310P3 | {health} | NA {temperature} 22618 / 22618 |
| 0 2 | 0000:49:00.0 | 0 47211/47302 |
| NPU Chip | Process id | Process name | Process memory(MB) |
| 16409 0 | 1266007 | VLLMWorker_TP | 45801 |
"""


def test_parse_mapping_and_ignore_process_rows():
    rows = guard.parse_devices(sample())
    assert rows == [dict(card=16409, name="310P3", health="OK", temperature_c=82, chip=0, device=2, pci="0000:49:00.0")]
    assert guard.admit(sample(), devices=(2,))["passed"]
    assert not guard.admit(sample())["passed"]


@pytest.mark.parametrize("temperature,health", [(90, "OK"), (102, "OK"), (82, "Warning"), (-1, "OK")])
def test_unhealthy_or_hot_device_blocks_benchmark(temperature, health):
    result = guard.admit(sample(temperature, health), devices=(2,))
    assert not result["passed"] and result["reasons"]


@pytest.mark.parametrize("text", ["", "device query failed", sample().replace("NA 82", "NA NA"), sample() * 2])
def test_malformed_or_duplicate_reports_fail_closed(text):
    with pytest.raises(ValueError):
        guard.parse_devices(text)


def test_timeout_never_waits_indefinitely_for_uninterruptible_ioctl(monkeypatch):
    process = Mock(pid=123)
    process.communicate.side_effect = subprocess.TimeoutExpired("npu-smi", 8)
    monkeypatch.setattr(guard.subprocess, "Popen", Mock(return_value=process))
    with pytest.raises(RuntimeError, match="timed out.*123"):
        guard.query()
    process.kill.assert_called_once()
    assert process.communicate.call_args_list[0].kwargs["timeout"] == guard.QUERY_TIMEOUT_SECONDS
    assert process.communicate.call_args_list[1].kwargs["timeout"] == guard.REAP_TIMEOUT_SECONDS


def test_unsuccessful_management_query_cannot_admit(monkeypatch):
    process = Mock(returncode=1)
    process.communicate.return_value = (sample(), "driver unavailable")
    monkeypatch.setattr(guard.subprocess, "Popen", Mock(return_value=process))
    with pytest.raises(RuntimeError, match="driver unavailable"):
        guard.query()


@pytest.mark.parametrize("devices", [(), (2, 2), (-1,)])
def test_invalid_expected_device_set_is_rejected(devices):
    with pytest.raises(ValueError, match="device IDs"):
        guard.admit(sample(), devices=devices)
