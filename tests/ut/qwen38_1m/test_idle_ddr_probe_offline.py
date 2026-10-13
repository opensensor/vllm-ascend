# SPDX-License-Identifier: Apache-2.0
"""Temperature admission must fail before any device child can start."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tools.qwen4exp import benchmark_idle_ddr_310 as probe


def snapshot(temperatures):
    return "\n".join(f"| {index} 310P3 | OK | NA {value} 0 / 0 |" for index, value in enumerate(temperatures))


@pytest.mark.parametrize("temperatures", [[45, 46, 55, 54, 72, 70], [72] * 6])
def test_all_six_cool_chips_are_admitted(temperatures):
    assert probe.admit(snapshot(temperatures)) == temperatures


@pytest.mark.parametrize("temperatures", [[45, 46, 55, 54, 72.1, 70], [95] * 6, [40] * 5])
def test_hot_or_missing_chip_never_spawns(tmp_path, monkeypatch, temperatures):
    spawn = Mock(side_effect=AssertionError("must not spawn on failed admission"))
    monkeypatch.setattr(probe.multiprocessing, "get_context", spawn)
    monkeypatch.setattr(probe.subprocess, "check_output", Mock(return_value=snapshot(temperatures)))
    args = SimpleNamespace(output=tmp_path / "probe", phase="device")
    with pytest.raises((RuntimeError, ValueError)):
        probe.run(args)
    spawn.assert_not_called()
