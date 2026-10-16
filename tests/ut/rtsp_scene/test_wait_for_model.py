import urllib.error
from unittest.mock import Mock

import pytest

from tools.rtsp_scene.wait_for_model import wait_for_health


class Clock:
    def __init__(self):
        self.now = 0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def test_readiness_retries_connection_failure_and_unhealthy_status():
    clock = Clock()
    probe = Mock(side_effect=[urllib.error.URLError("refused"), False, True])
    wait_for_health("http://localhost/health", 10, 2, probe=probe, clock=clock, sleep=clock.sleep)
    assert probe.call_count == 3 and clock.now == 4
    probe.assert_called_with("http://localhost/health", 2)


def test_readiness_times_out_without_exceeding_deadline():
    clock = Clock()
    probe = Mock(return_value=False)
    with pytest.raises(TimeoutError, match="did not become ready"):
        wait_for_health("http://localhost/health", 5, 2, probe=probe, clock=clock, sleep=clock.sleep)
    assert clock.now == 5 and probe.call_count == 3
    probe.assert_called_with("http://localhost/health", 1)


@pytest.mark.parametrize("timeout,interval", [(0, 2), (10, -1), (float("inf"), 2), (10, float("nan"))])
def test_readiness_rejects_invalid_bounds(timeout, interval):
    with pytest.raises(ValueError, match="finite positive"):
        wait_for_health("http://localhost/health", timeout, interval)
