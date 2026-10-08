# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline thermal hysteresis, failed-ack reconciliation and HTTP contract."""

import json
import threading
from http.client import IncompleteRead
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest

from tools.qwen4exp import thermal_controller
from tools.qwen4exp.thermal_controller import EngineControl, ThermalController, parse_temperatures


class Engine:
    def __init__(self, paused=False):
        self.paused = paused
        self.calls = []
        self.pause_failure = None
        self.resume_failure = None
        self.requests = ["decode-in-flight", "queued-prefill"]
        self.caches = {"state": object()}

    def is_paused(self):
        return self.paused

    def pause(self):
        self.calls.append("pause")
        if self.pause_failure == "before":
            raise OSError("pause rejected")
        self.paused = True
        if self.pause_failure == "after":
            raise TimeoutError("lost pause acknowledgment")

    def resume(self):
        self.calls.append("resume")
        if self.resume_failure == "before":
            raise OSError("resume rejected")
        self.paused = False
        if self.resume_failure == "after":
            raise TimeoutError("lost resume acknowledgment")


def test_any_device_holds_at_94_and_every_device_must_cool_to_85():
    engine = Engine()
    requests, caches = engine.requests, engine.caches
    controller = ThermalController(engine)
    assert controller.step([93, 80, 82, 84])["action"] == "none"
    assert controller.step([80, 94, 82, 84])["action"] == "pause"
    for temperatures in ([95, 95, 95, 95], [86, 85, 84, 83], [85, 85, 85, 86]):
        assert controller.step(temperatures)["action"] == "holding"
    assert controller.step([85, 84, 83, 85])["action"] == "resume"
    assert controller.step([93, 93, 93, 93])["action"] == "none"
    assert controller.step([94, 84, 84, 84])["action"] == "pause"
    assert engine.calls == ["pause", "resume", "pause"]
    assert engine.requests is requests and engine.caches is caches


def test_existing_manual_pause_is_never_resumed():
    engine = Engine(paused=True)
    controller = ThermalController(engine)
    assert controller.step([94] * 4)["action"] == "external_pause"
    controller.step([84] * 4)
    assert engine.paused and engine.calls == [] and not controller.owns_pause


@pytest.mark.parametrize("temperatures", [None, [], [84] * 3, [float("nan")] * 4, [float("inf")] * 4, [-1] * 4])
def test_missing_or_invalid_sensor_data_holds_until_valid_cool_sample(temperatures):
    engine = Engine()
    controller = ThermalController(engine)
    assert controller.step(temperatures)["action"] == "pause"
    assert controller.step([86] * 4)["action"] == "holding"
    assert controller.step([85] * 4)["action"] == "resume"


@pytest.mark.parametrize("failure", ["before", "after"])
def test_lost_pause_ack_is_reconciled_without_clearing_requests(failure):
    engine = Engine()
    engine.pause_failure = failure
    controller = ThermalController(engine)
    with pytest.raises(OSError):
        controller.step([94] * 4)
    engine.pause_failure = None
    controller.step([85] * 4)
    assert not engine.paused
    assert engine.calls == (["pause"] if failure == "before" else ["pause", "resume"])
    assert engine.requests == ["decode-in-flight", "queued-prefill"]


def test_failed_resume_does_not_release_latch_at_86():
    engine = Engine()
    controller = ThermalController(engine)
    controller.step([94] * 4)
    engine.resume_failure = "before"
    with pytest.raises(OSError):
        controller.step([85] * 4)
    assert controller.step([86] * 4)["action"] == "holding"
    assert engine.paused and engine.calls == ["pause", "resume"]
    engine.resume_failure = None
    assert controller.step([85] * 4)["action"] == "resume"


def test_lost_resume_ack_does_not_send_another_resume():
    engine = Engine()
    controller = ThermalController(engine)
    controller.step([94] * 4)
    engine.resume_failure = "after"
    with pytest.raises(OSError):
        controller.step([85] * 4)
    controller.step([85] * 4)
    assert engine.calls == ["pause", "resume"] and not controller.owns_pause


def test_hot_external_resume_is_reheld_until_low_target():
    engine = Engine()
    controller = ThermalController(engine)
    controller.step([94] * 4)
    engine.paused = False
    assert controller.step([90] * 4)["action"] == "pause"
    assert controller.step([85] * 4)["action"] == "resume"


def test_monitor_exit_on_pid_reuse_leaves_owned_hold(monkeypatch):
    engine = Engine()
    controller = ThermalController(engine)
    identities = iter(["original", "original", "reused"])
    monkeypatch.setattr(thermal_controller, "process_identity", lambda pid: next(identities))
    monkeypatch.setattr(thermal_controller.subprocess, "check_output", lambda *args, **kwargs: "ignored")
    monkeypatch.setattr(thermal_controller, "parse_temperatures", lambda *args: [94] * 4)
    monkeypatch.setattr(thermal_controller.time, "sleep", lambda seconds: None)
    thermal_controller.run(controller, 100, 1, 2)
    assert engine.paused and engine.calls == ["pause"]


def test_monitor_retries_incomplete_http_response_without_resuming(monkeypatch, capsys):
    engine = Engine()
    controller = ThermalController(engine)
    identities = iter(["original", "original", "original", "gone"])
    reads = iter([IncompleteRead(b"", 1), False])

    def pause_state():
        result = next(reads)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(engine, "is_paused", pause_state)
    monkeypatch.setattr(thermal_controller, "process_identity", lambda pid: next(identities))
    monkeypatch.setattr(thermal_controller.subprocess, "check_output", lambda *args, **kwargs: "ignored")
    monkeypatch.setattr(thermal_controller, "parse_temperatures", lambda *args: [94] * 4)
    monkeypatch.setattr(thermal_controller.time, "sleep", lambda seconds: None)
    thermal_controller.run(controller, 100, 1, 2)
    actions = [json.loads(line)["action"] for line in capsys.readouterr().out.splitlines()]
    assert actions == ["control_error", "pause"]
    assert engine.paused and engine.calls == ["pause"]


def test_parse_real_smi_columns_and_reject_partial_readings():
    rows = [f"| 8217 310P3 | OK | NA | {value} | 0 / 0 |" for value in (93, 94, 85, 84)]
    assert parse_temperatures("\n".join(rows)) == [93, 94, 85, 84]
    with pytest.raises(ValueError):
        parse_temperatures("\n".join(rows[:-1]))
    with pytest.raises(ValueError):
        parse_temperatures("\n".join(rows).replace("94", "NA"))


@pytest.mark.parametrize("high,low", [(85, 94), (85, 85), (94, -1), (float("nan"), 85)])
def test_invalid_thresholds_are_rejected(high, low):
    with pytest.raises(ValueError):
        ThermalController(Engine(), high, low)


def test_loopback_http_uses_keep_mode_without_cache_reset():
    state = SimpleNamespace(paused=False, calls=[])

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            assert self.path == "/is_paused"
            self.respond({"is_paused": state.paused})

        def do_POST(self):
            state.calls.append(self.path)
            if self.path == "/pause?mode=keep&clear_cache=false":
                state.paused = True
                self.respond({"status": "paused"})
            elif self.path == "/resume":
                state.paused = False
                self.respond({"status": "resumed"})
            else:
                raise AssertionError("Unexpected thermal control request")

        def respond(self, value):
            body = json.dumps(value).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        controller = ThermalController(EngineControl(f"http://127.0.0.1:{server.server_port}"))
        controller.step([94] * 4)
        controller.step([85] * 4)
        assert state.calls == ["/pause?mode=keep&clear_cache=false", "/resume"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.mark.parametrize("url", ["http://192.168.53.187:8001", "https://localhost", "http://user:pass@localhost"])
def test_control_urls_must_be_plain_loopback(url):
    with pytest.raises(ValueError):
        EngineControl(url)
