# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hold vLLM requests at 94C and preserve their state until all NPUs cool to 85C.

This controller never launches a server, offloads weights, or clears caches.
Use the local development pause API; retain a separate emergency watchdog.
Only this controller should use pause/resume while it owns a thermal hold.
"""

import argparse
import json
import math
import subprocess
import time
import urllib.request
from http.client import HTTPException
from pathlib import Path
from urllib.parse import urlsplit

HOLD_TEMPERATURE_C = 94
RESUME_TEMPERATURE_C = 85
EXPECTED_DEVICES = 4
MAX_SENSOR_TEMPERATURE_C = 150
SAMPLE_TIMEOUT_S = 2
API_TIMEOUT_S = 5
POLL_INTERVAL_S = 1


def parse_temperatures(output, expected_devices=EXPECTED_DEVICES):
    temperatures = []
    for line in output.splitlines():
        if "310P" in line:
            fields = [field.strip() for field in line.strip().strip("|").split("|")]
            if len(fields) == 3:
                # Current 26.0.rc1 output also folds hugepage usage into
                # the same cell: "NA 61 0 / 0". Do not mistake page counts
                # for temperature or accept truncated sensor rows.
                power_temperature_pages = fields[2].split()
                if len(power_temperature_pages) != 5 or power_temperature_pages[3] != "/":
                    raise ValueError("Incomplete NPU power/temperature/hugepage row")
                temperature = power_temperature_pages[1]
            elif len(fields) == 4:
                # npu-smi 26.0.rc1 groups Power(W) and Temp(C) in one
                # physical table cell. Hugepage usage follows that cell.
                power_temperature = fields[2].split()
                if len(power_temperature) != 2:
                    raise ValueError("Incomplete NPU power/temperature row")
                temperature = power_temperature[1]
            elif len(fields) == 5:
                temperature = fields[3]
            else:
                raise ValueError("Incomplete NPU temperature row")
            temperatures.append(float(temperature))
    if len(temperatures) != expected_devices or any(
        not math.isfinite(value) or not 0 <= value <= MAX_SENSOR_TEMPERATURE_C for value in temperatures
    ):
        raise ValueError("Expected valid temperature readings for every NPU")
    return temperatures


class EngineControl:
    def __init__(self, base_url, timeout=API_TIMEOUT_S):
        parsed = urlsplit(base_url)
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("API timeout must be finite and positive")
        if (
            parsed.scheme != "http"
            or parsed.hostname not in {"localhost", "127.0.0.1", "::1"}
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path not in {"", "/"}
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("Thermal controls require a plain loopback HTTP base URL")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def request(self, path, method):
        request = urllib.request.Request(self.base_url + path, data=b"" if method == "POST" else None, method=method)
        with self.opener.open(request, timeout=self.timeout) as response:
            result = json.loads(response.read())
        if not isinstance(result, dict):
            raise RuntimeError("Engine returned a non-object control response")
        return result

    def is_paused(self):
        paused = self.request("/is_paused", "GET").get("is_paused")
        if not isinstance(paused, bool):
            raise RuntimeError("Engine returned an invalid pause state")
        return paused

    def pause(self):
        if self.request("/pause?mode=keep&clear_cache=false", "POST").get("status") != "paused":
            raise RuntimeError("Engine did not acknowledge the thermal hold")

    def resume(self):
        if self.request("/resume", "POST").get("status") != "resumed":
            raise RuntimeError("Engine did not acknowledge resuming generation")


class ThermalController:
    def __init__(
        self,
        engine,
        high_c=HOLD_TEMPERATURE_C,
        low_c=RESUME_TEMPERATURE_C,
        expected_devices=EXPECTED_DEVICES,
    ):
        if not (math.isfinite(low_c) and math.isfinite(high_c) and 0 <= low_c < high_c):
            raise ValueError("Thermal thresholds require 0 <= low < high")
        if expected_devices < 1:
            raise ValueError("Expected device count must be positive")
        self.engine = engine
        self.high_c = high_c
        self.low_c = low_c
        self.expected_devices = expected_devices
        self.cooling = False
        self.owns_pause = False

    def step(self, temperatures):
        valid = (
            temperatures is not None
            and len(temperatures) == self.expected_devices
            and all(math.isfinite(value) and 0 <= value <= MAX_SENSOR_TEMPERATURE_C for value in temperatures)
        )
        maximum = max(temperatures) if valid else None
        if not valid or maximum >= self.high_c:
            self.cooling = True
        result = {"maximum_c": maximum, "sensor_valid": valid, "action": "none"}
        if self.cooling and (not valid or maximum > self.low_c):
            if not self.engine.is_paused():
                # Record the request before the HTTP call: a timeout can occur
                # after the engine applies it. A later state read reconciles it.
                self.owns_pause = True
                self.engine.pause()
                result["action"] = "pause"
            else:
                result["action"] = "holding" if self.owns_pause else "external_pause"
        elif self.cooling or self.owns_pause:
            if self.owns_pause and self.engine.is_paused():
                self.engine.resume()
                result["action"] = "resume"
            # Do not release the latch before a successful resume; failures
            # must not resume at 86-93C just because an earlier sample was 85C.
            self.owns_pause = False
            self.cooling = False
        return {**result, "cooling": self.cooling, "owns_pause": self.owns_pause}


def process_identity(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rpartition(")")[2].split()
        return None if fields[0] in {"Z", "X"} else fields[19]
    except (OSError, IndexError):
        return None


def run(controller, api_pid, poll_s, sample_timeout_s):
    identity = process_identity(api_pid)
    if identity is None:
        raise RuntimeError("API process is not alive")
    while process_identity(api_pid) == identity:
        sensor_error = None
        try:
            output = subprocess.check_output(["npu-smi", "info"], text=True, timeout=sample_timeout_s)
            temperatures = parse_temperatures(output, controller.expected_devices)
        except (OSError, subprocess.SubprocessError, ValueError) as error:
            temperatures = None
            sensor_error = str(error)
        try:
            result = controller.step(temperatures)
        except (OSError, ValueError, RuntimeError, HTTPException) as error:
            result = {"action": "control_error", "error": str(error), "owns_pause": controller.owns_pause}
        print(
            json.dumps({"time": time.time(), "temperatures_c": temperatures, "sensor_error": sensor_error, **result}),
            flush=True,
        )
        time.sleep(poll_s)
    # Exiting deliberately leaves an owned pause in place. Never resume a hot
    # server merely because its monitor was stopped or the API process died.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8001")
    parser.add_argument("--api-pid", type=int, required=True)
    parser.add_argument("--high-c", type=float, default=HOLD_TEMPERATURE_C)
    parser.add_argument("--low-c", type=float, default=RESUME_TEMPERATURE_C)
    parser.add_argument("--expected-devices", type=int, default=EXPECTED_DEVICES)
    parser.add_argument("--poll-s", type=float, default=POLL_INTERVAL_S)
    parser.add_argument("--sample-timeout-s", type=float, default=SAMPLE_TIMEOUT_S)
    args = parser.parse_args()
    if args.api_pid < 1 or any(
        not math.isfinite(value) or value <= 0 for value in (args.poll_s, args.sample_timeout_s)
    ):
        parser.error("PID, polling interval and sensor timeout must be positive")
    controller = ThermalController(EngineControl(args.base_url), args.high_c, args.low_c, args.expected_devices)
    run(controller, args.api_pid, args.poll_s, args.sample_timeout_s)


if __name__ == "__main__":
    main()
