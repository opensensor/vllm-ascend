# SPDX-License-Identifier: Apache-2.0
"""Admission check for repeated 310P benchmarks, not a hardware safety guarantee."""

import argparse
import contextlib
import json
import subprocess
from pathlib import Path

import regex as re

DEFAULT_MAX_TEMPERATURE_C = 90
QUERY_TIMEOUT_SECONDS = 8
REAP_TIMEOUT_SECONDS = 1


def parse_devices(text):
    devices = []
    pending = None
    for line in text.splitlines():
        cells = [cell.strip() for cell in line.split("|")[1:-1]]
        if len(cells) != 3:
            continue
        identity, state, values = cells
        fields = identity.split()
        if len(fields) == 2 and fields[0].isdigit() and fields[1].startswith("310P"):
            if pending is not None:
                raise ValueError("missing chip/device row")
            readings = values.split()
            if len(readings) < 2:
                raise ValueError("missing 310P temperature")
            pending = dict(card=int(fields[0]), name=fields[1], health=state, temperature_c=int(readings[1]))
        elif pending is not None and re.fullmatch(r"[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.\d", state):
            if len(fields) != 2 or not all(field.isdigit() for field in fields):
                raise ValueError("invalid chip/device mapping")
            devices.append(dict(pending, chip=int(fields[0]), device=int(fields[1]), pci=state))
            pending = None
    if pending is not None or not devices:
        raise ValueError("incomplete 310P device report")
    if len({device["device"] for device in devices}) != len(devices):
        raise ValueError("duplicate device ID in management report")
    return devices


def admit(text, *, devices=(0, 1, 2, 3), max_temperature_c=DEFAULT_MAX_TEMPERATURE_C):
    if not devices or len(set(devices)) != len(devices) or any(device < 0 for device in devices):
        raise ValueError("expected device IDs must be nonempty, unique and nonnegative")
    if not 0 < max_temperature_c <= 120:
        raise ValueError("benchmark temperature limit must be in (0,120] C")
    observed = parse_devices(text)
    by_id = {device["device"]: device for device in observed}
    reasons = []
    for identifier in devices:
        row = by_id.get(identifier)
        if row is None:
            reasons.append(f"missing device {identifier}")
        elif row["health"] != "OK" or not 0 <= row["temperature_c"] < max_temperature_c:
            reasons.append(f"device {identifier}: health={row['health']}, temperature={row['temperature_c']} C")
    return dict(passed=not reasons, limit_c=max_temperature_c, devices=observed, reasons=reasons)


def query():
    process = subprocess.Popen(["npu-smi", "info"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        stdout, stderr = process.communicate(timeout=QUERY_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired as error:
        process.kill()
        # A task in an uninterruptible driver ioctl may not reap immediately.
        # Never turn a bounded preflight into an unbounded wait after SIGKILL.
        with contextlib.suppress(subprocess.TimeoutExpired):
            process.communicate(timeout=REAP_TIMEOUT_SECONDS)
        raise RuntimeError(f"NPU management query timed out; inspect process {process.pid}") from error
    if process.returncode:
        raise RuntimeError(f"NPU management query failed ({process.returncode}): {stderr.strip()}")
    return stdout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=Path, help="validate a saved npu-smi report without querying hardware")
    parser.add_argument("--max-temperature-c", type=int, default=DEFAULT_MAX_TEMPERATURE_C)
    parser.add_argument("--devices", type=int, nargs="+", default=(0, 1, 2, 3))
    args = parser.parse_args()
    try:
        text = args.sample.read_text() if args.sample else query()
        result = admit(text, devices=args.devices, max_temperature_c=args.max_temperature_c)
    except (OSError, ValueError, RuntimeError) as error:
        result = dict(passed=False, error=str(error))
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 2)


if __name__ == "__main__":
    main()
