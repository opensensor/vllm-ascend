# SPDX-License-Identifier: Apache-2.0
"""Collect one already-running MTP arm, without changing its configuration.

Requires later NPU authorization. Requests and cooldown holds count toward
sustained wall throughput. A separate quality/image receipt must match the
serving alias and draft depth. The collector never starts or pauses a server.
"""

import argparse
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import median

from tools.qwen4exp.thermal_controller import process_identity
from tools.qwen38_decode_study.benchmark import metrics, request_json, stream_completion

MAX_TELEMETRY_GAP_SECONDS = 10
HOLD_C = 94
RESUME_C = 85
HARD_STOP_C = 96


def thermal_evidence(path: Path, start: float, end: float) -> dict:
    samples = []
    for line in path.read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue  # A concurrently appended final line can be incomplete.
        if isinstance(row.get("time"), (int, float)) and start - MAX_TELEMETRY_GAP_SECONDS <= row["time"] <= end:
            samples.append(row)
    if not samples:
        return {"max_core_c": None, "thermal_policy_pass": False, "thermal_shutdown": None}
    samples.sort(key=lambda row: row["time"])
    valid = samples[0]["time"] <= start and end - samples[-1]["time"] <= MAX_TELEMETRY_GAP_SECONDS
    temperatures = []
    previous = samples[0]["time"]
    latched = samples[0].get("cooling") is True
    for row in samples:
        valid &= row["time"] - previous <= MAX_TELEMETRY_GAP_SECONDS
        previous = row["time"]
        values = row.get("temperatures_c")
        if (
            not isinstance(values, list)
            or len(values) != 4
            or any(
                not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 150
                for value in values
            )
        ):
            valid = False
            continue
        maximum = max(values)
        temperatures.append(maximum)
        if maximum >= HOLD_C:
            latched = True
        if latched and maximum > RESUME_C:
            valid &= row.get("cooling") is True and row.get("action") in ("pause", "holding", "external_pause")
        if row.get("action") == "resume":
            valid &= maximum <= RESUME_C
        if maximum <= RESUME_C:
            latched = False
        valid &= row.get("sensor_valid") is True and row.get("action") != "control_error"
    maximum = max(temperatures) if temperatures else None
    return {
        "max_core_c": maximum,
        "thermal_policy_pass": bool(valid and maximum is not None and maximum < HARD_STOP_C),
        "thermal_shutdown": None,
        "hard_thermal_limit_exceeded": maximum >= HARD_STOP_C if maximum is not None else None,
    }


def collect_group(base, body, concurrency):
    def one():
        started = time.perf_counter()
        result = stream_completion(base, body)
        result["first_time"] = started + result["ttft_s"]
        result["last_time"] = result["first_time"] + result["decode_s"]
        if result["usage"]["completion_tokens"] != body["max_tokens"]:
            raise RuntimeError("fixed-output-length request ended early; arm incomplete")
        return result

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        responses = list(executor.map(lambda _: one(), range(concurrency)))
    return {
        "responses": responses,
        "wall_seconds": time.perf_counter() - started,
        "decode_seconds": max(r["last_time"] for r in responses) - min(r["first_time"] for r in responses),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-pid", type=int, required=True)
    parser.add_argument("--draft-length", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--quality-receipt", type=Path, required=True)
    parser.add_argument("--thermal-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--concurrencies", type=int, nargs="+", choices=(1, 2, 3), default=[1, 2, 3])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--min-arm-seconds", type=float, default=600)
    args = parser.parse_args()
    if (
        args.output.exists()
        or args.repeats < 3
        or args.min_arm_seconds < 600
        or not math.isfinite(args.min_arm_seconds)
    ):
        parser.error("new output, at least three repeats and >=600 sustained seconds required")
    body = json.loads(args.request.read_text())
    if type(body.get("max_tokens")) is not int or body["max_tokens"] < 64:
        parser.error("request needs max_tokens >=64")
    body.update(model=args.model, stream=True, stream_options={"include_usage": True}, ignore_eos=True)
    receipt = json.loads(args.quality_receipt.read_text())
    if receipt.get("model") != args.model or receipt.get("draft_length") != args.draft_length:
        parser.error("quality receipt model/depth differs from this arm")
    ids = {entry["id"] for entry in request_json(args.base_url, "/v1/models")["data"]}
    if args.model not in ids:
        parser.error("serving alias is not present")
    identity = process_identity(args.api_pid)
    if identity is None:
        parser.error("run collector on serving host with a live API PID")
    rows = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output:
        for concurrency in args.concurrencies:
            before = metrics(args.base_url)
            if any(before.get(name, 0) for name in ("vllm:num_requests_running", "vllm:num_requests_waiting")):
                raise RuntimeError("server has other traffic; arm comparison would be contaminated")
            start = time.time()
            thermal = thermal_evidence(args.thermal_log, start, start)
            if thermal["max_core_c"] is None or not thermal["thermal_policy_pass"]:
                raise RuntimeError("fresh complete thermal supervision is required")
            arm_rows = []
            try:
                for repeat in range(args.repeats):
                    groups = []
                    began = time.monotonic()
                    while not groups or time.monotonic() - began < args.min_arm_seconds / args.repeats:
                        if process_identity(args.api_pid) != identity:
                            raise RuntimeError("API identity changed; arm incomplete")
                        groups.append(collect_group(args.base_url, body, concurrency))
                    responses = [response for group in groups for response in group["responses"]]
                    arm_rows.append(
                        {
                            "draft_length": args.draft_length,
                            "concurrency": concurrency,
                            "repeat": repeat,
                            "request": body,
                            "completion_tokens": sum(r["usage"]["completion_tokens"] for r in responses),
                            "decode_rounds": len(groups),
                            "decode_seconds": sum(g["decode_seconds"] for g in groups),
                            "wall_seconds": sum(g["wall_seconds"] for g in groups),
                            "ttft_seconds": median(r["ttft_s"] for r in responses),
                            "quality_pass": receipt.get("quality_pass") is True,
                            "image_pass": receipt.get("image_pass") is True,
                            "request_ids": [r["request_id"] for r in responses],
                            "text_hashes": [r["text_sha256"] for r in responses],
                            "prompt_tokens": [r["usage"]["prompt_tokens"] for r in responses],
                            "timing_source": "client_stream_chunks_wall_includes_cooldown_holds",
                        }
                    )
            except Exception as error:
                # No partial arm can be selected as a successful thermal run.
                output.write(
                    json.dumps(
                        {
                            "arm_failed": True,
                            "draft_length": args.draft_length,
                            "concurrency": concurrency,
                            "error": str(error),
                        }
                    )
                    + "\n"
                )
                output.flush()
                raise
            if process_identity(args.api_pid) != identity:
                output.write(json.dumps({"arm_failed": True, "error": "API identity changed"}) + "\n")
                raise RuntimeError("API identity changed; arm incomplete")
            end = time.time()
            thermal = thermal_evidence(args.thermal_log, start, end)
            if thermal["max_core_c"] is None:
                raise RuntimeError("thermal evidence missing; arm incomplete")
            for row in arm_rows:
                row.update(sustained_seconds=end - start, **thermal)
                row["thermal_shutdown"] = False  # All requests completed under one unchanged API identity.
                output.write(json.dumps(row) + "\n")
                rows.append(row)
            output.flush()
    print(json.dumps({"rows": len(rows), "server_modified": False}))


if __name__ == "__main__":
    main()
