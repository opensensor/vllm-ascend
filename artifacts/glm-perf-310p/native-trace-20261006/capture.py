# SPDX-License-Identifier: Apache-2.0
"""Capture c1/c4 CANN traces on resident weights with deterministic cleanup."""

import json
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tools.glm_perf.resident_control import Control
from tools.glm_perf.resident_harness import ResidentClient


def mutate_profile(client, start, label):
    def matches(receipts):
        return all(
            (w.get("cann_active") is start and (not start or w.get("label") == label))
            or (
                w.get("cann_profile", {}).get("active") is start
                and (not start or w["cann_profile"].get("label") == label)
            )
            for w in receipts
        )

    return client._acknowledged_rpc("profile", start, label, matches=matches)


def main():
    root = Path(__file__).resolve().parent
    qualification = json.loads(
        Path("/home/matteius/experiments/glm-decode-trace-20261006/isolated-qualification.json").read_text()
    )
    assert qualification["passed"]
    baseline = Path("/home/matteius/experiments/glm-decode-flags-20261006/resident-combine.py").read_text()
    source = baseline.replace("def replacements(native_resources):", "def profile_base_replacements(native_resources):")
    source += "\n" + (root / "acl-profile.py").read_text() + "\n" + (root / "candidate.py").read_text()
    (root / "applied-candidate.py").write_text(source)
    client = ResidentClient("http://127.0.0.1:8001", timeout=900)
    before = client._acknowledged_rpc("resident_status", matches=client._status_matches)
    assert all(
        w["candidate"] == "native_disk_v29" and not w["graphs_dirty"] and not w.get("native_failed") for w in before
    )
    for worker in before:
        environment = dict(
            entry.split(b"=", 1)
            for entry in Path(f"/proc/{worker['pid']}/environ").read_bytes().split(b"\0")
            if b"=" in entry
        )
        assert environment.get(b"ASCEND_LAUNCH_BLOCKING", b"0") in (b"0", b""), "blocking launch profiling is invalid"
    (root / "before.json").write_text(json.dumps(before, indent=2) + "\n")
    active = False
    try:
        client.switch(Control(uuid.uuid4().hex, candidate="native_v29_cann_decode_trace", source=source))
        print("trace candidate captured with existing weights", flush=True)
        for label, concurrency in (("c1", 1), ("c4", 4)):
            client.request("/pause?mode=wait&clear_cache=true")
            # Mutate start once, then acknowledge via pure status (never repeat start).
            active = True
            mutate_profile(client, True, label)
            status = client._acknowledged_rpc("resident_status", matches=client._status_matches)
            assert all(w["cann_profile"]["active"] and w["cann_profile"]["label"] == label for w in status)
            client.request("/resume")

            def request(index):
                began = time.time_ns()
                payload = {
                    "model": "glm53-flash-selective-w3",
                    "prompt": f"# {uuid.uuid4().hex}\ndef add_{index}(a, b):\n    return",
                    "max_tokens": 32,
                    "temperature": 0,
                    "ignore_eos": True,
                }
                result = client.request("/v1/completions", payload)
                ended = time.time_ns()
                assert result["usage"]["completion_tokens"] == 32
                return {"index": index, "start_wall_ns": began, "end_wall_ns": ended, "response": result}

            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                rows = list(executor.map(request, range(concurrency)))
            (root / f"{label}-requests.json").write_text(json.dumps(rows, indent=2) + "\n")
            client.request("/pause?mode=wait&clear_cache=false")
            stopped = mutate_profile(client, False, label)
            status = client._acknowledged_rpc("resident_status", matches=client._status_matches)
            assert all(not w["cann_profile"]["active"] and not w["cann_profile"]["initialized"] for w in status)
            active = False
            (root / f"{label}-stopped.json").write_text(json.dumps(stopped, indent=2) + "\n")
            print(f"{label} completed, collection finalized on all ranks", flush=True)
    finally:
        if active:
            client.request("/pause?mode=wait&clear_cache=false")
            mutate_profile(client, False, "cleanup")
            status = client._acknowledged_rpc("resident_status", matches=client._status_matches)
            assert all(not w["cann_profile"]["active"] for w in status), "stop must precede graph recapture"
        restored = client.switch(Control(uuid.uuid4().hex, candidate="native_disk_v29", source=baseline))
        client.request("/resume")
        (root / "restored.json").write_text(json.dumps(restored, indent=2) + "\n")
        assert sorted((w["pid"], w["weight_storage_digest"]) for w in before) == sorted(
            (w["pid"], w["weight_storage_digest"]) for w in restored
        )
        print("qualified serving restored, same workers and weights", flush=True)
    result = client.request(
        "/v1/completions",
        {
            "model": "glm53-flash-selective-w3",
            "prompt": "def multiply(a, b): return",
            "max_tokens": 8,
            "ignore_eos": True,
        },
    )
    (root / "post-trace-request.json").write_text(json.dumps(result, indent=2) + "\n")
    assert result["usage"]["completion_tokens"] == 8


if __name__ == "__main__":
    main()
