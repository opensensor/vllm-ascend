# SPDX-License-Identifier: Apache-2.0
"""Bounded resident event sample; always restore the qualified fused candidate."""

import hashlib
import json
import time
import uuid
from pathlib import Path

from tools.glm_perf.resident_control import Control
from tools.glm_perf.resident_harness import ResidentClient


def main():
    root = Path(__file__).resolve().parent
    baseline = Path("/home/matteius/experiments/glm-prompt-profile-20261005/fused-baseline-candidate.py").read_text()
    expected = "348947ff4f965d2b6af532269f4b0e90eee1ba201d3e16dd60a063b8de423282"
    assert hashlib.sha256(baseline.encode()).hexdigest() == expected
    source = baseline.replace("def replacements(native_resources):", "def qualified_replacements(native_resources):")
    assert source != baseline
    source += "\n" + (root / "event-candidate.py").read_text()
    (root / "applied-events.py").write_text(source)
    client = ResidentClient("http://127.0.0.1:8001", timeout=300)
    before = client.rpc("resident_status")
    assert all(row["digest"] == expected and not row["graphs_dirty"] and not row.get("native_failed") for row in before)
    assert not client.request("/is_paused", method="GET")["is_paused"]
    (root / "events-before.json").write_text(json.dumps(before, indent=2) + "\n")
    try:
        client.switch(Control(uuid.uuid4().hex, candidate="prefill_events", source=source))
        print("event candidate captured", flush=True)
        # This RPC only arms closure state. It does not start torch_npu.profiler.
        armed = client.rpc("profile", True, "bounded_640")
        assert all(row.get("event_sampling") for row in armed), armed
        prompt = (
            f"# Event sample {uuid.uuid4().hex}\n"
            + "def update(items, key, value):\n    items[key] = value\n    return items\n" * 150
        )
        tokens = client.request("/tokenize", {"model": "glm53-flash-selective-w3", "prompt": prompt})["tokens"][:640]
        assert len(tokens) == 640
        (root / "events-metrics-before.txt").write_text(raw_metrics(client))
        began = time.perf_counter()
        output = client.request(
            "/v1/completions",
            {
                "model": "glm53-flash-selective-w3",
                "prompt": tokens,
                "max_tokens": 1,
                "temperature": 0,
                "ignore_eos": True,
            },
        )
        (root / "events-request.json").write_text(
            json.dumps({"output": output, "elapsed_s": time.perf_counter() - began}, indent=2) + "\n"
        )
        assert output["usage"]["prompt_tokens"] == 640 and output["usage"]["completion_tokens"] == 1
        client.request("/pause?mode=wait&clear_cache=true")
        client.rpc("profile", False, "bounded_640")
        measured = client.rpc("resident_status")
        (root / "events-measured.json").write_text(json.dumps(measured, indent=2) + "\n")
        (root / "events-metrics-after.txt").write_text(raw_metrics(client))
        assert all(
            row["prefill_events"]["records"] and all(r["ready"] for r in row["prefill_events"]["records"])
            for row in measured
        )
        print("bounded prompt sample complete", flush=True)
    finally:
        restored = client.switch(Control(uuid.uuid4().hex, candidate="completed_pools_sinkhorn", source=baseline))
        client.request("/resume")
        (root / "events-restored.json").write_text(json.dumps(restored, indent=2) + "\n")
        assert sorted((r["pid"], r["weight_storage_digest"]) for r in restored) == sorted(
            (r["pid"], r["weight_storage_digest"]) for r in before
        )
        print("qualified fusion restored; same workers and weights", flush=True)
    warmup = client.request(
        "/v1/completions",
        {
            "model": "glm53-flash-selective-w3",
            "prompt": "def add(a,b): return",
            "max_tokens": 8,
            "temperature": 0,
            "ignore_eos": True,
        },
    )
    (root / "events-restored-request.json").write_text(json.dumps(warmup, indent=2) + "\n")
    assert warmup["usage"]["completion_tokens"] == 8


def raw_metrics(client):
    import urllib.request

    with urllib.request.urlopen(client.base_url + "/metrics", timeout=30) as response:
        return response.read().decode()


if __name__ == "__main__":
    main()
