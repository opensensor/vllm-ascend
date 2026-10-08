# SPDX-License-Identifier: Apache-2.0
"""Run on the inference host; always restore the exact resident candidate."""

import hashlib
import json
import time
import uuid
from dataclasses import replace
from pathlib import Path

from tools.glm_perf.resident_control import Control
from tools.glm_perf.resident_harness import ResidentClient
from tools.glm_perf.suite import load_tokenizer_json, retrieval_case, run_request

root = Path(__file__).resolve().parent
client = ResidentClient("http://127.0.0.1:8001", timeout=1200)
baseline = (root / "baseline-candidate.py").read_text()
digest = hashlib.sha256(baseline.encode()).hexdigest()
initial = client.rpc("resident_status")
(root / "initial-status.json").write_text(json.dumps(initial, indent=2))
if any(worker["digest"] != digest or worker["mode"] != "graph" for worker in initial):
    raise RuntimeError("server is not on the expected graph baseline")
count_tokens, tokenizer_identity = load_tokenizer_json(
    Path("/srv/ai/models/GLM-5.3-Flash-selective-W3-310p/tokenizer.json")
)
source = (
    baseline.replace("def replacements(", "def serving_replacements(")
    + "\n"
    + (root / "profile-candidate.py").read_text()
)
active = False
profile_nonce = uuid.uuid4().hex


def set_profile(enabled):
    # Dispatch once: older positive replies may still occupy rank queues.
    raw = client.request("/collective_rpc", {"method": "profile", "args": [enabled, profile_nonce], "timeout": 1200})
    (root / ("profile-start-raw.json" if enabled else "profile-stop-raw.json")).write_text(json.dumps(raw, indent=2))
    for _ in range(16):
        receipts = client.request("/collective_rpc", {"method": "resident_status", "args": [], "timeout": 1200})[
            "results"
        ]
        if all(
            isinstance(r, dict)
            and r.get("prompt_profile", {}).get("nonce") == profile_nonce
            and r["prompt_profile"]["active"] == enabled
            for r in receipts
        ):
            return receipts
    raise RuntimeError("could not confirm prompt profiler state on all ranks")


try:
    receipts = client.switch(Control(uuid.uuid4().hex, candidate="prefill_trace", source=source))
    (root / "profile-installed.json").write_text(json.dumps(receipts, indent=2))
    client.request("/pause?mode=wait&clear_cache=true")
    active = True
    start = set_profile(True)
    (root / "profile-start.json").write_text(json.dumps(start, indent=2))
    client.resume()
    print("PROFILE_REQUEST_STARTED", flush=True)
    case = replace(retrieval_case(20480, variant=104205, count_tokens=count_tokens), max_tokens=32)
    result = run_request(
        case,
        client.base_url,
        "glm53-flash-selective-w3",
        42,
        "prefill_trace",
        tokenizer_identity=tokenizer_identity,
        timeout_s=1200,
    )
    (root / "profile-request.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k not in ("prompt", "request_settings")}), flush=True)
    client.request("/pause?mode=wait&clear_cache=false")
    stop = set_profile(False)
    active = False
    (root / "profile-stop.json").write_text(json.dumps(stop, indent=2))
finally:
    if active:
        client.request("/pause?mode=wait&clear_cache=false")
        set_profile(False)
    receipts = client.switch(Control(uuid.uuid4().hex, candidate="completed_pools", source=baseline))
    client.resume()
    (root / "restored.json").write_text(json.dumps(receipts, indent=2))
    print("PROFILE_RESTORED", flush=True)

client.request("/pause?mode=wait&clear_cache=true")
client.resume()
print("BASELINE_REQUEST_STARTED", flush=True)
case = replace(retrieval_case(8192, variant=104205, count_tokens=count_tokens), max_tokens=32)
result = run_request(
    case,
    client.base_url,
    "glm53-flash-selective-w3",
    42,
    "baseline_8k",
    tokenizer_identity=tokenizer_identity,
    timeout_s=1200,
)
(root / "baseline-8k.json").write_text(json.dumps(result, indent=2))
print(json.dumps({k: v for k, v in result.items() if k not in ("prompt", "request_settings")}), flush=True)
(root / "capture-complete.json").write_text(json.dumps({"completed_at_unix": time.time()}, indent=2))
print("CAPTURE_COMPLETE", flush=True)
