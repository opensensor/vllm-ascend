import hashlib
import json
import time
import urllib.request
from pathlib import Path

root = Path("/srv/ai/src/qwen-prefill-fairness-20261010")


def call(path, body=None, timeout=60):
    request = urllib.request.Request(
        "http://127.0.0.1:8001" + path,
        data=None if body is None else json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read()
        return json.loads(raw) if raw else None


def rpc(method, args=None, timeout=180):
    return call("/collective_rpc", {"method": method, "args": args or [], "timeout": timeout}, timeout + 10)


def save(name, value):
    (root / (name + ".json")).write_text(json.dumps(value, indent=2))
    return value


assert not call("/is_paused")["is_paused"]
metrics = urllib.request.urlopen("http://127.0.0.1:8001/metrics", timeout=10).read().decode()
assert all(
    float(line.rsplit(" ", 1)[1]) == 0
    for line in metrics.splitlines()
    if line.startswith(("vllm:num_requests_running{", "vllm:num_requests_waiting{"))
)
call("/pause?mode=wait&clear_cache=false", {})
try:
    before = save("profile-before", rpc("resident_status"))
    library = Path("/srv/ai/src/qwen-performance-evidence-20261010/diagnostic-marker-v14.so")
    manifest = {
        "name": "qwen_transfer_diagnostic_v16",
        "libraries": [{"path": str(library), "sha256": hashlib.sha256(library.read_bytes()).hexdigest()}],
        "assets": [],
        "operators": ["qwen_stage_diagnostic_v14::marker"],
        "validation_source": (root / "transfer-profile-validation.py").read_text(),
    }
    save("profile-manifest", manifest)
    prepared = save("profile-prepared", rpc("resident_native_prepare", [json.dumps(manifest)]))
    assert all("error" not in v for v in prepared["results"]), prepared
    digests = {v["native_digest"] for v in prepared["results"]}
    assert len(digests) == 1
    loaded = save("profile-loaded", rpc("resident_native_load", [next(iter(digests))]))
    assert all("error" not in v and v["validation"]["passed"] for v in loaded["results"]), loaded
    started = save("profile-started", rpc("resident_transfer_profile_v16", ["start"]))
    assert all(v["passed"] for v in started["results"]), started
finally:
    call("/resume", {})
try:
    start = time.perf_counter()
    response = call(
        "/v1/completions",
        {
            "model": "qwen38-flash-next",
            "prompt": "Transfer trace cold prefix: "
            + ("The archive describes a blue project with tests and a completed review.\n" * 32)
            + "Continue briefly.",
            "temperature": 0,
            "seed": 42,
            "max_tokens": 12,
            "ignore_eos": True,
        },
        timeout=180,
    )
    save("profile-request", {"elapsed_s": time.perf_counter() - start, "response": response})
finally:
    stopped = save("profile-stopped", rpc("resident_transfer_profile_v16", ["stop"]))
    assert all(v["passed"] for v in stopped["results"]), stopped
    after = save("profile-after", rpc("resident_status"))
    assert [v["weight_storage_digest"] for v in before["results"]] == [
        v["weight_storage_digest"] for v in after["results"]
    ]
    assert all(
        v["candidate"] == "baseline" and not v["graphs_dirty"] and not v["native_failed"] for v in after["results"]
    )
    print("Six-rank live decode profile collected; server remains baseline", flush=True)
