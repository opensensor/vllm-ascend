# SPDX-License-Identifier: Apache-2.0
import hashlib
import json
import time
import uuid
from pathlib import Path

import psutil

from tools.glm_perf.resident_control import Control
from tools.glm_perf.resident_harness import ResidentClient
from tools.glm_perf.resident_native import NativeManifest
from tools.glm_perf.suite import make_groups, run_groups, summarize
from tools.glm_perf.worker_affinity import apply_bindings, plan_bindings

root = Path(__file__).resolve().parent
client = ResidentClient("http://127.0.0.1:8001")
pid = int(Path("/home/matteius/experiments/glm-kpool-live-score-20261005/sinkhorn-normalize.pid").read_text())
for _ in range(450):
    try:
        client.request("/health", method="GET")
        break
    except OSError:
        if not psutil.pid_exists(pid):
            raise RuntimeError("server exited during startup")
        time.sleep(2)
else:
    raise TimeoutError("GLM startup")
workers = sorted((p for p in psutil.Process(pid).children(recursive=True) if "Worker_TP" in p.name()), key=lambda p: p.name())
assert len(workers) == 4
masks = [set(range(s, s + 4)) | set(range(s + 32, s + 36)) for s in (8, 12, 16, 20)]
bindings = plan_bindings(pid, {p.pid: mask for p, mask in zip(workers, masks)})
apply_bindings(bindings)
(root / "serving-affinity.json").write_text(json.dumps(bindings, indent=2))
previous = Path("/home/matteius/experiments/glm-completed-pools-20261005/applied-candidate.py").read_text()
prefix = previous.replace("def replacements(", "def serving_replacements(")
prefix += "\n" + (root / "mhc_sinkhorn_normalize.py").read_text().replace("def replacements(", "def sinkhorn_replacements(")


def artifact(name):
    path = root / name
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


manifest = {"name": "sinkhorn_normalize_v1", "libraries": [artifact("glm_sinkhorn_bridge_v1.so")],
            "assets": [artifact("sinkhorn_native.py"), artifact("normalize-v1.bin")],
            "operators": ["glm_sinkhorn_v1::launch"], "validation_source": (root / "native-validation.py").read_text()}
(root / "native-manifest.json").write_text(json.dumps(manifest, indent=2))
loaded = client.load_native(NativeManifest(manifest))
(root / "native-load-receipts.json").write_text(json.dumps(loaded, indent=2))
print("NATIVE LOAD PASSED ON FOUR RANKS", flush=True)
promoted = False
try:
    source = prefix + "\n" + (root / "shadow-audit.py").read_text()
    client.switch(Control(uuid.uuid4().hex, candidate="sinkhorn_shadow", source=source))
    groups = make_groups(["fault"], [])
    rows = run_groups(groups, client.base_url, "glm53-flash-selective-w3", 42, timeout_s=300)
    receipts = client.rpc("resident_status")
    (root / "shadow-requests.json").write_text(json.dumps(rows, indent=2))
    (root / "shadow-receipts.json").write_text(json.dumps(receipts, indent=2))
    assert all(r["sinkhorn_audit"]["elements"] > 10000 and r["sinkhorn_audit"]["fp32_mismatches"] == 0 for r in receipts), receipts
    print("REAL ACTIVATION SHADOW PASSED", flush=True)
    source = prefix + "\ndef replacements(native_resources):\n    changes = serving_replacements(native_resources)\n    changes.update(sinkhorn_replacements(native_resources))\n    return changes\n"
    (root / "applied-candidate.py").write_text(source)
    before = client.switch(Control(uuid.uuid4().hex, candidate="completed_pools_sinkhorn", source=source))
    (root / "serving-initial-status.json").write_text(json.dumps(before, indent=2))
    with (root / "serving-results.jsonl").open("w") as output:
        def record(row):
            output.write(json.dumps(row) + "\n")
            output.flush()
            print(json.dumps({k: row.get(k) for k in ("case_id", "passed", "ttft_s", "decode_tokens_per_s", "error")}), flush=True)
        rows = run_groups(make_groups(["short", "tool"], []), client.base_url, "glm53-flash-selective-w3", 42, on_result=record, timeout_s=600)
    result = summarize(rows)
    (root / "serving-summary.json").write_text(json.dumps(result, indent=2))
    print("SUMMARY", json.dumps(result), flush=True)
    assert result["passed"], result
    # Promotion decision follows inspection of throughput and generated text.
    promoted = True
    (root / "serving-final-status.json").write_text(json.dumps(client.rpc("resident_status"), indent=2))
finally:
    if not promoted:
        client.switch(Control(uuid.uuid4().hex, candidate="completed_pools", source=previous))
        client.resume()
        print("RESTORED COMPLETED POOLS", flush=True)
