# SPDX-License-Identifier: Apache-2.0
import hashlib
import json
import time
import urllib.request
import uuid
from pathlib import Path

import psutil

from tools.glm_perf.resident_control import Control
from tools.glm_perf.resident_harness import ResidentClient
from tools.glm_perf.resident_native import NativeManifest
from tools.glm_perf.suite import load_tokenizer_json, make_groups, retrieval_case, run_groups, summarize
from tools.glm_perf.worker_affinity import apply_bindings, plan_bindings

root = Path(__file__).resolve().parent
candidates = Path("/srv/ai/src/glm-selective-w3-nz-test-20261004/tools/glm_perf/resident_candidates")
client = ResidentClient("http://127.0.0.1:8001")
pid = int(Path("/home/matteius/experiments/glm-kpool-live-score-20261005/queue10-baseline2.pid").read_text())
for _ in range(600):
    if not psutil.pid_exists(pid):
        raise RuntimeError("server exited during startup")
    try:
        with urllib.request.urlopen("http://127.0.0.1:8001/health", timeout=2) as response:
            if response.status == 200:
                break
    except OSError:
        pass
    time.sleep(2)
else:
    raise TimeoutError("server startup")
workers = sorted(
    (p for p in psutil.Process(pid).children(recursive=True) if "Worker_TP" in p.name()), key=lambda p: p.name()
)
assert len(workers) == 4
masks = [set(range(s, s + 4)) | set(range(s + 32, s + 36)) for s in (8, 12, 16, 20)]
plan = plan_bindings(pid, {p.pid: mask for p, mask in zip(workers, masks)})
apply_bindings(plan)
(root / "affinity.json").write_text(json.dumps(plan, indent=2))
identities = None
summaries = {}
count_tokens, tokenizer_identity = load_tokenizer_json(
    Path("/srv/ai/models/GLM-5.3-Flash-selective-W3-310p/tokenizer.json")
)


def switch(name, source=""):
    global identities
    control = Control(uuid.uuid4().hex, candidate=name if source else "baseline", source=source)
    Control.from_dict(vars(control))
    receipts = client.switch(control)
    current = {(r["rank"], r["pid"], r["weight_storage_digest"]) for r in receipts}
    assert identities is None or current == identities
    identities = current
    (root / f"{name}-status.json").write_text(json.dumps(receipts, indent=2))
    print("SWITCH", name, flush=True)
    return receipts


try:
    if (root / "preflight-results.json").exists():
        preflight = json.loads((root / "preflight-results.json").read_text())
    else:
        switch("preflight", (root / "preflight.py").read_text())
        client.request("/pause?mode=wait&clear_cache=true")
        preflight = client.rpc("resident_status")
        (root / "preflight-results.json").write_text(json.dumps(preflight, indent=2))
    switch("preflight_restored")
    client.resume()
    names = {
        "kda_batched_qk": "kda_input_preparation.py",
        "kpool_decode_epilogue": "kpool_decode_epilogue.py",
        "moe_half_unpermute": "moe_half_unpermute.py",
        "direct_route_tokens": "direct_route_tokens.py",
        "indexer_projection": "indexer_projection.py",
    }
    admitted = [name for name in names if all(r["queue_preflight"][name]["passed"] for r in preflight)]
    if json.loads((root / "gate-beta-parity.json").read_text()).get("passed"):

        def asset(path):
            return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

        validation = (root / "staged/validate-kda-gate-beta.py").read_text()
        validation += "\ndef prepare():\n"
        validation += "    import torch\n    from tools.glm_perf.kda_gate_beta_native import GateBeta\n"
        validation += "    return GateBeta(" + repr(str(root / "gate-beta-build/kda-gate-beta-v1.bin"))
        validation += ", heads=16, lower_bound=-5.0, device=torch.device('npu', torch.npu.current_device()))\n"
        manifest = NativeManifest(
            {
                "name": "kda_gate_beta_v1",
                "libraries": [asset(root / "gate-beta-build/glm_kda_prepare_bridge_v1.so")],
                "assets": [asset(root / "gate-beta-build/kda-gate-beta-v1.bin")],
                "operators": ["glm_kda_prepare_v1::launch"],
                "validation_source": validation,
            }
        )
        (root / "gate-beta-manifest.json").write_text(manifest.payload)
        receipts = client.load_native(manifest)
        (root / "gate-beta-native-load.json").write_text(json.dumps(receipts, indent=2))
        names["kda_gate_beta"] = "kda_gate_beta.py"
        admitted.append("kda_gate_beta")
    print("ADMITTED", admitted, flush=True)
    (root / "python-admitted.json").write_text(json.dumps(admitted))
    with (root / "python-serving.jsonl").open("w") as output:
        for name in ["baseline", *admitted, "baseline_repeat"]:
            source = (candidates / names[name]).read_text() if name in names else ""
            try:
                switch(name, source)
            except Exception as exc:
                summaries[name] = {"switch_error": repr(exc)}
                (root / "python-summary.json").write_text(json.dumps(summaries, indent=2))
                switch(name + "_recovered")
                if client.request("/is_paused", method="GET")["is_paused"]:
                    client.resume()
                continue

            def record(row, candidate=name):
                row["candidate"] = candidate
                output.write(json.dumps(row) + "\n")
                output.flush()
                print(
                    json.dumps(
                        {k: row.get(k) for k in ("candidate", "case_id", "passed", "decode_tokens_per_s", "error")}
                    ),
                    flush=True,
                )

            groups = make_groups(["short", "fault", "fault4"], [])
            if name in ("baseline", "moe_half_unpermute", "direct_route_tokens", "baseline_repeat"):
                variant = ["baseline", "moe_half_unpermute", "direct_route_tokens", "baseline_repeat"].index(name) + 10
                groups.append(("cold8k", [retrieval_case(8192, variant=variant, count_tokens=count_tokens)]))
            rows = run_groups(
                groups,
                "http://127.0.0.1:8001",
                "glm53-flash-selective-w3",
                42,
                on_result=record,
                tokenizer_identity=tokenizer_identity,
            )
            summaries[name] = summarize(rows)
            (root / "python-summary.json").write_text(json.dumps(summaries, indent=2))
            with urllib.request.urlopen("http://127.0.0.1:8001/metrics", timeout=10) as response:
                (root / f"{name}-metrics.txt").write_bytes(response.read())
            print("SUMMARY", name, json.dumps(summaries[name]), flush=True)
            if not summaries[name]["valid"]:
                raise RuntimeError("invalid benchmark: " + name)
finally:
    switch("python_restored")
    if client.request("/is_paused", method="GET")["is_paused"]:
        client.resume()
