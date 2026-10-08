# SPDX-License-Identifier: Apache-2.0
"""Recover an absent GLM server; resident experiments use hot_swap.py."""

import argparse
import hashlib
import json
import os
import subprocess
import time
import uuid
from pathlib import Path

import psutil
from qualified_harness import ResidentClient

from tools.glm_perf.resident_control import Control
from tools.glm_perf.resident_native import NativeManifest
from tools.glm_perf.worker_affinity import apply_bindings, plan_bindings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("label", choices=("baseline", "candidate", "rollback"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    qualified = Path("/home/matteius/experiments/glm-prompt-profile-20261005")
    saved = json.loads((qualified / "package-restart-config.json").read_text())
    # Recovery must never terminate an already resident GLM worker.
    for record in root.glob("*-process.json"):
        previous = json.loads(record.read_text())
        try:
            existing = psutil.Process(previous["pid"])
            if existing.create_time() == previous["created"] and existing.is_running():
                raise RuntimeError("GLM is resident. Use hot_swap.py; this launcher cannot stop it.")
        except psutil.NoSuchProcess:
            pass
    command = list(saved["command"])
    command[command.index("--worker-extension-cls") + 1] = "decode_flags_extension.DecodeFlagsExtension"
    overrides_index = command.index("--hf-overrides") + 1
    overrides = json.loads(command[overrides_index])
    enabled = args.label == "candidate"
    overrides.update(ascend_glm_decode_swiglu=enabled, ascend_glm_decode_combine=enabled)
    command[overrides_index] = json.dumps(overrides, separators=(",", ":"))
    stack = [str(root / "opp/vendors/custom_transformer")] + saved["opp_stack"]
    assert all(Path(vendor).is_dir() for vendor in stack), stack
    environment = dict(os.environ)
    environment.update(
        ASCEND_CUSTOM_OPP_PATH=":".join(stack),
        PYTHONPATH=":".join(
            [
                str(root),
                "/home/matteius/experiments/glm-kpool-live-score-20261005",
                saved["cwd"],
                "/srv/ai/src/vllm-opensensor",
                environment.get("PYTHONPATH", ""),
            ]
        ),
        ASCEND_RT_VISIBLE_DEVICES="0,1,2,3",
        SOC_VERSION="ascend310p1",
        TASK_QUEUE_ENABLE="1",
        OMP_NUM_THREADS="1",
        VLLM_SERVER_DEV_MODE="1",
        VLLM_USE_BREAKABLE_CUDAGRAPH="1",
        VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="3000",
        VLLM_ASCEND_310P_ENABLE_MLA="1",
        VLLM_ASCEND_310P_GLM_HOST_KV="0",
        VLLM_ASCEND_KV_CACHE_FRACTION="0.75",
        VLLM_ASCEND_LOG_REQUEST_TIMINGS="1",
    )
    # Retain the qualified KDA host library while selecting the cached kernel.
    host_stack = [
        vendor.replace(str(qualified / "opp-score-cache"), "/srv/ai/src/kda-persistent-scores-opp") for vendor in stack
    ]
    environment["LD_LIBRARY_PATH"] = ":".join(
        [str(Path(vendor) / "op_api/lib") for vendor in host_stack] + [environment.get("LD_LIBRARY_PATH", "")]
    )
    environment.pop("ASCEND_LAUNCH_BLOCKING", None)
    log_path = root / f"serve-{args.label}.log"
    assert not log_path.exists(), f"preserve previous logs: {log_path}"
    (root / f"{args.label}-config.json").write_text(
        json.dumps(
            {
                "command": command,
                "cwd": saved["cwd"],
                "opp_stack": stack,
                "host_stack": host_stack,
                "launch_blocking": False,
                "binding_sha256": hashlib.sha256((root / "glm_decode_flags.so").read_bytes()).hexdigest(),
                "source_hashes": {
                    path: hashlib.sha256((Path(saved["cwd"]) / path).read_bytes()).hexdigest()
                    for path in (
                        "vllm_ascend/models/glm5next_w2/model.py",
                        "vllm_ascend/_310p/quantization/methods/w2_dynamic.py",
                    )
                },
            },
            indent=2,
        )
        + "\n"
    )
    with log_path.open("wb") as log:
        server = subprocess.Popen(
            command,
            cwd=saved["cwd"],
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    process = psutil.Process(server.pid)
    (root / f"{args.label}-process.json").write_text(
        json.dumps({"pid": process.pid, "created": process.create_time(), "log": str(log_path)}, indent=2) + "\n"
    )
    print(json.dumps({"launched": args.label, "pid": server.pid, "log": str(log_path)}), flush=True)
    client = ResidentClient("http://127.0.0.1:8001", timeout=900)
    for attempt in range(900):
        if server.poll() is not None:
            raise RuntimeError(f"server startup failed: {server.returncode}; see {log_path}")
        try:
            client.request("/health", method="GET")
            break
        except OSError:
            time.sleep(2)
        if attempt % 15 == 0:
            print(json.dumps({"loading": args.label, "seconds": attempt * 2}), flush=True)
    else:
        raise TimeoutError("GLM startup")
    workers = sorted(
        (child for child in process.children(recursive=True) if "Worker_TP" in child.name()),
        key=lambda child: child.name(),
    )
    assert len(workers) == 4
    masks = [set(range(start, start + 4)) | set(range(start + 32, start + 36)) for start in (8, 12, 16, 20)]
    affinity = plan_bindings(server.pid, {worker.pid: mask for worker, mask in zip(workers, masks)})
    apply_bindings(affinity)
    (root / f"{args.label}-affinity.json").write_text(json.dumps(affinity, indent=2, default=str) + "\n")
    # Restore existing qualified instrumentation once, identically on both launches.
    manifest = NativeManifest(
        json.loads(Path("/home/matteius/experiments/glm-sinkhorn-resident-20261005/native-manifest.json").read_text())
    )
    loaded = client.load_native(manifest)
    source = (qualified / "fused-baseline-candidate.py").read_text()
    assert (
        hashlib.sha256(source.encode()).hexdigest()
        == "348947ff4f965d2b6af532269f4b0e90eee1ba201d3e16dd60a063b8de423282"
    )
    status = client.switch(Control(uuid.uuid4().hex, candidate="completed_pools_sinkhorn", source=source))
    audit = client.rpc("decode_flags_status")
    assert len(audit) == 4 and all(row["banks"] for row in audit)
    assert all(
        bank["decode_swiglu"] == enabled and bank["decode_combine"] == enabled and not bank["offload_to_cpu"]
        for row in audit
        for bank in row["banks"]
    )
    (root / f"{args.label}-ready.json").write_text(
        json.dumps({"native_load": loaded, "status": status, "flags": audit}, indent=2) + "\n"
    )
    client.resume()
    print(
        json.dumps({"ready": args.label, "pid": server.pid, "workers": [worker.pid for worker in workers]}), flush=True
    )


if __name__ == "__main__":
    main()
