"""Launch one reviewed GLM card profile from the host's 4 TB model drive."""

import argparse
import fcntl
import hashlib
import json
import os
import socket
import subprocess
from pathlib import Path

from tools.glm_perf.deployment_profiles import profile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cards", type=int, choices=(2, 3), required=True)
    parser.add_argument("--attempt", type=int, default=1)
    args = parser.parse_args()
    selected = profile(args.cards)
    root = Path("/srv/ai/storage/artifacts/glm-card-profiles-20261010")
    name = "two" if args.cards == 2 else "three"
    model = Path(f"/srv/ai/storage/models/GLM-5.3-Flash-{name}-card-native-candidate")
    receipt = json.loads((root / f"{name}-card-storage-check.json").read_text())
    assert receipt["shard_checksums_verified"] and receipt["profile"] == selected.name
    source = Path("/srv/ai/src/glm-card-profiles-20261010-5652e3a81")
    saved = json.loads(Path("/home/matteius/experiments/glm-decode-flags-20261006/candidate-config.json").read_text())
    command = [
        "/srv/ai/venvs/qwen38-w4-test-ce1862/bin/python",
        "-m",
        "vllm.entrypoints.cli.main",
        *selected.launch_args(model)[1:],
        "--host",
        "127.0.0.1",
        "--gpu-memory-utilization",
        "0.90",
        "--trust-remote-code",
        "--enable-logging-iteration-details",
        "--worker-extension-cls",
        "card_worker_extension.CardWorkerExtension",
    ]
    environment = dict(os.environ)
    environment.update(
        ASCEND_CUSTOM_OPP_PATH=":".join(saved["opp_stack"]),
        PYTHONPATH=":".join([str(root), str(source), "/srv/ai/src/vllm-opensensor", environment.get("PYTHONPATH", "")]),
        ASCEND_RT_VISIBLE_DEVICES="2,3,4,5" if args.cards == 2 else "0,1,2,3,4,5",
        SOC_VERSION="ascend310p1",
        TASK_QUEUE_ENABLE="1",
        OMP_NUM_THREADS="1",
        VLLM_USE_BREAKABLE_CUDAGRAPH="1",
        VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="3000",
        VLLM_ASCEND_310P_ENABLE_MLA="1",
        VLLM_ASCEND_310P_GLM_HOST_KV="0",
        VLLM_ASCEND_KV_CACHE_FRACTION="0.75",
        VLLM_ASCEND_LOG_REQUEST_TIMINGS="1",
        HCCL_IF_BASE_PORT="54000",
    )
    environment["LD_LIBRARY_PATH"] = ":".join(
        [str(Path(path) / "op_api/lib") for path in saved["host_stack"]] + [environment.get("LD_LIBRARY_PATH", "")]
    )
    environment.pop("ASCEND_LAUNCH_BLOCKING", None)
    # Fail on missing auxiliary/ATB registrations before allocating model weights.
    subprocess.run(
        [
            command[0],
            "-c",
            "import card_worker_extension; "
            "from torch_npu.op_plugin.atb._atb_ops import _register_atb_extensions; "
            "_register_atb_extensions(); print('GLM_BINDING_ATB_PREFLIGHT_PASS')",
        ],
        cwd=source,
        env=environment,
        check=True,
    )
    with open("/tmp/glm-card-profile-8001.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with socket.socket() as check:
            assert check.connect_ex(("127.0.0.1", 8001)) != 0, "port is occupied"
        logfile = root / f"{name}-card-server-r{args.attempt}.log"
        hashes = {
            filename: hashlib.sha256((source / filename).read_bytes()).hexdigest()
            for filename in (
                "vllm_ascend/models/glm5next/model.py",
                "vllm_ascend/models/glm5next_w2/model.py",
                "vllm_ascend/model_loader/glm_native_int4.py",
                "vllm_ascend/_310p/model_runner_310p.py",
                "vllm_ascend/_310p/npu_input_batch.py",
                "vllm_ascend/distributed/device_communicators/pipeline_all_gather.py",
                "vllm_ascend/distributed/device_communicators/npu_communicator.py",
                "vllm_ascend/patch/worker/patch_distributed.py",
                "vllm_ascend/patch/platform/patch_distributed.py",
            )
        }
        with logfile.open("xb") as output:
            process = subprocess.Popen(
                command,
                cwd=source,
                env=environment,
                stdout=output,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
        record = {
            "pid": process.pid,
            "command": command,
            "cwd": str(source),
            "log": str(logfile),
            "devices": environment["ASCEND_RT_VISIBLE_DEVICES"],
            "source_sha256": hashes,
            "opp_stack": saved["opp_stack"],
            "host_stack": saved["host_stack"],
            "scope": "New real-weight profile: automatic memory profiling, no fixed-cache bypass",
        }
        (root / f"{name}-card-server-process.json").write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps(record), flush=True)
        raise SystemExit(process.wait())


if __name__ == "__main__":
    main()
