# SPDX-License-Identifier: Apache-2.0
"""One controlled kernel-package restart, preserving the live GLM settings."""

import argparse
import contextlib
import hashlib
import json
import subprocess
import time
import uuid
from pathlib import Path

import psutil

from tools.glm_perf.resident_control import Control
from tools.glm_perf.resident_harness import ResidentClient
from tools.glm_perf.resident_native import NativeManifest
from tools.glm_perf.worker_affinity import apply_bindings, plan_bindings


def stop(process, known_children=()):
    # Track process objects with creation times; never target unrelated jobs.
    processes = list(known_children)
    with contextlib.suppress(psutil.NoSuchProcess):
        processes.extend([process] + process.children(recursive=True))
    processes = list({item.pid: item for item in processes}.values())
    for item in processes:
        with contextlib.suppress(psutil.NoSuchProcess):
            item.terminate()
    _, alive = psutil.wait_procs(processes, timeout=30)
    for item in alive:
        with contextlib.suppress(psutil.NoSuchProcess):
            item.kill()
    psutil.wait_procs(alive, timeout=15)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-pid", required=True, type=int)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    baseline = Path("/home/matteius/experiments/glm-sinkhorn-resident-20261005")
    build = json.loads((root / "full-kda-build.json").read_text())
    assert hashlib.sha256(Path(build["binary"]).read_bytes()).hexdigest() == build["binary_sha256"]
    for label in ("control", "cached"):
        rows = json.loads((root / f"full-kda-{label}.json").read_text())
        safe = [row for row in rows if "_True_" in row["case"]]
        assert len(safe) == 17
        assert all(
            row["finite"] and sum(row["bit_mismatches"]) == 0 and sum(row["repeat_bit_mismatches"]) == 0 for row in safe
        ), f"{label}: full KDA qualification failed"
    source = (root / "fused-baseline-candidate.py").read_text()
    expected_digest = hashlib.sha256(source.encode()).hexdigest()
    client = ResidentClient("http://127.0.0.1:8001")
    before = client.rpc("resident_status")
    assert all(
        row["digest"] == expected_digest
        and row["candidate"] == "completed_pools_sinkhorn"
        and not row.get("native_failed")
        and not row["graphs_dirty"]
        for row in before
    )
    (root / "before-package-restart.json").write_text(json.dumps(before, indent=2) + "\n")
    process = psutil.Process(args.api_pid)
    assert "vllm.entrypoints.cli.main" in process.cmdline()
    command, cwd = process.cmdline(), process.cwd()
    # Retain the environment in memory, without writing credentials to artifacts.
    environment = process.environ()
    original_environment = dict(environment)
    old_vendor = "/srv/ai/src/kda-persistent-scores-opp/vendors/custom_transformer"
    new_vendor = str(Path(build["package"]) / "vendors/custom_transformer")
    stack = environment["ASCEND_CUSTOM_OPP_PATH"].split(":")
    assert stack.count(old_vendor) == 1, stack
    stack[stack.index(old_vendor)] = new_vendor
    environment["ASCEND_CUSTOM_OPP_PATH"] = ":".join(stack)
    # Host libraries and every other vendor retain their original priority.
    (root / "package-restart-config.json").write_text(
        json.dumps(
            {
                "command": command,
                "cwd": cwd,
                "opp_stack": stack,
                "previous_api_pid": process.pid,
                "kernel_sha256": build["binary_sha256"],
            },
            indent=2,
        )
        + "\n"
    )
    assert client.request("/pause?mode=wait&clear_cache=true")["status"] == "paused"
    stop(process)
    print("previous GLM processes stopped", flush=True)

    def launch(env, label):
        log = (root / f"serve-{label}.log").open("wb")
        server = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        log.close()
        (root / f"{label}-api.pid").write_text(str(server.pid) + "\n")
        print(json.dumps({"launch": label, "api_pid": server.pid}), flush=True)
        tree = psutil.Process(server.pid)
        known_children = {}
        try:
            ready = False
            for attempt in range(900):
                with contextlib.suppress(psutil.NoSuchProcess):
                    known_children.update({child.pid: child for child in tree.children(recursive=True)})
                if server.poll() is not None:
                    raise RuntimeError(f"{label} server exited: {server.returncode}")
                try:
                    client.request("/health", method="GET")
                    ready = True
                    break
                except OSError:
                    time.sleep(2)
                if attempt % 15 == 0:
                    print(json.dumps({"loading": label, "waited_s": attempt * 2}), flush=True)
            if not ready:
                raise TimeoutError(f"{label} startup")
        except Exception:
            stop(tree, known_children.values())
            raise
        return server

    def restore_fusion(server, label):
        workers = sorted(
            (child for child in psutil.Process(server.pid).children(recursive=True) if "Worker_TP" in child.name()),
            key=lambda child: child.name(),
        )
        assert len(workers) == 4
        masks = [set(range(start, start + 4)) | set(range(start + 32, start + 36)) for start in (8, 12, 16, 20)]
        apply_bindings(plan_bindings(server.pid, {worker.pid: mask for worker, mask in zip(workers, masks)}))
        manifest = NativeManifest(json.loads((baseline / "native-manifest.json").read_text()))
        receipts = client.load_native(manifest)
        (root / f"{label}-native-load.json").write_text(json.dumps(receipts, indent=2) + "\n")
        status = client.switch(Control(uuid.uuid4().hex, candidate="completed_pools_sinkhorn", source=source))
        client.resume()
        (root / f"{label}-live-status.json").write_text(json.dumps(status, indent=2) + "\n")
        # A successful request checks the execution queue, beyond HTTP health.
        warmup = client.request(
            "/v1/completions",
            {
                "model": "glm53-flash-selective-w3",
                "prompt": "Write a Python function that adds two integers.\n",
                "max_tokens": 16,
                "temperature": 0,
            },
        )
        assert warmup["usage"]["completion_tokens"] > 0
        (root / f"{label}-warmup.json").write_text(json.dumps(warmup, indent=2) + "\n")
        print(
            json.dumps(
                {
                    "ready": label,
                    "api_pid": server.pid,
                    "workers": [worker.pid for worker in workers],
                    "candidate": "completed_pools_sinkhorn",
                }
            ),
            flush=True,
        )

    server = None
    try:
        server = launch(environment, "score-cache")
        restore_fusion(server, "score-cache")
    except Exception:
        if server is not None and server.poll() is None:
            stop(psutil.Process(server.pid))
        rollback = launch(original_environment, "rollback-baseline")
        restore_fusion(rollback, "rollback-baseline")
        raise


if __name__ == "__main__":
    main()
