# SPDX-License-Identifier: Apache-2.0
"""Bounded, model-free idle attribution using the serving PyHCCL communicator.

Run each phase in a fresh process after cooling. CPU rendezvous never injects
an HCCL barrier into the idle window. Results identify a phase, not a cause:
telemetry may be stale, and profiling/driver traffic can outlive a process.
"""

import argparse
import gc
import json
import multiprocessing
import subprocess
import time
from datetime import timedelta
from pathlib import Path

from tools.qwen4exp.thermal_controller import parse_temperatures

PHASES = ("device", "allocation", "communicator", "eager", "graph", "retire", "close")
ADMISSION_C = 72.0
ABORT_C = 90.0
EXPECTED_DEVICES = 6
ALLOCATION_BYTES = 128 * 1024 * 1024


def admit(snapshot: str) -> list[float]:
    temperatures = parse_temperatures(snapshot, EXPECTED_DEVICES)
    if max(temperatures) > ADMISSION_C:
        raise RuntimeError(f"idle probe requires all chips at or below {ADMISSION_C}C: {temperatures}")
    return temperatures


def worker(rank, phase, rendezvous, port, output):
    # Load the device backend only inside admitted children. The parent and
    # rejection path never import torch_npu or initialize a device.
    import torch
    import torch_npu  # noqa: F401

    communicator = None
    graph = None
    try:
        torch.distributed.init_process_group(
            "gloo",
            rank=rank,
            world_size=EXPECTED_DEVICES,
            init_method=f"tcp://127.0.0.1:{port}",
            timeout=timedelta(seconds=60),
        )
        torch.npu.set_device(rank)
        device = torch.device(f"npu:{rank}")
        sentinel = torch.full((1024,), rank + 1, dtype=torch.float16, device=device)
        allocation = torch.empty(ALLOCATION_BYTES, dtype=torch.uint8, device=device) if phase == "allocation" else None
        if allocation is not None:
            allocation.zero_()
        if phase in PHASES[2:]:
            from vllm_ascend.distributed.device_communicators.pyhccl import PyHcclCommunicator

            communicator = PyHcclCommunicator(torch.distributed.group.WORLD, device)
            assert communicator.available and not communicator.disabled
        if phase in PHASES[3:]:
            for _ in range(3):
                result = communicator.all_reduce(sentinel)
            torch.npu.synchronize()
            assert torch.equal(result.cpu(), torch.full((1024,), 21, dtype=torch.float16))
        if phase in PHASES[4:]:
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                result = communicator.all_reduce(sentinel)
            graph.replay()
            torch.npu.synchronize()
            assert torch.equal(result.cpu(), torch.full((1024,), 21, dtype=torch.float16))
        if phase in ("retire", "close"):
            graph.reset()
            graph = None
            gc.collect()
            torch.npu.empty_cache()
        if phase == "close":
            communicator.close()
        torch.npu.synchronize()
        rendezvous.wait(timeout=60)
        # Parent samples the idle phase while allocations/graphs stay owned.
        rendezvous.wait(timeout=60)
        Path(output, f"rank-{rank}.json").write_text(json.dumps({"rank": rank, "phase": phase, "passed": True}) + "\n")
    except Exception as error:
        Path(output, f"rank-{rank}.json").write_text(
            json.dumps({"rank": rank, "phase": phase, "passed": False, "error": repr(error)}) + "\n"
        )
        rendezvous.abort()
        raise
    finally:
        if graph is not None:
            graph.reset()
        if communicator is not None:
            communicator.close()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    snapshot = subprocess.check_output(["npu-smi", "info"], text=True, timeout=5)
    try:
        temperatures = admit(snapshot)
    except (RuntimeError, ValueError) as error:
        (args.output / "result.json").write_text(
            json.dumps({"passed": False, "admitted": False, "device_children": 0, "error": str(error)}) + "\n"
        )
        raise
    context = multiprocessing.get_context("spawn")
    rendezvous = context.Barrier(EXPECTED_DEVICES + 1)
    children = [
        context.Process(target=worker, args=(rank, args.phase, rendezvous, args.port, str(args.output)))
        for rank in range(EXPECTED_DEVICES)
    ]
    report = {"phase": args.phase, "before_c": temperatures, "model_loaded": False, "passed": False}
    try:
        for child in children:
            child.start()
        rendezvous.wait(timeout=60)
        start = time.monotonic()
        with (args.output / "idle.jsonl").open("x") as samples:
            while time.monotonic() - start < args.seconds:
                raw = subprocess.check_output(["npu-smi", "info"], text=True, timeout=5)
                temperatures = parse_temperatures(raw, EXPECTED_DEVICES)
                record = {"time": time.time(), "temperatures_c": temperatures, "snapshot": raw}
                if max(temperatures) >= ABORT_C:
                    raise RuntimeError(f"thermal abort: {temperatures}")
                record["usage"] = [
                    {
                        "card": card,
                        "chip": chip,
                        "raw": subprocess.check_output(
                            ["npu-smi", "info", "-t", "usages", "-i", str(card), "-c", str(chip)], text=True, timeout=5
                        ),
                    }
                    for card in args.cards
                    for chip in (0, 1)
                ]
                samples.write(json.dumps(record) + "\n")
                samples.flush()
                time.sleep(1)
        rendezvous.wait(timeout=60)
        for child in children:
            child.join(timeout=10)
        report["exitcodes"] = [child.exitcode for child in children]
        report["passed"] = all(code == 0 for code in report["exitcodes"])
    except Exception as error:
        report["error"] = repr(error)
        raise
    finally:
        for child in children:
            if child.pid is not None and child.is_alive():
                child.terminate()
                child.join(timeout=5)
                if child.is_alive():
                    child.kill()
                    child.join(timeout=5)
        (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    if not report["passed"]:
        raise RuntimeError("idle probe did not pass; inspect per-rank receipts")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=PHASES, required=True)
    parser.add_argument("--cards", type=int, nargs=3, required=True)
    parser.add_argument("--seconds", type=int, choices=range(5, 21), default=10)
    parser.add_argument("--port", type=int, default=29593)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(set(args.cards)) != 3 or not 1024 <= args.port <= 65535:
        parser.error("three distinct card IDs and an unprivileged TCP port are required")
    run(args)


if __name__ == "__main__":
    main()
