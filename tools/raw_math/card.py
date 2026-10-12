# SPDX-License-Identifier: Apache-2.0
"""Measure simultaneous independent GEMMs on both chips of the assigned card.

This is aggregate throughput for two independent jobs, not one sharded GEMM.
The coordinator's wall interval includes barrier release, submission, device
completion and receipt of both worker messages. It is deliberately conservative.
"""

import argparse
import hashlib
import json
import multiprocessing
import os
import time
from datetime import UTC, datetime
from pathlib import Path

from .cases import Case, summary

DEVICES = (2, 3)
REPEATS = 4
TIMEOUT_SECONDS = 180


def aggregate_rate(case, repeats, elapsed_ms):
    if repeats < 1:
        raise ValueError("positive repeats required")
    summary([elapsed_ms])
    return len(DEVICES) * repeats * 2 * case.m * case.n * case.k / (elapsed_ms * 1e9)


def worker(index, case, barrier, pipe, samples):
    # Each subprocess owns its accelerator context; the parent stays CPU-only.
    import torch
    import torch_npu

    from .benchmark import NpuCase, changed_input, inputs, run_case, validate

    backend = None
    try:
        torch.set_num_threads(1)
        torch.npu.set_device(index)
        torch.npu.set_compile_mode(jit_compile=False)
        torch.npu.matmul.allow_hf32 = False
        source = inputs(case, torch)
        backend = NpuCase(case, source, torch, torch_npu, nz=True)
        qualification = run_case(case, source, backend, torch, 3, 3)
        pipe.send(dict(ready=True, physical_device=DEVICES[index], qualification=qualification))
        for _ in range(samples):
            barrier.wait(TIMEOUT_SECONDS)
            device, wall = backend.measure(REPEATS)
            pipe.send(dict(device_ms_per_operation=device, wall_ms_per_operation=wall))
        final = validate(case, (changed_input(case, source[0]), *source[1:]), backend.read(), torch)
        pipe.send(dict(final_validation=final))
    except Exception as error:
        pipe.send(dict(error=f"{type(error).__name__}: {error}"))
        barrier.abort()
    finally:
        if backend:
            backend.close()
        pipe.close()


def receive(pipe):
    if not pipe.poll(TIMEOUT_SECONDS):
        raise TimeoutError("assigned benchmark worker did not reply")
    result = pipe.recv()
    if "error" in result:
        raise RuntimeError(result["error"])
    return result


def run(case, samples):
    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(len(DEVICES) + 1)
    pairs = [context.Pipe() for _ in DEVICES]
    processes = [
        context.Process(target=worker, args=(i, case, barrier, pair[1], samples)) for i, pair in enumerate(pairs)
    ]
    try:
        for process, pair in zip(processes, pairs):
            process.start()
            pair[1].close()
        qualifications = [receive(pair[0]) for pair in pairs]
        measured = []
        for _ in range(samples):
            start = time.perf_counter()
            barrier.wait(TIMEOUT_SECONDS)
            workers = [receive(pair[0]) for pair in pairs]
            elapsed_ms = (time.perf_counter() - start) * 1000
            measured.append(
                dict(
                    elapsed_ms=elapsed_ms,
                    workers=workers,
                    aggregate_tera_ops_per_second=aggregate_rate(case, REPEATS, elapsed_ms),
                )
            )
        final = [receive(pair[0]) for pair in pairs]
        for process in processes:
            process.join(TIMEOUT_SECONDS)
            if process.is_alive() or process.exitcode != 0:
                raise RuntimeError("assigned benchmark worker did not exit cleanly")
        timing = summary([row["elapsed_ms"] for row in measured])
        return dict(
            case=case.to_dict(),
            repeats=REPEATS,
            samples=measured,
            coordinator_wall=timing,
            aggregate_tera_ops_per_second=aggregate_rate(case, REPEATS, timing["median_ms"]),
            qualification=qualifications,
            final_validation=final,
        )
    finally:
        barrier.abort()
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join()
        for pair in pairs:
            pair[0].close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=15)
    args = parser.parse_args()
    if args.output.exists() or args.samples < 3:
        parser.error("fresh output and at least three samples required")
    if os.environ.get("ASCEND_RT_VISIBLE_DEVICES") != "2,3":
        parser.error("this experiment owns only physical NPU devices 2,3")
    report = dict(
        completed=False,
        physical_devices=DEVICES,
        started_utc=datetime.now(UTC).isoformat(),
        contract=__doc__,
        cases=[],
        source_sha256={
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ("benchmark.py", "cases.py", "card.py")
        },
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    save()
    for dtype in ("fp16", "int8"):
        for size in (4096, 8192):
            case = Case(f"square{size}", "gemm", dtype, size, size, size)
            report["cases"].append(run(case, args.samples))
            save()
            print(
                json.dumps({key: report["cases"][-1][key] for key in ("case", "aggregate_tera_ops_per_second")}),
                flush=True,
            )
    report.update(completed=True, ended_utc=datetime.now(UTC).isoformat())
    save()


if __name__ == "__main__":
    main()
