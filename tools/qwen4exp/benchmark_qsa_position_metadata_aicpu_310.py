# SPDX-License-Identifier: Apache-2.0
"""Isolated QSA position-metadata AICPU gate; no serving integration."""

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

CASES = ((3, 5856, 512), (64, 10000, 512), (256, 10000, 512), (2048, 10000, 512))
RATIO = 4


def timed_ms(function, iterations, trials, torch):
    for _ in range(3):
        function()
    torch.npu.synchronize()
    samples = []
    for _ in range(trials):
        start = time.perf_counter()
        for _ in range(iterations):
            function()
        torch.npu.synchronize()
        samples.append(1000 * (time.perf_counter() - start) / iterations)
    return {"median_ms": statistics.median(samples), "trials_ms": samples}


def torch_geometry(positions, ratio, capacity, width, torch):
    following = positions + 1
    complete = torch.div(following, ratio, rounding_mode="floor")
    tail_starts = complete * ratio
    visible = complete.clamp_max(capacity)
    return visible, visible.clamp_max(width), tail_starts, following - tail_starts


def candidate_geometry(candidate, positions, ratio, capacity, width):
    metadata = candidate(positions, ratio, capacity, width)
    return metadata[0], metadata[1], metadata[2], metadata[3]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--trials", type=int, default=5)
    args = parser.parse_args()
    if args.iterations <= 0 or args.trials <= 0:
        parser.error("iterations and trials must be positive")
    if args.dry_run:
        print(json.dumps({"cases": CASES, "ratio": RATIO, "npu_used": False}, indent=2))
        return
    if args.output is None or args.output.exists():
        parser.error("--output must be a new path")

    import torch
    import torch_npu

    from vllm_ascend.utils import enable_custom_op

    if not torch_npu.npu.is_available() or "310" not in torch_npu.npu.get_device_name(0):
        raise RuntimeError("requires an Ascend 310P; no experiment was run")
    torch.npu.set_device(0)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    enable_custom_op()
    candidate = getattr(torch.ops._C_ascend, "npu_qsa_position_metadata_aicpu_310", None)
    if candidate is None:
        raise RuntimeError("experimental AI CPU metadata operator is not registered")

    source = Path(__file__).resolve().parents[2] / "csrc/attention/qsa_position_metadata_aicpu_v310/position_metadata.h"
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    with args.output.open("x") as output:
        for queries, capacity, width in CASES:
            for pattern in ("sequential", "padded"):
                host_positions = torch.arange(capacity * RATIO - queries, capacity * RATIO, dtype=torch.int32)
                if pattern == "padded":
                    host_positions[::8] = -1
                positions = host_positions.npu()
                expected = torch_geometry(positions, RATIO, capacity, width, torch)
                actual = candidate_geometry(candidate, positions, RATIO, capacity, width)
                for actual_field, expected_field in zip(actual, expected, strict=True):
                    torch.testing.assert_close(actual_field.cpu(), expected_field.cpu(), rtol=0, atol=0)
                positions.add_(2)
                changed_expected = torch_geometry(positions, RATIO, capacity, width, torch)
                changed_actual = candidate_geometry(candidate, positions, RATIO, capacity, width)
                for actual_field, expected_field in zip(changed_actual, changed_expected, strict=True):
                    torch.testing.assert_close(actual_field.cpu(), expected_field.cpu(), rtol=0, atol=0)
                positions.sub_(2)
                candidate_time = timed_ms(
                    lambda p=positions, c=capacity, w=width: candidate_geometry(candidate, p, RATIO, c, w),
                    args.iterations,
                    args.trials,
                    torch,
                )
                baseline_time = timed_ms(
                    lambda p=positions, c=capacity, w=width: torch_geometry(p, RATIO, c, w, torch),
                    args.iterations,
                    args.trials,
                    torch,
                )
                record = {
                    "queries": queries,
                    "capacity": capacity,
                    "selected_width": width,
                    "pattern": pattern,
                    "aicpu_ms": candidate_time,
                    "torch_ms": baseline_time,
                    "source_sha256": source_hash,
                    "exact_parity": True,
                    "model_ttft_measured": False,
                }
                line = json.dumps(record)
                print(line, flush=True)
                output.write(line + "\n")
                output.flush()


if __name__ == "__main__":
    main()
