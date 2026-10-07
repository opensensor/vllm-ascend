# SPDX-License-Identifier: Apache-2.0
"""Paired 310P grouped-W3 operator timing for canonical and NZ storage."""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
import torch_npu

from vllm_ascend.models.glm5next_w2.model import _pack_codes_nz_w3
from vllm_ascend.utils import enable_custom_op

EXPERTS = 8
WARMUP = 3
REPEATS = 12
GEOMETRIES = (("gate_up", 4096, 4096), ("down", 4096, 2048))
ROUTE_COUNTS = (
    ("singleton", [1] * EXPERTS),
    ("repeated", [32] + [0] * (EXPERTS - 1)),
    ("prefill", [18] * EXPERTS),
    ("mixed_fallback", [129, 0, 18] + [0] * (EXPERTS - 3)),
    ("large_prefill", [640] + [0] * (EXPERTS - 1)),
)


def timed_call(op, inputs, codes, scales, ends) -> float:
    torch.npu.synchronize()
    start = time.perf_counter()
    op(inputs, codes, scales, ends)
    torch.npu.synchronize()
    return (time.perf_counter() - start) * 1000


def benchmark_case(name: str, n: int, k: int, route: str, counts: list[int], repeats: int) -> dict:
    generator = torch.Generator().manual_seed(31003 + n + k + sum(counts))
    canonical_cpu = torch.randint(0, 256, (EXPERTS, n, k * 3 // 8), generator=generator, dtype=torch.uint8)
    nz_cpu = torch.stack([_pack_codes_nz_w3(canonical_cpu[expert], k) for expert in range(EXPERTS)])
    inputs = torch.randn(sum(counts), k, generator=generator).half().npu()
    scales = (torch.rand(EXPERTS, n // 32, k // 32, generator=generator) * 0.02 + 0.005).npu()
    ends = torch.tensor(counts, dtype=torch.int64).cumsum(0).npu()
    canonical = canonical_cpu.npu()
    nz = nz_cpu.view(torch.int8).npu()
    op = torch.ops._C_ascend.npu_w2_grouped_blocked_dequant_matmul_310

    expected = op(inputs, canonical, scales, ends).cpu()
    actual = op(inputs, nz, scales, ends).cpu()
    if not torch.equal(expected.view(torch.uint8), actual.view(torch.uint8)):
        raise AssertionError(f"{name}/{route}: NZ and canonical outputs differ")

    for _ in range(WARMUP):
        timed_call(op, inputs, canonical, scales, ends)
        timed_call(op, inputs, nz, scales, ends)
    samples = {"canonical": [], "nz": []}
    for repeat in range(repeats):
        order = ("canonical", "nz") if repeat % 2 == 0 else ("nz", "canonical")
        for layout in order:
            codes = canonical if layout == "canonical" else nz
            samples[layout].append(timed_call(op, inputs, codes, scales, ends))
    medians = {layout: statistics.median(values) for layout, values in samples.items()}
    return {
        "projection": name,
        "output_width": n,
        "input_width": k,
        "route": route,
        "rows": sum(counts),
        "expert_count": EXPERTS,
        "bitwise_equal": True,
        "canonical_median_ms": medians["canonical"],
        "nz_median_ms": medians["nz"],
        "speedup": medians["canonical"] / medians["nz"],
        "samples_ms": samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=REPEATS)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    torch.npu.set_device(0)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    enable_custom_op()
    cases = [
        benchmark_case(name, n, k, route, counts, args.repeats)
        for name, n, k in GEOMETRIES
        for route, counts in ROUTE_COUNTS
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"cases": cases}, indent=2) + "\n")
    for case in cases:
        print(
            f"{case['projection']} {case['route']}: "
            f"{case['canonical_median_ms']:.2f} -> {case['nz_median_ms']:.2f} ms "
            f"({case['speedup']:.2f}x)"
        )


if __name__ == "__main__":
    main()
