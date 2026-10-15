# SPDX-License-Identifier: Apache-2.0
"""Discover exact reduction order before implementing HC gated-mean fusion."""

import argparse
import json
from pathlib import Path

import torch

from tools.qwen4exp.benchmark_math_paths_310 import compare_outputs, gate_passed

PATTERNS = ("random", "zero", "large", "cancellation", "subnormal")


def alternatives(product):
    a, b, c, d = product.unbind(1)
    return {
        "serial0123": ((a + b) + c) + d,
        "serial3210": ((d + c) + b) + a,
        "pair01_23": (a + b) + (c + d),
        "pair02_13": (a + c) + (b + d),
        "pair03_12": (a + d) + (b + c),
    }


def inputs(rows, pattern):
    generator = torch.Generator().manual_seed(9163)
    normalized = torch.randn(rows, 4, 2560, generator=generator)
    gate = torch.sigmoid(torch.randn(rows, 4, 2560, generator=generator)).half()
    if pattern == "zero":
        normalized.zero_()
    elif pattern == "large":
        normalized.mul_(16384)
    elif pattern == "cancellation":
        normalized[:, 0].fill_(1)
        normalized[:, 1].fill_(-1)
        normalized[:, 2].fill_(2**-23)
        normalized[:, 3].fill_(2**-24)
        gate.fill_(1)
    elif pattern == "subnormal":
        normalized.mul_(2**-24)
    return normalized, gate


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("preserve existing evidence")
    import torch_npu

    torch.npu.set_device(0)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    report = {"scope": "synthetic_reduction_order_not_model_quality", "records": [], "exact_orders": None}
    passing = set(alternatives(torch.zeros(1, 4, 1)))
    for rows in (3, 6, 2560):
        for pattern in PATTERNS:
            normalized, gate = inputs(rows, pattern)
            normalized, gate = normalized.npu(), gate.npu()
            product = normalized * gate
            expected = product.mean(1).half()
            checks = {
                name: compare_outputs(expected, (value * 0.25).half()) for name, value in alternatives(product).items()
            }
            passing &= {name for name, check in checks.items() if gate_passed(check)}
            report["records"].append({"tokens": rows, "pattern": pattern, "checks": checks})
            report["exact_orders"] = sorted(passing)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(
                json.dumps({"tokens": rows, "pattern": pattern, "remaining_exact_orders": sorted(passing)}), flush=True
            )
    return 0 if passing else 1


if __name__ == "__main__":
    raise SystemExit(main())
