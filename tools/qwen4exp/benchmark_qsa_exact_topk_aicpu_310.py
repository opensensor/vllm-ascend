# SPDX-License-Identifier: Apache-2.0
"""Isolated 310P QSA selector gate. Never starts or modifies the model server.

Use --dry-run on a host without an NPU. A real run requires an isolated custom
OPP and extension containing npu_qsa_exact_topk_aicpu_310.
"""

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

CASES = ((3, 5856, 512), (64, 10000, 512), (256, 10000, 512), (2048, 10000, 512))
PATTERNS = ("random", "ties", "masked")


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--trials", type=int, default=5)
    args = parser.parse_args()
    if args.iterations <= 0 or args.trials <= 0:
        parser.error("iterations and trials must be positive")
    if args.dry_run:
        print(json.dumps({"cases": CASES, "patterns": PATTERNS, "npu_used": False}, indent=2))
        return
    if args.output is None or args.output.exists():
        parser.error("--output must be a new path")

    import torch
    import torch_npu

    from vllm_ascend.models.qwen4_exp.ops.qsa_indexer import _fast_topk_indices
    from vllm_ascend.utils import enable_custom_op

    if not torch_npu.npu.is_available() or "310" not in torch_npu.npu.get_device_name(0):
        raise RuntimeError("requires an Ascend 310P; no experiment was run")
    torch.npu.set_device(0)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    enable_custom_op()
    candidate = getattr(torch.ops._C_ascend, "npu_qsa_exact_topk_aicpu_310", None)
    if candidate is None:
        raise RuntimeError("experimental AI CPU selector is not registered")

    source = Path(__file__).resolve().parents[2] / "csrc/attention/qsa_exact_topk_aicpu_v310/exact_topk.h"
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    generator = torch.Generator().manual_seed(1024)
    with args.output.open("x") as output:
        for queries, groups, topk in CASES:
            for pattern in PATTERNS:
                if pattern == "ties":
                    host_scores = torch.randint(0, 16, (queries, groups), generator=generator).float()
                else:
                    host_scores = torch.rand((queries, groups), generator=generator)
                if pattern == "masked":
                    host_scores[:, groups // 2 :] = -torch.inf
                expected = torch.argsort(host_scores, dim=1, descending=True, stable=True)[:, :topk].int()
                device_scores = host_scores.npu()
                actual = candidate(device_scores, topk).cpu()
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                baseline = _fast_topk_indices(device_scores, topk)
                baseline_equal = bool(torch.equal(baseline.int().cpu(), expected))
                candidate_time = timed_ms(
                    lambda scores=device_scores, k=topk: candidate(scores, k), args.iterations, args.trials, torch
                )
                baseline_time = timed_ms(
                    lambda scores=device_scores, k=topk: _fast_topk_indices(scores, k),
                    args.iterations,
                    args.trials,
                    torch,
                )
                record = {
                    "queries": queries,
                    "groups": groups,
                    "topk": topk,
                    "pattern": pattern,
                    "candidate_exact": True,
                    "current_fast_topk_exact": baseline_equal,
                    "aicpu_ms": candidate_time,
                    "current_prefill_ms": baseline_time,
                    "selector_sha256": source_hash,
                    "comparison": "isolated selection only; scores precomputed on NPU",
                    "model_ttft_measured": False,
                }
                line = json.dumps(record)
                print(line, flush=True)
                output.write(line + "\n")
                output.flush()


if __name__ == "__main__":
    main()
