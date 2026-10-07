# SPDX-License-Identifier: Apache-2.0
"""Offline-prepared prefill benchmark. Run only with released NPU hardware."""

import argparse
import importlib.util
import json
import statistics
from pathlib import Path

import torch

from tools.glm_perf.kpool_prefill import select_prefill_request
from vllm_ascend.models.glm5next.kpool_ops import score_and_select_kpool_tokens


def main():
    # Lazy imports keep --help and source inspection independent of NPU init.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extension", required=True)
    parser.add_argument("--fixtures", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rows", type=int, default=640)
    parser.add_argument("--pools", type=int, nargs="+", default=[2048, 8192, 32768, 77760])
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    if args.rows <= 0 or args.repeats < 2 or any(p < args.rows or p % 8 for p in args.pools):
        parser.error("positive rows, >=2 repeats and aligned pool counts >= rows are required")
    import torch_npu

    from vllm_ascend.utils import enable_custom_op

    torch.npu.set_device(0)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    enable_custom_op()
    torch.ops.load_library(args.extension)
    spec = importlib.util.spec_from_file_location("prefill_fixtures", args.fixtures)
    fixtures = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixtures)
    records = []
    for pools in args.pools:
        values = fixtures.inputs(args.rows, pools)
        q, weights, storage, table, _, _ = values[:6]
        _, blocks, br, bs, rs, offset = values[6:]
        cache = storage.as_strided((blocks, br, 1, 128), (bs, rs, rs, 1), offset)
        positions = torch.arange(pools * 4 - args.rows, pools * 4, device=q.device, dtype=torch.int32)

        def baseline(pools=pools, q=q, cache=cache, table=table, br=br, weights=weights, positions=positions):
            ids = torch.arange(pools, device=q.device)
            keys = cache[table[0, ids // br].long(), ids % br, 0]
            return score_and_select_kpool_tokens(q, weights, keys, positions, 2048, 4)

        def candidate(q=q, weights=weights, cache=cache, table=table, positions=positions, pools=pools):
            return select_prefill_request(q, weights, cache, table, positions, pools, 2048, 4)

        expected, actual = baseline().cpu(), candidate().cpu()
        shape_equal = actual.shape == expected.shape
        exact = shape_equal and torch.equal(actual, expected)
        sets_equal = shape_equal and torch.equal(actual.sort().values, expected.sort().values)
        timings = {"baseline": [], "candidate": []}
        peaks = {"baseline": [], "candidate": []}
        functions = {"baseline": baseline, "candidate": candidate}
        for function in functions.values():
            for _ in range(2):
                function()
        torch.npu.synchronize()
        for repeat in range(args.repeats):
            names = list(functions) if repeat % 2 == 0 else list(reversed(functions))
            for name in names:
                torch.npu.reset_peak_memory_stats()
                before = torch.npu.memory_allocated()
                begin, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
                begin.record()
                output = functions[name]()
                end.record()
                end.synchronize()
                timings[name].append(begin.elapsed_time(end))
                peaks[name].append(torch.npu.max_memory_allocated() - before)
                del output
        result = {
            "rows": args.rows,
            "live_context_tokens": pools * 4,
            "selected_sets_equal": sets_equal,
            "selected_order_equal": exact,
            "median_ms": {name: statistics.median(samples) for name, samples in timings.items()},
            "samples_ms": timings,
            "peak_allocated_delta_bytes": peaks,
            "scope": "full prefill selector including query rotation, page gather, scoring and top-k",
        }
        records.append(result)
        Path(args.output).write_text(json.dumps(records, indent=2) + "\n")
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
