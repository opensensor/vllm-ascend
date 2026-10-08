# SPDX-License-Identifier: Apache-2.0
"""Measure the existing fixed-capacity selector against live paged scoring.

Use only with released NPU hardware; this script never launches a server.
"""

import argparse
import importlib.util
import json
import statistics
from pathlib import Path

import torch
import torch_npu

from vllm_ascend.models.glm5next.kpool_ops import expand_kpool_groups, select_kpool_groups
from vllm_ascend.utils import enable_custom_op


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--extension", required=True)
    parser.add_argument("--fixtures", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.npu.set_device(0)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    enable_custom_op()
    torch.ops.load_library(args.extension)
    spec = importlib.util.spec_from_file_location("fixtures", args.fixtures)
    fixtures = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixtures)
    records = []
    for rows, valid in [(2, 64), (8, 64), (2, 2048), (2, 32768), (2, 77760)]:
        for pools in [8192, 32768, 77760]:
            if valid > pools:
                continue
            values = fixtures.inputs(rows, pools, [valid] * rows)
            q, weights, storage, table, bounds, pos = values[:6]
            _, blocks, br, bs, rs, offset = values[6:]
            cache = storage.as_strided((blocks, br, 128), (bs, rs, 1), offset)

            def baseline(
                pools=pools, bounds=bounds, rows=rows, table=table, pos=pos, br=br, cache=cache, q=q, weights=weights
            ):
                outputs = []
                ids = torch.arange(pools, device="npu", dtype=torch.long)
                requests = torch.searchsorted(bounds, torch.arange(rows, device="npu", dtype=bounds.dtype), right=True)
                requests = requests.clamp(max=table.shape[0] - 1).long()
                for row in range(rows):
                    pages = table.index_select(0, requests[row : row + 1])[0]
                    mask = ids < (pos[row] + 1) // 4
                    physical = torch.where(mask, pages[ids // br].long(), 0)
                    keys = torch.where(mask[:, None], cache[physical, ids % br], 0)
                    logits = (q[row].float() @ keys.float().T).relu_()
                    score = (logits * weights[row, :, None]).sum(0, keepdim=True)
                    selected, _, starts, counts = select_kpool_groups(score, pos[row : row + 1], 2048, 4)
                    outputs.append(expand_kpool_groups(selected, starts, counts, 4))
                return outputs

            def candidate(values=values, rows=rows, pos=pos):
                scores = fixtures.op(values)
                outputs = []
                # Keep exactly the per-row top-k shape/order from serving.
                for row in range(rows):
                    selected, _, starts, counts = select_kpool_groups(
                        scores[row : row + 1], pos[row : row + 1], 2048, 4
                    )
                    outputs.append(expand_kpool_groups(selected, starts, counts, 4))
                return outputs

            # Compare selected sets and ordering separately: tiny score changes
            # can reorder near ties even when the selected key set is identical.
            old, new = baseline(), candidate()
            old_cpu = [value.cpu() for value in old]
            new_cpu = [value.cpu() for value in new]
            # All negative indices denote padding. Keep raw equality separate:
            # the existing backend can emit negative values other than -1.
            overlap = [float(torch.isin(b[b >= 0], a[a >= 0]).float().mean()) for a, b in zip(old_cpu, new_cpu)]
            exact = [bool(torch.equal(a, b)) for a, b in zip(old_cpu, new_cpu)]
            valid_order = [bool(torch.equal(a[a >= 0], b[b >= 0])) for a, b in zip(old_cpu, new_cpu)]
            padding_counts = [(int((a < 0).sum()), int((b < 0).sum())) for a, b in zip(old_cpu, new_cpu)]
            graphs = {}
            for label, fn in [("baseline", baseline), ("candidate", candidate)]:
                stream = torch.npu.Stream()
                stream.wait_stream(torch.npu.current_stream())
                with torch.npu.stream(stream):
                    for _ in range(3):
                        fn()
                torch.npu.current_stream().wait_stream(stream)
                graph = torch.npu.NPUGraph()
                torch.npu.synchronize()
                torch.npu.reset_peak_memory_stats()
                before = torch.npu.memory_allocated()
                with torch.npu.graph(graph, stream=stream):
                    output = fn()
                peak = torch.npu.max_memory_allocated() - before
                graphs[label] = (graph, output, peak)
            timings = {name: [] for name in graphs}
            for repeat in range(9):
                for name in list(graphs) if repeat % 2 == 0 else list(reversed(graphs)):
                    graph = graphs[name][0]
                    start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
                    start.record()
                    for _ in range(10):
                        graph.replay()
                    end.record()
                    end.synchronize()
                    timings[name].append(start.elapsed_time(end) / 10)
            result = {
                "rows": rows,
                "live_pools": valid,
                "capacity_pools": pools,
                "selected_overlap": overlap,
                "exact_order": exact,
                "valid_order_equal": valid_order,
                "padding_counts": padding_counts,
                "median_ms": {k: statistics.median(v) for k, v in timings.items()},
                "samples_ms": timings,
                "capture_peak_bytes": {k: v[2] for k, v in graphs.items()},
            }
            records.append(result)
            print(json.dumps(result), flush=True)
            Path(args.output).write_text(json.dumps(records, indent=2) + "\n")
            del graphs


if __name__ == "__main__":
    main()
