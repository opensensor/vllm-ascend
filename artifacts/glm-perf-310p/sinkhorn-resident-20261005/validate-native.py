# SPDX-License-Identifier: Apache-2.0
import json
import statistics
import time
from pathlib import Path

import torch
import torch_npu  # noqa: F401

from sinkhorn_native import SinkhornNormalize, reference_normalize


def measure(function):
    for _ in range(3):
        function()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        output = function()
    samples = []
    for _ in range(7):
        start = time.perf_counter()
        for _ in range(30):
            graph.replay()
        torch.npu.synchronize()
        samples.append((time.perf_counter() - start) * 1000 / 30)
    return statistics.median(samples), graph, output


@torch.inference_mode()
def main():
    root = Path(__file__).resolve().parent
    torch.npu.set_device(0)
    torch.npu.set_compile_mode(jit_compile=False)
    torch.ops.load_library(str(root / "glm_sinkhorn_bridge_v1.so"))
    records, timings = [], []
    generator = torch.Generator().manual_seed(310)
    for order in (2,):
        for iterations in (1, 20, 64):
            op = SinkhornNormalize(str(root / "normalize-v1.bin"), iterations=iterations, order=order)
            for rows in (1, 2, 4, 8):
                for scale in (0, 0.1, 1, 10, 100):
                    logits = (torch.randn(rows, 4, 4, generator=generator) * scale).npu()
                    mix = torch.softmax(logits, -1) + op.epsilon
                    expected = reference_normalize(mix, iterations, op.epsilon)
                    actual = op(mix)
                    torch.npu.synchronize()
                    a, b = actual.cpu(), expected.cpu()
                    records.append({"order": order, "iterations": iterations, "rows": rows, "scale": scale,
                                    "max_abs": (a-b).abs().max().item(), "fp32_mismatches": int((a != b).sum()),
                                    "fp16_mismatches": int((a.half() != b.half()).sum()),
                                    "finite": bool(torch.isfinite(a).all())})
                if iterations == 20:
                    base_ms, _, _ = measure(lambda: reference_normalize(mix, iterations, op.epsilon))
                    new_ms, graph, output = measure(lambda: op(mix))
                    replay = []
                    for _ in range(5):
                        logits = torch.randn(rows, 4, 4, generator=generator).npu()
                        mix.copy_(torch.softmax(logits, -1) + op.epsilon)
                        graph.replay()
                        expected = reference_normalize(mix, iterations, op.epsilon)
                        torch.npu.synchronize()
                        a, b = output.cpu(), expected.cpu()
                        replay.append({"max_abs": (a-b).abs().max().item(),
                                       "fp16_mismatches": int((a.half() != b.half()).sum())})
                    timings.append({"order": order, "rows": rows, "reference_ms": base_ms,
                                    "native_ms": new_ms, "replay": replay})
                    print(json.dumps(timings[-1]), flush=True)
    result = {"cases": records, "timings": timings}
    (root / "native-results.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({"cases": len(records), "max_abs": max(r["max_abs"] for r in records),
                      "fp16_mismatches": sum(r["fp16_mismatches"] for r in records)}), flush=True)


if __name__ == "__main__":
    main()
