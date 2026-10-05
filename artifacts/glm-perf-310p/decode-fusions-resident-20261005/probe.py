# SPDX-License-Identifier: Apache-2.0
"""Isolated graph timings in paused resident workers; no model dispatch change."""

import statistics

import torch

from tools.glm_perf.resident_worker import ResidentWorkerExtension
from vllm_ascend.models.glm5next.kpool_ops import hadamard128


def measure(functions):
    graphs = {}
    try:
        for name, fn in functions.items():
            stream = torch.npu.Stream()
            stream.wait_stream(torch.npu.current_stream())
            with torch.npu.stream(stream):
                for _ in range(3):
                    fn()
            torch.npu.current_stream().wait_stream(stream)
            torch.npu.synchronize()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph, stream=stream):
                output = fn()
            graphs[name] = (graph, output)
        samples = {name: [] for name in graphs}
        for repeat in range(7):
            names = list(graphs)
            for name in names if repeat % 2 == 0 else reversed(names):
                begin, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
                begin.record()
                for _ in range(20):
                    graphs[name][0].replay()
                end.record()
                end.synchronize()
                samples[name].append(begin.elapsed_time(end) / 20)
        return {
            name: {"median_ms": statistics.median(values), "samples_ms": values} for name, values in samples.items()
        }
    finally:
        torch.npu.synchronize()
        for graph, _ in graphs.values():
            graph.reset()


def audit():
    results = []
    generator = torch.Generator().manual_seed(310)
    for rows in (2, 8):
        x = torch.randn(rows, 4096, generator=generator).half().npu()
        residual = torch.randn(rows, 4, 4096, generator=generator).half().float().npu()
        post = torch.randn(rows, 4, 1, generator=generator).sigmoid().half().float().npu()
        comb = torch.randn(rows, 4, 4, generator=generator).softmax(-1).half().float().npu()
        query = torch.randn(rows, 32, 128, generator=generator).half().npu()

        def reference(x=x, residual=residual, post=post, comb=comb):
            return (torch.einsum("nij,nih->njh", comb, residual) + post * x.float().unsqueeze(1)).half().float()

        def native(x=x, residual=residual, post=post, comb=comb):
            return torch.ops._C_ascend.npu_glm_mhc_post_310(x, residual, post, comb)

        expected, actual = reference(), native()
        torch.testing.assert_close(actual, expected, rtol=1e-3, atol=2e-6)
        results.append(
            {
                "rows": rows,
                "post_max_abs_error": (actual - expected).abs().max().item(),
                "post_changed_elements": (actual != expected).sum().item(),
                "timing": measure(
                    {
                        "post_reference": reference,
                        "post_native": native,
                        "query_rotation": lambda query=query: hadamard128(query).bfloat16().half(),
                    }
                ),
            }
        )
    return {"results": results}


def replacements():
    result = {}

    def status(self):
        if not result:
            try:
                result.update(audit())
            except Exception as exc:
                result["error_detail"] = f"{type(exc).__name__}: {exc}"
        receipt = ResidentWorkerExtension.resident_status(self)
        receipt["decode_fusion_audit"] = result
        return receipt

    return {"vllm_ascend._310p.worker_310p:NPUWorker310.resident_status": status}
