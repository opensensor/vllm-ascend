# SPDX-License-Identifier: Apache-2.0
"""Isolated graph timings in paused resident workers; no model dispatch change."""

import statistics

import torch

from tools.glm_perf.resident_worker import ResidentWorkerExtension


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
    import torch_npu

    records = []
    generator = torch.Generator().manual_seed(310)
    for rows in (2, 8, 640):
        x = (torch.randn(rows, 4096, generator=generator) * 20).half().float().npu()
        weight = torch.randn(4096, generator=generator).half().npu()
        eps = 1e-5

        def reference(x=x, weight=weight, eps=eps):
            return (x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps) * weight.float()).half()

        def native(x=x, weight=weight, eps=eps):
            return torch_npu.npu_rms_norm(x, weight.float(), eps)[0].half()

        expected, actual = reference(), native()
        torch.testing.assert_close(actual, expected, rtol=1e-3, atol=2e-6)
        records.append(
            {
                "rows": rows,
                "max_abs": (actual - expected).abs().max().item(),
                "changed_elements": (actual != expected).sum().item(),
                "timing": measure({"reference": reference, "native": native}),
            }
        )
    return {"results": records}


def replacements():
    result = {}

    def status(self):
        if not result:
            try:
                result.update(audit())
            except Exception as exc:
                result["error_detail"] = f"{type(exc).__name__}: {exc}"
        receipt = ResidentWorkerExtension.resident_status(self)
        receipt["mhc_norm_audit"] = result
        return receipt

    return {"vllm_ascend._310p.worker_310p:NPUWorker310.resident_status": status}
