# SPDX-License-Identifier: Apache-2.0
import statistics
import time

import torch


def reference(logits, iterations=20, eps=1e-6):
    mix = torch.softmax(logits, dim=-1) + eps
    mix = mix / (mix.sum(dim=-2, keepdim=True) + eps)
    for _ in range(iterations - 1):
        mix = mix / (mix.sum(dim=-1, keepdim=True) + eps)
        mix = mix / (mix.sum(dim=-2, keepdim=True) + eps)
    return mix


def measure(function):
    for _ in range(3):
        function()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        output = function()
    times = []
    for _ in range(5):
        start = time.perf_counter()
        for _ in range(20):
            graph.replay()
        torch.npu.synchronize()
        times.append((time.perf_counter() - start) * 50)
    return statistics.median(times), output


@torch.inference_mode()
def probe(worker):
    inventory = []
    for name, module in worker._resident_wrappers()[0].runnable.named_modules():
        if hasattr(module, "use_310p_sinkhorn"):
            inventory.append({"name": name, "enabled": module.use_310p_sinkhorn})
    records = []
    gen = torch.Generator().manual_seed(310)
    for rows in (1, 2, 8, 160, 640):
        logits = torch.randn(rows, 4, 4, generator=gen).npu()
        baseline, expected = measure(lambda: reference(logits))
        native, actual = measure(lambda: torch.ops._C_ascend.mhc_sinkhorn_310(logits, 20, 1e-6))
        a, b = expected.cpu(), actual.cpu()
        records.append({"rows": rows, "reference_ms": baseline, "native_ms": native,
                        "max_abs": float((a-b).abs().max()), "mismatches": int((a != b).sum()),
                        "fp16_mismatches": int((a.half() != b.half()).sum())})
    return {"inventory": inventory, "measurements": records}


def replacements(native_resources=None):
    changes = serving_replacements(native_resources)
    target = "vllm_ascend._310p.worker_310p:NPUWorker310.resident_status"
    original = changes[target]
    evidence = {}

    def status(self):
        if not evidence:
            try:
                evidence.update(probe(self))
            except Exception as exc:
                evidence["error"] = repr(exc)
        result = original(self)
        result["sinkhorn_probe"] = evidence
        return result

    changes[target] = status
    return changes
