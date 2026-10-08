# SPDX-License-Identifier: Apache-2.0
"""Resident weight-layout probe, appended to the current candidate bundle."""

import statistics
import time
from functools import partial

import torch
import torch_npu


def measure_projection(function):
    for _ in range(3):
        function()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        output = function()
    times = []
    for _ in range(7):
        start = time.perf_counter()
        for _ in range(10):
            graph.replay()
        torch.npu.synchronize()
        times.append((time.perf_counter() - start) * 100)
    return {"median_ms": statistics.median(times), "samples_ms": times}, output


@torch.inference_mode()
def probe_projections(worker):
    candidates = []
    for name, module in worker._resident_wrappers()[0].runnable.named_modules():
        if hasattr(module, "_glm_kda_qkv_weight_nz"):
            for attr in ("f_b_proj", "g_b_proj", "o_proj"):
                projection = getattr(module, attr)
                candidates.append((name + "." + attr, projection))
    if not candidates:
        raise ValueError("no prepared KDA projections found")
    inventory = [
        {
            "name": name,
            "shape": list(layer.weight.shape),
            "dtype": str(layer.weight.dtype),
            "format": torch_npu.get_npu_format(layer.weight),
            "method": type(layer.quant_method).__name__,
        }
        for name, layer in candidates
    ]
    records, seen = [], set()
    generator = torch.Generator().manual_seed(310)
    for name, layer in candidates:
        geometry = tuple(layer.weight.shape)
        if geometry in seen or layer.weight.dtype != torch.float16:
            continue
        seen.add(geometry)
        # Original parameter is untouched; temporary packed storage only.
        packed = torch_npu.npu_format_cast(layer.weight.detach().T.contiguous(), 29).unsqueeze(0)
        for rows in (2, 8, 640):
            x = torch.randn(rows, geometry[1], generator=generator).half().to(layer.weight.device)
            ends = torch.tensor([rows], dtype=torch.int64, device=x.device)
            original = partial(layer.quant_method.apply, layer, x, bias=None)
            grouped = partial(
                torch_npu.npu_grouped_matmul, x=[x], weight=[packed], group_list=ends, split_item=2, group_type=0
            )

            def candidate(grouped=grouped):
                return grouped()[0]

            expected, actual = original(), candidate()
            torch.npu.synchronize()
            a, b = expected.cpu(), actual.cpu()
            record = {
                "name": name,
                "shape": geometry,
                "rows": rows,
                "mismatches": int((a != b).sum()),
                "max_abs": float((a.float() - b.float()).abs().max()),
            }
            record["original"], _ = measure_projection(original)
            record["grouped_nz"], _ = measure_projection(candidate)
            records.append(record)
    return {"inventory": inventory, "measurements": records}


def replacements(native_resources=None):
    changes = serving_replacements(native_resources)  # noqa: F821 - composed with current serving source
    target = "vllm_ascend._310p.worker_310p:NPUWorker310.resident_status"
    original = changes[target]
    evidence = {}

    def status(self):
        if not evidence:
            try:
                evidence.update(probe_projections(self))
            except Exception as exc:
                evidence["error"] = repr(exc)
        result = original(self)
        result["projection_probe"] = evidence
        return result

    changes[target] = status
    return changes
