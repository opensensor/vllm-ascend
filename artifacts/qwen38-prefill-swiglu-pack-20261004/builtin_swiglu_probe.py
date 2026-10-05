# SPDX-License-Identifier: Apache-2.0
"""Compare built-in 310P SwiGLU variants with the Qwen prefill path."""

import json
import statistics
import time

import torch
import torch.nn.functional as F
import torch_npu

from vllm_ascend.models.qwen4_exp.w4a8_int4 import pack_activation_device
from vllm_ascend.utils import enable_custom_op


def timed_ms(function) -> list[float]:
    for _ in range(2):
        function()
    torch.npu.synchronize()
    samples = []
    for _ in range(3):
        start = time.perf_counter()
        for _ in range(5):
            function()
        torch.npu.synchronize()
        samples.append((time.perf_counter() - start) * 200)
    return samples


def main() -> None:
    torch.npu.set_device(0)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    enable_custom_op()
    rows = 15360
    generator = torch.Generator().manual_seed(1000 + rows)
    gate_up = (torch.randn(rows, 1280, generator=generator) * 1.5).half().npu()

    def baseline():
        gate, up = gate_up.float().chunk(2, -1)
        return (F.silu(gate) * up).half()

    methods = {
        "torch": baseline,
        "builtin_fp32": lambda: torch_npu.npu_swiglu(gate_up.float(), dim=-1).half(),
        "builtin_fp16": lambda: torch_npu.npu_swiglu(gate_up, dim=-1),
    }
    reference = baseline().cpu()
    packed_reference = [tensor.cpu() for tensor in pack_activation_device(baseline())]
    report = {}
    for name, method in methods.items():
        try:
            candidate = method().cpu()
            packed = [tensor.cpu() for tensor in pack_activation_device(method())]
            counts = [int((value != target).sum()) for value, target in zip(packed, packed_reference)]
            activation_times = timed_ms(method)
            pack_times = timed_ms(lambda selected=method: pack_activation_device(selected()))
            report[name] = {
                "activation_mismatches": int((candidate != reference).sum()),
                "packed_mismatches": counts,
                "activation_median_ms": statistics.median(activation_times),
                "activation_pack_median_ms": statistics.median(pack_times),
                "activation_samples_ms": activation_times,
                "activation_pack_samples_ms": pack_times,
            }
        except Exception as error:
            report[name] = {"error": str(error)}
    print(json.dumps({"rows": rows, "results": report}), flush=True)


if __name__ == "__main__":
    main()
