# SPDX-License-Identifier: Apache-2.0
"""Check 768-token W3 grouped dispatch against two 640-bound calls on 310P."""

import json
import statistics
import time
from types import SimpleNamespace

import torch
import torch_npu

from vllm_ascend._310p.quantization.methods.w2_dynamic import (
    W2_GROUPED_MAX_ROUTES,
    AscendW2DynamicFusedMoEMethod310,
    _w2_grouped_mm_op,
)
from vllm_ascend.utils import enable_custom_op

LOCAL_EXPERTS = 72
HIDDEN = 4096
INTERMEDIATE = 2048
TOP_K = 8
CONTROL_CHUNK_TOKENS = 640
PREFILL_TOKENS = 768
REPEATS = 5


class GroupedBank(list):
    grouped_ready = True
    nz_packed_codes = True
    prefill_route_histogram = True
    local_expert_offset = 0
    num_local_experts = LOCAL_EXPERTS


def make_bank() -> GroupedBank:
    bank = GroupedBank([SimpleNamespace(hidden=HIDDEN, inter=INTERMEDIATE) for _ in range(LOCAL_EXPERTS)])
    gate_codes = torch.full((LOCAL_EXPERTS, INTERMEDIATE, HIDDEN * 3 // 8), 0x49, dtype=torch.int8, device="npu")
    gate_scales = torch.full((LOCAL_EXPERTS, INTERMEDIATE // 32, HIDDEN // 32), 0.01, dtype=torch.float32, device="npu")
    bank.gate_packed_bank = gate_codes
    bank.up_packed_bank = gate_codes
    bank.gate_scale_bank = gate_scales
    bank.up_scale_bank = gate_scales
    bank.gate_up_packed_bank = torch.cat((gate_codes, gate_codes), dim=1)
    bank.gate_up_scale_bank = torch.cat((gate_scales, gate_scales), dim=1)
    bank.down_packed_bank = torch.full(
        (LOCAL_EXPERTS, HIDDEN, INTERMEDIATE * 3 // 8), 0x49, dtype=torch.int8, device="npu"
    )
    bank.down_scale_bank = torch.full(
        (LOCAL_EXPERTS, HIDDEN // 32, INTERMEDIATE // 32), 0.01, dtype=torch.float32, device="npu"
    )
    return bank


def timed(operation) -> float:
    operation()
    torch.npu.synchronize()
    samples = []
    for _ in range(REPEATS):
        start = time.perf_counter()
        operation()
        torch.npu.synchronize()
        samples.append((time.perf_counter() - start) * 1000)
    return statistics.median(samples)


def main() -> None:
    if W2_GROUPED_MAX_ROUTES < PREFILL_TOKENS * TOP_K:
        raise RuntimeError("source does not admit 768-token top-8 grouped calls")
    torch.npu.set_device(0)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    enable_custom_op()
    grouped_op = _w2_grouped_mm_op()
    if grouped_op is None:
        raise RuntimeError("grouped packed operator unavailable")
    bank = make_bank()
    generator = torch.Generator().manual_seed(310)
    x = (torch.randn(PREFILL_TOKENS, HIDDEN, generator=generator) * 0.1).half().npu()
    ids = torch.randint(0, LOCAL_EXPERTS * 4, (PREFILL_TOKENS, TOP_K), generator=generator).npu()
    weights = torch.where(ids < LOCAL_EXPERTS, 1 / TOP_K, 0).float().npu()
    method = AscendW2DynamicFusedMoEMethod310()

    def direct():
        return method._apply_device_grouped(grouped_op, bank, x, weights, ids, None)

    def tiled():
        return torch.cat(
            [
                method._apply_device_grouped(
                    grouped_op,
                    bank,
                    x[start : start + CONTROL_CHUNK_TOKENS],
                    weights[start : start + CONTROL_CHUNK_TOKENS],
                    ids[start : start + CONTROL_CHUNK_TOKENS],
                    None,
                )
                for start in range(0, PREFILL_TOKENS, CONTROL_CHUNK_TOKENS)
            ]
        )

    reference = tiled()
    candidate = direct()
    torch.testing.assert_close(candidate, reference, atol=0, rtol=0)
    print(
        json.dumps(
            {
                "tokens": PREFILL_TOKENS,
                "routes": PREFILL_TOKENS * TOP_K,
                "local_experts": LOCAL_EXPERTS,
                "bitwise_equal": True,
                "tiled_ms": timed(tiled),
                "direct_ms": timed(direct),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
