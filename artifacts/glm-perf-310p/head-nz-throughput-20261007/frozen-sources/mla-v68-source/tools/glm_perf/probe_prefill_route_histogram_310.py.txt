# SPDX-License-Identifier: Apache-2.0
"""Compare GLM route counting and full dispatch on an idle 310P device."""

import json
import time

import torch
import torch_npu

from vllm_ascend.models.qwen4_exp.grouped_expert_dispatch import build_grouped_expert_dispatch

LOCAL_EXPERTS = 72
EXPERT_OFFSET = 72
TOP_K = 8
WARMUP = 5
REPEATS = 30


def elapsed_ms(operation) -> float:
    for _ in range(WARMUP):
        operation()
    torch.npu.synchronize()
    samples = []
    for _ in range(REPEATS):
        start = time.perf_counter()
        operation()
        torch.npu.synchronize()
        samples.append((time.perf_counter() - start) * 1000)
    samples.sort()
    return samples[len(samples) // 2]


def run(num_tokens: int) -> dict[str, float | int]:
    generator = torch.Generator().manual_seed(310)
    ids = torch.randint(0, LOCAL_EXPERTS * 3, (num_tokens, TOP_K), generator=generator).npu()
    weights = torch.rand((num_tokens, TOP_K), generator=generator).npu()
    pair_expert = ids.reshape(-1) - EXPERT_OFFSET
    pair_expert = torch.where((pair_expert >= 0) & (pair_expert < LOCAL_EXPERTS), pair_expert, LOCAL_EXPERTS)
    sort_key = pair_expert.to(torch.int32).to(torch.float32)
    local_ids = torch.arange(LOCAL_EXPERTS, device="npu", dtype=pair_expert.dtype)

    def compare_counts():
        return (pair_expert.unsqueeze(1) == local_ids).sum(dim=0)

    def histogram_counts():
        return torch.histc(sort_key, bins=LOCAL_EXPERTS + 1, min=0, max=LOCAL_EXPERTS)[:LOCAL_EXPERTS].to(torch.int64)

    def dispatch(mode: str):
        return build_grouped_expert_dispatch(
            weights,
            ids,
            num_local_experts=LOCAL_EXPERTS,
            expert_offset=EXPERT_OFFSET,
            weight_dtype=torch.float32,
            count_mode=mode,
        )

    torch.testing.assert_close(histogram_counts(), compare_counts(), atol=0, rtol=0)
    reference = dispatch("compare")
    candidate = dispatch("histogram")
    for field in ("counts", "order", "inverse_order", "token_indices", "route_weights"):
        torch.testing.assert_close(getattr(candidate, field), getattr(reference, field), atol=0, rtol=0)

    return {
        "tokens": num_tokens,
        "routes": num_tokens * TOP_K,
        "compare_counts_ms": elapsed_ms(compare_counts),
        "histogram_counts_ms": elapsed_ms(histogram_counts),
        "compare_dispatch_ms": elapsed_ms(lambda: dispatch("compare")),
        "histogram_dispatch_ms": elapsed_ms(lambda: dispatch("histogram")),
    }


if __name__ == "__main__":
    torch_npu.npu.set_compile_mode(jit_compile=False)
    for token_count in (128, 640, 768):
        print(json.dumps(run(token_count)), flush=True)
