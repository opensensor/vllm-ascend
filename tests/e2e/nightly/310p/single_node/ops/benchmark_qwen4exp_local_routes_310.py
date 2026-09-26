# SPDX-License-Identifier: Apache-2.0
"""Compare full and local-only TP4 routed-expert intermediates on 310P."""

import argparse
import json
import statistics
import time

import torch
import torch_npu

from vllm_ascend.models.qwen4_exp.grouped_expert_dispatch import build_grouped_expert_dispatch
from vllm_ascend.models.qwen4_exp.moe import _w8a8_packed_grouped_experts_npu
from vllm_ascend.utils import maybe_trans_nz

HIDDEN_SIZE = 2560
INTERMEDIATE_SIZE = 640
GLOBAL_EXPERTS = 512
LOCAL_EXPERTS = 128
TOP_K = 10


def full_routes(
    x: torch.Tensor,
    weights: torch.Tensor,
    ids: torch.Tensor,
    w13: torch.Tensor,
    s13: torch.Tensor,
    w2: torch.Tensor,
    s2: torch.Tensor,
) -> torch.Tensor:
    """The original packed route path, including peer-owned output rows."""
    num_tokens, hidden = x.shape
    dispatch = build_grouped_expert_dispatch(
        weights, ids, num_local_experts=LOCAL_EXPERTS, expert_offset=0, weight_dtype=x.dtype
    )
    sorted_x = x[dispatch.token_indices[dispatch.order]]
    group_list = dispatch.group_list
    local_rows = torch.arange(num_tokens * TOP_K, device=x.device) < group_list[-1]
    gate_up = torch_npu.npu_quant_grouped_matmul_dequant(sorted_x, w13, s13, group_list)
    gate_up = torch.where(local_rows[:, None], gate_up, 0)
    activated = torch_npu.npu_swiglu(gate_up)
    routed = torch_npu.npu_quant_grouped_matmul_dequant(activated, w2, s2, group_list)
    routed = torch.where(local_rows[:, None], routed, 0)
    routed = routed.float() * dispatch.route_weights[dispatch.order]
    return routed[dispatch.inverse_order].view(num_tokens, TOP_K, hidden).sum(dim=1).to(x.dtype)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.tokens < 2 or args.repeats < 1:
        parser.error("tokens must be >= 2 and repeats must be >= 1")
    torch.manual_seed(48)
    device = "npu:0"
    x = torch.randn((args.tokens, HIDDEN_SIZE), dtype=torch.float16, device=device)
    ids = torch.randint(GLOBAL_EXPERTS, (args.tokens, TOP_K), dtype=torch.int32, device=device)
    weights = torch.rand((args.tokens, TOP_K), dtype=torch.float16, device=device)
    weights = weights / weights.sum(dim=1, keepdim=True)
    w13 = maybe_trans_nz(
        torch.randint(-8, 8, (LOCAL_EXPERTS, 2 * INTERMEDIATE_SIZE, HIDDEN_SIZE), dtype=torch.int8, device=device)
    )
    w2 = maybe_trans_nz(
        torch.randint(-8, 8, (LOCAL_EXPERTS, HIDDEN_SIZE, INTERMEDIATE_SIZE), dtype=torch.int8, device=device)
    )
    s13 = torch.full((LOCAL_EXPERTS, 2 * INTERMEDIATE_SIZE), 0.001, dtype=torch.float32, device=device)
    s2 = torch.full((LOCAL_EXPERTS, HIDDEN_SIZE), 0.001, dtype=torch.float32, device=device)

    def local_routes() -> torch.Tensor:
        return _w8a8_packed_grouped_experts_npu(x, weights, ids, w13, s13, w2, s2, 0)

    def measure(fn) -> tuple[torch.Tensor, float]:
        fn()
        torch_npu.npu.synchronize()
        durations = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            output = fn()
            torch_npu.npu.synchronize()
            durations.append((time.perf_counter() - start) * 1000)
        return output, statistics.median(durations)

    full, full_ms = measure(lambda: full_routes(x, weights, ids, w13, s13, w2, s2))
    local, local_ms = measure(local_routes)
    error = (full.float() - local.float()).abs().max().item()
    print(
        json.dumps(
            {
                "tokens": args.tokens,
                "full_ms": full_ms,
                "local_ms": local_ms,
                "speedup": full_ms / local_ms,
                "max_abs_error": error,
            }
        )
    )


if __name__ == "__main__":
    main()
