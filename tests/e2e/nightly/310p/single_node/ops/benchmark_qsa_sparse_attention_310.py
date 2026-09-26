# SPDX-License-Identifier: Apache-2.0
"""Standalone 310P sparse-attention benchmark with the model's QSA geometry.

Run with a candidate vendor's op_api/lib first in LD_LIBRARY_PATH, then pass
its vendors/custom_transformer directory via --vendor-path. The benchmark does
not load model weights and reports synchronized operator time only.
"""

import argparse
import json
import os
import time

import torch
import torch_npu

from vllm_ascend.models.qwen4_exp.ops.qsa_indexer import QSAGroupSelection
from vllm_ascend.models.qwen4_exp.ops.qsa_sparse_attention_310 import qsa_sparse_attention_310
from vllm_ascend.utils import enable_custom_op


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vendor-path", required=True)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--groups", type=int, default=512)
    parser.add_argument("--cache-tokens", type=int, default=None)
    parser.add_argument("--vary-groups", action="store_true")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    if not torch_npu.npu.is_available() or "310" not in torch_npu.npu.get_device_name(0):
        raise RuntimeError("requires an Ascend 310P NPU")
    if args.tokens <= 0 or args.groups <= 0 or args.repeats <= 0:
        parser.error("tokens, groups, and repeats must be positive")

    enable_custom_op()
    os.environ["ASCEND_CUSTOM_OPP_PATH"] = args.vendor_path
    device = "npu:0"
    block_size = 512
    cache_tokens = args.cache_tokens or args.groups * 4
    if cache_tokens < args.groups * 4 or cache_tokens % block_size != 0:
        parser.error("cache-tokens must be a multiple of 512 and hold all selected groups")
    group_stride = (cache_tokens // 4) // args.groups
    num_query_heads = 24
    num_kv_heads = 2
    head_dim = 256
    num_blocks = cache_tokens // block_size
    query = torch.randn((args.tokens, num_query_heads, head_dim), dtype=torch.float16, device=device)
    cache_shape = (num_blocks, num_kv_heads * head_dim // 16, block_size, 16)
    key_cache = torch_npu.npu_format_cast(torch.randn(cache_shape, dtype=torch.float16, device=device), 29)
    value_cache = torch_npu.npu_format_cast(torch.randn(cache_shape, dtype=torch.float16, device=device), 29)
    group_indices = torch.arange(args.groups, dtype=torch.int32, device=device) * group_stride
    if args.vary_groups:
        token_offsets = torch.arange(args.tokens, dtype=torch.int32, device=device).unsqueeze(1)
        group_indices = (group_indices + token_offsets * (args.groups + 1)) % (cache_tokens // 4)
    else:
        group_indices = group_indices.expand(args.tokens, -1).contiguous()
    selection = QSAGroupSelection(
        group_indices=group_indices,
        group_counts=torch.full((args.tokens,), args.groups, dtype=torch.int32, device=device),
        tail_starts=torch.zeros(args.tokens, dtype=torch.int32, device=device),
        tail_counts=torch.zeros(args.tokens, dtype=torch.int32, device=device),
    )
    block_table = torch.arange(num_blocks, dtype=torch.int32, device=device).unsqueeze(0)
    query_start_loc = torch.tensor([0, args.tokens], dtype=torch.int32, device=device)

    def run() -> None:
        qsa_sparse_attention_310(query, key_cache, value_cache, selection, block_table, query_start_loc)
        torch_npu.npu.synchronize()

    run()
    samples = []
    for _ in range(args.repeats):
        start = time.perf_counter()
        run()
        samples.append((time.perf_counter() - start) * 1000)
    print(
        json.dumps(
            {
                "tokens": args.tokens,
                "groups": args.groups,
                "cache_tokens": cache_tokens,
                "vary_groups": args.vary_groups,
                "operator_ms": samples,
            }
        )
    )


if __name__ == "__main__":
    main()
