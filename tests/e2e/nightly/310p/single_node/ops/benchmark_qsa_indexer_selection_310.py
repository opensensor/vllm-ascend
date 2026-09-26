# SPDX-License-Identifier: Apache-2.0
"""Compare padded and visible-prefix 310P QSA index selection.

Run with the custom operator's op_api/lib in LD_LIBRARY_PATH. The script
measures synchronized score plus stable top-k time without loading weights.
"""

import argparse
import json
import os
import time

import torch
import torch_npu

from vllm_ascend.models.qwen4_exp.ops.qsa_indexer import _stable_topk_indices, qsa_indexer_select_groups_310
from vllm_ascend.utils import enable_custom_op


def wide_matmul_scores(
    query: torch.Tensor,
    cache: torch.Tensor,
    block_table: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    """Decode-size GEMM candidate for the native per-group QSA score op."""
    groups_per_block = cache.shape[1] - 3
    physical_blocks = block_table[0].to(torch.long).clamp_min(0)
    keys = torch.index_select(cache, 0, physical_blocks)[:, :groups_per_block].reshape(-1, query.shape[-1])
    scores = torch.matmul(query.float(), keys.float().t()).relu_().sum(dim=1)
    visible_groups = ((positions.to(torch.long) + 1) // 4).clamp_max(scores.shape[1])
    group_ids = torch.arange(scores.shape[1], device=scores.device)
    return scores.masked_fill(group_ids.unsqueeze(0) >= visible_groups.unsqueeze(1), -torch.inf)


def legacy_candidate_topk(scores: torch.Tensor, k: int) -> torch.Tensor:
    """Pre-optimization selector, retained only for repeatable comparisons."""
    initial_values = torch.topk(scores, k, dim=1, sorted=False).values
    cutoff = initial_values.amin(dim=1, keepdim=True)
    negative_inf = torch.full_like(scores, -torch.inf)
    better_values, better_indices = torch.topk(
        torch.where(scores > cutoff, scores, negative_inf), k, dim=1, sorted=False
    )
    better_valid = better_values > cutoff
    indices = torch.arange(scores.shape[1], device=scores.device).expand_as(scores)
    tie_priority = torch.where(scores == cutoff, -indices.to(scores.dtype), negative_inf)
    tie_priorities, tie_indices = torch.topk(tie_priority, k, dim=1, sorted=False)
    tie_valid = torch.isfinite(tie_priorities)
    tie_values = torch.where(tie_valid, cutoff.expand_as(tie_priorities), tie_priorities)
    candidate_indices = torch.cat((better_indices, tie_indices), dim=1)
    candidate_values = torch.cat((better_values, tie_values), dim=1)
    candidate_valid = torch.cat((better_valid, tie_valid), dim=1)
    order = torch.argsort(candidate_indices.to(torch.float32), dim=1, stable=True)
    candidate_indices = candidate_indices.gather(1, order)
    candidate_values = candidate_values.gather(1, order)
    candidate_valid = candidate_valid.gather(1, order)
    order = torch.argsort(candidate_valid.to(torch.float32), dim=1, descending=True, stable=True)
    candidate_indices = candidate_indices.gather(1, order)
    candidate_values = candidate_values.gather(1, order)
    order = torch.argsort(candidate_values, dim=1, descending=True, stable=True)
    return candidate_indices.gather(1, order)[:, :k]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vendor-path", required=True)
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--visible-groups", type=int, default=1700)
    parser.add_argument("--table-blocks", type=int, default=2048)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if not torch_npu.npu.is_available() or "310" not in torch_npu.npu.get_device_name(0):
        raise RuntimeError("requires an Ascend 310P NPU")
    if min(args.tokens, args.visible_groups, args.table_blocks, args.block_size, args.repeats) <= 0:
        parser.error("all numeric arguments must be positive")
    if args.block_size % 4:
        parser.error("block-size must be divisible by four")

    groups_per_block = args.block_size // 4
    if args.visible_groups > args.table_blocks * groups_per_block:
        parser.error("visible groups exceed table capacity")
    os.environ["ASCEND_CUSTOM_OPP_PATH"] = args.vendor_path
    enable_custom_op()
    device = "npu:0"
    query = torch.randn((args.tokens, 4, 128), dtype=torch.float16, device=device)
    cache = torch.randn((args.table_blocks, groups_per_block + 3, 128), dtype=torch.float16, device=device)
    block_table = torch.arange(args.table_blocks, dtype=torch.int32, device=device).unsqueeze(0)
    query_start_loc = torch.tensor([0, args.tokens], dtype=torch.int32, device=device)
    positions = torch.full((args.tokens,), args.visible_groups * 4 - 1, dtype=torch.int32, device=device)

    def run(max_visible_groups: int | None) -> torch.Tensor:
        selected = qsa_indexer_select_groups_310(
            query,
            cache,
            block_table,
            query_start_loc,
            positions,
            compress_ratio=4,
            token_topk=2048,
            max_visible_groups=max_visible_groups,
        )
        torch_npu.npu.synchronize()
        return selected.group_indices

    full_selection = run(None)
    bounded_selection = run(args.visible_groups)
    if not torch.equal(full_selection, bounded_selection):
        raise AssertionError("bounded selection differs from full-table selection")
    samples = {}
    for name, limit in (("full", None), ("bounded", args.visible_groups)):
        times = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            run(limit)
            times.append((time.perf_counter() - start) * 1000)
        samples[name] = times

    bounded_blocks = (args.visible_groups + groups_per_block - 1) // groups_per_block
    scores = torch.ops._C_ascend.npu_qsa_indexer_score_310(
        query, cache, block_table[:, :bounded_blocks].contiguous(), query_start_loc, positions, 4
    )
    bounded_table = block_table[:, :bounded_blocks].contiguous()
    matmul_scores = wide_matmul_scores(query, cache, bounded_table, positions)
    visible_scores = args.visible_groups
    max_score_error = (scores[:, :visible_scores] - matmul_scores[:, :visible_scores]).abs().max().item()
    topk_width = min(512, scores.shape[1])
    stable_selection = legacy_candidate_topk(scores, topk_width)
    argsort_selection = _stable_topk_indices(scores, topk_width)
    torch_npu.npu.synchronize()
    if not torch.equal(stable_selection, argsort_selection):
        raise AssertionError("stable top-k differs from full stable argsort")
    matmul_selection = _stable_topk_indices(matmul_scores, topk_width)
    for name, fn in (
        (
            "native_score",
            lambda: torch.ops._C_ascend.npu_qsa_indexer_score_310(
                query, cache, bounded_table, query_start_loc, positions, 4
            ),
        ),
        ("wide_matmul_score", lambda: wide_matmul_scores(query, cache, bounded_table, positions)),
    ):
        fn()
        torch_npu.npu.synchronize()
        times = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            fn()
            torch_npu.npu.synchronize()
            times.append((time.perf_counter() - start) * 1000)
        samples[name] = times
        # Score and stable selection are invoked together in every QSA layer.
        # Measure the combined path so a faster score kernel that merely
        # shifts work into sorting cannot appear to be a decode win.
        _stable_topk_indices(fn(), topk_width)
        torch_npu.npu.synchronize()
        combined_times = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            _stable_topk_indices(fn(), topk_width)
            torch_npu.npu.synchronize()
            combined_times.append((time.perf_counter() - start) * 1000)
        samples[f"{name}_and_stable_topk"] = combined_times
    for name, fn in (("candidate_topk_legacy", legacy_candidate_topk), ("stable_argsort", _stable_topk_indices)):
        times = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            fn(scores, topk_width)
            torch_npu.npu.synchronize()
            times.append((time.perf_counter() - start) * 1000)
        samples[name] = times
    sort_sweep = {}
    for width in (512, 1712, 4096, 8192, 32768):
        sweep_scores = torch.randn((args.tokens, width), dtype=torch.float32, device=device)
        sort_sweep[width] = {}
        for name, fn in (("candidate_topk_legacy", legacy_candidate_topk), ("stable_argsort", _stable_topk_indices)):
            fn(sweep_scores, min(512, width))
            torch_npu.npu.synchronize()
            times = []
            for _ in range(args.repeats):
                start = time.perf_counter()
                fn(sweep_scores, min(512, width))
                torch_npu.npu.synchronize()
                times.append((time.perf_counter() - start) * 1000)
            sort_sweep[width][name] = times
    print(
        json.dumps(
            {
                "tokens": args.tokens,
                "visible_groups": args.visible_groups,
                "table_blocks": args.table_blocks,
                "bounded_blocks": bounded_blocks,
                "wide_matmul_selection_equal": torch.equal(matmul_selection, argsort_selection),
                "wide_matmul_max_score_error": max_score_error,
                "milliseconds": samples,
                "sort_sweep_ms": sort_sweep,
            }
        )
    )


if __name__ == "__main__":
    main()
