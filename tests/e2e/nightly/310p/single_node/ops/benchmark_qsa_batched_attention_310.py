# SPDX-License-Identifier: Apache-2.0
"""Experimental gather + batched-matmul QSA comparison for Ascend 310P.

This is a diagnostic benchmark, not a serving implementation. Run with the
custom operator's op_api/lib in LD_LIBRARY_PATH.
"""

import argparse
import json
import time

import torch
import torch_npu

from vllm_ascend.models.qwen4_exp.ops.qsa_batched_attention_310 import qsa_batched_prefill_310
from vllm_ascend.models.qwen4_exp.ops.qsa_gather_nz_310 import (
    qsa_gather_key_transposed_nz_310,
    qsa_gather_value_nz_310,
)
from vllm_ascend.models.qwen4_exp.ops.qsa_indexer import QSAGroupSelection
from vllm_ascend.models.qwen4_exp.ops.qsa_sparse_attention_310 import qsa_sparse_attention_310
from vllm_ascend.utils import enable_custom_op


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=8)
    parser.add_argument("--groups", type=int, default=512)
    parser.add_argument("--cache-tokens", type=int, default=8192)
    parser.add_argument("--visible-tokens", type=int)
    parser.add_argument("--qk-scale", type=float, default=0.1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--tile-sizes", type=int, nargs="+", default=[64])
    parser.add_argument("--profile-stages", action="store_true")
    parser.add_argument(
        "--sort-groups",
        action="store_true",
        help="Include the cost of sorting selected groups into cache order before QSA gather",
    )
    parser.add_argument(
        "--test-key-layouts",
        action="store_true",
        help="Compare explicit ND and NZ score operands with the current implicit transpose",
    )
    parser.add_argument(
        "--test-nz-keys",
        action="store_true",
        help="Experiment with direct NZ key gather and transposed NZ score matmul",
    )
    parser.add_argument(
        "--test-transposed-nz-keys",
        action="store_true",
        help="Benchmark key gather that writes the transposed NZ score operand",
    )
    args = parser.parse_args()
    visible_tokens = args.cache_tokens if args.visible_tokens is None else args.visible_tokens
    block_size = 64
    num_query_heads = 24
    num_kv_heads = 2
    head_dim = 256
    compress_ratio = 4
    if not torch_npu.npu.is_available() or "310" not in torch_npu.npu.get_device_name(0):
        raise RuntimeError("requires an Ascend 310P NPU")
    if min(args.tokens, args.groups, args.cache_tokens, visible_tokens, args.repeats, *args.tile_sizes) <= 0:
        parser.error("all numeric arguments must be positive")
    if args.cache_tokens % block_size or args.groups * compress_ratio > visible_tokens:
        parser.error("visible cache must contain all selected groups and total cache must use whole pages")
    if visible_tokens > args.cache_tokens:
        parser.error("visible tokens cannot exceed cache tokens")
    enable_custom_op()
    device = "npu:0"
    torch.manual_seed(943)
    num_blocks = args.cache_tokens // block_size
    cache_shape = (num_blocks, num_kv_heads * head_dim // 16, block_size, 16)
    query = torch.randn((args.tokens, num_query_heads, head_dim), dtype=torch.float16, device=device) * args.qk_scale
    key_cache = torch_npu.npu_format_cast(
        torch.randn(cache_shape, dtype=torch.float16, device=device) * args.qk_scale, 29
    )
    value_cache = torch_npu.npu_format_cast(torch.randn(cache_shape, dtype=torch.float16, device=device) * 0.1, 29)
    group_ids = torch.stack(
        [torch.randperm(visible_tokens // compress_ratio)[: args.groups] for _ in range(args.tokens)]
    ).to(dtype=torch.int32, device=device)
    selection = QSAGroupSelection(
        group_indices=group_ids,
        group_counts=torch.full((args.tokens,), args.groups, dtype=torch.int32, device=device),
        tail_starts=torch.zeros(args.tokens, dtype=torch.int32, device=device),
        tail_counts=torch.zeros(args.tokens, dtype=torch.int32, device=device),
    )
    block_table = torch.arange(num_blocks, dtype=torch.int32, device=device).unsqueeze(0)
    query_start_loc = torch.tensor([0, args.tokens], dtype=torch.int32, device=device)

    def native() -> torch.Tensor:
        return qsa_sparse_attention_310(query, key_cache, value_cache, selection, block_table, query_start_loc)

    def to_token_major(cache: torch.Tensor) -> torch.Tensor:
        return (
            torch_npu.npu_format_cast(cache, 0)
            .view(num_blocks, num_kv_heads, head_dim // 16, block_size, 16)
            .permute(0, 3, 1, 2, 4)
            .contiguous()
            .view(args.cache_tokens, num_kv_heads, head_dim)
        )

    token_ids = (
        group_ids.unsqueeze(-1) * compress_ratio + torch.arange(compress_ratio, dtype=torch.int32, device=device)
    ).reshape(args.tokens, args.groups * compress_ratio)
    physical_blocks = block_table[0, token_ids // block_size]
    slot_ids = (physical_blocks * block_size + token_ids % block_size).reshape(-1)

    def batched() -> torch.Tensor:
        key_rows = (
            to_token_major(key_cache)
            .index_select(0, slot_ids)
            .view(args.tokens, args.groups * compress_ratio, num_kv_heads, head_dim)
        )
        value_rows = (
            to_token_major(value_cache)
            .index_select(0, slot_ids)
            .view(args.tokens, args.groups * compress_ratio, num_kv_heads, head_dim)
        )
        query_rows = query.view(args.tokens, num_kv_heads, num_query_heads // num_kv_heads, head_dim)
        key_rows = key_rows.permute(0, 2, 3, 1)
        value_rows = value_rows.permute(0, 2, 1, 3)
        logits = torch.matmul(query_rows, key_rows)
        probabilities = torch.softmax(logits.float() * head_dim**-0.5, dim=-1).to(query.dtype)
        return torch.matmul(probabilities, value_rows).reshape_as(query)

    def serving_batched() -> torch.Tensor:
        return qsa_batched_prefill_310(
            query,
            key_cache,
            value_cache,
            selection,
            block_table,
            query_start_loc,
            scale=head_dim**-0.5,
        )

    def serving_sorted() -> torch.Tensor:
        sorted_selection = QSAGroupSelection(
            group_indices=torch.sort(selection.group_indices, dim=-1).values,
            group_counts=selection.group_counts,
            tail_starts=selection.tail_starts,
            tail_counts=selection.tail_counts,
        )
        return qsa_batched_prefill_310(
            query,
            key_cache,
            value_cache,
            sorted_selection,
            block_table,
            query_start_loc,
            scale=head_dim**-0.5,
        )

    def serving_bounded(query_tile: int) -> torch.Tensor:
        return qsa_batched_prefill_310(
            query,
            key_cache,
            value_cache,
            selection,
            block_table,
            query_start_loc,
            scale=head_dim**-0.5,
            visible_blocks=(visible_tokens + block_size - 1) // block_size,
            query_tile=query_tile,
        )

    def serving_nz_keys(*, transposed: bool) -> torch.Tensor:
        """Diagnostic: gather both K/V directly from paged NZ cache."""
        outputs = []
        heads_per_kv_head = num_query_heads // num_kv_heads
        group_ranks = torch.arange(args.groups, dtype=torch.int32, device=device)
        tail_offsets = torch.arange(compress_ratio, dtype=torch.int32, device=device)
        for start in range(0, args.tokens, 64):
            end = min(start + 64, args.tokens)
            tile_selection = QSAGroupSelection(
                group_indices=selection.group_indices[start:end],
                group_counts=selection.group_counts[start:end],
                tail_starts=selection.tail_starts[start:end],
                tail_counts=selection.tail_counts[start:end],
            )
            group_valid = group_ranks.unsqueeze(0) < tile_selection.group_counts.unsqueeze(1)
            group_valid = group_valid.unsqueeze(-1).expand(-1, -1, compress_ratio).reshape(end - start, -1)
            tail_valid = tail_offsets.unsqueeze(0) < tile_selection.tail_counts.unsqueeze(1)
            valid = torch.cat((group_valid, tail_valid), dim=1)
            selected_keys = (
                qsa_gather_key_transposed_nz_310(key_cache, tile_selection, block_table, head_dim=head_dim)
                if transposed
                else qsa_gather_value_nz_310(key_cache, tile_selection, block_table, head_dim=head_dim)
            )
            selected_values = qsa_gather_value_nz_310(value_cache, tile_selection, block_table, head_dim=head_dim)
            selected_tokens = selected_keys.shape[3] if transposed else selected_keys.shape[2]
            valid = torch.nn.functional.pad(valid, (0, selected_tokens - valid.shape[1]), value=False)
            tile_query = query[start:end].view(end - start, num_kv_heads, heads_per_kv_head, head_dim)
            logits = torch.matmul(tile_query, selected_keys if transposed else selected_keys.transpose(-1, -2))
            logits = (logits.float() * head_dim**-0.5).masked_fill(~valid[:, None, None, :], -torch.inf)
            probabilities = torch.softmax(logits, dim=-1).to(query.dtype)
            outputs.append(torch.matmul(probabilities, selected_values).reshape(end - start, num_query_heads, head_dim))
        return torch.cat(outputs, dim=0)

    def measure(fn) -> tuple[torch.Tensor, list[float]]:
        fn()
        torch_npu.npu.synchronize()
        times = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            result = fn()
            torch_npu.npu.synchronize()
            times.append((time.perf_counter() - start) * 1000)
        return result, times

    native_result, native_times = measure(native)
    batched_result, batched_times = measure(batched)
    serving_result, serving_times = measure(serving_batched)
    sorted_result, sorted_times = measure(serving_sorted) if args.sort_groups else (None, None)
    bounded_results = {tile: measure(lambda tile=tile: serving_bounded(tile)) for tile in args.tile_sizes}
    nz_key_result, nz_key_times = (
        measure(lambda: serving_nz_keys(transposed=False)) if args.test_nz_keys else (None, None)
    )
    transposed_nz_key_result, transposed_nz_key_times = (
        measure(lambda: serving_nz_keys(transposed=True)) if args.test_transposed_nz_keys else (None, None)
    )
    stage_times = {}
    if args.profile_stages:
        # Isolate the five expensive device stages for one serving-sized tile.
        # Report them separately from end-to-end timing: precomputed inputs
        # intentionally exclude launch and selection overhead.
        tile_tokens = min(args.tokens, 64)
        tile_slots = slot_ids[: tile_tokens * args.groups * compress_ratio]
        _, stage_times["cache_format_ms"] = measure(lambda: (to_token_major(key_cache), to_token_major(value_cache)))
        _, stage_times["key_cache_format_ms"] = measure(lambda: torch_npu.npu_format_cast(key_cache, 0))
        key_nd = torch_npu.npu_format_cast(key_cache, 0)

        def key_token_major() -> torch.Tensor:
            return (
                key_nd.view(num_blocks, num_kv_heads, head_dim // 16, block_size, 16)
                .permute(0, 3, 1, 2, 4)
                .contiguous()
                .view(args.cache_tokens, num_kv_heads, head_dim)
            )

        _, stage_times["key_token_major_ms"] = measure(key_token_major)
        key_rows = to_token_major(key_cache)
        value_rows = to_token_major(value_cache)
        torch_npu.npu.synchronize()
        _, stage_times["key_gather_ms"] = measure(lambda: key_rows.index_select(0, tile_slots))

        def gather_rows() -> tuple[torch.Tensor, torch.Tensor]:
            selected_keys = key_rows.index_select(0, tile_slots).view(
                tile_tokens, args.groups * compress_ratio, num_kv_heads, head_dim
            )
            selected_values = value_rows.index_select(0, tile_slots).view(
                tile_tokens, args.groups * compress_ratio, num_kv_heads, head_dim
            )
            return selected_keys, selected_values

        (selected_keys, selected_values), stage_times["gather_ms"] = measure(gather_rows)
        tile_query = query[:tile_tokens].view(tile_tokens, num_kv_heads, num_query_heads // num_kv_heads, head_dim)
        keys_transposed = selected_keys.permute(0, 2, 3, 1)
        values_transposed = selected_values.permute(0, 2, 1, 3)
        logits, stage_times["score_matmul_ms"] = measure(lambda: torch.matmul(tile_query, keys_transposed))
        if args.test_key_layouts:
            contiguous_keys, stage_times["key_transpose_contiguous_ms"] = measure(keys_transposed.contiguous)
            nz_score_keys, stage_times["key_transpose_nz_format_ms"] = measure(
                lambda: torch_npu.npu_format_cast(contiguous_keys, 29)
            )
            contiguous_logits, stage_times["score_contiguous_matmul_ms"] = measure(
                lambda: torch.matmul(tile_query, contiguous_keys)
            )
            nz_score_logits, stage_times["score_nz_matmul_ms"] = measure(
                lambda: torch.matmul(tile_query, nz_score_keys)
            )
            _, stage_times["score_contiguous_total_ms"] = measure(
                lambda: torch.matmul(tile_query, keys_transposed.contiguous())
            )
            _, stage_times["score_nz_total_ms"] = measure(
                lambda: torch.matmul(tile_query, torch_npu.npu_format_cast(keys_transposed.contiguous(), 29))
            )
            stage_times["score_contiguous_max_abs_difference"] = (
                (contiguous_logits.float() - logits.float()).abs().max().item()
            )
            stage_times["score_nz_max_abs_difference"] = (nz_score_logits.float() - logits.float()).abs().max().item()
        probabilities, stage_times["softmax_ms"] = measure(
            lambda: torch.softmax(logits.float() * head_dim**-0.5, dim=-1).to(query.dtype)
        )
        _, stage_times["value_matmul_ms"] = measure(lambda: torch.matmul(probabilities, values_transposed))
        nz_values, stage_times["value_nz_format_ms"] = measure(
            lambda: torch_npu.npu_format_cast(values_transposed.contiguous(), 29)
        )
        nz_result, stage_times["value_nz_matmul_ms"] = measure(lambda: torch.matmul(probabilities, nz_values))
        stage_times["value_nz_max_abs_difference"] = (
            (nz_result.float() - torch.matmul(probabilities, values_transposed).float()).abs().max().item()
        )
        if hasattr(torch.ops._C_ascend, "qsa_gather_value_nz_310"):
            tile_selection = QSAGroupSelection(
                group_indices=selection.group_indices[:tile_tokens],
                group_counts=selection.group_counts[:tile_tokens],
                tail_starts=selection.tail_starts[:tile_tokens],
                tail_counts=selection.tail_counts[:tile_tokens],
            )
            if args.test_nz_keys:
                gathered_nz_keys, stage_times["custom_key_nz_gather_ms"] = measure(
                    lambda: qsa_gather_value_nz_310(key_cache, tile_selection, block_table, head_dim=head_dim)
                )
                custom_key_logits, stage_times["custom_key_nz_matmul_ms"] = measure(
                    lambda: torch.matmul(tile_query, gathered_nz_keys.transpose(-1, -2))
                )
                stage_times["custom_key_nz_score_max_abs_difference"] = (
                    (custom_key_logits[..., : selected_keys.shape[1]].float() - logits.float()).abs().max().item()
                )
            if args.test_transposed_nz_keys:
                gathered_transposed_keys, stage_times["custom_key_transposed_nz_gather_ms"] = measure(
                    lambda: qsa_gather_key_transposed_nz_310(key_cache, tile_selection, block_table, head_dim=head_dim)
                )
                transposed_key_logits, stage_times["custom_key_transposed_nz_matmul_ms"] = measure(
                    lambda: torch.matmul(tile_query, gathered_transposed_keys)
                )
                stage_times["custom_key_transposed_nz_score_max_abs_difference"] = (
                    (transposed_key_logits[..., : selected_keys.shape[1]].float() - logits.float()).abs().max().item()
                )
            gathered_nz, stage_times["custom_value_nz_gather_ms"] = measure(
                lambda: qsa_gather_value_nz_310(value_cache, tile_selection, block_table, head_dim=head_dim)
            )
            gathered_nd = torch_npu.npu_format_cast(gathered_nz, 0)[:, :, : args.groups * compress_ratio]
            stage_times["custom_value_nz_max_abs_difference"] = (
                (gathered_nd.float() - values_transposed.float()).abs().max().item()
            )
            padded_probabilities = torch.nn.functional.pad(
                probabilities, (0, gathered_nz.shape[-2] - probabilities.shape[-1])
            )
            custom_result, stage_times["custom_value_nz_matmul_ms"] = measure(
                lambda: torch.matmul(padded_probabilities, gathered_nz)
            )
            stage_times["custom_value_nz_output_max_abs_difference"] = (
                (custom_result.float() - torch.matmul(probabilities, values_transposed).float()).abs().max().item()
            )
    difference = batched_result.float() - native_result.float()
    serving_difference = serving_result.float() - native_result.float()
    sorted_difference = sorted_result.float() - serving_result.float() if sorted_result is not None else None
    nz_key_difference = nz_key_result.float() - native_result.float() if nz_key_result is not None else None
    transposed_nz_key_difference = (
        transposed_nz_key_result.float() - native_result.float() if transposed_nz_key_result is not None else None
    )
    bounded_metrics = {}
    for tile, (bounded_result, bounded_times) in bounded_results.items():
        bounded_difference = bounded_result.float() - native_result.float()
        bounded_metrics[tile] = {
            "ms": bounded_times,
            "max_abs_difference": bounded_difference.abs().max().item(),
            "relative_rms_difference": (
                bounded_difference.square().mean().sqrt() / native_result.float().square().mean().sqrt()
            ).item(),
        }
    print(
        json.dumps(
            {
                "tokens": args.tokens,
                "groups": args.groups,
                "cache_tokens": args.cache_tokens,
                "visible_tokens": visible_tokens,
                "qk_scale": args.qk_scale,
                "native_ms": native_times,
                "batched_ms": batched_times,
                "serving_batched_ms": serving_times,
                "sorted_groups_ms": sorted_times,
                "sorted_groups_max_abs_difference": sorted_difference.abs().max().item()
                if sorted_difference is not None
                else None,
                "serving_max_abs_difference": serving_difference.abs().max().item(),
                "serving_relative_rms_difference": (
                    serving_difference.square().mean().sqrt() / native_result.float().square().mean().sqrt()
                ).item(),
                "nz_key_ms": nz_key_times,
                "nz_key_max_abs_difference": nz_key_difference.abs().max().item()
                if nz_key_difference is not None
                else None,
                "nz_key_relative_rms_difference": (
                    nz_key_difference.square().mean().sqrt() / native_result.float().square().mean().sqrt()
                ).item()
                if nz_key_difference is not None
                else None,
                "transposed_nz_key_ms": transposed_nz_key_times,
                "transposed_nz_key_max_abs_difference": transposed_nz_key_difference.abs().max().item()
                if transposed_nz_key_difference is not None
                else None,
                "transposed_nz_key_relative_rms_difference": (
                    transposed_nz_key_difference.square().mean().sqrt() / native_result.float().square().mean().sqrt()
                ).item()
                if transposed_nz_key_difference is not None
                else None,
                "bounded_by_tile": bounded_metrics,
                "stage_profile": stage_times,
                "max_abs_difference": difference.abs().max().item(),
                "relative_rms_difference": (
                    difference.square().mean().sqrt() / native_result.float().square().mean().sqrt()
                ).item(),
            }
        )
    )


if __name__ == "__main__":
    main()
