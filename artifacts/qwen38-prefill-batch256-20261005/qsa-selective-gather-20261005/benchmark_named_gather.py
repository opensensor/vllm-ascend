# SPDX-License-Identifier: Apache-2.0
"""Compare named gathers and host-selected QSA tiles without loading model weights."""

import argparse
import hashlib
import importlib
import json
import statistics
import time
from functools import partial
from pathlib import Path

import torch
import torch_npu

from vllm_ascend.models.qwen4_exp.ops.qsa_indexer import QSAGroupSelection
from vllm_ascend.utils import enable_custom_op

BLOCK_SIZE = 64
HEAD_DIM = 256
KV_HEADS = 2
QUERY_HEADS = 24
SELECTED_GROUPS = 512
COMPRESS_RATIO = 4
TILE_QUERIES = 64
PREFILL_QUERIES = 2560
CACHE_BLOCKS = 16800
TABLE_CAPACITY = 4096
WALL_REPEATS = 12
EVENT_REPEATS = 6
EVENT_ITERATIONS = 10
ATTENTION_REPEATS = 5
MINIMUM_EVENT_GAIN = 0.05


def load_ops(binding: Path):
    enable_custom_op()
    torch.ops.load_library(str(binding))
    return torch.ops._C_ascend.qsa_gather_value_nz_310, torch.ops._C_ascend.qsa_gather_value_nz_zero_310


def gather(op, cache, selection, table, *, head_dim, transpose_output=False, output=None):
    heads = cache.shape[1] // (head_dim // 16)
    tokens = selection.group_indices.shape[0]
    width = ((selection.group_indices.shape[1] * COMPRESS_RATIO + COMPRESS_RATIO + 15) // 16) * 16
    if output is None:
        shape = (tokens, heads, head_dim, width) if transpose_output else (tokens, heads, width, head_dim)
        output = torch_npu.empty_with_format(size=shape, dtype=cache.dtype, device=cache.device, acl_format=29)
    op(
        cache,
        selection.group_indices,
        selection.group_counts,
        selection.tail_starts,
        selection.tail_counts,
        table,
        output,
        heads,
        head_dim,
        transpose_output,
    )
    return output


def compare_calls(calls):
    wall = {name: [] for name in calls}
    events = {name: [] for name in calls}
    names = tuple(calls)
    for call in calls.values():
        call()
        call()
    torch_npu.npu.synchronize()
    for repeat in range(WALL_REPEATS):
        for name in names if repeat % 2 == 0 else names[::-1]:
            start = time.perf_counter()
            calls[name]()
            torch_npu.npu.synchronize()
            wall[name].append((time.perf_counter() - start) * 1000)
    for repeat in range(EVENT_REPEATS):
        for name in names if repeat % 2 == 0 else names[::-1]:
            start, end = torch_npu.npu.Event(enable_timing=True), torch_npu.npu.Event(enable_timing=True)
            calls[name]()
            calls[name]()
            start.record()
            for _ in range(EVENT_ITERATIONS):
                calls[name]()
            end.record()
            end.synchronize()
            events[name].append(start.elapsed_time(end) / EVENT_ITERATIONS)
    return {
        "wall_median_ms": {name: statistics.median(values) for name, values in wall.items()},
        "event_median_ms": {name: statistics.median(values) for name, values in events.items()},
        "wall_samples_ms": wall,
        "event_samples_ms": events,
    }


def attention_comparison(module, ops, key_cache, value_cache, table, offset, cutoff, parallel):
    positions = torch.arange(offset + 1, offset + PREFILL_QUERIES + 1, dtype=torch.int32)
    complete_groups = positions // COMPRESS_RATIO
    counts = complete_groups.clamp(max=SELECTED_GROUPS)
    host_counts = counts.tolist()
    groups = torch.arange(SELECTED_GROUPS).unsqueeze(0) + torch.arange(PREFILL_QUERIES).unsqueeze(1) * 127
    groups = (groups % complete_groups.clamp(min=1).unsqueeze(1)).to(dtype=torch.int32, device="npu:0")
    selection = QSAGroupSelection(
        groups,
        counts.to("npu:0"),
        (complete_groups * COMPRESS_RATIO).to("npu:0"),
        (positions % COMPRESS_RATIO).to("npu:0"),
    )
    query = torch.randn(PREFILL_QUERIES, QUERY_HEADS, HEAD_DIM, dtype=torch.float16, device="npu:0") * 0.1
    query_start_loc = torch.tensor([0, PREFILL_QUERIES], dtype=torch.int32, device="npu:0")
    streams = module.QSAPrefillGatherStreams().get() if parallel else None
    selected_tiles = sum(
        host_counts[min(start + TILE_QUERIES, PREFILL_QUERIES) - 1] <= cutoff
        for start in range(0, PREFILL_QUERIES, TILE_QUERIES)
    )
    baseline, zero = ops

    def selected(cache, selection, table, *, head_dim, transpose_output=False, output=None):
        end = selection.group_counts.storage_offset() + selection.group_counts.numel()
        op = zero if host_counts[end - 1] <= cutoff else baseline
        return gather(op, cache, selection, table, head_dim=head_dim, transpose_output=transpose_output, output=output)

    original = module.qsa_gather_key_transposed_nz_310, module.qsa_gather_value_nz_310
    # A dense chunk reuses the baseline callables directly. The experimental
    # tile policy consults host metadata only and never reads device counts.
    callables = {
        "baseline": (partial(gather, baseline, transpose_output=True), partial(gather, baseline)),
        "selective": (
            (partial(selected, transpose_output=True), selected)
            if selected_tiles
            else (partial(gather, baseline, transpose_output=True), partial(gather, baseline))
        ),
    }
    times = {name: [] for name in callables}
    outputs = {}
    try:
        for repeat in range(ATTENTION_REPEATS + 1):
            names = tuple(callables) if repeat % 2 == 0 else tuple(callables)[::-1]
            for name in names:
                module.qsa_gather_key_transposed_nz_310, module.qsa_gather_value_nz_310 = callables[name]
                start = time.perf_counter()
                outputs[name] = module.qsa_batched_prefill_310(
                    query,
                    key_cache,
                    value_cache,
                    selection,
                    table,
                    query_start_loc,
                    scale=HEAD_DIM**-0.5,
                    gather_streams=streams,
                )
                torch_npu.npu.synchronize()
                if repeat:
                    times[name].append((time.perf_counter() - start) * 1000)
        torch.testing.assert_close(outputs["baseline"], outputs["selective"], rtol=0, atol=0)
        assert torch.isfinite(outputs["selective"]).all()
    finally:
        module.qsa_gather_key_transposed_nz_310, module.qsa_gather_value_nz_310 = original
    return {
        "query_offset": offset,
        "query_tokens": PREFILL_QUERIES,
        "parallel_gather": parallel,
        "selected_zero_tiles": selected_tiles,
        "total_tiles": PREFILL_QUERIES // TILE_QUERIES,
        "exact_output_parity": True,
        "output_sha256": hashlib.sha256(outputs["selective"].cpu().numpy().tobytes()).hexdigest(),
        "wall_median_ms": {name: statistics.median(values) for name, values in times.items()},
        "wall_samples_ms": times,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch_npu.npu.is_available() or "310" not in torch_npu.npu.get_device_name(0):
        raise RuntimeError("requires an Ascend 310P NPU")
    ops = load_ops(args.binding)
    module = importlib.import_module("vllm_ascend.models.qwen4_exp.ops.qsa_batched_attention_310")
    torch.manual_seed(3106)
    shape = (CACHE_BLOCKS, KV_HEADS * HEAD_DIM // 16, BLOCK_SIZE, 16)
    key_cache = torch_npu.npu_format_cast(torch.randn(shape, dtype=torch.float16, device="npu:0") * 0.1, 29)
    value_cache = torch_npu.npu_format_cast(torch.randn(shape, dtype=torch.float16, device="npu:0") * 0.1, 29)
    visible_blocks = (PREFILL_QUERIES * 2 + BLOCK_SIZE - 1) // BLOCK_SIZE
    table = torch.zeros(1, TABLE_CAPACITY, dtype=torch.int32, device="npu:0")
    table[0, :visible_blocks] = torch.randperm(CACHE_BLOCKS, dtype=torch.int32, device="npu:0")[:visible_blocks]
    groups = torch.stack(
        [torch.randperm(PREFILL_QUERIES // COMPRESS_RATIO)[:SELECTED_GROUPS] for _ in range(TILE_QUERIES)]
    )
    groups = groups.to(dtype=torch.int32, device="npu:0")
    results = {}
    with torch.no_grad():
        for active in (128, 256, 384, 512):
            selection = QSAGroupSelection(
                groups,
                torch.full((TILE_QUERIES,), active, dtype=torch.int32, device="npu:0"),
                torch.zeros(TILE_QUERIES, dtype=torch.int32, device="npu:0"),
                torch.zeros(TILE_QUERIES, dtype=torch.int32, device="npu:0"),
            )
            results[str(active)] = {}
            for kind, cache, transpose in (("key", key_cache, True), ("value", value_cache, False)):
                outputs = {
                    name: gather(op, cache, selection, table, head_dim=HEAD_DIM, transpose_output=transpose)
                    for name, op in zip(("baseline", "zero"), ops)
                }
                calls = {
                    name: partial(
                        gather,
                        op,
                        cache,
                        selection,
                        table,
                        head_dim=HEAD_DIM,
                        transpose_output=transpose,
                        output=outputs[name],
                    )
                    for name, op in zip(("baseline", "zero"), ops)
                }
                timing = compare_calls(calls)
                cpu = {name: torch_npu.npu_format_cast(out, 0).cpu() for name, out in outputs.items()}
                if transpose:
                    cpu = {name: out.transpose(-1, -2) for name, out in cpu.items()}
                torch.testing.assert_close(
                    cpu["baseline"][:, :, : active * COMPRESS_RATIO],
                    cpu["zero"][:, :, : active * COMPRESS_RATIO],
                    rtol=0,
                    atol=0,
                )
                assert torch.isfinite(cpu["zero"]).all()
                masked_nonzero = torch.count_nonzero(
                    cpu["zero"][:, :, active * COMPRESS_RATIO : SELECTED_GROUPS * COMPRESS_RATIO]
                ).item()
                assert masked_nonzero == 0
                results[str(active)][kind] = {
                    **timing,
                    "exact_valid_parity": True,
                    "masked_nonzero_count": masked_nonzero,
                }
        cutoff = 0
        for active in (128, 256, 384):
            if all(
                results[str(active)][kind]["event_median_ms"]["zero"]
                <= results[str(active)][kind]["event_median_ms"]["baseline"] * (1 - MINIMUM_EVENT_GAIN)
                for kind in ("key", "value")
            ):
                cutoff = active
            else:
                break
        attention = [
            attention_comparison(module, ops, key_cache, value_cache, table, offset, cutoff, parallel)
            for offset in (0, PREFILL_QUERIES)
            for parallel in (False, True)
        ]
    result = {
        "kv_heads": KV_HEADS,
        "cache_blocks": CACHE_BLOCKS,
        "block_table_width": TABLE_CAPACITY,
        "tile_queries": TILE_QUERIES,
        "event_repeats": EVENT_REPEATS,
        "event_iterations": EVENT_ITERATIONS,
        "zero_cutoff_groups": cutoff,
        "gathers": results,
        "attention": attention,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
