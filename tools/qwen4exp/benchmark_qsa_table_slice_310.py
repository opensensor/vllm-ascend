# SPDX-License-Identifier: Apache-2.0
"""Measure QSA NZ gather with full and visible-page block tables on one 310P."""

import argparse
import hashlib
import json
import statistics
import time

import torch
import torch_npu

from vllm_ascend.models.qwen4_exp.ops.qsa_batched_attention_310 import qsa_batched_prefill_310
from vllm_ascend.models.qwen4_exp.ops.qsa_gather_nz_310 import (
    qsa_gather_key_transposed_nz_310,
    qsa_gather_value_nz_310,
)
from vllm_ascend.models.qwen4_exp.ops.qsa_indexer import QSAGroupSelection
from vllm_ascend.utils import enable_custom_op

BLOCK_SIZE = 64
HEAD_DIM = 256
SELECTED_GROUPS = 512
COMPRESS_RATIO = 4
QUERY_TOKENS = 64
BLOCK_TABLE_CAPACITY = 4096
BLOCK_TABLE_ALIGNMENT = 8
QUEUED_ITERATIONS = 10
QUEUED_REPEATS = 8


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--visible-tokens", type=int, default=2560)
    parser.add_argument("--cache-blocks", type=int, default=4096)
    parser.add_argument("--active-groups", type=int, default=SELECTED_GROUPS)
    parser.add_argument("--repeats", type=int, default=24)
    args = parser.parse_args()
    if not torch_npu.npu.is_available() or "310" not in torch_npu.npu.get_device_name(0):
        raise RuntimeError("requires an Ascend 310P NPU")
    visible_blocks = (args.visible_tokens + BLOCK_SIZE - 1) // BLOCK_SIZE
    compact_width = ((visible_blocks + BLOCK_TABLE_ALIGNMENT - 1) // BLOCK_TABLE_ALIGNMENT) * BLOCK_TABLE_ALIGNMENT
    if (
        args.repeats < 1
        or visible_blocks < 1
        or compact_width > BLOCK_TABLE_CAPACITY
        or args.cache_blocks < visible_blocks
        or args.visible_tokens // COMPRESS_RATIO < SELECTED_GROUPS
        or args.active_groups < 1
        or args.active_groups > SELECTED_GROUPS
    ):
        parser.error("invalid visible tokens, cache blocks, or repeat count")

    enable_custom_op()
    torch.manual_seed(3105)
    device = "npu:0"
    cache_shape = (args.cache_blocks, HEAD_DIM // 16, BLOCK_SIZE, 16)
    key_cache = torch_npu.npu_format_cast(torch.randn(cache_shape, dtype=torch.float16, device=device), 29)
    value_cache = torch_npu.npu_format_cast(torch.randn(cache_shape, dtype=torch.float16, device=device), 29)
    visible_groups = args.visible_tokens // COMPRESS_RATIO
    groups = torch.stack(
        [torch.randperm(visible_groups, dtype=torch.int32)[:SELECTED_GROUPS] for _ in range(QUERY_TOKENS)]
    ).to(device)
    selection = QSAGroupSelection(
        group_indices=groups,
        group_counts=torch.full((QUERY_TOKENS,), args.active_groups, dtype=torch.int32, device=device),
        tail_starts=torch.zeros(QUERY_TOKENS, dtype=torch.int32, device=device),
        tail_counts=torch.zeros(QUERY_TOKENS, dtype=torch.int32, device=device),
    )
    full_table = torch.zeros((1, BLOCK_TABLE_CAPACITY), dtype=torch.int32, device=device)
    full_table[0, :visible_blocks] = torch.randperm(args.cache_blocks, dtype=torch.int32, device=device)[
        :visible_blocks
    ]
    compact_table = full_table[:, :compact_width].clone()
    query = torch.randn((QUERY_TOKENS, 24, HEAD_DIM), dtype=torch.float16, device=device) * 0.1
    query_start_loc = torch.tensor([0, QUERY_TOKENS], dtype=torch.int32, device=device)
    padded_tokens = ((SELECTED_GROUPS * COMPRESS_RATIO + COMPRESS_RATIO + 15) // 16) * 16
    outputs = {}
    for variant in ("full", "compact"):
        outputs[(variant, "key")] = torch_npu.empty_with_format(
            size=(QUERY_TOKENS, 1, HEAD_DIM, padded_tokens), dtype=torch.float16, device=device, acl_format=29
        )
        outputs[(variant, "value")] = torch_npu.empty_with_format(
            size=(QUERY_TOKENS, 1, padded_tokens, HEAD_DIM), dtype=torch.float16, device=device, acl_format=29
        )

    def run(variant: str, kind: str) -> None:
        table = full_table if variant == "full" else compact_table
        gather = qsa_gather_key_transposed_nz_310 if kind == "key" else qsa_gather_value_nz_310
        cache = key_cache if kind == "key" else value_cache
        gather(cache, selection, table, head_dim=HEAD_DIM, output=outputs[(variant, kind)])

    times = {(variant, kind): [] for variant in ("full", "compact") for kind in ("key", "value")}
    event_times = {kind: [] for kind in ("key", "value")}
    with torch.no_grad():
        for kind in ("key", "value"):
            for variant in ("full", "compact"):
                run(variant, kind)
            torch_npu.npu.synchronize()
            exact = torch.equal(outputs[("full", kind)], outputs[("compact", kind)])
            if not exact:
                raise AssertionError(f"{kind} gather changed values with compact block table")
            for repeat in range(args.repeats):
                order = ("full", "compact") if repeat % 2 == 0 else ("compact", "full")
                for variant in order:
                    start = time.perf_counter()
                    run(variant, kind)
                    torch_npu.npu.synchronize()
                    times[(variant, kind)].append((time.perf_counter() - start) * 1000)
            for _ in range(QUEUED_REPEATS):
                start_event = torch_npu.npu.Event(enable_timing=True)
                end_event = torch_npu.npu.Event(enable_timing=True)
                # A queued warmup keeps the first timed launch from waiting
                # for host dispatch after an idle device.
                run("full", kind)
                run("full", kind)
                start_event.record()
                for _ in range(QUEUED_ITERATIONS):
                    run("full", kind)
                end_event.record()
                end_event.synchronize()
                event_times[kind].append(start_event.elapsed_time(end_event) / QUEUED_ITERATIONS)
        attention = qsa_batched_prefill_310(
            query, key_cache, value_cache, selection, full_table, query_start_loc, scale=HEAD_DIM**-0.5
        )
        torch_npu.npu.synchronize()
        attention_sha256 = hashlib.sha256(attention.cpu().numpy().tobytes()).hexdigest()
        valid_tokens = args.active_groups * COMPRESS_RATIO
        valid_key_sha256 = hashlib.sha256(
            outputs[("full", "key")][:, :, :, :valid_tokens].contiguous().cpu().numpy().tobytes()
        ).hexdigest()
        valid_value_sha256 = hashlib.sha256(
            outputs[("full", "value")][:, :, :valid_tokens, :].contiguous().cpu().numpy().tobytes()
        ).hexdigest()
        masked_values = outputs[("full", "value")][:, :, valid_tokens : SELECTED_GROUPS * COMPRESS_RATIO]
        masked_keys = outputs[("full", "key")][:, :, :, valid_tokens : SELECTED_GROUPS * COMPRESS_RATIO]
        # Count on the host: a device float32 mean can round below one even
        # when every masked element is zero. This check is outside the timings.
        masked_value_nonzero_count = torch.count_nonzero(masked_values.cpu()).item()
        masked_key_nonzero_count = torch.count_nonzero(masked_keys.cpu()).item()
        masked_zero_fraction = 1 - masked_value_nonzero_count / masked_values.numel() if masked_values.numel() else None

    print(
        json.dumps(
            {
                "visible_tokens": args.visible_tokens,
                "cache_blocks": args.cache_blocks,
                "full_table_width": full_table.shape[1],
                "compact_table_width": compact_table.shape[1],
                "query_tokens": QUERY_TOKENS,
                "selected_groups": SELECTED_GROUPS,
                "active_groups": args.active_groups,
                "repeats": args.repeats,
                "exact_key_and_value_parity": True,
                "attention_sha256": attention_sha256,
                "valid_key_sha256": valid_key_sha256,
                "valid_value_sha256": valid_value_sha256,
                "masked_value_zero_fraction": masked_zero_fraction,
                "masked_value_nonzero_count": masked_value_nonzero_count,
                "masked_key_nonzero_count": masked_key_nonzero_count,
                "median_ms": {
                    f"{variant}_{kind}": statistics.median(values) for (variant, kind), values in times.items()
                },
                "samples_ms": {f"{variant}_{kind}": values for (variant, kind), values in times.items()},
                "queued_iterations": QUEUED_ITERATIONS,
                "queued_repeats": QUEUED_REPEATS,
                "event_median_ms": {kind: statistics.median(values) for kind, values in event_times.items()},
                "event_samples_ms": event_times,
            }
        )
    )


if __name__ == "__main__":
    main()
