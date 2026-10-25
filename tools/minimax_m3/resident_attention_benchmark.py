# SPDX-License-Identifier: Apache-2.0
"""Bounded head-batched attention comparisons on private scratch caches."""

import statistics
import time

import torch

from tools.minimax_m3.attention_candidate import paged_sparse_attention_batched_heads
from vllm_ascend._310p.ops.minimax_m3 import paged_sparse_attention

MIN_EXPERIMENT_FREE_BYTES = 2 * 1024**3
ATTENTION_CASES = ((1, 128), (1, 2048), (1, 16384), (64, 128), (64, 2048), (64, 16384))


def benchmark_attention(device, iterations):
    free, _ = torch.npu.mem_get_info()
    if free < MIN_EXPERIMENT_FREE_BYTES:
        raise ValueError("experimental attention requires at least 2 GiB free device memory")
    generator = torch.Generator().manual_seed(71)
    records = []
    for queries, context in ATTENTION_CASES:
        block, dim, heads, query_heads = 128, 128, 2, 32
        pages = (context + block - 1) // block

        def random(*shape):
            return torch.randn(*shape, generator=generator).half().to(device)

        order = torch.randperm(pages, generator=generator).to(device)
        index_order = torch.randperm(pages, generator=generator).to(device)
        kc, vc, ic = random(pages, block, heads, dim), random(pages, block, heads, dim), random(pages, block, dim)
        q, k, v = random(queries, query_heads, dim), random(queries, heads, dim), random(queries, heads, dim)
        iq, ik = random(queries, heads, dim), random(queries, dim)
        # Include an unfilled final page and a partial prefill, plus forced ties.
        length = context - 3
        positions = torch.arange(length - queries, length, device=device)
        slots = order[positions // block] * block + positions % block
        islots = index_order[positions // block] * block + positions % block
        kc[order[-1], -3:] = torch.nan
        vc[order[-1], -3:] = torch.nan
        ic[index_order[-1], -3:] = torch.nan
        args = (q, k, v, iq, ik, kc, vc, ic)
        kwargs = dict(
            query_starts=(0, queries),
            seq_lens=torch.tensor([length], device=device),
            block_table=order[None],
            index_block_table=index_order[None],
            slot_mapping=slots,
            index_slot_mapping=islots,
            max_seq_len=length,
            topk_blocks=16,
            scale=dim**-0.5,
            init_blocks=1,
            local_blocks=1,
        )
        timings, reference, error = {}, None, None
        for name, function in (
            ("baseline", paged_sparse_attention),
            ("batched_heads", paged_sparse_attention_batched_heads),
        ):
            for _ in range(2):
                function(*args, **kwargs)
            torch.npu.synchronize()
            samples = []
            for _ in range(iterations):
                start = time.perf_counter()
                result = function(*args, **kwargs)
                torch.npu.synchronize()
                samples.append(time.perf_counter() - start)
            actual = result.cpu()
            timings[name] = {"median_seconds": statistics.median(samples), "samples": samples}
            if reference is None:
                reference = actual
            else:
                error = float((actual.float() - reference.float()).abs().max())
        records.append(
            {"queries": queries, "context": length, "timings": timings, "max_abs": error, "passed": error == 0.0}
        )
    return {
        "passed": all(row["passed"] for row in records),
        "cases": records,
        "scope": "Synthetic scratch caches; existing request caches are not used",
    }
