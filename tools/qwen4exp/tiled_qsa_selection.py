# SPDX-License-Identifier: Apache-2.0
"""Bound the existing QSA selector's query-row workspace without changing policy."""

MAX_TILE_ELEMENTS = 1 << 22


def tiled_selector(original, max_tile_elements=MAX_TILE_ELEMENTS):
    if max_tile_elements < 1:
        raise ValueError("selection tile budget must be positive")

    def select(scores, k, policy):
        import torch

        rows = max(1, max_tile_elements // max(1, scores.shape[1]))
        if scores.shape[0] <= rows:
            return original(scores, k, policy)
        first = original(scores[:rows], k, policy)
        selected = torch.empty((scores.shape[0], k), dtype=first.dtype, device=scores.device)
        selected[:rows].copy_(first)
        del first
        for start in range(rows, scores.shape[0], rows):
            stop = min(start + rows, scores.shape[0])
            selected[start:stop].copy_(original(scores[start:stop], k, policy))
        return selected

    return select


def private_gate(worker):
    """Measure unchanged selected indices and workspace tiling on private data."""
    import statistics
    import time

    import torch

    from vllm_ascend.models.qwen4_exp.ops import qsa_indexer

    original = qsa_indexer._select_topk_indices
    candidate = tiled_selector(original)
    records = []
    generator = torch.Generator().manual_seed(241015)
    with torch.inference_mode():
        for columns in (5120, 8192):
            scores = torch.randn(1923, columns, generator=generator).to(worker.model_runner.device)
            policy = qsa_indexer.QSA_SELECTION_FAST_TOPK
            expected = original(scores, 512, policy)
            actual = candidate(scores, 512, policy)
            torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=0, atol=0)
            elapsed = {"original": [], "tiled": []}
            for iteration in range(4):
                variants = (("original", original), ("tiled", candidate))
                if iteration % 2:
                    variants = variants[::-1]
                for name, select in variants:
                    torch.npu.synchronize()
                    started = time.monotonic()
                    result = select(scores, 512, policy)
                    torch.npu.synchronize()
                    elapsed[name].append((time.monotonic() - started) * 1000)
                    del result
            records.append(
                dict(
                    columns=columns,
                    rows=1923,
                    exact=True,
                    milliseconds={name: statistics.median(values) for name, values in elapsed.items()},
                )
            )
            del scores, actual, expected
    return dict(rank=worker.rank, passed=True, cases=records)
