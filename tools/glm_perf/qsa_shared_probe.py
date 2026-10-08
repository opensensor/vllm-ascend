# SPDX-License-Identifier: Apache-2.0
"""Complete QSA exact parent/replay gates with independent dense references."""

import gc
import statistics
import time

import torch


def fixture(device, dim, tokens, shared, sparse=False, requests=1, logical_heads=1, physical_heads=1):
    generator = torch.Generator().manual_seed(98700 + dim + tokens)
    cache_cpu = torch.randn(14, physical_heads * dim // 16, 640, 16, generator=generator).half() * 0.1
    value_cpu = cache_cpu if shared else torch.randn(cache_cpu.shape, generator=generator).half() * 0.1
    query_cpu = torch.randn(tokens, 16 * logical_heads, dim, generator=generator).half() * 0.1
    key = cache_cpu.to(device)
    value = key if shared else value_cpu.to(device)
    query = query_cpu.to(device)
    groups = (torch.arange(512, dtype=torch.int32) * 3).expand(tokens, -1).contiguous()
    counts = torch.full((tokens,), 512 if sparse else 35, dtype=torch.int32)
    tails = torch.full((tokens,), 8000 if sparse else 0, dtype=torch.int32)
    lengths = torch.full((tokens,), 3 if sparse else -1, dtype=torch.int32)
    blocks = torch.zeros(requests, 9720, dtype=torch.int32)
    blocks[:, :14] = torch.arange(14, dtype=torch.int32)
    starts = torch.tensor([i * tokens // requests for i in range(requests + 1)], dtype=torch.int32)
    metadata = tuple(t.to(device) for t in (groups, counts, tails, lengths, blocks, starts))
    return query, key, value, metadata, query_cpu, cache_cpu, value_cpu


def gate_case(
    parent,
    candidate,
    dim,
    tokens,
    shared,
    sparse=False,
    requests=1,
    logical_heads=1,
    physical_heads=1,
    production=None,
    profile=False,
):
    data = fixture(candidate.device, dim, tokens, shared, sparse, requests, logical_heads, physical_heads)
    query, key, value, metadata, query_cpu, cache_cpu, value_cpu = data
    scale = 256**-0.5

    def call(op, query=query, key=key, value=value, metadata=metadata):
        return op(query, key, value, *metadata, scale, 4, logical_heads)

    expected = call(parent)
    actual = call(candidate)
    torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=0, atol=0)
    if production is not None:
        torch.testing.assert_close(actual.cpu(), call(production).cpu(), rtol=0, atol=0)
    reference = not sparse
    if reference:
        keys = cache_cpu[0, : dim // 16, :35].permute(1, 0, 2).reshape(35, dim).float()
        values = value_cpu[0, : dim // 16, :35].permute(1, 0, 2).reshape(35, dim).float()
        scores = query_cpu[0, :16].float() @ keys.T * scale
        result = (scores.softmax(-1) @ values).half()
        torch.testing.assert_close(actual.cpu()[0, :16], result, rtol=5e-3, atol=3e-3)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        captured = call(candidate)
    try:
        for phase in range(3):
            query.fill_(0.01 * (phase + 1))
            key.mul_(0.97)
            if not shared:
                value.add_(0.0001)
            if sparse:
                metadata[0].add_(1)
                metadata[3].fill_(phase + 1)
            else:
                metadata[1].fill_((1, 64, 129)[phase])
            graph.replay()
            torch.testing.assert_close(captured.cpu(), call(parent).cpu(), rtol=0, atol=0)
            torch.testing.assert_close(captured.cpu(), call(candidate).cpu(), rtol=0, atol=0)
    finally:
        graph.reset()
        torch.npu.synchronize()
    timings = {}
    if profile:
        for name, op in (("parent", parent), ("shared", candidate)):
            samples = []
            for _ in range(5):
                torch.npu.synchronize()
                start = time.perf_counter()
                call(op)
                torch.npu.synchronize()
                samples.append((time.perf_counter() - start) * 1000)
            timings[name] = dict(samples_ms=samples, median_ms=statistics.median(samples))
    result = dict(
        dim=dim,
        tokens=tokens,
        shared=shared,
        sparse=sparse,
        requests=requests,
        logical_heads=logical_heads,
        physical_heads=physical_heads,
        exact_parent=True,
        exact_production=production is not None,
        independent_dense_reference=reference,
        changed_replays=3,
        profiles=timings,
    )
    del captured, expected, actual, data, query, key, value, metadata
    gc.collect()
    torch.npu.empty_cache()
    return result


def qualify(parent, candidate, production=None, full=True, on_case=None):
    cases = []
    geometries = (
        ((256, 1), (256, 17), (512, 1), (512, 2), (512, 17), (512, 64), (512, 640))
        if full
        else ((256, 1), (512, 2), (512, 17), (512, 640))
    )
    for dim, tokens in geometries:
        for shared in (False, True):
            for sparse in (False, True):
                row = gate_case(
                    parent,
                    candidate,
                    dim,
                    tokens,
                    shared,
                    sparse,
                    production=production,
                    profile=full and shared and dim == 512 and tokens in (1, 640),
                )
                cases.append(row)
                if on_case is not None:
                    on_case(row)
    for shared in (False, True):
        row = gate_case(
            parent, candidate, 512, 17, shared, requests=2, logical_heads=1, physical_heads=2, production=production
        )
        cases.append(row)
        if on_case is not None:
            on_case(row)
    return dict(passed=True, cases=cases, changed_replay=True, no_math_change=True)
