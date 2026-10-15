# SPDX-License-Identifier: Apache-2.0
"""Drained, reversible mixed-QSA admission for the retained four-NPU service."""

import inspect

MIN_PREFILL_TOKENS = 128
MIN_DECODE_GROUPS = 2048


def matmul_dispatch(query_lens, num_tokens, capacity, num_requests, query_start_loc_size, max_decode_tokens):
    """Let a long prefill justify GEMM even when accompanying decode is short."""
    if query_lens is None:
        return (
            capacity > 0
            and num_requests == 1
            and query_start_loc_size == 2
            and (
                num_tokens >= MIN_PREFILL_TOKENS or (num_tokens <= max_decode_tokens and capacity >= MIN_DECODE_GROUPS)
            )
        )
    lengths = tuple(int(value) for value in query_lens)
    if len(lengths) != num_requests or any(value < 0 for value in lengths) or sum(lengths) != num_tokens:
        raise ValueError("invalid packed query lengths")
    if num_tokens <= 0 or capacity <= 0 or query_start_loc_size != num_requests + 1:
        return False
    has_prefill = any(value >= MIN_PREFILL_TOKENS for value in lengths)
    return all(
        value >= MIN_PREFILL_TOKENS or (value <= max_decode_tokens and (has_prefill or capacity >= MIN_DECODE_GROUPS))
        for value in lengths
        if value
    )


def install(worker, mode, factory_source, *, check_only=False):
    # Executed only by the authenticated controller after draining all requests.
    if mode != "score":
        raise ValueError("only score dispatch is qualified; mixed attention failed the live gate")
    from tools.qwen4exp.resident_candidates import qsa_graph_dispatch
    from vllm_ascend.models.qwen4_exp import model
    from vllm_ascend.models.qwen4_exp.ops import qsa_indexer

    if getattr(worker, "_qfn_mixed_qsa", None) is not None:
        raise RuntimeError("restore the prior mixed QSA candidate before replacing it")
    if not getattr(worker, "_qfn_mixed_qsa_gate", {}).get("passed"):
        raise RuntimeError("private score gate required before install")
    namespace = dict(qsa_graph_dispatch.__dict__)
    exec(compile(factory_source, "<mixed-qsa-factory>", "exec"), namespace)
    baseline = namespace["build_qualified_forward"]()
    if model._QSAAttention.forward.__code__ != baseline.__code__:
        raise RuntimeError("live QSA forward differs from the qualified graph-preserving factory")
    if check_only:
        return dict(rank=worker.rank, passed=True, mode=mode, forward_preserved=True)
    state = dict(mode=mode, originals=[], replacements=[], failed=False)
    worker._qfn_mixed_qsa = state
    replacements = [(qsa_indexer, "_use_qsa_matmul_score_batch", matmul_dispatch)]
    for owner, name, value in replacements:
        state["originals"].append((owner, name, name in vars(owner), vars(owner).get(name)))
        setattr(owner, name, value)
        state["replacements"].append((owner, name, inspect.getattr_static(owner, name)))
    return dict(rank=worker.rank, passed=True, mode=mode)


def restore(worker):
    state = worker._qfn_mixed_qsa
    if any(inspect.getattr_static(owner, name) is not value for owner, name, value in state["replacements"]):
        raise RuntimeError("mixed QSA ownership changed; retain state for inspection")
    for owner, name, present, original in reversed(state["originals"]):
        if present:
            setattr(owner, name, original)
        else:
            delattr(owner, name)
    worker._qfn_mixed_qsa = None
    return dict(rank=worker.rank, passed=True, restored=state["mode"])


def private_gate(worker):
    """Compare packed score selection on private device buffers."""
    import torch

    from vllm_ascend.models.qwen4_exp.ops import qsa_indexer as indexer

    device = worker.model_runner.device
    original = indexer._use_qsa_matmul_score_batch
    generator = torch.Generator().manual_seed(241015)
    records = []
    with torch.inference_mode():
        for groups, lengths in (
            (480, (3, 1920)),
            (1024, (3, 256)),
            (2048, (256, 3)),
            (1024, (3, 129, 2)),
            (8192, (3, 512)),
        ):
            count = sum(lengths)
            pages = groups // 16
            cache = torch.randn(pages + 11, 19, 128, generator=generator).half().to(device)
            table = (
                torch.stack([torch.randperm(pages + 11, generator=generator)[:pages] for _ in lengths]).int().to(device)
            )
            query = torch.randn(count, 4, 128, generator=generator).half().to(device)
            positions = torch.cat(
                [torch.arange(groups * 4 - n, groups * 4) if n > 3 else torch.arange(61, 61 + n) for n in lengths]
            ).to(device=device, dtype=torch.int32)
            bounds = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32, device=device)
            kwargs = dict(
                compress_ratio=4,
                token_topk=2048,
                max_visible_groups=groups,
                max_matmul_decode_tokens=8,
                query_lens=lengths,
            )
            baseline = indexer.qsa_indexer_select_groups_310(query, cache, table, bounds, positions, **kwargs)
            try:
                indexer._use_qsa_matmul_score_batch = matmul_dispatch
                candidate = indexer.qsa_indexer_select_groups_310(query, cache, table, bounds, positions, **kwargs)
            finally:
                indexer._use_qsa_matmul_score_batch = original
            for field in ("group_counts", "tail_starts", "tail_counts"):
                torch.testing.assert_close(
                    getattr(candidate, field).cpu(), getattr(baseline, field).cpu(), rtol=0, atol=0
                )
            a, b = candidate.group_indices.cpu(), baseline.group_indices.cpu()
            counts = baseline.group_counts.cpu().tolist()
            mismatches = sum(
                not torch.equal(a[row, :n].sort().values, b[row, :n].sort().values) for row, n in enumerate(counts)
            )
            if mismatches:
                raise RuntimeError(f"selected group sets differ: {mismatches}/{count} rows")
            records.append(dict(groups=groups, lengths=lengths, selected_sets_exact=True))
    worker._qfn_mixed_qsa_gate = dict(passed=True, cases=records)
    return dict(rank=worker.rank, passed=True, cases=records)


def status(worker):
    state = getattr(worker, "_qfn_mixed_qsa", None)
    return dict(
        rank=worker.rank,
        passed=state is not None and not state["failed"],
        mode=state["mode"] if state else None,
        private_gate=getattr(worker, "_qfn_mixed_qsa_gate", None),
    )
