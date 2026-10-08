# SPDX-License-Identifier: Apache-2.0
"""Math, padding and allocation contracts for kpool scratch reuse."""

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from vllm_ascend.models.glm5next import kpool_ops as ops


def reference_score(queries, weights, keys):
    rotated = ops.hadamard128(queries).bfloat16()
    logits = rotated.reshape(-1, 128).float() @ keys.float().T
    logits = logits.reshape(*queries.shape[:2], -1).relu_()
    return (logits * weights.float().unsqueeze(-1)).sum(1)


class AllocationAudit(TorchDispatchMode):
    def __init__(self):
        self.operations = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.operations.append((str(func), tuple(args[0].shape) if args and torch.is_tensor(args[0]) else None))
        return func(*args, **(kwargs or {}))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("rows", [1, 8, 17])
def test_score_reuses_scratch_without_changing_math_or_inputs(dtype, rows):
    generator = torch.Generator().manual_seed(71)
    queries = torch.randn(rows, 4, 128, generator=generator).to(dtype)
    # Negative head weights are intentional; multiply must follow ReLU.
    weights = torch.randn(rows, 4, generator=generator)
    keys = torch.randn(66, 128, generator=generator).to(dtype)[::2]
    inputs = [value.clone() for value in (queries, weights, keys)]
    expected = reference_score(queries, weights, keys)
    with AllocationAudit() as audit:
        actual = ops.score_kpool(queries, weights, keys)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for value, saved in zip((queries, weights, keys), inputs, strict=True):
        torch.testing.assert_close(value, saved, rtol=0, atol=0)
    assert ("aten.mul_.Tensor", (rows, 4, 33)) in audit.operations
    assert ("aten.mul.Tensor", (rows, 4, 33)) not in audit.operations


def test_query_chunks_convert_key_bank_once(monkeypatch):
    generator = torch.Generator().manual_seed(7)
    q = torch.randn(7, 4, 128, generator=generator).half()
    w = torch.randn(7, 4, generator=generator)
    keys = torch.randn(17, 128, generator=generator).half()
    positions = torch.arange(60, 67)
    monkeypatch.setattr(ops, "MAX_KPOOL_SCORE_ELEMENTS", 2 * 4 * 17)
    expected = []
    for start in range(0, 7, 2):
        scores = reference_score(q[start : start + 2], w[start : start + 2], keys)
        ids, _, tail, counts = ops.select_kpool_groups(scores, positions[start : start + 2], 16, 4)
        expected.append(ops.expand_kpool_groups(ids, tail, counts, 4))
    with AllocationAudit() as audit:
        actual = ops.score_and_select_kpool_tokens(q, w, keys, positions, 16, 4)
    torch.testing.assert_close(actual, torch.cat(expected), rtol=0, atol=0)
    assert audit.operations.count(("aten._to_copy.default", (17, 128))) == 1


@pytest.mark.parametrize("width", [0, 4, 17, 77760])
def test_causal_scores_preserve_order_ties_and_padding(width):
    positions = torch.tensor([0, 7, 15, 31, 65])
    # Exact ties are intentional, with different completed counts per row.
    logits = torch.zeros(5, width)
    valid = torch.arange(width)[None, :] < ((positions + 1) // 4)[:, None]
    causal = logits.masked_fill(~valid, -torch.inf)
    expected = ops.select_kpool_groups(logits, positions, 16, 4)
    with AllocationAudit() as audit:
        actual = ops.select_kpool_groups(causal, positions, 16, 4, scores_are_causal=True)
    for left, right in zip(actual, expected, strict=True):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    assert ("aten.masked_fill.Scalar", (5, width)) not in audit.operations


def test_causal_path_keeps_rank_and_index_padding_checks(monkeypatch):
    def garbage_topk(*args, **kwargs):
        class Result:
            indices = torch.tensor([[1, 0, 0, -7], [0, 2, 999, 0]])

        return Result()

    monkeypatch.setattr(torch, "topk", garbage_topk)
    ids, _, _, _ = ops.select_kpool_groups(torch.zeros(2, 32), torch.tensor([7, 3]), 16, 4, scores_are_causal=True)
    assert ids.tolist() == [[1, 0, -1, -1], [0, -1, -1, -1]]
