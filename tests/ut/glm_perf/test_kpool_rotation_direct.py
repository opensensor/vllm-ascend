# SPDX-License-Identifier: Apache-2.0
"""Resident decode fusion must preserve cache aliasing and baseline fallback."""

import pytest
import torch

from tools.glm_perf.resident_candidates import kpool_rotation_direct as candidate
from vllm_ascend.models.glm5next.kpool_ops import hadamard128


@pytest.mark.parametrize("rows", [1, 2, 3, 4, 5, 6, 7, 8])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_native_rotation_keeps_cache_view(monkeypatch, rows, dtype):
    storage = torch.zeros(16 + 3 * 160 * 144, dtype=torch.float16)
    cache = storage.as_strided((3, 160, 1, 128), (160 * 144, 144, 128, 1), 16)
    query = torch.randn(rows, 32, 128).to(dtype)
    weights = torch.randn(rows, 32)
    table = torch.tensor([[2, 1, 0]], dtype=torch.int64)
    ends = torch.tensor([rows], dtype=torch.int64)
    positions = torch.arange(rows, dtype=torch.int64)
    calls = []

    def scorer(*args):
        calls.append(args)
        return args[0]

    def rotate(q):
        assert q is query
        return hadamard128(q).bfloat16().half()

    monkeypatch.setattr(torch.ops._C_ascend, "npu_glm_kpool_score_310", scorer, raising=False)
    score = next(iter(candidate.replacements({"rotation_v1": rotate}).values()))
    result = score(query, weights, cache, table, ends, positions, 480)
    args = calls[0]
    torch.testing.assert_close(result, hadamard128(query).bfloat16().half(), rtol=0, atol=0)
    assert args[2].data_ptr() == storage.data_ptr()
    assert args[2].numel() == storage.numel()
    assert args[-5:] == (3, 160, 160 * 144, 144, 16)
    assert all(t.dtype == torch.int32 for t in args[3:6])
    torch.testing.assert_close(args[3], table.int())
    torch.testing.assert_close(args[4], ends.int())
    torch.testing.assert_close(args[5], positions.int())


@pytest.mark.parametrize("kind", ["prefill", "dtype", "strided", "head_dim"])
def test_unsupported_queries_use_original_without_rotation(monkeypatch, kind):
    query = torch.zeros(2, 32, 128, dtype=torch.float16)
    if kind == "prefill":
        query = torch.zeros(9, 32, 128, dtype=torch.float16)
    elif kind == "dtype":
        query = query.float()
    elif kind == "strided":
        query = torch.zeros(2, 32, 256, dtype=torch.float16)[..., ::2]
    elif kind == "head_dim":
        query = query[..., :64]
    seen = []
    monkeypatch.setattr(candidate, "baseline_score", lambda *args: seen.append(args) or "baseline")

    def rotate(q):
        pytest.fail("unsupported query reached native rotation")

    score = next(iter(candidate.replacements({"rotation_v1": rotate}).values()))
    args = (query, None, None, None, None, None, 480)
    assert score(*args) == "baseline"
    assert seen[0] == args
