# SPDX-License-Identifier: Apache-2.0
"""Offline contracts for the experimental tiled prefill integration."""

import ast
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tools.glm_perf import kpool_prefill as candidate
from vllm_ascend.models.glm5next import kpool_ops

ROOT = Path(__file__).resolve().parents[3]


def inputs(rows, pools):
    torch.manual_seed(310)
    block_rows = 160
    columns = (pools + block_rows - 1) // block_rows
    stride = block_rows * 144 + 256
    storage = torch.full((16 + (columns + 1) * stride,), torch.nan, dtype=torch.float16)
    cache = storage.as_strided((columns + 1, block_rows, 1, 128), (stride, 144, 128, 1), 16)
    cache[:-1].copy_(torch.randn(cache[:-1].shape).bfloat16().half())
    table = torch.arange(columns, dtype=torch.int32).flip(0).reshape(1, -1)
    q = torch.randn(rows, 32, 128).half()
    w = torch.randn(rows, 32)
    positions = torch.arange(pools * 4 - rows, pools * 4)
    return q, w, cache, table, positions


def reference_operator(q, weights, flat, table, ends, positions, pools, blocks, br, bs, rs, offset):
    cache = flat.as_strided((blocks, br, 128), (bs, rs, 1), offset)
    ids = torch.arange(pools)
    pages = table[0, ids // br].long()
    keys = cache[pages, ids % br].float()
    scores = (q.float().flatten(0, 1) @ keys.T).reshape(q.shape[0], 32, pools).relu()
    scores = (scores * weights[:, :, None]).sum(1)
    return scores.masked_fill(ids[None, :] >= ((positions + 1) // 4)[:, None], -torch.inf)


@pytest.mark.parametrize("rows,pools", [(1, 161), (3, 65), (4, 160), (5, 163), (127, 65), (128, 161)])
def test_native_padding_preserves_storage_and_valid_scores(monkeypatch, rows, pools):
    q, weights, cache, table, positions = inputs(rows, pools)
    calls = []

    def op(*args):
        calls.append(args)
        return reference_operator(*args)

    monkeypatch.setattr(torch.ops._C_ascend, "npu_glm_kpool_prefill_score_310", op, raising=False)
    actual = candidate.score_prefill_paged(q, weights, cache, table, positions, pools)
    args = calls[0]
    assert args[2].data_ptr() == cache.untyped_storage().data_ptr()
    assert args[0].shape[0] % 4 == 0
    assert args[6] % 8 == 0
    assert args[0].shape[0] <= 128
    assert (args[5][rows:] == -1).all()
    assert (args[0][rows:] == 0).all()
    torch.testing.assert_close(args[0][:rows], kpool_ops.hadamard128(q).bfloat16().half(), rtol=0, atol=0)
    ids = torch.arange(pools)
    keys = cache[table[0, ids // 160].long(), ids % 160, 0]
    expected = kpool_ops.score_kpool(q, weights, keys)
    expected.masked_fill_(ids[None, :] >= ((positions + 1) // 4)[:, None], -torch.inf)
    torch.testing.assert_close(actual, expected, rtol=3e-5, atol=1e-4)


@pytest.mark.parametrize("rows,budget", [(7, 3 * 32 * 165), (129, 1 << 26), (259, 1 << 26)])
@pytest.mark.parametrize("ties", [False, True])
def test_request_selection_keeps_baseline_topk_geometry(monkeypatch, rows, budget, ties):
    pools = 165
    q, weights, cache, table, positions = inputs(rows, pools)
    if ties:
        q.zero_()
    monkeypatch.setattr(torch.ops._C_ascend, "npu_glm_kpool_prefill_score_310", reference_operator, raising=False)
    monkeypatch.setattr(kpool_ops, "MAX_KPOOL_SCORE_ELEMENTS", budget)
    ids = torch.arange(pools)
    keys = cache[table[0, ids // 160].long(), ids % 160, 0]
    shapes = []
    original_topk = kpool_ops.topk_pool_indices

    def traced_topk(logits, count):
        shapes.append(tuple(logits.shape))
        return original_topk(logits, count)

    monkeypatch.setattr(kpool_ops, "topk_pool_indices", traced_topk)
    expected = kpool_ops.score_and_select_kpool_tokens(q, weights, keys, positions, 64, 4)
    baseline_shapes = shapes[:]
    shapes.clear()
    actual = candidate.select_prefill_request(q, weights, cache, table, positions, pools, 64, 4)
    assert shapes == baseline_shapes
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("problem", ["rows", "positions", "table", "short_table", "dtype", "stride"])
def test_reject_unsupported_geometry_before_native_dispatch(problem):
    q, w, cache, table, pos = inputs(5, 165)
    if problem == "rows":
        q, w, cache, table, pos = inputs(129, 165)
    elif problem == "positions":
        pos = pos[:-1]
    elif problem == "table":
        table = table.repeat(2, 1)
    elif problem == "short_table":
        table = table[:, :1]
    elif problem == "dtype":
        cache = cache.float()
    elif problem == "stride":
        cache = cache.transpose(1, 3)
    with pytest.raises(ValueError):
        candidate.score_prefill_paged(q, w, cache, table, pos, 165)


def test_native_geometry_on_cpu(tmp_path):
    source = tmp_path / "geometry.cpp"
    source.write_text(
        '#include "csrc/attention/glm_kpool_prefill_score_v310/score_geometry.h"\n'
        "#include <cassert>\n"
        "int main() { using NsGlmKpoolPrefillScore::ValidLayout;\n"
        "for(int n=1;n<=132;++n) {\n"
        "assert(ValidLayout(n,320,5,160,20736,128,16,103696)==(n<=128 && n%4==0)); }\n"
        "assert(!ValidLayout(4,319,5,160,20736,128,16,103696));\n"
        "assert(!ValidLayout(4,320,5,160,20736,128,16,103000));\n"
        "assert(!ValidLayout(4,320,5,160,128,128,16,103696));\n"
        "}\n"
    )
    binary = tmp_path / "geometry"
    subprocess.run(["c++", "-std=c++17", "-I", str(ROOT), str(source), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True)


def selector_methods():
    # Load the actual methods without initializing the NPU worker/import tree.
    tree = ast.parse((ROOT / "vllm_ascend/models/glm5next/sparse_attn_indexer_kpool.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "SparseAttnIndexerKpool")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_select_tokens")
    scope = dict(vars(kpool_ops), _cache_tensor=lambda cache: cache)
    exec(compile(ast.Module(body=[method], type_ignores=[]), "<baseline>", "exec"), scope)
    baseline = scope["_select_tokens"]
    tree = ast.parse((ROOT / "tools/glm_perf/resident_candidates/kpool_prefill_tiled.py").read_text())
    method = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "select_tokens")
    scope.update(
        BASELINE_SELECT=baseline,
        supports_prefill_score=candidate.supports_prefill_score,
        select_prefill_request=candidate.select_prefill_request,
    )
    exec(compile(ast.Module(body=[method], type_ignores=[]), "<candidate>", "exec"), scope)
    return baseline, scope["select_tokens"]


@pytest.mark.parametrize("fallback", [False, True])
def test_multiple_requests_empty_request_and_output_padding(monkeypatch, fallback):
    baseline, new = selector_methods()
    q, w, cache, table, pos = inputs(12, 165)
    if fallback:
        cache = cache.float()
    pos[5:] = torch.arange(25, 32)
    metadata = SimpleNamespace(
        num_actual_tokens=12,
        cum_query_lens_cpu=torch.tensor([0, 5, 5, 12]),
        seq_lens_cpu=torch.tensor([165, 0, 8]),
        block_table=table.repeat(3, 1),
    )
    owner = SimpleNamespace(k_cache=cache, topk_tokens=64, topk_indices_buffer=torch.full((16, 80), 123))
    monkeypatch.setattr(torch.ops._C_ascend, "npu_glm_kpool_prefill_score_310", reference_operator, raising=False)
    expected = baseline(owner, q, w, pos, 4, metadata).clone()
    owner.topk_indices_buffer.fill_(123)
    actual = new(owner, q, w, pos, 4, metadata)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert (actual[12:] == 123).all()
    assert (actual[:12, 67:] == -1).all()
