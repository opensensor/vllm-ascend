# SPDX-License-Identifier: Apache-2.0
import ast
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.models.glm5next.kpool_ops import expand_kpool_groups, hadamard128, select_kpool_groups
from vllm_ascend.models.glm5next.ops.kpool_native import score_kpool_paged, supports_live_kpool_score

ROOT = Path(__file__).resolve().parents[3]


def cache_view():
    backing = torch.zeros(16 + 5 * (160 * 128 + 256), dtype=torch.float16)
    return backing.as_strided((5, 160, 1, 128), (160 * 128 + 256, 128, 128, 1), 16)


def test_shared_cache_storage_reaches_operator_without_copy(monkeypatch):
    cache = cache_view()
    q = torch.randn(2, 32, 128).half()
    weights = torch.randn(2, 32)
    table = torch.tensor([[3, 1], [2, 0]], dtype=torch.int32)
    ends = torch.tensor([1, 2], dtype=torch.int32)
    positions = torch.tensor([511, 633], dtype=torch.int64)
    captured = []
    monkeypatch.setattr(
        torch.ops._C_ascend, "npu_glm_kpool_score_310", lambda *args: captured.append(args), raising=False
    )
    score_kpool_paged(q, weights, cache, table, ends, positions, 320)
    args = captured[0]
    assert args[2].data_ptr() == cache.untyped_storage().data_ptr()
    assert args[2].numel() == cache.untyped_storage().nbytes() // 2
    assert args[2].storage_offset() == 0
    assert args[6:] == (320, 5, 160, 160 * 128 + 256, 128, 16)
    assert args[5].dtype == torch.int32
    torch.testing.assert_close(args[0], hadamard128(q).bfloat16().half(), rtol=0, atol=0)


@pytest.mark.parametrize(
    "problem", ["pool", "rows", "heads", "dim", "dtype", "capacity", "zero", "offset", "transpose"]
)
def test_unsupported_layouts_fall_back(problem):
    q, cache, pool, capacity = torch.zeros(2, 32, 128), cache_view(), 4, 320
    if problem == "pool":
        pool = 8
    elif problem == "rows":
        q = torch.zeros(9, 32, 128)
    elif problem == "heads":
        q = torch.zeros(2, 16, 128)
    elif problem == "dim":
        q = torch.zeros(2, 32, 64)
    elif problem == "dtype":
        cache = cache.float()
    elif problem == "capacity":
        capacity = 319
    elif problem == "zero":
        capacity = 0
    elif problem == "offset":
        cache = cache.as_strided(cache.shape, cache.stride(), 1)
    else:
        cache = cache.transpose(1, 3)
    assert not supports_live_kpool_score(q, cache, pool, capacity)


def test_supported_strided_shared_pages():
    assert supports_live_kpool_score(torch.zeros(8, 32, 128), cache_view(), 4, 320)


def installer():
    source = (ROOT / "vllm_ascend/models/glm5next_w2/model.py").read_text()
    node = next(
        n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "_install_dsa_indexer"
    )
    scope = {"Iterable": list, "Any": object, "Glm5NextW2DtypePolicy": object, "_is_dsa_layer": lambda layer: True}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "<installer>", "exec"), scope)
    return scope["_install_dsa_indexer"]


@pytest.mark.parametrize("graph,enabled", [(False, False), (False, True), (True, False), (True, True)])
def test_opt_in_requires_graph_path(graph, enabled):
    op = SimpleNamespace()
    layer = SimpleNamespace(self_attn=SimpleNamespace(mla_attn=object(), indexer=SimpleNamespace(indexer_op=op)))
    config = SimpleNamespace(ascend_glm_mtp_full_graph=graph, ascend_glm_live_kpool_score=enabled)
    assert installer()([layer], config, None) == 1
    assert getattr(op, "live_kpool_score", False) == (graph and enabled)


def test_native_geometry_bounds(tmp_path):
    # Execute the host adapter's geometry contract without loading CANN/NPU.
    source = tmp_path / "geometry.cpp"
    source.write_text(
        '#include "csrc/attention/glm_kpool_score_v310/score_geometry.h"\n'
        "#include <cassert>\n"
        "int main() { using NsGlmKpoolScore::ValidLayout;\n"
        "assert(ValidLayout(8,320,5,160,20736,128,16,103696));\n"
        "assert(!ValidLayout(9,320,5,160,20736,128,16,103696));\n"
        "assert(!ValidLayout(8,319,5,160,20736,128,16,103696));\n"
        "assert(!ValidLayout(8,320,5,160,20736,128,16,103000));\n"
        "assert(!ValidLayout(8,320,5,160,128,128,16,103696));\n"
        "assert(!ValidLayout(8,320,5,160,20736,128,1,103696));\n"
        "assert(!ValidLayout(8,320,5,160,20736,127,16,103696));\n"
        "}\n"
    )
    binary = tmp_path / "geometry"
    subprocess.run(["c++", "-std=c++17", "-I", str(ROOT), str(source), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True)


def test_selector_native_dispatch_keeps_per_row_selection_and_clears_padding():
    source = (ROOT / "vllm_ascend/models/glm5next/sparse_attn_indexer_kpool.py").read_text()
    cls = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == "SparseAttnIndexerKpool")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_select_tokens_fixed")
    seen = []
    scores = torch.randn(2, 800)

    def native(*args):
        seen.append(args)
        # The native producer guarantees -inf outside each row's live pools.
        complete = (args[5] + 1) // 4
        return scores.masked_fill(torch.arange(scores.shape[1])[None, :] >= complete[:, None], -torch.inf)

    scope = dict(
        torch=torch,
        _cache_tensor=lambda layer: layer,
        supports_live_kpool_score=supports_live_kpool_score,
        score_kpool_paged=native,
        select_kpool_groups=select_kpool_groups,
        expand_kpool_groups=expand_kpool_groups,
    )
    exec(compile(ast.Module(body=[method], type_ignores=[]), "<selector>", "exec"), scope)
    cache = cache_view()
    output = torch.full((8, 2056), 99, dtype=torch.int32)
    owner = SimpleNamespace(
        k_cache=cache, max_model_len=3200, live_kpool_score=True, topk_tokens=2048, topk_indices_buffer=output
    )
    queries, weights = torch.randn(2, 32, 128), torch.randn(2, 32)
    positions = torch.tensor([2051, 2063])
    metadata = SimpleNamespace(block_table=torch.zeros(2, 5, dtype=torch.int32), cum_query_lens=torch.tensor([1, 2]))
    result = scope["_select_tokens_fixed"](owner, queries, weights, positions, 4, metadata)
    assert result.data_ptr() == output.data_ptr()
    assert len(seen) == 1 and seen[0][-1] == 800
    for row in range(2):
        selected, _, starts, counts = select_kpool_groups(scores[row : row + 1], positions[row : row + 1], 2048, 4)
        expected = expand_kpool_groups(selected, starts, counts, 4)
        torch.testing.assert_close(
            result[row : row + 1, : expected.shape[1]], expected.to(result.dtype), rtol=0, atol=0
        )
        assert (result[row, expected.shape[1] :] == -1).all()
    assert (result[2:] == 99).all()


def test_topk_padding_cannot_alias_completed_pool(monkeypatch):
    # Hardware may return arbitrary indices for -inf padding: both negative
    # integers and duplicate in-range IDs have been observed at large bounds.
    invalid = torch.tensor([[2, 0, 1, 0, -(2**31), 1, -123, 2]])
    monkeypatch.setattr(torch, "topk", lambda *args, **kwargs: SimpleNamespace(indices=invalid))
    selected, count, _, _ = select_kpool_groups(torch.zeros(1, 16), torch.tensor([11]), 32, 4)
    torch.testing.assert_close(selected, torch.tensor([[2, 0, 1, -1, -1, -1, -1, -1]], dtype=torch.int32))
    assert count.tolist() == [3]
