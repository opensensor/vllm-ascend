# SPDX-License-Identifier: Apache-2.0
"""Logical attention and route tests runnable without loading torch_npu."""

import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[4]
SPEC = importlib.util.spec_from_file_location("noncausal_mask", ROOT / "vllm_ascend/_310p/attention/noncausal_mask.py")
mask_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mask_module)


@pytest.mark.parametrize("window", [None, 1, 3, 2048])
@pytest.mark.parametrize("query_lengths,kv_lengths", [([3], [3]), ([2, 3], [6, 4]), ([1, 4], [9, 11])])
def test_noncausal_mask_and_attention_match_independent_reference(window, query_lengths, kv_lengths):
    mask = mask_module.noncausal_draft_mask(
        query_lengths, kv_lengths, device=torch.device("cpu"), sliding_window=window, max_key_length=32
    )
    generator = torch.Generator().manual_seed(12)
    row = 0
    for q_count, kv_count in zip(query_lengths, kv_lengths):
        query = torch.randn(q_count, 4, generator=generator)
        keys = torch.randn(max(kv_lengths), 4, generator=generator)
        values = torch.randn(max(kv_lengths), 5, generator=generator)
        actual = torch.softmax(query @ keys.T + mask[row : row + q_count], dim=-1) @ values
        for index in range(q_count):
            absolute_query = kv_count - q_count + index
            allowed = [k for k in range(kv_count) if window is None or abs(k - absolute_query) <= window - 1]
            assert mask[row + index].eq(0).nonzero().flatten().tolist() == allowed
            expected = torch.softmax(query[index] @ keys[allowed].T, dim=-1) @ values[allowed]
            torch.testing.assert_close(actual[index], expected)
        row += q_count


def test_window_preserves_prefix_and_future_queries():
    mask = mask_module.noncausal_draft_mask([3], [7], device=torch.device("cpu"), sliding_window=3, max_key_length=8)
    # Queries occupy logical positions 4,5,6, with a cached prefix at 0..3.
    assert mask.eq(0).tolist() == [
        [False, False, True, True, True, True, True],
        [False, False, False, True, True, True, True],
        [False, False, False, False, True, True, True],
    ]


@pytest.mark.parametrize(
    "queries,keys,window",
    [([], [], None), ([2], [1], None), ([0], [1], None), ([1], [33], None), ([1], [1, 2], None), ([1], [1], 0)],
)
def test_invalid_lengths_or_window_fail_closed(queries, keys, window):
    with pytest.raises(ValueError):
        mask_module.noncausal_draft_mask(
            queries, keys, device=torch.device("cpu"), sliding_window=window, max_key_length=32
        )


def load_method(name, namespace):
    path = ROOT / "vllm_ascend/_310p/attention/attention_v1.py"
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "AscendAttentionBackendImpl310")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


def test_causal_and_noncausal_chunked_routes_remain_distinct():
    calls = []
    namespace = {
        "torch": torch,
        "get_query_lens_cpu": lambda metadata: torch.tensor([2], dtype=torch.int32),
        "AttentionMaskBuilder310": SimpleNamespace(get_compressed_splitfuse_mask=lambda device: "causal-mask"),
        "MASK_TYPE_NORM_COMPRESS_PAGED_ATTENTION": 5,
        "torch_npu": SimpleNamespace(_npu_paged_attention_splitfuse_v2=lambda **kwargs: calls.append(kwargs)),
    }
    method = load_method("forward_chunked_prefill_310", namespace)
    impl = SimpleNamespace(
        support_compressed_mask=True,
        key_cache="key-cache",
        value_cache="value-cache",
        num_heads=2,
        num_kv_heads=1,
        scale=0.5,
        forward_noncausal_chunked_prefill_310=lambda *args: "noncausal-route",
    )
    metadata = SimpleNamespace(causal=False, num_actual_tokens=2, block_tables="blocks", seq_lens=torch.tensor([5]))
    query = torch.ones(2, 2, 4)
    output = torch.empty_like(query)
    assert method(impl, query, metadata, output) == "noncausal-route"
    assert not calls
    metadata.causal = True
    assert method(impl, query, metadata, output) is output
    assert len(calls) == 1 and calls[0]["mask"] == "causal-mask"


@pytest.mark.parametrize(
    "unsupported",
    [
        None,
        "capture",
        "sinks",
        "alibi",
        "cache_dtype",
        "token_count",
        "short_query",
        "short_output",
        "query_dtype",
        "output_dtype",
        "query_layout",
        "output_layout",
        "head_width",
        "block_rows",
        "block_dtype",
    ],
)
def test_noncausal_route_uses_explicit_mask_and_rejects_unsupported(monkeypatch, unsupported):
    calls = []
    monkeypatch.setattr(
        torch, "npu", SimpleNamespace(is_current_stream_capturing=lambda: unsupported == "capture"), raising=False
    )
    namespace = {
        "torch": torch,
        "get_query_lens_cpu": lambda metadata: torch.tensor([2, 1], dtype=torch.int32),
        "AttentionMaskBuilder310": SimpleNamespace(
            get_noncausal_splitfuse_mask=lambda q, k, device, window: mask_module.noncausal_draft_mask(
                q, k, device=device, sliding_window=window, max_key_length=16
            )
        ),
        "torch_npu": SimpleNamespace(_npu_paged_attention_splitfuse=lambda **kwargs: calls.append(kwargs)),
    }
    method = load_method("forward_noncausal_chunked_prefill_310", namespace)
    impl = SimpleNamespace(
        sinks=1 if unsupported == "sinks" else None,
        alibi_slopes=1 if unsupported == "alibi" else None,
        key_cache=torch.empty(1, dtype=torch.float32 if unsupported == "cache_dtype" else torch.float16),
        value_cache=torch.empty(1, dtype=torch.float16),
        sliding_window=3,
        head_size=4,
        num_heads=2,
        num_kv_heads=1,
        scale=0.5,
    )
    metadata = SimpleNamespace(
        num_actual_tokens=4 if unsupported == "token_count" else 3,
        seq_lens=torch.tensor([6, 4]),
        block_tables=torch.zeros(
            1 if unsupported == "block_rows" else 2,
            1,
            dtype=torch.int64 if unsupported == "block_dtype" else torch.int32,
        ),
    )
    query = torch.zeros(3, 2, 4, dtype=torch.float16)
    output = torch.zeros_like(query)
    if unsupported == "short_query":
        query = query[:2]
    elif unsupported == "short_output":
        output = output[:2]
    elif unsupported == "query_dtype":
        query = query.float()
    elif unsupported == "output_dtype":
        output = output.float()
    elif unsupported == "query_layout":
        query = torch.zeros(3, 2, 8, dtype=torch.float16)[:, :, ::2]
    elif unsupported == "output_layout":
        output = torch.zeros(3, 2, 8, dtype=torch.float16)[:, :, ::2]
    elif unsupported == "head_width":
        query = torch.zeros(3, 2, 5, dtype=torch.float16)
    if unsupported:
        with pytest.raises((NotImplementedError, ValueError)):
            method(impl, query, metadata, output)
        assert not calls
    else:
        assert method(impl, query, metadata, output) is output
        assert len(calls) == 1
        assert calls[0]["seq_len"].tolist() == [2, 1]
        assert calls[0]["context_lens"].tolist() == [6, 4]
        assert calls[0]["context_lens"].dtype == torch.int32
        assert calls[0]["context_lens"].is_contiguous()
        expected = mask_module.noncausal_draft_mask(
            [2, 1], [6, 4], device=query.device, sliding_window=3, max_key_length=16
        )
        torch.testing.assert_close(calls[0]["mask"], expected)
