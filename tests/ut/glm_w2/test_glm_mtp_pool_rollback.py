# SPDX-License-Identifier: Apache-2.0
"""Run the actual pool writer on CPU with only device import stubs."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
from torch import nn

from vllm_ascend.models.glm5next.kpool_ops import compress_kpool
from vllm_ascend.models.glm5next.ops.mtp_norm import mtp_eh_norm


@pytest.fixture
def writer(monkeypatch):
    class CustomOp(nn.Module):
        @classmethod
        def register(cls, name):
            return lambda subclass: subclass

    for name, attrs in {
        "torch_npu": {},
        "vllm.compilation.breakable_cudagraph": {"BreakableCUDAGraphCapture": object},
        "vllm.forward_context": {"get_forward_context": lambda: None},
        "vllm.model_executor.custom_op": {"CustomOp": CustomOp},
    }.items():
        module = ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/models/glm5next/sparse_attn_indexer_kpool.py"
    spec = importlib.util.spec_from_file_location("glm_mtp_pool_writer_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    instance = module.SparseAttnIndexerKpool.__new__(module.SparseAttnIndexerKpool)
    nn.Module.__init__(instance)
    instance.head_dim = 128
    instance.k_cache = SimpleNamespace(kv_cache=torch.zeros(8, 4, 1, 128, dtype=torch.float16))
    instance.tail_cache = SimpleNamespace(kv_cache=torch.zeros(8, 4, 256), num_speculative_tokens=2)
    return instance


def write(writer, keys, gates, start):
    count = keys.shape[0]
    positions = torch.arange(start, start + count, dtype=torch.int32)
    slots = torch.where((positions + 1) % 4 == 0, positions // 4, -1)
    writer._write_pools(
        keys,
        gates,
        torch.zeros(4, 128),
        positions,
        4,
        SimpleNamespace(
            slot_mapping=slots, cum_query_lens=torch.tensor([count]), raw_seq_lens=torch.tensor([start + count])
        ),
        SimpleNamespace(slot_mapping=positions),
    )


@pytest.mark.parametrize("accepted", [1, 2, 3])
def test_rejection_across_pool_boundary_matches_clean_sequence(writer, accepted):
    generator = torch.Generator().manual_seed(420)
    keys = torch.randn(9, 128, generator=generator).half()
    gates = torch.randn(9, 128, generator=generator)
    write(writer, keys[:2], gates[:2], 0)
    write(writer, keys[2:5], gates[2:5], 2)
    # Verify three tokens, then replace the rejected suffix with new tokens.
    end = 2 + accepted
    replacements = torch.full((8 - end, 128), 0.25, dtype=torch.float16)
    replacement_gates = torch.full((8 - end, 128), -0.125)
    write(writer, replacements, replacement_gates, end)
    actual_keys = torch.cat((keys[:end], replacements)).reshape(2, 4, 128)
    actual_gates = torch.cat((gates[:end], replacement_gates)).reshape(2, 4, 128)
    expected = compress_kpool(actual_keys, actual_gates, torch.zeros(4, 128)).half()
    torch.testing.assert_close(writer.k_cache.kv_cache[0, :2, 0], expected, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_mtp_norm_masks_position_zero_and_preserves_recycled_hidden(dtype):
    embeddings = torch.tensor([[float("nan"), float("inf")], [2, 3]], dtype=dtype)
    hidden = torch.tensor([[1, 2], [4, 5]], dtype=dtype)
    before = hidden.clone()
    positions = torch.tensor([0, 8])
    ew, hw = torch.tensor([2.0, 3.0]), torch.tensor([4.0, 5.0])
    result = mtp_eh_norm(positions, embeddings, hidden, ew, hw, 1e-6)
    expected_e = torch.tensor([[0.0, 0.0], [2.0, 3.0]])
    expected_e = expected_e / torch.sqrt(expected_e.square().mean(-1, keepdim=True) + 1e-6) * ew
    expected_h = hidden.float() / torch.sqrt(hidden.float().square().mean(-1, keepdim=True) + 1e-6) * hw
    torch.testing.assert_close(result, torch.cat((expected_e, expected_h), -1).to(dtype))
    assert torch.equal(hidden, before) and torch.isfinite(result).all()


@pytest.mark.parametrize("positions", [[3, 5, 7], [19, 20, 23]])
def test_fixed_selector_matches_host_selection_after_request_move(writer, positions):
    from vllm_ascend.models.glm5next.kpool_ops import score_and_select_kpool_tokens

    torch.manual_seed(41)
    cache = writer.k_cache.kv_cache
    cache.normal_()
    writer.max_model_len = 32
    writer.topk_tokens = 8
    writer.topk_indices_buffer = torch.empty(3, 12, dtype=torch.int32)
    query = torch.randn(3, 2, 128).half()
    weights = torch.rand(3, 2)
    positions = torch.tensor(positions)
    # Different request pages and changing query boundaries exercise replay inputs.
    for ends, table in [([1, 3], [[2, 3], [0, 1]]), ([2, 3], [[0, 1], [2, 3]])]:
        metadata = SimpleNamespace(cum_query_lens=torch.tensor(ends), block_table=torch.tensor(table))
        actual = writer._select_tokens_fixed(query, weights, positions, 4, metadata).clone()
        start = 0
        for req, end in enumerate(ends):
            keys = cache[metadata.block_table[req]].reshape(-1, 128)
            reference = score_and_select_kpool_tokens(
                query[start:end], weights[start:end], keys, positions[start:end], 8, 4
            )
            # Dense pool order is immaterial to attention; compare selected sets.
            torch.testing.assert_close(
                actual[start:end, :11].sort(-1).values, reference.sort(-1).values, rtol=0, atol=0
            )
            start = end
