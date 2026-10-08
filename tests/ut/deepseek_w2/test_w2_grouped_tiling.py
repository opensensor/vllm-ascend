# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU control-flow coverage for prefill chunks above the grouped route cap.

Run with ``--noconftest``; the sibling method test installs the host stubs.
"""

import types

import pytest
import torch

from tests.ut.deepseek_w2.test_w2_method import (  # noqa: F401
    W2_CUBE_MIN_INPUT_DIM,
    W2_GROUPED_MAX_ROUTES,
    AscendW2DynamicFusedMoEMethod310,
)
from vllm_ascend._310p.quantization.methods import w2_dynamic


def _grouped_bank():
    hidden = inter = W2_CUBE_MIN_INPUT_DIM

    class GroupedBank(list):
        grouped_ready = True
        nz_packed_codes = True
        local_expert_offset = 2
        num_local_experts = 2

    bank = GroupedBank([types.SimpleNamespace(hidden=hidden, inter=inter) for _ in range(4)])
    bank.gate_packed_bank = torch.zeros(2, inter, hidden // 2, dtype=torch.uint8)
    bank.up_packed_bank = torch.zeros_like(bank.gate_packed_bank)
    bank.down_packed_bank = torch.zeros(2, hidden, inter // 2, dtype=torch.uint8)
    bank.gate_scale_bank = torch.ones(2, inter // 32, hidden // 32)
    bank.up_scale_bank = torch.ones_like(bank.gate_scale_bank)
    bank.down_scale_bank = torch.ones(2, hidden // 32, inter // 32)
    return bank


@pytest.mark.parametrize(
    "num_tokens,route_cap", [(768, None), (769, None), (1537, None), (1280, 10240), (2560, 20480), (2561, 20480)]
)
def test_grouped_prefill_tiles_without_changing_routes_or_shared_expert(monkeypatch, num_tokens, route_cap):
    top_k = 8
    max_routes = W2_GROUPED_MAX_ROUTES if route_cap is None else route_cap
    max_chunk_tokens = max_routes // top_k
    bank = _grouped_bank()
    bank.grouped_max_routes = route_cap
    torch.manual_seed(124)
    x = torch.randn(num_tokens, W2_CUBE_MIN_INPUT_DIM)
    topk_ids = torch.tensor([[2, 3, 2, 3, 2, 3, 2, 3]]).expand(num_tokens, -1)
    topk_weights = torch.full((num_tokens, top_k), 1 / top_k)
    topk_weights[:, 1] = 0  # Mask a peer-owned route through the sentinel path.
    observed_rows = []

    def fake_grouped_op(inputs, codes, scales, group_ends):
        del scales, group_ends
        observed_rows.append(inputs.shape[0])
        return inputs[:, : codes.shape[1]].to(torch.float16)

    class SharedExpert:
        def __init__(self):
            self.calls = []

        def forward(self, inputs):
            self.calls.append(inputs.shape[0])
            return torch.full_like(inputs, 0.125)

    method = AscendW2DynamicFusedMoEMethod310()
    reference_shared = SharedExpert()
    reference = method._apply_device_grouped(fake_grouped_op, bank, x, topk_weights, topk_ids, reference_shared)
    observed_rows.clear()
    bank.prefill_route_histogram = True
    monkeypatch.setattr(w2_dynamic, "_w2_grouped_mm_op", lambda: fake_grouped_op)
    shared = SharedExpert()
    actual = method._apply_device(bank, x, topk_weights, topk_ids, shared)

    torch.testing.assert_close(actual, reference, atol=0, rtol=0)
    assert shared.calls == reference_shared.calls == [num_tokens]
    assert len(observed_rows) == 3 * ((num_tokens + max_chunk_tokens - 1) // max_chunk_tokens)
    assert max(observed_rows) <= max_routes


def test_prefill_histogram_opt_in_keeps_short_decode_on_comparison_counts(monkeypatch):
    bank = _grouped_bank()
    bank.prefill_route_histogram = True
    x = torch.ones((4, W2_CUBE_MIN_INPUT_DIM), dtype=torch.float16)
    ids = torch.tensor([[2, 3, 2, 3, 2, 3, 2, 3]]).expand(4, -1)
    weights = torch.full((4, 8), 1 / 8)

    def fake_grouped_op(inputs, codes, scales, group_ends):
        del scales, group_ends
        return torch.zeros((inputs.shape[0], codes.shape[1]), dtype=inputs.dtype)

    monkeypatch.setattr(torch, "histc", lambda *args, **kwargs: pytest.fail("decode must use comparison counts"))
    output = AscendW2DynamicFusedMoEMethod310()._apply_device_grouped(fake_grouped_op, bank, x, weights, ids, None)
    assert output.shape == x.shape
