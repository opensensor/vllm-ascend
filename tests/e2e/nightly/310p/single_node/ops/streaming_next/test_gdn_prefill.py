# SPDX-License-Identifier: Apache-2.0
"""Every causal row and real TP4/TP6 GDN head geometry, including varlen state."""

from unittest.mock import patch

import pytest


@pytest.mark.parametrize("heads", [6, 9, 12])
@pytest.mark.parametrize("tokens", [64, 128, 2560])
@pytest.mark.parametrize("batch", [1, 2])
def test_gdn_output_every_causal_row(gdn_operators, heads, tokens, batch):
    torch = gdn_operators
    q = torch.full((batch, heads // 3, tokens, 128), 0.01, dtype=torch.float16, device="npu")
    v = torch.full((batch, heads, tokens, 128), 0.01, dtype=torch.float16, device="npu")
    state = torch.zeros(batch, heads, (tokens // 64) * 128, 128, dtype=torch.float16, device="npu")
    g = torch.zeros(batch, heads, tokens, dtype=torch.float32, device="npu")
    output = (
        torch.ops._C_ascend.chunk_fwd_o_vllm(
            q,
            q,
            v,
            state,
            128**-0.5,
            g=g,
            g_gamma=None,
            cu_seqlens=None,
            chunk_indices=None,
            chunk_size=64,
            transpose_state_layout=False,
        )
        .cpu()
        .float()
    )
    # Use the actual FP16 constant; check every head and row, not only row zero.
    constant = float(q.cpu()[0, 0, 0, 0])
    expected = ((torch.arange(tokens) % 64) + 1) * constant**3 * 128 * 128**-0.5
    expected = expected[None, None, :, None].expand_as(output)
    torch.testing.assert_close(output, expected, rtol=0, atol=1e-6)


@pytest.mark.parametrize("key_heads,value_heads", [(4, 12), (3, 9), (2, 6)])
@pytest.mark.parametrize("tokens", [128, 2560, 320])
def test_native_gdn_prefill_output_and_state_match_cpu_reference(gdn_operators, key_heads, value_heads, tokens):
    torch = gdn_operators
    # Lazy hardware imports follow explicit admission in the fixture.
    import torch_npu

    from vllm_ascend._310p.ops.fla.chunk_gated_delta_rule import (
        chunk_gated_delta_rule_310,
        chunk_gated_delta_rule_pytorch,
    )

    torch.manual_seed(4100 + key_heads + tokens)
    q = torch.randn(1, tokens, key_heads, 128).half() * 0.05
    k = torch.randn_like(q) * 0.05
    v = torch.randn(1, tokens, value_heads, 128).half() * 0.1
    g = torch.full((1, tokens, value_heads), -0.1, dtype=torch.float32)
    beta = torch.full((1, tokens, value_heads), 0.3, dtype=torch.float16)
    boundaries = torch.tensor([0, 128, 320], dtype=torch.int64) if tokens == 320 else None
    state = torch.randn(2 if tokens == 320 else 1, value_heads, 128, 128) * 0.01

    def rms(x, weight, eps):
        result = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps)
        return (result * weight.float()).to(x.dtype), None

    kwargs = dict(
        output_final_state=True,
        head_first=False,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=boundaries,
    )
    with torch.inference_mode(), patch.object(torch_npu, "npu_rms_norm", rms):
        reference, reference_state = chunk_gated_delta_rule_pytorch(q, k, v, g, beta, initial_state=state, **kwargs)
    with torch.inference_mode():
        output, final_state = chunk_gated_delta_rule_310(
            q.npu(),
            k.npu(),
            v.npu(),
            g.npu(),
            beta.npu(),
            initial_state=state.npu(),
            **kwargs,
        )
    for actual, expected in [(output.cpu().float(), reference.float()), (final_state.cpu(), reference_state)]:
        assert torch.isfinite(actual).all()
        relative_l2 = (actual - expected).norm() / expected.norm().clamp_min(1e-12)
        assert relative_l2 < 0.001
