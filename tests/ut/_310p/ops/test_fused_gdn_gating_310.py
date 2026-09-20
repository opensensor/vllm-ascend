import pytest
import torch

from vllm_ascend._310p.ops.fla.fused_gdn_gating import (
    fused_gdn_gating_pytorch,
    gdn_gating_constants,
)

# Both softplus branches must be exercised: values far above `threshold` take the
# linear tail, values far below take log1p(exp(.)).
_LINEAR_TAIL_INPUT = 40.0
_LOG1P_BRANCH_INPUT = -40.0


def _reference_gating(A_log, a, b, dt_bias, beta=1.0, threshold=20.0):
    """The pre-optimization expression, kept verbatim as the parity oracle."""
    compute_dtype = torch.float32
    A_log_f = A_log.to(compute_dtype)
    a_f = a.to(compute_dtype)
    b_f = b.to(compute_dtype)
    dt_bias_f = dt_bias.to(compute_dtype)

    batch = a.shape[0]
    A_log_expanded = A_log_f.unsqueeze(0).expand(batch, -1)
    dt_bias_expanded = dt_bias_f.unsqueeze(0).expand(batch, -1)

    x = a_f + dt_bias_expanded
    beta_x = beta * x
    softplus_x = torch.where(
        beta_x <= threshold,
        (1.0 / beta) * torch.log1p(torch.exp(beta_x)),
        x,
    )
    g = (-torch.exp(A_log_expanded) * softplus_x).unsqueeze(0)
    beta_output = torch.sigmoid(b_f).to(b.dtype).unsqueeze(0)
    return g, beta_output


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("num_tokens,num_heads", [(1, 48), (37, 8), (512, 16)])
@pytest.mark.parametrize("beta,threshold", [(1.0, 20.0), (0.7, 12.0)])
def test_fused_gdn_gating_matches_reference(dtype, num_tokens, num_heads, beta, threshold):
    torch.manual_seed(0)

    A_log = torch.randn(num_heads, dtype=dtype)
    dt_bias = torch.randn(num_heads, dtype=dtype)
    a = torch.randn(num_tokens, num_heads, dtype=dtype)
    b = torch.randn(num_tokens, num_heads, dtype=dtype)
    a[0] = _LINEAR_TAIL_INPUT
    a[-1] = _LOG1P_BRANCH_INPUT

    ref_g, ref_beta = _reference_gating(A_log, a, b, dt_bias, beta, threshold)
    g, beta_out = fused_gdn_gating_pytorch(A_log, a, b, dt_bias, beta, threshold)

    assert g.shape == (1, num_tokens, num_heads)
    assert g.dtype == torch.float32
    assert beta_out.dtype == dtype
    torch.testing.assert_close(g, ref_g, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(beta_out, ref_beta, rtol=0, atol=0)


def test_precomputed_constants_match_inline_path():
    """Passing cached constants must not change the result."""
    torch.manual_seed(0)
    num_tokens, num_heads = 5, 48

    A_log = torch.randn(num_heads, dtype=torch.float16)
    dt_bias = torch.randn(num_heads, dtype=torch.float16)
    a = torch.randn(num_tokens, num_heads, dtype=torch.float16)
    b = torch.randn(num_tokens, num_heads, dtype=torch.float16)

    inline_g, inline_beta = fused_gdn_gating_pytorch(A_log, a, b, dt_bias)
    constants = gdn_gating_constants(A_log, dt_bias)
    cached_g, cached_beta = fused_gdn_gating_pytorch(A_log, a, b, dt_bias, constants=constants)

    torch.testing.assert_close(cached_g, inline_g, rtol=0, atol=0)
    torch.testing.assert_close(cached_beta, inline_beta, rtol=0, atol=0)


def test_gdn_gating_constants_shapes_and_values():
    A_log = torch.tensor([0.0, 1.0, -2.0], dtype=torch.float16)
    dt_bias = torch.tensor([0.5, -1.5, 3.0], dtype=torch.float16)

    neg_exp_A_log, dt_bias_f32 = gdn_gating_constants(A_log, dt_bias)

    assert neg_exp_A_log.dtype == torch.float32
    assert dt_bias_f32.dtype == torch.float32
    torch.testing.assert_close(neg_exp_A_log, -torch.exp(A_log.float()), rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(dt_bias_f32, dt_bias.float(), rtol=0, atol=0)


def test_tiled_constants_broadcast_matches_untiled():
    """The kernel takes constants pre-tiled; every row must be the same."""
    from vllm_ascend._310p.ops.fla.fused_gdn_gating import (
        _gating_tile_rows,
        gdn_gating_tiled_constants,
    )

    num_heads = 12  # 48 GDN heads at TP4: the unaligned case the tiling exists for
    A_log = torch.randn(num_heads, dtype=torch.float16)
    dt_bias = torch.randn(num_heads, dtype=torch.float16)

    neg_exp, dt_f32 = gdn_gating_constants(A_log, dt_bias)
    neg_exp_tiled, dt_tiled = gdn_gating_tiled_constants(A_log, dt_bias)

    tile_rows = _gating_tile_rows(num_heads)
    assert neg_exp_tiled.shape == (tile_rows, num_heads)
    assert dt_tiled.shape == (tile_rows, num_heads)
    assert neg_exp_tiled.is_contiguous() and dt_tiled.is_contiguous()
    for row in range(tile_rows):
        torch.testing.assert_close(neg_exp_tiled[row], neg_exp, rtol=0, atol=0)
        torch.testing.assert_close(dt_tiled[row], dt_f32, rtol=0, atol=0)


def test_dispatch_falls_back_when_kernel_absent():
    """Without the custom op built, results must match the reference exactly."""
    from unittest.mock import patch

    import vllm_ascend._310p.ops.fla.fused_gdn_gating as mod

    torch.manual_seed(0)
    num_tokens, num_heads = 3, 12
    A_log = torch.randn(num_heads, dtype=torch.float16)
    dt_bias = torch.randn(num_heads, dtype=torch.float16)
    a = torch.randn(num_tokens, num_heads, dtype=torch.float16)
    b = torch.randn(num_tokens, num_heads, dtype=torch.float16)

    with patch.object(mod, "_gating_op", return_value=None):
        g, beta_out = mod.fused_gdn_gating_310(A_log, a, b, dt_bias)
    ref_g, ref_beta = mod.fused_gdn_gating_pytorch(A_log, a, b, dt_bias)

    torch.testing.assert_close(g, ref_g, rtol=0, atol=0)
    torch.testing.assert_close(beta_out, ref_beta, rtol=0, atol=0)


@pytest.mark.parametrize("num_tokens", [4, 8, 12, 32])
def test_kernel_dispatch_narrows_aligned_outputs(num_tokens):
    """Exactly-tiling shapes go to the op; its padded rows are narrowed away."""
    from unittest.mock import Mock, patch

    import vllm_ascend._310p.ops.fla.fused_gdn_gating as mod

    num_heads = 12  # TP4: tile_rows == 4, so these token counts tile exactly
    tile_rows = mod._gating_tile_rows(num_heads)
    assert num_tokens % tile_rows == 0
    padded_tokens = num_tokens + tile_rows  # op may over-allocate; caller narrows
    padded_g = torch.randn(padded_tokens, num_heads, dtype=torch.float32)
    padded_beta = torch.randn(padded_tokens, num_heads, dtype=torch.float16)
    op = Mock(return_value=(padded_g, padded_beta))
    A_log = torch.randn(num_heads, dtype=torch.float16)
    dt_bias = torch.randn(num_heads, dtype=torch.float16)
    a = torch.randn(num_tokens, num_heads, dtype=torch.float16)
    b = torch.randn(num_tokens, num_heads, dtype=torch.float16)
    tiled_constants = mod.gdn_gating_tiled_constants(A_log, dt_bias)

    with patch.object(mod, "_gating_op", return_value=op):
        g, beta_out = mod.fused_gdn_gating_310(A_log, a, b, dt_bias, tiled_constants=tiled_constants)

    assert g.shape == (1, num_tokens, num_heads)
    assert beta_out.shape == (1, num_tokens, num_heads)
    assert g.untyped_storage().data_ptr() == padded_g.untyped_storage().data_ptr()
    assert beta_out.untyped_storage().data_ptr() == padded_beta.untyped_storage().data_ptr()
    op.assert_called_once()


@pytest.mark.parametrize("num_tokens", [1, 2, 5, 10, 31, 37])
def test_ragged_shapes_bypass_the_kernel(num_tokens):
    """Shapes that do not tile exactly must use the op chain, not the kernel.

    Padding them to a whole tile was measured slower than the seven framework
    ops the kernel replaces (T=10, H=12: 117.7us padded vs 90.1us op-chain),
    because F.pad allocates, fills and copies. MTP k=4 with max_num_seqs=2
    produces exactly these shapes, so this path is the common one at decode.
    """
    from unittest.mock import Mock, patch

    import vllm_ascend._310p.ops.fla.fused_gdn_gating as mod

    num_heads = 12
    assert num_tokens % mod._gating_tile_rows(num_heads) != 0
    op = Mock()
    A_log = torch.randn(num_heads, dtype=torch.float16)
    dt_bias = torch.randn(num_heads, dtype=torch.float16)
    a = torch.randn(num_tokens, num_heads, dtype=torch.float16)
    b = torch.randn(num_tokens, num_heads, dtype=torch.float16)

    with patch.object(mod, "_gating_op", return_value=op):
        g, beta_out = mod.fused_gdn_gating_310(
            A_log,
            a,
            b,
            dt_bias,
            tiled_constants=mod.gdn_gating_tiled_constants(A_log, dt_bias),
        )
    ref_g, ref_beta = mod.fused_gdn_gating_pytorch(A_log, a, b, dt_bias)

    op.assert_not_called()
    torch.testing.assert_close(g, ref_g, rtol=0, atol=0)
    torch.testing.assert_close(beta_out, ref_beta, rtol=0, atol=0)


def test_stable_softplus_matches_guarded_form_at_model_defaults():
    """The kernel uses max(bx,0)+log1p(exp(-|bx|)); assert the identity holds.

    This is the CPU oracle for the AscendC kernel's math, including the linear
    tail where the reference switches branches.
    """
    x = torch.cat(
        [
            torch.linspace(-60.0, 60.0, 5000, dtype=torch.float64),
            torch.tensor([0.0, 19.999, 20.0, 20.001], dtype=torch.float64),
        ]
    )
    guarded = torch.where(x <= 20.0, torch.log1p(torch.exp(x)), x)
    stable = torch.clamp(x, min=0.0) + torch.log1p(torch.exp(-x.abs()))
    # Far below one fp16 ULP (9.77e-4), so invisible after the downstream cast.
    torch.testing.assert_close(stable, guarded, rtol=0, atol=1e-8)
