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
