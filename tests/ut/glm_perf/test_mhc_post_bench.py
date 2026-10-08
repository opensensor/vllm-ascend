# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from tools.glm_perf.benchmark_mhc_post_310 import reference_post, streaming_post


@pytest.mark.parametrize("leading", [(1,), (4,), (2, 3)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("fused_add", [False, True])
def test_streaming_post_matches_mixing_axes_and_preserves_inputs(leading, dtype, fused_add):
    torch.manual_seed(310)
    streams, hidden = 4, 32
    # Asymmetric mixes catch accidentally transposing the input/output streams.
    x = torch.randn(*leading, hidden).to(dtype)
    residual = torch.randn(*leading, streams, hidden).to(dtype)
    post = torch.randn(*leading, streams, 1)
    comb = torch.randn(*leading, streams, streams)
    inputs = (x, residual, post, comb)
    copies = [value.clone() for value in inputs]
    output = streaming_post(*inputs, fused_add=fused_add)
    assert output.shape == residual.shape and output.dtype == dtype
    torch.testing.assert_close(output, reference_post(*inputs), rtol=1e-3, atol=2e-6)
    for actual, before in zip(inputs, copies):
        assert torch.equal(actual, before)


def test_streaming_post_requires_a_residual_stream():
    with pytest.raises(ValueError, match="at least one"):
        streaming_post(torch.ones(1, 32), torch.empty(1, 0, 32), torch.empty(1, 0, 1), torch.empty(1, 0, 0))
