from unittest.mock import Mock

import pytest
import torch
from torch import nn
from torch.nn import functional

from tools.rtsp_scene.vision_norm import VISION_WIDTH, VisionZeroAddNorm


def normalize(x, zero, weight, bias, eps, additional_output):
    assert additional_output is True
    combined = x + zero
    return functional.layer_norm(combined, (VISION_WIDTH,), weight, bias, eps), None, None, combined


def test_supported_shapes_preserve_values_inputs_and_shared_storage():
    original = nn.LayerNorm(VISION_WIDTH, eps=1e-6).half()
    zero = torch.zeros(16, 1, VISION_WIDTH, dtype=torch.float16)
    native = Mock(side_effect=normalize)
    candidate = VisionZeroAddNorm(original, zero, native)
    other = VisionZeroAddNorm(original, zero, native)
    with torch.inference_mode():
        for rows, magnitude in ((1, 0.0001), (8, 0.1), (16, 3)):
            x = torch.randn(rows, 1, VISION_WIDTH).half() * magnitude
            saved = x.clone()
            assert torch.equal(candidate(x), original(x))
            assert torch.equal(saved, x)
            assert native.call_args.args[1].data_ptr() == zero.data_ptr()
            assert native.call_args.args[-1] is True
    assert candidate.zero.data_ptr() == other.zero.data_ptr()
    assert torch.count_nonzero(zero) == 0
    assert "zero" not in candidate.state_dict()


def test_unsupported_layouts_and_dtypes_delegate_without_native_calls():
    original = nn.LayerNorm(VISION_WIDTH).half()
    original.forward = Mock(side_effect=lambda x: x)
    native = Mock()
    candidate = VisionZeroAddNorm(original, torch.zeros(16, 1, VISION_WIDTH, dtype=torch.float16), native)
    inputs = (
        torch.zeros(16, VISION_WIDTH, dtype=torch.float16),
        torch.zeros(17, 1, VISION_WIDTH, dtype=torch.float16),
        torch.zeros(16, 1, VISION_WIDTH),
        torch.zeros(16, 1, 2 * VISION_WIDTH, dtype=torch.float16)[:, :, ::2],
        torch.zeros(0, 1, VISION_WIDTH, dtype=torch.float16),
    )
    for x in inputs:
        assert candidate(x) is x
        original.forward.assert_called_with(x)
    assert original.forward.call_count == len(inputs)
    native.assert_not_called()


def test_first_activation_gate_runs_once_and_refuses_a_failed_candidate():
    original = nn.LayerNorm(VISION_WIDTH).half()
    zero = torch.zeros(16, 1, VISION_WIDTH, dtype=torch.float16)
    x = torch.randn_like(zero)
    verify = Mock(return_value=True)
    candidate = VisionZeroAddNorm(original, zero, normalize, verify)
    candidate(x)
    candidate(x + 1)
    assert verify.call_count == 1
    candidate = VisionZeroAddNorm(original, zero, normalize, Mock(return_value=False))
    with pytest.raises(ValueError, match="first model-activation"):
        candidate(x)
    assert candidate.verify is not None


@pytest.mark.parametrize(
    "norm,shape",
    [
        (nn.LayerNorm(4608), (16, 1, VISION_WIDTH)),
        (nn.LayerNorm(VISION_WIDTH, elementwise_affine=False), (16, 1, VISION_WIDTH)),
        (nn.LayerNorm(VISION_WIDTH), (16, VISION_WIDTH)),
        (nn.LayerNorm(VISION_WIDTH), (0, 1, VISION_WIDTH)),
    ],
)
def test_constructor_rejects_unvalidated_geometry(norm, shape):
    with pytest.raises(ValueError, match="bounded three-dimensional"):
        VisionZeroAddNorm(norm, torch.zeros(shape), Mock())
