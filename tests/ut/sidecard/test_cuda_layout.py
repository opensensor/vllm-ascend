# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest

from tools.sidecard.device_backend import CUDA_NIBBLE_PLANES, CUDA_ROW_MAJOR, CUDA_WORD_PLANES, encode_cuda_weights


@pytest.mark.parametrize("bits", [2, 3, 4])
def test_nibble_planes_preserve_signed_codes_and_halve_storage(bits):
    values = np.random.default_rng(bits).integers(-(1 << (bits - 1)), 1 << (bits - 1), size=(2, 64, 96), dtype=np.int8)
    encoded = encode_cuda_weights(values, CUDA_NIBBLE_PLANES)
    words = np.frombuffer(encoded, dtype="<u2").reshape(2, 3, 8, 64)
    lanes = np.stack([((words >> (4 * lane)) & 15).astype(np.int8) - 8 for lane in range(4)], axis=-1)
    restored = lanes.transpose(0, 3, 1, 2, 4).reshape(values.shape)
    assert restored.tobytes() == values.tobytes()
    assert len(encoded) == values.nbytes // 2


@pytest.mark.parametrize("code", [-9, 8])
def test_nibble_planes_reject_out_of_range_codes(code):
    with pytest.raises(ValueError, match="four-bit"):
        encode_cuda_weights(np.full((1, 32, 32), code, dtype=np.int8), CUDA_NIBBLE_PLANES)


@pytest.mark.parametrize("shape", [(2, 64, 96), (1, 4096, 4096), (1, 4096, 2048)])
def test_word_planes_preserve_every_signed_integer_code(shape):
    values = np.random.default_rng(53).integers(-128, 128, size=shape, dtype=np.int8)
    encoded = encode_cuda_weights(values, CUDA_WORD_PLANES)
    experts, channels, width = shape
    planes = np.frombuffer(encoded, dtype=np.int8).reshape(experts, width // 32, 8, channels, 4)
    restored = planes.transpose(0, 3, 1, 2, 4).reshape(shape)
    assert restored.tobytes() == values.tobytes()
    assert len(encoded) == values.nbytes
    assert encode_cuda_weights(values, CUDA_ROW_MAJOR) == values.tobytes()


@pytest.mark.parametrize(
    "values,layout",
    [
        (np.zeros((1, 32, 32), dtype=np.int8), "unknown"),
        (np.zeros((1, 32, 32), dtype=np.uint8), CUDA_WORD_PLANES),
        (np.zeros((1, 32, 31), dtype=np.int8), CUDA_WORD_PLANES),
        (np.zeros((1, 31, 32), dtype=np.int8), CUDA_WORD_PLANES),
        (np.zeros((32, 32), dtype=np.int8), CUDA_WORD_PLANES),
        (np.zeros((0, 32, 32), dtype=np.int8), CUDA_WORD_PLANES),
    ],
)
def test_invalid_word_plane_geometry(values, layout):
    with pytest.raises(ValueError):
        encode_cuda_weights(values, layout)
