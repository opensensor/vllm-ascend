import hashlib
import json
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest
import torch

from tools.rtsp_scene.dense_runtime import DenseProjection
from tools.rtsp_scene.input_quant import INPUT_WIDTHS, InputQuantizer, checked_input_build


def test_frozen_input_build_requires_complete_matching_binaries(tmp_path):
    hashes = {}
    for inner in INPUT_WIDTHS:
        name = f"input-k{inner}.bin"
        data = str(inner).encode()
        (tmp_path / name).write_bytes(data)
        hashes[name] = hashlib.sha256(data).hexdigest()
    manifest = {"max_rows": 8, "arguments": 4, "sha256": hashes}
    (tmp_path / "build.json").write_text(json.dumps(manifest))
    assert checked_input_build(tmp_path) == manifest
    (tmp_path / "input-k4096.bin").write_bytes(b"changed")
    with pytest.raises(ValueError, match="fingerprint"):
        checked_input_build(tmp_path)
    manifest["max_rows"] = 16
    (tmp_path / "build.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="eight-row"):
        checked_input_build(tmp_path)


def quantizer_stub():
    instance = InputQuantizer.__new__(InputQuantizer)
    instance.backend = SimpleNamespace(
        empty=Mock(side_effect=lambda rows, inner, dtype, device, nz: torch.empty(rows, inner, dtype=dtype)),
        launch=Mock(),
    )
    instance.get_format = Mock(return_value=2)
    instance.kernels = {inner: object() for inner in INPUT_WIDTHS}
    instance.configs = {rows: torch.tensor([rows]) for rows in range(1, 9)}
    return instance


def test_supported_input_uses_nz_output_and_bounded_cached_config():
    instance = quantizer_stub()
    x = torch.zeros(3, 2048, dtype=torch.float16)
    output, scales = instance(x)
    assert output.shape == (3, 2048) and output.dtype == torch.int8
    assert scales.shape == (24,) and scales.dtype == torch.float32
    instance.backend.empty.assert_called_once_with(3, 2048, torch.int8, x.device, True)
    args = instance.backend.launch.call_args.args[1]
    assert args[0] is x and args[-1] is instance.configs[3]


@pytest.mark.parametrize(
    "x",
    [
        torch.zeros(0, 2048).half(),
        torch.zeros(9, 2048).half(),
        torch.zeros(1, 1152).half(),
        torch.zeros(1, 2048),
        torch.zeros(1, 4096).half()[:, ::2],
    ],
)
def test_unsupported_input_never_launches(x):
    instance = quantizer_stub()
    with pytest.raises(ValueError, match="supported FP16"):
        instance(x)
    instance.backend.launch.assert_not_called()


def test_non_nd_physical_storage_uses_reference_codes_and_strided_scales():
    instance = quantizer_stub()
    instance.get_format.return_value = 29
    x = torch.zeros(3, 2048, dtype=torch.float16)
    codes = torch.ones_like(x, dtype=torch.int8)
    scales = torch.tensor([0.1, 0.2, 0.3])
    instance.backend.dynamic_quant = Mock(return_value=(codes, scales))
    instance.backend.format_cast = Mock(side_effect=lambda value, format: value)
    actual, actual_scales = instance(x)
    assert actual is codes
    assert torch.equal(actual_scales.reshape(3, 8), scales[:, None].expand(3, 8))
    instance.backend.launch.assert_not_called()
    instance.backend.dynamic_quant.assert_called_once_with(x)
    instance.backend.format_cast.assert_has_calls([call(x, 2), call(codes, 29)])


def test_complete_projection_uses_fused_codes_and_strided_scales():
    x = torch.zeros(3, 2048, dtype=torch.float16)
    codes, scales = torch.zeros_like(x, dtype=torch.int8), torch.zeros(24)
    backend = SimpleNamespace(
        max_decode_rows=8,
        manifest={"columns": 256},
        format_cast=lambda value, format: value,
        input_quant=Mock(return_value=(codes, scales)),
        dynamic_quant=Mock(),
        matrices={2048: object()},
        empty=Mock(side_effect=lambda rows, columns, dtype, device, nz: torch.empty(rows, columns, dtype=dtype)),
        launch=Mock(),
    )
    layer = SimpleNamespace(weight=torch.zeros(2048, 256, dtype=torch.int8), weight_scale=torch.ones(256), bias=None)
    projection = DenseProjection(layer, backend)
    assert projection(x).shape == (3, 256)
    backend.input_quant.assert_called_once_with(x)
    backend.dynamic_quant.assert_not_called()
    arguments = backend.launch.call_args.args[1]
    assert arguments[0] is codes and arguments[3].data_ptr() == scales.data_ptr()
    assert arguments[-1].tolist() == [3, 256, 0, 1, 8]
