import json
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from torch import nn

from tools.rtsp_scene.dense_runtime import NZMLP, DecodeLinearMethod, eligible
from tools.rtsp_scene.dense_setup import RUNTIME_CONFIG, checked_qkv_build, prepare_overlay
from tools.rtsp_scene.launch import build_command


def test_dense_worker_is_explicit_and_requires_int8_tp2():
    command = build_command("/private/overlay", 8003, 8, True, 2, dense=True)
    assert command[command.index("--worker-cls") + 1] == "tools.rtsp_scene.dense_worker.SceneDenseWorker"
    assert "--worker-cls" not in build_command("/private/control", 8003, 8, True, 2)
    for int8, size in ((False, 2), (True, 1)):
        with pytest.raises(ValueError):
            build_command("/private/overlay", 8003, 8, int8, size, dense=True)


def test_overlay_is_separate_and_preserves_original_checkpoint(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    (model / "model.safetensors.index.json").write_text("{}")
    (model / "weights.safetensors").write_bytes(b"original")
    gate = "dense-gate-rank0.json"
    (model / gate).write_text("prior-gate")
    vision_gate = "vision-gate-rank0.json"
    (model / vision_gate).write_text("prior-vision-gate")
    output = tmp_path / "overlay"
    with patch("tools.rtsp_scene.dense_setup.checked_build", return_value={"sha256": {}}):
        prepare_overlay(model, tmp_path / "build", output, "ffn")
        with pytest.raises(ValueError):
            prepare_overlay(model, tmp_path / "build", model / "nested", "ffn")
        with pytest.raises(FileExistsError):
            prepare_overlay(model, tmp_path / "build", output, "ffn")
    assert (output / "weights.safetensors").resolve() == model / "weights.safetensors"
    assert not (model / RUNTIME_CONFIG).exists()
    assert not (output / gate).exists()
    assert not (output / vision_gate).exists()
    (output / gate).write_text("new-gate")
    assert (model / gate).read_text() == "prior-gate"
    assert json.loads((output / RUNTIME_CONFIG).read_text())["mode"] == "ffn"
    assert "vision_zero_add_norm" not in json.loads((output / RUNTIME_CONFIG).read_text())
    with patch("tools.rtsp_scene.dense_setup.checked_build", return_value={"sha256": {}}):
        candidate = prepare_overlay(model, tmp_path / "build", tmp_path / "vision-overlay", "all", True)
    assert candidate["vision_zero_add_norm"] is True
    assert (model / vision_gate).read_text() == "prior-vision-gate"


def test_input_quantization_overlay_freezes_its_build_and_rejects_row_mismatch(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    (model / "model.safetensors.index.json").write_text("{}")
    quant = tmp_path / "quant"
    frozen = {"max_rows": 8, "arguments": 4, "sha256": {}}
    with (
        patch("tools.rtsp_scene.dense_setup.checked_build", return_value={"sha256": {}}),
        patch("tools.rtsp_scene.dense_setup.checked_input_build", return_value=frozen) as check,
    ):
        config = prepare_overlay(model, tmp_path / "matrix", tmp_path / "selected", "all", True, quant)
    check.assert_called_once_with(quant)
    assert config["input_quant"] == {"build": str(quant.resolve()), "manifest": frozen}
    assert config["vision_zero_add_norm"] is True
    with (
        patch("tools.rtsp_scene.dense_setup.checked_build", return_value={"max_decode_rows": 16}),
        pytest.raises(ValueError, match="eight-row"),
    ):
        prepare_overlay(model, tmp_path / "matrix", tmp_path / "rejected", "all", True, quant)
    assert not (tmp_path / "rejected").exists()


def test_device_image_overlay_changes_only_private_architecture_and_freezes_build(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    original = {"architectures": ["Qwen3VLForConditionalGeneration"], "model_type": "qwen3_vl"}
    (model / "config.json").write_text(json.dumps(original))
    (model / "model.safetensors.index.json").write_text("{}")
    frozen = {"version": 2, "sha256": {"kernel": "frozen"}}
    build, image = tmp_path / "matrix", tmp_path / "image"
    with (
        patch("tools.rtsp_scene.dense_setup.checked_build", return_value={"sha256": {}}),
        patch("tools.rtsp_scene.dense_setup.checked_image_build", return_value=frozen),
    ):
        output = tmp_path / "selected"
        config = prepare_overlay(model, build, output, "all", True, device_image_build=image)
        assert config["device_image_prepare"] == {"build": str(image.resolve()), "manifest": frozen}
        assert json.loads((model / "config.json").read_text()) == original
        assert not (output / "config.json").is_symlink()
        assert json.loads((output / "config.json").read_text())["architectures"] == [
            "SceneDeviceQwen3VLForConditionalGeneration"
        ]
        (model / "config.json").write_text('{"architectures":["OtherModel"]}')
        with pytest.raises(ValueError, match="Qwen3-VL"):
            prepare_overlay(model, build, tmp_path / "wrong", "all", device_image_build=image)
        assert not (tmp_path / "wrong").exists()
    with (
        patch("tools.rtsp_scene.dense_setup.checked_build", return_value={"sha256": {}}),
        patch("tools.rtsp_scene.dense_setup.checked_image_build", return_value={"version": 1}),
        pytest.raises(ValueError, match="version-two"),
    ):
        prepare_overlay(model, build, tmp_path / "legacy", "all", device_image_build=image)
    assert not (tmp_path / "legacy").exists()


def test_qkv_build_has_independent_registry_and_matching_decode_capacity(tmp_path):
    control = {"namespace": "rtsp_dense_control", "max_decode_rows": 8}
    candidate = {"namespace": "rtsp_dense_qkv", "max_decode_rows": 8, "sha256": {"kernel": "frozen"}}
    with patch("tools.rtsp_scene.dense_setup.checked_build", return_value=candidate):
        assert checked_qkv_build(tmp_path, control) == {"build": str(tmp_path.resolve()), "manifest": candidate}
    for invalid, message in [
        ({**candidate, "namespace": control["namespace"]}, "distinct native namespaces"),
        ({**candidate, "max_decode_rows": 16}, "decode row limit"),
    ]:
        with (
            patch("tools.rtsp_scene.dense_setup.checked_build", return_value=invalid),
            pytest.raises(ValueError, match=message),
        ):
            checked_qkv_build(tmp_path, control)


def test_qkv_overlay_preserves_primary_kernels_and_rejects_ffn_only_mode(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    (model / "model.safetensors.index.json").write_text("{}")
    control = {"namespace": "rtsp_dense_control", "max_decode_rows": 8}
    candidate = {"namespace": "rtsp_dense_qkv", "max_decode_rows": 8}
    build, qkv = tmp_path / "primary", tmp_path / "qkv"
    with patch("tools.rtsp_scene.dense_setup.checked_build", side_effect=[control, candidate]):
        config = prepare_overlay(model, build, tmp_path / "selected", "all", qkv_build=qkv)
    assert config["build"] == str(build.resolve()) and config["manifest"] == control
    assert config["qkv_projection_build"] == {"build": str(qkv.resolve()), "manifest": candidate}
    with (
        patch("tools.rtsp_scene.dense_setup.checked_build", side_effect=[control, candidate]),
        pytest.raises(ValueError, match="mode all"),
    ):
        prepare_overlay(model, build, tmp_path / "wrong", "ffn", qkv_build=qkv)
    assert not (tmp_path / "wrong").exists()


def test_linear_fallback_preserves_bias_and_forward_arguments():
    original = Mock()
    projection = Mock(inner=4, max_decode_rows=8)
    method = DecodeLinearMethod(original, projection)
    x = torch.zeros(1, 4, dtype=torch.float16)
    method.apply("layer", x)
    projection.assert_called_once_with(x)
    original.apply.assert_not_called()
    bias = torch.ones(4)
    method.apply("layer", x, bias, tp_rank=1)
    original.apply.assert_called_once_with("layer", x, bias, tp_rank=1)
    for unsupported in (torch.zeros(9, 4, dtype=torch.float16), x.float(), torch.zeros(1, 5, dtype=torch.float16)):
        assert not eligible(unsupported, 4)
        method.apply("layer", unsupported)
    assert original.apply.call_count == 4


class OriginalMLP(nn.Module):
    def __init__(self, reduce_results=True):
        super().__init__()
        self.gate_up_proj = SimpleNamespace(weight=torch.empty(4096, 12288, device="meta"))
        self.down_proj = SimpleNamespace(
            weight=torch.empty(6144, 4096, device="meta"),
            tp_size=2,
            input_is_parallel=True,
            bias=None,
            reduce_results=reduce_results,
        )
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        return x + 2


class FakeProjection(nn.Module):
    def __init__(self, layer, backend, **kwargs):
        super().__init__()

    def forward(self, x):
        return torch.zeros(x.shape[0], 12288, dtype=torch.float16)

    def quantized(self, q, scales):
        return torch.zeros(q.shape[0], 4096, dtype=torch.float16)


def test_nz_mlp_keeps_one_collective_and_prefill_fallback():
    backend = Mock(max_decode_rows=8)
    backend.empty.side_effect = lambda rows, columns, dtype, device, nz: torch.zeros(rows, columns, dtype=dtype)
    reduce = Mock(side_effect=lambda x: x + 1)
    original = OriginalMLP()
    # Meta weights keep this CPU routing test small; use CPU geometry buffers.
    original.down_proj.weight = torch.empty(6144, 4096, device="cpu")
    with patch("tools.rtsp_scene.dense_runtime.DenseProjection", FakeProjection):
        candidate = NZMLP(original, backend, reduce)
    x = torch.zeros(1, 4096, dtype=torch.float16)
    assert torch.equal(candidate(x), torch.ones_like(x))
    assert reduce.call_count == 1 and original.calls == 0
    prefill = torch.zeros(9, 4096, dtype=torch.float16)
    assert torch.equal(candidate(prefill), prefill + 2)
    assert reduce.call_count == 1 and original.calls == 1
    candidate.reduce_results = False
    assert torch.equal(candidate(x), torch.zeros_like(x))
    assert reduce.call_count == 1


def test_extended_row_limit_keeps_prefill_and_bias_fallback():
    original = Mock()
    projection = Mock(inner=4, max_decode_rows=16)
    method = DecodeLinearMethod(original, projection)
    for rows in (8, 9, 15, 16):
        x = torch.zeros(rows, 4, dtype=torch.float16)
        method.apply("layer", x)
        projection.assert_called_with(x)
    assert projection.call_count == 4 and original.apply.call_count == 0
    for x in (torch.zeros(17, 4, dtype=torch.float16), torch.zeros(16, 8, dtype=torch.float16)[:, ::2]):
        assert not eligible(x, 4, 16)
        method.apply("layer", x)
    method.apply("layer", torch.zeros(16, 4, dtype=torch.float16), torch.ones(4))
    assert original.apply.call_count == 3 and projection.call_count == 4


def test_extended_nz_mlp_processes_all_rows_and_keeps_collective():
    backend = Mock(max_decode_rows=16)
    backend.empty.side_effect = lambda rows, columns, dtype, device, nz: torch.zeros(rows, columns, dtype=dtype)
    reduce = Mock(side_effect=lambda x: x + 1)
    original = OriginalMLP()
    original.down_proj.weight = torch.empty(6144, 4096)
    with patch("tools.rtsp_scene.dense_runtime.DenseProjection", FakeProjection):
        candidate = NZMLP(original, backend, reduce)
    assert set(candidate.activation_configs) == set(range(1, 17))
    for rows in (9, 16):
        x = torch.zeros(rows, 4096, dtype=torch.float16)
        assert torch.equal(candidate(x), torch.ones_like(x))
        assert backend.launch.call_args.args[1][3][0] == rows
    assert reduce.call_count == 2 and original.calls == 0
    x = torch.zeros(17, 4096, dtype=torch.float16)
    assert torch.equal(candidate(x), x + 2)
    assert reduce.call_count == 2 and original.calls == 1
