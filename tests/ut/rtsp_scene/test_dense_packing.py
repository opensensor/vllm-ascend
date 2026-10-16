import hashlib
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from tools.rtsp_scene.dense_bench import checked_build, load_projection, pack_weight
from tools.rtsp_scene.kernel_compare import check_build_pair


def test_build_fingerprints_reject_a_mixed_native_stack(tmp_path):
    hashes = {}
    for name in ("dense-bridge.so", "dense-k2048.bin", "dense-k4096.bin", "dense-k6144.bin", "nz-swiglu.bin"):
        data = name.encode()
        (tmp_path / name).write_bytes(data)
        hashes[name] = hashlib.sha256(data).hexdigest()
    manifest = {"columns": 128, "inner_tile": 512, "sha256": hashes}
    (tmp_path / "build.json").write_text(json.dumps(manifest))
    assert checked_build(tmp_path) == manifest
    (tmp_path / "dense-k4096.bin").write_bytes(b"different build")
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        checked_build(tmp_path)


def test_paired_builds_reject_duplicate_registries_and_tampered_binaries(tmp_path):
    directories = tmp_path / "control", tmp_path / "candidate"
    for directory in directories:
        directory.mkdir()
        hashes = {}
        for name in ("dense-bridge.so", "dense-k2048.bin", "dense-k4096.bin", "dense-k6144.bin", "nz-swiglu.bin"):
            (directory / name).write_bytes(name.encode())
            hashes[name] = hashlib.sha256(name.encode()).hexdigest()
        (directory / "build.json").write_text(json.dumps({"columns": 256, "inner_tile": 256, "sha256": hashes}))
    with pytest.raises(ValueError, match="different frozen Torch namespaces"):
        check_build_pair(*directories)
    manifest = json.loads((directories[1] / "build.json").read_text())
    manifest.update(namespace="rtsp_dense_scatter16", max_decode_rows=16)
    (directories[1] / "build.json").write_text(json.dumps(manifest))
    assert check_build_pair(*directories)[1]["max_decode_rows"] == 16
    (directories[1] / "nz-swiglu.bin").write_bytes(b"mismatched")
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        check_build_pair(*directories)


def test_packing_preserves_every_cube_cell():
    weight = torch.arange(512 * 4096, dtype=torch.int32).reshape(512, 4096)
    for columns in (64, 128, 256, 512):
        packed = pack_weight(weight, columns)
        assert packed.is_contiguous()
        restored = packed.permute(0, 2, 3, 1, 4).contiguous().reshape_as(weight)
        assert torch.equal(weight, restored)
        for n, k in ((0, 0), (15, 31), (16, 32), (63, 255), (128, 256), (255, 4095)):
            assert packed[n // columns, k // 32, (n % columns) // 16, n % 16, k % 32] == weight[n, k]


def test_unsupported_cube_geometry_is_rejected():
    for shape, tile in (((33, 32), 64), ((128, 31), 128), ((128, 32), 1024)):
        with pytest.raises(ValueError):
            pack_weight(torch.zeros(shape, dtype=torch.int8), tile)


def write_checkpoint(directory: Path, tensors):
    save_file(tensors, directory / "weights.safetensors")
    (directory / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {name: "weights.safetensors" for name in tensors}})
    )


def test_gate_up_slices_each_piece_and_its_scales(tmp_path):
    prefix = "model.language_model.layers.0.mlp."
    tensors = {}
    for name, offset in (("gate_proj", 0), ("up_proj", 100)):
        tensors[prefix + name + ".weight"] = (torch.arange(8 * 32).reshape(8, 32) + offset).to(torch.int8)
        tensors[prefix + name + ".weight_scale"] = torch.arange(8).float()[:, None] + offset
    write_checkpoint(tmp_path, tensors)
    weight, scales = load_projection(tmp_path, "gate_up", 1, 2)
    assert torch.equal(weight, torch.cat([tensors[prefix + name + ".weight"][4:] for name in ("gate_proj", "up_proj")]))
    assert torch.equal(scales, torch.tensor([4, 5, 6, 7, 104, 105, 106, 107]).float())


def test_down_retains_original_per_channel_scale(tmp_path):
    prefix = "model.language_model.layers.0.mlp.down_proj"
    weight = torch.arange(4 * 64).reshape(4, 64).to(torch.int8)
    scales = torch.tensor([0.001, 0.5, 0.02, 0.3])
    write_checkpoint(tmp_path, {prefix + ".weight": weight, prefix + ".weight_scale": scales[:, None]})
    actual_weight, actual_scales = load_projection(tmp_path, "down", 1, 2)
    assert torch.equal(actual_weight, weight[:, 32:])
    assert torch.equal(actual_scales, scales)


def test_requested_layer_uses_its_own_weights_and_scales(tmp_path):
    tensors = {}
    for layer in (0, 1):
        prefix = f"model.language_model.layers.{layer}.mlp.down_proj"
        tensors[prefix + ".weight"] = torch.full((4, 64), layer + 1, dtype=torch.int8)
        tensors[prefix + ".weight_scale"] = torch.full((4, 1), layer + 0.1)
    write_checkpoint(tmp_path, tensors)
    weight, scales = load_projection(tmp_path, "down", 1, 2, layer=1)
    assert torch.equal(weight, torch.full((4, 32), 2, dtype=torch.int8))
    assert torch.equal(scales, torch.full((4,), 1.1))
    for layer in (-1, True, "1"):
        with pytest.raises(ValueError, match="nonnegative integer"):
            load_projection(tmp_path, "down", 1, 2, layer=layer)
