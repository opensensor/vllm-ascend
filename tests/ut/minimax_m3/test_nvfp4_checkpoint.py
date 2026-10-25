# SPDX-License-Identifier: Apache-2.0
import dataclasses
import json

import pytest
import torch
from safetensors.torch import save_file

from tools.glm_perf.resident_control import Control, PatchSession
from tools.minimax_m3 import nvfp4_checkpoint
from tools.minimax_m3.export_nvfp4 import MANIFEST, export
from tools.minimax_m3.export_w4 import file_digest
from tools.minimax_m3.mixed_checkpoint import dequant_nvfp4
from tools.minimax_m3.nvfp4 import unpack_source
from tools.minimax_m3.nvfp4_checkpoint import read_source_manifest, source_bank
from tools.minimax_m3.resident import candidate_source
from vllm_ascend._310p.ops.minimax_m3 import PackedM3Experts


@pytest.fixture
def bundle(tmp_path):
    source, destination = tmp_path / "source", tmp_path / "converted"
    source.mkdir()
    weights, records = {}, {}
    generator = torch.Generator().manual_seed(51)
    for expert in range(2):
        for projection in ("w1", "w2", "w3"):
            base = f"language_model.model.layers.0.block_sparse_moe.experts.{expert}.{projection}"
            weights[base + ".weight_packed"] = torch.randint(0, 256, (256, 128), dtype=torch.uint8, generator=generator)
            weights[base + ".weight_scale"] = torch.randint(
                0, 100, (256, 16), dtype=torch.uint8, generator=generator
            ).view(torch.float8_e4m3fn)
            weights[base + ".weight_global_scale"] = torch.tensor(100.0 + expert)
            weights[base + ".input_global_scale"] = torch.tensor(32.0)
            records[base + ".weight"] = {"expert": True}
    save_file(weights, source / "model.safetensors")
    (source / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": dict.fromkeys(weights, "model.safetensors")})
    )
    (source / "config.json").write_text(
        json.dumps(
            {
                "quantization_config": {"moe_quant_format": "nvfp4-pack-quantized", "group_size": 16},
                "text_config": {
                    "moe_layer_freq": [1],
                    "num_local_experts": 2,
                    "hidden_size": 256,
                    "intermediate_size": 256,
                },
            }
        )
    )
    result = export(source, destination)
    dense = {key: result[key] for key in ("config_sha256", "index_sha256")}
    dense["weights"] = records
    return source, destination, dense, weights


def test_export_source_identity_exact_decode_and_resume(bundle):
    source, destination, dense, weights = bundle
    manifest = read_source_manifest(destination, dense)
    assert len(manifest["banks"]) == 6
    for record in manifest["banks"].values():
        data = dict(source_bank(destination, record))
        assert all(tensor.untyped_storage().resizable() for tensor in data.values())
        base = f"language_model.model.layers.0.block_sparse_moe.experts.{record['ep_rank']}.{record['projection']}"
        expected = dequant_nvfp4(
            *(weights[base + suffix] for suffix in (".weight_packed", ".weight_scale", ".weight_global_scale"))
        ).half()
        assert torch.equal(unpack_source(*(data[k][0] for k in ("codes", "scales", "lookup"))), expected)
    before = {p: p.stat().st_mtime_ns for p in destination.glob("*.safetensors")}
    assert export(source, destination) == manifest
    assert before == {p: p.stat().st_mtime_ns for p in before}


def test_source_bundle_rejects_missing_bank_and_wrong_provenance(bundle):
    _, destination, dense, _ = bundle
    with pytest.raises(ValueError, match="provenance"):
        read_source_manifest(destination, {**dense, "config_sha256": "wrong"})
    path = destination / MANIFEST
    manifest = json.loads(path.read_text())
    manifest["banks"].pop(next(iter(manifest["banks"])))
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="expert set"):
        read_source_manifest(destination, dense)


def test_source_bundle_rejects_corruption(bundle):
    source, destination, dense, _ = bundle
    manifest = read_source_manifest(destination, dense)
    record = next(iter(manifest["banks"].values()))
    path = destination / record["file"]
    content = bytearray(path.read_bytes())
    content[-1] ^= 1
    path.write_bytes(content)
    with pytest.raises(ValueError, match="checksum"):
        dict(source_bank(destination, record))
    with pytest.raises(ValueError, match="Corrupt"):
        export(source, destination)


def test_full_bank_loader_keeps_raw_scale_bytes_in_forward(bundle, tmp_path, monkeypatch):
    _, destination, dense, _ = bundle
    native = tmp_path / "native"
    native.mkdir()
    (native / "build-options.json").write_text('{"source_nvfp4": true}')
    (native / "grouped_w4.bin").write_bytes(b"fixture")
    (native / "bridge.so").write_bytes(b"fixture")

    def entry(name):
        path = native / name
        return {"path": str(path), "sha256": file_digest(path)}

    spec = {
        "name": "source_test",
        "libraries": [entry("bridge.so")],
        "assets": [entry("build-options.json"), entry("grouped_w4.bin")],
        "operators": ["source_test::launch"],
        "validation_source": "def validate(): return {'passed': True}",
    }
    native_path = native / "manifest.json"
    native_path.write_text(json.dumps(spec))
    calls = []

    class Kernel:
        max_compact_routes = 4

        def __init__(self, *args):
            pass

        def __call__(self, x, codes, scales, ends, lookup):
            assert scales.dtype == torch.uint8
            assert codes.dtype == torch.int8
            assert ends.dtype == torch.int32
            calls.append(lookup)
            return torch.zeros(x.shape[0], codes.shape[1], dtype=x.dtype)

    monkeypatch.setattr(torch.ops, "load_library", lambda path: None)
    monkeypatch.setattr(nvfp4_checkpoint, "GroupedW4Kernel", Kernel)
    module = PackedM3Experts(256, 256, 2, 2, 1, 2, prefix="model.layers.0.experts")
    assert nvfp4_checkpoint.load_source_banks(module, destination, dense, native_path, 2048) == 3
    assert module.source_nvfp4 and module.max_grouped_routes == 2048
    result = module(torch.ones(1, 256).half(), torch.tensor([[0.0, 1.0]]))
    assert result.shape == (1, 256) and len(calls) == 3
    assert calls[0] is module.w1_lookup and calls[1] is module.w3_lookup and calls[2] is module.w2_lookup


def test_source_hot_reload_preserves_lookup_identity():
    arguments = []

    def kernel(*args):
        arguments.append(args)
        return args[0]

    projector = nvfp4_checkpoint.SourceProjector(None, {})
    codes = torch.zeros(1, 16, 128, dtype=torch.int8)
    lookup = torch.zeros(1, 4096).half()
    projector.lookups[codes.data_ptr()] = lookup
    original = type(projector).__call__
    session = PatchSession({"source_v2": kernel})
    candidate = Control("1" * 32, "direct-target", "m3_candidate", candidate_source(source_native_name="source_v2"))
    session.prepare(dataclasses.asdict(candidate))
    x = torch.ones(1, 256).half()
    try:
        session.apply(candidate.generation)
        assert projector(x, codes, None, None) is x
        assert arguments[0][-1] is lookup
    finally:
        baseline = Control("2" * 32, "direct-target", "baseline", "")
        session.prepare(dataclasses.asdict(baseline))
        session.apply(baseline.generation)
    assert type(projector).__call__ is original
    with pytest.raises(ValueError, match="not both"):
        candidate_source(native_name="w4_v2", source_native_name="source_v2")
