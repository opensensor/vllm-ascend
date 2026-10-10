# SPDX-License-Identifier: Apache-2.0
"""Native CPU export preserves code values, scale contract and full coverage."""

import json

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from tools.deepseek_w2.w2_format import pack_codes
from tools.glm_perf.export_native_cpu import export
from tools.glm_perf.fused_weight_layout import FP16_SCALE_LAYOUT, unpack_cube
from tools.glm_perf.native_checkpoint import INDEX, MANIFEST, file_digest, validate_scale_contract, write_json


def source_and_bundle(tmp_path, bits):
    source, bundle = tmp_path / "canonical", tmp_path / "bundle"
    source.mkdir()
    bundle.mkdir()
    write_json(
        source / "config.json",
        {"text_config": {"n_routed_experts": 4, "hidden_size": 256, "moe_intermediate_size": 256}},
    )
    tensors, values = {"lm_head.weight": torch.eye(4).half()}, {}
    for layer in (3, 4):
        for expert in range(4):
            for projection in ("gate", "up", "down"):
                stem = f"model.language_model.layers.{layer}.mlp.experts.{expert}.{projection}_proj"
                code = (torch.arange(256 * 256).reshape(256, 256) % (1 << bits) - (1 << (bits - 1))).to(torch.int8)
                values[stem + "_codes"] = code
                tensors[stem + "_codes"] = pack_codes(code, bits)
                tensors[stem + "_scale"] = torch.linspace(0.001, 0.02, 64).reshape(8, 8)
    save_file(tensors, str(source / "canonical.safetensors"))
    write_json(source / INDEX, {"metadata": {}, "weight_map": dict.fromkeys(tensors, "canonical.safetensors")})
    (source / "tokenizer.json").write_text("{}")
    (source / "LICENSE").write_text("Original license text")
    options = dict(
        fused_moe=True,
        prepared_weight_layout=True,
        fp16_weight_scales=True,
        version=17,
        helper_package="helpers",
        namespace="native17",
    )
    provenance = {"_build": options, "_helpers": {}}
    for name in (
        "glm_reconstruction_bridge_v17.so",
        "glm_fused_gate_up.bin",
        "glm_fused_down.bin",
        "glm_fused_pack.bin",
        "glm_fused_reduce.bin",
    ):
        (bundle / name).write_bytes(name.encode())
        key = "reconstruction_bridge.cpp" if name.endswith(".so") else name
        provenance[key] = {"binary_sha256": file_digest(bundle / name)}
    write_json(bundle / "provenance.json", provenance)
    return source, bundle, tensors, values, options


@pytest.mark.parametrize("bits", [2, 3, 4])
@pytest.mark.parametrize("world", [2, 4])
def test_export_matches_original_codes_and_publishes_partition(tmp_path, bits, world):
    source, bundle, original, codes, options = source_and_bundle(tmp_path, bits)
    before = {p.name: file_digest(p) for p in source.iterdir()}
    output = tmp_path / "native"
    report = export(source, output, bundle, world_size=world, dense_shard_bytes=64)
    assert report["complete"] and report["world_size"] == world
    assert len(report["native_shards"]) == 2 * world
    assert report["hardware_validation"] == "not_run"
    validate_scale_contract(report, FP16_SCALE_LAYOUT, options)
    index = json.loads((output / INDEX).read_text())["weight_map"]
    assert index.keys() == original.keys()
    for filename in set(index.values()):
        with safe_open(str(output / filename), framework="pt", device="cpu") as handle:
            assert set(handle.keys()) == {name for name, path in index.items() if path == filename}
            for name in sorted(handle.keys()):
                actual = handle.get_tensor(name)
                if name.endswith("_codes"):
                    assert torch.equal(unpack_cube(actual[None].view(torch.int8), 256)[0], codes[name])
                elif name.endswith("_scale"):
                    assert actual.dtype == torch.float16
                    assert torch.equal(actual.view(torch.int16), original[name].half().view(torch.int16))
                else:
                    assert torch.equal(actual, original[name])
    assert not (output / "canonical.safetensors").exists()
    assert (output / "LICENSE").read_text() == "Original license text"
    assert {p.name: file_digest(p) for p in source.iterdir()} == before
    with pytest.raises(FileExistsError):
        export(source, output, bundle, world_size=world)


def test_invalid_partition_fails_before_creating_destination(tmp_path):
    source, bundle, _, _, _ = source_and_bundle(tmp_path, 2)
    output = tmp_path / "native"
    with pytest.raises(ValueError, match="partition"):
        export(source, output, bundle, world_size=3)
    assert not output.exists()


def test_scale_overflow_leaves_incomplete_manifest(tmp_path):
    source, bundle, tensors, _, _ = source_and_bundle(tmp_path, 3)
    name = next(n for n in tensors if n.endswith("_scale"))
    tensors[name] = torch.full((8, 8), 1e10)
    save_file(tensors, str(source / "canonical.safetensors"))
    output = tmp_path / "native"
    with pytest.raises(ValueError, match="overflows"):
        export(source, output, bundle, world_size=2)
    assert json.loads((output / MANIFEST).read_text())["complete"] is False
    assert not (output / INDEX).exists()


def test_already_native_codes_are_never_repacked(tmp_path):
    source, bundle, _, _, _ = source_and_bundle(tmp_path, 3)
    c = json.loads((source / "config.json").read_text())
    c["ascend_glm_expert_layout"] = "cube_n128_k256_v1"
    write_json(source / "config.json", c)
    with pytest.raises(ValueError, match="already-native"):
        export(source, tmp_path / "native", bundle, world_size=2)


def test_decode_bundle_attachment_preserves_all_tensor_files(tmp_path):
    from tools.glm_perf.export_native_cpu import attach_decode_bundle

    source, bundle, _, _, _ = source_and_bundle(tmp_path, 4)
    output = tmp_path / "native"
    export(source, output, bundle, world_size=2)
    before = {p.name: file_digest(p) for p in output.glob("*.safetensors")}
    with pytest.raises(ValueError, match="distinct"):
        attach_decode_bundle(output, bundle)
    provenance = json.loads((bundle / "provenance.json").read_text())
    provenance["_build"].update(namespace="native18", helper_package="helpers18")
    write_json(bundle / "provenance.json", provenance)
    manifest = attach_decode_bundle(output, bundle)
    assert manifest["decode_kernel_bundle"] == "native-decode-kernels"
    assert {p.name: file_digest(p) for p in output.glob("*.safetensors")} == before
    assert file_digest(output / "native-decode-kernels/provenance.json") == manifest["decode_kernel_provenance_sha256"]
    with pytest.raises(ValueError, match="already has"):
        attach_decode_bundle(output, bundle)


def test_offline_publication_verifier_rejects_unindexed_payload_and_corruption(tmp_path):
    from tools.glm_perf.verify_card_checkpoint import verify

    source, bundle, _, _, _ = source_and_bundle(tmp_path, 3)
    output = tmp_path / "native"
    manifest = export(source, output, bundle, world_size=2)
    report = verify(output)
    assert report["code_tensors"] == report["scale_tensors"] == 24
    assert report["shard_checksums_verified"]
    extra = output / "unused.safetensors"
    save_file({"unused": torch.ones(2)}, str(extra))
    with pytest.raises(ValueError, match="unindexed tensor shards"):
        verify(output)
    extra.unlink()
    shard = output / manifest["native_shards"][0]["file"]
    data = bytearray(shard.read_bytes())
    data[-1] ^= 1
    shard.write_bytes(data)
    with pytest.raises(ValueError, match="checksum"):
        verify(output)


@pytest.mark.parametrize("bits", [2, 3, 4])
def test_byte_transcode_matches_independent_cube_packer(bits):
    from tools.glm_perf.cpu_weight_layout import pack_canonical_cpu
    from tools.glm_perf.fused_weight_layout import pack_cube

    generator = torch.Generator().manual_seed(31)
    codes = torch.randint(-(1 << (bits - 1)), 1 << (bits - 1), (384, 768), dtype=torch.int8, generator=generator)
    packed = pack_codes(codes, bits)
    result = pack_canonical_cpu(packed, 768)
    assert torch.equal(result, pack_cube(codes[None], bits)[0].view(torch.uint8))
    assert torch.equal(pack_canonical_cpu(packed.view(torch.int8), 768), result)


def test_resume_verifies_reuses_and_repairs_only_incomplete_candidate(tmp_path):
    source, bundle, tensors, _, _ = source_and_bundle(tmp_path, 4)
    bad = next(n for n in tensors if ".layers.4." in n and n.endswith("_scale"))
    good = tensors[bad].clone()
    tensors[bad].fill_(1e10)
    save_file(tensors, str(source / "canonical.safetensors"))
    output = tmp_path / "native"
    with pytest.raises(ValueError, match="overflows"):
        export(source, output, bundle, world_size=2)
    preserved = output / "native-rank0-layer003.safetensors"
    stat = preserved.stat()
    # Simulate a truncated write in a different already prepared partition.
    corrupted = output / "native-rank1-layer003.safetensors"
    corrupted.write_bytes(b"partial")
    tensors[bad] = good
    save_file(tensors, str(source / "canonical.safetensors"))
    report = export(source, output, bundle, world_size=2, resume=True)
    assert report["complete"]
    assert preserved.stat().st_ino == stat.st_ino and preserved.stat().st_mtime_ns == stat.st_mtime_ns
    with safe_open(str(corrupted), framework="pt", device="cpu") as handle:
        assert len(handle.keys()) == 12
    with pytest.raises(ValueError, match="complete"):
        export(source, output, bundle, world_size=2, resume=True)


@pytest.mark.parametrize("change", ["partition", "kernel", "shard_budget"])
def test_resume_rejects_changed_contract(tmp_path, change):
    source, bundle, tensors, _, _ = source_and_bundle(tmp_path, 4)
    bad = next(n for n in tensors if n.endswith("_scale"))
    tensors[bad].fill_(1e10)
    save_file(tensors, str(source / "canonical.safetensors"))
    output = tmp_path / "native"
    with pytest.raises(ValueError, match="overflows"):
        export(source, output, bundle, world_size=2)
    kwargs = dict(world_size=2, resume=True)
    if change == "partition":
        kwargs["world_size"] = 4
    elif change == "shard_budget":
        kwargs["dense_shard_bytes"] = 17
    else:
        provenance = json.loads((bundle / "provenance.json").read_text())
        provenance["_build"]["namespace"] = "other"
        write_json(bundle / "provenance.json", provenance)
    with pytest.raises(ValueError, match="resume"):
        export(source, output, bundle, **kwargs)


@pytest.mark.parametrize("width", [0, 255, 257])
def test_byte_transcode_rejects_invalid_width(width):
    from tools.glm_perf.cpu_weight_layout import pack_canonical_cpu

    with pytest.raises(ValueError):
        pack_canonical_cpu(torch.zeros(256, 128, dtype=torch.uint8), width)


def test_indexer_attachment_is_relocatable_and_preserves_weights(tmp_path):
    import shutil
    from pathlib import Path

    from tests.ut.glm_perf.test_indexer_bundle import make_bundle
    from tools.glm_perf.export_native_cpu import attach_indexer_bundle
    from tools.glm_perf.indexer_bundle import verified_manifest

    source, bundle, _, _, _ = source_and_bundle(tmp_path, 3)
    output = tmp_path / "native"
    export(source, output, bundle, world_size=2)
    before = {p.name: file_digest(p) for p in output.glob("*.safetensors")}
    attach_indexer_bundle(output, make_bundle(tmp_path / "indexer-source"))
    assert {p.name: file_digest(p) for p in output.glob("*.safetensors")} == before
    moved = tmp_path / "hub-snapshot"
    shutil.copytree(output, moved)
    manifest = json.loads((moved / "indexer-kernels/manifest.json").read_text())
    assert all(not Path(row["path"]).is_absolute() for row in manifest["libraries"] + manifest["assets"])
    verified_manifest(moved / "indexer-kernels")
    with pytest.raises(ValueError, match="without an indexer"):
        attach_indexer_bundle(output, tmp_path / "indexer-source")


def test_export_detects_an_entire_missing_layer_before_creating_output(tmp_path):
    source, bundle, _, _, _ = source_and_bundle(tmp_path, 4)
    config = json.loads((source / "config.json").read_text())
    config["text_config"].update(
        num_hidden_layers=5, num_nextn_predict_layers=0, mlp_layer_types=["dense"] * 3 + ["sparse"] * 2
    )
    write_json(source / "config.json", config)
    index = json.loads((source / INDEX).read_text())
    index["weight_map"] = {name: value for name, value in index["weight_map"].items() if ".layers.4." not in name}
    write_json(source / INDEX, index)
    output = tmp_path / "native"
    with pytest.raises(ValueError, match="entire routed"):
        export(source, output, bundle, world_size=2)
    assert not output.exists()


def test_original_attribution_is_copied_and_checked_without_changing_weights(tmp_path):
    from tools.glm_perf.export_native_cpu import attach_attribution
    from tools.glm_perf.verify_card_checkpoint import verify

    source, bundle, _, _, _ = source_and_bundle(tmp_path, 3)
    (source / "README.md").write_text("Original model attribution and license instructions")
    output = tmp_path / "native"
    export(source, output, bundle, world_size=2)
    before = {p.name: file_digest(p) for p in output.glob("*.safetensors")}
    manifest = attach_attribution(output, source)
    assert (output / "SOURCE_README.md").read_bytes() == (source / "README.md").read_bytes()
    assert {p.name: file_digest(p) for p in output.glob("*.safetensors")} == before
    assert len(manifest["source_attribution"]["files"]) == 2
    verify(output)
    (output / "LICENSE").write_text("changed")
    with pytest.raises(ValueError, match="attribution"):
        verify(output)
    with pytest.raises(ValueError, match="attribution"):
        attach_attribution(output, source)
