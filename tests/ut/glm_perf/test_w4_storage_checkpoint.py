# SPDX-License-Identifier: Apache-2.0
"""Lossless code widening, memory admission and permanent loader integration."""

import ast
import json
import shutil
import struct
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from tools.glm_perf import native_checkpoint as checkpoint
from tools.glm_perf import w4_storage_checkpoint as storage
from tools.glm_perf.fused_weight_layout import PREROUNDED_SCALE_LAYOUT, pack_cube, unpack_cube
from tools.glm_perf.prerounded_scale_checkpoint import export as export_scales
from tools.glm_perf.w4_storage_probe import checkpoint_weights
from vllm_ascend.models.glm5next_w2.model import _PackedW2ExpertBank


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    config = dict(
        ascend_glm_expert_layout=checkpoint.LAYOUT,
        text_config=dict(n_routed_experts=2, hidden_size=256, moe_intermediate_size=256),
    )
    checkpoint.write_json(root / "config.json", config)
    codes, dense = {}, {"lm_head.weight": torch.ones(2, 2)}
    generator = torch.Generator().manual_seed(37)
    for layer in (3, 4):
        for expert in range(2):
            for projection in ("gate", "up", "down"):
                prefix = f"model.language_model.layers.{layer}.mlp.experts.{expert}.{projection}_proj"
                bits = 2 if layer == 4 else 3
                values = torch.randint(
                    -(1 << (bits - 1)), 1 << (bits - 1), (1, 256, 256), dtype=torch.int8, generator=generator
                )
                codes[prefix + "_codes"] = pack_cube(values, bits)[0].view(torch.uint8)
                dense[prefix + "_scale"] = torch.rand(8, 8, generator=generator)
    save_file(codes, str(root / "codes.safetensors"), metadata={"layout": checkpoint.LAYOUT})
    save_file(dense, str(root / "dense.safetensors"))
    checkpoint.write_json(
        root / checkpoint.INDEX,
        dict(
            metadata={},
            weight_map=dict.fromkeys(codes, "codes.safetensors") | dict.fromkeys(dense, "dense.safetensors"),
        ),
    )
    bundle = root / "native-kernels"
    bundle.mkdir()
    options = dict(
        fused_moe=True, prepared_weight_layout=True, route_packed_input=True, version=956, helper_package="helpers"
    )
    provenance = {"_build": options, "_helpers": {}}
    for name in (
        "glm_reconstruction_bridge_v956.so",
        "glm_fused_gate_up.bin",
        "glm_fused_down.bin",
        "glm_fused_pack.bin",
        "glm_fused_reduce.bin",
        "glm_fused_route_input.bin",
    ):
        (bundle / name).write_bytes(name.encode())
        key = "reconstruction_bridge.cpp" if name.endswith(".so") else name
        provenance[key] = dict(binary_sha256=checkpoint.file_digest(bundle / name))
    checkpoint.write_json(bundle / "provenance.json", provenance)
    indexer = root / "indexer"
    indexer.mkdir()
    (indexer / "candidate.py").write_bytes(b"frozen indexer fixture")
    checkpoint.write_json(
        root / checkpoint.MANIFEST,
        dict(
            schema_version=1,
            complete=True,
            layout=checkpoint.LAYOUT,
            world_size=2,
            num_experts=2,
            layers=[3, 4],
            kernel_bundle="native-kernels",
            kernel_provenance_sha256=checkpoint.file_digest(bundle / "provenance.json"),
            indexer_kernel_bundle="indexer",
            native_shards=[dict(file="codes.safetensors")],
        ),
    )
    return root


@pytest.mark.parametrize("bits", [2, 3])
@pytest.mark.parametrize("dtype", [torch.int8, torch.uint8])
@pytest.mark.parametrize("shape", [(1, 128, 256), (2, 256, 512)])
def test_every_signed_code_and_physical_tile_matches_independent_reference(bits, dtype, shape):
    values = (
        ((torch.arange(torch.tensor(shape).prod()) % (1 << bits)) - (1 << (bits - 1))).to(torch.int8).reshape(shape)
    )
    old = pack_cube(values, bits).view(dtype)
    snapshot = old.clone()
    promoted = storage.promote(old, shape[-1])
    assert promoted.dtype == dtype and promoted.is_contiguous()
    assert torch.equal(promoted, pack_cube(values, 4).view(dtype))
    assert torch.equal(unpack_cube(promoted, shape[-1]), values)
    assert torch.equal(old, snapshot) and promoted.data_ptr() != old.data_ptr()


@pytest.mark.parametrize(
    "bad,k",
    [
        (torch.zeros(1, 128, 128, dtype=torch.uint8), 256),
        (torch.zeros(1, 128, 96), 256),
        (torch.zeros(1, 64, 96, dtype=torch.uint8), 256),
        (torch.empty(1, 128, 96, dtype=torch.uint8, device="meta"), 256),
    ],
)
def test_invalid_promotion_fails(bad, k):
    with pytest.raises(ValueError):
        storage.promote(bad, k)


def test_plan_uses_headers_only_and_exact_whole_bank_memory(source, monkeypatch):
    monkeypatch.setattr(storage, "safe_open", lambda *a, **kw: pytest.fail("planning mapped tensor payload"))
    inventory = storage.plan(source, extra_bytes_per_rank=0)
    assert len(inventory["banks"]) == 4 and sum(row["eligible"] for row in inventory["banks"]) == 2
    assert inventory["additional_bytes_per_rank"] == [0, 0]
    proposal = storage.plan(source, ["layers.3.gate_up", "layers.3.down"], extra_bytes_per_rank=24576)
    assert proposal["additional_bytes_per_rank"] == [24576, 24576] and proposal["fits_allowance"]
    assert proposal["new_payload_bytes"] == 6 * 256 * 256 // 2
    assert not proposal["precision_changed"] and not proposal["cube_operation_count_changed"]
    assert not storage.plan(source, ["layers.3.gate_up"], extra_bytes_per_rank=16383)["fits_allowance"]
    assert storage.plan(source, ["layers.4.down"], extra_bytes_per_rank=16384, source_bits=2)["fits_allowance"]


@pytest.mark.parametrize(
    "banks,budget,bits",
    [([], 99999, 3), (["layers.3.gate_up"], 16383, 3), (["layers.3.gate"], 99999, 3), (["layers.4.down"], 99999, 3)],
)
def test_invalid_selection_fails_before_output(source, tmp_path, banks, budget, bits):
    output = tmp_path / "invalid"
    with pytest.raises(ValueError):
        storage.export(source, output, banks, extra_bytes_per_rank=budget, source_bits=bits)
    assert not output.exists()


@pytest.mark.parametrize("budget,bits", [(-1, 3), (True, 3), (1.5, 3), (0, 4), (0, 3.0)])
def test_invalid_budget_and_width_are_rejected(source, budget, bits):
    with pytest.raises(ValueError):
        storage.plan(source, extra_bytes_per_rank=budget, source_bits=bits)


def test_export_preserves_source_scales_indexer_and_kernel_contract(source, tmp_path):
    hashes = {
        str(path.relative_to(source)): checkpoint.file_digest(path) for path in source.rglob("*") if path.is_file()
    }
    output = tmp_path / "widened"
    manifest = storage.export(source, output, ["layers.3.gate_up"], extra_bytes_per_rank=16384)
    assert manifest["complete"] and manifest["hardware_validation"] == "not_run"
    assert len(manifest["storage_promotion_shards"]) == 2
    index = json.loads((output / checkpoint.INDEX).read_text())["weight_map"]
    for name, filename in index.items():
        with safe_open(str(output / filename), framework="pt", device="cpu") as handle:
            actual = handle.get_tensor(name)
        original_file = "codes.safetensors" if name.endswith("_codes") else "dense.safetensors"
        with safe_open(str(source / original_file), framework="pt", device="cpu") as handle:
            original = handle.get_tensor(name)
        if ".layers.3." in name and name.endswith(("gate_proj_codes", "up_proj_codes")):
            assert actual.shape == (256, 128)
            assert torch.equal(unpack_cube(actual[None], 256), unpack_cube(original[None], 256))
        else:
            assert torch.equal(actual, original) and filename == original_file
    for name in ("codes.safetensors", "dense.safetensors"):
        assert (output / name).stat().st_ino == (source / name).stat().st_ino
    for relative, sha in hashes.items():
        assert checkpoint.file_digest(source / relative) == sha
        if relative.startswith(("native-kernels/", "indexer/")):
            assert checkpoint.file_digest(output / relative) == sha
    with pytest.raises(FileExistsError):
        storage.export(source, output, ["layers.3.gate_up"], extra_bytes_per_rank=16384)


def loader_iterator():
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/model_loader/glm_native_int4.py"
    owner = next(node for node in ast.parse(path.read_text()).body if isinstance(node, ast.ClassDef))
    method = next(
        node for node in owner.body if isinstance(node, ast.FunctionDef) and node.name == "_get_weights_iterator"
    )
    namespace = dict(
        Path=Path,
        json=json,
        time=time,
        safe_open=safe_open,
        torch=torch,
        FP16_SCALE_LAYOUT="fp16_storage_fp32_compute_v1",
        file_digest=checkpoint.file_digest,
        selected_weights=checkpoint.selected_weights,
        INDEX=checkpoint.INDEX,
        LAYOUT=checkpoint.LAYOUT,
    )
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["_get_weights_iterator"]


@pytest.mark.parametrize("rank,draft", [(0, False), (1, False), (0, True), (1, True)])
def test_actual_loader_and_resident_bank_accept_both_selected_and_unchanged_widths(source, tmp_path, rank, draft):
    output = tmp_path / "loader"
    manifest = storage.export(source, output, ["layers.3.gate_up"], extra_bytes_per_rank=16384)
    worker = SimpleNamespace(
        folder=output,
        native_manifest=manifest,
        counter_before_loading_weights=0.0,
        native_code_tensors=0,
        local_range=(rank, rank + 1),
        layer_keys=("layers.3",),
        draft=draft,
    )
    values = dict(loader_iterator()(worker, SimpleNamespace(model_or_path=str(output), prefix="", subfolder=None)))
    assert worker.native_code_tensors == 3
    bank = _PackedW2ExpertBank(
        layer_key="layers.3",
        num_experts=2,
        num_local_experts=1,
        local_expert_offset=rank,
        hidden=256,
        inter=256,
        offload_to_cpu=False,
        nz_packed_codes=False,
    )
    for projection in ("gate", "up", "down"):
        prefix = f"model.language_model.layers.3.mlp.experts.{rank}.{projection}_proj"
        bank.place_resident_tensor(rank, projection + "_packed", values[prefix + "_codes"], device="cpu")
        bank.place_resident_tensor(rank, projection + "_scale", values[prefix + "_scale"], device="cpu")
    assert bank.gate_up_packed_bank.shape == (1, 512, 128)
    assert bank.down_packed_bank.shape == (1, 256, 96)
    address = bank.gate_up_packed_bank.data_ptr()
    bank.finalize_grouped_storage()
    assert bank.grouped_ready and bank.gate_up_packed_bank.data_ptr() == address
    assert torch.equal(
        bank[rank].gate_packed, values[f"model.language_model.layers.3.mlp.experts.{rank}.gate_proj_codes"]
    )


def test_interrupted_export_retains_incomplete_marker(source, tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "promote", lambda *a: (_ for _ in ()).throw(ValueError("interrupted")))
    output = tmp_path / "partial"
    with pytest.raises(ValueError, match="interrupted"):
        storage.export(source, output, ["layers.3.down"], extra_bytes_per_rank=8192)
    assert json.loads((output / checkpoint.MANIFEST).read_text())["complete"] is False


def test_changed_kernel_rejected_before_export(source, tmp_path):
    (source / "native-kernels/glm_fused_route_input.bin").write_bytes(b"changed producer")
    output = tmp_path / "bad-kernel"
    with pytest.raises(ValueError, match="provenance"):
        storage.export(source, output, ["layers.3.down"], extra_bytes_per_rank=8192)
    assert not output.exists()


@pytest.mark.parametrize("raw", [b"", struct.pack("<Q", storage.MAX_HEADER_BYTES + 1), struct.pack("<Q", 100) + b"{}"])
def test_truncated_or_overlong_headers_rejected(tmp_path, raw):
    path = tmp_path / "bad.safetensors"
    path.write_bytes(raw)
    with pytest.raises(ValueError):
        storage.header(path)


def test_truncated_code_payload_rejected_before_output(source):
    path = source / "codes.safetensors"
    path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(ValueError, match="truncated"):
        storage.plan(source, extra_bytes_per_rank=0)


def test_loader_rejects_changed_promoted_code_bytes(source, tmp_path):
    output = tmp_path / "corrupt"
    manifest = storage.export(source, output, ["layers.3.gate_up"], extra_bytes_per_rank=16384)
    worker = SimpleNamespace(
        folder=output,
        native_manifest=manifest,
        counter_before_loading_weights=0.0,
        native_code_tensors=0,
        local_range=(0, 1),
        layer_keys=("layers.3",),
        draft=False,
    )
    path = output / manifest["storage_promotion_shards"][0]["file"]
    original = path.read_bytes()
    path.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
    with pytest.raises(ValueError, match="W4 storage shard.*checksum"):
        dict(loader_iterator()(worker, SimpleNamespace(model_or_path=str(output), prefix="", subfolder=None)))


def test_real_fixture_follows_promoted_index_and_both_local_ranks(source, tmp_path):
    output = tmp_path / "fixture"
    storage.export(source, output, ["layers.3.gate_up"], extra_bytes_per_rank=16384)
    manifest = json.loads((output / checkpoint.MANIFEST).read_text())
    manifest["world_size"] = 1
    checkpoint.write_json(output / checkpoint.MANIFEST, manifest)
    codes, scales = checkpoint_weights(output, 3, experts=2)
    assert codes[0].shape == (2, 512, 128) and codes[1].shape == (2, 256, 96)
    assert scales[0].shape == (2, 16, 8) and scales[1].shape == (2, 8, 8)
    for layer, first, experts in ((3, 0, 1), (3, 1, 2), (5, 0, 2)):
        with pytest.raises(ValueError):
            checkpoint_weights(output, layer, first, experts)
    with pytest.raises(ValueError, match="resident expert rank"):
        checkpoint_weights(source, 3, experts=2)


def test_widening_composes_with_permanently_prerounded_scales(source, tmp_path):
    bundle = tmp_path / "rounded-kernels"
    shutil.copytree(source / "native-kernels", bundle)
    provenance = json.loads((bundle / "provenance.json").read_text())
    provenance["_build"]["prerounded_weight_scales"] = True
    checkpoint.write_json(bundle / "provenance.json", provenance)
    rounded = tmp_path / "rounded"
    parent = export_scales(source, rounded, bundle)
    output = tmp_path / "rounded-widened"
    manifest = storage.export(rounded, output, ["layers.3.down"], extra_bytes_per_rank=8192)
    assert manifest["weight_scale_layout"] == PREROUNDED_SCALE_LAYOUT
    assert manifest["scale_shards"] == parent["scale_shards"]
    for row in manifest["scale_shards"]:
        assert checkpoint.file_digest(output / row["file"]) == row["sha256"]
    assert manifest["kernel_provenance_sha256"] == parent["kernel_provenance_sha256"]


def test_incremental_different_layer_export_keeps_prior_code_integrity_records(source, tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    parent = storage.export(source, first, ["layers.3.down"], extra_bytes_per_rank=8192)
    manifest = storage.export(first, second, ["layers.4.down"], extra_bytes_per_rank=16384, source_bits=2)
    assert manifest["storage_promotion_shards"][:2] == parent["storage_promotion_shards"]
    assert len(manifest["storage_promotion_shards"]) == 4
    assert json.loads((second / checkpoint.INDEX).read_text())["metadata"]["lossless_w4_storage_banks"] == [
        "layers.3.down",
        "layers.4.down",
    ]


def test_mixed_gate_and_up_widths_rejected_before_allocation(source):
    path = source / "codes.safetensors"
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        tensors = {name: handle.get_tensor(name).clone() for name in tuple(handle.keys())}
    name = "model.language_model.layers.3.mlp.experts.0.up_proj_codes"
    tensors[name] = storage.promote(tensors[name][None], 256)[0]
    save_file(tensors, str(path), metadata={"layout": checkpoint.LAYOUT})
    with pytest.raises(ValueError, match="mixed widths"):
        storage.plan(source, extra_bytes_per_rank=99999)


@pytest.mark.parametrize("change", [{"complete": False}, {"schema_version": 2}, {"layers": [3]}])
def test_incomplete_or_mismatching_manifest_rejected(source, change):
    path = source / checkpoint.MANIFEST
    manifest = json.loads(path.read_text()) | change
    checkpoint.write_json(path, manifest)
    with pytest.raises(ValueError):
        storage.plan(source, extra_bytes_per_rank=99999)
