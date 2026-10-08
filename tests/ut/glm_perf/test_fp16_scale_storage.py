# SPDX-License-Identifier: Apache-2.0
"""Permanent FP16 scales: exact FP32 promotion, admission and resident storage."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from tests.ut.glm_perf import test_prerounded_scale_checkpoint as rounded
from tools.glm_perf import build_reconstruction as builder
from tools.glm_perf import native_checkpoint as checkpoint
from tools.glm_perf.fused_weight_layout import FP16_SCALE_LAYOUT, PREROUNDED_SCALE_LAYOUT
from tools.glm_perf.glm_fused_moe import FusedGeometry, NativeFusedMoE
from tools.glm_perf.prerounded_scale_checkpoint import export, round_scales
from tools.glm_perf.resident_candidates.prefill_decode import PrefillDecodeNative
from vllm_ascend.models.glm5next_w2.model import _PackedW2ExpertBank


@pytest.fixture
def source_bundle(tmp_path):
    source, bundle, index, scales = rounded.source_bundle.__wrapped__(tmp_path)
    path = bundle / "provenance.json"
    provenance = json.loads(path.read_text())
    provenance["_build"].update(prerounded_weight_scales=False, fp16_weight_scales=True)
    checkpoint.write_json(path, provenance)
    return source, bundle, index, scales


def test_every_finite_half_pattern_promotes_exactly_including_signed_zero():
    values = torch.arange(-32768, 32768, dtype=torch.int32).to(torch.int16).view(torch.float16)
    values[~torch.isfinite(values)] = 0
    values = values.reshape(256, 256)
    compact = round_scales(values.float(), fp16_storage=True)
    assert compact.dtype == torch.float16
    assert compact.element_size() == 2
    assert torch.equal(compact.view(torch.int16), values.view(torch.int16))
    assert torch.equal(compact.float().view(torch.int32), round_scales(values.float()).view(torch.int32))
    assert torch.equal(round_scales(compact, True).view(torch.int16), compact.view(torch.int16))


@pytest.mark.parametrize("value", [torch.full((2, 2), float("nan")), torch.full((2, 2), 1e10), torch.ones(3)])
def test_invalid_scales_never_export(value):
    with pytest.raises(ValueError):
        round_scales(value, True)


def test_export_and_actual_loader_preserve_fp16_storage_and_exact_promoted_values(source_bundle, tmp_path):
    source, bundle, old_index, originals = source_bundle
    source_hashes = {path.name: checkpoint.file_digest(path) for path in source.iterdir()}
    output = tmp_path / "half-scales"
    manifest = export(source, output, bundle, fp16_storage=True)
    index = json.loads((output / checkpoint.INDEX).read_text())
    assert manifest["weight_scale_layout"] == FP16_SCALE_LAYOUT
    assert manifest["scale_storage_dtype"] == "float16" and manifest["scale_compute_dtype"] == "float32"
    assert manifest["scale_tensor_bytes"] == sum(
        value.numel() * 2 for name, value in originals.items() if name.endswith("_scale")
    )
    assert manifest["hardware_validation"] == "not_run" and manifest["transformed_code_tensors"] == 0
    assert json.loads((output / "config.json").read_text())["ascend_glm_expert_scale_layout"] == FP16_SCALE_LAYOUT
    assert index["metadata"]["expert_scale_layout"] == FP16_SCALE_LAYOUT
    for name, old_file in old_index["weight_map"].items():
        if not name.endswith("_scale"):
            assert index["weight_map"][name] == old_file
            assert (output / old_file).stat().st_ino == (source / old_file).stat().st_ino
            continue
        with safe_open(str(output / index["weight_map"][name]), framework="pt", device="cpu") as handle:
            actual = handle.get_tensor(name)
            assert actual.dtype == torch.float16
            assert handle.metadata()["weight_scale_layout"] == FP16_SCALE_LAYOUT
            assert torch.equal(actual.float().view(torch.int32), originals[name].half().float().view(torch.int32))
    worker = SimpleNamespace(
        folder=output,
        native_manifest=manifest,
        counter_before_loading_weights=0.0,
        native_code_tensors=0,
        local_range=(0, 1),
        layer_keys=("layers.3",),
        draft=False,
    )
    values = dict(
        rounded.loader_iterator()(worker, SimpleNamespace(model_or_path=str(output), prefix="", subfolder=None))
    )
    bank = _PackedW2ExpertBank(256, 256, 2, local_expert_offset=0, num_local_experts=1, offload_to_cpu=False)
    for projection in ("gate", "up", "down"):
        name = f"model.language_model.layers.3.mlp.experts.0.{projection}_proj_scale"
        bank.place_resident_tensor(0, projection + "_scale", values[name], device="cpu")
    assert bank.gate_up_scale_bank.dtype == torch.float16 and bank.down_scale_bank.dtype == torch.float16
    assert bank[0].gate_scale.data_ptr() == bank.gate_up_scale_bank[0].data_ptr()
    assert sum(value.numel() * value.element_size() for value in (bank.gate_up_scale_bank, bank.down_scale_bank)) == 384
    assert all(checkpoint.file_digest(source / name) == sha for name, sha in source_hashes.items())


@pytest.mark.parametrize("source_rounded", [False, True])
def test_export_from_raw_or_prerounded_fp32_is_identical(source_bundle, tmp_path, source_rounded):
    source, bundle, _, originals = source_bundle
    if source_rounded:
        provenance = json.loads((bundle / "provenance.json").read_text())
        provenance["_build"].update(prerounded_weight_scales=True, fp16_weight_scales=False)
        checkpoint.write_json(bundle / "provenance.json", provenance)
        export(source, tmp_path / "rounded", bundle)
        source = tmp_path / "rounded"
        provenance["_build"].update(prerounded_weight_scales=False, fp16_weight_scales=True)
        checkpoint.write_json(bundle / "provenance.json", provenance)
    output = tmp_path / "compact"
    export(source, output, bundle, fp16_storage=True)
    index = json.loads((output / checkpoint.INDEX).read_text())
    for name, expected in originals.items():
        if name.endswith("_scale"):
            with safe_open(str(output / index["weight_map"][name]), framework="pt", device="cpu") as handle:
                assert torch.equal(handle.get_tensor(name).view(torch.int16), expected.half().view(torch.int16))


@pytest.mark.parametrize("marker,enabled", [(None, True), (PREROUNDED_SCALE_LAYOUT, True), (FP16_SCALE_LAYOUT, False)])
def test_checkpoint_and_kernel_scale_abis_cannot_be_crossed(marker, enabled):
    manifest = dict(weight_scale_layout=marker)
    with pytest.raises(ValueError, match="matching checkpoint and kernel"):
        checkpoint.validate_scale_contract(manifest, marker, {"fp16_weight_scales": enabled})


def test_loader_rejects_fp32_payload_under_fp16_marker_even_with_matching_checksum(source_bundle, tmp_path):
    source, bundle, _, _ = source_bundle
    output = tmp_path / "wrong-dtype"
    manifest = export(source, output, bundle, fp16_storage=True)
    record = manifest["scale_shards"][0]
    path = output / record["file"]
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        tensors = {name: handle.get_tensor(name).float() for name in handle.keys()}  # noqa: SIM118 -- safe_open is not a dict.
    save_file(tensors, str(path), metadata={"weight_scale_layout": FP16_SCALE_LAYOUT})
    record["sha256"] = checkpoint.file_digest(path)
    worker = SimpleNamespace(
        folder=output,
        native_manifest=manifest,
        counter_before_loading_weights=0.0,
        native_code_tensors=0,
        local_range=(0, 1),
        layer_keys=("layers.3",),
        draft=False,
    )
    with pytest.raises(ValueError, match="dtype differs"):
        dict(rounded.loader_iterator()(worker, SimpleNamespace(model_or_path=str(output), prefix="", subfolder=None)))


@pytest.mark.parametrize("fp16", [False, True])
def test_actual_pipeline_guards_scale_dtype_before_any_kernel_submission(fp16):
    native = NativeFusedMoE.__new__(NativeFusedMoE)
    native.fp16_weight_scales = fp16
    assert native.required_weight_scale_layout == (FP16_SCALE_LAYOUT if fp16 else None)
    native.launch = lambda *args: pytest.fail("wrong dtype submitted")
    wrong = torch.ones(1, dtype=torch.float32 if fp16 else torch.float16)
    geometry = FusedGeometry(2, 1, 1, 256, 256, 4, 4, 4)
    with pytest.raises(ValueError, match="dtype differs"):
        native.grouped(None, None, wrong, None, wrong, None, None, None, geometry)


@pytest.mark.parametrize("flags", [{}, {"prerounded_weight_scales": True}])
def test_builder_rejects_missing_preparation_or_conflicting_scale_formats(tmp_path, flags):
    if flags:
        flags.update(fused_moe=True, all_bits=True, tile_pipeline=True, output_columns=128, prepared_weight_layout=True)
    with pytest.raises(ValueError):
        builder.build(tmp_path / "bad", tmp_path, tmp_path, fp16_weight_scales=True, **flags)
    assert not (tmp_path / "bad").exists()


def test_builder_compiles_fp16_scale_abi_into_generic_w3_and_w4_stages(tmp_path, monkeypatch):
    commands = []

    def fake_compile(command, **kwargs):
        commands.append(command)
        target = command[command.index("-o") + 1] if command[0] == "c++" else command[2]
        Path(target).write_bytes(b"mock; no hardware binary")

    monkeypatch.setattr(builder.subprocess, "run", fake_compile)
    monkeypatch.setattr(
        builder.importlib.util, "find_spec", lambda name: SimpleNamespace(origin=str(tmp_path / "__init__.py"))
    )
    builder.build(
        tmp_path / "built",
        tmp_path,
        tmp_path,
        version=962,
        fused_moe=True,
        prepared_weight_layout=True,
        all_bits=True,
        tile_pipeline=True,
        output_columns=128,
        specialize_w3=True,
        compact_w4_scratch=True,
        fp16_weight_scales=True,
    )
    stages = [command for command in commands if len(command) > 1 and command[1].endswith("glm_fused_moe.cpp")]
    assert len(stages) == 6
    assert all(
        "-DGLM_FP16_WEIGHT_SCALES" in command and "-DGLM_PREROUNDED_WEIGHT_SCALES" not in command for command in stages
    )


def test_mixed_decode_prefill_keeps_the_same_fp16_scale_contract():
    common = dict(
        device="cpu",
        input_dtype=torch.float16,
        activation_bits=4,
        prepared_weight_layout=True,
        fp16_route_workspace=True,
        required_weight_scale_layout=FP16_SCALE_LAYOUT,
        fp16_weight_scales=True,
    )
    decode, prefill = SimpleNamespace(**common), SimpleNamespace(**common)
    mixed = PrefillDecodeNative(prefill, decode)
    assert mixed.required_weight_scale_layout == FP16_SCALE_LAYOUT and mixed.fp16_weight_scales
    assert mixed.selected(16) is decode and mixed.selected(17) is prefill
    decode.required_weight_scale_layout = None
    with pytest.raises(ValueError, match="weight scale storage"):
        PrefillDecodeNative(prefill, decode)
