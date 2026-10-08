# SPDX-License-Identifier: Apache-2.0
"""Exact CPU scale arithmetic and complete permanent checkpoint publication."""

import ast
import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from tools.glm_perf import native_checkpoint as checkpoint
from tools.glm_perf.build_reconstruction import build
from tools.glm_perf.fused_weight_layout import FP16_SCALE_LAYOUT, PREROUNDED_SCALE_LAYOUT
from tools.glm_perf.prerounded_scale_checkpoint import export, round_scales
from tools.glm_perf.resident_candidates.expert_reconstruction import wrap_fused_moe


@pytest.fixture
def source_bundle(tmp_path):
    source, bundle = tmp_path / "source", tmp_path / "bundle"
    source.mkdir()
    bundle.mkdir()
    config = {
        "ascend_glm_expert_layout": checkpoint.LAYOUT,
        "text_config": {"n_routed_experts": 2, "hidden_size": 256, "moe_intermediate_size": 256},
    }
    checkpoint.write_json(source / "config.json", config)
    codes, scales = {}, {"lm_head.weight": torch.ones(2, 2)}
    for expert in range(2):
        for projection in ("gate", "up", "down"):
            prefix = f"model.language_model.layers.3.mlp.experts.{expert}.{projection}_proj"
            codes[prefix + "_codes"] = torch.zeros(256, 128, dtype=torch.uint8)
            scales[prefix + "_scale"] = torch.linspace(0.001, 0.01, 64).reshape(8, 8)
    save_file(codes, str(source / "codes.safetensors"), metadata={"layout": checkpoint.LAYOUT})
    save_file(scales, str(source / "dense.safetensors"))
    index = dict(
        metadata={}, weight_map=dict.fromkeys(codes, "codes.safetensors") | dict.fromkeys(scales, "dense.safetensors")
    )
    checkpoint.write_json(source / checkpoint.INDEX, index)
    checkpoint.write_json(
        source / checkpoint.MANIFEST,
        dict(
            schema_version=1,
            complete=True,
            layout=checkpoint.LAYOUT,
            world_size=2,
            num_experts=2,
            layers=[3],
            kernel_bundle="native-kernels",
            native_shards=[{"file": "codes.safetensors"}],
        ),
    )
    options = dict(
        fused_moe=True,
        prepared_weight_layout=True,
        prerounded_weight_scales=True,
        fp16_route_workspace=True,
        route_packed_input=True,
        native_route_columns=True,
        version=956,
        helper_package="fake_helpers",
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
        path = bundle / name
        path.write_bytes(name.encode())
        key = "reconstruction_bridge.cpp" if name.endswith(".so") else name
        provenance[key] = {"binary_sha256": checkpoint.file_digest(path)}
    checkpoint.write_json(bundle / "provenance.json", provenance)
    return source, bundle, index, scales


def test_rounding_is_idempotent_for_every_finite_half_pattern_and_preserves_signed_zero():
    bits = torch.arange(-32768, 32768, dtype=torch.int32).to(torch.int16)
    values = bits.view(torch.float16).float()
    values[~torch.isfinite(values)] = 0
    values = values.reshape(256, 256)
    assert torch.equal(round_scales(values).view(torch.int32), values.view(torch.int32))
    ties = torch.tensor([[0.0, -0.0, 2**-25, 2**-24, 1 + 2**-11, 1 + 3 * 2**-11]])
    assert torch.equal(round_scales(ties).view(torch.int32), ties.half().float().view(torch.int32))


@pytest.mark.parametrize(
    "value",
    [
        torch.ones(2, 2).half(),
        torch.ones(4),
        torch.ones(2, 2, device="meta"),
        torch.full((2, 2), float("nan")),
        torch.full((2, 2), 1e10),
    ],
)
def test_unqualified_or_overflowing_scales_fail(value):
    with pytest.raises(ValueError):
        round_scales(value)


def test_export_changes_only_scale_index_and_preserves_code_and_dense_bytes(source_bundle, tmp_path):
    source, bundle, old_index, originals = source_bundle
    output = tmp_path / "rounded"
    source_hashes = {path.name: checkpoint.file_digest(path) for path in source.iterdir()}
    manifest = export(source, output, bundle)
    assert manifest["complete"] and manifest["scale_tensor_count"] == 6 and len(manifest["scale_shards"]) == 2
    assert manifest["transformed_code_tensors"] == 0 and manifest["hardware_validation"] == "not_run"
    assert (output / "codes.safetensors").stat().st_ino == (source / "codes.safetensors").stat().st_ino
    assert (output / "dense.safetensors").stat().st_ino == (source / "dense.safetensors").stat().st_ino
    index = json.loads((output / checkpoint.INDEX).read_text())
    for name, old_file in old_index["weight_map"].items():
        if name.endswith("_scale"):
            with safe_open(str(output / index["weight_map"][name]), framework="pt", device="cpu") as handle:
                actual = handle.get_tensor(name)
                assert actual.dtype == torch.float32
                assert torch.equal(actual.view(torch.int32), originals[name].half().float().view(torch.int32))
                assert handle.metadata()["weight_scale_layout"] == PREROUNDED_SCALE_LAYOUT
        else:
            assert index["weight_map"][name] == old_file
    assert (output / "native-kernels/glm_fused_route_input.bin").exists()
    assert (output / "native-kernels/glm_fused_reduce.bin").exists()
    assert all(checkpoint.file_digest(source / name) == sha for name, sha in source_hashes.items())
    with pytest.raises(FileExistsError):
        export(source, output, bundle)


def test_failed_scale_export_never_publishes_complete_marker(source_bundle, tmp_path, monkeypatch):
    source, bundle, _, _ = source_bundle
    output = tmp_path / "failed"
    monkeypatch.setattr(
        "tools.glm_perf.prerounded_scale_checkpoint.round_scales",
        lambda value: (_ for _ in ()).throw(ValueError("interrupted")),
    )
    with pytest.raises(ValueError, match="interrupted"):
        export(source, output, bundle)
    assert json.loads((output / checkpoint.MANIFEST).read_text())["complete"] is False


def test_missing_coupled_producer_rejects_before_creating_output(source_bundle, tmp_path):
    source, bundle, _, _ = source_bundle
    (bundle / "glm_fused_route_input.bin").unlink()
    output = tmp_path / "bad"
    with pytest.raises(FileNotFoundError):
        export(source, output, bundle)
    assert not output.exists()


def test_prerounded_kernel_rejects_raw_resident_banks_before_dispatch():
    native = SimpleNamespace(required_weight_scale_layout=PREROUNDED_SCALE_LAYOUT)
    bank = SimpleNamespace(native_weight_layout=checkpoint.LAYOUT)
    for operation in (
        checkpoint.NativeInt4MoEMethod(native)._apply_device_grouped,
        wrap_fused_moe(None, native, {}).__get__(object()),
    ):
        with pytest.raises(ValueError, match="matching permanent weight scales"):
            operation(None, bank, None, None, None, None)


def test_prerounded_builder_requires_prepared_fused_geometry_before_creating_output(tmp_path):
    with pytest.raises(ValueError, match="prepared fused MoE"):
        build(tmp_path / "missing", tmp_path, tmp_path, prerounded_weight_scales=True)
    assert not (tmp_path / "missing").exists()


def test_loader_contract_rejects_wrong_config_raw_scales_and_incomplete_coverage():
    base = dict(num_experts=2, layers=[3], world_size=1)
    with pytest.raises(ValueError, match="permanent rounded scales"):
        checkpoint.validate_scale_contract(base, None, {"prerounded_weight_scales": True})
    with pytest.raises(ValueError, match="markers disagree"):
        checkpoint.validate_scale_contract(base, PREROUNDED_SCALE_LAYOUT, {})
    rounded = dict(
        base,
        weight_scale_layout=PREROUNDED_SCALE_LAYOUT,
        scale_tensor_count=6,
        scale_shards=[{"tensor_count": 6, "layer": 3, "rank": 0, "file": "prepared.safetensors"}],
    )
    assert (
        checkpoint.validate_scale_contract(rounded, PREROUNDED_SCALE_LAYOUT, {"prerounded_weight_scales": True})
        == PREROUNDED_SCALE_LAYOUT
    )
    rounded["scale_tensor_count"] = 5
    with pytest.raises(ValueError, match="tensor count"):
        checkpoint.validate_scale_contract(rounded, PREROUNDED_SCALE_LAYOUT, {})


def loader_iterator():
    # Execute the actual CPU iterator without importing the unavailable vLLM/
    # torch-npu runtime. This verifies authoritative file selection and hashes.
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
        FP16_SCALE_LAYOUT=FP16_SCALE_LAYOUT,
        file_digest=checkpoint.file_digest,
        selected_weights=checkpoint.selected_weights,
        INDEX=checkpoint.INDEX,
        LAYOUT=checkpoint.LAYOUT,
    )
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["_get_weights_iterator"]


def test_actual_loader_reads_only_local_prepared_scale_files_and_rejects_changed_bytes(source_bundle, tmp_path):
    source, bundle, _, _ = source_bundle
    output = tmp_path / "loader"
    manifest = export(source, output, bundle)
    worker = SimpleNamespace(
        folder=output,
        native_manifest=manifest,
        counter_before_loading_weights=0.0,
        native_code_tensors=0,
        local_range=(0, 1),
        layer_keys=("layers.3",),
        draft=False,
    )
    model_source = SimpleNamespace(model_or_path=str(output), prefix="", subfolder=None)
    values = dict(loader_iterator()(worker, model_source))
    assert worker.native_code_tensors == 3
    assert len(values) == 7 and all("experts.1." not in key for key in values)
    assert all(value.dtype == torch.float32 for key, value in values.items() if key.endswith("_scale"))
    path = output / manifest["scale_shards"][0]["file"]
    original = path.read_bytes()
    path.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
    with pytest.raises(ValueError, match="checksum"):
        dict(loader_iterator()(worker, model_source))
