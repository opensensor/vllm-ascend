# SPDX-License-Identifier: Apache-2.0
"""Permanent native bytes, authoritative index and incomplete export rejection."""

import json
from types import SimpleNamespace

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from tools.deepseek_w2.w2_format import pack_codes
from tools.glm_perf import native_checkpoint as checkpoint
from tools.glm_perf.fused_weight_layout import pack_cube, tensor_digest


def fixture(tmp_path, bits):
    source = tmp_path / "source"
    source.mkdir()
    config = {"text_config": {"n_routed_experts": 2, "hidden_size": 256, "moe_intermediate_size": 256}}
    (source / "config.json").write_text(json.dumps(config))
    tensors = {"lm_head.weight": torch.ones(256, 256).half()}
    projections = []
    for projection in ("gate", "up", "down"):
        signed = torch.randint(-(1 << (bits - 1)), 1 << (bits - 1), (2, 256, 256), dtype=torch.int8)
        projections.append(signed)
        for expert in range(2):
            name = f"model.language_model.layers.3.mlp.experts.{expert}.{projection}_proj"
            tensors[name + "_codes"] = pack_codes(signed[expert], bits)
            tensors[name + "_scale"] = torch.ones(8, 8)
    save_file(tensors, str(source / "original.safetensors"))
    index = {"metadata": {}, "weight_map": dict.fromkeys(tensors, "original.safetensors")}
    (source / checkpoint.INDEX).write_text(json.dumps(index))
    native = [pack_cube(torch.cat(projections[:2], 1), bits), pack_cube(projections[2], bits)]
    output = tmp_path / "output"
    output.mkdir()
    (output / checkpoint.INDEX).write_text(json.dumps(index))
    checkpoint.write_json(
        output / checkpoint.MANIFEST,
        {
            "schema_version": 1,
            "layout": checkpoint.LAYOUT,
            "complete": False,
            "world_size": 1,
            "num_experts": 2,
            "layers": [3],
        },
    )
    layout = SimpleNamespace(
        synchronize=lambda: None,
        backups=[(t, None, None) for t in native],
        receipt={"prepared": True, "bank_byte_hashes": [{"prepared": tensor_digest(t)} for t in native]},
    )
    return source, output, layout, tensors


@pytest.mark.parametrize("bits", (2, 3, 4))
def test_export_uses_prepared_bytes_and_keeps_scales_authoritative(tmp_path, bits):
    source, output, layout, tensors = fixture(tmp_path, bits)
    result = checkpoint.export_rank(layout, source, output, 0, 1)
    assert result["passed"] and result["repacked_banks"] == 0
    manifest = checkpoint.finalize(output)
    assert manifest["complete"] and manifest["native_tensor_count"] == 6
    index = json.loads((output / checkpoint.INDEX).read_text())["weight_map"]
    assert all(index[name] == "original.safetensors" for name in tensors if not name.endswith("_codes"))
    shard = output / "native-rank0-layer003.safetensors"
    with safe_open(str(shard), framework="pt") as handle:
        gate = handle.get_tensor("model.language_model.layers.3.mlp.experts.0.gate_proj_codes")
        torch.testing.assert_close(gate, layout.backups[0][0][0, :256].view(torch.uint8), rtol=0, atol=0)


def test_incomplete_export_never_commits_native_manifest(tmp_path):
    _, output, _, _ = fixture(tmp_path, 4)
    with pytest.raises(FileNotFoundError):
        checkpoint.finalize(output)
    assert not json.loads((output / checkpoint.MANIFEST).read_text())["complete"]


def test_changed_device_bytes_reject_before_export(tmp_path):
    source, output, layout, _ = fixture(tmp_path, 4)
    layout.backups[0][0][0, 0, 0] ^= 1
    with pytest.raises(ValueError, match="differs from prepared"):
        checkpoint.export_rank(layout, source, output, 0, 1)
    assert not list(output.glob("*.safetensors"))


def test_swapped_projection_banks_reject_layer_mapping(tmp_path):
    source, output, layout, _ = fixture(tmp_path, 4)
    layout.backups.reverse()
    layout.receipt["bank_byte_hashes"].reverse()
    with pytest.raises(ValueError, match="geometry"):
        checkpoint.export_rank(layout, source, output, 0, 1)


def test_native_weight_selection_limits_mtp_and_peer_experts():
    index = {
        "model.language_model.layers.3.mlp.experts.0.gate_proj_codes": "old",
        "model.language_model.layers.3.mlp.experts.1.gate_proj_codes": "native",
        "model.language_model.layers.4.mlp.experts.1.gate_proj_codes": "native",
        "model.language_model.layers.4.mlp.experts.1.gate_proj_scale": "scale",
        "model.language_model.layers.4.shared.weight": "dense",
        "lm_head.weight": "dense",
    }
    selected = set(checkpoint.selected_weights(index, (1, 2), ("layers.4",), True))
    assert selected == set(list(index)[2:5])
    assert "lm_head.weight" in set(checkpoint.selected_weights(index, (1, 2), ("layers.3",)))


def test_permanent_native_method_bypasses_legacy_dispatch_and_preserves_shared():
    seen = []
    native = lambda *args: seen.append(args) or args[0].float()
    method = checkpoint.NativeInt4MoEMethod(native)
    bank = SimpleNamespace(
        native_weight_layout=checkpoint.LAYOUT,
        gate_up_packed_bank=object(),
        gate_up_scale_bank=object(),
        down_packed_bank=object(),
        down_scale_bank=object(),
        local_expert_offset=72,
    )
    layer = SimpleNamespace(w2_experts=bank, w2_shared_expert=SimpleNamespace(forward=lambda x: 2 * x))
    x = torch.ones(2, 256)
    result = method.apply(layer, x, torch.ones(2, 1), torch.full((2, 1), 72), None, None)
    assert torch.equal(result, 3 * x) and seen[0][-1] == 72 and seen[0][0].dtype == torch.float16
    bank.native_weight_layout = None
    with pytest.raises(ValueError, match="permanent Cube"):
        method.apply(layer, x, torch.ones(2, 1), torch.full((2, 1), 72), None, None)
