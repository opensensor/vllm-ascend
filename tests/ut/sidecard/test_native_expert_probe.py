# SPDX-License-Identifier: Apache-2.0
"""Native single-expert admission without device initialization."""

import json
import math
import struct

import pytest

from tools.sidecard.native_expert_probe import NATIVE_LAYOUT, SCALE_LAYOUT, contained, select_expert


def native_fixture(root, bits=4, failure=None):
    config = dict(
        model_type="glm5_next_text",
        num_hidden_layers=45,
        first_k_dense_replace=3,
        n_routed_experts=288,
        num_experts_per_tok=8,
        hidden_size=4096,
        moe_intermediate_size=2048,
        routed_scaling_factor=2.5,
        ascend_glm_expert_layout=NATIVE_LAYOUT,
        ascend_glm_expert_scale_layout=SCALE_LAYOUT,
    )
    if failure == "config":
        config["ascend_glm_expert_layout"] = "canonical"
    (root / "config.json").write_text(json.dumps(config))
    filename = "native.safetensors"
    record = {"file": filename, "sha256": "a" * 64}
    manifest = {
        "complete": failure != "incomplete",
        "schema_version": 1,
        "layout": NATIVE_LAYOUT,
        "weight_scale_layout": SCALE_LAYOUT,
        "layer_bits": {"3": bits},
        "activation_bits": 4,
        "native_shards": [record],
        "scale_shards": [dict(record)],
        "kernel_bundle": "native-kernels",
        "kernel_provenance_sha256": "b" * 64,
    }
    if failure == "receipt":
        manifest["scale_shards"][0]["sha256"] = "c" * 64
    (root / "native-layout.json").write_text(json.dumps(manifest))
    header = {"__metadata__": {"layout": NATIVE_LAYOUT, "weight_scale_layout": SCALE_LAYOUT}}
    if failure == "metadata":
        header["__metadata__"]["layout"] = "canonical"
    index, offset = {}, 0
    for projection in ("gate", "up", "down"):
        rows, columns = (4096, 2048) if projection == "down" else (2048, 4096)
        for kind in ("codes", "scale"):
            name = f"model.language_model.layers.3.mlp.experts.72.{projection}_proj_{kind}"
            shape = [rows, columns * bits // 8] if kind == "codes" else [rows // 32, columns // 32]
            dtype = "I8" if kind == "codes" else "F16"
            size = math.prod(shape) * (1 if kind == "codes" else 2)
            if failure == "dtype" and kind == "scale":
                dtype, size = "F32", size * 2
            if failure == "shape" and kind == "codes":
                shape[0] //= 2
                size //= 2
            header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + size]}
            offset += size
            index[name] = filename
    (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": index}))
    raw = json.dumps(header).encode()
    with (root / filename).open("wb") as handle:
        handle.write(struct.pack("<Q", len(raw)))
        handle.write(raw)
        handle.truncate(8 + len(raw) + offset)


@pytest.mark.parametrize("bits", [2, 3, 4])
def test_native_probe_header_admits_all_packed_precisions_without_claiming_hardware(tmp_path, bits):
    native_fixture(tmp_path, bits)
    plan = select_expert(tmp_path, 3, 72)
    assert plan["bits"] == bits
    assert len(plan["tensor_files"]) == 6
    assert not plan["expert_math_validated"]
    assert not plan["complete_owner_bank_loaded"]


@pytest.mark.parametrize("failure", ["config", "incomplete", "receipt", "metadata", "dtype", "shape"])
def test_native_probe_rejects_bad_layout_receipts_or_geometry(tmp_path, failure):
    native_fixture(tmp_path, failure=failure)
    with pytest.raises(ValueError):
        select_expert(tmp_path, 3, 72)


@pytest.mark.parametrize("layer,expert", [(45, 72), (2, 72), (True, 72), (3, 288), (3, -1), (3, True)])
def test_native_probe_excludes_draft_and_bad_expert_identity(tmp_path, layer, expert):
    native_fixture(tmp_path)
    with pytest.raises(ValueError):
        select_expert(tmp_path, layer, expert)


def test_native_probe_rejects_paths_and_symlinks_outside_checkpoint(tmp_path):
    root = tmp_path / "checkpoint"
    root.mkdir()
    (root / "linked").symlink_to(tmp_path / "outside")
    with pytest.raises(ValueError, match="escapes"):
        contained(root, "../outside")
    with pytest.raises(ValueError, match="escapes"):
        contained(root, "linked")
