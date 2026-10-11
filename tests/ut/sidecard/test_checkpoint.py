# SPDX-License-Identifier: Apache-2.0
"""Header-only ownership planning, using sparse fixtures without model payloads."""

import json
import math
import os
import shutil
import struct

import pytest

from tools.sidecard.checkpoint import INDEX_NAME, MAX_HEADER_BYTES, plan_checkpoint, read_header
from tools.sidecard.placement import glm_placement


def write_sparse(path, header):
    raw = json.dumps(header).encode()
    payload_bytes = max(
        (entry["data_offsets"][1] for name, entry in header.items() if name != "__metadata__"), default=0
    )
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", len(raw)))
        handle.write(raw)
        handle.truncate(8 + len(raw) + payload_bytes)


@pytest.mark.parametrize("failure", ["negative", "dtype", "size", "overlap", "outside", "shape"])
def test_bad_safetensors_header(failure, tmp_path):
    tensor = {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}
    header = {"a": tensor}
    if failure == "negative":
        tensor["data_offsets"] = [-4, 0]
    elif failure == "dtype":
        tensor["dtype"] = "UNKNOWN"
    elif failure == "size":
        tensor["shape"] = [2]
    elif failure == "overlap":
        header["b"] = dict(tensor)
    elif failure == "shape":
        tensor["shape"] = [True]
    path = tmp_path / "bad.safetensors"
    write_sparse(path, header)
    if failure == "outside":
        with path.open("r+b") as handle:
            handle.truncate(path.stat().st_size - 1)
    with pytest.raises(ValueError):
        read_header(path)


def test_oversized_header_is_rejected_before_reading_it(tmp_path):
    path = tmp_path / "bad.safetensors"
    path.write_bytes(struct.pack("<Q", MAX_HEADER_BYTES + 1))
    with pytest.raises(ValueError, match="header length"):
        read_header(path)


def test_duplicate_header_names_are_rejected(tmp_path):
    path = tmp_path / "duplicate.safetensors"
    raw = b'{"a":{"dtype":"F32","shape":[0],"data_offsets":[0,0]},"a":{"dtype":"F32","shape":[0],"data_offsets":[0,0]}}'
    path.write_bytes(struct.pack("<Q", len(raw)) + raw)
    with pytest.raises(ValueError, match="duplicate"):
        read_header(path)


@pytest.fixture(scope="module")
def canonical_source(tmp_path_factory):
    source = tmp_path_factory.mktemp("canonical-sidecard")
    config = dict(
        model_type="glm5_next_text",
        num_hidden_layers=45,
        first_k_dense_replace=3,
        n_routed_experts=288,
        num_experts_per_tok=8,
        hidden_size=4096,
        moe_intermediate_size=2048,
        routed_scaling_factor=2.5,
    )
    (source / "config.json").write_text(json.dumps(config))
    filename = "weights.safetensors"
    header, index = {}, {}
    offset = 0

    def add(name, dtype, shape, element_bytes):
        nonlocal offset
        stop = offset + math.prod(shape) * element_bytes
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, stop]}
        index[name] = filename
        offset = stop

    for layer in range(3, 45):
        for expert in range(288):
            for projection in ("gate", "up", "down"):
                rows, columns = (4096, 2048) if projection == "down" else (2048, 4096)
                prefix = f"model.language_model.layers.{layer}.mlp.experts.{expert}.{projection}_proj"
                add(prefix + "_codes", "U8", [rows, columns // 4], 1)
                add(prefix + "_scale", "F32", [rows // 32, columns // 32], 4)
    add("model.language_model.embed_tokens.weight", "F16", [1, 4096], 2)
    add("model.language_model.layers.45.mlp.experts.0.gate_proj_codes", "U8", [1], 1)
    add("model.visual.weight", "F16", [1], 2)
    write_sparse(source / filename, header)
    (source / INDEX_NAME).write_text(json.dumps({"weight_map": index}))
    return source


def test_full_canonical_plan_counts_source_bytes_and_arbitrary_local_ids(canonical_source):
    plan = plan_checkpoint(canonical_source, glm_placement(), include_tensor_slices=True)
    cuda = plan["owners"]["cuda:0"]
    npu = plan["owners"]["npu:0"]
    assert cuda["expert_tensor_count"] == 42 * 72 * 6
    assert npu["expert_tensor_count"] == 42 * 54 * 6
    assert npu["layers"]["3"] == list(range(72, 126))
    assert cuda["tensor_count"] == cuda["expert_tensor_count"] + 1
    assert (
        sum(owner["source_tensor_bytes"] for owner in plan["owners"].values())
        == 42 * 288 * 3 * (4096 * 2048 // 4 + 4096 * 2048 // 1024 * 4) + 8192
    )
    assert plan["excluded_tensor_counts"] == {"vision": 1, "draft": 1}
    assert plan["payload_checksums_verified"] is False
    assert plan["native_layout_exported"] is False
    assert plan["capacity_admitted"] is False
    record = npu["tensor_slices"][0]
    assert record["file_offsets"][0] >= plan["shards"][record["shard"]]["payload_offset"]


@pytest.mark.parametrize("failure", ["missing", "absent", "escape"])
def test_incomplete_or_unsafe_checkpoint_plan(canonical_source, tmp_path, failure):
    shutil.copyfile(canonical_source / "config.json", tmp_path / "config.json")
    os.link(canonical_source / "weights.safetensors", tmp_path / "weights.safetensors")
    index = json.loads((canonical_source / INDEX_NAME).read_text())
    name = next(iter(index["weight_map"]))
    if failure == "missing":
        del index["weight_map"][name]
    elif failure == "absent":
        index["weight_map"]["model.language_model.nonexistent.weight"] = "weights.safetensors"
    else:
        index["weight_map"][name] = "../outside.safetensors"
    (tmp_path / INDEX_NAME).write_text(json.dumps(index))
    with pytest.raises(ValueError):
        plan_checkpoint(tmp_path, glm_placement())


@pytest.mark.parametrize("nested", [False, True])
def test_native_layout_is_not_silently_reclassified_as_canonical(canonical_source, tmp_path, nested):
    config = json.loads((canonical_source / "config.json").read_text())
    config["ascend_glm_expert_layout"] = "cube_n128_k256_v1"
    if nested:
        config = {"text_config": config}
    (tmp_path / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="native export adapter"):
        plan_checkpoint(tmp_path, glm_placement())
