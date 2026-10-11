# SPDX-License-Identifier: Apache-2.0
"""Permanent storage, original-FP8 admission and local partition regression gates."""

import json

import pytest
import torch
from safetensors.torch import save_file

from tools.glm_perf.channel_scale_checkpoint import export, load_partition, read_fp8_projection


def original_source(root):
    root.mkdir()
    (root / "config.json").write_text(
        json.dumps(
            {
                "text_config": {"num_hidden_layers": 4},
                "quantization_config": {"quant_method": "fp8", "weight_block_size": [128, 128]},
            }
        )
    )
    values, index = {}, {}
    for expert in (0, 1):
        for projection in ("gate", "up", "down"):
            prefix = f"model.language_model.layers.3.mlp.experts.{expert}.{projection}_proj"
            weight = (torch.arange(256 * 256).reshape(256, 256) % 17 - 8).float().to(torch.float8_e4m3fn)
            values[prefix + ".weight"] = weight
            values[prefix + ".weight_scale_inv"] = torch.tensor([[0.01, 0.02], [0.03, 0.04]])
    save_file(values, str(root / "source.safetensors"))
    index.update({name: "source.safetensors" for name in values})
    (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": index}))
    return index


def test_original_fp8_blocks_dequantize_in_correct_rows_and_columns(tmp_path):
    root = tmp_path / "original"
    index = original_source(root)
    result = read_fp8_projection(root, index, 3, 0, "gate")
    for row, column, factor in ((0, 0, 0.01), (0, 128, 0.02), (128, 0, 0.03), (128, 128, 0.04)):
        integer = (row * 256 + column) % 17 - 8
        assert result[row, column].item() == pytest.approx(integer * factor)


def test_export_loads_selected_partition_without_transforming_native_storage(tmp_path):
    source = tmp_path / "original"
    original_source(source)
    target = tmp_path / "overlay"
    manifest = export(source, target, [3], [0, 1], 256, 256)
    assert manifest["complete"] and len(manifest["experts"]) == 2
    both = load_partition(target, 3, 0, 2, host_budget_bytes=16 * 1024**2)
    one = load_partition(target, 3, 1, 2, host_budget_bytes=16 * 1024**2)
    for bank, selected in zip(both, one):
        for field in ("codes", "scales", "sums", "smoothing"):
            assert torch.equal(getattr(bank, field)[1:], getattr(selected, field))
    with pytest.raises(ValueError, match="budget"):
        load_partition(target, 3, 0, 2, host_budget_bytes=1)
    filename = target / manifest["experts"]["3:1"]["file"]
    payload = bytearray(filename.read_bytes())
    payload[-1] ^= 1
    filename.write_bytes(payload)
    with pytest.raises(ValueError, match="checksum"):
        load_partition(target, 3, 1, 2, host_budget_bytes=16 * 1024**2)


def test_export_rejects_low_bit_source_and_incomplete_overlay(tmp_path):
    source = tmp_path / "original"
    original_source(source)
    config = json.loads((source / "config.json").read_text())
    config["quantization_config"]["quant_method"] = "w2"
    (source / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="original"):
        export(source, tmp_path / "rejected", [3], [0], 256, 256)
    assert not (tmp_path / "rejected").exists()
    target = tmp_path / "unfinished"
    target.mkdir()
    (target / "manifest.json").write_text(json.dumps({"complete": False}))
    with pytest.raises(ValueError, match="complete"):
        load_partition(target, 3, 0, 1, host_budget_bytes=1024)


def test_resume_uses_durable_journal_without_requantizing_completed_experts(tmp_path, monkeypatch):
    import tools.glm_perf.channel_scale_checkpoint as checkpoint

    source = tmp_path / "original"
    original_source(source)
    target = tmp_path / "overlay"
    expected = export(source, target, [3], [0, 1], 256, 256)
    manifest = {**expected, "complete": False, "experts": {}}
    (target / "manifest.json").write_text(json.dumps(manifest))

    def forbidden(*args, **kwargs):
        pytest.fail("completed expert was requantized instead of admitted from its journal")

    monkeypatch.setattr(checkpoint, "read_fp8_projection", forbidden)
    assert export(source, target, [3], [0, 1], 256, 256, resume=True) == expected
    assert not (target / "manifest.pending").exists()
    with pytest.raises(ValueError, match="identical"):
        export(source, target, [3], [0, 1], 256, 256, clipping=0.9, resume=True)
    index = source / "model.safetensors.index.json"
    index.write_text(index.read_text() + "\n")
    with pytest.raises(ValueError, match="identical"):
        export(source, target, [3], [0, 1], 256, 256, resume=True)


@pytest.mark.parametrize("corruption", ["bytes", "duplicate", "unselected"])
def test_resume_rejects_corrupted_or_ambiguous_journal(tmp_path, corruption):
    source = tmp_path / "original"
    original_source(source)
    target = tmp_path / "overlay"
    manifest = export(source, target, [3], [0], 256, 256)
    journal = target / "completed.jsonl"
    if corruption == "bytes":
        path = target / manifest["experts"]["3:0"]["file"]
        path.write_bytes(path.read_bytes()[:-1] + b"x")
    elif corruption == "duplicate":
        journal.write_text(journal.read_text() * 2)
    else:
        record = json.loads(journal.read_text())
        record["key"] = "4:99"
        journal.write_text(json.dumps(record) + "\n")
    with pytest.raises(ValueError, match="journal"):
        export(source, target, [3], [0], 256, 256, resume=True)


@pytest.mark.parametrize("field,value", [("tensor_bytes", 1), ("gate_shape", [1, 128, 256])])
def test_partition_rejects_manifest_geometry_that_understates_stored_bank(tmp_path, field, value):
    source = tmp_path / "original"
    original_source(source)
    target = tmp_path / "overlay"
    manifest = export(source, target, [3], [0], 256, 256)
    manifest["experts"]["3:0"][field] = value
    (target / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="admitted manifest geometry"):
        load_partition(target, 3, 0, 1, host_budget_bytes=16 * 1024**2)


def test_storage_verifier_checks_complete_selection_and_native_headers(tmp_path):
    from tools.glm_perf.verify_channel_scale_overlay import verify

    source = tmp_path / "original"
    original_source(source)
    target = tmp_path / "overlay"
    manifest = export(source, target, [3], [0, 1], 256, 256)
    report = verify(target)
    assert report["shards"] == 2 and report["tensor_bytes"] == sum(
        entry["tensor_bytes"] for entry in manifest["experts"].values()
    )
    assert report["file_bytes"] > report["tensor_bytes"]
    manifest["experts"]["3:1"]["gate_shape"] = [1, 128, 256]
    (target / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="tensor header"):
        verify(target)
    del manifest["experts"]["3:1"]
    (target / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="coverage"):
        verify(target)
