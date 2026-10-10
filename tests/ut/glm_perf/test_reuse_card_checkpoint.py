# SPDX-License-Identifier: Apache-2.0
import base64
import json

import pytest
import torch
from safetensors.torch import save_file

from tools.glm_perf.reuse_card_checkpoint import digest, header, make_template, rebuild


def fixture(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    tensors = {"codes": torch.arange(32, dtype=torch.uint8), "scale": torch.tensor([0.125], dtype=torch.float16)}
    index = {}
    for name, tensor in tensors.items():
        shard = f"{name}.safetensors"
        save_file({name: tensor}, str(source / shard))
        index[name] = shard
    (source / "model.safetensors.index.json").write_text(json.dumps({"weight_map": index}))
    original = tmp_path / "export.safetensors"
    save_file(tensors, str(original), metadata={"layout": "permanent"})
    with original.open("rb") as stream:
        raw, _ = header(stream)
    template = {
        original.name: {
            "header": base64.b64encode(raw).decode(),
            "sha256": digest(original),
            "bytes": original.stat().st_size,
        }
    }
    return source, original, template


def test_reuse_recreates_export_exactly_and_resumes(tmp_path):
    source, original, template = fixture(tmp_path)
    output = tmp_path / "output"
    report = rebuild(template, [source], output)
    assert report["rebuilt_shards"] == 1
    assert (output / original.name).read_bytes() == original.read_bytes()
    assert rebuild(template, [source], output)["reused_shards"] == 1


def test_different_tensor_bytes_do_not_replace_previous_output(tmp_path):
    source, original, template = fixture(tmp_path)
    save_file({"codes": torch.zeros(32, dtype=torch.uint8)}, str(source / "codes.safetensors"))
    output = tmp_path / "output"
    output.mkdir()
    target = output / original.name
    target.write_bytes(b"previous")
    with pytest.raises(ValueError, match="differ"):
        rebuild(template, [source], output)
    assert target.read_bytes() == b"previous"
    assert list(output.iterdir()) == [target]


def test_missing_source_tensor_is_not_published(tmp_path):
    source, original, template = fixture(tmp_path)
    index_path = source / "model.safetensors.index.json"
    index_path.write_text(json.dumps({"weight_map": {}}))
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="no host-local source"):
        rebuild(template, [source], output)
    assert not list(output.iterdir())


def test_reject_escaping_output_path(tmp_path):
    source, original, template = fixture(tmp_path)
    template["../escaped.safetensors"] = template.pop(original.name)
    with pytest.raises(ValueError, match="escapes"):
        rebuild(template, [source], tmp_path / "output")


def test_template_reads_exported_headers_and_preserves_manifest_digest(tmp_path):
    source, original, template = fixture(tmp_path)
    record = {"file": original.name, "sha256": digest(original), "bytes": original.stat().st_size}
    manifest = {"complete": True, "native_shards": [record], "scale_shards": [record], "dense_shards": []}
    path = tmp_path / "native-layout.json"
    path.write_text(json.dumps(manifest))
    assert make_template(tmp_path) == template
    manifest["complete"] = False
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="complete"):
        make_template(tmp_path)
