# SPDX-License-Identifier: Apache-2.0
"""Permanent indexer hooks affect loaded instances and preserve other models."""

import json
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from tools.glm_perf import indexer_bundle


def score_kpool_paged(value):
    return "original:" + value


class Sparse(torch.nn.Module):
    def _select_tokens_fixed(self, value):
        return score_kpool_paged(value)

    def _write_pools(self, value):
        return "original-write:" + value


class Indexer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.indexer_op = Sparse()

    def forward(self, value):
        return "original-forward:" + value


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    for name, attributes in {
        "vllm_ascend.models.glm5next.attention": {"Indexer": Indexer},
        "vllm_ascend.models.glm5next.sparse_attn_indexer_kpool": {"SparseAttnIndexerKpool": Sparse},
    }.items():
        module = ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
    (tmp_path / "manifest.json").write_text("{}")
    (tmp_path / "options.json").write_text(json.dumps({"namespace": "native_test"}))
    verified = []
    monkeypatch.setattr(
        indexer_bundle,
        "verified_manifest",
        lambda root: (
            verified.append(True) or SimpleNamespace(value={"operators": ["native_test::launch"], "libraries": []}),
            {"namespace": "native_test"},
        ),
    )
    cast = object()

    def replacements(resources):
        assert resources["bf16_cast_v1"] is cast

        def write(self, value):
            return "native-write:" + value

        return {
            "vllm_ascend.models.glm5next.attention:Indexer.forward": lambda self, value: "native-forward:" + value,
            "vllm_ascend.models.glm5next.sparse_attn_indexer_kpool:SparseAttnIndexerKpool._write_pools": write,
            "vllm_ascend.models.glm5next.sparse_attn_indexer_kpool:score_kpool_paged": lambda value: "native:" + value,
        }

    def module(path, name):
        return (
            SimpleNamespace(NativeBF16Cast=lambda *args: cast)
            if path.name == "bf16_cast.py"
            else SimpleNamespace(replacements=replacements)
        )

    monkeypatch.setattr(indexer_bundle, "read_module", module)
    return tmp_path, verified, cast


def test_composition_binds_each_instance_without_changing_class_methods(prepared):
    root, verified, cast = prepared
    untouched = Indexer()
    model = torch.nn.Sequential(Indexer(), Indexer())
    assert indexer_bundle.install(model, root) == 2
    for instance in model:
        assert instance("x") == "native-forward:x"
        assert instance.indexer_op._write_pools("x") == "native-write:x"
        assert instance.indexer_op._select_tokens_fixed("x") == "native:x"
        assert instance._native_bf16_cast is cast
    assert untouched("x") == "original-forward:x"
    assert untouched.indexer_op._select_tokens_fixed("x") == "original:x"
    assert untouched.indexer_op._write_pools("x") == "original-write:x"
    assert verified == [True]


def test_composition_rejects_model_without_indexers(prepared):
    with pytest.raises(ValueError, match="no GLM indexer"):
        indexer_bundle.install(torch.nn.Sequential(), prepared[0])


def make_bundle(root):
    from tools.glm_perf.native_checkpoint import file_digest, write_json

    root.mkdir()
    files = ("bridge.so", "bf16_cast.py", "candidate.py", "options.json")
    for filename in files:
        (root / filename).write_text(filename)
    write_json(root / "options.json", {"namespace": "indexer_test"})
    manifest = dict(
        name="indexer_test_v1",
        operators=["indexer_test::launch"],
        validation_source='def validate(resources):\n    return {"passed": True}\n',
        libraries=[],
        assets=[],
    )
    for filename in files:
        field = "libraries" if filename.endswith(".so") else "assets"
        manifest[field].append({"path": str(root / filename), "sha256": file_digest(root / filename)})
    write_json(root / "manifest.json", manifest)
    return root


def test_old_absolute_receipt_is_rebased_and_hashes_remain_enforced(tmp_path):
    import shutil

    source = make_bundle(tmp_path / "old-location")
    moved = tmp_path / "downloaded-snapshot"
    shutil.copytree(source, moved)
    manifest, options = indexer_bundle.verified_manifest(moved)
    assert options["namespace"] == "indexer_test"
    assert all(entry["path"].startswith(str(moved)) for entry in manifest.value["libraries"] + manifest.value["assets"])
    (moved / "candidate.py").write_text("changed")
    with pytest.raises(ValueError, match="digest mismatch"):
        indexer_bundle.verified_manifest(moved)


def test_portable_receipt_cannot_read_outside_its_bundle(tmp_path):
    source = make_bundle(tmp_path / "bundle")
    path = source / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["assets"][0]["path"] = "../outside.py"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="escapes"):
        indexer_bundle.verified_manifest(source)
