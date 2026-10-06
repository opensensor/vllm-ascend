# SPDX-License-Identifier: Apache-2.0
"""Loader control tests with CPU files and isolated upstream loader stubs."""

import dataclasses
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from tools.glm_perf.native_checkpoint import INDEX, LAYOUT, MANIFEST


@pytest.fixture
def loader_module(monkeypatch):
    @dataclasses.dataclass
    class LoadConfig:
        load_format: str = "glm_native_int4"
        safetensors_load_strategy: str = "lazy"
        model_loader_extra_config: dict = dataclasses.field(default_factory=dict)

    class Default:
        def __init__(self, config):
            self.counter_before_loading_weights = 0.0

        def load_weights(self, model, config):
            assert not model.bank.nz_packed_codes
            model.loaded = dict(
                self._get_weights_iterator(SimpleNamespace(model_or_path=config.model, prefix="", subfolder=None))
            )

    stubs = {
        "vllm.config.load": {"LoadConfig": LoadConfig},
        "vllm.logger": {"logger": SimpleNamespace(info=lambda *args: None)},
        "vllm.model_executor.model_loader": {"register_model_loader": lambda name: lambda cls: cls},
        "vllm.model_executor.model_loader.default_loader": {"DefaultModelLoader": Default},
    }
    for name, attributes in stubs.items():
        module = ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/model_loader/glm_native_int4.py"
    spec = importlib.util.spec_from_file_location("native_loader_test_module", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(
        module,
        "frozen_helper",
        lambda root: (SimpleNamespace(NativeFusedMoE=lambda *args, **kwargs: object()), {"namespace": "test_native"}),
    )
    return module, LoadConfig


def checkpoint(tmp_path):
    prefix = "model.language_model.layers.3.mlp.experts.1."
    tensors = {
        prefix + p + "_proj_codes": torch.full((256, 128), 17, dtype=torch.uint8) for p in ("gate", "up", "down")
    }
    save_file(tensors, str(tmp_path / "native.safetensors"), metadata={"layout": LAYOUT})
    dense = {"lm_head.weight": torch.ones(2), prefix + "gate_proj_scale": torch.ones(8, 8)}
    # Deliberately include superseded canonical bytes: the index must prevent reading them.
    dense.update({name: torch.zeros_like(value) for name, value in tensors.items()})
    save_file(dense, str(tmp_path / "original.safetensors"))
    index = {name: "original.safetensors" for name in dense}
    index.update(dict.fromkeys(tensors, "native.safetensors"))
    (tmp_path / INDEX).write_text(json.dumps({"weight_map": index}))
    manifest = {
        "schema_version": 1,
        "layout": LAYOUT,
        "complete": True,
        "world_size": 2,
        "num_experts": 2,
        "activation_bits": 4,
        "kernel_bundle": "bundle",
        "native_shards": [{"file": "native.safetensors"}],
    }
    (tmp_path / MANIFEST).write_text(json.dumps(manifest))
    bank = SimpleNamespace(
        local_expert_offset=1, num_local_experts=1, offload_to_cpu=False, layer_key="layers.0", nz_packed_codes=True
    )
    owner = SimpleNamespace(w2_experts=bank, routed_experts_forward=lambda: None)
    model = SimpleNamespace(bank=bank, named_modules=lambda: [("model.layers.3.mlp_w2", owner)])
    config = SimpleNamespace(model=str(tmp_path), hf_config=SimpleNamespace(ascend_glm_expert_layout=LAYOUT))
    return model, config, owner


def test_direct_loader_uses_authoritative_native_bytes_and_never_packs(tmp_path, loader_module):
    module, config_type = loader_module
    model, config, owner = checkpoint(tmp_path)
    loader = module.GlmNativeInt4Loader(config_type())
    loader.load_weights(model, config)
    assert loader.layer_keys == ("layers.3",)  # Resolve identity from module path, not bank fallback label.
    assert torch.all(model.loaded["model.language_model.layers.3.mlp.experts.1.gate_proj_codes"] == 17)
    assert model.bank.native_weight_layout == LAYOUT and owner._method.native is not None
    assert model._native_int4_load_report["transformed_code_tensors"] == 0
    assert model._native_int4_load_report["layout_backup_bytes"] == 0


def test_loader_rejects_incomplete_checkpoint_before_weight_read(tmp_path, loader_module):
    module, config_type = loader_module
    model, config, _ = checkpoint(tmp_path)
    path = tmp_path / MANIFEST
    data = json.loads(path.read_text())
    data["complete"] = False
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="incomplete"):
        module.GlmNativeInt4Loader(config_type()).load_weights(model, config)
    assert model.bank.nz_packed_codes and not hasattr(model, "loaded")


def test_direct_loader_stages_native_code_bytes_without_changing_layout(tmp_path, loader_module, monkeypatch):
    module, config_type = loader_module
    model, config, _ = checkpoint(tmp_path)
    original_clone = torch.Tensor.clone
    copied = []

    def record_clone(tensor, *args, **kwargs):
        result = original_clone(tensor, *args, **kwargs)
        copied.append((tensor.data_ptr(), result.data_ptr(), torch.equal(tensor, result)))
        return result

    monkeypatch.setattr(torch.Tensor, "clone", record_clone)
    module.GlmNativeInt4Loader(config_type()).load_weights(model, config)
    assert len(copied) == 3
    assert all(source != staged and equal for source, staged, equal in copied)


def test_direct_loader_composes_optional_permanent_indexer_bundle(tmp_path, loader_module, monkeypatch):
    module, config_type = loader_module
    model, config, _ = checkpoint(tmp_path)
    path = tmp_path / MANIFEST
    data = json.loads(path.read_text())
    data["indexer_kernel_bundle"] = "indexer-kernels-v5"
    path.write_text(json.dumps(data))
    calls = []

    def install(loaded_model, bundle):
        assert loaded_model is model
        assert len(model.loaded) > 0
        calls.append(bundle)
        return 1

    monkeypatch.setattr(module, "install_indexer_bundle", install)
    module.GlmNativeInt4Loader(config_type()).load_weights(model, config)
    assert calls == [tmp_path / "indexer-kernels-v5"]
    assert model._native_int4_load_report["aicore_bf16_indexers"] == 1
    assert model._native_int4_load_report["transformed_code_tensors"] == 0
