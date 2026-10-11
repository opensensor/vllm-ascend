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
from safetensors import safe_open
from safetensors.torch import save_file

from tools.glm_perf.fused_weight_layout import FP16_SCALE_LAYOUT
from tools.glm_perf.native_checkpoint import INDEX, LAYOUT, MANIFEST, file_digest, write_json


@pytest.fixture
def loader_module(monkeypatch):
    @dataclasses.dataclass
    class LoadConfig:
        load_format: str = "glm_native_int4"
        safetensors_load_strategy: str = "lazy"
        download_dir: str | None = None
        model_loader_extra_config: dict = dataclasses.field(default_factory=dict)

    class Default:
        def __init__(self, config):
            self.counter_before_loading_weights = 0.0
            self.load_config = config

        def load_weights(self, model, config):
            assert not model.bank.nz_packed_codes
            model.loaded = dict(
                self._get_weights_iterator(SimpleNamespace(model_or_path=config.model, prefix="", subfolder=None))
            )

    stubs = {
        "vllm.config.load": {"LoadConfig": LoadConfig},
        "vllm.logger": {"logger": SimpleNamespace(info=lambda *args: None)},
        "vllm.distributed": {"get_pp_group": lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True)},
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
    (tmp_path / "bundle").mkdir()
    (tmp_path / "bundle/provenance.json").write_text(json.dumps({"_build": {"namespace": "test_native"}}))
    bank = SimpleNamespace(
        local_expert_offset=1, num_local_experts=1, offload_to_cpu=False, layer_key="layers.0", nz_packed_codes=True
    )
    owner = SimpleNamespace(w2_experts=bank, routed_experts_forward=lambda: None)
    model = SimpleNamespace(
        bank=bank,
        named_modules=lambda: [("model.layers.3.mlp_w2", owner)],
        vllm_config=SimpleNamespace(
            parallel_config=SimpleNamespace(pipeline_parallel_size=1),
            scheduler_config=SimpleNamespace(max_num_batched_tokens=640),
        ),
    )
    config = SimpleNamespace(
        model=str(tmp_path), revision=None, hf_config=SimpleNamespace(ascend_glm_expert_layout=LAYOUT)
    )
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


@pytest.mark.parametrize("half_routes", [False, True])
def test_direct_loader_uses_compiled_route_workspace_dtype(tmp_path, loader_module, monkeypatch, half_routes):
    module, config_type = loader_module
    model, config, owner = checkpoint(tmp_path)
    calls = []
    instance = object()

    def create(*args, **kwargs):
        calls.append(kwargs)
        return instance

    monkeypatch.setattr(
        module,
        "frozen_helper",
        lambda root: (
            SimpleNamespace(NativeFusedMoE=create),
            {"namespace": "test_native", "fp16_route_workspace": half_routes},
        ),
    )
    module.GlmNativeInt4Loader(config_type()).load_weights(model, config)
    expected = {
        "namespace": "test_native",
        "activation_bits": 4,
        "prepared_weight_layout": True,
        "shared_prefill_max_tokens": 640,
    }
    if half_routes:
        expected["fp16_route_workspace"] = True
    assert calls == [expected] and owner._method.native is instance
    assert model._native_int4_load_report["transformed_code_tensors"] == 0


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


def test_loader_rejects_raw_scales_with_prerounded_kernel_before_reading_weights(tmp_path, loader_module):
    module, config_type = loader_module
    model, config, _ = checkpoint(tmp_path)
    (tmp_path / "bundle/provenance.json").write_text(json.dumps({"_build": {"prerounded_weight_scales": True}}))
    with pytest.raises(ValueError, match="permanent rounded scales"):
        module.GlmNativeInt4Loader(config_type()).load_weights(model, config)
    assert not hasattr(model, "loaded") and model.bank.nz_packed_codes


@pytest.mark.parametrize("fp16_scales", [False, True])
def test_direct_loader_stages_all_checkpoint_tensors_without_changing_layout(
    tmp_path, loader_module, monkeypatch, fp16_scales
):
    module, config_type = loader_module
    model, config, _ = checkpoint(tmp_path)
    if fp16_scales:
        path = tmp_path / "original.safetensors"
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            names = handle.keys()
            tensors = {name: handle.get_tensor(name).clone() for name in names}
        for name in tensors:
            if name.endswith("_proj_scale"):
                tensors[name] = tensors[name].half()
        save_file(tensors, str(path), metadata={"weight_scale_layout": FP16_SCALE_LAYOUT})
        manifest_path = tmp_path / MANIFEST
        manifest = json.loads(manifest_path.read_text())
        manifest["weight_scale_layout"] = FP16_SCALE_LAYOUT
        manifest["scale_shards"] = [{"file": path.name, "sha256": file_digest(path)}]
        write_json(manifest_path, manifest)
        monkeypatch.setattr(module, "validate_scale_contract", lambda *args: FP16_SCALE_LAYOUT)
    original_clone = torch.Tensor.clone
    copied = []

    def record_clone(tensor, *args, **kwargs):
        result = original_clone(tensor, *args, **kwargs)
        copied.append((tensor.data_ptr(), result.data_ptr(), torch.equal(tensor, result)))
        return result

    monkeypatch.setattr(torch.Tensor, "clone", record_clone)
    module.GlmNativeInt4Loader(config_type()).load_weights(model, config)
    assert len(copied) == 5  # Three code matrices, the expert scale, and the dense weight.
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


@pytest.mark.parametrize("overlap", [None, False, True])
def test_owner_option_is_separate_from_quantization_and_installed_after_arithmetic(
    tmp_path, loader_module, monkeypatch, overlap
):
    module, config_type = loader_module
    model, config, _ = checkpoint(tmp_path)
    path = tmp_path / MANIFEST
    data = json.loads(path.read_text())
    data["indexer_kernel_bundle"] = "indexer-kernels-v5"
    path.write_text(json.dumps(data))
    calls = []
    monkeypatch.setattr(module, "plan_install", lambda model, **options: calls.append(("admit", options["owner_rank"])))
    monkeypatch.setattr(module, "install_indexer_bundle", lambda *args: calls.append(("arithmetic",)) or 1)

    def install(loaded_model, **options):
        assert loaded_model is model and "loaded" in vars(model)
        assert calls[-1] == ("arithmetic",)
        calls.append(("owner", options))
        return {"hardware_qualified": False}

    monkeypatch.setattr(module, "install_indexer_owner", install)
    options = dict(indexer_owner_rank=1, indexer_exchange_budget_bytes=12 * 1024**2)
    expected = dict(owner_rank=1, budget_bytes=12 * 1024**2)
    if overlap is not None:
        options["indexer_overlap_prefill"] = overlap
        expected["overlap_prefill"] = overlap
    loader = module.GlmNativeInt4Loader(config_type(model_loader_extra_config=options))
    assert loader.load_config.model_loader_extra_config == {}
    loader.load_weights(model, config)
    assert calls == [("admit", 1), ("arithmetic",), ("owner", expected)]
    assert model._native_int4_load_report["activation_bits"] == 4
    assert model._native_int4_load_report["indexer_owner"] == {"hardware_qualified": False}


def test_owner_config_rejected_before_any_checkpoint_or_weight_io(loader_module, monkeypatch):
    module, config_type = loader_module
    monkeypatch.setattr(module, "resolve_native_folder", lambda *args: pytest.fail("read before ownership admission"))

    def reject(*args, **kwargs):
        raise ValueError("incompatible ownership config")

    monkeypatch.setattr(module, "plan_install", reject)
    loader = module.GlmNativeInt4Loader(
        config_type(model_loader_extra_config=dict(indexer_owner_rank=0, indexer_exchange_budget_bytes=1))
    )
    with pytest.raises(ValueError, match="ownership config"):
        loader.load_weights(SimpleNamespace(vllm_config=object()), object())


def test_pipeline_loader_skips_nonlocal_head_and_caps_prefill_scratch(tmp_path, loader_module, monkeypatch):
    module, config_type = loader_module
    model, config, owner = checkpoint(tmp_path)
    model.vllm_config.parallel_config.pipeline_parallel_size = 3
    model.model = SimpleNamespace(start_layer=3, end_layer=4)
    monkeypatch.setattr(module, "get_pp_group", lambda: SimpleNamespace(is_first_rank=False, is_last_rank=False))
    calls = []
    monkeypatch.setattr(
        module,
        "frozen_helper",
        lambda root: (
            SimpleNamespace(NativeFusedMoE=lambda *args, **kwargs: calls.append(kwargs) or object()),
            {"namespace": "test_native"},
        ),
    )
    loader = module.GlmNativeInt4Loader(config_type())
    loader.load_weights(model, config)
    assert "lm_head.weight" not in model.loaded
    assert loader.pipeline_partition == (3, 4, False, False)
    assert calls[0]["shared_prefill_max_tokens"] == 640
    assert owner._method.native is not None


def test_decode_bundle_is_admitted_before_weight_read_and_dispatches_separately(tmp_path, loader_module, monkeypatch):
    from tools.glm_perf.native_checkpoint import SplitNativeMoE, file_digest

    module, config_type = loader_module
    model, config, owner = checkpoint(tmp_path)
    path = tmp_path / MANIFEST
    manifest = json.loads(path.read_text())
    decode_root = tmp_path / "decode"
    decode_root.mkdir()
    provenance = decode_root / "provenance.json"
    provenance.write_text(json.dumps({"_build": {"namespace": "decode", "helper_package": "decode_helpers"}}))
    manifest.update(decode_kernel_bundle="decode", decode_kernel_provenance_sha256="invalid")
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="decode kernel provenance"):
        module.GlmNativeInt4Loader(config_type()).load_weights(model, config)
    assert not hasattr(model, "loaded") and model.bank.nz_packed_codes
    manifest["decode_kernel_provenance_sha256"] = file_digest(provenance)
    path.write_text(json.dumps(manifest))
    calls = []

    class Backend:
        required_weight_scale_layout = None

        def __init__(self, root, **kwargs):
            self.root = root
            calls.append((root, kwargs))

        def __call__(self, x):
            return self.root.name

    monkeypatch.setattr(
        module,
        "frozen_helper",
        lambda root: (
            SimpleNamespace(NativeFusedMoE=Backend),
            json.loads((root / "provenance.json").read_text())["_build"],
        ),
    )
    module.GlmNativeInt4Loader(config_type()).load_weights(model, config)
    assert isinstance(owner._method.native, SplitNativeMoE)
    assert owner._method.native(torch.zeros(2, 4)) == "decode"
    assert owner._method.native(torch.zeros(640, 4)) == "bundle"
    assert calls[0][1]["shared_prefill_max_tokens"] == 640
    assert "shared_prefill_max_tokens" not in calls[1][1]


@pytest.mark.parametrize(
    "plan_name,options",
    [
        ("plan_router_owner", dict(router_owner_rank=0, router_exchange_budget_bytes=1)),
        ("plan_indexer_heads", dict(indexer_head_sharding=True, indexer_head_budget_bytes=1, indexer_score_elements=1)),
    ],
)
def test_new_rank_math_admission_precedes_checkpoint_resolution(loader_module, monkeypatch, plan_name, options):
    module, config = loader_module
    loader = module.GlmNativeInt4Loader(config(model_loader_extra_config=options))
    assert loader.load_config.model_loader_extra_config == {}

    def reject(*args, **kwargs):
        raise ValueError("offline reservation rejected")

    monkeypatch.setattr(module, plan_name, reject)
    monkeypatch.setattr(module, "resolve_native_folder", lambda *args: pytest.fail("resolved weights before admission"))
    with pytest.raises(ValueError, match="reservation"):
        loader.load_weights(SimpleNamespace(), SimpleNamespace())
