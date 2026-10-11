# SPDX-License-Identifier: Apache-2.0
"""Serving-loader selection over real CPU safetensors with upstream stubs."""

import dataclasses
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file


@pytest.fixture
def module(monkeypatch):
    @dataclasses.dataclass
    class Config:
        load_format: str = "glm_channel_scale_candidate"
        model_loader_extra_config: dict = dataclasses.field(
            default_factory=lambda: {
                "channel_overlay": "overlay",
                "channel_bundle": "bundle",
                "channel_host_budget_bytes": 32 * 1024**3,
                "channel_device_budget_bytes": 31 * 1024**3,
            }
        )

    class Default:
        def __init__(self, config):
            self.load_config = config

        def load_weights(self, model, config):
            model.loaded = dict(
                self._get_weights_iterator(SimpleNamespace(model_or_path=config.model, prefix="", subfolder=None))
            )

    stubs = {
        "vllm.logger": {"logger": SimpleNamespace(info=lambda *args: None)},
        "vllm.distributed": {"get_pp_group": lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True)},
        "vllm.model_executor.model_loader": {"register_model_loader": lambda name: lambda cls: cls},
        "vllm.model_executor.model_loader.default_loader": {"DefaultModelLoader": Default},
    }
    for name, attrs in stubs.items():
        stub = ModuleType(name)
        stub.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, stub)
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/model_loader/glm_channel_scale.py"
    spec = importlib.util.spec_from_file_location("channel_loader_under_test", path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result, Config


@pytest.mark.parametrize("stage", [0, 1, 2])
def test_pipeline_loader_reads_only_stage_dense_weights_and_never_legacy_experts(tmp_path, module, monkeypatch, stage):
    loaded, config = module
    layer = (3, 18, 33)[stage]
    tensors = {f"model.language_model.layers.{n}.self_attn.q_proj.weight": torch.ones(2) * n for n in (3, 18, 33)}
    tensors["model.language_model.embed_tokens.weight"] = torch.ones(2)
    tensors["model.language_model.norm.weight"] = torch.ones(2)
    tensors["lm_head.weight"] = torch.ones(2)
    legacy = f"model.language_model.layers.{layer}.mlp.experts.0.gate_proj_codes"
    tensors[legacy] = torch.ones(256, 128, dtype=torch.uint8)
    save_file(tensors, str(tmp_path / "base.safetensors"))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {n: "base.safetensors" for n in tensors}})
    )
    model = SimpleNamespace(
        _glm_text_config=SimpleNamespace(num_experts_per_token=8),
        vllm_config=SimpleNamespace(
            parallel_config=SimpleNamespace(pipeline_parallel_size=3),
            speculative_config=None,
            scheduler_config=SimpleNamespace(max_num_batched_tokens=640),
        ),
        model=SimpleNamespace(start_layer=stage * 15, end_layer=(stage + 1) * 15),
    )
    monkeypatch.setattr(
        loaded, "get_pp_group", lambda: SimpleNamespace(is_first_rank=stage == 0, is_last_rank=stage == 2)
    )
    prepared = []

    def prepare(*args, **kwargs):
        prepared.append(kwargs)
        return {"layers": [layer]}

    monkeypatch.setattr(loaded, "prepare_stage", prepare)
    loader = loaded.GlmChannelScaleLoader(config())
    loader.load_weights(model, SimpleNamespace(model=str(tmp_path)))
    assert legacy not in model.loaded
    assert f"model.language_model.layers.{layer}.self_attn.q_proj.weight" in model.loaded
    assert all(f"layers.{other}." not in name for name in model.loaded for other in set((3, 18, 33)) - {layer})
    assert ("model.language_model.embed_tokens.weight" in model.loaded) == (stage == 0)
    assert ("lm_head.weight" in model.loaded) == (stage == 2)
    assert prepared[0]["prefill_bits"] == prepared[0]["decode_bits"] == 8
    assert loader.load_config.model_loader_extra_config == {}


def test_loader_requires_explicit_reservations_and_rejects_unknown_options(module):
    loaded, config = module
    with pytest.raises(ValueError, match="reservations"):
        loaded.GlmChannelScaleLoader(config(model_loader_extra_config={}))
    options = config().model_loader_extra_config
    options["unknown_option"] = True
    with pytest.raises(ValueError, match="reservations"):
        loaded.GlmChannelScaleLoader(config(model_loader_extra_config=options))


def test_owner_requires_separate_reservation_and_preserves_disabled_default(module):
    loaded, config = module
    assert loaded.GlmChannelScaleLoader(config()).indexer_owner_options is None
    options = config().model_loader_extra_config
    options["indexer_owner_rank"] = 0
    with pytest.raises(ValueError, match="both"):
        loaded.GlmChannelScaleLoader(config(model_loader_extra_config=options))
    options["indexer_exchange_budget_bytes"] = 12 * 1024**2
    loader = loaded.GlmChannelScaleLoader(config(model_loader_extra_config=options))
    assert loader.indexer_owner_options == dict(owner_rank=0, budget_bytes=12 * 1024**2)


@pytest.mark.parametrize("bundle_path", ["../escape", "indexer"])
def test_indexer_manifest_is_admitted_before_native_stage_allocation(tmp_path, module, monkeypatch, bundle_path):
    loaded, config = module
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {}}))
    (tmp_path / loaded.MANIFEST).write_text(
        json.dumps({"indexer_kernel_bundle": bundle_path, "indexer_manifest_sha256": "wrong"})
    )
    if bundle_path == "indexer":
        (tmp_path / bundle_path).mkdir()
        (tmp_path / bundle_path / "manifest.json").write_text("{}")
    model = SimpleNamespace(
        _glm_text_config=SimpleNamespace(num_experts_per_token=8),
        vllm_config=SimpleNamespace(
            parallel_config=SimpleNamespace(pipeline_parallel_size=3),
            speculative_config=None,
            scheduler_config=SimpleNamespace(max_num_batched_tokens=640),
        ),
    )

    def forbidden(*args, **kwargs):
        pytest.fail("allocated expert banks before validating the base indexer manifest")

    monkeypatch.setattr(loaded, "prepare_stage", forbidden)
    with pytest.raises(ValueError, match="indexer bundle"):
        loaded.GlmChannelScaleLoader(config()).load_weights(model, SimpleNamespace(model=str(tmp_path)))


@pytest.mark.parametrize(
    "plan_name,options",
    [
        ("plan_router_owner", dict(router_owner_rank=0, router_exchange_budget_bytes=1)),
        ("plan_indexer_heads", dict(indexer_head_sharding=True, indexer_head_budget_bytes=1, indexer_score_elements=1)),
    ],
)
def test_new_rank_math_admission_precedes_expert_preparation(module, tmp_path, monkeypatch, plan_name, options):
    loaded, config = module
    cfg = config()
    cfg.model_loader_extra_config.update(options)
    (tmp_path / "model.safetensors.index.json").write_text("{}")
    model = SimpleNamespace(
        _glm_text_config=SimpleNamespace(num_experts_per_token=8),
        vllm_config=SimpleNamespace(
            parallel_config=SimpleNamespace(),
            speculative_config=None,
            scheduler_config=SimpleNamespace(max_num_batched_tokens=640),
        ),
    )

    def reject(*args, **kwargs):
        raise ValueError("offline reservation rejected")

    monkeypatch.setattr(loaded, plan_name, reject)
    monkeypatch.setattr(
        loaded, "prepare_stage", lambda *args, **kwargs: pytest.fail("allocated experts before admission")
    )
    with pytest.raises(ValueError, match="reservation"):
        loaded.GlmChannelScaleLoader(cfg).load_weights(model, SimpleNamespace(model=str(tmp_path)))
