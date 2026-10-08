"""CPU checks for opt-in GLM W2 safetensors pre-read filtering."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from vllm.config.load import LoadConfig
from vllm.model_executor.model_loader import get_model_loader
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader

import vllm_ascend.model_loader.glm_w2_safetensors as loader_module
from vllm_ascend.model_loader.glm_w2_safetensors import (
    LOAD_FORMAT,
    GlmW2FilteredSafetensorsLoader,
    local_decoder_expert_range,
    should_skip_glm_expert,
)


def model_config(path: str, *, architecture: str = "Glm5NextW2ForCausalLM") -> SimpleNamespace:
    return SimpleNamespace(
        model=path,
        revision=None,
        architectures=[architecture],
        hf_config=SimpleNamespace(model_type="glm5_next"),
        hf_text_config=SimpleNamespace(
            model_type="glm5_next_text", n_routed_experts=4, first_k_dense_replace=3, num_hidden_layers=5
        ),
    )


def test_tp_contiguous_range_matches_model_bank(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(loader_module, "_ep_rank_size", lambda: (1, 2))
    assert local_decoder_expert_range(model_config("/tmp/model")) == (2, 4, 3, 5)
    tensor = "model.language_model.layers.3.mlp.experts.1.gate_proj_codes"
    assert should_skip_glm_expert(tensor, local_range=(2, 4), dense_layers=3, decoder_layers=5, num_experts=4) == (
        True,
        "codes",
    )
    assert should_skip_glm_expert(
        tensor.replace("experts.1", "experts.2"), local_range=(2, 4), dense_layers=3, decoder_layers=5, num_experts=4
    ) == (False, "codes")
    assert should_skip_glm_expert(
        tensor.replace("layers.3", "layers.5"), local_range=(2, 4), dense_layers=3, decoder_layers=5, num_experts=4
    ) == (False, None)


def test_skip_before_get_tensor_and_preserve_default_file_selection(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prefix = "model.language_model.layers.3.mlp.experts."
    peer_codes = prefix + "1.gate_proj_codes"
    peer_scale = prefix + "1.gate_proj_scale"
    local_codes = prefix + "2.gate_proj_codes"
    mtp_codes = "model.language_model.layers.5.mlp.experts.1.gate_proj_codes"
    dense = "model.language_model.layers.3.self_attn.q_proj.weight"
    vision = "model.visual.proj.weight"
    shard = tmp_path / "model-00001-of-00001.safetensors"
    save_file(
        {
            peer_codes: torch.zeros((2, 3), dtype=torch.uint8),
            peer_scale: torch.zeros((2, 3), dtype=torch.float32),
            local_codes: torch.ones((2, 3), dtype=torch.uint8),
            mtp_codes: torch.ones((2, 3), dtype=torch.uint8),
            dense: torch.ones(1),
            vision: torch.ones(1),
        },
        shard,
    )
    loader = GlmW2FilteredSafetensorsLoader(LoadConfig(load_format=LOAD_FORMAT, use_tqdm_on_load=False))
    loader._local_range = (2, 4)
    loader._dense_layers = 3
    loader._decoder_layers = 5
    loader._num_experts = 4
    loader._model_path = str(tmp_path)
    original_open = loader_module.safe_open
    materialized: list[str] = []

    class SpyHandle:
        def __init__(self, path: str, framework: str):
            self.handle = original_open(path, framework=framework)

        def __enter__(self):
            self.handle.__enter__()
            return self

        def __exit__(self, *args):
            return self.handle.__exit__(*args)

        def keys(self):
            return self.handle.keys()

        def get_slice(self, name: str):
            return self.handle.get_slice(name)

        def get_tensor(self, name: str):
            materialized.append(name)
            return self.handle.get_tensor(name)

    monkeypatch.setattr(loader_module, "safe_open", SpyHandle)
    source = DefaultModelLoader.Source(str(tmp_path), revision=None)
    selected = dict(loader._get_weights_iterator(source))
    assert set(selected) == {local_codes, mtp_codes, dense, vision}
    assert set(materialized) == set(selected)
    assert loader.skipped_tensors == 2
    assert loader.skipped_bytes == 6 + 24
    assert loader.matched_expert_tensors == 3


def test_opt_in_registration_and_rejections(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert type(get_model_loader(LoadConfig(load_format="safetensors"))) is DefaultModelLoader
    assert isinstance(get_model_loader(LoadConfig(load_format=LOAD_FORMAT)), GlmW2FilteredSafetensorsLoader)
    with pytest.raises(ValueError, match="lazy"):
        GlmW2FilteredSafetensorsLoader(LoadConfig(load_format=LOAD_FORMAT, safetensors_load_strategy="eager"))
    with pytest.raises(ValueError, match="architecture"):
        local_decoder_expert_range(model_config(str(tmp_path), architecture="OtherModel"))
    monkeypatch.setattr(loader_module, "_ep_rank_size", lambda: (0, 2))
    wrong = model_config(str(tmp_path))
    wrong.hf_text_config.model_type = "other"
    with pytest.raises(ValueError, match="checkpoint"):
        local_decoder_expert_range(wrong)
    peer = "model.language_model.layers.3.mlp.experts.9.gate_proj_codes"
    with pytest.raises(ValueError, match="out-of-range"):
        should_skip_glm_expert(peer, local_range=(0, 2), dense_layers=3, decoder_layers=5, num_experts=4)


def test_index_skips_superseded_local_expert_before_get_tensor(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    expert = "model.language_model.layers.3.mlp.experts.2.gate_proj_codes"
    shared = "model.language_model.layers.3.self_attn.q_proj.weight"
    original = tmp_path / "model-00001-of-00002.safetensors"
    overlay = tmp_path / "model-00002-of-00002.safetensors"
    save_file({expert: torch.zeros(2, dtype=torch.uint8), shared: torch.ones(1)}, original)
    save_file({expert: torch.ones(2, dtype=torch.uint8)}, overlay)
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {expert: overlay.name, shared: original.name}})
    )
    loader = GlmW2FilteredSafetensorsLoader(LoadConfig(load_format=LOAD_FORMAT, use_tqdm_on_load=False))
    loader._local_range = (2, 4)
    loader._dense_layers = 3
    loader._decoder_layers = 5
    loader._num_experts = 4
    loader._model_path = str(tmp_path)
    real_open = loader_module.safe_open
    materialized = []

    class SpyHandle:
        def __init__(self, path: str, framework: str):
            self.handle = real_open(path, framework=framework)

        def __enter__(self):
            self.handle.__enter__()
            return self

        def __exit__(self, *args):
            return self.handle.__exit__(*args)

        def keys(self):
            return self.handle.keys()

        def get_slice(self, name: str):
            return self.handle.get_slice(name)

        def get_tensor(self, name: str):
            materialized.append(name)
            return self.handle.get_tensor(name)

    monkeypatch.setattr(loader_module, "safe_open", SpyHandle)
    weights = list(loader._get_weights_iterator(DefaultModelLoader.Source(str(tmp_path), revision=None)))

    assert {name for name, _ in weights} == {expert, shared}
    assert len(weights) == 2
    assert sum(name == expert for name in materialized) == 1
    assert loader.skipped_superseded_tensors == 1
    assert loader.skipped_superseded_bytes == 2
    assert dict(weights)[expert].tolist() == [1, 1]


def test_wrong_checkpoint_signature_fails(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    save_file({"model.language_model.layers.3.self_attn.q_proj.weight": torch.ones(1)}, tmp_path / "model.safetensors")
    monkeypatch.setattr(loader_module, "_ep_rank_size", lambda: (0, 2))

    def consume(self, model, config):
        list(self.get_all_weights(config, model))

    monkeypatch.setattr(DefaultModelLoader, "load_weights", consume)
    loader = GlmW2FilteredSafetensorsLoader(LoadConfig(load_format=LOAD_FORMAT, use_tqdm_on_load=False))
    with pytest.raises(ValueError, match="no GLM W2 decoder expert"):
        loader.load_weights(SimpleNamespace(), model_config(str(tmp_path)))


def test_mtp_loader_selects_draft_layer_and_filters_peer_experts(tmp_path, monkeypatch):
    candidate = model_config(str(tmp_path), architecture="Glm5NextW2MTPModel")
    candidate.hf_config.model_type = candidate.hf_text_config.model_type = "glm5_next_mtp"
    candidate.hf_text_config.num_nextn_predict_layers = 1
    monkeypatch.setattr(loader_module, "_ep_rank_size", lambda: (1, 2))
    assert local_decoder_expert_range(candidate) == (2, 4, 5, 6)
    prefix = "model.language_model.layers.5."
    shard = tmp_path / "model-00001-of-00001.safetensors"
    save_file(
        {
            "model.language_model.layers.3.mlp.experts.2.gate_proj_codes": torch.ones(2, 3, dtype=torch.uint8),
            prefix + "mlp.experts.1.gate_proj_codes": torch.ones(2, 3, dtype=torch.uint8),
            prefix + "mlp.experts.2.gate_proj_codes": torch.ones(2, 3, dtype=torch.uint8),
            prefix + "enorm.weight": torch.ones(1),
            "model.language_model.embed_tokens.weight": torch.ones(1),
        },
        shard,
    )
    loader = GlmW2FilteredSafetensorsLoader(LoadConfig(load_format=LOAD_FORMAT, use_tqdm_on_load=False))
    loader._local_range = (2, 4)
    loader._dense_layers, loader._decoder_layers, loader._num_experts = 5, 6, 4
    loader._model_path = str(tmp_path)
    loader._draft_layer_prefixes = (prefix,)
    selected = dict(loader._get_weights_iterator(DefaultModelLoader.Source(str(tmp_path), revision=None)))
    assert set(selected) == {prefix + "mlp.experts.2.gate_proj_codes", prefix + "enorm.weight"}
    assert loader.skipped_tensors == 1
