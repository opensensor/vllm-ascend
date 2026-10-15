# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Named checkpoint loading for the isolated real-weight HC experiment."""

import json

import pytest
import torch
from safetensors.torch import save_file

from tools.qwen4exp.hc_checkpoint import PARAMETER_SUFFIXES, load_hyperconnection

PREFIX = "model.language_model.layers.0.attn_hyper_connection"


def checkpoint(tmp_path, *, nested=True, omit=None, wrong_shape=False):
    config = {"hc_count": 2, "hidden_size": 16, "hc_lowrank": 8, "rms_norm_eps": 1e-6}
    (tmp_path / "config.json").write_text(json.dumps({"text_config": config} if nested else config))
    tensors = {
        f"{PREFIX}.hc_norm.weight": torch.arange(32).float() / 100,
        f"{PREFIX}.input_mix_weight_down.weight": torch.arange(8 * 32).float().view(8, 32) / 1000,
        f"{PREFIX}.input_mix_weight_up.weight": torch.ones(32, 8),
        f"{PREFIX}.block_inject_weight.weight": torch.zeros(2, 32),
        "unrelated.expert.weight": torch.full((4, 4), float("nan")),
    }
    if omit:
        del tensors[f"{PREFIX}.{omit}"]
    if wrong_shape:
        tensors[f"{PREFIX}.block_inject_weight.weight"] = torch.zeros(1, 32)
    save_file(tensors, tmp_path / "weights.safetensors")
    index = {"weight_map": dict.fromkeys(tensors, "weights.safetensors")}
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))
    return tensors


@pytest.mark.parametrize("nested", [True, False])
def test_only_named_weights_loaded_with_serving_dtype(tmp_path, nested):
    tensors = checkpoint(tmp_path, nested=nested)
    module, keys = load_hyperconnection(tmp_path, PREFIX)
    assert len(keys) == 4
    assert module.eps == 1e-6
    for attribute, suffix in PARAMETER_SUFFIXES:
        parameter = getattr(module, attribute)
        assert parameter.device.type == "cpu"
        assert parameter.dtype == torch.float16
        assert torch.equal(parameter, tensors[f"{PREFIX}.{suffix}"].half())


def test_incomplete_module_rejected_before_loading(tmp_path):
    checkpoint(tmp_path, omit="input_mix_weight_up.weight")
    with pytest.raises(ValueError, match="no complete combinable hyperconnection"):
        load_hyperconnection(tmp_path, PREFIX)


def test_shape_mismatch_rejected_instead_of_broadcast(tmp_path):
    checkpoint(tmp_path, wrong_shape=True)
    with pytest.raises(ValueError, match="expected.*got"):
        load_hyperconnection(tmp_path, PREFIX)
