# SPDX-License-Identifier: Apache-2.0
"""Placement and admission failures must be explicit before device allocation."""

import copy

import pytest

from tools.minimax_m3.serving_budget import GIB, make_budget, placement


def record(shape=(32, 32), dtype="F16", expert=False):
    return {"expert": expert, "tensors": {"weight": {"shape": shape, "dtype": dtype}}}


def fixture():
    weights = {f"language_model.model.layers.{layer}.self_attn.q_proj.weight": record() for layer in range(3)}
    weights["vision_tower.weight"] = record()
    return {"tensor_count": len(weights), "weights": weights}, {
        "text_config": {
            "num_hidden_layers": 3,
            "max_position_embeddings": 131072,
            "num_attention_heads": 64,
            "num_key_value_heads": 4,
            "num_local_experts": 128,
            "vocab_size": 200064,
            "head_dim": 128,
            "sparse_attention_config": {
                "sparse_index_dim": 128,
                "sparse_block_size": 128,
                "sparse_attention_freq": [0, 1, 1],
            },
        }
    }


def test_replicated_router_index_key_and_vision():
    prefix = "language_model.model.layers.3."
    assert placement(prefix + "self_attn.k_proj.weight", record()) == (3, 1024)
    assert placement(prefix + "self_attn.index_k_proj.weight", record()) == (3, 2048)
    assert placement(prefix + "block_sparse_moe.gate.weight", record(dtype="F32")) == (3, 4096)
    assert placement("vision_tower.encoder.layers.3.weight", record()) == ("vision", 2048)
    assert placement(prefix + "block_sparse_moe.experts.0.w1.weight", record(expert=True)) == (3, 1024)


def test_cache_rounding_replication_and_budget_failure():
    manifest, config = fixture()
    result = make_budget(manifest, config, context=129, sequences=2, device_gib=1, reserve_gib=2)
    assert result["best"]["partition"] == [1, 1, 1]
    assert result["cache_bytes_per_token_all_ranks"] == 2 * (1024 + 1280 + 1280)
    stages = result["best"]["stages"]
    assert stages[0]["kv_bytes_per_rank"] == 512 * 1024
    assert stages[1]["kv_bytes_per_rank"] == 512 * 1280
    assert all(s["weight_bytes_per_rank"] == 3072 for s in stages)
    assert stages[0]["runtime_reserve_bytes"] == 2 * GIB
    assert result["best"]["fits_planning_budget"] is False
    assert make_budget(manifest, config, device_gib=10)["best"]["fits_planning_budget"] is True


@pytest.mark.parametrize(
    "kwargs",
    [
        {"context": 0},
        {"sequences": 0},
        {"device_gib": -1},
        {"reserve_gib": -1},
        {"utilization": 1.1},
        {"context": 131073},
    ],
)
def test_bad_admission_arguments(kwargs):
    with pytest.raises(ValueError):
        make_budget(*fixture(), **kwargs)


def test_unknown_weights_and_incompatible_heads_fail_closed():
    with pytest.raises(ValueError, match="Unknown"):
        placement("language_model.model.layers.0.surprise.weight", record())
    manifest, config = fixture()
    bad = copy.deepcopy(config)
    bad["text_config"]["num_key_value_heads"] = 3
    with pytest.raises(ValueError, match="TP2"):
        make_budget(manifest, bad)


def test_printed_command_keeps_six_rank_topology_and_explicit_budget():
    from tools.minimax_m3.runtime_plan import serve_command

    command = serve_command("/models/MiniMax-M3-W4")
    assert command[command.index("--gpu-memory-utilization") + 1] == "0.95"
    assert command[command.index("--tensor-parallel-size") + 1] == "2"
    assert command[command.index("--pipeline-parallel-size") + 1] == "3"
    assert "VLLM_PP_LAYER_PARTITION=21,20,19" in command
    with pytest.raises(ValueError):
        serve_command("/models/MiniMax-M3-W4", memory_utilization=2)
