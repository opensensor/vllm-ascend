# SPDX-License-Identifier: Apache-2.0
"""The Ascend context path must store returned RoPE output, including on 310P."""

import ast
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch
import torch.nn.functional as F


def method():
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/patch/worker/patch_qwen3_dflash.py"
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "precompute_and_store_context_kv")
    scope = {"torch": torch, "F": F}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), scope)
    return scope[node.name]


@pytest.mark.parametrize("tokens", [1, 3, 128])
@pytest.mark.parametrize("mapping", ["shared", "per_layer", "dummy"])
def test_stores_rotated_keys_and_original_values(tokens, mapping):
    torch.manual_seed(804)
    states = torch.randn(tokens, 8)
    positions = torch.arange(tokens) + 17
    weights = torch.randn(2 * 2 * 2 * 4, 8)
    slots = torch.arange(tokens) + 129
    writes = []

    def rotary(pos, q, k):
        # Deliberately out of place: ignoring this return silently stores
        # unrotated keys, although startup and tensor shapes still succeed.
        return q + pos[:, None], k + pos[:, None]

    layers = [NS(self_attn=NS(k_norm=lambda x: x * 2, rotary_emb=rotary)) for _ in range(2)]
    owner = NS(
        _num_attn_layers=2,
        _kv_size=8,
        _head_dim=4,
        _num_kv_heads=2,
        hidden_norm=lambda x: x / 2,
        _fused_kv_weight=weights,
        _fused_kv_bias=None,
        layers=layers,
        _attn_layers=[
            NS(kv_cache=object(), impl=NS(do_kv_cache_update=lambda *args: writes.append(args))) for _ in range(2)
        ],
    )
    passed_slots = None if mapping == "dummy" else ([slots, None] if mapping == "per_layer" else slots)
    method()(owner, states, positions, passed_slots)
    expected = F.linear(states / 2, weights).view(tokens, 2, 2, 2, 4).permute(2, 1, 0, 3, 4)
    assert len(writes) == {"dummy": 0, "per_layer": 1, "shared": 2}[mapping]
    for i, (attn, key, value, cache, received_slots) in enumerate(writes):
        assert attn is owner._attn_layers[i] and cache is attn.kv_cache and received_slots is slots
        torch.testing.assert_close(key, expected[0, i] * 2 + positions[:, None, None])
        torch.testing.assert_close(value, expected[1, i])


def test_context_patch_is_loaded_outside_310p_exclusion():
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/patch/worker/__init__.py"
    tree = ast.parse(path.read_text())
    assert any(
        isinstance(n, ast.Import) and any(a.name.endswith("patch_qwen3_dflash") for a in n.names) for n in tree.body
    )
