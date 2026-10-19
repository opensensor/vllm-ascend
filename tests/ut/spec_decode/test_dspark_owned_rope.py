# SPDX-License-Identifier: Apache-2.0
"""A draft must not borrow the target's rotary frequencies or positions."""

import ast
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch


@pytest.mark.parametrize("tokens", [1, 4, 20, 640])
@pytest.mark.parametrize("global_cache", ["missing", "stale"])
def test_draft_rope_uses_own_cache(tokens, global_cache):
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/_310p/ops/rotary_embedding.py"
    node = next(
        n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "_rope_forward_oot"
    )
    seen = []

    def apply(q, k, cos, sin, rotary_mode):
        seen.append((cos, sin))
        assert rotary_mode == "half"

        def rotate(x):
            a, b = x.chunk(2, dim=-1)
            return x * cos + torch.cat((-b, a), dim=-1) * sin

        return rotate(q), rotate(k)

    def unexpected_update(_):
        raise AssertionError("draft touched global target cache")

    cache = torch.randn(2048, 128)
    stale = None if global_cache == "missing" else torch.zeros(1, 1, 1, 64)
    scope = {
        "torch": torch,
        "torch_npu": NS(npu_apply_rotary_pos_emb=apply),
        "update_cos_sin": unexpected_update,
        "get_cos_and_sin_slice": lambda: (stale, stale),
    }
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), scope)
    owner = NS(cos_sin_cache=cache, head_size=128, rotary_dim=128, use_model_owned_rope=True)
    positions = (torch.arange(tokens) * 7) % 2048
    q, k = torch.randn(tokens, 3 * 128), torch.randn(tokens, 128)
    actual_q, actual_k = scope[node.name](owner, positions, q, k, True)
    cos, sin = cache[positions].chunk(2, dim=-1)
    cos, sin = cos.repeat(1, 2)[:, None], sin.repeat(1, 2)[:, None]
    for original, actual in ((q, actual_q), (k, actual_k)):
        x = original.view(tokens, -1, 128)
        a, b = x.chunk(2, dim=-1)
        expected = (x * cos + torch.cat((-b, a), dim=-1) * sin).reshape(original.shape)
        torch.testing.assert_close(actual, expected)
    assert len(seen) == 1
