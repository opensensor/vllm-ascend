# SPDX-License-Identifier: Apache-2.0
"""Projection layout selection preserves non-linear and absorbed operands."""

from tools.sidecard.layer_math import FRACTAL_NZ, prepare_projection_layout


def test_formats_only_linear_consumers():
    names = (
        "self_attn.in_proj",
        "self_attn.f_b_proj.weight",
        "self_attn.g_b_proj.weight",
        "self_attn.o_proj.weight",
        "self_attn.q_a_proj.weight",
        "self_attn.q_b_proj.weight",
        "self_attn.kv_a_proj_with_mqa.weight",
        "mlp.shared_experts.gate_proj.weight",
        "mlp.shared_experts.up_proj.weight",
        "mlp.shared_experts.down_proj.weight",
    )
    unchanged = (
        "self_attn.kv_b_proj.weight",
        "self_attn.indexer.wq_b.weight",
        "self_attn.conv",
        "mlp.gate.weight",
        "hc_attn_fn",
        "input_layernorm.weight",
    )
    weights = {name: object() for name in (*names, *unchanged)}
    original = dict(weights)
    calls = []

    def convert(weight, layout):
        calls.append((weight, layout))
        return (weight, layout)

    prepare_projection_layout(weights, convert)
    assert len(calls) == len(names)
    for name in names:
        assert weights[name] == (original[name], FRACTAL_NZ)
    for name in unchanged:
        assert weights[name] is original[name]
