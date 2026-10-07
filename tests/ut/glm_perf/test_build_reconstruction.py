# SPDX-License-Identifier: Apache-2.0
"""Reject incompatible precision experiments before creating build artifacts."""

import pytest

from tools.glm_perf.build_bf16_cast import build as build_bf16
from tools.glm_perf.build_reconstruction import build


def test_direct_hidden_gather_requires_fused_pipeline(tmp_path):
    output = tmp_path / "direct"
    with pytest.raises(ValueError, match="direct hidden gather requires fused"):
        build(output, tmp_path, tmp_path, direct_hidden_gather=True)
    assert not output.exists()


def test_quad_hidden_quant_requires_fused_pipeline(tmp_path):
    output = tmp_path / "quad"
    with pytest.raises(ValueError, match="four-row hidden quantization requires fused"):
        build(output, tmp_path, tmp_path, quad_hidden_quant=True)
    assert not output.exists()


def test_quad_hidden_quant_rejects_conflicting_batch_size(tmp_path):
    output = tmp_path / "quad"
    with pytest.raises(ValueError, match="alternative schedules"):
        build(
            output,
            tmp_path,
            tmp_path,
            output_columns=128,
            tile_pipeline=True,
            all_bits=True,
            fused_moe=True,
            quad_hidden_quant=True,
            pair_hidden_quant=True,
        )
    assert not output.exists()


@pytest.mark.parametrize("invalid", [1, "true"])
def test_quad_hidden_quant_rejects_non_boolean_flag(tmp_path, invalid):
    output = tmp_path / "quad"
    with pytest.raises(ValueError, match="flags must be boolean"):
        build(output, tmp_path, tmp_path, quad_hidden_quant=invalid)
    assert not output.exists()


def test_wide_prefill_rows_require_prepared_vector_scales(tmp_path):
    output = tmp_path / "wide"
    with pytest.raises(ValueError, match="wide prefill rows require"):
        build(output, tmp_path, tmp_path, prefill_rows_32=True)
    assert not output.exists()


@pytest.mark.parametrize(
    "conflict",
    ["weight_decode_lut", "gather_product_matrix", "fp16_swiglu", "strided_product_copy", "repeat_product_cast"],
)
def test_wide_prefill_rejects_incompatible_scratch_before_creation(tmp_path, conflict):
    output = tmp_path / "wide"
    with pytest.raises(ValueError, match="lifetime-compatible scratch"):
        build(
            output,
            tmp_path,
            tmp_path,
            output_columns=128,
            tile_pipeline=True,
            all_bits=True,
            fused_moe=True,
            prepared_weight_layout=True,
            vector_scale_products=True,
            prefill_rows_32=True,
            **{conflict: True},
        )
    assert not output.exists()


def test_wide_prefill_rejects_unvalidated_k128_layout(tmp_path):
    output = tmp_path / "wide"
    with pytest.raises(ValueError, match="default K64"):
        build(
            output,
            tmp_path,
            tmp_path,
            output_columns=128,
            tile_pipeline=True,
            all_bits=True,
            fused_moe=True,
            prepared_weight_layout=True,
            vector_scale_products=True,
            prefill_rows_32=True,
            wide_cube_k=128,
        )
    assert not output.exists()


def test_fp16_swiglu_requires_fused_pipeline_before_creating_output(tmp_path):
    output = tmp_path / "build"
    with pytest.raises(ValueError, match="FP16 SwiGLU requires fused MoE"):
        build(output, tmp_path, tmp_path, fp16_swiglu=True)
    assert not output.exists()


def test_bf16_cast_version_rejected_before_creating_output(tmp_path):
    output = tmp_path / "cast"
    with pytest.raises(ValueError, match="version must be positive"):
        build_bf16(output, tmp_path / "missing-compiler", 0, tmp_path)
    assert not output.exists()


def test_paired_hidden_quantization_requires_fused_pipeline_before_creating_output(tmp_path):
    output = tmp_path / "build"
    with pytest.raises(ValueError, match="paired hidden quantization requires fused MoE"):
        build(output, tmp_path, tmp_path, pair_hidden_quant=True)
    assert not output.exists()


@pytest.mark.parametrize("width", [True, -1, 64, 512])
def test_invalid_wide_cube_geometry_is_rejected_before_creating_output(tmp_path, width):
    output = tmp_path / "build"
    with pytest.raises(ValueError, match="wide Cube K"):
        build(output, tmp_path, tmp_path, wide_cube_k=width)
    assert not output.exists()


@pytest.mark.parametrize(
    "options",
    [{"wide_cube_k": 128}, {"wide_cube_k": 256}, {"pair_prefill_scale_groups": True}, {"weight_decode_lut": True}],
)
def test_native_packing_requires_prepared_paired_fused_pipeline(tmp_path, options):
    output = tmp_path / "build"
    with pytest.raises(ValueError, match="require prepared paired fused MoE"):
        build(output, tmp_path, tmp_path, **options)
    assert not output.exists()


def test_strided_product_copy_requires_fused_pipeline_before_creating_output(tmp_path):
    output = tmp_path / "build"
    with pytest.raises(ValueError, match="strided product copy requires fused MoE"):
        build(output, tmp_path, tmp_path, strided_product_copy=True)
    assert not output.exists()


def test_repeated_product_cast_requires_fused_pipeline_before_creating_output(tmp_path):
    output = tmp_path / "build"
    with pytest.raises(ValueError, match="repeated product cast requires fused MoE"):
        build(output, tmp_path, tmp_path, repeat_product_cast=True)
    assert not output.exists()


def test_product_readback_alternatives_rejected_before_creating_output(tmp_path):
    output = tmp_path / "build"
    with pytest.raises(ValueError, match="alternative readback schedules"):
        build(
            output,
            tmp_path,
            tmp_path,
            output_columns=128,
            tile_pipeline=True,
            all_bits=True,
            fused_moe=True,
            strided_product_copy=True,
            repeat_product_cast=True,
        )
    assert not output.exists()


def test_w3_fragment_reconstruction_requires_fused_pipeline(tmp_path):
    output = tmp_path / "build"
    with pytest.raises(ValueError, match="W3 fragment reconstruction requires fused MoE"):
        build(output, tmp_path, tmp_path, w3_float_fragments=True)
    assert not output.exists()


def test_w3_reconstruction_alternatives_rejected_before_creating_output(tmp_path):
    output = tmp_path / "build"
    with pytest.raises(ValueError, match="alternative reconstruction paths"):
        build(
            output,
            tmp_path,
            tmp_path,
            output_columns=128,
            tile_pipeline=True,
            all_bits=True,
            fused_moe=True,
            w3_float_fragments=True,
            weight_decode_lut=True,
        )
    assert not output.exists()


def test_w3_specialization_requires_fused_pipeline(tmp_path):
    output = tmp_path / "build"
    with pytest.raises(ValueError, match="W3 specialization requires fused MoE"):
        build(output, tmp_path, tmp_path, specialize_w3=True)
    assert not output.exists()
