# SPDX-License-Identifier: Apache-2.0
"""Reject incompatible precision experiments before creating build artifacts."""

import pytest

from tools.glm_perf.build_bf16_cast import build as build_bf16
from tools.glm_perf.build_reconstruction import build


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
