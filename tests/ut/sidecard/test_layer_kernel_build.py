# SPDX-License-Identifier: Apache-2.0
"""Clipping patches fail closed when audited frozen insertion points change."""

import pytest

from tools.sidecard.layer_kernel_build import NATIVE_INSERTION, clip_cuda, clip_native


def test_native_clip_precedes_sigmoid_and_preserves_negative_gate_range():
    result = clip_native(NATIVE_INSERTION)
    assert result.index("Mins(low, low, 10.0f") < result.index("Muls(accum")
    assert "Maxs(low" not in result
    assert "Mins(high, high, 10.0f" in result and "Maxs(high, high, -10.0f" in result


@pytest.mark.parametrize("source", ["", NATIVE_INSERTION * 2])
def test_native_source_drift_is_rejected(source):
    with pytest.raises(ValueError, match="audited"):
        clip_native(source)


def test_cuda_clip_is_asymmetric_and_does_not_change_projection():
    source = (
        "  float gate = __half2float(gate_up[row * 2 * intermediate + channel]);\n"
        "  float up = __half2float(gate_up[row * 2 * intermediate + intermediate + channel]);\n"
        "  float other = 42.0F;\n"
    )
    result = clip_cuda(source)
    assert "float gate = fminf(10.0F, __half2float(" in result
    assert "float up = fminf(10.0F, fmaxf(-10.0F, __half2float(" in result
    assert "float other = 42.0F;" in result
    with pytest.raises(ValueError):
        clip_cuda(source + source)
