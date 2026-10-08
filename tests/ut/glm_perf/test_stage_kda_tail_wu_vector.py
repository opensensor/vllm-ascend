# SPDX-License-Identifier: Apache-2.0
"""Pin isolated staging and the scalar-order finite-half dot-product contract."""

import hashlib
import shutil
import subprocess
from pathlib import Path

import pytest
import torch

from tools.glm_perf.stage_kda_tail_wu_vector import EXTRA_UB_BYTES, FLAG, VECTOR, stage, transform

ROOT = Path(__file__).resolve().parents[3]
HEADER = ROOT / "csrc/attention/chunk_kda_fwd/op_kernel/chunk_kda_fwd_post_wu.h"


def test_disabled_vector_flag_preserves_complete_scalar_tail_method(tmp_path):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("requires host preprocessor")
    original = HEADER.read_text()
    results = []
    for source in (original, transform(original)):
        begin = source.index("    __aicore__ inline void ComputeTailWuRow(")
        end = source.index("    __aicore__ inline void ComputeTailWuVector(", begin)
        path = tmp_path / "tail.cpp"
        path.write_text(source[begin:end])
        results.append(
            subprocess.run(
                [compiler, "-E", "-P", "-D__CCE_AICORE__=200", str(path)], check=True, capture_output=True, text=True
            ).stdout.split()
        )
    assert results[0] == results[1]
    assert original == HEADER.read_text()


def test_frozen_parent_and_separate_buffer_staging(tmp_path):
    source = tmp_path / "parent.h"
    source.write_bytes(HEADER.read_bytes())
    expected = hashlib.sha256(source.read_bytes()).hexdigest()
    target = tmp_path / "candidate.h"
    record = stage(source, target, expected)
    assert record["compiler_define"] == FLAG and record["extra_ub_bytes"] == EXTRA_UB_BYTES == 5504
    assert source.read_bytes() == HEADER.read_bytes()
    assert record["candidate_sha256"] == hashlib.sha256(target.read_bytes()).hexdigest()
    assert not record["full_kda_evaluated"] and not record["serving_evaluated"]
    assert "BT_ == 64 && curT > 0 && curT < 64 && dim == 128" in target.read_text()
    with pytest.raises(FileExistsError):
        stage(source, target, expected)
    with pytest.raises(ValueError, match="qualified parent"):
        stage(source, tmp_path / "wrong.h", "0" * 64)
    assert not (tmp_path / "wrong.h").exists()
    with pytest.raises(ValueError, match="already staged"):
        transform(target.read_text())


def test_finite_half_products_are_exact_before_fp32_accumulation():
    bits = torch.arange(65536, dtype=torch.int32)
    finite = (bits & 0x7C00) != 0x7C00
    half = bits.short().view(torch.float16)[finite]
    for shift in (0, 1, 997, 32000):
        other = half.roll(shift)
        product = half.float() * other.float()
        exact = half.double() * other.double()
        assert torch.equal(product.double(), exact)


@pytest.mark.parametrize("count", [1, 3, 16, 31, 63])
@pytest.mark.parametrize("alias", [False, True])
def test_vector_channel_operations_preserve_each_sequential_sum_and_alias_order(count, alias):
    generator = torch.Generator().manual_seed(count)
    coefficients = torch.tril((torch.randn(count, count, generator=generator) * 0.1).half())
    coefficients.diagonal().fill_(1)
    source = (torch.randn(count, 128, generator=generator) * 100).half()
    source[:, ::7] = -0.0
    scalar_source, vector_source = source.clone(), source.clone()
    scalar_output = scalar_source if alias else torch.empty_like(source)
    vector_output = vector_source if alias else torch.empty_like(source)
    order = range(count - 1, -1, -1) if alias else range(count)
    # Keep all j terms, including zero coefficients for aliased rows already
    # written. Arithmetic and overwrite order mirror the original kernel.
    for row in order:
        for column in range(128):
            accumulator = torch.tensor(0.0, dtype=torch.float32)
            for j in range(count):
                product = float(coefficients[row, j]) * float(scalar_source[j, column])
                accumulator = (accumulator.double() + product).float()
            scalar_output[row, column] = accumulator.clamp(-65504, 65504).half()
        accumulator = torch.zeros(128, dtype=torch.float32)
        for j in range(count):
            product = coefficients[row, j].float() * vector_source[j].float()
            accumulator = accumulator + product
        vector_output[row] = accumulator.clamp(-65504, 65504).half()
    assert torch.equal(scalar_output.view(torch.int16), vector_output.view(torch.int16))


def test_live_accumulator_never_shares_scalar_read_or_mixed_type_storage():
    assert "GetValue(" not in VECTOR
    assert "CopyVectorIn(coefficientHalf, preparedAqk_, akkBase, 64);" in VECTOR
    assert "tailValuesHalfBuf_" in VECTOR and "tailValuesFloatBuf_" in VECTOR
    assert "tailAccumulatorBuf_" in VECTOR and "tailOutputBuf_" in VECTOR
    assert "Gather(broadcast, coefficientBroadcast[j * 8]" in VECTOR
    assert "Add(accumulator, accumulator, product" in VECTOR
