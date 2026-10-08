# SPDX-License-Identifier: Apache-2.0
"""Column permutation commutes with stable, per-column route arithmetic."""

import json

import pytest
import torch

from tools.glm_perf.build_reconstruction import build
from tools.glm_perf.route_columns_probe import run, verify_pair


def stable_reduce(rows, weights, ranks, *, native):
    columns = torch.arange(128)
    permutation = columns // 2 + (columns % 2) * 64
    # Mirror the down epilogue: round each route to FP16 before weighting.
    values = rows.half()
    if not native:
        values = values[:, permutation]
    output = []
    for token_ranks in ranks:
        accum = torch.zeros(128, dtype=torch.float32)
        for row in token_ranks.tolist():
            if row < 0 or row >= rows.shape[0]:
                continue
            value = values[row].float() * weights[row]
            accum = accum + value
        output.append(accum[permutation] if native else accum)
    return torch.stack(output)


@pytest.mark.parametrize("tokens", [2, 8, 17, 640])
@pytest.mark.parametrize("kind", ["random", "half_patterns", "cancellation"])
def test_deferred_permutation_preserves_bits_and_masked_route_order(tokens, kind):
    generator = torch.Generator().manual_seed(319)
    count = tokens * 8
    rows = torch.randn(count, 128, generator=generator) * 40
    if kind == "half_patterns":
        # All finite FP16 patterns across the large case, including signed zero
        # and subnormals. Keep NaNs out of the bitwise equality oracle.
        bits = (torch.arange(count * 128) % 65536).to(torch.int16)
        rows = bits.view(torch.float16).float().reshape(count, 128)
        rows[~torch.isfinite(rows)] = 0
    elif kind == "cancellation":
        rows[1::2] = -rows[::2]
        rows[:, ::7] = -0.0
    weights = torch.rand(count, generator=generator)
    weights[::8] = 0
    if kind == "cancellation":
        weights[1::2] = weights[::2]
    ranks = torch.arange(count).reshape(tokens, 8)
    ranks[:, 2] = -1
    ranks[:, 6] = count  # peer/unused suffix must never be read
    old = stable_reduce(rows, weights, ranks, native=False)
    fused = stable_reduce(rows, weights, ranks, native=True)
    assert torch.equal(old.view(torch.int32), fused.view(torch.int32))


@pytest.mark.parametrize(
    "options", [{}, {"fused_moe": True, "all_bits": True, "tile_pipeline": True, "output_columns": 128}]
)
def test_native_columns_reject_an_unpaired_workspace_before_build(tmp_path, options):
    target = tmp_path / "build"
    with pytest.raises(ValueError, match="paired fused MoE"):
        build(target, tmp_path, tmp_path, native_route_columns=True, **options)
    assert not target.exists()


@pytest.mark.parametrize("invalid", [1, "true"])
def test_native_columns_are_an_explicit_boolean_experiment(tmp_path, invalid):
    with pytest.raises(ValueError, match="flags must be boolean"):
        build(tmp_path / "build", tmp_path, tmp_path, native_route_columns=invalid)


def test_full_moe_probe_cannot_select_a_device_by_default(tmp_path):
    with pytest.raises(ValueError, match="explicit"):
        run(tmp_path / "missing", tmp_path / "missing", tmp_path / "report.json")
    assert not (tmp_path / "report.json").exists()


def test_paired_probe_rejects_another_schedule_change_before_loading_binaries(tmp_path):
    baseline, candidate = tmp_path / "baseline", tmp_path / "candidate"
    for path, options in (
        (baseline, dict(native_route_columns=False, fp16_route_workspace=True, wide_cube_k=128)),
        (candidate, dict(native_route_columns=True, fp16_route_workspace=True, wide_cube_k=0)),
    ):
        path.mkdir()
        (path / "provenance.json").write_text(json.dumps({"_build": options}))
    with pytest.raises(ValueError, match="another kernel schedule"):
        verify_pair(baseline, candidate)
