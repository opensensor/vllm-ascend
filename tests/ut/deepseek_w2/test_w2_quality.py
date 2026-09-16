# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the host-side W2 quality / mixed-bit sensitivity harness.

Validates, on synthetic FP4<->W2 pairs (with a known perturbation) and on a FEW
real sampled experts from the artifact + source (never a full bank):

* the functional-error metric is *monotonic* in a controlled perturbation
  (rel MSE up, cosine down) — so bigger weight error always ranks worse;
* the ranking orders experts correctly by perturbation magnitude;
* the raw per-tensor W2 reconstruction respects the ``scale/2`` half-step bound;
* the harness runs end-to-end on real sampled experts + Engram rows and yields
  finite errors and a coherent mixed-bit proposal;
* reads stay bounded (no full shard / bank materialised).

Run: ``python3 -m pytest -q --noconftest tests/ut/deepseek_w2/test_w2_quality.py``
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from tools.deepseek_w2.w2_format import (
    W2_BITS,
    W2_BLOCK_COLS,
    W2_BLOCK_ROWS,
    broadcast_block_scales,
    quantize_weight,
)
from tools.deepseek_w2.w2_quality import (
    ArtifactReader,
    ExpertSample,
    build_ranking,
    build_report,
    default_layer_sample,
    engram_row_error,
    functional_error,
    make_activations,
    measure_expert,
    mixed_bit_proposal,
    swiglu_ffn,
    tensor_recon_stats,
)

_SOURCE_DIR = Path("/run/media/matteius/20TB-drive/models/dealignai/DeepSeek-V4.1-Flash-UNCENSORED-FP8")
_ARTIFACT_DIR = Path("/run/media/matteius/20TB-drive/models/DeepSeek-V4.1-W2-310p")

# Small synthetic expert geometry (both dims multiples of the 32x32 block).
_HIDDEN = 64
_INTER = 32
_SAMPLE_LAYER = 0
_SAMPLE_EXPERTS = [0, 1]


def _have_source() -> bool:
    return (_SOURCE_DIR / "model.safetensors.index.json").exists()


def _have_artifact() -> bool:
    return (_ARTIFACT_DIR / "model.safetensors.index.json").exists()


def _current_rss_bytes() -> int:
    """Resident set size of this process, in bytes (Linux /proc)."""
    with open("/proc/self/statm") as handle:
        pages = int(handle.read().split()[1])
    return pages * 4096


def _synthetic_expert(seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """A seeded synthetic (w1, w3, w2) SwiGLU expert."""
    gen = torch.Generator().manual_seed(seed)
    w1 = torch.randn(_INTER, _HIDDEN, generator=gen)
    w3 = torch.randn(_INTER, _HIDDEN, generator=gen)
    w2 = torch.randn(_HIDDEN, _INTER, generator=gen)
    return w1, w3, w2


def _perturb(
    expert: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    eps: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Add ``eps``-scaled seeded gaussian noise to every weight of an expert."""
    gen = torch.Generator().manual_seed(seed)
    return tuple(w + eps * torch.randn(w.shape, generator=gen) for w in expert)


# --- synthetic: metric monotonic in the perturbation -------------------------
def test_functional_error_monotonic_in_perturbation():
    source = _synthetic_expert(seed=1)
    acts = make_activations(_HIDDEN, seed=7)
    rel_mses = []
    cosines = []
    for eps in (0.0, 0.05, 0.1, 0.2, 0.4):
        recon = _perturb(source, eps, seed=99)
        result = functional_error(source, recon, acts)
        rel_mses.append(result["rel_mse_mean"])
        cosines.append(result["cosine_mean"])
    # Zero perturbation is exact; error strictly grows, cosine strictly falls.
    assert rel_mses[0] == pytest.approx(0.0, abs=1e-12)
    assert cosines[0] == pytest.approx(1.0, abs=1e-9)
    assert all(np.diff(rel_mses) > 0), rel_mses
    assert all(np.diff(cosines) < 0), cosines
    assert all(np.isfinite(rel_mses)) and all(np.isfinite(cosines))


# --- synthetic: ranking orders experts by degradation ------------------------
def test_ranking_orders_experts_by_perturbation():
    source = _synthetic_expert(seed=2)
    acts = make_activations(_HIDDEN, seed=3)
    # Experts with increasing perturbation -> increasing functional error.
    eps_by_expert = {10: 0.05, 20: 0.2, 30: 0.1, 40: 0.4}
    samples = []
    for expert, eps in eps_by_expert.items():
        recon = _perturb(source, eps, seed=expert)
        w2 = functional_error(source, recon, acts)
        w3 = functional_error(source, _perturb(source, eps / 2, seed=expert), acts)
        samples.append(ExpertSample(layer=0, expert=expert, w2_functional=w2, w3_functional=w3))
    ranking = build_ranking(samples)
    ranked_experts = [r["expert"] for r in ranking]
    expected = sorted(eps_by_expert, key=lambda e: eps_by_expert[e], reverse=True)
    assert ranked_experts == expected
    # Ranking is sorted descending by rel MSE.
    rel = [r["w2_rel_mse"] for r in ranking]
    assert rel == sorted(rel, reverse=True)


# --- synthetic: raw reconstruction respects the half-step bound --------------
def test_tensor_recon_within_half_step_bound():
    gen = torch.Generator().manual_seed(5)
    w = torch.randn(_HIDDEN, _INTER, generator=gen)
    codes, block_scale = quantize_weight(w, W2_BITS, W2_BLOCK_ROWS, W2_BLOCK_COLS)
    full = broadcast_block_scales(block_scale, _HIDDEN, _INTER)
    w_recon = (codes.double() * full).to(torch.float32)
    stats = tensor_recon_stats(w, w_recon, block_scale.to(torch.float32))
    # Round-to-nearest on the two-tailed grid guarantees err <= scale/2.
    assert stats["max_err_over_bound"] <= 1.0 + 1e-6
    assert stats["max_abs_err"] <= stats["half_step_bound_max"] + 1e-6
    assert np.isfinite(stats["rms_err"])


# --- synthetic: swiglu proxy is exact when weights match ---------------------
def test_swiglu_zero_error_on_identical_weights():
    source = _synthetic_expert(seed=8)
    acts = make_activations(_HIDDEN, seed=8)
    result = functional_error(source, source, acts)
    assert result["rel_mse_mean"] == pytest.approx(0.0, abs=1e-12)
    assert result["cosine_mean"] == pytest.approx(1.0, abs=1e-9)
    # Sanity: the FFN actually produces the expected output shape.
    out = swiglu_ffn(acts["unit_normal"], *(_synthetic_expert(seed=8)[i] for i in (0, 1, 2)))
    assert out.shape == (acts["unit_normal"].shape[0], _HIDDEN)


# --- real: end-to-end on a few sampled experts + engram ----------------------
@pytest.mark.skipif(not (_have_source() and _have_artifact()), reason="source or W2 artifact not present")
def test_real_sampled_experts_produce_finite_errors_and_proposal():
    areader = ArtifactReader(_ARTIFACT_DIR)
    samples = [measure_expert(_SOURCE_DIR, areader, _SAMPLE_LAYER, expert, seed=0) for expert in _SAMPLE_EXPERTS]
    for s in samples:
        assert np.isfinite(s.w2_functional["rel_mse_mean"])
        assert np.isfinite(s.w2_functional["cosine_mean"])
        assert np.isfinite(s.w3_functional["rel_mse_mean"])
        # W3 (3-bit) is a strictly finer grid than W2 -> never worse functionally.
        assert s.w3_functional["rel_mse_mean"] <= s.w2_functional["rel_mse_mean"] + 1e-6
        for proj in ("w1", "w2", "w3"):
            recon = s.tensor_recon[proj]
            # Artifact scale matches the source weight -> within the half-step bound.
            assert recon["max_err_over_bound"] <= 1.0 + 1e-3
            assert np.isfinite(recon["rms_err"])

    ranking = build_ranking(samples)
    assert [r["w2_rel_mse"] for r in ranking] == sorted((r["w2_rel_mse"] for r in ranking), reverse=True)
    proposal = mixed_bit_proposal(ranking, areader)
    assert proposal["model_routed_experts"] > 0
    assert proposal["model_routed_w2_bytes"] > 0
    assert 0.0 <= proposal["fraction_over_budget"] <= 1.0
    assert np.isfinite(proposal["escalation_byte_delta_bytes"])


@pytest.mark.skipif(not (_have_source() and _have_artifact()), reason="source or W2 artifact not present")
def test_real_engram_w4_row_error_is_finite_and_small():
    areader = ArtifactReader(_ARTIFACT_DIR)
    layer = 1
    if f"layers.{layer}.engram.embed_codes.p0" not in areader.weight_map:
        pytest.skip("engram embed not present in artifact")
    result = engram_row_error(_SOURCE_DIR, areader, layer, rows=[0, 12345, 6_000_000])
    assert result["num_rows"] == 3
    assert np.isfinite(result["rel_mse_mean"])
    # W4 (4-bit) row reconstruction should be far tighter than the 2-bit experts.
    assert result["rel_mse_max"] < 0.5
    for row in result["per_row"]:
        assert np.isfinite(row["rel_mse"])


# --- real: reads stay bounded (no full shard / bank in RSS) -------------------
@pytest.mark.skipif(not (_have_source() and _have_artifact()), reason="source or W2 artifact not present")
def test_bounded_working_set_on_real_experts():
    areader = ArtifactReader(_ARTIFACT_DIR)
    # Warm one expert so importer/allocator growth is not attributed to the loop.
    measure_expert(_SOURCE_DIR, areader, _SAMPLE_LAYER, 0, seed=0)
    rss_samples = [_current_rss_bytes()]
    for expert in _SAMPLE_EXPERTS:
        measure_expert(_SOURCE_DIR, areader, _SAMPLE_LAYER, expert, seed=0)
        rss_samples.append(_current_rss_bytes())
    growth = max(rss_samples) - rss_samples[0]
    # One expert's three float weights + its W2/W3 copies are ~0.3 GiB; a full
    # shard is ~5 GiB and a bank is 476 GB. A <1.5 GiB envelope proves neither
    # a shard nor a bank is being materialised.
    assert growth < 1.5 * (1 << 30), f"RSS grew {growth / (1 << 30):.2f} GiB across experts"


def test_default_layer_sample_covers_ends_and_stride():
    layers = default_layer_sample(40, every=4)
    assert layers[0] == 0
    assert layers[-1] == 39
    assert 4 in layers and 8 in layers
    assert layers == sorted(set(layers))


# --- report assembly shape (synthetic-free structural check) -----------------
@pytest.mark.skipif(not (_have_source() and _have_artifact()), reason="source or W2 artifact not present")
def test_build_report_structure(tmp_path):
    report = build_report(
        _SOURCE_DIR,
        _ARTIFACT_DIR,
        layers=[0],
        experts=[0],
        engram_layer=1,
        engram_rows=[0, 42],
        seed=0,
    )
    assert set(report) >= {"headline", "ranking", "per_expert", "engram_w4", "mixed_bit_proposal"}
    assert report["headline"]["worst_expert"] is not None
    assert len(report["per_expert"]) == 1
    out = tmp_path / "quality_report.json"
    out.write_text(json.dumps(report))
    assert json.loads(out.read_text())["harness"].endswith("w2_quality.py")
