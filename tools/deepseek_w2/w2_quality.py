# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host-side W2 quality / mixed-bit sensitivity harness for DeepSeek V4.1 (E-risk).

This measures the *functional* degradation the 2-bit routed-expert format
(``w2_format.py`` / ``w2_convert.py``) introduces versus the FP4 (E2M1 + ue8m0)
source checkpoint, **without any Ascend hardware**. It is a host proxy: it does
not run the real fused-MoE kernel, it runs the mathematically equivalent
per-expert SwiGLU FFN in float and compares the FP4-source expert against the
already-converted W2 artifact expert on identical synthetic activations. The
weight difference is therefore isolated: both paths see the same inputs, so the
output delta is caused *only* by W2 requantisation.

What it produces
================
1. **Per-expert functional error** for a SAMPLE of experts (spread across
   layers): dequant the FP4 source expert (w1/w2/w3) and the W2 artifact expert
   to float, push a seeded synthetic activation batch (a few token profiles)
   through ``silu(x@w1.T) * (x@w3.T) @ w2.T`` BOTH ways, and report relative MSE
   and cosine similarity of the outputs, plus the raw per-tensor W2
   reconstruction error against the ``scale/2`` half-step bound. As a mixed-bit
   probe it *also* requantises the same source expert in-memory to each
   escalation grid (W3, W4) and measures their functional error, so the W2->Wk
   gains are real, not extrapolated.
2. **Engram W4 error**: sample rows of ``engram.embed`` and compare the W4
   artifact row reconstruction against the FP8 (E4M3) source row (relative MSE).
3. **Ranking + mixed-bit proposal**: rank experts by functional degradation and
   pick, per over-budget expert, the smallest escalation grid (W3, then W4) that
   meets the error budget; report the sample fractions and the host/HBM byte
   delta of that mixed allocation.

Bounded reads
=============
Every read is row-bounded through :class:`SafetensorsShardReader` (memory-mapped,
sliced by row). At most one expert's three weight matrices are materialised in
float at a time (~0.14 GB); a full expert bank (476 GB) or a full shard is never
loaded. Engram rows are read one row at a time from the 98 GB source table.

The numeric primitives are imported from the E1.1 modules, never reimplemented:
``dequant_mxfp4`` / ``dequant_fp8_e4m3`` / ``SafetensorsShardReader`` from
``w2_convert``; ``unpack_codes`` / ``dequantize_packed`` / ``quantize_weight`` /
``broadcast_block_scales`` from ``w2_format``.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from tools.deepseek_w2.w2_convert import (
    _ST_NP_DTYPE as _SHARD_NP_DTYPE,
)
from tools.deepseek_w2.w2_convert import (
    DEFAULT_CHUNK_ROWS,
    SafetensorsShardReader,
    dequant_fp8_e4m3,
    dequant_mxfp4,
)
from tools.deepseek_w2.w2_format import (
    W2_BITS,
    W2_BLOCK_COLS,
    W2_BLOCK_ROWS,
    W4_BITS,
    broadcast_block_scales,
    dequantize_packed,
    quantize_weight,
    unpack_codes,
)

# The W2/W4 artifact stores packed codes as unsigned bytes (safetensors dtype
# "U8"); the E1.1 reader's dtype map only knows the source "I8". Register it so
# the same bounded reader can slice the artifact shards. (Mutating the map, not
# the file: the reader is imported and never edited.)
_SHARD_NP_DTYPE.setdefault("U8", np.uint8)

# --- defaults ----------------------------------------------------------------
DEFAULT_SOURCE_DIR = "/run/media/matteius/20TB-drive/models/dealignai/DeepSeek-V4.1-Flash-UNCENSORED-FP8"
DEFAULT_ARTIFACT_DIR = "/run/media/matteius/20TB-drive/models/DeepSeek-V4.1-W2-310p"
INDEX_FILE = "model.safetensors.index.json"

# W3 (3-bit) grid the mixed-bit proposal escalates to. The packer stores 3-bit
# codes at 2 codes/byte (see w2_format.pack_codes), but the *quality* is the
# honest 3-bit grid {-4..3}; the byte estimate below models a bit-tight 3-bit
# store (3 bits/weight) as the achievable HBM footprint.
W3_BITS = 3

# In-memory escalation grids probed per expert (finest first target last). Each
# is a real requant of the *same* source weight, so the W2->Wk gain is measured,
# not extrapolated. W4 is included because a 5%-ish budget is typically out of
# reach for a naive 3-bit RTN grid (see the report), so the proposal needs the
# next rung to be honest about what actually meets budget.
ESCALATION_BITS: tuple[int, ...] = (3, 4)

# Default synthetic activation batch: a few token "profiles" over the hidden
# axis, all seeded. Plain float32 activations are used (NOT per-token INT8 act
# quant): both the FP4 and W2 experts see the *identical* activations, so the
# measured output delta isolates the weight-quantisation error, which is exactly
# what this harness is de-risking. The magnitudes bracket a plausible RMSNorm'd
# residual-stream scale so the SwiGLU non-linearity is exercised across regimes.
DEFAULT_ACT_PROFILES: tuple[tuple[str, int, float], ...] = (
    ("unit_normal", 8, 1.0),
    ("wide_normal", 8, 3.0),
    ("narrow_normal", 8, 0.25),
)

# Relative-MSE error budget the mixed-bit proposal must meet per expert.
DEFAULT_REL_MSE_BUDGET = 0.05


# --- artifact reader ---------------------------------------------------------
class ArtifactReader:
    """Weight-map-aware bounded reader over a sharded W2/W4 artifact directory.

    Resolves a tensor name to its shard via ``model.safetensors.index.json`` and
    caches one :class:`SafetensorsShardReader` per shard. All slicing is
    row-bounded through the underlying memory-mapped reader.
    """

    def __init__(self, artifact_dir: str | Path):
        self.dir = Path(artifact_dir)
        with (self.dir / INDEX_FILE).open() as handle:
            self.weight_map: dict[str, str] = json.load(handle)["weight_map"]
        self._readers: dict[str, SafetensorsShardReader] = {}

    def has(self, name: str) -> bool:
        return name in self.weight_map

    def reader_for(self, name: str) -> SafetensorsShardReader:
        shard = self.weight_map[name]
        reader = self._readers.get(shard)
        if reader is None:
            reader = self._readers[shard] = SafetensorsShardReader(self.dir / shard)
        return reader

    def shape(self, name: str) -> tuple[int, ...]:
        return self.reader_for(name).shape(name)

    def read_rows(self, name: str, row_start: int, row_end: int) -> torch.Tensor:
        return torch.from_numpy(self.reader_for(name).read_rows(name, row_start, row_end).copy())


# --- dequant helpers (bounded, row-chunked) ----------------------------------
def dequant_source_weight(
    reader: SafetensorsShardReader,
    weight_name: str,
    scale_name: str,
    chunk_rows: int = DEFAULT_CHUNK_ROWS,
) -> torch.Tensor:
    """Dequant one FP4 (mxfp4) source weight tensor to float32, row-chunked.

    Reads at most ``chunk_rows`` source rows at once; returns the whole expert
    weight (one expert, ~47 MB float32 — bounded, never a bank).
    """
    out_features = reader.shape(weight_name)[0]
    tiles: list[torch.Tensor] = []
    for row_start in range(0, out_features, chunk_rows):
        row_end = min(row_start + chunk_rows, out_features)
        w = torch.from_numpy(reader.read_rows(weight_name, row_start, row_end).copy())
        s = torch.from_numpy(reader.read_rows(scale_name, row_start, row_end).copy())
        tiles.append(dequant_mxfp4(w, s).to(torch.float32))
    return torch.cat(tiles, dim=0)


def dequant_w2_weight(
    areader: ArtifactReader,
    stem: str,
    n_bits: int = W2_BITS,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dequant a W2/W4 artifact weight (``{stem}_codes`` + ``{stem}_scale``).

    Returns ``(weight_float32, block_scale_float32)`` where the weight is
    ``[out, in]`` and the block scale is ``[out/32, in/32]``.
    """
    codes_name = f"{stem}_codes"
    scale_name = f"{stem}_scale"
    out_features = areader.shape(codes_name)[0]
    codes_per_byte = 8 // n_bits
    in_features = areader.shape(codes_name)[1] * codes_per_byte
    codes = areader.read_rows(codes_name, 0, out_features)
    scale = areader.read_rows(scale_name, 0, areader.shape(scale_name)[0]).to(torch.float32)
    weight = dequantize_packed(codes, scale, out_features, in_features, n_bits).to(torch.float32)
    return weight, scale


# --- functional error metrics ------------------------------------------------
def make_activations(
    hidden: int,
    seed: int = 0,
    profiles: tuple[tuple[str, int, float], ...] = DEFAULT_ACT_PROFILES,
) -> dict[str, torch.Tensor]:
    """Seeded synthetic activation batches, one ``[T, hidden]`` per profile."""
    generator = torch.Generator().manual_seed(seed)
    acts: dict[str, torch.Tensor] = {}
    for name, tokens, scale in profiles:
        acts[name] = torch.randn(tokens, hidden, generator=generator, dtype=torch.float32) * scale
    return acts


def swiglu_ffn(x: torch.Tensor, w1: torch.Tensor, w3: torch.Tensor, w2: torch.Tensor) -> torch.Tensor:
    """DeepSeek routed-expert SwiGLU FFN: ``silu(x@w1.T) * (x@w3.T) @ w2.T``.

    ``w1`` = gate ``[inter, hidden]``, ``w3`` = up ``[inter, hidden]``,
    ``w2`` = down ``[hidden, inter]``; ``x`` = ``[T, hidden]``.
    """
    gate = x @ w1.transpose(0, 1)
    up = x @ w3.transpose(0, 1)
    hidden = torch.nn.functional.silu(gate) * up
    return hidden @ w2.transpose(0, 1)


def _rel_mse(reference: torch.Tensor, other: torch.Tensor) -> float:
    """Relative MSE ``||other - reference||^2 / ||reference||^2`` (Frobenius)."""
    denom = float((reference.double() ** 2).sum())
    if denom == 0.0:
        return 0.0
    return float(((other.double() - reference.double()) ** 2).sum() / denom)


def _mean_cosine(reference: torch.Tensor, other: torch.Tensor) -> float:
    """Mean per-token cosine similarity between two ``[T, H]`` output batches."""
    cos = torch.nn.functional.cosine_similarity(reference.double(), other.double(), dim=-1)
    return float(cos.mean())


def functional_error(
    source: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    recon: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    activations: dict[str, torch.Tensor],
) -> dict:
    """Functional error of a reconstructed expert vs the source expert.

    ``source`` / ``recon`` are ``(w1, w3, w2)`` float tensors. Returns per-profile
    relative MSE + cosine and the profile-mean aggregates used for ranking.
    """
    w1_s, w3_s, w2_s = source
    w1_r, w3_r, w2_r = recon
    per_profile: dict[str, dict[str, float]] = {}
    for name, x in activations.items():
        out_s = swiglu_ffn(x, w1_s, w3_s, w2_s)
        out_r = swiglu_ffn(x, w1_r, w3_r, w2_r)
        per_profile[name] = {
            "rel_mse": _rel_mse(out_s, out_r),
            "cosine": _mean_cosine(out_s, out_r),
        }
    rel_mse_mean = float(np.mean([p["rel_mse"] for p in per_profile.values()]))
    cosine_mean = float(np.mean([p["cosine"] for p in per_profile.values()]))
    return {
        "per_profile": per_profile,
        "rel_mse_mean": rel_mse_mean,
        "cosine_mean": cosine_mean,
    }


def tensor_recon_stats(
    w_source: torch.Tensor,
    w_recon: torch.Tensor,
    block_scale: torch.Tensor,
    block_rows: int = W2_BLOCK_ROWS,
    block_cols: int = W2_BLOCK_COLS,
) -> dict:
    """Raw per-tensor reconstruction error vs the ``scale/2`` half-step bound.

    The W2 format guarantees ``|w - dequant(quant(w))| <= block_scale / 2`` per
    element. ``max_err_over_bound`` should be ``<= 1`` (up to float32 scale
    rounding); a value materially above 1 would mean the artifact scale does not
    match the source weight (a real defect this proxy would catch).
    """
    out_features, in_features = w_source.shape
    err = (w_source.double() - w_recon.double()).abs()
    bound = broadcast_block_scales(block_scale, out_features, in_features, block_rows, block_cols) / 2.0
    ratio = err / bound.clamp_min(torch.finfo(torch.float64).tiny)
    return {
        "max_abs_err": float(err.max()),
        "rms_err": float((err**2).mean().sqrt()),
        "half_step_bound_max": float(bound.max()),
        "max_err_over_bound": float(ratio.max()),
    }


# --- per-expert sampling -----------------------------------------------------
def _requant_expert(
    src: dict[str, torch.Tensor],
    bits: int,
) -> dict[str, torch.Tensor]:
    """In-memory requant of a source expert's (w1, w2, w3) to a ``bits`` grid."""
    out: dict[str, torch.Tensor] = {}
    for proj, w in src.items():
        codes, bscale = quantize_weight(w, bits, W2_BLOCK_ROWS, W2_BLOCK_COLS)
        full = broadcast_block_scales(bscale, w.shape[0], w.shape[1])
        out[proj] = (codes.double() * full).to(torch.float32)
    return out


@dataclass
class ExpertSample:
    """One sampled expert's measured errors: W2 functional, escalation probes, raw.

    ``escalation`` maps a bit width (e.g. 3, 4) to that grid's functional-error
    dict. ``w3_functional`` mirrors ``escalation[3]`` for convenience.
    """

    layer: int
    expert: int
    w2_functional: dict = field(default_factory=dict)
    w3_functional: dict = field(default_factory=dict)
    escalation: dict = field(default_factory=dict)
    tensor_recon: dict = field(default_factory=dict)

    def as_json(self) -> dict:
        return {
            "layer": self.layer,
            "expert": self.expert,
            "w2_functional": self.w2_functional,
            "w3_functional": self.w3_functional,
            "escalation": {str(bits): fe for bits, fe in self.escalation.items()},
            "tensor_recon": self.tensor_recon,
        }


def measure_expert(
    source_dir: str | Path,
    areader: ArtifactReader,
    layer: int,
    expert: int,
    seed: int = 0,
    profiles: tuple[tuple[str, int, float], ...] = DEFAULT_ACT_PROFILES,
    escalation_bits: tuple[int, ...] = ESCALATION_BITS,
    chunk_rows: int = DEFAULT_CHUNK_ROWS,
) -> ExpertSample:
    """Measure one real sampled expert: W2 functional error + escalation probes + raw.

    Dequants the FP4 source and the W2 artifact expert (bounded), runs the SwiGLU
    proxy both ways, requantises the *same* source expert to each escalation grid
    in-memory (so the W2->Wk gain is measured, not extrapolated), and records
    per-tensor reconstruction stats against the ``scale/2`` bound.
    """
    source_dir = Path(source_dir)
    weight_map = json.load((source_dir / INDEX_FILE).open())["weight_map"]

    src: dict[str, torch.Tensor] = {}
    w2art: dict[str, torch.Tensor] = {}
    art_scale: dict[str, torch.Tensor] = {}
    for proj in ("w1", "w2", "w3"):
        wname = f"layers.{layer}.ffn.experts.{expert}.{proj}.weight"
        sname = f"layers.{layer}.ffn.experts.{expert}.{proj}.scale"
        reader = SafetensorsShardReader(source_dir / weight_map[wname])
        src[proj] = dequant_source_weight(reader, wname, sname, chunk_rows)
        stem = f"layers.{layer}.ffn.experts.{expert}.{proj}"
        w_w2, scale_w2 = dequant_w2_weight(areader, stem, W2_BITS)
        w2art[proj] = w_w2
        art_scale[proj] = scale_w2

    hidden = src["w1"].shape[1]
    activations = make_activations(hidden, seed=seed, profiles=profiles)
    source_triple = (src["w1"], src["w3"], src["w2"])
    sample = ExpertSample(layer=layer, expert=expert)
    sample.w2_functional = functional_error(source_triple, (w2art["w1"], w2art["w3"], w2art["w2"]), activations)
    for bits in escalation_bits:
        rq = _requant_expert(src, bits)
        sample.escalation[bits] = functional_error(source_triple, (rq["w1"], rq["w3"], rq["w2"]), activations)
    sample.w3_functional = sample.escalation.get(W3_BITS, {})
    sample.tensor_recon = {
        proj: tensor_recon_stats(src[proj], w2art[proj], art_scale[proj]) for proj in ("w1", "w2", "w3")
    }
    return sample


# --- engram W4 row error -----------------------------------------------------
def _engram_parts(areader: ArtifactReader, layer: int) -> tuple[list[str], list[int], list[str]]:
    """Ordered ``embed_codes`` part names, their cumulative row starts, and the
    matching ``embed_scale`` part names for one engram layer."""
    prefix = f"layers.{layer}.engram.embed_codes.p"
    parts = sorted(
        (name for name in areader.weight_map if name.startswith(prefix)),
        key=lambda n: int(n.rsplit(".p", 1)[1]),
    )
    starts: list[int] = []
    running = 0
    for name in parts:
        starts.append(running)
        running += areader.shape(name)[0]
    scale_parts = [name.replace("embed_codes", "embed_scale") for name in parts]
    return parts, starts, scale_parts


def engram_row_error(
    source_dir: str | Path,
    areader: ArtifactReader,
    layer: int,
    rows: list[int],
) -> dict:
    """Per-row W4 reconstruction error of ``engram.embed`` vs the FP8 source.

    Reads one row at a time from both the 98 GB FP8 source table and the W4
    artifact parts (fully bounded). Returns per-row relative MSE and aggregates.
    """
    source_dir = Path(source_dir)
    weight_map = json.load((source_dir / INDEX_FILE).open())["weight_map"]
    src_wname = f"layers.{layer}.engram.embed.weight"
    src_sname = f"layers.{layer}.engram.embed.scale"
    src_reader = SafetensorsShardReader(source_dir / weight_map[src_wname])

    parts, starts, scale_parts = _engram_parts(areader, layer)
    part_rows = [areader.shape(name)[0] for name in parts]
    scale_rows_per_part = [areader.shape(name)[0] for name in scale_parts]
    block_rows = part_rows[0] // scale_rows_per_part[0]
    in_features = areader.shape(parts[0])[1] * (8 // W4_BITS)

    per_row: list[dict] = []
    for global_row in rows:
        # Source FP8 row -> float.
        w_src = torch.from_numpy(src_reader.read_rows(src_wname, global_row, global_row + 1).copy())
        s_src = torch.from_numpy(src_reader.read_rows(src_sname, global_row, global_row + 1).copy())
        row_src = dequant_fp8_e4m3(w_src, s_src).to(torch.float32)
        # W4 artifact row -> float (find its part, sign-extend + block-scale).
        part_idx = max(i for i, start in enumerate(starts) if start <= global_row)
        local = global_row - starts[part_idx]
        codes_row = areader.read_rows(parts[part_idx], local, local + 1)
        scale_row = areader.read_rows(scale_parts[part_idx], local // block_rows, local // block_rows + 1)
        codes = unpack_codes(codes_row, in_features, W4_BITS).double()
        full = scale_row.double().repeat_interleave(W2_BLOCK_COLS, dim=1)[:, :in_features]
        row_w4 = (codes * full).to(torch.float32)
        per_row.append({"row": int(global_row), "rel_mse": _rel_mse(row_src, row_w4)})

    rel = [r["rel_mse"] for r in per_row]
    return {
        "layer": layer,
        "num_rows": len(per_row),
        "per_row": per_row,
        "rel_mse_mean": float(np.mean(rel)) if rel else 0.0,
        "rel_mse_max": float(np.max(rel)) if rel else 0.0,
    }


# --- ranking + mixed-bit proposal --------------------------------------------
def _routed_expert_element_count(areader: ArtifactReader) -> tuple[int, int]:
    """Total routed-expert weight elements + number of experts in the artifact.

    Derived from the artifact index: counts every ``*_codes`` expert tensor and
    multiplies by its element count (all experts share geometry). Bounded — only
    shard headers are touched.
    """
    stems = sorted(
        {name[: -len("_codes")] for name in areader.weight_map if ".ffn.experts." in name and name.endswith("_codes")}
    )
    total = 0
    expert_ids: set[tuple[int, int]] = set()
    per_stem_elems: dict[str, int] = {}
    for stem in stems:
        codes_name = f"{stem}_codes"
        out_features, packed_cols = areader.shape(codes_name)
        elems = out_features * packed_cols * (8 // W2_BITS)
        total += elems
        per_stem_elems[stem] = elems
        parts = stem.split(".")
        expert_ids.add((int(parts[1]), int(parts[4])))
    return total, len(expert_ids)


def build_ranking(samples: list[ExpertSample]) -> list[dict]:
    """Rank sampled experts by W2 functional relative MSE (worst first).

    Each row carries the W2 error, the W3 error (kept as a named column for
    convenience), and an ``escalation`` map ``{bits: rel_mse}`` over every probed
    grid so the proposal can pick the minimal bit width that meets budget.
    """
    rows = [
        {
            "layer": s.layer,
            "expert": s.expert,
            "w2_rel_mse": s.w2_functional["rel_mse_mean"],
            "w2_cosine": s.w2_functional["cosine_mean"],
            "w3_rel_mse": s.w3_functional.get("rel_mse_mean", float("nan")),
            "w3_cosine": s.w3_functional.get("cosine_mean", float("nan")),
            "escalation": {bits: fe["rel_mse_mean"] for bits, fe in s.escalation.items()},
        }
        for s in samples
    ]
    return sorted(rows, key=lambda r: r["w2_rel_mse"], reverse=True)


def _min_bits_meeting_budget(row: dict, budget: float) -> int | None:
    """Smallest probed escalation bit width whose rel MSE meets ``budget``.

    Returns ``2`` if W2 already meets budget, the minimal escalation width if one
    does, or ``None`` if no probed grid meets it.
    """
    if row["w2_rel_mse"] <= budget:
        return W2_BITS
    for bits in sorted(row.get("escalation", {})):
        if row["escalation"][bits] <= budget:
            return bits
    return None


def mixed_bit_proposal(
    ranking: list[dict],
    areader: ArtifactReader,
    budget: float = DEFAULT_REL_MSE_BUDGET,
) -> dict:
    """Propose a minimal-bit-width escalation for experts over the error budget.

    A sampled expert is "over budget" if its W2 relative MSE exceeds ``budget``.
    For each such expert the proposal picks the smallest probed grid (W3, then
    W4) that meets budget; experts no probed grid rescues are flagged
    unresolved. The per-bit fractions of the sample estimate the full-model
    allocation; the byte delta sums ``elems * (bits - 2) / 8`` over the escalated
    fractions (bit-tight store), i.e. the extra HBM/host bytes vs all-W2.
    """
    n = len(ranking)
    total_elems, num_experts = _routed_expert_element_count(areader)
    routed_w2_bytes = total_elems * W2_BITS / 8

    # Per-expert chosen bit width (2 = stays W2; None = unresolved by any probe).
    chosen = [_min_bits_meeting_budget(r, budget) for r in ranking]
    over_budget = [r for r, b in zip(ranking, chosen) if b != W2_BITS]
    unresolved = [r for r, b in zip(ranking, chosen) if b is None]

    # Fraction of the sample escalated to each width, and the mean extra bits.
    escalation_bits_seen = sorted({b for b in chosen if b not in (None, W2_BITS)})
    fraction_by_bits = {bits: sum(1 for b in chosen if b == bits) / n if n else 0.0 for bits in escalation_bits_seen}
    # Unresolved experts are costed at the finest probed grid (best effort).
    finest = max(ranking[0]["escalation"]) if ranking and ranking[0].get("escalation") else W3_BITS
    extra_bits_per_weight = sum((bits - W2_BITS) * frac for bits, frac in fraction_by_bits.items()) + (
        len(unresolved) / n if n else 0.0
    ) * (finest - W2_BITS)
    byte_delta = total_elems * extra_bits_per_weight / 8

    frac_over = len(over_budget) / n if n else 0.0
    worst_10pct_cutoff = max(1, int(round(0.10 * n))) if n else 0
    return {
        "rel_mse_budget": budget,
        "sampled_experts": n,
        "num_over_budget": len(over_budget),
        "fraction_over_budget": frac_over,
        "escalation_fraction_by_bits": {str(bits): frac for bits, frac in fraction_by_bits.items()},
        "num_unresolved_by_any_probe": len(unresolved),
        "budget_met_by_escalation": len(unresolved) == 0 and frac_over > 0.0,
        "worst_10pct_experts": [
            {"layer": r["layer"], "expert": r["expert"], "w2_rel_mse": r["w2_rel_mse"]}
            for r in ranking[:worst_10pct_cutoff]
        ],
        "model_routed_experts": num_experts,
        "model_routed_expert_elements": total_elems,
        "model_routed_w2_bytes": routed_w2_bytes,
        "model_routed_w2_gib": routed_w2_bytes / (1 << 30),
        "mean_extra_bits_per_weight": extra_bits_per_weight,
        "escalation_byte_delta_bytes": byte_delta,
        "escalation_byte_delta_gib": byte_delta / (1 << 30),
        "escalation_note": (
            "byte delta assumes a bit-tight store (k bits/weight); today's "
            "pack_codes writes W3 at 2 codes/byte (4 bits/weight) and W4 at "
            "2 codes/byte (4 bits/weight), so a W3 escalation costs the same "
            "on-disk bytes as W4 with the current packer."
        ),
    }


# --- report orchestration ----------------------------------------------------
def default_layer_sample(num_layers: int, every: int = 4) -> list[int]:
    """Every ``every``-th layer plus the first and last layer, sorted-unique."""
    layers = set(range(0, num_layers, every)) | {0, num_layers - 1}
    return sorted(layers)


def build_report(
    source_dir: str | Path,
    artifact_dir: str | Path,
    layers: list[int],
    experts: list[int],
    engram_layer: int | None,
    engram_rows: list[int],
    seed: int = 0,
    profiles: tuple[tuple[str, int, float], ...] = DEFAULT_ACT_PROFILES,
    budget: float = DEFAULT_REL_MSE_BUDGET,
    chunk_rows: int = DEFAULT_CHUNK_ROWS,
) -> dict:
    """Run the full harness over a sample and assemble the JSON report dict."""
    areader = ArtifactReader(artifact_dir)
    samples: list[ExpertSample] = []
    for layer in layers:
        for expert in experts:
            samples.append(
                measure_expert(source_dir, areader, layer, expert, seed=seed, profiles=profiles, chunk_rows=chunk_rows)
            )

    ranking = build_ranking(samples)
    proposal = mixed_bit_proposal(ranking, areader, budget=budget)

    engram = None
    if engram_layer is not None and engram_rows:
        engram = engram_row_error(source_dir, areader, engram_layer, engram_rows)

    rel_all = [r["w2_rel_mse"] for r in ranking]
    cos_all = [r["w2_cosine"] for r in ranking]
    escalation_medians = {}
    for bits in sorted({b for r in ranking for b in r.get("escalation", {})}):
        vals = [r["escalation"][bits] for r in ranking if bits in r["escalation"]]
        escalation_medians[str(bits)] = float(np.median(vals)) if vals else None
    headline = {
        "worst_expert": ranking[0] if ranking else None,
        "best_expert": ranking[-1] if ranking else None,
        "w2_rel_mse_median": float(np.median(rel_all)) if rel_all else None,
        "w2_rel_mse_mean": float(np.mean(rel_all)) if rel_all else None,
        "w2_cosine_median": float(np.median(cos_all)) if cos_all else None,
        "escalation_rel_mse_median_by_bits": escalation_medians,
        "engram_w4_rel_mse_mean": engram["rel_mse_mean"] if engram else None,
    }
    return {
        "harness": "tools/deepseek_w2/w2_quality.py",
        "proxy_note": (
            "Host-side float SwiGLU proxy of the routed-expert MoE; identical "
            "seeded activations through FP4-source vs W2-artifact experts isolate "
            "the weight-quantisation error. Not the real Ascend fused-MoE kernel: "
            "final quality still needs a runtime/rental to confirm."
        ),
        "activation_note": (
            "Plain float32 activations (no per-token INT8 act quant); both paths "
            "share them so the output delta is purely the weight-quant effect."
        ),
        "config": {
            "source_dir": str(source_dir),
            "artifact_dir": str(artifact_dir),
            "sample_layers": layers,
            "sample_experts": experts,
            "seed": seed,
            "profiles": [{"name": n, "tokens": t, "scale": s} for (n, t, s) in profiles],
            "rel_mse_budget": budget,
        },
        "headline": headline,
        "ranking": ranking,
        "per_expert": [s.as_json() for s in samples],
        "engram_w4": engram,
        "mixed_bit_proposal": proposal,
    }


def _render_markdown(report: dict) -> str:
    """Short markdown summary with the headline numbers and the proposal."""
    h = report["headline"]
    p = report["mixed_bit_proposal"]
    e = report.get("engram_w4")
    worst = h["worst_expert"]
    lines = [
        "# DeepSeek V4.1 W2 quality / mixed-bit host proxy",
        "",
        "> Host-side float SwiGLU proxy. Identical seeded activations run through the",
        "> FP4 source expert and the converted W2 artifact expert; the output delta is",
        "> the weight-quantisation error. **Not** the real Ascend fused-MoE kernel —",
        "> final quality still needs a runtime/rental.",
        "",
        "## Headline",
        "",
        "| metric | value |",
        "| --- | --- |",
        f"| sampled experts | {p['sampled_experts']} |",
        f"| worst-expert W2 rel MSE | {worst['w2_rel_mse']:.4g} (L{worst['layer']} E{worst['expert']}, "
        f"cos {worst['w2_cosine']:.4f}) |",
        f"| median W2 rel MSE | {h['w2_rel_mse_median']:.4g} |",
        f"| mean W2 rel MSE | {h['w2_rel_mse_mean']:.4g} |",
        f"| median W2 cosine | {h['w2_cosine_median']:.4f} |",
    ]
    for bits, med in h.get("escalation_rel_mse_median_by_bits", {}).items():
        if med is not None:
            lines.append(f"| median W{bits} rel MSE (in-memory probe) | {med:.4g} |")
    if e:
        lines.append(f"| Engram W4 rel MSE (mean/max) | {e['rel_mse_mean']:.4g} / {e['rel_mse_max']:.4g} |")
    frac_by_bits = ", ".join(
        f"W{bits}: {frac * 100:.1f}%" for bits, frac in p.get("escalation_fraction_by_bits", {}).items()
    )
    lines += [
        "",
        "## Mixed-bit proposal",
        "",
        f"- Error budget: relative MSE <= **{p['rel_mse_budget']}**.",
        f"- Over budget at W2 in sample: **{p['num_over_budget']}/{p['sampled_experts']}** "
        f"(~{p['fraction_over_budget'] * 100:.1f}% of experts).",
        f"- Minimal escalation that meets budget (sample fractions): **{frac_by_bits or 'none'}**; "
        f"unresolved by any probed grid: **{p['num_unresolved_by_any_probe']}** "
        f"(budget fully met by escalation: **{p['budget_met_by_escalation']}**).",
        f"- Routed-expert W2 footprint: **{p['model_routed_w2_gib']:.1f} GiB** ({p['model_routed_experts']} experts).",
        f"- Mixed-bit escalation byte delta: **+{p['escalation_byte_delta_gib']:.2f} GiB** "
        f"(mean +{p['mean_extra_bits_per_weight']:.2f} bits/weight, bit-tight store).",
        f"- Note: {p['escalation_note']}",
        "",
        "## Verdict",
        "",
        _verdict(report),
        "",
    ]
    return "\n".join(lines)


def _verdict(report: dict) -> str:
    """One-paragraph host-proxy verdict (honest about proxy limits)."""
    h = report["headline"]
    p = report["mixed_bit_proposal"]
    frac = p["fraction_over_budget"]
    median = h["w2_rel_mse_median"]
    if frac == 0.0:
        stance = (
            f"Every sampled expert meets the {p['rel_mse_budget']} relative-MSE budget on this host proxy "
            f"(median {median:.3g}); **W2 looks viable** for the routed experts with no escalation needed."
        )
    elif p["budget_met_by_escalation"]:
        stance = (
            f"~{frac * 100:.0f}% of sampled experts exceed the {p['rel_mse_budget']} budget at W2, but a minimal "
            f"per-expert bit escalation (+{p['escalation_byte_delta_gib']:.1f} GiB) brings every one back under it; "
            f"**W2 + targeted higher-bit experts looks viable**."
        )
    else:
        stance = (
            f"~{frac * 100:.0f}% of sampled experts exceed the {p['rel_mse_budget']} budget at W2 and "
            f"{p['num_unresolved_by_any_probe']} are not rescued by any probed grid (W3/W4); "
            f"**naive round-to-nearest W2 looks risky** here — calibration (GPTQ/AWQ-style), not just wider bits, "
            f"is likely the real mitigation, and a runtime/rental check is needed."
        )
    return (
        stance + " This is a functional weight-error proxy only: it does not capture the INT8 activation-quant path, "
        "routing, or end-to-end task accuracy, so the final go/no-go still needs a runtime or rental run."
    )


def _cli() -> None:
    parser = argparse.ArgumentParser(description="DeepSeek V4.1 W2 quality / mixed-bit host proxy")
    parser.add_argument("--source-dir", default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--artifact-dir", default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--layers", type=int, nargs="+", default=None, help="explicit layer sample")
    parser.add_argument("--every", type=int, default=4, help="sample every Nth layer when --layers absent")
    parser.add_argument("--num-layers", type=int, default=40, help="layer count for the default sample")
    parser.add_argument("--experts", type=int, nargs="+", default=[0, 191, 383])
    parser.add_argument("--engram-layer", type=int, default=1)
    parser.add_argument("--engram-rows", type=int, default=64, help="number of engram rows to sample")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--budget", type=float, default=DEFAULT_REL_MSE_BUDGET)
    parser.add_argument("--out-json", default="artifacts/deepseek-v41-w2/quality_report.json")
    parser.add_argument("--out-md", default="artifacts/deepseek-v41-w2/quality_report.md")
    args = parser.parse_args()

    layers = args.layers if args.layers is not None else default_layer_sample(args.num_layers, args.every)
    generator = np.random.default_rng(args.seed)
    areader = ArtifactReader(args.artifact_dir)
    engram_layer = args.engram_layer
    engram_rows: list[int] = []
    if engram_layer is not None and f"layers.{engram_layer}.engram.embed_codes.p0" in areader.weight_map:
        _, starts, _ = _engram_parts(areader, engram_layer)
        total_rows = starts[-1] + areader.shape(f"layers.{engram_layer}.engram.embed_codes.p{len(starts) - 1}")[0]
        engram_rows = sorted(int(r) for r in generator.integers(0, total_rows, size=args.engram_rows))
    else:
        engram_layer = None

    report = build_report(
        args.source_dir,
        args.artifact_dir,
        layers,
        args.experts,
        engram_layer,
        engram_rows,
        seed=args.seed,
        budget=args.budget,
    )

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w") as handle:
        json.dump(report, handle, indent=2)
    Path(args.out_md).write_text(_render_markdown(report))
    print(f"wrote {out_json} and {args.out_md}")
    print(json.dumps(report["headline"], indent=2))
    print(json.dumps(report["mixed_bit_proposal"], indent=2))


if __name__ == "__main__":
    _cli()


__all__ = [
    "DEFAULT_SOURCE_DIR",
    "DEFAULT_ARTIFACT_DIR",
    "DEFAULT_ACT_PROFILES",
    "DEFAULT_REL_MSE_BUDGET",
    "W3_BITS",
    "ArtifactReader",
    "ExpertSample",
    "dequant_source_weight",
    "dequant_w2_weight",
    "make_activations",
    "swiglu_ffn",
    "functional_error",
    "tensor_recon_stats",
    "measure_expert",
    "engram_row_error",
    "build_ranking",
    "mixed_bit_proposal",
    "default_layer_sample",
    "build_report",
]
