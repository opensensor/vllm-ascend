#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Full-model resumable FP8 -> W2/FP16 converter for GLM-5.3-Flash (glm5_next).

Produces the W2-on-310P artifact for GLM-5.3-Flash. It is a *thin* GLM-specific
driver over the already-validated DeepSeek W2 machinery: the packing/block-scale
primitives (:mod:`tools.deepseek_w2.w2_format`), the memory-mapped shard reader +
``ConversionStats`` (:mod:`tools.deepseek_w2.w2_convert`), and the resumable
sharding / atomic-IO / page-cache / manifest helpers
(:mod:`tools.deepseek_w2.convert_full`) are all *imported*, never reimplemented.

GLM-5.3-Flash is much simpler than DeepSeek V4.1: **no Engram, no indexer table,
no MXFP4**. Only the routed MoE experts are quantised; everything else is either
dequantised FP8 or cast to FP16.

The one piece the DeepSeek machinery does not already provide is the source
dequant: GLM's FP8 checkpoint uses a **plain float32 per-[128,128]-block scale**
(``weight_block_size = [128, 128]``), NOT the ue8m0 power-of-two exponent scale
DeepSeek uses. :func:`dequant_fp8_e4m3_f32block` supplies exactly that -- decode
the ``F8_E4M3`` code, broadcast the plain F32 block scale, multiply.

Family -> target precision routing
==================================
======================================  ======================  =========  ================================
family                                  source (dtype)          target     how
======================================  ======================  =========  ================================
routed expert gate/up/down_proj.weight  F8_E4M3 + F32 block     W2         dequant_fp8_e4m3_f32block -> 2-bit
  ``mlp.experts.{E}.*_proj.weight``
other F8_E4M3 weights                    F8_E4M3 + F32 block     FP16       dequant_fp8_e4m3_f32block -> f16
  (dense MLP of first_k_dense layers,
   shared_experts, MLA kv_a_proj_with_mqa
   / o_proj / q_a_proj / q_b_proj)
everything else                          BF16 / F32              FP16       cast -> float16
  (linear-attn, norms, router gate,
   hyper-connection, embed, lm_head,
   MTP eh_proj/enorm/hnorm/shared_head)
``.weight_scale_inv``                    F32                     SCALE      consumed with its weight
``model.visual.*``                       (any)                   EXCLUDE    recorded excluded, not written
======================================  ======================  =========  ================================

Output layout, resumability, atomic shard IO, per-shard sha256, ``progress.json``
and the ``POSIX_FADV_DONTNEED`` page-cache release are identical to the DeepSeek
driver (reused verbatim); see :mod:`tools.deepseek_w2.convert_full`.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from tools.deepseek_w2.convert_full import (
    DEFAULT_CHUNK_ROWS,
    INDEX_FILE,
    MANIFEST_FILE,
    PROGRESS_FILE,
    ConvItem,
    OutputTensor,
    RunResult,
    SafetensorsShardReader,
    _codes_bytes,
    _copy_side_files,
    _dense_part_rows,
    _drop_page_cache,
    _load_headers,
    _load_progress,
    _native_from_rows,
    _packed_part_rows,
    _part_ranges,
    _read_full_native,
    _save_shard_atomic,
    _scale_bytes,
    _sha256_file,
    _shard_name,
    _suffix,
    _write_json_atomic,
    bin_shards,
    current_rss_bytes,
)
from tools.deepseek_w2.w2_convert import ConversionStats
from tools.deepseek_w2.w2_format import (
    W2_BITS,
    W2_BLOCK_COLS,
    W2_BLOCK_ROWS,
    W4_BITS,
    pack_codes,
    quantize_weight,
)
from tools.glm_w2.build_manifest import classify

import re as _re


def _expert_layer_index(name: str) -> int | None:
    """Decoder layer index for a ``...layers.{L}.mlp.experts...`` tensor, else None."""
    m = _re.search(r"\.layers\.(\d+)\.", name)
    return int(m.group(1)) if m else None


def _routed_expert_bits(name: str, default_bits: int, w4_layers: frozenset[int]) -> int:
    """Per-layer code width for a routed-expert weight: W4 for layers in
    ``w4_layers`` (or when the global default is 4), else ``default_bits``."""
    if default_bits == W4_BITS:
        return W4_BITS
    li = _expert_layer_index(name)
    return W4_BITS if (li is not None and li in w4_layers) else default_bits

# --- defaults ----------------------------------------------------------------
DEFAULT_SOURCE_DIR = "/run/media/matteius/20TB-drive/models/GLM-5.3-Flash-FP8"
DEFAULT_OUT_DIR = "/run/media/matteius/20TB-drive/models/GLM-5.3-Flash-W2-310p"
DEFAULT_SHARD_TARGET_BYTES = 5 * (1 << 30)  # ~5 GB output shards

# GLM FP8 checkpoint uses a fixed [128, 128] weight block (config
# ``weight_block_size``); the scale is a PLAIN float32 per-block value.
GLM_FP8_BLOCK_ROWS = 128
GLM_FP8_BLOCK_COLS = 128

# Source-format tags for GLM (distinct from DeepSeek's ue8m0 "fp8_e4m3").
FMT_FP8_F32BLOCK = "fp8_e4m3_f32block"
FMT_CAST = "cast"

_SCALE_SUFFIX = ".weight_scale_inv"


# --- source-format dequant: F8_E4M3 + plain F32 [128,128] block scale --------
def dequant_fp8_e4m3_f32block(
    weight_fp8: torch.Tensor,
    scale_f32: torch.Tensor,
    block: tuple[int, int] = (GLM_FP8_BLOCK_ROWS, GLM_FP8_BLOCK_COLS),
) -> torch.Tensor:
    """Dequantise ``F8_E4M3`` weights with a plain float32 per-block scale.

    This is the GLM analogue of :func:`tools.deepseek_w2.w2_convert.dequant_fp8_e4m3`,
    but the scale is used **directly** as a float32 multiplier (GLM's
    ``weight_scale_inv``), not decoded from a ue8m0 power-of-two exponent.

    Args:
        weight_fp8: ``[R, C]`` weight. ``uint8`` / ``int8`` bytes are bit-cast to
            ``float8_e4m3fn``; a ``float8_e4m3fn`` tensor is used as-is.
        scale_f32: ``[ceil(R / block_rows), ceil(C / block_cols)]`` float32 block
            scales (one scalar per ``[block_rows, block_cols]`` weight block).
        block: ``(block_rows, block_cols)`` -- ``[128, 128]`` for GLM-5.3-Flash.

    Returns:
        ``[R, C]`` float32 dequantised weight = ``e4m3_value * f32_block_scale``.
    """
    if weight_fp8.dtype in (torch.uint8, torch.int8):
        weight = weight_fp8.view(torch.float8_e4m3fn).to(torch.float32)
    else:
        weight = weight_fp8.to(torch.float32)
    rows, cols = weight.shape
    block_rows, block_cols = block
    s_rows, s_cols = scale_f32.shape
    if s_rows < math.ceil(rows / block_rows) or s_cols < math.ceil(cols / block_cols):
        raise ValueError(
            f"scale grid {(s_rows, s_cols)} too small for weight {(rows, cols)} at block {(block_rows, block_cols)}"
        )
    scale = scale_f32.to(torch.float32)
    full = scale.repeat_interleave(block_rows, dim=0).repeat_interleave(block_cols, dim=1)[:rows, :cols]
    return weight * full


# --- routing -----------------------------------------------------------------
@dataclass
class Route:
    """The target precision + source handling for one source weight tensor."""

    target: str  # "W2" | "FP16" | "EXCLUDE" | "SCALE"
    source_format: str  # "fp8_e4m3_f32block" | "cast" | ""


def route_tensor(
    name: str,
    dtype: str,
    default_expert_bits: int = W2_BITS,
    w4_layers: frozenset[int] = frozenset(),
) -> Route:
    """Route one tensor to its target precision from family + header dtype.

    ``SCALE`` (a ``.weight_scale_inv`` companion) is consumed with its weight and
    never emitted standalone. ``EXCLUDE`` covers the vision tower (text-only).
    Routed experts go to ``"W2"`` or ``"W4"`` per :func:`_routed_expert_bits`
    (mixed precision: W4 for the listed/most-sensitive deep layers, W2 elsewhere).
    """
    family = classify(name)
    if family == "vision":
        return Route("EXCLUDE", "")
    if name.endswith(_SCALE_SUFFIX):
        return Route("SCALE", "")
    if family == "routed_expert_weight":
        bits = _routed_expert_bits(name, default_expert_bits, w4_layers)
        return Route("W4" if bits == W4_BITS else "W2", FMT_FP8_F32BLOCK)
    # Everything else deploys at FP16: dequantise F8_E4M3 blocks, else cast.
    return Route("FP16", FMT_FP8_F32BLOCK if dtype == "F8_E4M3" else FMT_CAST)


# --- planning ----------------------------------------------------------------
def plan_items(
    weight_map: dict[str, str], headers: dict[str, dict], target_bytes: int,
    default_expert_bits: int = W2_BITS, w4_layers: frozenset[int] = frozenset(),
) -> tuple[list[ConvItem], list[dict]]:
    """Turn the source weight-map into a deterministic list of conversion items.

    Returns ``(items, excluded)``; ``items`` are ordered by source tensor name
    (reproducible across runs) and ``excluded`` records the vision tensors left
    out of the text-only deployment.
    """
    items: list[ConvItem] = []
    excluded: list[dict] = []
    for name in sorted(weight_map):
        meta = headers[weight_map[name]][name]
        dtype = meta["dtype"]
        shape = tuple(meta["shape"])
        route = route_tensor(name, dtype, default_expert_bits, w4_layers)
        if route.target == "SCALE":
            continue
        if route.target == "EXCLUDE":
            excluded.append({"tensor": name, "family": classify(name), "dtype": dtype, "shape": list(shape)})
            continue
        if route.target in ("W2", "W4"):
            items.extend(_plan_packed(name, dtype, shape, route, target_bytes))
        else:
            items.extend(_plan_fp16(name, dtype, shape, route, target_bytes))
    return items, excluded


def _scale_name_for(name: str) -> str:
    stem = name[: -len(".weight")] if name.endswith(".weight") else name
    return f"{stem}{_SCALE_SUFFIX}"


def _plan_packed(name: str, dtype: str, shape: tuple[int, ...], route: Route, target_bytes: int) -> list[ConvItem]:
    bits = W4_BITS if route.target == "W4" else W2_BITS
    out_features, in_features = shape  # GLM FP8 experts are stored [out, in] (not packed)
    stem = name[: -len(".weight")] if name.endswith(".weight") else name
    total_bytes = _codes_bytes(out_features, in_features, bits) + _scale_bytes(out_features, in_features)
    if total_bytes <= target_bytes:
        ranges = [(0, out_features)]
    else:
        ranges = _part_ranges(out_features, _packed_part_rows(in_features, bits, target_bytes))
    n_parts = len(ranges)
    items: list[ConvItem] = []
    for k, (r0, r1) in enumerate(ranges):
        rows = r1 - r0
        codes = OutputTensor(_suffix(f"{stem}_codes", k, n_parts), "codes", "U8", _codes_bytes(rows, in_features, bits))
        scale = OutputTensor(_suffix(f"{stem}_scale", k, n_parts), "scale", "F32", _scale_bytes(rows, in_features))
        items.append(
            ConvItem(
                stem=stem,
                weight_name=name,
                scale_name=_scale_name_for(name),
                family=classify(name),
                target=route.target,
                source_format=route.source_format,
                source_dtype=dtype,
                source_shape=shape,
                out_features=out_features,
                in_features=in_features,
                row0=r0,
                row1=r1,
                part_index=k,
                n_parts=n_parts,
                outputs=[codes, scale],
            )
        )
    return items


def _plan_fp16(name: str, dtype: str, shape: tuple[int, ...], route: Route, target_bytes: int) -> list[ConvItem]:
    numel = int(np.prod(shape)) if shape else 1
    total_bytes = numel * 2
    scale_name = _scale_name_for(name) if route.source_format == FMT_FP8_F32BLOCK else None
    # Only 2-D *cast* tensors partition row-wise (e.g. the big BF16 embed /
    # lm_head). FP8-sourced FP16 tensors carry a block-scale grid whose rows must
    # not be split off-block; they are small and stay whole.
    if len(shape) == 2 and route.source_format == FMT_CAST and total_bytes > target_bytes:
        ranges = _part_ranges(shape[0], _dense_part_rows(shape[1], target_bytes))
    else:
        ranges = [(0, shape[0] if shape else 1)]
    n_parts = len(ranges)
    out_features = shape[0] if shape else 1
    in_features = shape[1] if len(shape) == 2 else (shape[0] if shape else 1)
    items: list[ConvItem] = []
    for k, (r0, r1) in enumerate(ranges):
        rows = r1 - r0
        part_bytes = (rows * shape[1] * 2) if len(shape) == 2 else total_bytes
        out_name = _suffix(name, k, n_parts)
        items.append(
            ConvItem(
                stem=name,
                weight_name=name,
                scale_name=scale_name,
                family=classify(name),
                target="FP16",
                source_format=route.source_format,
                source_dtype=dtype,
                source_shape=shape,
                out_features=out_features,
                in_features=in_features,
                row0=r0,
                row1=r1,
                part_index=k,
                n_parts=n_parts,
                outputs=[OutputTensor(out_name, "dense", "F16", part_bytes)],
            )
        )
    return items


# --- per-item conversion (bounded) ------------------------------------------
def _source_dequant_f32block(
    reader: SafetensorsShardReader,
    weight_name: str,
    scale_name: str,
    row_start: int,
    row_end: int,
) -> torch.Tensor:
    """Dequantise one row-chunk of an F8_E4M3 + F32-block tensor to float32.

    The chunk must start on a block-row boundary (all GLM callers step by a
    multiple of the 128-row block), so ``[row_start, row_end)`` maps to a whole
    number of scale rows with no partial-block straddle at the top.
    """
    w_rows = torch.from_numpy(reader.read_rows(weight_name, row_start, row_end).copy())
    weight_rows = reader.shape(weight_name)[0]
    scale_rows = reader.shape(scale_name)[0]
    weight_cols = reader.shape(weight_name)[1]
    scale_cols = reader.shape(scale_name)[1]
    block_rows = weight_rows // scale_rows
    block_cols = weight_cols // scale_cols
    if row_start % block_rows:
        raise ValueError(f"fp8 chunk start {row_start} must align to the {block_rows}-row scale block")
    s_start = row_start // block_rows
    s_end = -(-row_end // block_rows)  # ceil
    s_rows = torch.from_numpy(reader.read_rows(scale_name, s_start, s_end).copy())
    return dequant_fp8_e4m3_f32block(w_rows, s_rows, (block_rows, block_cols))


def _convert_packed_range(
    reader: SafetensorsShardReader,
    item: ConvItem,
    chunk_rows: int,
) -> tuple[torch.Tensor, torch.Tensor, ConversionStats]:
    """Stream a ``[row0, row1)`` row range -> packed W2 codes + fp32 block scale.

    Reuses the DeepSeek E1.1 primitives (``quantize_weight`` + ``pack_codes``) and
    pads the final partial 32-row block with zeros so a row count that is not a
    multiple of 32 still tiles the ``[32, 32]`` block-scale grid. The transient
    working set is one float tile of ``chunk_rows x in_features``.
    """
    bits = W4_BITS if item.target == "W4" else W2_BITS
    in_features = item.in_features
    part_rows = item.row1 - item.row0
    padded = math.ceil(part_rows / W2_BLOCK_ROWS) * W2_BLOCK_ROWS
    codes_per_byte = 8 // bits
    packed = torch.zeros(padded, in_features // codes_per_byte, dtype=torch.uint8)
    block_scale = torch.zeros(padded // W2_BLOCK_ROWS, in_features // W2_BLOCK_COLS, dtype=torch.float32)
    stats = ConversionStats(
        out_features=part_rows,
        in_features=in_features,
        full_float_bytes=part_rows * in_features * 4,
    )
    for c0 in range(item.row0, item.row1, chunk_rows):
        c1 = min(c0 + chunk_rows, item.row1)
        tile = _source_dequant_f32block(reader, item.weight_name, item.scale_name, c0, c1)
        stats.observe_tile(tile)
        height = c1 - c0
        padded_height = math.ceil(height / W2_BLOCK_ROWS) * W2_BLOCK_ROWS
        if padded_height != height:
            tile = torch.cat([tile, torch.zeros(padded_height - height, in_features, dtype=tile.dtype)], dim=0)
        # MSE-optimal per-block scale (default) lifts W2 weight cosine ~0.83 -> ~0.92
        # vs the fp8 source at zero memory cost (same code format/kernels). Set
        # GLM_W2_SCALE_METHOD=minmax to reproduce the original no-clip scale.
        codes, scale = quantize_weight(
            tile, bits, W2_BLOCK_ROWS, W2_BLOCK_COLS,
            method=os.environ.get("GLM_W2_SCALE_METHOD", "mse"),
        )
        dst = c0 - item.row0
        packed[dst : dst + padded_height] = pack_codes(codes, bits)
        block_scale[dst // W2_BLOCK_ROWS : dst // W2_BLOCK_ROWS + padded_height // W2_BLOCK_ROWS] = scale.to(
            torch.float32
        )
        del tile, codes, scale
    return packed, block_scale, stats


def _convert_fp16(
    reader: SafetensorsShardReader,
    item: ConvItem,
    chunk_rows: int,
) -> tuple[torch.Tensor, ConversionStats]:
    """Convert one tensor (or row range) to float16, bounded by row-chunks."""
    shape = item.source_shape
    stats = ConversionStats(
        out_features=item.row1 - item.row0,
        in_features=item.in_features,
        full_float_bytes=(item.row1 - item.row0) * item.in_features * 4,
    )
    if len(shape) != 2:
        native = _read_full_native(reader, item.weight_name)
        out = native.to(torch.float32).to(torch.float16)
        stats.observe_tile(native.to(torch.float32))
        return out, stats
    out = torch.empty(item.row1 - item.row0, shape[1], dtype=torch.float16)
    step = chunk_rows
    if item.source_format == FMT_FP8_F32BLOCK:
        block_rows = shape[0] // reader.shape(item.scale_name)[0]
        step = max(block_rows, (chunk_rows // block_rows) * block_rows)
    for c0 in range(item.row0, item.row1, step):
        c1 = min(c0 + step, item.row1)
        if item.source_format == FMT_FP8_F32BLOCK:
            tile = _source_dequant_f32block(reader, item.weight_name, item.scale_name, c0, c1)
        else:
            tile = _native_from_rows(reader.read_rows(item.weight_name, c0, c1).copy(), item.source_dtype).to(
                torch.float32
            )
        stats.observe_tile(tile)
        out[c0 - item.row0 : c1 - item.row0] = tile.to(torch.float16)
        del tile
    return out, stats


def convert_item(
    reader: SafetensorsShardReader, item: ConvItem, chunk_rows: int
) -> tuple[dict[str, torch.Tensor], ConversionStats]:
    """Convert one item to its output tensor(s) (bounded working set)."""
    if item.target in ("W2", "W4"):
        packed, scale, stats = _convert_packed_range(reader, item, chunk_rows)
        return {item.outputs[0].name: packed.contiguous(), item.outputs[1].name: scale.contiguous()}, stats
    tensor, stats = _convert_fp16(reader, item, chunk_rows)
    return {item.outputs[0].name: tensor.contiguous()}, stats


# --- provenance --------------------------------------------------------------
def _source_dequant_desc(source_format: str) -> str:
    if source_format == FMT_FP8_F32BLOCK:
        return "e4m3_value * f32_block_scale (plain per-[128,128]-block scale)"
    return "bitcast native dtype -> float16 (no dequant)"


def _provenance_entry(stem_items: list[ConvItem], stats: ConversionStats | None) -> dict:
    head = stem_items[0]
    entry: dict = {
        "source_tensor": head.weight_name,
        "source_dtype": head.source_dtype,
        "source_shape": list(head.source_shape),
        "source_format": head.source_format,
        "source_dequant": _source_dequant_desc(head.source_format),
        "family": head.family,
        "target_precision": head.target,
        "out_features": head.out_features,
        "in_features": head.in_features,
        "n_parts": head.n_parts,
        "outputs": [o.name for it in stem_items for o in it.outputs],
        "part_rows": [[it.row0, it.row1] for it in stem_items],
    }
    if head.target in ("W2", "W4"):
        _bits = W4_BITS if head.target == "W4" else W2_BITS
        entry["packing"] = {
            "n_bits": _bits,
            "codes_per_byte": 8 // _bits,
            "grid": "signed two's-complement, symmetric",
            "block_rows": W2_BLOCK_ROWS,
            "block_cols": W2_BLOCK_COLS,
            "endianness": "little-endian by field index within byte",
            "row_pad_to_block": head.out_features % W2_BLOCK_ROWS != 0,
        }
    if stats is not None:
        entry["peak_float_tile_bytes"] = stats.peak_float_tile_bytes
    return entry


# --- driver ------------------------------------------------------------------
def run(
    source_dir: str | Path,
    out_dir: str | Path,
    shard_target_bytes: int = DEFAULT_SHARD_TARGET_BYTES,
    chunk_rows: int = DEFAULT_CHUNK_ROWS,
    resume: bool = True,
    default_expert_bits: int = W2_BITS,
    w4_layers: frozenset[int] = frozenset(),
) -> RunResult:
    """Convert the full GLM-5.3-Flash model, streaming + resumable + sharded.

    Reuses the DeepSeek driver's atomic shard IO, per-shard sha256,
    ``progress.json`` resumability and ``POSIX_FADV_DONTNEED`` page-cache release.
    ``resume=False`` reconverts every shard.
    """
    source_dir = Path(source_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    weight_map = json.loads((source_dir / INDEX_FILE).read_text())["weight_map"]
    headers = _load_headers(source_dir, weight_map)
    items, excluded = plan_items(weight_map, headers, shard_target_bytes, default_expert_bits, w4_layers)
    shards = bin_shards(items, shard_target_bytes)
    n_shards = len(shards)
    shard_names = [_shard_name(i, n_shards) for i in range(n_shards)]

    out_weight_map: dict[str, str] = {}
    for shard_idx, item_indices in enumerate(shards):
        for idx in item_indices:
            for out_t in items[idx].outputs:
                out_weight_map[out_t.name] = shard_names[shard_idx]

    progress = _load_progress(out_dir) if resume else {}
    progress_shards: dict[str, dict] = progress.get("shards", {}) if resume else {}

    prov_stats: dict[str, ConversionStats] = {}
    result = RunResult(num_shards=n_shards, excluded_count=len(excluded))
    peak_rss = current_rss_bytes()

    for shard_idx, item_indices in enumerate(shards):
        shard_file = shard_names[shard_idx]
        recorded = progress_shards.get(shard_file)
        if (
            recorded
            and recorded.get("done")
            and (out_dir / shard_file).exists()
            and recorded.get("sha256") == _sha256_file(out_dir / shard_file)
        ):
            result.shard_status[shard_file] = "skipped"
            continue

        tensors: dict[str, torch.Tensor] = {}
        readers: dict[str, SafetensorsShardReader] = {}
        for idx in item_indices:
            item = items[idx]
            src_shard = weight_map[item.weight_name]
            reader = readers.get(src_shard)
            if reader is None:
                reader = readers[src_shard] = SafetensorsShardReader(source_dir / src_shard)
            outputs, stats = convert_item(reader, item, chunk_rows)
            tensors.update(outputs)
            prov_stats.setdefault(item.stem, stats)
            peak_rss = max(peak_rss, current_rss_bytes())

        sha, nbytes = _save_shard_atomic(out_dir, shard_file, tensors)
        progress_shards[shard_file] = {"done": True, "sha256": sha, "bytes": nbytes}
        result.shard_status[shard_file] = "converted"
        src_shards_used = {weight_map[items[idx].weight_name] for idx in item_indices}
        del tensors, readers
        for src_shard in src_shards_used:
            _drop_page_cache(source_dir / src_shard)
        _drop_page_cache(out_dir / shard_file)
        peak_rss = max(peak_rss, current_rss_bytes())
        _write_json_atomic(
            out_dir / PROGRESS_FILE,
            {"plan": {shard_names[i]: shards[i] for i in range(n_shards)}, "shards": progress_shards},
        )

    result.total_output_bytes = sum(s.get("bytes", 0) for s in progress_shards.values())
    result.peak_rss_bytes = peak_rss

    by_stem: dict[str, list[ConvItem]] = {}
    for item in items:
        by_stem.setdefault(item.stem, []).append(item)
    tensor_entries = [_provenance_entry(group, prov_stats.get(stem)) for stem, group in by_stem.items()]

    _write_json_atomic(
        out_dir / INDEX_FILE,
        {
            "metadata": {"total_size": result.total_output_bytes, "converter": "tools/glm_w2/convert_full.py"},
            "weight_map": dict(sorted(out_weight_map.items())),
        },
    )
    copied = _copy_side_files(source_dir, out_dir)
    manifest = {
        "schema_version": 1,
        "converter": "tools/glm_w2/convert_full.py",
        "model": "GLM-5.3-Flash (glm5_next)",
        "source_dir": str(source_dir),
        "out_dir": str(out_dir),
        "shard_target_bytes": shard_target_bytes,
        "chunk_rows": chunk_rows,
        "routing_table": {
            "routed_expert mlp.experts.{E}.*_proj.weight (F8_E4M3 + F32 block)": "W2 (2-bit codes + fp32 scale)",
            "other F8_E4M3 weights (dense MLP / shared_experts / MLA projections)": "FP16 (dequant_fp8_e4m3_f32block)",
            "BF16/F32 (linear-attn / norms / gate / hc / embed / lm_head / MTP)": "FP16 (cast)",
            "*.weight_scale_inv": "consumed with its weight (not emitted)",
            "model.visual.*": "EXCLUDE (text-only)",
        },
        "source_scale_format": "plain F32 per-[128,128]-block scale (weight_block_size=[128,128]; NOT ue8m0)",
        "num_output_shards": n_shards,
        "total_output_bytes": result.total_output_bytes,
        "peak_rss_bytes": peak_rss,
        "copied_files": copied,
        "shards": [
            {"file": shard_names[i], **progress_shards.get(shard_names[i], {"done": False})} for i in range(n_shards)
        ],
        "excluded": {
            "reason": "text-only W2-on-310P deployment; vision tower excluded",
            "count": len(excluded),
            "tensors": excluded,
        },
        "tensors": tensor_entries,
    }
    _write_json_atomic(out_dir / MANIFEST_FILE, manifest)
    return result


def _cli(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Full-model FP8 -> W2/FP16 converter for GLM-5.3-Flash (glm5_next)")
    parser.add_argument("--source-dir", default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--shard-target-bytes", type=int, default=DEFAULT_SHARD_TARGET_BYTES)
    parser.add_argument("--chunk-rows", type=int, default=DEFAULT_CHUNK_ROWS)
    parser.add_argument("--no-resume", action="store_true", help="ignore progress.json and reconvert every shard")
    parser.add_argument(
        "--expert-bits", type=int, default=W2_BITS, choices=(W2_BITS, W4_BITS),
        help="default routed-expert code width (2=W2, 4=W4). 4 makes ALL experts W4.",
    )
    parser.add_argument(
        "--w4-layers", default="",
        help="comma-separated decoder layer indices to quantize experts at W4 while the "
             "rest use --expert-bits (mixed precision, e.g. '34,35,...,44'); 'all' = every layer.",
    )
    args = parser.parse_args(argv)

    if args.w4_layers.strip().lower() == "all":
        w4_layers = frozenset(range(0, 10000))
    else:
        w4_layers = frozenset(int(x) for x in args.w4_layers.split(",") if x.strip())

    started = time.time()
    result = run(
        args.source_dir,
        args.out_dir,
        shard_target_bytes=args.shard_target_bytes,
        chunk_rows=args.chunk_rows,
        resume=not args.no_resume,
        default_expert_bits=args.expert_bits,
        w4_layers=w4_layers,
    )
    converted = sum(1 for s in result.shard_status.values() if s == "converted")
    skipped = sum(1 for s in result.shard_status.values() if s == "skipped")
    print(
        f"done: {result.num_shards} shards ({converted} converted, {skipped} skipped), "
        f"{result.total_output_bytes} output bytes, peak RSS {result.peak_rss_bytes} bytes, "
        f"{result.excluded_count} excluded, {time.time() - started:.1f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())


__all__ = [
    "DEFAULT_SOURCE_DIR",
    "DEFAULT_OUT_DIR",
    "DEFAULT_SHARD_TARGET_BYTES",
    "GLM_FP8_BLOCK_ROWS",
    "GLM_FP8_BLOCK_COLS",
    "FMT_FP8_F32BLOCK",
    "FMT_CAST",
    "INDEX_FILE",
    "MANIFEST_FILE",
    "PROGRESS_FILE",
    "SafetensorsShardReader",
    "dequant_fp8_e4m3_f32block",
    "Route",
    "route_tensor",
    "plan_items",
    "convert_item",
    "run",
    "RunResult",
    "current_rss_bytes",
    "_sha256_file",
    "_load_headers",
]
