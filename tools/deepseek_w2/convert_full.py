#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Full-model resumable FP8/FP4 -> W2/W4/FP16 converter for DeepSeek V4.1 (E1.4).

This is the tool that produces the actual 552B-at-2-bit W2 artifact for Ascend
310P. It walks the source ``model.safetensors.index.json``, routes **every
deployed tensor** to its target precision by family, streams the conversion in
bounded row-chunks (a full expert bank -- 476 GB -- is never materialised), and
writes sharded safetensors output with a weight-map index, a per-tensor
provenance manifest, and per-output-shard sha256. Conversion is **resumable**:
each output shard is written atomically (temp file + rename) and recorded with
its sha256 in ``progress.json``; a re-run skips already-finished shards.

It is a thin orchestration layer over the *already-validated* per-tensor
primitives in :mod:`tools.deepseek_w2.w2_convert` and
:mod:`tools.deepseek_w2.w2_format` (E1.1) -- those are imported, never
reimplemented. The family classifier is reused from
:mod:`tools.deepseek_w2.build_manifest` (E0.1).

Family -> target precision routing
==================================
Driven by the E0.1 family classifier + the authoritative header dtype:

======================  =========================  ==================  ==============================
family                  source (dtype)             target              how
======================  =========================  ==================  ==============================
routed_expert_weight    FP4 int8-packed (I8)+ue8m0  W2 (2-bit)          convert_weight_streaming(bits=2)
engram_embed / engram   F8_E4M3 + ue8m0             W4 (4-bit)          convert_weight_streaming(bits=4)
  (embed.weight,          "                          "                   / padded range wrapper
   wkv.weight)
shared_expert / mla_*   F8_E4M3 + ue8m0             FP16                dequant_fp8_e4m3 -> float16
  mla_indexer/compressor
  mtp_main_proj
dense / lm_head /       BF16 / F16 / F32            FP16                cast -> float16
  embed_tokens / norms /
  router/hc gates /
  engram q_weight,k_weight
  mtp heads
vision / aligner /      (any)                       EXCLUDE             recorded excluded, not written
  image tokens
======================  =========================  ==================  ==============================

The ``*.scale`` companion of a quantised weight is consumed *with* its weight
(fed to the dequant) and is never emitted as a standalone output tensor; W2/W4
outputs carry their own freshly-computed ``*_scale`` (fp32, ``[out/32, in/32]``).

Output layout (``--out-dir``)
=============================
* ``model-{i:05d}-of-{N:05d}.safetensors`` -- target ~5 GB shards
  (``safetensors.torch.save_file``). W2/W4 emit ``{stem}_codes`` (uint8) +
  ``{stem}_scale`` (fp32); FP16 emit the original name (float16). An oversized
  tensor (e.g. the 384M-row Engram embed table) is row-partitioned into
  ``{name}.p{k}`` parts, each <= the shard target, so RSS stays bounded and
  every part fits ``save_file``.
* ``model.safetensors.index.json`` -- output ``weight_map`` (tensor -> shard).
* ``conversion_manifest.json`` -- per-tensor provenance (source dtype/shape,
  source_dequant, target precision, packing params, partition map), per-output-
  shard sha256 + byte size, excluded (vision) tensors, total output bytes, and
  peak RSS.
* ``progress.json`` -- the deterministic shard plan + per-shard done/sha256 for
  resumability.
* ``config.json`` + tokenizer files copied verbatim.

Resumability
============
The shard plan is computed deterministically from the source index *before* any
conversion (output byte sizes are known from source shapes + target precision),
so a re-run reproduces the identical plan. Each shard is written to
``<name>.tmp`` then ``os.replace``-d into place (atomic: a partial shard is
never mistaken for complete) and its sha256 recorded in ``progress.json``. On
re-run a shard whose recorded sha256 matches the on-disk file is SKIPPED. It is
safe to Ctrl-C and restart.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import struct
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

from tools.deepseek_w2.build_manifest import classify
from tools.deepseek_w2.w2_convert import (
    DEFAULT_CHUNK_ROWS,
    FP4_CODES_PER_BYTE,
    ConversionStats,
    SafetensorsShardReader,
    _source_dequant,
    convert_weight_streaming,
)
from tools.deepseek_w2.w2_format import (
    W2_BITS,
    W2_BLOCK_COLS,
    W2_BLOCK_ROWS,
    W4_BITS,
    pack_codes,
    quantize_weight,
)

# --- defaults ----------------------------------------------------------------
DEFAULT_OUT_DIR = "/run/media/matteius/20TB-drive/models/DeepSeek-V4.1-W2-310p"
DEFAULT_SHARD_TARGET_BYTES = 5 * (1 << 30)  # ~5 GB output shards
INDEX_FILE = "model.safetensors.index.json"
MANIFEST_FILE = "conversion_manifest.json"
PROGRESS_FILE = "progress.json"
SHARD_STEM = "model"

# config + tokenizer files copied verbatim into the output checkpoint.
_COPY_CANDIDATES = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "tokenizer.model",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "chat_template.jinja",
    "chat_template.json",
)

# Vision / aligner / image-token families are excluded (text-only deployment).
_VISION_FAMILIES = frozenset({"vision", "vision_aligner", "vision_image_token"})

# safetensors dtype tag -> (numpy read dtype, native torch dtype, item bytes).
_ST_NP_DTYPE = {
    "I8": np.int8,
    "F8_E8M0": np.uint8,
    "F8_E4M3": np.uint8,
    "BF16": np.uint16,
    "F16": np.uint16,
    "F32": np.float32,
}
_ST_TORCH_DTYPE = {
    "I8": torch.int8,
    "F8_E8M0": torch.uint8,
    "F8_E4M3": torch.float8_e4m3fn,
    "BF16": torch.bfloat16,
    "F16": torch.float16,
    "F32": torch.float32,
}


# --- process memory accounting ----------------------------------------------
def current_rss_bytes() -> int:
    """Resident set size of this process, in bytes (Linux ``/proc``)."""
    try:
        with open("/proc/self/statm") as handle:
            resident_pages = int(handle.read().split()[1])
        return resident_pages * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError):
        return 0


# --- routing -----------------------------------------------------------------
@dataclass
class Route:
    """The target precision + source handling for one source weight tensor."""

    target: str  # "W2" | "W4" | "FP16" | "EXCLUDE" | "SCALE"
    source_format: str  # "mxfp4" | "fp8_e4m3" | "cast" | ""


def route_tensor(name: str, dtype: str) -> Route:
    """Route one tensor to its target precision from family + header dtype.

    ``SCALE`` tensors (the ``*.scale`` companion of a quantised weight) are
    consumed with their weight, not emitted standalone. ``EXCLUDE`` covers the
    vision tower / aligner / image tokens (text-only deployment).
    """
    family = classify(name)
    if family in _VISION_FAMILIES:
        return Route("EXCLUDE", "")
    if name.endswith(".scale"):
        return Route("SCALE", "")
    if family == "routed_expert_weight":
        return Route("W2", "mxfp4")
    if family in ("engram_embed", "engram") and dtype == "F8_E4M3":
        return Route("W4", "fp8_e4m3")
    # Everything else deploys at FP16: dequantise FP8 blocks, or cast BF16/F32.
    return Route("FP16", "fp8_e4m3" if dtype == "F8_E4M3" else "cast")


# --- planning ----------------------------------------------------------------
@dataclass
class OutputTensor:
    """One planned output tensor (a codes/scale field, or an FP16 tensor)."""

    name: str
    kind: str  # "codes" | "scale" | "dense"
    dtype: str  # output dtype tag: "U8" | "F32" | "F16"
    nbytes: int


@dataclass
class ConvItem:
    """A unit of conversion work: one source weight over a row range.

    Large tensors are split into several items (``n_parts`` > 1), each covering
    a contiguous ``[row0, row1)`` row range and producing its own output
    tensor(s), so peak RSS stays bounded and every part fits one shard.
    """

    stem: str
    weight_name: str
    scale_name: str | None
    family: str
    target: str
    source_format: str
    source_dtype: str
    source_shape: tuple[int, ...]
    out_features: int
    in_features: int
    row0: int
    row1: int
    part_index: int
    n_parts: int
    outputs: list[OutputTensor]

    @property
    def nbytes(self) -> int:
        return sum(o.nbytes for o in self.outputs)


def _codes_bytes(rows: int, in_features: int, bits: int) -> int:
    return rows * (in_features // (8 // bits))


def _scale_bytes(rows: int, in_features: int) -> int:
    return math.ceil(rows / W2_BLOCK_ROWS) * (in_features // W2_BLOCK_COLS) * 4


def _packed_part_rows(in_features: int, bits: int, target_bytes: int) -> int:
    """Largest 32-row-multiple whose codes fit ``target_bytes`` (>= 32 rows)."""
    bytes_per_row = in_features // (8 // bits)
    rows = max(1, target_bytes // max(1, bytes_per_row))
    return max(W2_BLOCK_ROWS, (rows // W2_BLOCK_ROWS) * W2_BLOCK_ROWS)


def _dense_part_rows(cols: int, target_bytes: int) -> int:
    bytes_per_row = cols * 2  # float16
    return max(1, target_bytes // max(1, bytes_per_row))


def _part_ranges(total_rows: int, rows_per_part: int) -> list[tuple[int, int]]:
    return [(r0, min(r0 + rows_per_part, total_rows)) for r0 in range(0, total_rows, rows_per_part)]


def _suffix(base: str, part_index: int, n_parts: int) -> str:
    return base if n_parts == 1 else f"{base}.p{part_index}"


def plan_items(
    weight_map: dict[str, str], headers: dict[str, dict], target_bytes: int
) -> tuple[list[ConvItem], list[dict]]:
    """Turn the source weight-map into a deterministic list of conversion items.

    Returns ``(items, excluded)`` where ``items`` are ordered by source tensor
    name (deterministic across runs) and ``excluded`` records the vision tensors
    left out of the deployment.
    """
    items: list[ConvItem] = []
    excluded: list[dict] = []
    for name in sorted(weight_map):
        meta = headers[weight_map[name]][name]
        dtype = meta["dtype"]
        shape = tuple(meta["shape"])
        route = route_tensor(name, dtype)
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


def _plan_packed(name: str, dtype: str, shape: tuple[int, ...], route: Route, target_bytes: int) -> list[ConvItem]:
    bits = W2_BITS if route.target == "W2" else W4_BITS
    out_features = shape[0]
    in_features = shape[1] * FP4_CODES_PER_BYTE if route.source_format == "mxfp4" else shape[1]
    stem = name[: -len(".weight")] if name.endswith(".weight") else name
    scale_name = f"{stem}.scale"
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
                scale_name=scale_name,
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
    scale_name = (
        f"{name[: -len('.weight')]}.scale" if (route.source_format == "fp8_e4m3" and name.endswith(".weight")) else None
    )
    # Only 2-D *cast* tensors partition row-wise (the huge BF16 embed / lm_head).
    # FP8-sourced FP16 tensors carry a block-scale grid whose rows must not be
    # split off-block; they are small in practice (<= a shared expert) and stay
    # whole.
    if len(shape) == 2 and route.source_format == "cast" and total_bytes > target_bytes:
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


def bin_shards(items: list[ConvItem], target_bytes: int) -> list[list[int]]:
    """Greedily bin item indices into shards each <= ``target_bytes``.

    Deterministic given the (deterministic) item order, so the shard plan is
    reproducible across runs -- the basis of resumability.
    """
    shards: list[list[int]] = []
    current: list[int] = []
    current_bytes = 0
    for idx, item in enumerate(items):
        if current and current_bytes + item.nbytes > target_bytes:
            shards.append(current)
            current, current_bytes = [], 0
        current.append(idx)
        current_bytes += item.nbytes
    if current:
        shards.append(current)
    return shards


# --- per-item conversion (bounded) ------------------------------------------
def _read_full_native(reader: SafetensorsShardReader, name: str) -> torch.Tensor:
    """Read a whole (small, e.g. 1-D) tensor into a native-dtype torch tensor."""
    meta = reader.meta(name)
    dtype = meta["dtype"]
    shape = tuple(meta["shape"])
    begin = reader._data_start + meta["data_offsets"][0]
    end = reader._data_start + meta["data_offsets"][1]
    raw = np.array(reader._mmap[begin:end], dtype=np.uint8)  # contiguous copy
    tensor = torch.from_numpy(raw).view(_ST_TORCH_DTYPE[dtype])
    return tensor.reshape(shape) if shape else tensor


def _native_from_rows(arr: np.ndarray, dtype: str) -> torch.Tensor:
    """Reinterpret a read_rows numpy block as its native-dtype torch tensor."""
    if dtype in ("BF16", "F16"):
        u8 = torch.from_numpy(np.ascontiguousarray(arr).view(np.uint8))
        return u8.view(_ST_TORCH_DTYPE[dtype]).reshape(arr.shape)
    tensor = torch.from_numpy(np.ascontiguousarray(arr))
    if dtype == "F8_E4M3":
        return tensor.view(torch.float8_e4m3fn)
    return tensor


def _convert_packed_range(
    reader: SafetensorsShardReader,
    item: ConvItem,
    bits: int,
    chunk_rows: int,
) -> tuple[torch.Tensor, torch.Tensor, ConversionStats]:
    """Stream a ``[row0, row1)`` row range -> packed codes + fp32 block scale.

    Reuses the E1.1 primitives (``_source_dequant`` + ``quantize_weight`` +
    ``pack_codes``) and pads the final partial 32-row block with zeros so a
    tensor whose row count is not a multiple of 32 (e.g. the Engram embed table)
    still tiles the ``[32, 32]`` block-scale grid. The transient working set is
    one float tile of ``chunk_rows x in_features``.
    """
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
        tile = _source_dequant(reader, item.weight_name, item.scale_name, item.source_format, c0, c1)
        stats.observe_tile(tile)
        height = c1 - c0
        padded_height = math.ceil(height / W2_BLOCK_ROWS) * W2_BLOCK_ROWS
        if padded_height != height:
            tile = torch.cat([tile, torch.zeros(padded_height - height, in_features, dtype=tile.dtype)], dim=0)
        codes, scale = quantize_weight(tile, bits, W2_BLOCK_ROWS, W2_BLOCK_COLS)
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
    if item.source_format == "fp8_e4m3":
        block_rows = shape[0] // reader.shape(item.scale_name)[0]
        step = max(block_rows, (chunk_rows // block_rows) * block_rows)
    for c0 in range(item.row0, item.row1, step):
        c1 = min(c0 + step, item.row1)
        if item.source_format == "fp8_e4m3":
            tile = _source_dequant(reader, item.weight_name, item.scale_name, "fp8_e4m3", c0, c1)
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
        bits = W2_BITS if item.target == "W2" else W4_BITS
        aligned = item.n_parts == 1 and item.out_features % W2_BLOCK_ROWS == 0
        if aligned:
            packed, scale, stats = convert_weight_streaming(
                reader, item.weight_name, item.scale_name, item.source_format, bits, chunk_rows
            )
        else:
            packed, scale, stats = _convert_packed_range(reader, item, bits, chunk_rows)
        codes_name = item.outputs[0].name
        scale_name = item.outputs[1].name
        return {codes_name: packed.contiguous(), scale_name: scale.contiguous()}, stats
    tensor, stats = _convert_fp16(reader, item, chunk_rows)
    return {item.outputs[0].name: tensor.contiguous()}, stats


# --- provenance --------------------------------------------------------------
def _source_dequant_desc(source_format: str) -> str:
    if source_format == "mxfp4":
        return "fp4_value * 2**(ue8m0_byte-127)"
    if source_format == "fp8_e4m3":
        return "e4m3_value * 2**(ue8m0_byte-127)"
    return "bitcast native dtype -> float16 (no dequant)"


def _provenance_entry(stem_items: list[ConvItem], stats: ConversionStats | None) -> dict:
    head = stem_items[0]
    bits = W2_BITS if head.target == "W2" else W4_BITS if head.target == "W4" else None
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
    if bits is not None:
        entry["packing"] = {
            "n_bits": bits,
            "codes_per_byte": 8 // bits,
            "grid": "signed two's-complement, symmetric",
            "block_rows": W2_BLOCK_ROWS,
            "block_cols": W2_BLOCK_COLS,
            "endianness": "little-endian by field index within byte",
            "row_pad_to_block": head.out_features % W2_BLOCK_ROWS != 0,
        }
    if stats is not None:
        entry["peak_float_tile_bytes"] = stats.peak_float_tile_bytes
    return entry


# --- atomic shard IO ---------------------------------------------------------
def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _save_shard_atomic(out_dir: Path, shard_file: str, tensors: dict[str, torch.Tensor]) -> tuple[str, int]:
    tmp = out_dir / f"{shard_file}.tmp"
    final = out_dir / shard_file
    save_file(tensors, str(tmp), metadata={"converter": "tools/deepseek_w2/convert_full.py", "format": "pt"})
    os.replace(tmp, final)
    return _sha256_file(final), final.stat().st_size


def _write_json_atomic(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    os.replace(tmp, path)


# --- source headers ----------------------------------------------------------
def _read_header(path: Path) -> dict:
    with path.open("rb") as handle:
        (header_len,) = struct.unpack("<Q", handle.read(8))
        header = json.loads(handle.read(header_len))
    header.pop("__metadata__", None)
    return header


def _load_headers(source_dir: Path, weight_map: dict[str, str]) -> dict[str, dict]:
    headers: dict[str, dict] = {}
    for shard in sorted(set(weight_map.values())):
        headers[shard] = _read_header(source_dir / shard)
    return headers


# --- driver ------------------------------------------------------------------
@dataclass
class RunResult:
    """Summary of one conversion run (per-shard status for the test/spy)."""

    shard_status: dict[str, str] = field(default_factory=dict)  # "converted" | "skipped"
    num_shards: int = 0
    total_output_bytes: int = 0
    peak_rss_bytes: int = 0
    excluded_count: int = 0


def _shard_name(index: int, total: int) -> str:
    return f"{SHARD_STEM}-{index + 1:05d}-of-{total:05d}.safetensors"


def _copy_side_files(source_dir: Path, out_dir: Path) -> list[str]:
    copied = []
    for candidate in _COPY_CANDIDATES:
        src = source_dir / candidate
        if src.exists():
            shutil.copy2(src, out_dir / candidate)
            copied.append(candidate)
    return copied


def run(
    source_dir: str | Path,
    out_dir: str | Path,
    shard_target_bytes: int = DEFAULT_SHARD_TARGET_BYTES,
    chunk_rows: int = DEFAULT_CHUNK_ROWS,
    resume: bool = True,
) -> RunResult:
    """Convert the full model, streaming + resumable + sharded.

    Set ``resume=False`` to ignore any existing ``progress.json`` and reconvert
    every shard. The default (``resume=True``) skips shards already recorded
    complete with a matching on-disk sha256.
    """
    source_dir = Path(source_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    weight_map = json.loads((source_dir / INDEX_FILE).read_text())["weight_map"]
    headers = _load_headers(source_dir, weight_map)
    items, excluded = plan_items(weight_map, headers, shard_target_bytes)
    shards = bin_shards(items, shard_target_bytes)
    n_shards = len(shards)
    shard_names = [_shard_name(i, n_shards) for i in range(n_shards)]

    # Output weight_map: every output tensor -> its shard file (from the plan).
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
        del tensors, readers
        peak_rss = max(peak_rss, current_rss_bytes())
        _write_json_atomic(
            out_dir / PROGRESS_FILE,
            {"plan": {shard_names[i]: shards[i] for i in range(n_shards)}, "shards": progress_shards},
        )

    result.total_output_bytes = sum(s.get("bytes", 0) for s in progress_shards.values())
    result.peak_rss_bytes = peak_rss

    # Group items by source stem for one provenance entry per source tensor.
    by_stem: dict[str, list[ConvItem]] = {}
    for item in items:
        by_stem.setdefault(item.stem, []).append(item)
    tensor_entries = [_provenance_entry(group, prov_stats.get(stem)) for stem, group in by_stem.items()]

    _write_json_atomic(
        out_dir / INDEX_FILE,
        {
            "metadata": {"total_size": result.total_output_bytes, "converter": "convert_full.py (E1.4)"},
            "weight_map": dict(sorted(out_weight_map.items())),
        },
    )
    copied = _copy_side_files(source_dir, out_dir)
    manifest = {
        "schema_version": 1,
        "converter": "tools/deepseek_w2/convert_full.py (E1.4)",
        "source_dir": str(source_dir),
        "out_dir": str(out_dir),
        "shard_target_bytes": shard_target_bytes,
        "chunk_rows": chunk_rows,
        "routing_table": {
            "routed_expert_weight": "W2 (FP4/ue8m0 -> 2-bit codes + fp32 scale)",
            "engram_embed/engram (F8_E4M3)": "W4 (FP8/ue8m0 -> 4-bit codes + fp32 scale)",
            "shared_expert/mla_*/mtp_main_proj (F8_E4M3)": "FP16 (dequant_fp8_e4m3)",
            "dense/lm_head/embed/norms/gates/engram q,k/mtp heads (BF16/F16/F32)": "FP16 (cast)",
            "vision/aligner/image_token": "EXCLUDE (text-only)",
        },
        "num_output_shards": n_shards,
        "total_output_bytes": result.total_output_bytes,
        "peak_rss_bytes": peak_rss,
        "copied_files": copied,
        "shards": [
            {"file": shard_names[i], **progress_shards.get(shard_names[i], {"done": False})} for i in range(n_shards)
        ],
        "excluded": {
            "reason": "text-only W2-on-310P deployment; vision tower + aligner + image tokens excluded",
            "count": len(excluded),
            "tensors": excluded,
        },
        "tensors": tensor_entries,
    }
    _write_json_atomic(out_dir / MANIFEST_FILE, manifest)
    return result


def _load_progress(out_dir: Path) -> dict:
    path = out_dir / PROGRESS_FILE
    if path.exists():
        try:
            return json.loads(path.read_text())
        except (OSError, ValueError):
            return {}
    return {}


def _cli(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Full-model FP8/FP4 -> W2/W4/FP16 converter (E1.4)")
    parser.add_argument(
        "--source-dir",
        default="/run/media/matteius/20TB-drive/models/dealignai/DeepSeek-V4.1-Flash-UNCENSORED-FP8",
    )
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--shard-target-bytes", type=int, default=DEFAULT_SHARD_TARGET_BYTES)
    parser.add_argument("--chunk-rows", type=int, default=DEFAULT_CHUNK_ROWS)
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="ignore progress.json and reconvert every shard",
    )
    args = parser.parse_args(argv)

    started = time.time()
    result = run(
        args.source_dir,
        args.out_dir,
        shard_target_bytes=args.shard_target_bytes,
        chunk_rows=args.chunk_rows,
        resume=not args.no_resume,
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
    "DEFAULT_OUT_DIR",
    "DEFAULT_SHARD_TARGET_BYTES",
    "Route",
    "route_tensor",
    "ConvItem",
    "OutputTensor",
    "plan_items",
    "bin_shards",
    "convert_item",
    "run",
    "RunResult",
    "current_rss_bytes",
]
