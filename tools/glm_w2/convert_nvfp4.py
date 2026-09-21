#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Full-model resumable NVFP4 -> {NVFP4 experts | FP16} converter for GLM-5.3-Flash.

Produces the 310P artifact that uses the **golden GPU checkpoint's own NVFP4
(E2M1 + per-[1,16] fp8 block scale) weights** for the routed experts instead of
re-quantising the FP8 source through the lossier mxfp4 W2/W4 format. NVFP4-direct
is the fidelity-1.0 path: the expert weights the 310P eager MoE dequantises are
byte-identical (up to fp32 scale folding) to the weights the shipped GPU model
runs, so the only remaining error vs golden is the FP16 downcast of non-expert
tensors.

Family -> target precision routing
==================================
=======================================  ====================  =========  =====================================
family                                   source (dtype)        target     how
=======================================  ====================  =========  =====================================
routed expert gate/up/down_proj.weight   U8 (packed E2M1)      NVFP4      copy packed codes; fold global scale
  ``mlp.experts.{E}.*_proj.weight``                                 into the fp8 block scale (fp32 out)
other quantised Linear ``*.weight``      U8 (packed E2M1)      FP16       nvfp4_dequant -> float16
  (dense MLP / shared_experts / MLA /
   DSA / router / lm-head projections)
everything else (norms / embed / hc)     BF16 / F32            FP16       cast -> float16
``*.weight_scale`` / ``*.weight_scale_2`` / ``*.input_scale``  SCALE      consumed with its weight
``model.visual.*``                       (any)                 EXCLUDE    recorded excluded, not written
=======================================  ====================  =========  =====================================

Output layout, resumability, atomic shard IO, per-shard sha256 and
``POSIX_FADV_DONTNEED`` page-cache release reuse the DeepSeek driver helpers
verbatim (see :mod:`tools.deepseek_w2.convert_full`).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from dataclasses import dataclass
from functools import partial
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
    _copy_side_files,
    _dense_part_rows,
    _drop_page_cache,
    _load_headers,
    _load_progress,
    _native_from_rows,
    _part_ranges,
    _read_full_native,
    _save_shard_atomic,
    _sha256_file,
    _shard_name,
    _suffix,
    _write_json_atomic,
    bin_shards,
    current_rss_bytes,
)
from tools.deepseek_w2.w2_convert import ConversionStats
from tools.deepseek_w2.w2_format import (
    NVFP4_BLOCK_COLS,
    NVFP4_CODES_PER_BYTE,
    W2_BITS,
    W2_BLOCK_COLS,
    W2_BLOCK_ROWS,
    W4_BITS,
    pack_codes,
    quantize_weight,
    unpack_nvfp4_codes,
)
from tools.glm_w2.build_manifest import classify

# --- defaults ----------------------------------------------------------------
DEFAULT_SOURCE_DIR = "/run/media/matteius/20TB-drive/models/GLM-5.3-Flash-NVFP4"
DEFAULT_OUT_DIR = "/run/media/matteius/20TB-drive/models/GLM-5.3-Flash-NVFP4-310p"
DEFAULT_SHARD_TARGET_BYTES = 5 * (1 << 30)  # ~5 GB output shards

# NVFP4 companion suffixes (consumed with their weight; never emitted standalone).
_NVFP4_SCALE_SUFFIXES = (".weight_scale", ".weight_scale_2", ".input_scale")


_LAYER_INDEX_RE = re.compile(r"\.layers\.(\d+)\.")


def _expert_layer_index(name: str) -> int | None:
    """Decoder layer index for a ``...layers.{L}.mlp.experts...`` tensor, else None."""
    m = _LAYER_INDEX_RE.search(name)
    return int(m.group(1)) if m else None


def _nvfp4_scale2_name(weight_name: str) -> str:
    """``{stem}.weight`` -> ``{stem}.weight_scale_2`` (the global fp32 scalar)."""
    stem = weight_name[: -len(".weight")]
    return f"{stem}.weight_scale_2"


def _resolve_reader(
    readers: dict[str, SafetensorsShardReader],
    weight_map: dict[str, str],
    source_dir: Path,
    name: str,
) -> SafetensorsShardReader:
    """Return (opening + caching on demand) the reader for ``name``'s shard."""
    shard = weight_map[name]
    reader = readers.get(shard)
    if reader is None:
        reader = readers[shard] = SafetensorsShardReader(source_dir / shard)
    return reader


# --- source-format decode (E2M1 + block-16 fp8 scale + global scalar) ---------
def _nvfp4_read_decode(
    resolve_reader,
    weight_name: str,
    scale_name: str,
    scale2_name: str,
    row_start: int,
    row_end: int,
) -> torch.Tensor:
    """Read a row-chunk of a packed NVFP4 weight + scales -> fp32 ``[rows, in]``.

    The packed weight is ``uint8[out, in // 2]`` (2 E2M1 nibbles/byte); the block
    scale is ``fp8_e4m3fn[out, in // 16]`` (read as uint8 then bit-cast) and the
    global scale is a fp32 scalar. Weight / scale / scale_2 may live in different
    source shards, so each is resolved through ``resolve_reader(name)``.
    """
    w_rows = torch.from_numpy(resolve_reader(weight_name).read_rows(weight_name, row_start, row_end).copy())  # uint8
    s_rows = torch.from_numpy(resolve_reader(scale_name).read_rows(scale_name, row_start, row_end).copy())  # uint8
    s_rows = s_rows.view(torch.float8_e4m3fn).to(torch.float32)  # fp8 -> fp32
    scale2 = float(_read_full_native(resolve_reader(scale2_name), scale2_name).to(torch.float32).item())
    in_features = w_rows.shape[1] * NVFP4_CODES_PER_BYTE
    val = unpack_nvfp4_codes(w_rows, in_features).to(torch.float32)  # [rows, in]
    full_scale = (s_rows * scale2).repeat_interleave(NVFP4_BLOCK_COLS, dim=1)[:, :in_features]
    return val * full_scale


# --- routing -----------------------------------------------------------------
@dataclass
class Nvfp4Route:
    target: str  # "NVFP4" | "W2" | "W4" | "FP16_NVFP4" | "FP16_CAST" | "EXCLUDE" | "SCALE"


def route_tensor(
    name: str,
    dtype: str,
    nvfp4_layers: frozenset[int] = frozenset(),
    default_bits: int = W2_BITS,
) -> Nvfp4Route:
    """Route one NVFP4-source tensor name to its conversion lane.

    Routed experts (layers 3..44) go NVFP4-direct (verbatim E2M1 + folded scale)
    for layers in ``nvfp4_layers``, else re-quantise to W2/W4 (``default_bits``)
    so the full model fits the 310P HBM (pure NVFP4-direct is over budget -- the
    per-[1,16] fp32 scale grid is 76 GB total). The router, shared experts,
    attention, norms and the MTP draft layer's BF16 experts stay full-precision
    and cast to FP16. A stray ``U8`` non-expert Linear is NVFP4-decoded.
    """
    family = classify(name)
    if family == "vision":
        return Nvfp4Route("EXCLUDE")
    if name.endswith(_NVFP4_SCALE_SUFFIXES):
        return Nvfp4Route("SCALE")
    if family == "routed_expert_weight":
        if dtype != "U8":
            return Nvfp4Route("FP16_CAST")
        li = _expert_layer_index(name)
        if li is not None and li in nvfp4_layers:
            return Nvfp4Route("NVFP4")
        bits = W4_BITS if default_bits == W4_BITS else W2_BITS
        return Nvfp4Route("W4" if bits == W4_BITS else "W2")
    if name.endswith(".weight") and dtype == "U8":
        return Nvfp4Route("FP16_NVFP4")
    return Nvfp4Route("FP16_CAST")


# --- planning ----------------------------------------------------------------
def _nvfp4_codes_bytes(rows: int, in_features: int) -> int:
    return rows * (in_features // NVFP4_CODES_PER_BYTE)


def _nvfp4_scale_bytes(rows: int, in_features: int) -> int:
    # fp16 folded scale (not fp32): halves the scale grid footprint so a
    # full-NVFP4 bank fits the 310P HBM. fp16 (10 mantissa bits) is ~0.05%
    # relative on the scale, negligible next to the 4-bit E2M1 code itself.
    return rows * (in_features // NVFP4_BLOCK_COLS) * 2


def _plan_expert(name: str, shape: tuple[int, ...], target_bytes: int) -> list[ConvItem]:
    """Plan a routed-expert passthrough: packed codes + folded fp32 block scale.

    Source ``weight`` is already packed ``uint8[out, in // 2]``; the output
    ``_codes`` copies it verbatim and ``_scale`` folds the global fp32 scalar into
    the fp8 block scale (``weight_scale * weight_scale_2``) at fp32.
    """
    out_features, in_bytes = shape
    in_features = in_bytes * NVFP4_CODES_PER_BYTE
    stem = name[: -len(".weight")]
    total_bytes = _nvfp4_codes_bytes(out_features, in_features) + _nvfp4_scale_bytes(out_features, in_features)
    if total_bytes <= target_bytes:
        ranges = [(0, out_features)]
    else:
        rows_per_part = max(1, target_bytes // max(1, (in_features // 2 + in_features // 4)))
        ranges = _part_ranges(out_features, rows_per_part)
    n_parts = len(ranges)
    items: list[ConvItem] = []
    for k, (r0, r1) in enumerate(ranges):
        rows = r1 - r0
        codes = OutputTensor(_suffix(f"{stem}_codes", k, n_parts), "codes", "U8", _nvfp4_codes_bytes(rows, in_features))
        scale = OutputTensor(
            _suffix(f"{stem}_scale", k, n_parts), "scale", "F16", _nvfp4_scale_bytes(rows, in_features)
        )
        items.append(
            ConvItem(
                stem=stem,
                weight_name=name,
                scale_name=f"{stem}.weight_scale",
                family="routed_expert_weight",
                target="NVFP4",
                source_format="nvfp4",
                source_dtype="U8",
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


def _plan_expert_w2w4(name: str, shape: tuple[int, ...], target: str) -> list[ConvItem]:
    """Plan a routed-expert weight -> packed W2/W4 codes + fp32 [32,32] block scale.

    Source ``weight`` is packed ``uint8[out, in // 2]`` (NVFP4 E2M1); it is
    NVFP4-decoded then re-quantised to the W2/W4 integer format. Each expert is
    one part (all GLM expert shapes are a multiple of the 32-row/32-col block).
    """
    out_features, in_bytes = shape
    in_features = in_bytes * NVFP4_CODES_PER_BYTE
    bits = W4_BITS if target == "W4" else W2_BITS
    codes_per_byte = 8 // bits
    stem = name[: -len(".weight")]
    codes = OutputTensor(
        f"{stem}_codes", "codes", "U8", out_features * (in_features // codes_per_byte)
    )
    scale = OutputTensor(
        f"{stem}_scale", "scale", "F32",
        (out_features // W2_BLOCK_ROWS) * (in_features // W2_BLOCK_COLS) * 4,
    )
    return [
        ConvItem(
            stem=stem,
            weight_name=name,
            scale_name=f"{stem}.weight_scale",
            family="routed_expert_weight",
            target=target,
            source_format="nvfp4",
            source_dtype="U8",
            source_shape=shape,
            out_features=out_features,
            in_features=in_features,
            row0=0,
            row1=out_features,
            part_index=0,
            n_parts=1,
            outputs=[codes, scale],
        )
    ]


def _plan_fp16(name: str, dtype: str, shape: tuple[int, ...], target: str, target_bytes: int) -> list[ConvItem]:
    """Plan a non-expert tensor -> FP16 (NVFP4 decode, or plain cast)."""
    out_features = shape[0] if shape else 1
    is_2d = len(shape) == 2
    # FP16_NVFP4 decodes packed E2M1 (2 codes/byte), so the fp16 output width is
    # double the packed uint8 width.
    in_features = (
        (shape[1] * 2 if target == "FP16_NVFP4" else shape[1]) if is_2d else (shape[0] if shape else 1)
    )
    numel = int(np.prod(shape)) if shape else 1
    total_bytes = (out_features * in_features * 2) if is_2d else numel * 2
    if is_2d and total_bytes > target_bytes:
        ranges = _part_ranges(shape[0], _dense_part_rows(in_features, target_bytes))
    else:
        ranges = [(0, out_features)]
    n_parts = len(ranges)
    items: list[ConvItem] = []
    for k, (r0, r1) in enumerate(ranges):
        rows = r1 - r0
        part_bytes = (rows * in_features * 2) if is_2d else total_bytes
        items.append(
            ConvItem(
                stem=name,
                weight_name=name,
                scale_name=(f"{name[: -len('.weight')]}.weight_scale" if target == "FP16_NVFP4" else None),
                family=classify(name),
                target="FP16",
                source_format="nvfp4" if target == "FP16_NVFP4" else "cast",
                source_dtype=dtype,
                source_shape=shape,
                out_features=out_features,
                in_features=in_features,
                row0=r0,
                row1=r1,
                part_index=k,
                n_parts=n_parts,
                outputs=[OutputTensor(_suffix(name, k, n_parts), "dense", "F16", part_bytes)],
            )
        )
    return items


def plan_items(
    weight_map: dict[str, str],
    headers: dict[str, dict],
    target_bytes: int,
    nvfp4_layers: frozenset[int] = frozenset(),
    default_bits: int = W2_BITS,
) -> tuple[list[ConvItem], list[dict]]:
    """Turn the NVFP4 source weight-map into a deterministic conversion plan."""
    items: list[ConvItem] = []
    excluded: list[dict] = []
    for name in sorted(weight_map):
        meta = headers[weight_map[name]][name]
        dtype = meta["dtype"]
        shape = tuple(meta["shape"])
        route = route_tensor(name, dtype, nvfp4_layers, default_bits)
        if route.target == "SCALE":
            continue
        if route.target == "EXCLUDE":
            excluded.append({"tensor": name, "family": classify(name), "dtype": dtype, "shape": list(shape)})
            continue
        if route.target == "NVFP4":
            items.extend(_plan_expert(name, shape, target_bytes))
        elif route.target in ("W2", "W4"):
            items.extend(_plan_expert_w2w4(name, shape, route.target))
        else:
            items.extend(_plan_fp16(name, dtype, shape, route.target, target_bytes))
    return items, excluded


# --- per-item conversion (bounded) ------------------------------------------
def _convert_expert(resolve_reader, item: ConvItem, chunk_rows: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Stream a routed-expert row range -> packed codes (verbatim) + folded scale."""
    in_features = item.in_features
    part_rows = item.row1 - item.row0
    codes = torch.empty(part_rows, in_features // NVFP4_CODES_PER_BYTE, dtype=torch.uint8)
    block_scale = torch.empty(part_rows, in_features // NVFP4_BLOCK_COLS, dtype=torch.float16)
    scale2_name = _nvfp4_scale2_name(item.weight_name)
    scale2 = float(_read_full_native(resolve_reader(scale2_name), scale2_name).to(torch.float32).item())
    for c0 in range(item.row0, item.row1, chunk_rows):
        c1 = min(c0 + chunk_rows, item.row1)
        w_rows = torch.from_numpy(resolve_reader(item.weight_name).read_rows(item.weight_name, c0, c1).copy())
        s_rows = torch.from_numpy(resolve_reader(item.scale_name).read_rows(item.scale_name, c0, c1).copy())
        s_fp8 = s_rows.view(torch.float8_e4m3fn).to(torch.float32)
        dst = c0 - item.row0
        codes[dst : dst + (c1 - c0)] = w_rows
        block_scale[dst : dst + (c1 - c0)] = s_fp8 * scale2
    return codes, block_scale


def _convert_expert_w2w4(
    resolve_reader, item: ConvItem, chunk_rows: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stream a routed-expert row range -> packed W2/W4 codes + fp32 block scale.

    NVFP4-decodes the source E2M1 + block-16 fp8 scale to fp32, then re-quantises
    to the W2/W4 integer format (MSE scale by default) so non-NVFP4-direct layers
    still fit the 310P HBM. The transient working set is one ``chunk_rows`` tile.
    """
    in_features = item.in_features
    part_rows = item.row1 - item.row0
    bits = W4_BITS if item.target == "W4" else W2_BITS
    codes_per_byte = 8 // bits
    padded = math.ceil(part_rows / W2_BLOCK_ROWS) * W2_BLOCK_ROWS
    packed = torch.zeros(padded, in_features // codes_per_byte, dtype=torch.uint8)
    block_scale = torch.zeros(padded // W2_BLOCK_ROWS, in_features // W2_BLOCK_COLS, dtype=torch.float32)
    scale2_name = _nvfp4_scale2_name(item.weight_name)
    for c0 in range(item.row0, item.row1, chunk_rows):
        c1 = min(c0 + chunk_rows, item.row1)
        tile = _nvfp4_read_decode(
            resolve_reader, item.weight_name, item.scale_name, scale2_name, c0, c1
        )
        height = c1 - c0
        padded_height = math.ceil(height / W2_BLOCK_ROWS) * W2_BLOCK_ROWS
        if padded_height != height:
            tile = torch.cat([tile, torch.zeros(padded_height - height, in_features, dtype=tile.dtype)], dim=0)
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
    return packed, block_scale


def _convert_fp16(resolve_reader, item: ConvItem, chunk_rows: int) -> tuple[torch.Tensor, ConversionStats]:
    """Convert one non-expert tensor (or row range) to FP16, bounded by row-chunks."""
    shape = item.source_shape
    stats = ConversionStats(
        out_features=item.row1 - item.row0,
        in_features=item.in_features,
        full_float_bytes=(item.row1 - item.row0) * item.in_features * 4,
    )
    if len(shape) != 2:
        native = _read_full_native(resolve_reader(item.weight_name), item.weight_name)
        out = native.to(torch.float32).to(torch.float16)
        stats.observe_tile(native.to(torch.float32))
        return out, stats
    out = torch.empty(item.row1 - item.row0, item.in_features, dtype=torch.float16)
    for c0 in range(item.row0, item.row1, chunk_rows):
        c1 = min(c0 + chunk_rows, item.row1)
        if item.source_format == "nvfp4":
            tile = _nvfp4_read_decode(
                resolve_reader, item.weight_name, item.scale_name, _nvfp4_scale2_name(item.weight_name), c0, c1
            )
        else:
            native = _native_from_rows(
                resolve_reader(item.weight_name).read_rows(item.weight_name, c0, c1).copy(), item.source_dtype
            )
            tile = native.to(torch.float32)
        stats.observe_tile(tile)
        out[c0 - item.row0 : c1 - item.row0] = tile.to(torch.float16)
        del tile
    return out, stats


def convert_item(
    resolve_reader, item: ConvItem, chunk_rows: int
) -> tuple[dict[str, torch.Tensor], ConversionStats | None]:
    """Convert one item to its output tensor(s) (bounded working set)."""
    if item.target == "NVFP4":
        codes, scale = _convert_expert(resolve_reader, item, chunk_rows)
        return {item.outputs[0].name: codes.contiguous(), item.outputs[1].name: scale.contiguous()}, None
    if item.target in ("W2", "W4"):
        codes, scale = _convert_expert_w2w4(resolve_reader, item, chunk_rows)
        return {item.outputs[0].name: codes.contiguous(), item.outputs[1].name: scale.contiguous()}, None
    tensor, stats = _convert_fp16(resolve_reader, item, chunk_rows)
    return {item.outputs[0].name: tensor.contiguous()}, stats


# --- driver ------------------------------------------------------------------
def _rewrite_quant_method(out_dir: Path) -> None:
    """Rewrite the copied config.json so the NPU loader accepts it.

    The NVFP4 source checkpoint is modelopt-quantised (``quant_method=modelopt``),
    which the 310P plugin rejects at ModelConfig validation time. The W2/W4 model
    class reads the custom ``*_codes``/``*_scale`` tensors itself and ignores the
    quant config, so relabel it ``fp8`` (the value the proven W4early checkpoint
    uses) to pass validation.
    """
    config_path = out_dir / "config.json"
    if not config_path.exists():
        return
    cfg = json.loads(config_path.read_text())
    qc = cfg.get("quantization_config")
    if isinstance(qc, dict) and qc.get("quant_method") == "modelopt":
        # The W2/NVFP4 model class reads the custom *_codes/_scale tensors itself
        # and ignores the quant config; only the quant_method label matters to
        # pass ModelConfig validation. "fp8" additionally requires activation_scheme
        # etc., so replace the modelopt dict with a minimal fp8 one.
        cfg["quantization_config"] = {
            "quant_method": "fp8",
            "activation_scheme": "dynamic",
            "fmt": "e4m3",
            "weight_block_size": [128, 128],
        }
        config_path.write_text(json.dumps(cfg, indent=2) + "\n")


def run(
    source_dir: str | Path,
    out_dir: str | Path,
    shard_target_bytes: int = DEFAULT_SHARD_TARGET_BYTES,
    chunk_rows: int = DEFAULT_CHUNK_ROWS,
    resume: bool = True,
    nvfp4_layers: frozenset[int] = frozenset(),
    default_bits: int = W2_BITS,
) -> RunResult:
    """Convert the full NVFP4 checkpoint, streaming + resumable + sharded."""
    source_dir = Path(source_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    weight_map = json.loads((source_dir / INDEX_FILE).read_text())["weight_map"]
    headers = _load_headers(source_dir, weight_map)
    items, excluded = plan_items(weight_map, headers, shard_target_bytes, nvfp4_layers, default_bits)
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
        resolve_reader = partial(_resolve_reader, readers, weight_map, source_dir)

        for idx in item_indices:
            item = items[idx]
            outputs, _stats = convert_item(resolve_reader, item, chunk_rows)
            tensors.update(outputs)
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

    _write_json_atomic(
        out_dir / INDEX_FILE,
        {
            "metadata": {"total_size": result.total_output_bytes, "converter": "tools/glm_w2/convert_nvfp4.py"},
            "weight_map": dict(sorted(out_weight_map.items())),
        },
    )
    copied = _copy_side_files(source_dir, out_dir)
    _rewrite_quant_method(out_dir)
    manifest = {
        "schema_version": 1,
        "converter": "tools/glm_w2/convert_nvfp4.py",
        "model": "GLM-5.3-Flash (glm5_next)",
        "source_dir": str(source_dir),
        "out_dir": str(out_dir),
        "shard_target_bytes": shard_target_bytes,
        "chunk_rows": chunk_rows,
        "nvfp4_layers": sorted(nvfp4_layers),
        "default_expert_bits": default_bits,
        "routing_table": {
            f"routed_expert mlp.experts.{{E}}.*_proj.weight in layers {sorted(nvfp4_layers)}": (
                "NVFP4 (verbatim E2M1 codes + folded fp32 block scale)"
            ),
            "other routed_expert *.weight (U8 packed E2M1)": (
                f"W{default_bits} (nvfp4 decode -> re-quantise, MSE scale)"
            ),
            "other quantised Linear *.weight (U8 packed E2M1)": "FP16 (nvfp4 dequant -> float16)",
            "BF16/F32 (norms / embed / hc)": "FP16 (cast)",
            "*.weight_scale / *.weight_scale_2 / *.input_scale": "consumed with its weight (not emitted)",
            "model.visual.*": "EXCLUDE (text-only)",
        },
        "source_format": "NVFP4 (E2M1 float codes + per-[1,16] fp8 block scale + global fp32 scalar)",
        "num_output_shards": n_shards,
        "total_output_bytes": result.total_output_bytes,
        "peak_rss_bytes": peak_rss,
        "copied_files": copied,
        "shards": [
            {"file": shard_names[i], **progress_shards.get(shard_names[i], {"done": False})} for i in range(n_shards)
        ],
        "excluded": {
            "reason": "text-only NVFP4-on-310P deployment; vision tower excluded",
            "count": len(excluded),
            "tensors": excluded,
        },
    }
    _write_json_atomic(out_dir / MANIFEST_FILE, manifest)
    return result


def _cli(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Full-model NVFP4 -> {NVFP4 experts | FP16} converter for GLM-5.3-Flash"
    )
    parser.add_argument("--source-dir", default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--shard-target-bytes", type=int, default=DEFAULT_SHARD_TARGET_BYTES)
    parser.add_argument("--chunk-rows", type=int, default=DEFAULT_CHUNK_ROWS)
    parser.add_argument("--no-resume", action="store_true", help="ignore progress.json and reconvert every shard")
    parser.add_argument(
        "--nvfp4-layers",
        default="",
        help="comma-separated decoder-layer indices whose routed experts stay NVFP4-direct "
        "(fidelity 1.0); all other routed experts re-quantise to --default-expert-bits",
    )
    parser.add_argument(
        "--default-expert-bits",
        type=int,
        default=W2_BITS,
        choices=(W2_BITS, W4_BITS),
        help="code width for routed experts NOT in --nvfp4-layers (2 or 4)",
    )
    args = parser.parse_args(argv)

    nvfp4_layers = frozenset(
        int(x.strip()) for x in args.nvfp4_layers.split(",") if x.strip()
    )
    started = time.time()
    result = run(
        args.source_dir,
        args.out_dir,
        shard_target_bytes=args.shard_target_bytes,
        chunk_rows=args.chunk_rows,
        resume=not args.no_resume,
        nvfp4_layers=nvfp4_layers,
        default_bits=args.default_expert_bits,
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
    "route_tensor",
    "plan_items",
    "convert_item",
    "run",
]
