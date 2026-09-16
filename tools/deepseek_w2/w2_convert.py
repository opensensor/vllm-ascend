# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Streaming FP4/FP8 -> packed-W2 converter for DeepSeek V4.1 (plan E1.1).

This is the critical-path host-side converter that makes DeepSeek V4.1 552B fit
on Ascend 310P: it reads the FP8-checkpoint routed-expert weights (which are
**FP4 (E2M1) packed into int8** with a **ue8m0 (F8_E8M0)** per-32 block scale),
dequantises them *exactly* to float, requantises to the 2-bit packed-W2 format
(``w2_format.py``), and writes the packed codes + per-block scale. Engram weight
tables convert the same way to 4-bit (W4). Every read is bounded (row-chunked,
memory-mapped): a full expert bank (476 GB) is never materialised.

Source expert format (validated bit-for-bit against the vLLM fork's MXFP4
emulation, ``tests/quantization/reference_mxfp4.py::dq_mxfp4_torch``)
======================================================================

* ``layers.{L}.ffn.experts.{E}.{w1,w2,w3}.weight``: dtype ``I8``, shape
  ``[out, in // 2]``. Two FP4 codes are packed per int8 byte, **low nibble =
  first (even) element** along the input axis, **high nibble = second (odd)**.
  Each 4-bit code is E2M1: bit3 = sign, bits2..1 = exponent (bias 1), bit0 =
  mantissa, giving magnitudes ``{0, 0.5, 1, 1.5, 2, 3, 4, 6}``.
* ``...weight.scale``: dtype ``F8_E8M0``, shape ``[out, in // 32]``. Each byte
  ``b`` is an unsigned power-of-two exponent; its value is ``2 ** (b - 127)``.
  One scale per contiguous run of 32 FP4 codes along the input axis.
* Dequantised weight = ``fp4_value * 2 ** (scale_byte - 127)``.

The FP8 dense/shared/engram path (``F8_E4M3`` weight + ``F8_E8M0`` block scale)
uses the same ue8m0 exponent scale over ``[32, 32]`` (or per-row ``[1, 32]``)
blocks; the E4M3 code is decoded via ``torch.float8_e4m3fn``.

Provenance
==========
Every converted tensor records source dtype/shape, the exact source-format
dequant, the target precision, and the packing parameters into an output
manifest so the load path (E1.2) and the kernel bridge (E1.3) can verify the
contract.
"""

from __future__ import annotations

import argparse
import json
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from tools.deepseek_w2.w2_format import (
    W2_BITS,
    W2_BLOCK_COLS,
    W2_BLOCK_ROWS,
    pack_codes,
    quantize_weight,
)

# --- source format constants -------------------------------------------------
UE8M0_EXP_BIAS = 127
"""ue8m0 (F8_E8M0) exponent bias: value = 2 ** (byte - 127)."""

MXFP4_GROUP_SIZE = 32
"""FP4 codes sharing one ue8m0 scale, along the input axis."""

FP4_CODES_PER_BYTE = 2
"""Two FP4 (E2M1) codes packed per source int8 byte."""

# E2M1 value table for FP4 codes 0..15 (index = 4-bit code; bit3 = sign).
# Magnitudes {0, .5, 1, 1.5, 2, 3, 4, 6}; matches the fork MXFP4 emulation.
_FP4_E2M1_LUT = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)

# Chunk height (rows) for streaming; a multiple of the 32-row block so every
# [32, 32] target block is fully contained in one chunk.
DEFAULT_CHUNK_ROWS = 256

_ST_NP_DTYPE = {
    "I8": np.int8,
    "F8_E8M0": np.uint8,
    "F8_E4M3": np.uint8,
    "BF16": np.uint16,
    "F16": np.uint16,
    "F32": np.float32,
}


def ue8m0_to_scale(scale_bytes: torch.Tensor) -> torch.Tensor:
    """Decode ue8m0 exponent bytes to float32 power-of-two scales.

    ``value = 2 ** (byte - 127)``; exact in float32 for the whole byte range.
    """
    if scale_bytes.dtype != torch.uint8:
        raise ValueError("ue8m0 scale must be uint8 exponent bytes")
    exp = scale_bytes.to(torch.int32) - UE8M0_EXP_BIAS
    return torch.exp2(exp.to(torch.float32))


def dequant_mxfp4(
    packed_i8: torch.Tensor,
    scale_e8m0: torch.Tensor,
    group_size: int = MXFP4_GROUP_SIZE,
) -> torch.Tensor:
    """Dequantise FP4 (int8-packed) + ue8m0 block scale to exact float32.

    Args:
        packed_i8: ``[R, in // 2]`` int8 (or uint8) FP4-packed weight bytes.
        scale_e8m0: ``[R, in // group_size]`` uint8 ue8m0 exponents.
        group_size: FP4 codes per scale (32).

    Returns:
        ``[R, in]`` float32 dequantised weight (exact: FP4 values are exactly
        representable and the scale is a power of two).
    """
    packed = packed_i8.view(torch.uint8).to(torch.int64)
    rows, packed_cols = packed.shape
    in_features = packed_cols * FP4_CODES_PER_BYTE
    codes = torch.empty(rows, in_features, dtype=torch.int64)
    codes[:, 0::2] = packed & 0x0F  # low nibble -> first (even) element
    codes[:, 1::2] = (packed >> 4) & 0x0F  # high nibble -> second (odd)
    values = _FP4_E2M1_LUT.to(torch.float32)[codes]
    scale = ue8m0_to_scale(scale_e8m0)  # [R, in // group_size]
    values = values.view(rows, in_features // group_size, group_size)
    values = values * scale[..., None]
    return values.view(rows, in_features)


def dequant_fp8_e4m3(
    fp8_bytes: torch.Tensor,
    scale_e8m0: torch.Tensor,
) -> torch.Tensor:
    """Dequantise F8_E4M3 weight + ue8m0 block scale to float32.

    Block shape is inferred from the weight/scale shape ratio (``[32, 32]`` for
    dense/shared experts, ``[1, 32]`` per-row for the Engram tables).
    """
    weight = fp8_bytes.view(torch.float8_e4m3fn).to(torch.float32)
    rows, cols = weight.shape
    s_rows, s_cols = scale_e8m0.shape
    if rows % s_rows or cols % s_cols:
        raise ValueError(f"weight shape {(rows, cols)} not divisible by scale grid {(s_rows, s_cols)}")
    block_rows, block_cols = rows // s_rows, cols // s_cols
    scale = ue8m0_to_scale(scale_e8m0)
    full = scale.repeat_interleave(block_rows, dim=0).repeat_interleave(block_cols, dim=1)[:rows, :cols]
    return weight * full


# --- streaming safetensors reader -------------------------------------------
class SafetensorsShardReader:
    """Memory-mapped safetensors reader that slices tensors by row-chunk.

    Only the requested byte ranges are faulted in by the OS; the whole shard is
    never read into RSS.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        with self.path.open("rb") as handle:
            (header_len,) = struct.unpack("<Q", handle.read(8))
            self._header = json.loads(handle.read(header_len))
        self._data_start = 8 + header_len
        self._mmap = np.memmap(self.path, dtype=np.uint8, mode="r")

    def meta(self, name: str) -> dict:
        return self._header[name]

    def dtype(self, name: str) -> str:
        return self._header[name]["dtype"]

    def shape(self, name: str) -> tuple[int, ...]:
        return tuple(self._header[name]["shape"])

    def read_rows(self, name: str, row_start: int, row_end: int) -> np.ndarray:
        """Return rows ``[row_start:row_end]`` of a 2-D tensor as a numpy view."""
        meta = self._header[name]
        shape = meta["shape"]
        if len(shape) != 2:
            raise ValueError(f"{name}: only 2-D tensors support row slicing")
        rows, cols = shape
        row_end = min(row_end, rows)
        np_dtype = np.dtype(_ST_NP_DTYPE[meta["dtype"]])
        itemsize = np_dtype.itemsize
        row_bytes = cols * itemsize
        begin = self._data_start + meta["data_offsets"][0] + row_start * row_bytes
        end = begin + (row_end - row_start) * row_bytes
        raw = np.frombuffer(self._mmap[begin:end], dtype=np_dtype)
        return raw.reshape(row_end - row_start, cols)


# --- conversion --------------------------------------------------------------
@dataclass
class ConversionStats:
    """Bounded-memory accounting for one streamed conversion."""

    out_features: int = 0
    in_features: int = 0
    n_chunks: int = 0
    peak_float_tile_bytes: int = 0
    full_float_bytes: int = 0

    def observe_tile(self, tile: torch.Tensor) -> None:
        self.n_chunks += 1
        self.peak_float_tile_bytes = max(self.peak_float_tile_bytes, tile.numel() * tile.element_size())


def _source_dequant(
    reader: SafetensorsShardReader,
    weight_name: str,
    scale_name: str,
    source_format: str,
    row_start: int,
    row_end: int,
) -> torch.Tensor:
    """Dequantise one row-chunk of the source tensor to float32."""
    w_rows = torch.from_numpy(reader.read_rows(weight_name, row_start, row_end).copy())
    if source_format == "mxfp4":
        s_rows = torch.from_numpy(reader.read_rows(scale_name, row_start, row_end).copy())
        return dequant_mxfp4(w_rows, s_rows)
    if source_format == "fp8_e4m3":
        # Map the chunk's rows onto the (coarser) block-scale grid. Chunks are
        # aligned to the 32-row block, so [row_start, row_end) maps to a whole
        # number of scale rows with no partial-block straddle.
        block_rows = reader.shape(weight_name)[0] // reader.shape(scale_name)[0]
        if row_start % block_rows or (row_end % block_rows and row_end != reader.shape(weight_name)[0]):
            raise ValueError("fp8 chunk must align to the scale block rows")
        s_start = row_start // block_rows
        s_end = -(-row_end // block_rows)  # ceil
        s_rows = torch.from_numpy(reader.read_rows(scale_name, s_start, s_end).copy())
        return dequant_fp8_e4m3(w_rows, s_rows)
    raise ValueError(f"unknown source_format {source_format!r}")


def convert_weight_streaming(
    reader: SafetensorsShardReader,
    weight_name: str,
    scale_name: str,
    source_format: str = "mxfp4",
    target_bits: int = W2_BITS,
    chunk_rows: int = DEFAULT_CHUNK_ROWS,
) -> tuple[torch.Tensor, torch.Tensor, ConversionStats]:
    """Stream a source weight -> packed target codes + per-block scale.

    Reads the source in row-chunks (aligned to the 32-row block), dequantises
    each chunk to float32, requantises to ``target_bits`` packed codes, and
    assembles the packed output. The transient working set is one float tile of
    ``chunk_rows x in_features``; the full float weight is never held.

    Returns ``(packed, block_scale, stats)``:
        packed: ``[out, in // codes_per_byte]`` uint8.
        block_scale: ``[out // 32, in // 32]`` float32.
        stats: :class:`ConversionStats` (bounded-memory evidence).
    """
    if source_format == "mxfp4":
        out_features = reader.shape(weight_name)[0]
        in_features = reader.shape(weight_name)[1] * FP4_CODES_PER_BYTE
    else:
        out_features, in_features = reader.shape(weight_name)
    if chunk_rows % W2_BLOCK_ROWS:
        raise ValueError(f"chunk_rows={chunk_rows} must be a multiple of {W2_BLOCK_ROWS}")
    codes_per_byte = 8 // target_bits
    packed = torch.empty(out_features, in_features // codes_per_byte, dtype=torch.uint8)
    block_scale = torch.empty(out_features // W2_BLOCK_ROWS, in_features // W2_BLOCK_COLS, dtype=torch.float32)
    stats = ConversionStats(
        out_features=out_features,
        in_features=in_features,
        full_float_bytes=out_features * in_features * 4,
    )
    for row_start in range(0, out_features, chunk_rows):
        row_end = min(row_start + chunk_rows, out_features)
        tile = _source_dequant(reader, weight_name, scale_name, source_format, row_start, row_end)
        stats.observe_tile(tile)
        codes, scale = quantize_weight(tile, target_bits, W2_BLOCK_ROWS, W2_BLOCK_COLS)
        packed[row_start:row_end] = pack_codes(codes, target_bits)
        block_scale[row_start // W2_BLOCK_ROWS : row_end // W2_BLOCK_ROWS] = scale.to(torch.float32)
        del tile, codes, scale
    return packed, block_scale, stats


@dataclass
class Provenance:
    """Per-tensor conversion provenance recorded into the output manifest."""

    entries: list[dict] = field(default_factory=list)

    def record(
        self,
        tensor: str,
        source_dtype: str,
        source_shape: tuple[int, ...],
        source_format: str,
        target_precision: str,
        target_bits: int,
        stats: ConversionStats,
    ) -> None:
        self.entries.append(
            {
                "tensor": tensor,
                "source_dtype": source_dtype,
                "source_shape": list(source_shape),
                "source_format": source_format,
                "source_dequant": (
                    "fp4_value * 2**(ue8m0_byte-127)"
                    if source_format == "mxfp4"
                    else "e4m3_value * 2**(ue8m0_byte-127)"
                ),
                "target_precision": target_precision,
                "packing": {
                    "n_bits": target_bits,
                    "codes_per_byte": 8 // target_bits,
                    "grid": "signed two's-complement, symmetric",
                    "block_rows": W2_BLOCK_ROWS,
                    "block_cols": W2_BLOCK_COLS,
                    "endianness": "little-endian by field index within byte",
                },
                "out_features": stats.out_features,
                "in_features": stats.in_features,
                "n_chunks": stats.n_chunks,
                "peak_float_tile_bytes": stats.peak_float_tile_bytes,
                "full_float_bytes": stats.full_float_bytes,
            }
        )

    def to_manifest(self, source_dir: str) -> dict:
        return {
            "schema_version": 1,
            "converter": "tools/deepseek_w2/w2_convert.py (E1.1)",
            "source_dir": source_dir,
            "source_expert_format": {
                "weight_dtype": "I8 (FP4 E2M1, 2 codes/byte, low-nibble-first)",
                "scale_dtype": "F8_E8M0 (ue8m0, value=2**(byte-127))",
                "group_size": MXFP4_GROUP_SIZE,
            },
            "target_format": "packed-W2 (tools/deepseek_w2/w2_format.py)",
            "tensors": self.entries,
        }


def _index_map(source_dir: Path) -> dict[str, str]:
    with (source_dir / "model.safetensors.index.json").open() as handle:
        return json.load(handle)["weight_map"]


def convert_experts(
    source_dir: str | Path,
    layer: int,
    experts: list[int],
    projections: tuple[str, ...] = ("w1", "w2", "w3"),
    out_dir: str | Path | None = None,
    chunk_rows: int = DEFAULT_CHUNK_ROWS,
) -> tuple[dict[str, tuple[torch.Tensor, torch.Tensor]], dict]:
    """Convert a handful of real routed experts FP4 -> packed-W2.

    Returns ``(tensors, manifest)`` where ``tensors`` maps
    ``"layers.{L}.ffn.experts.{E}.{proj}"`` to ``(packed, block_scale)``. If
    ``out_dir`` is given, a small sample artifact + provenance manifest is
    written there (never hundreds of GB).
    """
    source_dir = Path(source_dir)
    weight_map = _index_map(source_dir)
    readers: dict[str, SafetensorsShardReader] = {}
    tensors: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    prov = Provenance()

    for expert in experts:
        for proj in projections:
            wname = f"layers.{layer}.ffn.experts.{expert}.{proj}.weight"
            sname = f"layers.{layer}.ffn.experts.{expert}.{proj}.scale"
            shard = weight_map[wname]
            reader = readers.get(shard)
            if reader is None:
                reader = readers[shard] = SafetensorsShardReader(source_dir / shard)
            packed, scale, stats = convert_weight_streaming(reader, wname, sname, "mxfp4", W2_BITS, chunk_rows)
            key = f"layers.{layer}.ffn.experts.{expert}.{proj}"
            tensors[key] = (packed, scale)
            prov.record(
                wname,
                reader.dtype(wname),
                reader.shape(wname),
                "mxfp4",
                "W2 (2-bit routed expert)",
                W2_BITS,
                stats,
            )

    manifest = prov.to_manifest(str(source_dir))
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for key, (packed, scale) in tensors.items():
            safe = key.replace(".", "_")
            torch.save(
                {"packed": packed, "block_scale": scale},
                out_dir / f"{safe}.w2.pt",
            )
        with (out_dir / "conversion_manifest.json").open("w") as handle:
            json.dump(manifest, handle, indent=2)
    return tensors, manifest


def _cli() -> None:
    parser = argparse.ArgumentParser(description="FP4 -> packed-W2 sample converter")
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--layer", type=int, default=6)
    parser.add_argument("--experts", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--out-dir", default="artifacts/deepseek-v41-w2/sample")
    parser.add_argument("--chunk-rows", type=int, default=DEFAULT_CHUNK_ROWS)
    args = parser.parse_args()
    _, manifest = convert_experts(
        args.source_dir,
        args.layer,
        args.experts,
        out_dir=args.out_dir,
        chunk_rows=args.chunk_rows,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    _cli()


__all__ = [
    "UE8M0_EXP_BIAS",
    "MXFP4_GROUP_SIZE",
    "FP4_CODES_PER_BYTE",
    "DEFAULT_CHUNK_ROWS",
    "ue8m0_to_scale",
    "dequant_mxfp4",
    "dequant_fp8_e4m3",
    "SafetensorsShardReader",
    "ConversionStats",
    "Provenance",
    "convert_weight_streaming",
    "convert_experts",
]
