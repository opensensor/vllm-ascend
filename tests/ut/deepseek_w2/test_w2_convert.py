# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""E1.1 tests: packed-W2 format + streaming FP4/ue8m0 -> W2 converter.

Validates, on both synthetic tiles and a FEW real DeepSeek V4.1 experts (never
the full 476 GB checkpoint):

* the FP4 (E2M1) + ue8m0 source dequant, against a hand-computed value and the
  vLLM fork's MXFP4 emulation;
* pack/unpack as exact inverses on the 2-bit (and 4-bit) codes;
* the packed-W2 format matching the E0.4 reference bit-for-bit;
* the W2 weight round-trip within its declared half-step tolerance;
* bounded working-set memory during streaming conversion.

Run: ``python3 -m pytest -q --noconftest tests/ut/deepseek_w2/test_w2_convert.py``
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch

from tools.deepseek_w2 import w2_format
from tools.deepseek_w2.w2_convert import (
    SafetensorsShardReader,
    convert_experts,
    convert_weight_streaming,
    dequant_mxfp4,
    ue8m0_to_scale,
)

# --- declared tolerances -----------------------------------------------------
# Pack/unpack is a bijection on the code field -> exact integer equality.
W2_PACK_EXACT = 0
# Weight round-trip: the signed int2 grid {-2,-1,0,1} with a per-block scale
# sized from both tails guarantees |w - dequant(quant(w))| <= block_scale / 2
# (half a grid step) per element. This is the honest, provable W2 bound; we
# assert it directly, plus a tiny float slack for the float64 arithmetic.
W2_ROUNDTRIP_HALF_STEP_SLACK = 1e-9
# The on-disk block scale is stored as float32, so the reconstructed weight
# uses a scale rounded to float32; the half-step bound then holds up to the
# relative precision of that float32 scale (a boundary element at |w/scale| =
# code_max + 0.5 can round the wrong way by ~1 ulp).
W2_ROUNDTRIP_REL_SLACK = 1e-6
# Streaming vs. one-shot conversion must be bit-identical (same math, chunked).
CONVERT_STREAM_EXACT = 0

_SOURCE_DIR = Path("/run/media/matteius/20TB-drive/models/dealignai/DeepSeek-V4.1-Flash-UNCENSORED-FP8")
_FORK_DIR = Path("/run/media/matteius/20TB-drive/vllm")
_SAMPLE_LAYER = 6
_SAMPLE_EXPERTS = [0, 1]


def _current_rss_bytes() -> int:
    """Resident set size of this process, in bytes (Linux /proc)."""
    with open("/proc/self/statm") as handle:
        resident_pages = int(handle.read().split()[1])
    return resident_pages * os.sysconf("SC_PAGE_SIZE")


def _have_source() -> bool:
    return (_SOURCE_DIR / "model.safetensors.index.json").exists()


# ===========================================================================
# Source-format dequant: FP4 (E2M1) + ue8m0
# ===========================================================================
def test_ue8m0_hand_values():
    # value = 2 ** (byte - 127): byte 127 -> 1, 128 -> 2, 126 -> 0.5, 137 -> 1024.
    bytes_ = torch.tensor([127, 128, 126, 137, 117], dtype=torch.uint8)
    got = ue8m0_to_scale(bytes_)
    expected = torch.tensor([1.0, 2.0, 0.5, 1024.0, 2.0**-10])
    assert torch.equal(got, expected)


def test_fp4_ue8m0_dequant_hand_value():
    # One packed byte holds two FP4 codes: low nibble = first element.
    # code 0b0011 = +1.5, code 0b1010 = -1.0  ->  byte = (0b1010 << 4) | 0b0011.
    byte = torch.tensor([[(0b1010 << 4) | 0b0011]], dtype=torch.uint8).view(torch.int8)
    # Pad the row to a full 32-code group so the single ue8m0 scale applies.
    packed = torch.zeros(1, 16, dtype=torch.int8)
    packed[0, 0] = byte
    scale = torch.full((1, 1), 128, dtype=torch.uint8)  # 2 ** (128-127) = 2
    deq = dequant_mxfp4(packed, scale)
    assert deq.shape == (1, 32)
    # First two codes scaled by 2: +1.5*2 = 3.0, -1.0*2 = -2.0; rest are 0.
    assert deq[0, 0].item() == pytest.approx(3.0)
    assert deq[0, 1].item() == pytest.approx(-2.0)
    assert torch.count_nonzero(deq[0, 2:]).item() == 0


def _load_fork_mxfp4():
    """Load the fork's MXFP4 emulation by file path (its ``tests`` package
    collides with this repo's, so a plain import can't reach it)."""
    import importlib.util

    ref_path = _FORK_DIR / "tests" / "quantization" / "reference_mxfp4.py"
    if not ref_path.exists():
        return None
    spec = importlib.util.spec_from_file_location("_fork_reference_mxfp4", ref_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.skipif(not _have_source(), reason="source checkpoint not present")
def test_dequant_matches_fork_mxfp4_on_real_expert():
    fork = _load_fork_mxfp4()
    if fork is None:
        pytest.skip("vLLM fork reference_mxfp4 not present")

    weight_map = json.load((_SOURCE_DIR / "model.safetensors.index.json").open())["weight_map"]
    wname = f"layers.{_SAMPLE_LAYER}.ffn.experts.0.w1.weight"
    sname = f"layers.{_SAMPLE_LAYER}.ffn.experts.0.w1.scale"
    reader = SafetensorsShardReader(_SOURCE_DIR / weight_map[wname])

    w = torch.from_numpy(reader.read_rows(wname, 0, 64).copy())
    s = torch.from_numpy(reader.read_rows(sname, 0, 64).copy())
    mine = dequant_mxfp4(w, s)
    fork_deq = fork.dq_mxfp4_torch(w.view(torch.uint8), s, torch.bfloat16).float()
    # FP4 values * power-of-two scale are exactly representable in bf16, so the
    # two dequants agree bit-for-bit.
    assert torch.equal(mine, fork_deq)


# ===========================================================================
# Pack / unpack: exact inverses
# ===========================================================================
@pytest.mark.parametrize("n_bits", [w2_format.W2_BITS, w2_format.W4_BITS])
def test_pack_unpack_exact(n_bits):
    torch.manual_seed(0)
    code_min = -(1 << (n_bits - 1))
    code_max = (1 << (n_bits - 1)) - 1
    codes = torch.randint(code_min, code_max + 1, (17, 64), dtype=torch.int8)
    packed = w2_format.pack_codes(codes, n_bits)
    assert packed.dtype == torch.uint8
    assert packed.shape == (17, 64 // (8 // n_bits))
    back = w2_format.unpack_codes(packed, 64, n_bits)
    assert (codes - back).abs().max().item() == W2_PACK_EXACT


def test_pack_field_order_little_endian():
    # W2: first code -> low 2 bits, ..., fourth code -> high 2 bits.
    codes = torch.tensor([[1, -1, -2, 0]], dtype=torch.int8)  # 01, 11, 10, 00
    packed = w2_format.pack_codes(codes, w2_format.W2_BITS)
    # byte = 0b00_10_11_01 = 0x2D
    assert int(packed[0, 0].item()) == 0b00101101


# ===========================================================================
# Packed-W2 format matches the E0.4 reference bit-for-bit
# ===========================================================================
def test_format_matches_e04_reference():
    ref = pytest.importorskip("tests.ut.deepseek_w2.reference.w2_moe_reference")
    torch.manual_seed(1)
    w = torch.randn(64, 96, dtype=torch.float64) * 0.05

    ref_scale = ref.compute_w2_block_scales(w)
    my_scale = w2_format.compute_block_scales(w, w2_format.W2_BITS)
    assert torch.equal(ref_scale, my_scale)

    ref_codes, _ = ref.quantize_weight_w2(w)
    my_codes, _ = w2_format.quantize_weight_w2(w)
    assert torch.equal(ref_codes, my_codes)

    ref_packed = ref.pack_w2_codes(ref_codes)
    my_packed = w2_format.pack_w2_codes(my_codes)
    assert torch.equal(ref_packed, my_packed)

    ref_unpacked = ref.unpack_w2_codes(ref_packed, 96)
    my_unpacked = w2_format.unpack_w2_codes(my_packed, 96)
    assert torch.equal(ref_unpacked, my_unpacked)

    ref_deq = ref.unpack_w2_to_int8(ref_packed, ref_scale, 64, 96)
    my_deq = w2_format.dequantize_w2(my_packed, my_scale, 64, 96)
    assert torch.equal(ref_deq, my_deq)


# ===========================================================================
# W2 weight round-trip within the declared half-step tolerance
# ===========================================================================
def test_w2_roundtrip_half_step_bound():
    torch.manual_seed(2)
    w = torch.randn(96, 128, dtype=torch.float64) * 0.1
    codes, scale = w2_format.quantize_weight_w2(w)
    packed = w2_format.pack_w2_codes(codes)
    recon = w2_format.dequantize_w2(packed, scale, 96, 128)

    full_scale = w2_format.broadcast_block_scales(scale, 96, 128)
    err = (w - recon).abs()
    # Every element is within half a grid step of the source weight.
    bound = 0.5 * full_scale * (1 + W2_ROUNDTRIP_REL_SLACK) + W2_ROUNDTRIP_HALF_STEP_SLACK
    assert torch.all(err <= bound)
    # And pack/unpack round-trips the codes exactly.
    assert torch.equal(w2_format.unpack_w2_codes(packed, 128).to(torch.int8), codes)


# ===========================================================================
# Streaming converter on a FEW real experts
# ===========================================================================
@pytest.mark.skipif(not _have_source(), reason="source checkpoint not present")
def test_streaming_matches_oneshot_on_real_expert():
    weight_map = json.load((_SOURCE_DIR / "model.safetensors.index.json").open())["weight_map"]
    wname = f"layers.{_SAMPLE_LAYER}.ffn.experts.0.w1.weight"
    sname = f"layers.{_SAMPLE_LAYER}.ffn.experts.0.w1.scale"
    reader = SafetensorsShardReader(_SOURCE_DIR / weight_map[wname])
    out_features, packed_cols = reader.shape(wname)
    in_features = packed_cols * 2

    # One-shot: dequant the whole expert, then quantize+pack in one go.
    w_full = torch.from_numpy(reader.read_rows(wname, 0, out_features).copy())
    s_full = torch.from_numpy(reader.read_rows(sname, 0, out_features).copy())
    deq_full = dequant_mxfp4(w_full, s_full)
    codes_full, scale_full = w2_format.quantize_weight_w2(deq_full)
    packed_oneshot = w2_format.pack_w2_codes(codes_full)

    # Streaming (chunked).
    packed_stream, scale_stream, stats = convert_weight_streaming(
        reader, wname, sname, "mxfp4", w2_format.W2_BITS, chunk_rows=256
    )
    assert packed_stream.shape == (out_features, in_features // 4)
    assert scale_stream.shape == (out_features // 32, in_features // 32)
    # Chunking changes nothing numerically.
    assert (packed_stream.to(torch.int64) - packed_oneshot.to(torch.int64)).abs().max().item() == CONVERT_STREAM_EXACT
    assert torch.equal(scale_stream, scale_full.to(torch.float32))

    # Round-trip: reconstructed W2 weight is within half a step of the exact
    # source dequant.
    recon = w2_format.dequantize_w2(packed_stream, scale_stream.double(), out_features, in_features)
    full_scale = w2_format.broadcast_block_scales(scale_stream.double(), out_features, in_features)
    err = (deq_full - recon).abs()
    bound = 0.5 * full_scale * (1 + W2_ROUNDTRIP_REL_SLACK) + W2_ROUNDTRIP_HALF_STEP_SLACK
    assert torch.all(err <= bound)


@pytest.mark.skipif(not _have_source(), reason="source checkpoint not present")
def test_bounded_working_set_on_real_experts():
    # The transient float tile is chunk-bounded and far below the full weight;
    # steady-state RSS does not grow as more experts are converted (no bank is
    # ever materialised).
    weight_map = json.load((_SOURCE_DIR / "model.safetensors.index.json").open())["weight_map"]
    chunk_rows = 256

    rss_samples = []
    peak_tile = 0
    full_float = 0
    for i in range(6):
        expert = i % 2
        proj = ["w1", "w2", "w3"][i % 3]
        wname = f"layers.{_SAMPLE_LAYER}.ffn.experts.{expert}.{proj}.weight"
        sname = f"layers.{_SAMPLE_LAYER}.ffn.experts.{expert}.{proj}.scale"
        reader = SafetensorsShardReader(_SOURCE_DIR / weight_map[wname])
        packed, scale, stats = convert_weight_streaming(
            reader, wname, sname, "mxfp4", w2_format.W2_BITS, chunk_rows=chunk_rows
        )
        peak_tile = max(peak_tile, stats.peak_float_tile_bytes)
        full_float = max(full_float, stats.full_float_bytes)
        del packed, scale, reader
        rss_samples.append(_current_rss_bytes())

    # The working tile never holds the whole float weight.
    assert 0 < peak_tile <= chunk_rows * (5120) * 4
    assert peak_tile < full_float
    # Steady-state RSS (after warmup) grows by less than one full float weight
    # across repeated conversions -> no per-expert accumulation / bank build-up.
    warm = rss_samples[1:]
    assert max(warm) - min(warm) < full_float


@pytest.mark.skipif(not _have_source(), reason="source checkpoint not present")
def test_convert_experts_writes_sample_artifact(tmp_path):
    tensors, manifest = convert_experts(
        _SOURCE_DIR,
        _SAMPLE_LAYER,
        _SAMPLE_EXPERTS,
        out_dir=tmp_path,
        chunk_rows=256,
    )
    assert len(tensors) == len(_SAMPLE_EXPERTS) * 3
    assert (tmp_path / "conversion_manifest.json").exists()
    entry = manifest["tensors"][0]
    assert entry["source_dtype"] == "I8"
    assert entry["source_format"] == "mxfp4"
    assert entry["packing"]["n_bits"] == 2
    assert entry["packing"]["block_rows"] == 32
    assert entry["packing"]["codes_per_byte"] == 4
    # provenance records the exact source dequant.
    assert "2**(ue8m0_byte-127)" in entry["source_dequant"]
