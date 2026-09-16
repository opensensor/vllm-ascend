# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""E1.4 tests: full-model resumable FP8/FP4 -> W2/W4/FP16 converter.

Validates ``tools/deepseek_w2/convert_full.py`` on a small SYNTHETIC source
checkpoint (never the 476 GB model). The synthetic checkpoint is written with
the *exact* safetensors dtype tags of the real model (``I8`` FP4-packed routed
experts + ``F8_E8M0`` ue8m0 scales, ``F8_E4M3`` shared/MLA/engram weights,
``BF16``/``F32`` dense/norm/gate tensors, and a ``vision.*`` tensor that must be
excluded) so the family -> precision router is exercised end to end.

Covered:
* routing: routed experts -> W2 (codes uint8 + scale fp32); engram embed/wkv
  -> W4; shared/MLA/mtp FP8 -> FP16 (dequant); BF16/F32 dense -> FP16 (cast);
  vision EXCLUDED from output + recorded in the manifest;
* output layout: shards exist, index weight_map + manifest enumerate the output
  tensors with correct dtype/shape, per-shard sha256 recorded and correct;
* the non-32-row-aligned + partitioned Engram embed table (padding path);
* resumability: a second run SKIPS every completed shard (no rewrite -- checked
  via the run summary AND unchanged shard mtimes) and the output is identical;
* bounded RSS: the per-tensor float working tile is far below a full shard.

Run: ``python3 -m pytest -q --noconftest tests/ut/deepseek_w2/test_convert_full.py``
"""

from __future__ import annotations

import json
import struct

import numpy as np
import pytest
import torch
from safetensors.torch import load_file

from tools.deepseek_w2 import convert_full
from tools.deepseek_w2.w2_format import W2_BLOCK_ROWS, dequantize_w2, dequantize_w4

# --- tolerances --------------------------------------------------------------
# W2/W4 reconstruction is bounded by half a grid step per element (E1.1 proof);
# add a tiny float slack for the float32 on-disk scale.
HALF_STEP_REL_SLACK = 1e-6
HALF_STEP_ABS_SLACK = 1e-9
# FP16 cast of a value already representable exactly stays exact; general values
# are within one float16 ulp of the source.
FP16_RTOL = 1e-3


# --- synthetic safetensors writer (controls the exact dtype tags) ------------
def _bytes_for(t: torch.Tensor) -> bytes:
    return t.contiguous().view(torch.uint8).numpy().tobytes()


def _write_safetensors(path, tensors: dict[str, tuple[torch.Tensor, str]]) -> None:
    """Write a safetensors file with explicit dtype tags (name -> (tensor, tag))."""
    header: dict[str, dict] = {}
    blob = bytearray()
    for name, (tensor, tag) in tensors.items():
        raw = _bytes_for(tensor)
        header[name] = {"dtype": tag, "shape": list(tensor.shape), "data_offsets": [len(blob), len(blob) + len(raw)]}
        blob += raw
    hjson = json.dumps(header).encode("utf-8")
    with open(path, "wb") as handle:
        handle.write(struct.pack("<Q", len(hjson)))
        handle.write(hjson)
        handle.write(blob)


def _i8_fp4(rows: int, cols_packed: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 256, (rows, cols_packed), generator=g, dtype=torch.int32).to(torch.uint8).view(torch.int8)


def _ue8m0(rows: int, cols: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    # Exponent bytes clustered around 127 (scale ~1) so dequant values are sane.
    return (120 + torch.randint(0, 15, (rows, cols), generator=g, dtype=torch.int32)).to(torch.uint8)


def _f8_e4m3(rows: int, cols: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(rows, cols, generator=g) * 0.3).to(torch.float8_e4m3fn)


def _bf16(shape, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(*shape, generator=g) * 0.2).to(torch.bfloat16)


def _f32(shape, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(*shape, generator=g) * 0.2).to(torch.float32)


def _build_synthetic_source(root):
    """Write a tiny multi-family DeepSeek-V4.1-shaped source checkpoint."""
    root.mkdir(parents=True, exist_ok=True)
    # in_features via packed cols: routed w [64, 16] -> in=32; scale [64, 1].
    shard0: dict[str, tuple[torch.Tensor, str]] = {}
    shard1: dict[str, tuple[torch.Tensor, str]] = {}

    # --- routed experts (I8 FP4 + F8_E8M0) -> W2 ---
    for expert in (0, 1):
        for proj in ("w1", "w2", "w3"):
            shard0[f"layers.0.ffn.experts.{expert}.{proj}.weight"] = (
                _i8_fp4(64, 16, expert * 10 + {"w1": 1, "w2": 2, "w3": 3}[proj]),
                "I8",
            )
            shard0[f"layers.0.ffn.experts.{expert}.{proj}.scale"] = (_ue8m0(64, 1, expert * 3 + 1), "F8_E8M0")

    # --- shared expert (F8_E4M3 + F8_E8M0, block 32x32) -> FP16 ---
    shard0["layers.0.ffn.shared_experts.w1.weight"] = (_f8_e4m3(64, 32, 101), "F8_E4M3")
    shard0["layers.0.ffn.shared_experts.w1.scale"] = (_ue8m0(2, 1, 102), "F8_E8M0")

    # --- MLA attn: one FP8 proj and one BF16/F32 tensor -> FP16 ---
    shard0["layers.0.attn.wq_b.weight"] = (_f8_e4m3(32, 32, 111), "F8_E4M3")
    shard0["layers.0.attn.wq_b.scale"] = (_ue8m0(1, 1, 112), "F8_E8M0")
    shard0["layers.0.attn.q_norm.weight"] = (_bf16((32,), 113), "BF16")

    # --- router gate + norms (BF16/F32 1-D & 2-D) -> FP16 cast ---
    shard0["layers.0.ffn.gate.weight"] = (_bf16((8, 32), 121), "BF16")
    shard0["layers.0.ffn.gate.bias"] = (_f32((8,), 122), "F32")
    shard0["layers.0.attn_norm.weight"] = (_bf16((32,), 123), "BF16")

    # --- embed / lm_head / final norm -> FP16 cast ---
    shard1["embed.weight"] = (_bf16((96, 32), 201), "BF16")
    shard1["head.weight"] = (_bf16((96, 32), 202), "BF16")
    shard1["norm.weight"] = (_bf16((32,), 203), "BF16")

    # --- engram: wkv (F8_E4M3, 32-aligned) -> W4; embed (non-aligned) -> W4;
    #     q_weight/k_weight (BF16) -> FP16 ---
    shard1["layers.1.engram.wkv.weight"] = (_f8_e4m3(64, 64, 211), "F8_E4M3")
    shard1["layers.1.engram.wkv.scale"] = (_ue8m0(2, 2, 212), "F8_E8M0")
    shard1["layers.1.engram.embed.weight"] = (_f8_e4m3(40, 64, 213), "F8_E4M3")  # 40 rows: not a multiple of 32
    shard1["layers.1.engram.embed.scale"] = (_ue8m0(40, 2, 214), "F8_E8M0")  # per-row (block_rows=1)
    shard1["layers.1.engram.q_weight"] = (_bf16((4, 32), 215), "BF16")
    shard1["layers.1.engram.k_weight"] = (_bf16((4, 32), 216), "BF16")

    # --- MTP block (FP8 main_proj -> FP16; BF16 markov head -> FP16) ---
    shard1["mtp.0.main_proj.weight"] = (_f8_e4m3(32, 32, 221), "F8_E4M3")
    shard1["mtp.0.main_proj.scale"] = (_ue8m0(1, 1, 222), "F8_E8M0")
    shard1["mtp.2.markov_head.embed.weight"] = (_bf16((16, 32), 223), "BF16")

    # --- vision / aligner / image token -> EXCLUDED ---
    shard0["vision.patch_embed.proj.weight"] = (_bf16((8, 16), 231), "BF16")
    shard0["aligner.w1.weight"] = (_bf16((8, 16), 232), "BF16")
    shard0["image_start"] = (_bf16((32,), 233), "BF16")

    _write_safetensors(root / "model-00001-of-00002.safetensors", shard0)
    _write_safetensors(root / "model-00002-of-00002.safetensors", shard1)
    weight_map = {name: "model-00001-of-00002.safetensors" for name in shard0}
    weight_map.update({name: "model-00002-of-00002.safetensors" for name in shard1})
    (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    (root / "config.json").write_text(
        json.dumps({"model_type": "deepseek_v41", "architectures": ["DeepseekV41ForCausalLM"]})
    )
    (root / "tokenizer.json").write_text('{"synthetic": true}')
    (root / "tokenizer_config.json").write_text('{"synthetic": true}')
    return root


@pytest.fixture
def synthetic_source(tmp_path):
    return _build_synthetic_source(tmp_path / "src")


# ===========================================================================
# Routing (unit, no IO)
# ===========================================================================
@pytest.mark.parametrize(
    ("name", "dtype", "target", "source_format"),
    [
        ("layers.0.ffn.experts.3.w2.weight", "I8", "W2", "mxfp4"),
        ("layers.0.ffn.experts.3.w2.scale", "F8_E8M0", "SCALE", ""),
        ("layers.1.engram.embed.weight", "F8_E4M3", "W4", "fp8_e4m3"),
        ("layers.1.engram.wkv.weight", "F8_E4M3", "W4", "fp8_e4m3"),
        ("layers.1.engram.q_weight", "BF16", "FP16", "cast"),
        ("layers.0.ffn.shared_experts.w1.weight", "F8_E4M3", "FP16", "fp8_e4m3"),
        ("layers.0.attn.wq_b.weight", "F8_E4M3", "FP16", "fp8_e4m3"),
        ("layers.0.attn.q_norm.weight", "BF16", "FP16", "cast"),
        ("mtp.0.main_proj.weight", "F8_E4M3", "FP16", "fp8_e4m3"),
        ("embed.weight", "BF16", "FP16", "cast"),
        ("head.weight", "BF16", "FP16", "cast"),
        ("norm.weight", "BF16", "FP16", "cast"),
        ("layers.0.ffn.gate.bias", "F32", "FP16", "cast"),
        ("vision.patch_embed.proj.weight", "BF16", "EXCLUDE", ""),
        ("aligner.w1.weight", "BF16", "EXCLUDE", ""),
        ("image_start", "BF16", "EXCLUDE", ""),
    ],
)
def test_route_tensor(name, dtype, target, source_format):
    route = convert_full.route_tensor(name, dtype)
    assert route.target == target
    assert route.source_format == source_format


# ===========================================================================
# End-to-end conversion of the synthetic subset
# ===========================================================================
def test_full_conversion_outputs_and_index(synthetic_source, tmp_path):
    out_dir = tmp_path / "out"
    # Small shard target so the subset spans several shards and the engram embed
    # table (its W4 codes) is row-partitioned.
    result = convert_full.run(synthetic_source, out_dir, shard_target_bytes=8192, chunk_rows=256)

    assert result.num_shards >= 2
    index = json.loads((out_dir / "model.safetensors.index.json").read_text())
    manifest = json.loads((out_dir / convert_full.MANIFEST_FILE).read_text())
    weight_map = index["weight_map"]

    # config + tokenizer copied.
    assert (out_dir / "config.json").exists()
    assert (out_dir / "tokenizer.json").exists()
    assert set(manifest["copied_files"]) >= {"config.json", "tokenizer.json", "tokenizer_config.json"}

    # Load every output shard and check dtype/shape against the plan.
    shard_tensors: dict[str, torch.Tensor] = {}
    for shard_file in {v for v in weight_map.values()}:
        shard_tensors.update(load_file(str(out_dir / shard_file)))
    assert set(shard_tensors) == set(weight_map)

    # W2 routed expert: codes uint8, scale fp32, right shapes.
    codes = shard_tensors["layers.0.ffn.experts.0.w1_codes"]
    scale = shard_tensors["layers.0.ffn.experts.0.w1_scale"]
    assert codes.dtype == torch.uint8 and codes.shape == (64, 32 // 4)  # in=32, 4 codes/byte
    assert scale.dtype == torch.float32 and scale.shape == (64 // 32, 32 // 32)

    # W4 engram wkv: codes uint8 (2 codes/byte), scale fp32.
    wkv_codes = shard_tensors["layers.1.engram.wkv_codes"]
    wkv_scale = shard_tensors["layers.1.engram.wkv_scale"]
    assert wkv_codes.dtype == torch.uint8 and wkv_codes.shape == (64, 64 // 2)
    assert wkv_scale.dtype == torch.float32 and wkv_scale.shape == (64 // 32, 64 // 32)

    # FP16 casts: dense 2-D, 1-D norm, and gate bias.
    assert shard_tensors["embed.weight"].dtype == torch.float16
    assert shard_tensors["embed.weight"].shape == (96, 32)
    assert shard_tensors["norm.weight"].dtype == torch.float16
    assert shard_tensors["norm.weight"].shape == (32,)
    assert shard_tensors["layers.0.ffn.gate.bias"].dtype == torch.float16
    # FP16 dequant of an FP8 tensor.
    assert shard_tensors["layers.0.ffn.shared_experts.w1.weight"].dtype == torch.float16
    assert shard_tensors["layers.0.attn.wq_b.weight"].dtype == torch.float16

    # Vision excluded: not in output; recorded in manifest.
    assert not any(name.startswith(("vision.", "aligner.", "image_")) for name in weight_map)
    excluded_names = {e["tensor"] for e in manifest["excluded"]["tensors"]}
    assert excluded_names == {"vision.patch_embed.proj.weight", "aligner.w1.weight", "image_start"}
    assert manifest["excluded"]["count"] == 3

    # No raw source ``*.scale`` leaked as an output tensor.
    assert not any(name.endswith(".scale") and not name.endswith("_scale") for name in weight_map)

    # Per-shard sha256 recorded and correct.
    for shard in manifest["shards"]:
        assert shard["done"] is True
        on_disk = convert_full._sha256_file(out_dir / shard["file"])
        assert shard["sha256"] == on_disk
        assert shard["bytes"] == (out_dir / shard["file"]).stat().st_size


def test_engram_embed_partitioned_and_padded(synthetic_source, tmp_path):
    out_dir = tmp_path / "out"
    # Tiny target forces the 40-row engram embed W4 codes into >1 part.
    convert_full.run(synthetic_source, out_dir, shard_target_bytes=1024, chunk_rows=256)
    manifest = json.loads((out_dir / convert_full.MANIFEST_FILE).read_text())
    entry = next(e for e in manifest["tensors"] if e["source_tensor"] == "layers.1.engram.embed.weight")
    assert entry["target_precision"] == "W4"
    assert entry["n_parts"] >= 2  # partitioned
    assert entry["packing"]["row_pad_to_block"] is True  # 40 not a multiple of 32
    # Parts cover all 40 rows contiguously.
    part_rows = entry["part_rows"]
    assert part_rows[0][0] == 0 and part_rows[-1][1] == 40
    for (a0, a1), (b0, _b1) in zip(part_rows, part_rows[1:]):
        assert a1 == b0


def test_w2_w4_reconstruction_within_half_step(synthetic_source, tmp_path):
    """The written W2/W4 codes+scale reconstruct the exact source dequant to
    within the E1.1 half-step bound (routing + packing are numerically sound)."""
    from tools.deepseek_w2.w2_convert import dequant_fp8_e4m3, dequant_mxfp4

    out_dir = tmp_path / "out"
    convert_full.run(synthetic_source, out_dir, shard_target_bytes=1 << 20, chunk_rows=256)
    index = json.loads((out_dir / "model.safetensors.index.json").read_text())
    shard_tensors: dict[str, torch.Tensor] = {}
    for shard_file in set(index["weight_map"].values()):
        shard_tensors.update(load_file(str(out_dir / shard_file)))

    # W2 routed expert vs. exact FP4 dequant.
    src = convert_full.SafetensorsShardReader(synthetic_source / "model-00001-of-00002.safetensors")
    w = torch.from_numpy(src.read_rows("layers.0.ffn.experts.0.w1.weight", 0, 64).copy())
    s = torch.from_numpy(src.read_rows("layers.0.ffn.experts.0.w1.scale", 0, 64).copy())
    exact = dequant_mxfp4(w, s)
    recon = dequantize_w2(
        shard_tensors["layers.0.ffn.experts.0.w1_codes"],
        shard_tensors["layers.0.ffn.experts.0.w1_scale"].double(),
        64,
        32,
    )
    from tools.deepseek_w2.w2_format import broadcast_block_scales

    full = broadcast_block_scales(shard_tensors["layers.0.ffn.experts.0.w1_scale"].double(), 64, 32)
    bound = 0.5 * full * (1 + HALF_STEP_REL_SLACK) + HALF_STEP_ABS_SLACK
    assert torch.all((exact - recon).abs() <= bound)

    # W4 engram wkv vs. exact FP8 dequant.
    src1 = convert_full.SafetensorsShardReader(synthetic_source / "model-00002-of-00002.safetensors")
    ww = torch.from_numpy(src1.read_rows("layers.1.engram.wkv.weight", 0, 64).copy()).view(torch.float8_e4m3fn)
    ws = torch.from_numpy(src1.read_rows("layers.1.engram.wkv.scale", 0, 2).copy())
    exact_wkv = dequant_fp8_e4m3(ww, ws)
    recon_wkv = dequantize_w4(
        shard_tensors["layers.1.engram.wkv_codes"],
        shard_tensors["layers.1.engram.wkv_scale"].double(),
        64,
        64,
    )
    full_wkv = broadcast_block_scales(shard_tensors["layers.1.engram.wkv_scale"].double(), 64, 64)
    bound_wkv = 0.5 * full_wkv * (1 + HALF_STEP_REL_SLACK) + HALF_STEP_ABS_SLACK
    assert torch.all((exact_wkv - recon_wkv).abs() <= bound_wkv)


def test_fp16_cast_matches_source(synthetic_source, tmp_path):
    out_dir = tmp_path / "out"
    convert_full.run(synthetic_source, out_dir, shard_target_bytes=1 << 20, chunk_rows=256)
    index = json.loads((out_dir / "model.safetensors.index.json").read_text())
    shard_tensors: dict[str, torch.Tensor] = {}
    for shard_file in set(index["weight_map"].values()):
        shard_tensors.update(load_file(str(out_dir / shard_file)))

    src = convert_full.SafetensorsShardReader(synthetic_source / "model-00002-of-00002.safetensors")
    raw = torch.from_numpy(np.ascontiguousarray(src.read_rows("embed.weight", 0, 96)).view(np.uint8).copy())
    expected = raw.view(torch.bfloat16).reshape(96, 32).to(torch.float16)
    assert torch.equal(shard_tensors["embed.weight"], expected)


# ===========================================================================
# Resumability
# ===========================================================================
def test_resume_skips_completed_shards(synthetic_source, tmp_path):
    out_dir = tmp_path / "out"
    first = convert_full.run(synthetic_source, out_dir, shard_target_bytes=4096, chunk_rows=256)
    assert all(status == "converted" for status in first.shard_status.values())

    # Snapshot shard bytes + mtimes.
    shard_files = list(first.shard_status)
    before = {f: (out_dir / f).stat().st_mtime_ns for f in shard_files}
    before_bytes = {f: (out_dir / f).read_bytes() for f in shard_files}

    second = convert_full.run(synthetic_source, out_dir, shard_target_bytes=4096, chunk_rows=256)
    # Every shard skipped -- no reconversion.
    assert all(status == "skipped" for status in second.shard_status.values())
    for f in shard_files:
        assert (out_dir / f).stat().st_mtime_ns == before[f]  # untouched on disk
        assert (out_dir / f).read_bytes() == before_bytes[f]  # bit-identical


def test_resume_after_partial_interruption(synthetic_source, tmp_path):
    """Simulate a Ctrl-C after some shards: delete a shard + its progress entry,
    re-run, and confirm only the missing shard is reconverted, others skipped."""
    out_dir = tmp_path / "out"
    convert_full.run(synthetic_source, out_dir, shard_target_bytes=4096, chunk_rows=256)

    progress = json.loads((out_dir / convert_full.PROGRESS_FILE).read_text())
    victim = sorted(progress["shards"])[0]
    (out_dir / victim).unlink()
    del progress["shards"][victim]
    (out_dir / convert_full.PROGRESS_FILE).write_text(json.dumps(progress))

    result = convert_full.run(synthetic_source, out_dir, shard_target_bytes=4096, chunk_rows=256)
    assert result.shard_status[victim] == "converted"
    assert sum(1 for s in result.shard_status.values() if s == "skipped") == result.num_shards - 1
    # A stale/partial .tmp is never mistaken for a finished shard.
    assert not list(out_dir.glob("*.tmp"))


# ===========================================================================
# Bounded RSS
# ===========================================================================
def test_bounded_working_set(synthetic_source, tmp_path):
    out_dir = tmp_path / "out"
    chunk_rows = 256
    baseline_rss = convert_full.current_rss_bytes()
    result = convert_full.run(synthetic_source, out_dir, shard_target_bytes=8192, chunk_rows=chunk_rows)
    manifest = json.loads((out_dir / convert_full.MANIFEST_FILE).read_text())
    # Every per-tensor float working tile is bounded by the chunk geometry
    # (chunk_rows x in_features x 4 bytes) -- never the full bank. At real scale
    # this is a few MB against 5 GB shards; here it is a few KB.
    for entry in manifest["tensors"]:
        peak = entry.get("peak_float_tile_bytes", 0)
        assert peak <= chunk_rows * entry["in_features"] * 4
    # RSS growth over the pre-run baseline stays tiny for this synthetic subset
    # (the torch/interpreter floor dominates absolute RSS): no bank accumulates.
    assert result.peak_rss_bytes - baseline_rss < 128 * (1 << 20)


def test_convert_item_tile_bounded_by_chunk(synthetic_source, tmp_path):
    """A single large-ish item's peak float tile is bounded by chunk_rows, not
    the whole tensor."""
    weight_map = json.loads((synthetic_source / "model.safetensors.index.json").read_text())["weight_map"]
    headers = convert_full._load_headers(synthetic_source, weight_map)
    items, _ = convert_full.plan_items(weight_map, headers, convert_full.DEFAULT_SHARD_TARGET_BYTES)
    embed_item = next(it for it in items if it.weight_name == "embed.weight")
    reader = convert_full.SafetensorsShardReader(synthetic_source / weight_map["embed.weight"])
    _out, stats = convert_full.convert_item(reader, embed_item, chunk_rows=W2_BLOCK_ROWS)
    # chunk_rows=32 over a 96-row tensor -> tile <= 32 rows, strictly below full.
    assert 0 < stats.peak_float_tile_bytes <= 32 * 32 * 4
    assert stats.peak_float_tile_bytes < stats.full_float_bytes
