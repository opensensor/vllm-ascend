# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests: full-model resumable FP8 -> W2/FP16 converter for GLM-5.3-Flash.

Validates ``tools/glm_w2/convert_full.py`` on a small SYNTHETIC glm5_next-shaped
source checkpoint (never the 306 GB model), plus a FEW real routed experts when
the source checkpoint is present.

The synthetic checkpoint uses the *exact* safetensors dtype tags of the real
model: ``F8_E4M3`` routed experts + ``F32`` ``weight_scale_inv`` block scales,
``F8_E4M3`` dense-MLP / shared-expert / MLA projections, ``BF16`` / ``F32``
dense/norm/gate tensors, and ``model.visual.*`` tensors that must be excluded --
so the family -> precision router is exercised end to end.

Covered:
* the F8_E4M3 + plain-F32-block dequant (hand value + against a real expert);
* routing: routed experts -> W2 (codes uint8 + scale fp32); other F8_E4M3 ->
  FP16 (dequant); BF16/F32 -> FP16 (cast); vision EXCLUDED;
* output layout: shards + index weight_map + manifest enumerate outputs with the
  right dtype/shape; per-shard sha256 recorded and correct;
* the non-32-row-aligned routed expert (zero-pad path);
* W2 round-trip within the E1.1 half-step bound; FP16 dequant/cast fidelity;
* resumability (second run skips every shard); bounded RSS.

Run: ``python3 -m pytest -q --noconftest tests/ut/glm_w2/test_convert_full.py``
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import load_file

from tools.deepseek_w2.w2_format import broadcast_block_scales, dequantize_w2
from tools.glm_w2 import convert_full
from tools.glm_w2.convert_full import dequant_fp8_e4m3_f32block

# --- tolerances --------------------------------------------------------------
# W2 reconstruction is bounded by half a grid step per element (E1.1 proof);
# add a tiny float slack for the float32 on-disk scale.
HALF_STEP_REL_SLACK = 1e-6
HALF_STEP_ABS_SLACK = 1e-9

_SOURCE_DIR = Path("/run/media/matteius/20TB-drive/models/GLM-5.3-Flash-FP8")
_LP = "model.language_model.layers"


def _have_source() -> bool:
    return (_SOURCE_DIR / "model.safetensors.index.json").exists()


# --- synthetic safetensors writer (controls the exact dtype tags) ------------
def _bytes_for(t: torch.Tensor) -> bytes:
    return t.contiguous().view(torch.uint8).numpy().tobytes()


def _write_safetensors(path, tensors: dict[str, tuple[torch.Tensor, str]]) -> None:
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


def _f8_e4m3(rows: int, cols: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(rows, cols, generator=g) * 0.3).to(torch.float8_e4m3fn)


def _f32_scale(rows: int, cols: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    # Small positive block scales (like the real weight_scale_inv ~1e-3..1e-1).
    return (torch.rand(rows, cols, generator=g) * 0.05 + 0.01).to(torch.float32)


def _bf16(shape, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(*shape, generator=g) * 0.2).to(torch.bfloat16)


def _f32(shape, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(*shape, generator=g) * 0.2).to(torch.float32)


def _build_synthetic_source(root):
    """Write a tiny multi-family glm5_next-shaped source checkpoint."""
    root.mkdir(parents=True, exist_ok=True)
    shard0: dict[str, tuple[torch.Tensor, str]] = {}
    shard1: dict[str, tuple[torch.Tensor, str]] = {}

    # --- routed experts (F8_E4M3 + F32 [32,32] block scale) -> W2 ---
    for expert in (0, 1):
        for proj, seed in (("gate_proj", 1), ("up_proj", 2), ("down_proj", 3)):
            shard0[f"{_LP}.3.mlp.experts.{expert}.{proj}.weight"] = (_f8_e4m3(64, 64, expert * 10 + seed), "F8_E4M3")
            shard0[f"{_LP}.3.mlp.experts.{expert}.{proj}.weight_scale_inv"] = (
                _f32_scale(2, 2, expert * 3 + seed),
                "F32",
            )
    # --- a routed expert whose out (48) is NOT a multiple of 32 (pad path) ---
    #     block 16x32 -> scale [3, 2].
    shard0[f"{_LP}.3.mlp.experts.2.gate_proj.weight"] = (_f8_e4m3(48, 64, 71), "F8_E4M3")
    shard0[f"{_LP}.3.mlp.experts.2.gate_proj.weight_scale_inv"] = (_f32_scale(3, 2, 72), "F32")

    # --- other F8_E4M3 weights -> FP16 (dequant) ---
    # shared expert (block 32x32).
    shard0[f"{_LP}.3.mlp.shared_experts.down_proj.weight"] = (_f8_e4m3(32, 64, 101), "F8_E4M3")
    shard0[f"{_LP}.3.mlp.shared_experts.down_proj.weight_scale_inv"] = (_f32_scale(1, 2, 102), "F32")
    # dense MLP of a first_k_dense layer (block 32x32).
    shard0[f"{_LP}.0.mlp.down_proj.weight"] = (_f8_e4m3(64, 64, 111), "F8_E4M3")
    shard0[f"{_LP}.0.mlp.down_proj.weight_scale_inv"] = (_f32_scale(2, 2, 112), "F32")
    # MLA q_a_proj (block 32x32).
    shard0[f"{_LP}.3.self_attn.q_a_proj.weight"] = (_f8_e4m3(32, 64, 121), "F8_E4M3")
    shard0[f"{_LP}.3.self_attn.q_a_proj.weight_scale_inv"] = (_f32_scale(1, 2, 122), "F32")

    # --- BF16/F32 -> FP16 (cast) ---
    shard0[f"{_LP}.3.self_attn.q_proj.weight"] = (_bf16((32, 64), 131), "BF16")
    shard0[f"{_LP}.3.mlp.gate.weight"] = (_bf16((8, 64), 141), "BF16")
    shard0[f"{_LP}.3.mlp.gate.e_score_correction_bias"] = (_f32((8,), 142), "F32")
    shard0[f"{_LP}.3.input_layernorm.weight"] = (_bf16((64,), 143), "BF16")

    shard1["model.language_model.embed_tokens.weight"] = (_bf16((96, 32), 201), "BF16")
    shard1["lm_head.weight"] = (_bf16((96, 32), 202), "BF16")
    shard1["model.language_model.norm.weight"] = (_bf16((64,), 203), "BF16")
    # MTP layer extras (BF16 -> FP16 cast).
    shard1[f"{_LP}.5.eh_proj.weight"] = (_bf16((64, 128), 211), "BF16")
    shard1[f"{_LP}.5.shared_head.norm.weight"] = (_bf16((64,), 212), "BF16")

    # --- vision -> EXCLUDED ---
    shard0["model.visual.blocks.0.attn.qkv.weight"] = (_bf16((24, 16), 231), "BF16")
    shard0["model.visual.patch_embed.proj.weight"] = (_bf16((16, 16), 232), "BF16")

    _write_safetensors(root / "model-00001-of-00002.safetensors", shard0)
    _write_safetensors(root / "model-00002-of-00002.safetensors", shard1)
    weight_map = {name: "model-00001-of-00002.safetensors" for name in shard0}
    weight_map.update({name: "model-00002-of-00002.safetensors" for name in shard1})
    (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    (root / "config.json").write_text(
        json.dumps({"model_type": "glm5_next", "architectures": ["Glm5NextForConditionalGeneration"]})
    )
    (root / "tokenizer.json").write_text('{"synthetic": true}')
    (root / "tokenizer_config.json").write_text('{"synthetic": true}')
    (root / "generation_config.json").write_text('{"synthetic": true}')
    return root


@pytest.fixture
def synthetic_source(tmp_path):
    return _build_synthetic_source(tmp_path / "src")


# ===========================================================================
# Source-format dequant: F8_E4M3 + plain F32 block scale
# ===========================================================================
def test_dequant_fp8_f32block_hand_value():
    # 0xE8 decodes to -64.0 in E4M3 (sign=1, exp=13, man=0 -> -(1.0)*2**(13-7)).
    # Two-element row, one 1x2 block, scale 0.5 -> [-64*0.5, +? ]. Use a clean
    # per-block scale so the product is exact.
    weight = torch.tensor([[0xE8, 0x3C]], dtype=torch.uint8)  # -64.0, +1.5
    assert weight.view(torch.float8_e4m3fn)[0, 0].item() == -64.0
    assert weight.view(torch.float8_e4m3fn)[0, 1].item() == 1.5
    scale = torch.tensor([[0.5]], dtype=torch.float32)
    deq = dequant_fp8_e4m3_f32block(weight, scale, block=(1, 2))
    assert deq.shape == (1, 2)
    assert deq[0, 0].item() == pytest.approx(-32.0)
    assert deq[0, 1].item() == pytest.approx(0.75)


def test_dequant_fp8_f32block_broadcasts_per_block():
    # 4x4 weight, 2x2 block scale -> each [2,2] quadrant scaled by its own value.
    weight = (torch.arange(16, dtype=torch.float32).reshape(4, 4) * 0.1).to(torch.float8_e4m3fn)
    scale = torch.tensor([[2.0, 3.0], [4.0, 5.0]], dtype=torch.float32)
    deq = dequant_fp8_e4m3_f32block(weight, scale, block=(2, 2))
    ref = weight.to(torch.float32) * scale.repeat_interleave(2, 0).repeat_interleave(2, 1)
    assert torch.equal(deq, ref)


@pytest.mark.skipif(not _have_source(), reason="source checkpoint not present")
def test_dequant_matches_manual_on_real_expert():
    """Decode one real routed-expert weight element by hand (E4M3 bit-unpack +
    its F32 block scale) and confirm the vectorised dequant agrees exactly."""
    weight_map = json.loads((_SOURCE_DIR / "model.safetensors.index.json").read_text())["weight_map"]
    wname = f"{_LP}.3.mlp.experts.0.down_proj.weight"
    sname = f"{wname}_scale_inv"
    reader = convert_full.SafetensorsShardReader(_SOURCE_DIR / weight_map[wname])
    rows, cols = reader.shape(wname)
    s_rows, s_cols = reader.shape(sname)
    block_rows, block_cols = rows // s_rows, cols // s_cols  # 128, 128
    w = torch.from_numpy(reader.read_rows(wname, 0, block_rows * 2).copy())
    s = torch.from_numpy(reader.read_rows(sname, 0, 2).copy())
    deq = dequant_fp8_e4m3_f32block(w, s, (block_rows, block_cols))

    # Hand-decode element [130, 300]: block (1, 2).
    r, c = 130, 300
    byte = int(w.view(torch.uint8)[r, c].item())
    sign = -1.0 if (byte >> 7) & 1 else 1.0
    exp = (byte >> 3) & 0xF
    man = byte & 0x7
    mag = (man / 8.0) * 2 ** (1 - 7) if exp == 0 else (1 + man / 8.0) * 2 ** (exp - 7)
    fp8_val = sign * mag
    scale_val = s[r // block_rows, c // block_cols].item()
    assert deq[r, c].item() == pytest.approx(fp8_val * scale_val, rel=1e-6, abs=1e-12)
    # And torch's own E4M3 decode matches the hand decode.
    assert w.view(torch.float8_e4m3fn)[r, c].to(torch.float32).item() == pytest.approx(fp8_val)


# ===========================================================================
# Routing
# ===========================================================================
@pytest.mark.parametrize(
    ("name", "dtype", "target", "source_format"),
    [
        (f"{_LP}.3.mlp.experts.7.gate_proj.weight", "F8_E4M3", "W2", "fp8_e4m3_f32block"),
        (f"{_LP}.3.mlp.experts.7.gate_proj.weight_scale_inv", "F32", "SCALE", ""),
        (f"{_LP}.3.mlp.shared_experts.up_proj.weight", "F8_E4M3", "FP16", "fp8_e4m3_f32block"),
        (f"{_LP}.0.mlp.down_proj.weight", "F8_E4M3", "FP16", "fp8_e4m3_f32block"),
        (f"{_LP}.3.self_attn.o_proj.weight", "F8_E4M3", "FP16", "fp8_e4m3_f32block"),
        (f"{_LP}.3.self_attn.q_proj.weight", "BF16", "FP16", "cast"),
        (f"{_LP}.3.mlp.gate.e_score_correction_bias", "F32", "FP16", "cast"),
        (f"{_LP}.3.input_layernorm.weight", "BF16", "FP16", "cast"),
        ("model.language_model.embed_tokens.weight", "BF16", "FP16", "cast"),
        ("lm_head.weight", "BF16", "FP16", "cast"),
        ("model.language_model.norm.weight", "BF16", "FP16", "cast"),
        (f"{_LP}.45.eh_proj.weight", "BF16", "FP16", "cast"),
        ("model.visual.blocks.0.attn.qkv.weight", "BF16", "EXCLUDE", ""),
        ("model.visual.patch_embed.proj.weight", "BF16", "EXCLUDE", ""),
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
    result = convert_full.run(synthetic_source, out_dir, shard_target_bytes=8192, chunk_rows=256)

    assert result.num_shards >= 2
    index = json.loads((out_dir / "model.safetensors.index.json").read_text())
    manifest = json.loads((out_dir / convert_full.MANIFEST_FILE).read_text())
    weight_map = index["weight_map"]

    # config + tokenizer copied.
    assert (out_dir / "config.json").exists()
    assert set(manifest["copied_files"]) >= {"config.json", "tokenizer.json", "tokenizer_config.json"}

    shard_tensors: dict[str, torch.Tensor] = {}
    for shard_file in set(weight_map.values()):
        shard_tensors.update(load_file(str(out_dir / shard_file)))
    assert set(shard_tensors) == set(weight_map)

    # W2 routed expert: codes uint8 (4 codes/byte), scale fp32.
    codes = shard_tensors[f"{_LP}.3.mlp.experts.0.gate_proj_codes"]
    scale = shard_tensors[f"{_LP}.3.mlp.experts.0.gate_proj_scale"]
    assert codes.dtype == torch.uint8 and codes.shape == (64, 64 // 4)
    assert scale.dtype == torch.float32 and scale.shape == (64 // 32, 64 // 32)

    # FP16 dequant of an F8_E4M3 weight (shared expert / dense MLP / MLA proj).
    assert shard_tensors[f"{_LP}.3.mlp.shared_experts.down_proj.weight"].dtype == torch.float16
    assert shard_tensors[f"{_LP}.0.mlp.down_proj.weight"].dtype == torch.float16
    assert shard_tensors[f"{_LP}.3.self_attn.q_a_proj.weight"].dtype == torch.float16
    # FP16 casts: dense 2-D, 1-D norm, gate bias.
    assert shard_tensors["model.language_model.embed_tokens.weight"].dtype == torch.float16
    assert shard_tensors["model.language_model.embed_tokens.weight"].shape == (96, 32)
    assert shard_tensors["model.language_model.norm.weight"].dtype == torch.float16
    assert shard_tensors[f"{_LP}.3.mlp.gate.e_score_correction_bias"].dtype == torch.float16

    # No raw source scale leaked, and the *_codes/_scale suffixes are present.
    assert not any(name.endswith(".weight_scale_inv") for name in weight_map)

    # Vision excluded: not in output; recorded in the manifest.
    assert not any(".visual." in name or name.startswith("model.visual") for name in weight_map)
    excluded_names = {e["tensor"] for e in manifest["excluded"]["tensors"]}
    assert excluded_names == {
        "model.visual.blocks.0.attn.qkv.weight",
        "model.visual.patch_embed.proj.weight",
    }
    assert manifest["excluded"]["count"] == 2

    # Per-shard sha256 recorded and correct.
    for shard in manifest["shards"]:
        assert shard["done"] is True
        assert shard["sha256"] == convert_full._sha256_file(out_dir / shard["file"])
        assert shard["bytes"] == (out_dir / shard["file"]).stat().st_size


def test_non_aligned_routed_expert_padded(synthetic_source, tmp_path):
    out_dir = tmp_path / "out"
    convert_full.run(synthetic_source, out_dir, shard_target_bytes=1 << 20, chunk_rows=256)
    manifest = json.loads((out_dir / convert_full.MANIFEST_FILE).read_text())
    entry = next(e for e in manifest["tensors"] if e["source_tensor"] == f"{_LP}.3.mlp.experts.2.gate_proj.weight")
    assert entry["target_precision"] == "W2"
    assert entry["out_features"] == 48
    assert entry["packing"]["row_pad_to_block"] is True  # 48 not a multiple of 32

    index = json.loads((out_dir / "model.safetensors.index.json").read_text())
    shard_tensors: dict[str, torch.Tensor] = {}
    for shard_file in set(index["weight_map"].values()):
        shard_tensors.update(load_file(str(out_dir / shard_file)))
    codes = shard_tensors[f"{_LP}.3.mlp.experts.2.gate_proj_codes"]
    scale = shard_tensors[f"{_LP}.3.mlp.experts.2.gate_proj_scale"]
    # Padded to 64 rows (2 x 32-row blocks); 64 cols -> 16 packed bytes.
    assert codes.shape == (64, 64 // 4)
    assert scale.shape == (64 // 32, 64 // 32)


def test_w2_reconstruction_within_half_step(synthetic_source, tmp_path):
    out_dir = tmp_path / "out"
    convert_full.run(synthetic_source, out_dir, shard_target_bytes=1 << 20, chunk_rows=256)
    index = json.loads((out_dir / "model.safetensors.index.json").read_text())
    shard_tensors: dict[str, torch.Tensor] = {}
    for shard_file in set(index["weight_map"].values()):
        shard_tensors.update(load_file(str(out_dir / shard_file)))

    src = convert_full.SafetensorsShardReader(synthetic_source / "model-00001-of-00002.safetensors")
    wname = f"{_LP}.3.mlp.experts.0.gate_proj.weight"
    w = torch.from_numpy(src.read_rows(wname, 0, 64).copy())
    s = torch.from_numpy(src.read_rows(f"{wname}_scale_inv", 0, 2).copy())
    exact = dequant_fp8_e4m3_f32block(w, s, (32, 32))

    codes = shard_tensors[f"{_LP}.3.mlp.experts.0.gate_proj_codes"]
    blk = shard_tensors[f"{_LP}.3.mlp.experts.0.gate_proj_scale"].double()
    recon = dequantize_w2(codes, blk, 64, 64)
    full = broadcast_block_scales(blk, 64, 64)
    bound = 0.5 * full * (1 + HALF_STEP_REL_SLACK) + HALF_STEP_ABS_SLACK
    assert torch.all((exact - recon).abs() <= bound)


def test_fp16_dequant_and_cast_fidelity(synthetic_source, tmp_path):
    out_dir = tmp_path / "out"
    convert_full.run(synthetic_source, out_dir, shard_target_bytes=1 << 20, chunk_rows=256)
    index = json.loads((out_dir / "model.safetensors.index.json").read_text())
    shard_tensors: dict[str, torch.Tensor] = {}
    for shard_file in set(index["weight_map"].values()):
        shard_tensors.update(load_file(str(out_dir / shard_file)))

    # FP16 dequant of an F8_E4M3 tensor == dequant(...).half(), exactly.
    src0 = convert_full.SafetensorsShardReader(synthetic_source / "model-00001-of-00002.safetensors")
    wname = f"{_LP}.3.mlp.shared_experts.down_proj.weight"
    w = torch.from_numpy(src0.read_rows(wname, 0, 32).copy())
    s = torch.from_numpy(src0.read_rows(f"{wname}_scale_inv", 0, 1).copy())
    expected = dequant_fp8_e4m3_f32block(w, s, (32, 32)).to(torch.float16)
    assert torch.equal(shard_tensors[wname], expected)

    # FP16 cast of a BF16 tensor == bf16 -> f16, exactly.
    src1 = convert_full.SafetensorsShardReader(synthetic_source / "model-00002-of-00002.safetensors")
    raw = torch.from_numpy(
        np.ascontiguousarray(src1.read_rows("model.language_model.embed_tokens.weight", 0, 96)).view(np.uint8).copy()
    )
    expected_embed = raw.view(torch.bfloat16).reshape(96, 32).to(torch.float16)
    assert torch.equal(shard_tensors["model.language_model.embed_tokens.weight"], expected_embed)


# ===========================================================================
# Resumability
# ===========================================================================
def test_resume_skips_completed_shards(synthetic_source, tmp_path):
    out_dir = tmp_path / "out"
    first = convert_full.run(synthetic_source, out_dir, shard_target_bytes=4096, chunk_rows=256)
    assert all(status == "converted" for status in first.shard_status.values())

    shard_files = list(first.shard_status)
    before = {f: (out_dir / f).stat().st_mtime_ns for f in shard_files}
    before_bytes = {f: (out_dir / f).read_bytes() for f in shard_files}

    second = convert_full.run(synthetic_source, out_dir, shard_target_bytes=4096, chunk_rows=256)
    assert all(status == "skipped" for status in second.shard_status.values())
    for f in shard_files:
        assert (out_dir / f).stat().st_mtime_ns == before[f]
        assert (out_dir / f).read_bytes() == before_bytes[f]


def test_resume_after_partial_interruption(synthetic_source, tmp_path):
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
    for entry in manifest["tensors"]:
        peak = entry.get("peak_float_tile_bytes", 0)
        assert peak <= chunk_rows * entry["in_features"] * 4
    assert result.peak_rss_bytes - baseline_rss < 128 * (1 << 20)


# ===========================================================================
# A FEW real routed experts (skipped if the 306 GB source is absent)
# ===========================================================================
@pytest.mark.skipif(not _have_source(), reason="source checkpoint not present")
def test_real_experts_w2_roundtrip_and_bounded_rss():
    weight_map = json.loads((_SOURCE_DIR / "model.safetensors.index.json").read_text())["weight_map"]
    chunk_rows = 256
    rss_samples = []
    for i in range(4):
        proj = ("gate_proj", "up_proj", "down_proj")[i % 3]
        wname = f"{_LP}.3.mlp.experts.{i}.{proj}.weight"
        sname = f"{wname}_scale_inv"
        reader = convert_full.SafetensorsShardReader(_SOURCE_DIR / weight_map[wname])
        out_features, in_features = reader.shape(wname)
        item = convert_full.ConvItem(
            stem=wname[: -len(".weight")],
            weight_name=wname,
            scale_name=sname,
            family="routed_expert_weight",
            target="W2",
            source_format=convert_full.FMT_FP8_F32BLOCK,
            source_dtype="F8_E4M3",
            source_shape=(out_features, in_features),
            out_features=out_features,
            in_features=in_features,
            row0=0,
            row1=out_features,
            part_index=0,
            n_parts=1,
            outputs=[
                convert_full.OutputTensor(f"{wname}_codes", "codes", "U8", 0),
                convert_full.OutputTensor(f"{wname}_scale", "scale", "F32", 0),
            ],
        )
        packed, blk, stats = convert_full._convert_packed_range(reader, item, chunk_rows)
        # Exact source dequant (bounded read: full expert is only a few MB).
        w = torch.from_numpy(reader.read_rows(wname, 0, out_features).copy())
        s_rows, s_cols = reader.shape(sname)
        s = torch.from_numpy(reader.read_rows(sname, 0, s_rows).copy())
        exact = dequant_fp8_e4m3_f32block(w, s, (out_features // s_rows, in_features // s_cols))
        recon = dequantize_w2(packed, blk.double(), out_features, in_features)
        full = broadcast_block_scales(blk.double(), out_features, in_features)
        bound = 0.5 * full * (1 + HALF_STEP_REL_SLACK) + HALF_STEP_ABS_SLACK
        assert torch.all((exact - recon).abs() <= bound)
        # Working tile is chunk-bounded, never the whole expert.
        assert 0 < stats.peak_float_tile_bytes <= chunk_rows * in_features * 4
        assert stats.peak_float_tile_bytes < stats.full_float_bytes
        del packed, blk, reader, w, s, exact, recon, full
        rss_samples.append(convert_full.current_rss_bytes())
    # No per-expert accumulation across repeated conversions.
    warm = rss_samples[1:]
    assert max(warm) - min(warm) < 256 * (1 << 20)
