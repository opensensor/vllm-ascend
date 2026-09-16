# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit test for the DeepSeek V4.1 W2 checkpoint manifest builder (plan E0.1).

Runs the real builder against the FP8 source checkpoint and asserts the tensor
families are enumerated with the dtypes observed in the safetensors headers.
Skips (CI-safe) when the source checkpoint is absent; on the dev box the source
is present, so this runs for real.

Run with ``--noconftest`` (the shared tests/ut/conftest.py is unrelated and
fails to import here):

    python3 -m pytest -q --noconftest \
        tests/ut/deepseek_w2/test_build_manifest.py
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_BUILDER_PATH = _REPO_ROOT / "tools" / "deepseek_w2" / "build_manifest.py"

# FP8 source checkpoint (read-only; 476 GB — never loaded, headers only).
_SOURCE_DIR = Path("/run/media/matteius/20TB-drive/models/dealignai/DeepSeek-V4.1-Flash-UNCENSORED-FP8")


def _load_builder():
    spec = importlib.util.spec_from_file_location("deepseek_w2_build_manifest", _BUILDER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pytestmark = pytest.mark.skipif(
    not (_SOURCE_DIR / "model.safetensors.index.json").exists(),
    reason=f"DeepSeek V4.1 FP8 source checkpoint not found at {_SOURCE_DIR}",
)


@pytest.fixture(scope="module")
def manifest():
    builder = _load_builder()
    # size-only shard records: never hash 476 GB in a unit test.
    return builder.build_manifest(_SOURCE_DIR, with_sha256=False)


def test_architecture_and_geometry(manifest):
    assert manifest["architectures"] == ["DeepseekV41ForCausalLM"]
    geo = manifest["geometry"]
    assert geo["num_hidden_layers"] == 40
    assert geo["hidden_size"] == 5120
    assert geo["vocab_size"] == 129280
    assert geo["n_routed_experts"] == 384
    assert geo["num_experts_per_tok"] == 6
    assert geo["moe_intermediate_size"] == 2304
    assert geo["n_shared_experts"] == 1
    assert geo["q_lora_rank"] == 1280
    assert geo["qk_rope_head_dim"] == 64
    assert geo["num_nextn_predict_layers"] == 3
    # EOS / special tokens carried through from config.
    assert manifest["eos_token_id"] == 1
    assert manifest["bos_token_id"] == 0
    assert manifest["pad_token_id"] == 2


def test_quantization_block(manifest):
    quant = manifest["quantization"]
    assert quant["quant_method"] == "fp8"
    assert quant["expert_dtype"] == "fp4"
    assert quant["weight_block_size"] == [32, 32]
    assert quant["scale_fmt"] == "ue8m0"


def test_families_enumerated(manifest):
    counts = manifest["family_counts"]
    # Acceptance families: experts, engram, indexer, MLA, shared_expert.
    required = (
        "routed_expert_weight",
        "routed_expert_scale",
        "engram",
        "engram_embed",
        "mla_indexer",
        "mla_attn",
        "shared_expert_weight",
    )
    for fam in required:
        assert fam in counts, f"missing family {fam}"
        assert counts[fam] > 0, f"family {fam} has zero tensors"
    # 40 layers x 384 experts x 3 projections routed experts (main block).
    assert manifest["family_counts_by_block"]["main"]["routed_expert_weight"] == 46080
    assert manifest["family_counts_by_block"]["main"]["routed_expert_scale"] == 46080


def test_dtype_map_matches_headers(manifest):
    dtypes = manifest["family_dtypes"]
    # Routed experts are FP4 packed into int8; per-block scales are ue8m0.
    assert "I8" in dtypes["routed_expert_weight"]
    assert "F8_E8M0" in dtypes["routed_expert_scale"]
    # Shared (dense) experts + MLA projections are FP8 e4m3 with ue8m0 scales.
    assert "F8_E4M3" in dtypes["shared_expert_weight"]
    assert "F8_E4M3" in dtypes["mla_attn"]
    # Engram embedding table is FP8 e4m3 with a ue8m0 scale table.
    assert "F8_E4M3" in dtypes["engram_embed"]
    # Indexer + norms + embeddings stay BF16 in the source.
    assert "BF16" in dtypes["mla_indexer"]
    assert "BF16" in dtypes["embed_tokens"]
    assert "BF16" in dtypes["lm_head"]


def test_engram_tables(manifest):
    tables = manifest["engram_tables"]
    # engram_layer_ids == [1, 14]
    assert sorted(tables) == ["1", "14"]
    for layer_id, spec in tables.items():
        embed = spec["embed_weight"]
        assert embed["dtype"] == "F8_E4M3"
        # (num_embeddings, engram_head_dim=256)
        assert embed["shape"][1] == 256
        assert embed["shape"][0] > 384_000_000
        # ue8m0 scale table accompanies the FP8 embedding.
        assert spec["embed_scale"]["dtype"] == "F8_E8M0"
        assert spec["wkv_weight"]["dtype"] == "F8_E4M3"


def test_target_precision_per_family(manifest):
    targets = manifest["target_precision"]
    assert targets["routed_expert_weight"].startswith("W2")
    assert targets["engram_embed"].startswith("W4") or "W4" in targets["engram_embed"]
    assert "FP16" in targets["mla_attn"]
    assert "FP16" in targets["lm_head"]
    assert manifest["g2_accuracy_threshold"]["status"] == "pending"


def test_shards_recorded(manifest):
    assert manifest["num_shards"] == 48
    assert len(manifest["shards"]) == 48
    for shard in manifest["shards"]:
        assert shard["bytes"] > 0
        assert "sha256" not in shard  # size-only in the test
    assert manifest["sha256_present"] is False
    assert manifest["source_dir"] == str(_SOURCE_DIR)


def test_vision_flagged_text_only(manifest):
    # Vision tower is recorded but flagged out of the text-only deployment.
    assert manifest["vision"]["present"] is True
    assert manifest["vision"]["included_in_deployment"] is False
    assert manifest["vision"]["tensor_count"] > 0
