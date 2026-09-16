# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit test for the GLM-5.3-Flash (glm5_next) W2 checkpoint manifest builder.

Runs the real builder against the FP8 source checkpoint and asserts the tensor
families are enumerated with the dtypes observed in the safetensors headers, and
that the GLM-specific facts hold: routed experts are F8_E4M3 + a *plain F32*
per-[128,128]-block scale (``weight_scale_inv``), there is no Engram and no
quantised indexer table, and the vision tower is flagged out of the deployment.

Skips (CI-safe) when the source checkpoint is absent; on the dev box the source
is present, so this runs for real.

Run: ``python3 -m pytest -q --noconftest tests/ut/glm_w2/test_build_manifest.py``
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_BUILDER_PATH = _REPO_ROOT / "tools" / "glm_w2" / "build_manifest.py"

# FP8 source checkpoint (read-only; 306 GB -- never loaded, headers only).
_SOURCE_DIR = Path("/run/media/matteius/20TB-drive/models/GLM-5.3-Flash-FP8")


def _load_builder():
    spec = importlib.util.spec_from_file_location("glm_w2_build_manifest", _BUILDER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pytestmark = pytest.mark.skipif(
    not (_SOURCE_DIR / "model.safetensors.index.json").exists(),
    reason=f"GLM-5.3-Flash FP8 source checkpoint not found at {_SOURCE_DIR}",
)


@pytest.fixture(scope="module")
def manifest():
    builder = _load_builder()
    # size-only shard records: never hash 306 GB in a unit test.
    return builder.build_manifest(_SOURCE_DIR, with_sha256=False)


def test_architecture_and_geometry(manifest):
    assert manifest["architectures"] == ["Glm5NextForConditionalGeneration"]
    assert manifest["model_type"] == "glm5_next"
    geo = manifest["geometry"]
    assert geo["num_hidden_layers"] == 45
    assert geo["hidden_size"] == 4096
    assert geo["vocab_size"] == 154880
    assert geo["n_routed_experts"] == 288
    assert geo["num_experts_per_tok"] == 8
    assert geo["moe_intermediate_size"] == 2048
    assert geo["n_shared_experts"] == 1
    assert geo["first_k_dense_replace"] == 3
    assert geo["num_nextn_predict_layers"] == 1


def test_quantization_block(manifest):
    quant = manifest["quantization"]
    assert quant["quant_method"] == "fp8"
    assert quant["fmt"] == "e4m3"
    # weight_block_size 128x128 -- a plain F32 block scale, NOT ue8m0.
    assert quant["weight_block_size"] == [128, 128]
    assert quant["scale_fmt"] is None
    assert "NOT ue8m0" in quant["scale_dtype"]


def test_families_enumerated(manifest):
    counts = manifest["family_counts"]
    required = (
        "routed_expert_weight",
        "routed_expert_scale",
        "shared_expert_weight",
        "dense_mlp_weight",
        "self_attn",
        "mla_indexer",
        "router_gate",
        "embed_tokens",
        "lm_head",
    )
    for fam in required:
        assert fam in counts, f"missing family {fam}"
        assert counts[fam] > 0, f"family {fam} has zero tensors"
    # 43 sparse layers (3..44 main + MTP layer 45) x 288 experts x 3 projections.
    assert counts["routed_expert_weight"] == 37152
    assert counts["routed_expert_scale"] == 37152
    # main = 42 sparse layers x 288 x 3; MTP layer 45 = 288 x 3.
    assert manifest["family_counts_by_block"]["main"]["routed_expert_weight"] == 36288
    assert manifest["family_counts_by_block"]["mtp"]["routed_expert_weight"] == 864


def test_no_engram_no_indexer_quant(manifest):
    """GLM-5.3-Flash has no Engram and no quantised indexer table -- only routed
    experts are quantised; the sparse-attention indexer stays BF16 -> FP16."""
    counts = manifest["family_counts"]
    assert not any("engram" in fam for fam in counts)
    # The indexer is present but BF16 (deploys FP16), never W2/W4.
    assert "F8_E4M3" not in manifest["family_dtypes"]["mla_indexer"]
    assert "FP16" in manifest["target_precision"]["mla_indexer"]


def test_dtype_map_matches_headers(manifest):
    dtypes = manifest["family_dtypes"]
    # Routed experts are F8_E4M3; their block scales are plain F32.
    assert dtypes["routed_expert_weight"] == {"F8_E4M3": 37152}
    assert dtypes["routed_expert_scale"] == {"F32": 37152}
    # Other quantised weights (dense MLP, shared experts, some MLA projections)
    # are also F8_E4M3 with F32 block scales.
    assert "F8_E4M3" in dtypes["shared_expert_weight"]
    assert "F8_E4M3" in dtypes["dense_mlp_weight"]
    assert "F8_E4M3" in dtypes["self_attn"]  # kv_a_proj_with_mqa / o_proj / q_a/q_b
    # Norms / embeddings / LM head / indexer stay BF16 in the source.
    assert "BF16" in dtypes["embed_tokens"]
    assert "BF16" in dtypes["lm_head"]
    assert "BF16" in dtypes["mla_indexer"]


def test_routed_expert_example_geometry(manifest):
    example = manifest["routed_expert_example"]
    assert set(example) == {"gate_proj", "up_proj", "down_proj"}
    down = example["down_proj"]
    assert down["weight"]["dtype"] == "F8_E4M3"
    assert down["weight"]["shape"] == [4096, 2048]
    # F32 block scale: [4096/128, 2048/128] = [32, 16].
    assert down["weight_scale_inv"]["dtype"] == "F32"
    assert down["weight_scale_inv"]["shape"] == [32, 16]
    gate = example["gate_proj"]
    assert gate["weight"]["shape"] == [2048, 4096]
    assert gate["weight_scale_inv"]["shape"] == [16, 32]


def test_target_precision_per_family(manifest):
    targets = manifest["target_precision"]
    assert targets["routed_expert_weight"].startswith("W2")
    assert "FP16" in targets["shared_expert_weight"]
    assert "FP16" in targets["self_attn"]
    assert "FP16" in targets["lm_head"]
    assert "consumed" in targets["routed_expert_scale"]
    # No W4 anywhere (no Engram).
    assert not any("W4" in v for v in targets.values())


def test_shards_recorded(manifest):
    assert manifest["num_shards"] == 62
    assert len(manifest["shards"]) == 62
    for shard in manifest["shards"]:
        assert shard["bytes"] > 0
        assert "sha256" not in shard  # size-only in the test
    assert manifest["sha256_present"] is False
    assert manifest["source_dir"] == str(_SOURCE_DIR)


def test_vision_flagged_text_only(manifest):
    assert manifest["vision"]["present"] is True
    assert manifest["vision"]["included_in_deployment"] is False
    assert manifest["vision"]["tensor_count"] == 347
    # Vision tensors are not routed to any output precision (EXCLUDE).
    assert manifest["target_precision"]["vision"].startswith("n/a")


def test_mtp_layer_recorded(manifest):
    mtp = manifest["mtp"]
    assert mtp["num_nextn_predict_layers"] == 1
    assert mtp["mtp_layer_index"] == 45
    assert mtp["tensor_count"] > 0
