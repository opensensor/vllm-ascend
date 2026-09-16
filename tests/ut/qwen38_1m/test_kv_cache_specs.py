# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU unit tests for Qwen4Exp QSA KV-cache specs on Ascend 310P (plan T1.4).

Everything runs host-side with NO NPU and NO Triton kernel: the QSA cache specs,
byte-math and slot mapping live in the host-safe
``vllm_ascend.models.qwen4_exp.kv_cache`` module (imports torch + vLLM base
specs + the dtype policy only).

Run with ``--noconftest`` (the shared tests/ut/conftest.py fails to import on
this host):

    python3 -m pytest -q --noconftest \
        tests/ut/qwen38_1m/test_kv_cache_specs.py
"""

import sys

import pytest
import torch
from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    MambaSpec,
    MLAAttentionSpec,
)

from vllm_ascend.models.qwen4_exp import kv_cache as kvc
from vllm_ascend.models.qwen4_exp.dtype_policy import (
    ASCEND_QWEN4EXP_DTYPE_POLICY as POLICY,
)

_MIB = 1024 * 1024


# --- 1M reference block math (feeds T8.x) --------------------------------
# max_model_len = 262_144 native YaRN window * factor 4 = 1_048_576.
# compress_ratio 4, indexer head_dim 128, 1 kv head, attention block 128.
#   compressed storage rows   = 1_048_576 / 4          = 262_144
#   compressed pages          = ceil(1_048_576 / 128)  = 8_192
#   BF16 compressed page      = (128/4) * 1 * 128 * 2  = 8_192 B
#   C8   compressed page      = (128/4) * 1 * 128 * 1  = 4_096 B
#   BF16 compressed total     = 8_192 * 8_192          = 64 MiB
#   C8   compressed total     = 4_096 * 8_192          = 32 MiB
#   raw ring page (key-only)  = 4 * 1 * 128 * 2        = 1_024 B (per request)
_EXPECTED_1M = {
    "BF16": {
        "compressed_page_bytes": 8_192,
        "compressed_bytes": 64 * _MIB,
        "ring_page_bytes": 1_024,
        "per_layer_bytes": 64 * _MIB + 1_024,
        "element_size_bytes": 2,
    },
    "C8": {
        "compressed_page_bytes": 4_096,
        "compressed_bytes": 32 * _MIB,
        "ring_page_bytes": 1_024,
        "per_layer_bytes": 32 * _MIB + 1_024,
        "element_size_bytes": 1,
    },
}


def test_module_import_does_not_pull_torch_npu_or_triton():
    # The 310P host lane has no NPU; the QSA spec module must stay light.
    assert "torch_npu" not in sys.modules
    # The backend advertises a Triton-free page path.
    assert kvc.AscendQSAStateBackend.uses_triton() is False


def test_raw_ring_spec_is_key_only_bytes():
    """The QSA raw ring is key-only; its page must NOT double for a V tensor."""
    ring = kvc.make_qsa_raw_ring_spec()
    assert isinstance(ring, kvc.AscendQSARawRingSpec)
    assert ring.block_size == kvc.qsa_ring_capacity(kvc.INDEXER_COMPRESS_RATIO)
    assert ring.num_kv_heads == kvc.INDEXER_KV_HEADS
    assert ring.head_size == kvc.INDEXER_HEAD_DIM
    # Key-only page-size math.
    expected = ring.block_size * ring.num_kv_heads * ring.head_size * get_dtype_size(ring.dtype)
    assert ring.page_size_bytes == expected == 1_024
    # A generic full-attention spec of the same geometry doubles (K + V);
    # proves the override is doing real, 310P-specific byte accounting.
    full = FullAttentionSpec(
        block_size=ring.block_size,
        num_kv_heads=ring.num_kv_heads,
        head_size=ring.head_size,
        dtype=ring.dtype,
    )
    assert full.page_size_bytes == 2 * expected
    # One block per request for the ring's lifetime.
    assert ring.max_num_blocks_per_req(None, kvc.QWEN4EXP_MAX_MODEL_LEN_1M) == 1
    assert ring.max_memory_usage_bytes(None) == ring.page_size_bytes
    assert ring.prefix_cacheable is False


def test_ring_capacity_rounds_to_whole_groups_with_speculation():
    assert kvc.qsa_ring_capacity(4, 0) == 4
    assert kvc.qsa_ring_capacity(4, 1) == 8  # 4 * ceil(5/4)
    assert kvc.qsa_ring_capacity(4, 2) == 8  # 4 * ceil(6/4)
    assert kvc.qsa_ring_capacity(4, 4) == 8  # 4 * ceil(8/4)
    assert kvc.qsa_ring_capacity(4, 5) == 12  # 4 * ceil(9/4)
    with pytest.raises(ValueError):
        kvc.qsa_ring_capacity(0)


def test_block_size_lcm_alignment():
    """Raw-ring capacity must align to (divide) the scheduler block size."""
    attn = kvc.DEFAULT_ATTENTION_BLOCK_SIZE  # 128
    cap = kvc.qsa_ring_capacity(kvc.INDEXER_COMPRESS_RATIO)  # 4
    compressed_block = attn
    sched = kvc.qsa_scheduler_block_size(attn, cap, compressed_block)
    # 4 divides 128, so the scheduler block size stays the attention block.
    assert sched == 128
    assert sched % cap == 0
    assert sched % attn == 0
    assert sched % compressed_block == 0
    # A ratio whose capacity does not divide the attention block raises the
    # scheduler block size via the LCM (documented deviation path).
    cap3 = kvc.qsa_ring_capacity(3)  # 3
    sched3 = kvc.qsa_scheduler_block_size(attn, cap3, attn)
    assert sched3 == 384
    assert sched3 % cap3 == 0 and sched3 % attn == 0
    # Speculative lookahead grows the ring to 8, still dividing 128.
    cap_spec = kvc.qsa_ring_capacity(4, 2)  # 8
    sched_spec = kvc.qsa_scheduler_block_size(attn, cap_spec, attn)
    assert sched_spec == 128 and sched_spec % cap_spec == 0


def test_compressed_spec_shape_and_bytes():
    comp = kvc.make_qsa_compressed_spec()
    assert isinstance(comp, MLAAttentionSpec)
    assert comp.block_size == kvc.DEFAULT_ATTENTION_BLOCK_SIZE
    assert comp.num_kv_heads == kvc.INDEXER_KV_HEADS
    assert comp.head_size == kvc.INDEXER_HEAD_DIM
    # compress ratio 4 exposed through the version-appropriate field.
    ratio = getattr(comp, "compress_ratio", None)
    if ratio is None:
        ratio = comp.tokens_per_state
    assert ratio == kvc.INDEXER_COMPRESS_RATIO == 4
    # BF16 layout page = (block/ratio) * heads * head_dim * 2.
    assert comp.page_size_bytes == 8_192
    c8 = kvc.make_qsa_compressed_spec(c8=True)
    assert c8.dtype == kvc.QSA_C8_DTYPE
    assert c8.page_size_bytes == 4_096  # exactly half of BF16
    with pytest.raises(ValueError):
        kvc.make_qsa_compressed_spec(attention_block_size=130)  # not / ratio


def test_dtype_policy_drives_element_sizes():
    # BF16 layout element size comes from the authoritative policy (fp16 on
    # 310P == 2 bytes, matching bf16 width on the reference forks).
    assert get_dtype_size(POLICY.qsa_indexer_dtype) == 2
    # C8 is the 1-byte e4m3 compressed layout.
    assert get_dtype_size(kvc.QSA_C8_DTYPE) == 1


@pytest.mark.parametrize("layout", ["BF16", "C8"])
def test_1m_bytes_per_chip_table_is_exact(layout):
    table = kvc.qsa_cache_bytes_table()
    row = table[layout]
    exp = _EXPECTED_1M[layout]
    assert row["max_model_len"] == 1_048_576
    assert row["compress_ratio"] == 4
    assert row["compressed_storage_rows"] == 262_144
    assert row["compressed_num_blocks"] == 8_192
    assert row["compressed_page_bytes"] == exp["compressed_page_bytes"]
    assert row["compressed_bytes"] == exp["compressed_bytes"]
    assert row["ring_page_bytes"] == exp["ring_page_bytes"]
    assert row["per_layer_bytes"] == exp["per_layer_bytes"]
    assert row["element_size_bytes"] == exp["element_size_bytes"]


def test_c8_compressed_is_exactly_half_of_bf16():
    table = kvc.qsa_cache_bytes_table()
    assert table["C8"]["compressed_bytes"] * 2 == table["BF16"]["compressed_bytes"]
    # The raw ring is unquantized in both layouts.
    assert table["C8"]["ring_page_bytes"] == table["BF16"]["ring_page_bytes"]


def test_bytes_table_scales_by_num_qsa_layers():
    table = kvc.qsa_cache_bytes_table(num_qsa_layers=8)
    for layout in ("BF16", "C8"):
        row = table[layout]
        assert row["total_bytes"] == row["per_layer_bytes"] * 8


def _tiny_qwen4exp_spec_set():
    """Build a tiny hybrid Qwen4Exp spec set (full-attn + GDN + QSA pair)."""
    full = FullAttentionSpec(
        block_size=kvc.DEFAULT_ATTENTION_BLOCK_SIZE,
        num_kv_heads=2,
        head_size=64,
        dtype=POLICY.kv_cache_dtype,
    )
    gdn = MambaSpec(
        shapes=((4, 8), (8,)),
        dtypes=(POLICY.mamba_conv_cache_dtype, POLICY.mamba_ssm_cache_dtype),
        block_size=kvc.DEFAULT_ATTENTION_BLOCK_SIZE,
    )
    ring = kvc.make_qsa_raw_ring_spec()
    comp = kvc.make_qsa_compressed_spec()
    return full, gdn, ring, comp


def test_build_kv_cache_config_groups_and_block_sizes():
    full, gdn, ring, comp = _tiny_qwen4exp_spec_set()
    config = kvc.build_qwen4exp_kv_cache_config(
        full_attention_layers={"layers.0.attn": full},
        mamba_layers={"layers.1.gdn": gdn},
        qsa_raw_ring_layers={"layers.0.qsa.ring": ring},
        qsa_compressed_layers={"layers.0.qsa.compressed": comp},
        num_blocks=4,
    )
    assert isinstance(config, KVCacheConfig)
    assert config.num_blocks == 4
    groups = config.kv_cache_groups
    assert len(groups) == 4
    specs = [g.kv_cache_spec for g in groups]
    assert any(isinstance(s, FullAttentionSpec) and not isinstance(s, MLAAttentionSpec) for s in specs)
    assert any(isinstance(s, MambaSpec) for s in specs)
    assert any(isinstance(s, kvc.AscendQSARawRingSpec) for s in specs)
    assert any(isinstance(s, MLAAttentionSpec) and not isinstance(s, kvc.AscendQSARawRingSpec) for s in specs)
    # Block sizes: full-attn / GDN / compressed use the attention block; the QSA
    # ring uses its (aligned) capacity.
    block_sizes = {type(g.kv_cache_spec).__name__: g.kv_cache_spec.block_size for g in groups}
    assert block_sizes["FullAttentionSpec"] == 128
    assert block_sizes["MambaSpec"] == 128
    assert block_sizes["MLAAttentionSpec"] == 128
    assert block_sizes["AscendQSARawRingSpec"] == 4
    # Every ring capacity must divide the scheduler LCM.
    sched = kvc.qsa_scheduler_block_size(*(g.kv_cache_spec.block_size for g in groups))
    assert sched % 4 == 0 and sched % 128 == 0
    # One placeholder tensor per group.
    assert len(config.kv_cache_tensors) == 4


def test_merged_layers_share_one_group():
    ring = kvc.make_qsa_raw_ring_spec()
    groups = kvc.build_qwen4exp_kv_cache_groups(
        qsa_raw_ring_layers={"layers.0.qsa.ring": ring, "layers.2.qsa.ring": ring},
    )
    assert len(groups) == 1
    assert sorted(groups[0].layer_names) == ["layers.0.qsa.ring", "layers.2.qsa.ring"]


# --- Triton-free page/slot mapping (pure torch) --------------------------
def test_circular_slot_mapping_torch_no_triton():
    # 2 requests, ring capacity 4. block_table[req, 0] is the fixed ring block.
    block_table = torch.tensor([[5, 0], [9, 0]], dtype=torch.int64)
    token_to_req = torch.tensor([0, 0, 1], dtype=torch.int64)
    logical_positions = torch.tensor([6, 7, 3], dtype=torch.int64)
    slots = kvc.circular_qsa_slot_mapping(block_table, token_to_req, logical_positions, compressor_state_size=4)
    # req0 block 5: pos6 -> 5*4 + (6%4)=22 ; pos7 -> 5*4 + 3 = 23
    # req1 block 9: pos3 -> 9*4 + 3 = 39
    assert slots.tolist() == [22, 23, 39]


def test_compressed_slot_mapping_is_boundary_only():
    block_table = torch.tensor([[3, 4]], dtype=torch.int64)
    token_to_req = torch.tensor([0, 0, 0, 0], dtype=torch.int64)
    logical_positions = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
    slots = kvc.compressed_qsa_slot_mapping(
        block_table,
        token_to_req,
        logical_positions,
        storage_block_size=32,
        compress_ratio=4,
    )
    # Only the group boundary (pos where (pos+1) % ratio == 0, i.e. pos 3) maps.
    # compressed_pos = 3//4 = 0 -> block_table[0,0]=3 -> 3*32 + 0 = 96.
    assert slots.tolist() == [-1, -1, -1, 96]


def test_metadata_builder_selects_torch_ring_vs_compressed():
    builder = kvc.AscendQSAStateBackend.get_metadata_builder()
    assert builder is kvc.build_qsa_metadata_torch
    block_table = torch.tensor([[2, 0]], dtype=torch.int64)
    token_to_req = torch.tensor([0, 0], dtype=torch.int64)
    logical_positions = torch.tensor([3, 4], dtype=torch.int64)
    ring_slots = builder(
        block_table=block_table,
        token_to_req=token_to_req,
        logical_positions=logical_positions,
        storage_block_size=32,
        compress_ratio=4,
        circular_buffer_size=4,
    )
    # ring: pos3 -> 2*4+3=11 ; pos4 -> 2*4+(4%4)=8
    assert ring_slots.tolist() == [11, 8]
    comp_slots = builder(
        block_table=block_table,
        token_to_req=token_to_req,
        logical_positions=logical_positions,
        storage_block_size=32,
        compress_ratio=4,
        circular_buffer_size=0,
    )
    # compressed boundary only: pos3 is a boundary (2*... ) -> 2*32+0=64 ; pos4 not.
    assert comp_slots.tolist() == [64, -1]
