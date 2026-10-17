# SPDX-License-Identifier: Apache-2.0
"""Scheduler metadata for the eager 310P M3 ND-cache path."""

from dataclasses import dataclass
from typing import ClassVar

import torch
from vllm.v1.attention.backend import AttentionCGSupport, AttentionMetadata, AttentionMetadataBuilder


@dataclass
class M3EagerMetadata(AttentionMetadata):
    query_starts: tuple[int, ...]
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    max_seq_len: int
    num_actual_tokens: int


class M3EagerMetadataBuilder(AttentionMetadataBuilder[M3EagerMetadata]):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.NEVER
    reorder_batch_threshold = 1

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self._init_reorder_batch_threshold(1, supports_spec_as_decode=False)

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        common = common_attn_metadata
        if common_prefix_len:
            raise ValueError("M3 310P does not support cascade attention")
        if common.query_start_loc_cpu.device.type != "cpu":
            raise ValueError("M3 eager query boundaries must be scheduler-owned CPU metadata")
        starts = tuple(common.query_start_loc_cpu[: common.num_reqs + 1].tolist())
        if not starts or starts[0] != 0 or starts[-1] != common.num_actual_tokens:
            raise ValueError("M3 query boundaries disagree with actual token count")
        if any(a > b for a, b in zip(starts, starts[1:])):
            raise ValueError("M3 query boundaries must be monotone")
        return M3EagerMetadata(
            starts,
            common.seq_lens,
            common.block_table_tensor,
            common.slot_mapping,
            common.max_seq_len,
            common.num_actual_tokens,
        )
