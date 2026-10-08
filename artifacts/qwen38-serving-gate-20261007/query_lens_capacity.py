# SPDX-License-Identifier: Apache-2.0
"""Backport the pinned qLens capacity fix into an older resident 310P runtime.

The controller drains requests and recaptures target/draft graphs before
resuming. No model weights, cache geometry, or numerical kernels change.
"""

import torch


def fill_query_lens_cpu(self, num_reqs, query_start_loc_cpu, is_drafting=False):
    if self._query_lens_cpu_buffer is None:
        return (query_start_loc_cpu[1 : num_reqs + 1] - query_start_loc_cpu[:num_reqs]).contiguous()
    if num_reqs > self._query_lens_cpu_buffer.numel():
        # Never let out= resize a slice of the pinned allocation. Expanded MTP
        # metadata can have more rows than scheduler.max_num_seqs.
        self._query_lens_cpu_buffer = torch.empty(
            num_reqs,
            dtype=torch.int32,
            device="cpu",
            pin_memory=self._query_lens_cpu_buffer.is_pinned(),
        )
    buffer = self._query_lens_cpu_buffer[:num_reqs]
    if is_drafting:
        buffer = buffer.clone()
    torch.sub(query_start_loc_cpu[1 : num_reqs + 1], query_start_loc_cpu[:num_reqs], out=buffer)
    return buffer


def replacements():
    return {
        "vllm_ascend._310p.attention.metadata_builder:AscendAttentionMetadataBuilder310._fill_query_lens_cpu": (
            fill_query_lens_cpu
        )
    }
