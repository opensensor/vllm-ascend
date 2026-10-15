# SPDX-License-Identifier: Apache-2.0
"""Device metadata materialization for admitted, exact-size MTP2 decode.

Request/page allocation remains outside this graph. It consumes the live compact
block table, sequence lengths and normalized accepted counts, and updates the
same buffers already captured by the target model. No tensor value is read back
except in explicit verification mode.
"""

from dataclasses import replace
from types import MethodType

import numpy as np
import torch

MTP_WIDTH = 3
MAX_REQUESTS = 4
COMPARE_FIELDS = (
    "spec_query_start_loc",
    "spec_state_indices_tensor",
    "spec_sequence_masks",
    "spec_token_indx",
    "non_spec_token_indx",
    "num_accepted_tokens",
)


def uniform_decode_rows(builder, common, accepted, drafts, actual):
    """Admission uses existing CPU layout metadata, never device token values."""
    n = common.num_reqs
    if (
        not 1 <= n <= MAX_REQUESTS
        or actual != n
        or not builder.use_full_cuda_graph
        or not builder.use_spec_decode
        or builder.num_spec != MTP_WIDTH - 1
        or builder._USE_COMMON_KERNEL_METADATA
        or common.num_actual_tokens != n * MTP_WIDTH
        or common.max_query_len != MTP_WIDTH
        or accepted is None
        or accepted.shape != (n,)
        or drafts is None
        or drafts.device.type != "cpu"
        or drafts.shape != (n,)
        or common.query_start_loc_cpu.device.type != "cpu"
        or common.query_start_loc_cpu.shape != (n + 1,)
        or common.is_prefilling is None
        or common.is_prefilling.device.type != "cpu"
        or common.is_prefilling.shape != (n,)
    ):
        return False
    return (
        np.all(drafts.numpy() == MTP_WIDTH - 1)
        and np.array_equal(common.query_start_loc_cpu.numpy(), np.arange(n + 1) * MTP_WIDTH)
        and not np.any(common.is_prefilling.numpy())
    )


def fill_uniform_metadata(table, lengths, accepted, outputs, *, block_size, boundaries, tokens):
    """Pure tensor transform; physical block ownership is an admission contract."""
    n = lengths.numel()
    if (
        table.ndim != 2
        or not 1 <= n <= MAX_REQUESTS
        or table.shape[0] != n
        or accepted.shape != (n,)
        or type(block_size) is not int
        or block_size <= 0
    ):
        raise ValueError("invalid uniform MTP2 metadata geometry")
    offsets = torch.arange(MTP_WIDTH, dtype=torch.int32, device=table.device)
    columns = ((lengths - 1) // block_size).clamp(min=0).unsqueeze(1) + offsets
    outputs["spec_state_indices_tensor"].copy_(table.gather(1, columns.to(torch.int64)))
    outputs["spec_sequence_masks"].fill_(True)
    outputs["spec_query_start_loc"].copy_(boundaries)
    outputs["spec_token_indx"].copy_(tokens)
    outputs["num_accepted_tokens"].copy_(accepted)


class GDNMetadataGraphs:
    def __init__(self, runner, group, builder, metadata_type):
        if (
            builder.vllm_config.cache_config.mamba_cache_mode != "align"
            or builder.kv_cache_spec.num_speculative_blocks != 2
        ):
            raise ValueError("metadata graphs require align-mode MTP2 state")
        self.runner, self.group, self.builder = runner, group, builder
        self.original = builder.build
        self.enabled = self.verify = self.failed = False
        self.replays = self.checked = self.fallbacks = 0
        self.records = {}
        table = runner.input_batch.block_table.block_tables[group].block_table.gpu
        self.table = table
        self.lengths = torch.full((MAX_REQUESTS,), 3, dtype=torch.int32, device=table.device)
        self.accepted = torch.ones_like(self.lengths)
        self.stream = torch.npu.Stream(device=table.device)
        self.stream.wait_stream(torch.npu.current_stream())
        # Retain all owners before submission; failed capture must not release
        # memory still referenced by the runtime.
        with torch.npu.stream(self.stream):
            for n in range(1, MAX_REQUESTS + 1):
                metadata = metadata_type(
                    num_prefills=0,
                    num_prefill_tokens=0,
                    num_decodes=0,
                    num_decode_tokens=0,
                    num_spec_decodes=n,
                    num_spec_decode_tokens=n * MTP_WIDTH,
                    num_actual_tokens=n * MTP_WIDTH,
                    spec_query_start_loc=builder.spec_query_start_loc[: n + 1],
                    spec_state_indices_tensor=builder.spec_state_indices_tensor[:n],
                    spec_sequence_masks=builder.spec_sequence_masks[:n],
                    spec_token_indx=builder.spec_token_indx[: n * MTP_WIDTH],
                    non_spec_token_indx=builder.non_spec_token_indx[:0],
                    num_accepted_tokens=builder.num_accepted_tokens[:n],
                )
                metadata.non_spec_prefill_metadata = metadata.non_spec_decode_metadata = None
                builder._attach_spec_decode_metadata(metadata)
                record = dict(
                    metadata=metadata,
                    graph=torch.npu.NPUGraph(),
                    boundaries=torch.arange(n + 1, dtype=torch.int32, device=table.device) * MTP_WIDTH,
                    tokens=torch.arange(n * MTP_WIDTH, dtype=torch.int32, device=table.device),
                    outputs={key: getattr(metadata, key) for key in COMPARE_FIELDS},
                )
                self.records[n] = record

                def fill(n=n, record=record):
                    fill_uniform_metadata(
                        self.table[:n],
                        self.lengths[:n],
                        self.accepted[:n],
                        record["outputs"],
                        block_size=builder.kv_cache_spec.block_size,
                        boundaries=record["boundaries"],
                        tokens=record["tokens"],
                    )

                for _ in range(3):
                    fill()
                torch.npu.synchronize()
                with torch.npu.graph(record["graph"], stream=self.stream):
                    fill()
                torch.npu.synchronize()
        torch.npu.current_stream().wait_stream(self.stream)
        torch.npu.synchronize()

    def build(
        self,
        common_prefix_len,
        common_attn_metadata,
        num_accepted_tokens=None,
        num_decode_draft_tokens_cpu=None,
        fast_build=False,
        num_actual_reqs=None,
    ):
        if self.failed:
            raise RuntimeError("metadata graph failed; retain owners and stop submission")
        common = common_attn_metadata
        args = dict(
            common_prefix_len=common_prefix_len,
            common_attn_metadata=common,
            num_accepted_tokens=num_accepted_tokens,
            num_decode_draft_tokens_cpu=num_decode_draft_tokens_cpu,
            fast_build=fast_build,
            num_actual_reqs=num_actual_reqs,
        )
        n = common.num_reqs
        if not self.enabled or not uniform_decode_rows(
            self.builder, common, num_accepted_tokens, num_decode_draft_tokens_cpu, num_actual_reqs
        ):
            self.fallbacks += 1
            return self.original(**args)
        if (
            common.block_table_tensor.data_ptr() != self.table.data_ptr()
            or common.block_table_tensor.shape != self.table[:n].shape
            or common.block_table_tensor.stride() != self.table.stride()
            or common.seq_lens.dtype != self.lengths.dtype
            or common.seq_lens.shape != (n,)
            or common.seq_lens.device != self.table.device
            or num_accepted_tokens.device != self.table.device
            or num_accepted_tokens.dtype != self.accepted.dtype
        ):
            self.fallbacks += 1
            return self.original(**args)
        expected = None
        if self.verify:
            reference = self.original(**args)
            expected = {name: getattr(reference, name).clone() for name in COMPARE_FIELDS}
        self.failed = True
        record = self.records[n]
        self.lengths[:n].copy_(common.seq_lens)
        self.accepted[:n].copy_(num_accepted_tokens)
        record["graph"].replay()
        if expected is not None:
            for name, value in expected.items():
                torch.testing.assert_close(record["outputs"][name].cpu(), value.cpu(), rtol=0, atol=0)
            self.checked += 1
        result = replace(record["metadata"])
        result.non_spec_prefill_metadata = result.non_spec_decode_metadata = None
        self.builder._attach_spec_decode_metadata(result)
        self.replays += 1
        self.failed = False
        return result

    def install(self):
        owner = self

        def build(builder, *args, **kwargs):
            if builder is not owner.builder:
                raise RuntimeError("metadata owner belongs to another builder")
            return owner.build(*args, **kwargs)

        self.builder.build = MethodType(build, self.builder)

    def status(self):
        return dict(
            group=self.group,
            enabled=self.enabled,
            verify=self.verify,
            failed=self.failed,
            replays=self.replays,
            checked=self.checked,
            fallbacks=self.fallbacks,
            capture_requests=sorted(self.records),
        )
