#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The per-step chunk plan must reproduce the per-layer padding exactly."""

import torch

from vllm_ascend._310p.ops.fla.chunk_gated_delta_rule import (
    _apply_varlen_chunk_plan,
    _ceil_div,
    _chunk_indices_from_offsets,
    _pad_varlen_to_chunk,
    build_varlen_chunk_plan,
)

CHUNK = 64
H_QK, H_V, D = 4, 12, 128


def _inputs(total_tokens: int):
    torch.manual_seed(total_tokens)
    q = torch.randn(1, total_tokens, H_QK, D, dtype=torch.float16)
    k = torch.randn(1, total_tokens, H_QK, D, dtype=torch.float16)
    v = torch.randn(1, total_tokens, H_V, D, dtype=torch.float16)
    g = torch.randn(1, total_tokens, H_V, dtype=torch.float32)
    beta = torch.rand(1, total_tokens, H_V, dtype=torch.float32)
    return q, k, v, g, beta


def _reference_pad(cu, q, k, v, g, beta):
    """The layout the op produced before the plan was hoisted out of it."""
    parts = {name: [] for name in "qkvgb"}
    seq_ranges = []
    padded_cu = [0]
    cursor = 0
    for i in range(cu.numel() - 1):
        start, end = int(cu[i].item()), int(cu[i + 1].item())
        seq_len = end - start
        padded_len = _ceil_div(seq_len, CHUNK) * CHUNK if seq_len > 0 else 0
        pad = padded_len - seq_len
        for name, x in zip("qkvgb", (q, k, v, g, beta)):
            seg = x[:, start:end]
            if pad > 0:
                seg = torch.cat((seg, x.new_zeros((1, pad, *x.shape[2:]))), dim=1)
            parts[name].append(seg)
        seq_ranges.append((0, cursor, cursor + seq_len))
        cursor += seq_len
        padded_cu.append(padded_cu[-1] + padded_len)
    padded = tuple(torch.cat(parts[n], dim=1) for n in "qkvgb")
    return padded, seq_ranges, padded_cu


SEQ_SETS = [
    [0, 8192],           # one full chunk-aligned sequence: the editor case
    [0, 5839],           # one sequence needing padding
    [0, 64],             # exactly one chunk
    [0, 1, 65, 200],     # several ragged sequences, one of length 1
    [0, 128, 128, 300],  # an empty sequence in the middle
    [0],                 # no sequences at all
]


def test_plan_matches_previous_padding():
    for offsets in SEQ_SETS:
        total = offsets[-1] if offsets else 0
        cu = torch.tensor(offsets, dtype=torch.int64)
        q, k, v, g, beta = _inputs(max(total, 1))
        if total == 0:
            q, k, v, g, beta = (x[:, :0] for x in (q, k, v, g, beta))

        (ref_padded, ref_ranges, ref_cu) = _reference_pad(cu, q, k, v, g, beta) if len(offsets) > 1 else (
            tuple(x[:, :0] for x in (q, k, v, g, beta)), [], [0]
        )

        plan = build_varlen_chunk_plan(cu, CHUNK)
        got = _apply_varlen_chunk_plan(q, k, v, g, beta, plan)

        assert plan.seq_ranges == ref_ranges, offsets
        assert plan.cu_list == ref_cu, offsets
        for name, a, b in zip("qkvgb", got, ref_padded):
            assert a.shape == b.shape, (offsets, name, a.shape, b.shape)
            torch.testing.assert_close(a, b, msg=f"{offsets} {name}")


def test_plan_agrees_with_the_wrapper_it_replaced():
    for offsets in SEQ_SETS:
        if len(offsets) < 2:
            continue
        cu = torch.tensor(offsets, dtype=torch.int64)
        q, k, v, g, beta = _inputs(offsets[-1])
        *wrapped, ranges, cu_kernel = _pad_varlen_to_chunk(q, k, v, g, beta, cu, CHUNK)
        plan = build_varlen_chunk_plan(cu, CHUNK)
        direct = _apply_varlen_chunk_plan(q, k, v, g, beta, plan)
        assert ranges == plan.seq_ranges
        assert cu_kernel.tolist() == plan.cu_list
        for a, b in zip(wrapped, direct):
            torch.testing.assert_close(a, b)


def test_chunk_indices_match_offsets():
    for offsets in SEQ_SETS:
        cu = torch.tensor(offsets, dtype=torch.int64)
        plan = build_varlen_chunk_plan(cu, CHUNK)
        assert plan.chunk_indices == _chunk_indices_from_offsets(plan.cu_list, CHUNK)
        # every padded sequence contributes exactly ceil(len / chunk) chunks
        expected = sum(
            _ceil_div(plan.cu_list[i + 1] - plan.cu_list[i], CHUNK) for i in range(len(plan.cu_list) - 1)
        )
        assert len(plan.chunk_indices) == 2 * expected


def test_single_sequence_padding_is_not_copied_twice():
    """One aligned sequence should hand back the input, not a cat of one part."""
    cu = torch.tensor([0, 8192], dtype=torch.int64)
    q, k, v, g, beta = _inputs(8192)
    plan = build_varlen_chunk_plan(cu, CHUNK)
    q_pad, *_ = _apply_varlen_chunk_plan(q, k, v, g, beta, plan)
    assert q_pad.data_ptr() == q.data_ptr()
