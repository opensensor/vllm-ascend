#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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
# This file is a part of the vllm-ascend project.
#
"""The recurrent GDN op's per-step inputs must be derived once, not per layer.

`flat_state_indices` / `actual_seq_lengths` depend only on per-step metadata, so
recomputing them in each of the 48 GDN layers costs ~7 NPU launches x 47
redundant layers per decoded token on a device bound by launch count.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from vllm_ascend._310p.ops.fla.chunk_gated_delta_rule import CHUNK_SIZE, build_varlen_chunk_plan
from vllm_ascend._310p.ops.fla.gdn_310 import _cached_chunk_plan, _cached_recurrent_step_meta


def _cu_and_indices(num_decodes):
    cu_seqlens = torch.arange(num_decodes + 1, dtype=torch.int32)
    ssm_state_indices = torch.arange(num_decodes, dtype=torch.int32)
    return cu_seqlens, ssm_state_indices


def test_derived_values_are_correct():
    num_decodes = 4
    cu, idx = _cu_and_indices(num_decodes)
    meta = SimpleNamespace()

    flat, lens = _cached_recurrent_step_meta(meta, "decode", cu, idx, num_decodes)

    torch.testing.assert_close(lens, torch.ones(num_decodes, dtype=torch.int32), rtol=0, atol=0)
    torch.testing.assert_close(flat, idx.to(torch.int32), rtol=0, atol=0)
    assert flat.dtype == torch.int32
    assert lens.dtype == torch.int32


def test_second_call_reuses_the_same_tensors():
    """The whole point: layers 1..47 must not redo the work layer 0 did."""
    num_decodes = 3
    cu, idx = _cu_and_indices(num_decodes)
    meta = SimpleNamespace()

    first = _cached_recurrent_step_meta(meta, "decode", cu, idx, num_decodes)
    second = _cached_recurrent_step_meta(meta, "decode", cu, idx, num_decodes)

    assert first[0] is second[0]
    assert first[1] is second[1]


def test_spec_and_decode_slots_do_not_collide():
    """A mixed batch derives both; one must not serve the other's tensors."""
    meta = SimpleNamespace()
    cu_d, idx_d = _cu_and_indices(2)
    cu_s, idx_s = _cu_and_indices(5)

    decode = _cached_recurrent_step_meta(meta, "decode", cu_d, idx_d, 2)
    spec = _cached_recurrent_step_meta(meta, "spec", cu_s, idx_s, 5)

    assert decode[0] is not spec[0]
    assert decode[1].numel() == 2
    assert spec[1].numel() == 5


def test_cache_is_per_metadata_object():
    """A fresh GDNAttentionMetadata is built each step, so a new object must
    not see the previous step's tensors."""
    cu, idx = _cu_and_indices(2)
    first = _cached_recurrent_step_meta(SimpleNamespace(), "decode", cu, idx, 2)
    second = _cached_recurrent_step_meta(SimpleNamespace(), "decode", cu, idx, 2)

    assert first[0] is not second[0]
    torch.testing.assert_close(first[0], second[0], rtol=0, atol=0)


def test_chunk_plan_is_built_once_per_step():
    """Layers 1..47 must reuse layer 0's plan, not repeat its D2H copy."""
    meta = SimpleNamespace()
    cu = torch.tensor([0, 8192], dtype=torch.int32)

    first = _cached_chunk_plan(meta, cu)
    for _ in range(47):
        assert _cached_chunk_plan(meta, cu) is first


def test_chunk_plan_is_rebuilt_for_a_different_cu_seqlens():
    """A second cu_seqlens in one step must not be served the first one's plan."""
    meta = SimpleNamespace()
    cu_a = torch.tensor([0, 8192], dtype=torch.int32)
    cu_b = torch.tensor([0, 64, 300], dtype=torch.int32)

    plan_a = _cached_chunk_plan(meta, cu_a)
    plan_b = _cached_chunk_plan(meta, cu_b)

    assert plan_b is not plan_a
    assert plan_a.cu_list == [0, 8192]
    assert plan_b.cu_list == [0, 64, 64 + 256]
    # and going back re-derives rather than returning stale state
    assert _cached_chunk_plan(meta, cu_a).cu_list == plan_a.cu_list


def test_chunk_plan_matches_a_direct_build():
    meta = SimpleNamespace()
    cu = torch.tensor([0, 100, 100, 260], dtype=torch.int32)

    cached = _cached_chunk_plan(meta, cu)
    direct = build_varlen_chunk_plan(cu.to(torch.int64).cpu(), CHUNK_SIZE)

    assert cached.cu_list == direct.cu_list
    assert cached.seq_ranges == direct.seq_ranges
    assert cached.segments == direct.segments
    assert cached.chunk_indices == direct.chunk_indices


def test_chunk_plan_cache_does_not_leak_between_steps():
    """The builder makes a fresh metadata object per step, so a new one must miss."""
    cu = torch.tensor([0, 8192], dtype=torch.int32)
    first = _cached_chunk_plan(SimpleNamespace(), cu)
    second = _cached_chunk_plan(SimpleNamespace(), cu)
    assert first is not second
    assert first.cu_list == second.cu_list
