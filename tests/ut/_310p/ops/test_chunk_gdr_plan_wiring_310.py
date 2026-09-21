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
"""Passing a prebuilt chunk plan must not change what reaches the kernels.

The unit tests next door check the padding helpers on their own. This one runs
``chunk_gated_delta_rule_310`` itself with the AscendC ops stubbed out, so it
covers the wiring: that the same ``cu_seqlens``/``chunk_indices``/state count
and the same padded tensors reach the kernel whether the plan was handed in or
derived inside, and that the output is unpadded identically.
"""

import torch

from vllm_ascend._310p.ops.fla import chunk_gated_delta_rule as cgdr

CHUNK = 64
H_QK, H_V, D = 4, 12, 128


class _RecordingOps:
    """Stands in for torch.ops._C_ascend, recording what the op layer receives."""

    def __init__(self):
        self.calls = {}

    def chunk_gated_delta_rule_fwd_h(self, k, w, u, *, g, gk, initial_state, output_final_state,
                                     chunk_size, save_new_value, cu_seqlens, chunk_indices,
                                     use_exp2, transpose_state_layout):
        self.calls["fwd_h"] = dict(
            k=k.clone(), w=w.clone(), u=u.clone(), g=g.clone(),
            cu_seqlens=list(cu_seqlens), chunk_indices=list(chunk_indices),
            initial_state=initial_state.clone(),
        )
        # h is only forwarded to chunk_fwd_o; v_new keeps u's shape.
        return torch.zeros(1), u, initial_state

    def chunk_fwd_o(self, q, k, v_new, h, scale, *, g, g_gamma, cu_seqlens, chunk_indices,
                    chunk_size, transpose_state_layout):
        self.calls["fwd_o"] = dict(q=q.clone(), cu_seqlens=list(cu_seqlens),
                                   chunk_indices=list(chunk_indices), scale=scale)
        # The real kernel emits value heads, not query heads, so shape this off
        # v_new. The ramp makes it position-dependent, so a wrong unpadding
        # shows up as a wrong value and not only as a wrong shape.
        b, h_v, t, d_v = v_new.shape
        ramp = torch.arange(t, dtype=torch.float32).view(1, 1, t, 1)
        return (v_new.to(torch.float32) + ramp).to(torch.float16)


def _run(offsets, *, with_plan, ops, monkeypatch):
    total = offsets[-1]
    torch.manual_seed(total + len(offsets))
    q = torch.randn(total, H_QK, D, dtype=torch.float16)
    k = torch.randn(total, H_QK, D, dtype=torch.float16)
    v = torch.randn(total, H_V, D, dtype=torch.float16)
    g = torch.nn.functional.logsigmoid(torch.randn(total, H_V, dtype=torch.float32))
    beta = torch.rand(total, H_V, dtype=torch.float32)
    cu = torch.tensor(offsets, dtype=torch.int64)

    monkeypatch.setattr(torch.ops, "_C_ascend", ops, raising=False)
    monkeypatch.setattr(cgdr, "_require_ascend_chunk_ops", lambda *a, **kw: None)

    plan = cgdr.build_varlen_chunk_plan(cu, CHUNK) if with_plan else None
    out, state = cgdr.chunk_gated_delta_rule_310(
        q=q, k=k, v=v, g=g, beta=beta,
        initial_state=None, output_final_state=True,
        cu_seqlens=cu, head_first=False, use_qk_l2norm_in_kernel=False,
        chunk_plan=plan,
    )
    return out, state


OFFSET_SETS = [
    [0, 8192],            # one chunk-aligned sequence
    [0, 5839],            # one sequence needing padding
    [0, 64, 192, 333],    # ragged batch
    [0, 100, 100, 260],   # an empty sequence in the middle
]


def test_prebuilt_plan_matches_derived_plan(monkeypatch):
    for offsets in OFFSET_SETS:
        rec_a, rec_b = _RecordingOps(), _RecordingOps()
        out_a, st_a = _run(offsets, with_plan=False, ops=rec_a, monkeypatch=monkeypatch)
        out_b, st_b = _run(offsets, with_plan=True, ops=rec_b, monkeypatch=monkeypatch)

        assert rec_a.calls["fwd_h"]["cu_seqlens"] == rec_b.calls["fwd_h"]["cu_seqlens"], offsets
        assert rec_a.calls["fwd_h"]["chunk_indices"] == rec_b.calls["fwd_h"]["chunk_indices"], offsets
        assert rec_a.calls["fwd_o"]["cu_seqlens"] == rec_b.calls["fwd_o"]["cu_seqlens"], offsets
        for field in ("k", "w", "u", "g", "initial_state"):
            torch.testing.assert_close(
                rec_a.calls["fwd_h"][field], rec_b.calls["fwd_h"][field], msg=f"{offsets} {field}"
            )
        torch.testing.assert_close(rec_a.calls["fwd_o"]["q"], rec_b.calls["fwd_o"]["q"])
        torch.testing.assert_close(out_a, out_b)
        torch.testing.assert_close(st_a, st_b)


def test_output_is_unpadded_back_to_the_input_length(monkeypatch):
    for offsets in OFFSET_SETS:
        out, state = _run(offsets, with_plan=True, ops=_RecordingOps(), monkeypatch=monkeypatch)
        assert out.shape == (offsets[-1], H_V, D), offsets
        assert state.shape == (len(offsets) - 1, H_V, D, D), offsets


def test_padded_token_count_is_chunk_aligned_per_sequence(monkeypatch):
    for offsets in OFFSET_SETS:
        rec = _RecordingOps()
        _run(offsets, with_plan=True, ops=rec, monkeypatch=monkeypatch)
        cu_padded = rec.calls["fwd_h"]["cu_seqlens"]
        assert cu_padded[0] == 0
        for i in range(len(cu_padded) - 1):
            seg = cu_padded[i + 1] - cu_padded[i]
            assert seg % CHUNK == 0, (offsets, seg)
            raw = offsets[i + 1] - offsets[i]
            assert 0 <= seg - raw < CHUNK, (offsets, raw, seg)
