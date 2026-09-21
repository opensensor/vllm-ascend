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
"""Grouping the WY gram matrix by K head must not change what the kernels get.

Q/K carry 16 heads and V carries 48, so the live path expands K threefold before
the WY matmuls. But (k_beta @ key^T)[i, j] == beta[i] * <key[i], key[j]>, so the
gram matrix is the same for all three V heads in a group and only the beta
scaling differs. VLLM_ASCEND_GDN_WY_GROUPED_GRAM=1 builds it once per K head.
"""

import torch

from vllm_ascend._310p.ops.fla import chunk_gated_delta_rule as cgdr

CHUNK = 64
B, T, H_QK, H_V, D = 1, 256, 4, 12, 128


def _inputs(seed=5):
    torch.manual_seed(seed)
    # l2-normalised q/k: use_qk_l2norm_in_kernel=True is what the live path
    # passes, and without it the UT inverse overflows fp16 on random inputs.
    q = torch.nn.functional.normalize(torch.randn(B, T, H_QK, D), dim=-1).to(torch.float16)
    k = torch.nn.functional.normalize(torch.randn(B, T, H_QK, D), dim=-1).to(torch.float16)
    v = torch.randn(B, T, H_V, D, dtype=torch.float16)
    g = torch.nn.functional.logsigmoid(torch.randn(B, T, H_V, dtype=torch.float32))
    beta = torch.rand(B, T, H_V, dtype=torch.float32)
    return q, k, v, g, beta


def _build(monkeypatch, grouped, args):
    monkeypatch.setattr(cgdr, "_WY_GROUPED_GRAM", grouped)
    return cgdr._compute_kernel_inputs_from_torch_wy(*args, CHUNK)


def _max_ulps(want, got):
    """Largest difference between two fp16 tensors, in units of the last place.

    The two forms are algebraically identical but multiply in a different order,
    so the fp16 result can land one representable value apart. Measuring in ULPs
    says that directly; a relative tolerance just has to be guessed loose enough
    to cover it.
    """
    a, b = want.to(torch.float32), got.to(torch.float32)
    if want.dtype != torch.float16:
        return 0.0 if torch.equal(a, b) else float("inf")
    # Floored at fp16's smallest subnormal step: near zero the exponent-derived
    # spacing collapses, and dividing by it turns a legitimate one-step rounding
    # into a meaningless ratio.
    ulp = torch.pow(2.0, torch.floor(torch.log2(a.abs().clamp(min=2.0 ** -24))) - 10)
    ulp = ulp.clamp(min=2.0 ** -24)
    return ((a - b).abs() / ulp).max().item()


def test_grouped_gram_matches_the_expanded_form(monkeypatch):
    args = _inputs()
    default = _build(monkeypatch, False, args)
    grouped = _build(monkeypatch, True, args)

    names = ("q_kernel", "k_kernel", "w_kernel", "u_kernel", "g_kernel")
    for name, want, got in zip(names, default, grouped):
        assert want.shape == got.shape, name
        assert want.dtype == got.dtype, name
        assert torch.isfinite(want.to(torch.float32)).all(), f"{name} reference is not finite"
        ulps = _max_ulps(want, got)
        assert ulps <= 1.0, f"{name} differs by {ulps:.2f} fp16 ULPs"

    # q/k/g are not touched by either form, so they must be bit-identical.
    for name, want, got in zip(names, default, grouped):
        if name in ("q_kernel", "k_kernel", "g_kernel"):
            assert torch.equal(want, got), f"{name} should be untouched"


def test_grouped_gram_is_off_by_default():
    """The prefill path for every request: it must opt in, not opt out."""
    assert cgdr._WY_GROUPED_GRAM is False


def test_grouped_path_declines_a_non_divisible_head_count(monkeypatch):
    """5 V heads over 4 K heads is a caller error, and must still raise."""
    monkeypatch.setattr(cgdr, "_WY_GROUPED_GRAM", True)
    torch.manual_seed(1)
    q = torch.randn(B, CHUNK, H_QK, D, dtype=torch.float16)
    k = torch.randn(B, CHUNK, H_QK, D, dtype=torch.float16)
    v = torch.randn(B, CHUNK, 5, D, dtype=torch.float16)
    g = torch.randn(B, CHUNK, 5, dtype=torch.float32)
    beta = torch.rand(B, CHUNK, 5, dtype=torch.float32)
    try:
        cgdr._compute_kernel_inputs_from_torch_wy(q, k, v, g, beta, CHUNK)
    except ValueError:
        return
    raise AssertionError("expected ValueError for Hv=5 over Hqk=4")


def test_grouped_gram_handles_several_chunks_and_seeds(monkeypatch):
    for seed in (0, 11, 42):
        args = _inputs(seed)
        default = _build(monkeypatch, False, args)
        grouped = _build(monkeypatch, True, args)
        for want, got in zip(default, grouped):
            ulps = _max_ulps(want, got)
            assert ulps <= 1.0, f"seed {seed} differs by {ulps:.2f} fp16 ULPs"
