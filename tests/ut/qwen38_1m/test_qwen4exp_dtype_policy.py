# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the authoritative Qwen4Exp dtype policy (plan T1.2, PRD R4).

The dtype policy is the single source of truth every downstream Qwen4Exp
component (T1.3/T1.4/T1.5/T3.1/T4.x/T5.x/T6.1) and the T0.5/T8.3 byte-math
harnesses consume. These tests pin the policy table shape and enforce that
the layer modules never hardcode dtype literals (they must READ the policy).

Run with ``--noconftest`` (the shared tests/ut/conftest.py fails to import on
this host):

    python3 -m pytest -q --noconftest \
        tests/ut/qwen38_1m/test_qwen4exp_dtype_policy.py
"""

import re
from pathlib import Path

import pytest
import torch

from vllm_ascend.models.qwen4_exp.dtype_policy import (
    ASCEND_QWEN4EXP_DTYPE_POLICY,
    REQUIRED_CAST_SITES,
    Qwen4ExpDtypePolicy,
)

_PKG_DIR = Path(__file__).parents[3] / "vllm_ascend" / "models" / "qwen4_exp"

# Every layer module must read dtype from the policy; only dtype_policy.py may
# spell dtype literals.
_LAYER_MODULES = (
    "model.py",
    "ple_layer.py",
    "qsa.py",
    "indexer_qsa.py",
    "ngram_embedding.py",
    "mtp.py",
)


def test_default_policy_is_singleton_frozen_float16():
    assert isinstance(ASCEND_QWEN4EXP_DTYPE_POLICY, Qwen4ExpDtypePolicy)
    # PRD R4 / §5.3: 310P main compute is pinned to float16.
    assert ASCEND_QWEN4EXP_DTYPE_POLICY.main_dtype is torch.float16
    # fp32 accumulation is the documented CANN assumption.
    assert ASCEND_QWEN4EXP_DTYPE_POLICY.accumulation_dtype is torch.float32
    # Frozen dataclass: attributes may not be reassigned.
    with pytest.raises((AttributeError, TypeError)):
        ASCEND_QWEN4EXP_DTYPE_POLICY.main_dtype = torch.bfloat16


def test_policy_exposes_every_required_entry():
    policy = ASCEND_QWEN4EXP_DTYPE_POLICY
    required = {
        "main_dtype",
        "accumulation_dtype",
        "embedding_dtype",
        "lm_head_dtype",
        "logits_dtype",
        "router_dtype",
        "shared_expert_dtype",
        "expert_dtype",
        "attention_dtype",
        "attention_accumulation_dtype",
        "qsa_main_dtype",
        "qsa_indexer_dtype",
        "kv_cache_dtype",
        "mamba_ssm_cache_dtype",
        "mamba_conv_cache_dtype",
        "ple_projection_dtype",
        "ple_norm_accumulation_dtype",
        "gated_residual_dtype",
        "gated_residual_accumulation_dtype",
        "hyperconnection_params_dtype",
        "ngram_embedding_dtype",
    }
    for name in required:
        value = getattr(policy, name)
        assert isinstance(value, torch.dtype), f"{name} must be a torch.dtype"


def test_required_cast_sites_present_and_resolve():
    # The cast sites named by the PRD must be addressable through the policy.
    for site in (
        "attention",
        "router",
        "shared_expert",
        "ple_projection",
        "gated_residual",
        "lm_head",
    ):
        assert site in REQUIRED_CAST_SITES
        assert isinstance(ASCEND_QWEN4EXP_DTYPE_POLICY.cast_site(site), torch.dtype)


def test_accumulation_sites_are_fp32():
    policy = ASCEND_QWEN4EXP_DTYPE_POLICY
    # Router and every *_accumulation site accumulate in fp32 on 310P.
    assert policy.router_dtype is torch.float32
    assert policy.accumulation_dtype is torch.float32
    assert policy.attention_accumulation_dtype is torch.float32
    assert policy.ple_norm_accumulation_dtype is torch.float32
    assert policy.gated_residual_accumulation_dtype is torch.float32
    assert policy.logits_dtype is torch.float32


def test_eager_linear_keeps_npu_weights_in_storage_dtype():
    from vllm_ascend.models.qwen4_exp.model import _linear_operand_dtype

    assert _linear_operand_dtype("npu", torch.float16, torch.float32) is torch.float16
    assert _linear_operand_dtype("cpu", torch.float16, torch.float32) is torch.float32
    assert _linear_operand_dtype("cpu", torch.float64, torch.float64) is torch.float64


def test_ssm_cache_dtype_is_fp32_and_main_caches_fp16():
    policy = ASCEND_QWEN4EXP_DTYPE_POLICY
    # SSM recurrent state must stay fp32 for numerical stability at 1M ctx.
    assert policy.mamba_ssm_cache_dtype is torch.float32
    # KV / conv / QSA caches ride the float16 main dtype.
    assert policy.kv_cache_dtype is torch.float16
    assert policy.mamba_conv_cache_dtype is torch.float16
    assert policy.qsa_main_dtype is torch.float16
    assert policy.qsa_indexer_dtype is torch.float16


def test_for_310p_matches_default_singleton():
    assert Qwen4ExpDtypePolicy.for_310p() == ASCEND_QWEN4EXP_DTYPE_POLICY


def test_layer_modules_have_no_bare_dtype_literals():
    """Enforce: only dtype_policy.py may spell float16/bfloat16 literals."""
    offenders = {}
    pattern = re.compile(r"torch\.(float16|bfloat16|half)\b")
    for name in _LAYER_MODULES:
        path = _PKG_DIR / name
        assert path.exists(), f"missing layer module {name}"
        hits = []
        for lineno, line in enumerate(path.read_text().splitlines(), start=1):
            code = line.split("#", 1)[0]
            if pattern.search(code):
                hits.append((lineno, line.strip()))
        if hits:
            offenders[name] = hits
    assert not offenders, f"layer modules must read dtype from the policy, found literals: {offenders}"
