# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the Ascend DeepSeek V4.1 package + registration (plan E2.1).

Everything runs host-side with NO NPU. The E2.1 model classes SUBCLASS the
shipped ``deepseek_v4`` model, whose top-level import pulls heavy vLLM
machinery (``FusedMoEFactory``) that is unavailable on this CPU host and also
pulls ``muls_add_triton`` (Triton). So -- exactly like the Qwen4Exp case
(``tests/ut/qwen38_1m/test_qwen4exp_registration.py``) -- these tests assert
**registration + dtype-policy + import-hygiene** rather than full construction.
The shipped base is resolved lazily, so importing the V4.1 package/model stays
Triton-free on the 310P path.

Run with ``--noconftest`` (the shared tests/ut/conftest.py fails to import on
this host):

    python3 -m pytest -q --noconftest \
        tests/ut/deepseek_w2/test_deepseekv41_package.py
"""

import subprocess
import sys
from pathlib import Path

import pytest
import torch

_PKG_DIR = Path(__file__).parents[3] / "vllm_ascend" / "models" / "deepseek_v41"

# arch name -> "module_path:ClassName" the Ascend registry must resolve to.
_EXPECTED_V41_REGISTRATIONS = {
    "DeepseekV41ForCausalLM": "vllm_ascend.models.deepseek_v41.model:AscendDeepseekV41ForCausalLM",
    "DeepseekV41ForConditionalGeneration": (
        "vllm_ascend.models.deepseek_v41.model:AscendDeepseekV41ForConditionalGeneration"
    ),
    "DeepSeekV41MTPModel": "vllm_ascend.models.deepseek_v41.mtp:DeepSeekV41MTP",
}

# The shipped DeepSeek V4 registrations that MUST remain intact (additive-only).
_EXPECTED_V4_REGISTRATIONS = {
    "DeepseekV4ForCausalLM": "vllm_ascend.models.deepseek_v4.model:AscendDeepseekV4ForCausalLM",
    "DeepseekV4ForConditionalGeneration": (
        "vllm_ascend.models.deepseek_v4.vl_model:AscendDeepseekV4ForConditionalGeneration"
    ),
    "DeepSeekV4MTPModel": "vllm_ascend.models.deepseek_v4.mtp:DeepSeekV4MTP",
}


def _record_registrations() -> dict:
    from unittest.mock import patch

    from vllm_ascend.models import register_model

    recorded: dict = {}

    def _record(arch, target):
        recorded[arch] = target

    with patch("vllm_ascend.models.ModelRegistry.register_model", side_effect=_record):
        register_model()
    return recorded


# ---------------------------------------------------------------------------
# Registration (additive; V4 intact)
# ---------------------------------------------------------------------------


def test_v41_arch_names_register_to_ascend_classes():
    recorded = _record_registrations()
    for arch, target in _EXPECTED_V41_REGISTRATIONS.items():
        assert arch in recorded, f"{arch} was not registered"
        assert recorded[arch] == target, f"{arch} -> {recorded[arch]!r}, expected {target!r}"


def test_shipped_deepseek_v4_registrations_still_present():
    recorded = _record_registrations()
    for arch, target in _EXPECTED_V4_REGISTRATIONS.items():
        assert arch in recorded, f"shipped {arch} registration was clobbered"
        assert recorded[arch] == target, f"shipped {arch} -> {recorded[arch]!r}, expected {target!r}"


# ---------------------------------------------------------------------------
# dtype policy (authoritative table)
# ---------------------------------------------------------------------------


def test_dtype_policy_singleton_and_required_cast_sites():
    from vllm_ascend.models.deepseek_v41.dtype_policy import (
        ASCEND_DEEPSEEKV41_DTYPE_POLICY,
        REQUIRED_CAST_SITES,
        DeepseekV41DtypePolicy,
    )

    policy = ASCEND_DEEPSEEKV41_DTYPE_POLICY
    assert isinstance(policy, DeepseekV41DtypePolicy)

    # Every declared cast site resolves to a real torch.dtype via cast_site().
    assert REQUIRED_CAST_SITES, "cast-site table must not be empty"
    for name, attr in REQUIRED_CAST_SITES.items():
        assert hasattr(policy, attr), f"policy missing field {attr!r} for cast site {name!r}"
        assert isinstance(policy.cast_site(name), torch.dtype)

    # Unknown cast site fails loudly.
    with pytest.raises(KeyError):
        policy.cast_site("does_not_exist")


def test_dtype_policy_pins_w2_int8_w4_fp16_fp32():
    from vllm_ascend.models.deepseek_v41.dtype_policy import (
        ASCEND_DEEPSEEKV41_DTYPE_POLICY as p,
    )

    # W2 routed experts / INT8 dynamic activation.
    assert p.cast_site("expert_weight") is torch.uint8  # packed 2-bit codes
    assert p.cast_site("expert_activation") is torch.int8
    assert p.cast_site("expert_accumulation") is torch.float32
    # ~W4 Engram tables.
    assert p.cast_site("engram_table") is torch.uint8  # packed 4-bit rows
    assert p.cast_site("engram") is torch.float16
    # FP16 MLA / indexer / dense / shared / LM-head.
    for site in ("mla", "indexer", "dense", "shared_expert", "lm_head", "main"):
        assert p.cast_site(site) is torch.float16, site
    # FP32 accumulation everywhere it matters.
    for site in ("accumulation", "router", "logits", "attention_accumulation", "mla_accumulation"):
        assert p.cast_site(site) is torch.float32, site


def test_dtype_policy_from_vllm_config_is_pinned():
    from types import SimpleNamespace

    from vllm_ascend.models.deepseek_v41.dtype_policy import (
        DeepseekV41DtypePolicy,
    )

    # None config -> pinned 310P policy.
    assert DeepseekV41DtypePolicy.from_vllm_config(None).main_dtype is torch.float16
    # An explicit torch.dtype KV override is honoured; everything else pinned.
    cfg = SimpleNamespace(cache_config=SimpleNamespace(cache_dtype=torch.float32))
    derived = DeepseekV41DtypePolicy.from_vllm_config(cfg)
    assert derived.kv_cache_dtype is torch.float32
    assert derived.main_dtype is torch.float16
    assert derived.expert_activation_dtype is torch.int8


# ---------------------------------------------------------------------------
# Import isolation / Triton-free package (grep-gate)
# ---------------------------------------------------------------------------


def test_package_source_has_no_triton_import():
    pattern_hits: dict = {}
    for path in _PKG_DIR.rglob("*.py"):
        for lineno, line in enumerate(path.read_text().splitlines(), start=1):
            code = line.split("#", 1)[0]  # comments are documentation, not code
            if "import triton" in code or "from triton" in code:
                pattern_hits.setdefault(path.name, []).append((lineno, line))
            if "triton" in code and "import" in code:
                pattern_hits.setdefault(path.name, []).append((lineno, line))
    assert not pattern_hits, f"triton import reachable in package code: {pattern_hits}"


def test_model_import_path_does_not_pull_shipped_triton_op():
    # Fresh interpreter: importing the V4.1 package + model module must NOT
    # eagerly load the shipped ``deepseek_v4.model`` base (which top-level pulls
    # ``muls_add_triton``). The base is resolved lazily; the shipped MoE Triton
    # op (``vllm_ascend.ops.triton.mul_add``) is the E3.3 swap target and must
    # stay out of this package's import path on the 310P flag.
    #
    # NB: ``import vllm_ascend`` itself loads Triton globally via the plugin's
    # top-level init, so a bare ``'triton' in sys.modules`` check is not a valid
    # signal here (mirrors the Qwen4Exp test, which greps source instead). The
    # honest gate is that *this package's* import path pulls neither the shipped
    # base module nor its Triton MoE op.
    code = (
        "import os, sys\n"
        "os.environ['VLLM_ASCEND_ENABLE_310P'] = '1'\n"
        "import vllm_ascend.models.deepseek_v41 as pkg\n"
        "import vllm_ascend.models.deepseek_v41.model as m\n"
        "assert hasattr(pkg, 'ASCEND_DEEPSEEKV41_DTYPE_POLICY')\n"
        "assert hasattr(pkg, 'w2_active_moe_forward')\n"  # E1.2 export intact
        "assert hasattr(m, 'DeepSeekV41MTP')\n"
        "assert 'vllm_ascend.models.deepseek_v4.model' not in sys.modules, "
        "'shipped V4 base was eagerly imported (should be lazy)'\n"
        "assert 'vllm_ascend.ops.triton.mul_add' not in sys.modules, "
        "'shipped muls_add_triton op pulled into deepseek_v41 import path'\n"
        "print('OK')\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, f"subprocess failed:\nSTDOUT:{proc.stdout}\nSTDERR:{proc.stderr}"
    assert "OK" in proc.stdout


# ---------------------------------------------------------------------------
# E1.2 exports preserved / lazy model surface
# ---------------------------------------------------------------------------


def test_e12_w2_unpack_exports_still_importable():
    from vllm_ascend.models.deepseek_v41 import (  # noqa: F401
        ActiveExpertWeights,
        route_topk_w2,
        swiglu_gate_up,
        unpack_active_experts,
        w2_active_moe_forward,
        w2_group_qdq_linear,
    )


def test_model_module_exposes_lazy_classes_and_mtp_stub():
    import vllm_ascend.models.deepseek_v41.model as m

    # The heavy subclasses are lazy (PEP 562) but discoverable via dir().
    exported = set(dir(m))
    assert "AscendDeepseekV41ForCausalLM" in exported
    assert "AscendDeepseekV41ForConditionalGeneration" in exported

    # MTP-3 stub is a registration target that fails fast if constructed.
    with pytest.raises(NotImplementedError):
        m.DeepSeekV41MTP()
    assert m.DeepSeekV41MTP.num_nextn_predict_layers == 3
    assert m.DeepSeekV41MTP.dspark_target_layer_ids == (37, 38, 39)


def test_reject_multimodal_first_gate():
    from types import SimpleNamespace

    import vllm_ascend.models.deepseek_v41.model as m

    # Text-only config passes the gate (no exception).
    text_only = SimpleNamespace(
        model_config=SimpleNamespace(multimodal_config=None, hf_config=SimpleNamespace(vision_config=None))
    )
    m._reject_multimodal(text_only)

    # A vision_config or multimodal_config trips the first gate.
    with_vision = SimpleNamespace(
        model_config=SimpleNamespace(multimodal_config=None, hf_config=SimpleNamespace(vision_config=object()))
    )
    with pytest.raises(NotImplementedError):
        m._reject_multimodal(with_vision)

    with_mm = SimpleNamespace(
        model_config=SimpleNamespace(multimodal_config=object(), hf_config=SimpleNamespace(vision_config=None))
    )
    with pytest.raises(NotImplementedError):
        m._reject_multimodal(with_mm)


def test_v41_config_contract_constants():
    import vllm_ascend.models.deepseek_v41.model as m

    assert m.DEEPSEEKV41_NUM_HIDDEN_LAYERS == 40
    assert m.DEEPSEEKV41_N_ROUTED_EXPERTS == 384
    assert m.DEEPSEEKV41_NUM_EXPERTS_PER_TOK == 6
    assert m.DEEPSEEKV41_Q_LORA_RANK == 1280
    assert m.DEEPSEEKV41_ENGRAM_LAYER_IDS == (1, 14)
    assert m.DEEPSEEKV41_NUM_NEXTN_PREDICT_LAYERS == 3
    assert m.DEEPSEEKV41_COMPRESS_RATIOS == (0, 1, 2)
