# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the Ascend GLM-5.3-Flash W2 package + registration (plan G3).

Everything runs host-side with NO NPU and NO Triton on the import path. The G3
model classes SUBCLASS the shipped ``glm5next`` model, whose top-level import
pulls heavy vLLM machinery (``FusedMoEFactory``) unavailable on this CPU host
and also pulls the Triton KDA op (``vllm_ascend.ops.triton.kda.kda`` via
``glm5next.kda``). So -- exactly like the DeepSeek V4.1 case
(``tests/ut/deepseek_w2/test_deepseekv41_package.py``) -- these tests assert
**registration + dtype-policy + import-hygiene** rather than full construction.
The shipped base is resolved lazily, so importing the W2 package/model stays
Triton-free on the 310P path.

Run with ``--noconftest`` (the shared tests/ut/conftest.py fails to import on
this host):

    python3 -m pytest -q --noconftest \
        tests/ut/glm_w2/test_glm5next_w2_package.py
"""

import subprocess
import sys
from pathlib import Path

import pytest
import torch

_PKG_DIR = Path(__file__).parents[3] / "vllm_ascend" / "models" / "glm5next_w2"

# arch name -> "module_path:ClassName" the Ascend registry must resolve to.
_EXPECTED_W2_REGISTRATIONS = {
    "Glm5NextW2ForCausalLM": "vllm_ascend.models.glm5next_w2.model:AscendGlm5NextW2ForCausalLM",
    "Glm5NextW2ForConditionalGeneration": (
        "vllm_ascend.models.glm5next_w2.model:AscendGlm5NextW2ForConditionalGeneration"
    ),
    "Glm5NextW2MTPModel": "vllm_ascend.models.glm5next_w2.model:Glm5NextW2MTP",
}

# The shipped glm5next registrations that MUST remain intact (additive-only).
_EXPECTED_SHIPPED_REGISTRATIONS = {
    "Glm5NextForCausalLM": "vllm_ascend.models.glm5next.model:Glm5NextForCausalLM",
    "Glm5NextForConditionalGeneration": "vllm_ascend.models.glm5next.model:Glm5NextForConditionalGeneration",
    "Glm5NextMTPModel": "vllm_ascend.models.glm5next.mtp:Glm5NextMTP",
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
# Registration (additive; shipped glm5next intact)
# ---------------------------------------------------------------------------


def test_w2_arch_names_register_to_ascend_classes():
    recorded = _record_registrations()
    for arch, target in _EXPECTED_W2_REGISTRATIONS.items():
        assert arch in recorded, f"{arch} was not registered"
        assert recorded[arch] == target, f"{arch} -> {recorded[arch]!r}, expected {target!r}"


def test_shipped_glm5next_registrations_still_present():
    recorded = _record_registrations()
    for arch, target in _EXPECTED_SHIPPED_REGISTRATIONS.items():
        assert arch in recorded, f"shipped {arch} registration was clobbered"
        assert recorded[arch] == target, f"shipped {arch} -> {recorded[arch]!r}, expected {target!r}"


# ---------------------------------------------------------------------------
# dtype policy (authoritative table)
# ---------------------------------------------------------------------------


def test_dtype_policy_singleton_and_required_cast_sites():
    from vllm_ascend.models.glm5next_w2.dtype_policy import (
        ASCEND_GLM5NEXT_W2_DTYPE_POLICY,
        REQUIRED_CAST_SITES,
        Glm5NextW2DtypePolicy,
    )

    policy = ASCEND_GLM5NEXT_W2_DTYPE_POLICY
    assert isinstance(policy, Glm5NextW2DtypePolicy)

    # Every declared cast site resolves to a real torch.dtype via cast_site().
    assert REQUIRED_CAST_SITES, "cast-site table must not be empty"
    for name, attr in REQUIRED_CAST_SITES.items():
        assert hasattr(policy, attr), f"policy missing field {attr!r} for cast site {name!r}"
        assert isinstance(policy.cast_site(name), torch.dtype)

    # Unknown cast site fails loudly.
    with pytest.raises(KeyError):
        policy.cast_site("does_not_exist")


def test_dtype_policy_has_no_engram_sites():
    # GLM-5.3-Flash has NO Engram n-gram tables (unlike DeepSeek V4.1).
    from vllm_ascend.models.glm5next_w2.dtype_policy import REQUIRED_CAST_SITES

    for name in REQUIRED_CAST_SITES:
        assert "engram" not in name, f"unexpected Engram cast site {name!r} in GLM W2 policy"


def test_dtype_policy_pins_w2_int8_fp16_fp32():
    from vllm_ascend.models.glm5next_w2.dtype_policy import (
        ASCEND_GLM5NEXT_W2_DTYPE_POLICY as p,
    )

    # W2 routed experts / INT8 dynamic activation / FP32 accumulation.
    assert p.cast_site("expert_weight") is torch.uint8  # packed 2-bit codes
    assert p.cast_site("expert_activation") is torch.int8
    assert p.cast_site("expert_accumulation") is torch.float32
    # FP16 for the GLM compute sites (KDA linear-attn, DSA sparse-attn, dense
    # MLP, shared expert, LM head, main).
    for site in ("kda", "dsa", "dense", "shared_expert", "lm_head", "main"):
        assert p.cast_site(site) is torch.float16, site
    # FP32 accumulation everywhere it matters.
    for site in ("accumulation", "router", "logits"):
        assert p.cast_site(site) is torch.float32, site


def test_dtype_policy_from_vllm_config_is_pinned():
    from types import SimpleNamespace

    from vllm_ascend.models.glm5next_w2.dtype_policy import Glm5NextW2DtypePolicy

    # None config -> pinned 310P policy (float16 main).
    assert Glm5NextW2DtypePolicy.from_vllm_config(None).main_dtype is torch.float16
    # An explicit torch.dtype KV override is honoured; everything else pinned.
    cfg = SimpleNamespace(cache_config=SimpleNamespace(cache_dtype=torch.float32))
    derived = Glm5NextW2DtypePolicy.from_vllm_config(cfg)
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


def test_model_import_path_does_not_pull_shipped_triton_kda():
    # Fresh interpreter: importing the W2 package + model module must NOT
    # eagerly load the shipped ``glm5next.model`` base (which top-level pulls
    # ``FusedMoEFactory`` and, via ``glm5next.kda``, the Triton KDA op at
    # ``vllm_ascend.ops.triton.kda.kda``). The base is resolved lazily; that
    # Triton op is the G4 swap target and must stay out of this package's
    # import path on the 310P flag.
    #
    # NB: ``import vllm_ascend`` itself loads Triton globally via the plugin's
    # top-level init, so a bare ``'triton' in sys.modules`` check is not a valid
    # signal here (mirrors the DeepSeek V4.1 test, which greps source instead).
    # The honest gate is that *this package's* import path pulls neither the
    # shipped base module nor its Triton KDA op.
    code = (
        "import os, sys\n"
        "os.environ['VLLM_ASCEND_ENABLE_310P'] = '1'\n"
        "import vllm_ascend.models.glm5next_w2 as pkg\n"
        "import vllm_ascend.models.glm5next_w2.model as m\n"
        "assert hasattr(pkg, 'ASCEND_GLM5NEXT_W2_DTYPE_POLICY')\n"
        "assert hasattr(m, 'Glm5NextW2MTP')\n"
        "assert 'vllm_ascend.models.glm5next.model' not in sys.modules, "
        "'shipped glm5next base was eagerly imported (should be lazy)'\n"
        "assert 'vllm_ascend.ops.triton.kda.kda' not in sys.modules, "
        "'shipped Triton KDA op pulled into glm5next_w2 import path'\n"
        "print('OK')\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, f"subprocess failed:\nSTDOUT:{proc.stdout}\nSTDERR:{proc.stderr}"
    assert "OK" in proc.stdout


# ---------------------------------------------------------------------------
# Lazy model surface / reject-multimodal gate / config contract
# ---------------------------------------------------------------------------


def test_model_module_exposes_lazy_classes_and_mtp_stub():
    import vllm_ascend.models.glm5next_w2.model as m

    # The heavy subclasses are lazy (PEP 562) but discoverable via dir().
    exported = set(dir(m))
    assert "AscendGlm5NextW2ForCausalLM" in exported
    assert "AscendGlm5NextW2ForConditionalGeneration" in exported

    # MTP-1 stub is a registration target that fails fast if constructed.
    with pytest.raises(NotImplementedError):
        m.Glm5NextW2MTP()
    assert m.Glm5NextW2MTP.num_nextn_predict_layers == 1


def test_reject_multimodal_first_gate():
    from types import SimpleNamespace

    import vllm_ascend.models.glm5next_w2.model as m

    # Text-only config passes the gate (no exception).
    text_only = SimpleNamespace(
        model_config=SimpleNamespace(multimodal_config=None, hf_config=SimpleNamespace(vision_config=None))
    )
    m._reject_multimodal(text_only)

    # A vision_config (GLM ships ``model.visual.*``) or multimodal_config trips
    # the first gate.
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


def test_w2_config_contract_constants():
    import vllm_ascend.models.glm5next_w2.model as m

    assert m.GLM5NEXT_NUM_HIDDEN_LAYERS == 45
    assert m.N_ROUTED_EXPERTS == 288
    assert m.NUM_EXPERTS_PER_TOK == 8
    assert m.FIRST_K_DENSE_REPLACE == 3
    assert m.FULL_ATTN_LAYERS == (3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43)
    assert len(m.FULL_ATTN_LAYERS) == 11  # 11 DSA layers, the other 34 KDA
    assert m.KDA_NUM_HEADS == 64
    assert m.KDA_HEAD_DIM == 128
    assert m.ROUTED_SCALING_FACTOR == 2.5
    assert m.NUM_NEXTN_PREDICT_LAYERS == 1
