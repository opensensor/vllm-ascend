# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the GLM-5.3-Flash W2 eager assembly + dummy-weight CPU boot (G7).

Everything runs host-side with NO NPU and NO Triton. The assembly wires the REAL
G4 KDA / G5 DSA / G6 W2-MoE components into GLM's hybrid decoder stack at a tiny
config and boots it on CPU: construct -> forward -> sample. ``load_weights`` maps
a synthetic real-shaped W2 index into the E1.3 fused param banks.

Run with ``--noconftest``:

    python3 -m pytest -q --noconftest tests/ut/glm_w2/test_glm5next_w2_assembly.py
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Import-hygiene capture FIRST: importing the assembly must not pull torch_npu or
# the heavy 310P W2 method. Freeze the import delta BEFORE installing the npu
# stubs (which the E1.3 method needs only to *resolve* at forward time).
# ---------------------------------------------------------------------------
_MODULES_BEFORE = set(sys.modules)

import vllm_ascend.models.glm5next_w2.assembly as _assembly  # noqa: E402,F401

_ASSEMBLY_ADDED = frozenset(sys.modules) - _MODULES_BEFORE

_ASSEMBLY_SRC = Path(__file__).parents[3] / "vllm_ascend" / "models" / "glm5next_w2" / "assembly.py"


# ---------------------------------------------------------------------------
# torch_npu / device stubs the E1.3 method import needs (mirrors the DeepSeek
# assembly test). GLM's Glm5NextW2MoE reuses the same resolve_w2_moe_method +
# AscendW2DynamicFusedMoEMethod310, so the same stubs make its CPU host path run.
# ---------------------------------------------------------------------------
def _mod(name, **attrs):
    m = types.ModuleType(name)
    m.__spec__ = importlib.util.spec_from_loader(name, loader=None)
    for key, value in attrs.items():
        setattr(m, key, value)
    sys.modules[name] = m
    return m


def _pkg(name, path=None, **attrs):
    m = _mod(name, **attrs)
    m.__path__ = [path] if path else []
    return m


def _install_npu_stubs():
    if "torch_npu" not in sys.modules:
        tn = _pkg("torch_npu")
        tn.npu = MagicMock()
        tn._C = MagicMock()
        sys.modules["torch_npu._C"] = tn._C
    sys.modules.setdefault("triton.runtime", _mod("triton.runtime"))
    sys.modules.setdefault("vllm_ascend._build_info", _mod("vllm_ascend._build_info", __device_type__="A2"))

    import vllm.distributed.utils as _vllm_dist_utils

    if not hasattr(_vllm_dist_utils, "is_weak_contiguous"):
        _vllm_dist_utils.is_weak_contiguous = lambda *a, **k: True  # type: ignore[attr-defined]

    _pkg("vllm_ascend.ops")
    _pkg("vllm_ascend.ops.fused_moe")
    _pkg("vllm_ascend.ops.fused_moe.dataclass")
    _mod(
        "vllm_ascend.ops.fused_moe.dataclass.fused_experts",
        MoEWeights=type("MoEWeights", (), {}),
        build_fused_experts_input=lambda **k: MagicMock(),
    )
    _mod("vllm_ascend.ops.fused_moe.dataclass.moe_mlp", MoEMlpComputeInput=type("MoEMlpComputeInput", (), {}))
    _mod("vllm_ascend.ops.fused_moe.routed_experts", AscendRoutedExperts=type("AscendRoutedExperts", (), {}))
    _mod(
        "vllm_ascend.ops.fused_moe.moe_utils",
        maybe_normalize_mxfp_scale_layout=lambda x: x,
        cumsum_group_list=lambda *a, **k: None,
    )
    _mod(
        "vllm_ascend.ops.linear",
        AscendRowParallelLinear=type("AscendRowParallelLinear", (), {}),
        AscendUnquantizedLinearMethod=type("AscendUnquantizedLinearMethod", (), {}),
    )
    _mod("vllm_ascend.ascend_forward_context", _EXTRA_CTX=MagicMock())

    import vllm_ascend

    va_dir = os.path.dirname(vllm_ascend.__file__)
    _pkg("vllm_ascend.quantization.methods", path=os.path.join(va_dir, "quantization", "methods"))
    _pkg("vllm_ascend._310p", path=os.path.join(va_dir, "_310p"))
    _pkg("vllm_ascend._310p.quantization", path=os.path.join(va_dir, "_310p", "quantization"))


_install_npu_stubs()

import pytest  # noqa: E402
import torch  # noqa: E402

from vllm_ascend.models.glm5next_w2.assembly import (  # noqa: E402
    AscendGlm5NextW2EagerForCausalLM,
)


def _tiny_glm_config(prefix: str, *, num_hidden_layers: int, first_k_dense_replace: int) -> SimpleNamespace:
    """A tiny hybrid GLM config that constructs + forwards on CPU in milliseconds.

    KDA head geometry and DSA MLA-NoPE geometry are shrunk to small tiling-valid
    values; ``full_attn_layers`` marks the DSA layers (every 4th, like GLM).
    """
    full_attn_layers = [i for i in range(num_hidden_layers) if i % 4 == 3]
    return SimpleNamespace(
        model_type="glm5_next",
        vocab_size=128,
        hidden_size=64,
        num_hidden_layers=num_hidden_layers,
        rms_norm_eps=1e-5,
        # MoE (routed experts): 32-tiling-valid hidden/inter.
        n_routed_experts=4,
        num_experts_per_token=2,
        moe_intermediate_size=32,
        intermediate_size=32,
        n_shared_experts=1,
        first_k_dense_replace=first_k_dense_replace,
        norm_topk_prob=True,
        routed_scaling_factor=2.5,
        scoring_func="sigmoid",
        n_group=1,
        topk_group=1,
        # hybrid layer types
        full_attn_layers=full_attn_layers,
        # KDA head geometry (num_heads * head_dim need not equal hidden).
        kda_num_heads=2,
        kda_head_dim=16,
        kda_short_conv_kernel_size=4,
        # DSA MLA-NoPE geometry (small, tiling-valid).
        num_attention_heads=4,
        q_lora_rank=32,
        kv_lora_rank=16,
        qk_nope_head_dim=16,
        v_head_dim=16,
        index_n_heads=2,
        index_head_dim=16,
        index_topk=8,
        index_kpool=2,
        # mhc residual streams.
        mhc_num_residual_streams=4,
        # MTP-1.
        num_nextn_predict_layers=1,
        mtp_layer_index=num_hidden_layers,
    )


@pytest.fixture(scope="module")
def boot_config():
    # 8 layers, first 2 dense -> covers dense-MLP + MoE and KDA + DSA (layers 3,7).
    return _tiny_glm_config("glmboot", num_hidden_layers=8, first_k_dense_replace=2)


@pytest.fixture(scope="module")
def boot_model(boot_config):
    torch.manual_seed(0)
    return AscendGlm5NextW2EagerForCausalLM(config=boot_config)


def _input_ids(n: int = 6) -> torch.Tensor:
    return torch.arange(1, n + 1, dtype=torch.int64)


# ---------------------------------------------------------------------------
# Import hygiene
# ---------------------------------------------------------------------------


def test_assembly_source_has_no_triton_import():
    hits = []
    for lineno, line in enumerate(_ASSEMBLY_SRC.read_text().splitlines(), start=1):
        code = line.split("#", 1)[0]
        if "import triton" in code or "from triton" in code or "ops.triton" in code:
            hits.append((lineno, line))
    assert not hits, f"triton import reachable in assembly.py: {hits}"


def test_importing_assembly_did_not_pull_torch_npu():
    # Importing the eager assembly must not require torch_npu (frozen delta taken
    # before the stubs were installed for the forward-time method resolution).
    assert "torch_npu" not in _ASSEMBLY_ADDED
    assert "vllm_ascend._310p.quantization.methods.w2_dynamic" not in _ASSEMBLY_ADDED


# ---------------------------------------------------------------------------
# Construct / forward / sample
# ---------------------------------------------------------------------------


def test_construct_forward_and_sample(boot_model):
    ids = _input_ids()
    hidden = boot_model.forward(ids)
    assert hidden.shape == (ids.shape[0], boot_model.config.hidden_size)
    assert torch.isfinite(hidden).all()

    logits = boot_model.compute_logits(hidden)
    assert logits.shape == (ids.shape[0], boot_model.config.vocab_size)

    token = boot_model.sample_token(ids)
    assert 0 <= int(token) < boot_model.config.vocab_size


def test_forward_is_deterministic(boot_model):
    ids = _input_ids()
    first = boot_model.forward(ids)
    second = boot_model.forward(ids)
    assert torch.equal(first, second)
    assert int(boot_model.sample_token(ids)) == int(boot_model.sample_token(ids))


def test_hybrid_layer_kinds_present(boot_model):
    # 8 layers: DSA at 3 and 7; the rest KDA. MoE from layer 2 on (first 2 dense).
    assert boot_model.model.dsa_layers == (3, 7)
    assert set(boot_model.model.kda_layers) == {0, 1, 2, 4, 5, 6}
    assert boot_model.model.moe_layers == (2, 3, 4, 5, 6, 7)


def test_mtp_draft_runs(boot_model):
    assert boot_model.num_nextn_predict_layers == 1
    token = boot_model.mtp_draft(_input_ids())
    assert 0 <= int(token) < boot_model.config.vocab_size


def test_kv_group_report_hybrid(boot_model):
    rep = boot_model.kv_group_report()
    assert rep["num_dsa_layers"] == 2
    assert rep["num_kda_layers"] == 6
    assert rep["dsa_layers"] == [3, 7]
    assert "mla_latent" in rep["dsa_spec"]
    assert "recurrent" in rep["kda_spec"]


# ---------------------------------------------------------------------------
# load_weights: synthetic real-shaped W2 index -> fused banks
# ---------------------------------------------------------------------------


def _synthetic_w2_weights(config):
    """Yield (name, tensor) for one MoE block's experts + a few FP16 + a vision tensor."""
    from vllm_ascend.models.glm5next_w2.weight_mapping import expected_expert_shape

    hidden = config.hidden_size
    inter = config.moe_intermediate_size
    n_exp = config.n_routed_experts
    layer = config.first_k_dense_replace  # first MoE layer
    L = f"model.language_model.layers.{layer}.mlp.experts"
    slot_for = {"gate_proj": "w1", "up_proj": "w3", "down_proj": "w2"}
    for e in range(n_exp):
        for proj, slot in slot_for.items():
            for kind, dtype in (("codes", torch.uint8), ("scale", torch.float32)):
                shape = expected_expert_shape(slot, kind, {"hidden_size": hidden, "moe_intermediate_size": inter})
                t = (torch.randint(0, 256, shape, dtype=torch.uint8) if kind == "codes"
                     else torch.randn(shape, dtype=torch.float32))
                yield f"{L}.{e}.{proj}_{kind}", t.to(dtype)
    # non-expert FP16 + a vision tensor (excluded).
    yield f"model.language_model.layers.{layer}.mlp.gate.weight", torch.randn(n_exp, hidden, dtype=torch.float16)
    yield "model.language_model.layers.0.mlp.down_proj.weight", torch.randn(hidden, inter, dtype=torch.float16)
    yield "model.visual.blocks.0.attn.qkv.weight", torch.randn(8, 8, dtype=torch.float16)


def test_load_weights_places_experts_and_accounts_lanes(boot_config):
    model = AscendGlm5NextW2EagerForCausalLM(config=boot_config)
    report = model.load_weights(_synthetic_w2_weights(boot_config))
    n_exp = boot_config.n_routed_experts
    assert report["w2_expert"] == n_exp * 6  # gate/up/down x codes/scale
    assert report["fp16"] == 2
    assert report["exclude"] == 1
    assert report["expert_blocks"] == 1
    # The fused bank for the loaded block has the fused w13 (2*inter rows) filled.
    block = f"layers.{boot_config.first_k_dense_replace}"
    bank = model._expert_param_banks[block]
    assert bank["w13_codes"].shape == (n_exp, 2 * boot_config.moe_intermediate_size, boot_config.hidden_size // 4)
    assert bank["w2_codes"].shape == (n_exp, boot_config.hidden_size, boot_config.moe_intermediate_size // 4)


def test_load_weights_rejects_wrong_shape(boot_config):
    model = AscendGlm5NextW2EagerForCausalLM(config=boot_config)
    bad = [("model.language_model.layers.2.mlp.experts.0.gate_proj_codes", torch.zeros(8, 8, dtype=torch.uint8))]
    with pytest.raises(ValueError):
        model.load_weights(bad)
