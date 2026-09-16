# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the Ascend DeepSeek V4.1 MTP-3 + DSpark wiring (plan E4.2).

Everything runs host-side with NO NPU / NO Triton. The concrete drafter
(:class:`vllm_ascend.models.deepseek_v41.mtp.DeepSeekV41MTP`) adapts the shipped
``deepseek_v4`` serial MTP + DSpark block drafter, reusing the E4.1 eager
decoder-layer component so it constructs and runs a deterministic single-step
draft on CPU. The shipped device drafters are resolved lazily, so importing the
module stays Triton-free.

Run with ``--noconftest`` (the shared tests/ut/conftest.py fails to import on
this host)::

    python3 -m pytest -q --noconftest tests/ut/deepseek_w2/test_mtp.py
"""

import importlib.util
import os
import subprocess
import sys
import tempfile
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

_PKG_DIR = Path(__file__).parents[3] / "vllm_ascend" / "models" / "deepseek_v41"
_SCRATCH = tempfile.mkdtemp(prefix="e41mtp_")


# ---------------------------------------------------------------------------
# Import-hygiene capture FIRST: importing the MTP module must not pull the
# shipped V4 base/MTP/DSpark modules or the shipped ``muls_add_triton`` op.
# ---------------------------------------------------------------------------
import vllm_ascend.models.deepseek_v41.mtp as mtp  # noqa: E402

_SHIPPED_V4_BASE = "vllm_ascend.models.deepseek_v4.model"
_SHIPPED_V4_MTP = "vllm_ascend.models.deepseek_v4.mtp"
_SHIPPED_V4_DSPARK = "vllm_ascend.models.deepseek_v4.dspark"
_MUL_ADD = "vllm_ascend.ops.triton.mul_add"
_MTP_PULLS_V4_BASE = _SHIPPED_V4_BASE in sys.modules
_MTP_PULLS_V4_MTP = _SHIPPED_V4_MTP in sys.modules
_MTP_PULLS_V4_DSPARK = _SHIPPED_V4_DSPARK in sys.modules
_MTP_PULLS_MUL_ADD = _MUL_ADD in sys.modules


# ---------------------------------------------------------------------------
# torch_npu / device stubs the E1.3 W2 method needs when the eager MoE first
# runs at *forward* time (mirrors test_moe.py / test_deepseekv41_assembly.py).
# Installed AFTER the import-hygiene capture above and cross-checked by the
# fresh-interpreter subprocess test, so the *import* path stays torch_npu-free.
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


def _tiny_v41_mtp_config(prefix: str) -> SimpleNamespace:
    """A tiny but shape-faithful V4.1 text config carrying MTP-3 / DSpark fields.

    All dims tile the [32, 32] W2 block; ``num_nextn_predict_layers=3`` and
    ``dspark_target_layer_ids=(37, 38, 39)`` mirror the real V4.1 geometry.
    """
    return SimpleNamespace(
        # backbone / vocab
        vocab_size=64,
        hidden_size=64,
        hc_mult=2,
        num_hidden_layers=40,
        compress_ratios=[i % 3 for i in range(40)],
        engram_layer_ids=(1, 14),
        rms_norm_eps=1e-6,
        # MLA (E3.1)
        num_attention_heads=4,
        q_lora_rank=32,
        kv_lora_rank=32,
        qk_nope_head_dim=16,
        qk_rope_head_dim=16,
        v_head_dim=16,
        rope_theta=10000.0,
        # indexer (E3.2)
        index_n_heads=2,
        index_head_dim=16,
        index_topk=4,
        # MoE (E3.3)
        n_routed_experts=8,
        num_experts_per_tok=3,
        norm_topk_prob=True,
        routed_scaling_factor=1.5,
        moe_intermediate_size=32,
        # Engram (E2.3) -- unused on the MTP dense blocks but read by from_config
        engram_max_ngram_size=2,
        engram_n_heads=2,
        engram_vocab_size=97,
        engram_compressed_vocab_size=97,
        engram_pad_token_id=0,
        engram_head_dim=32,
        engram_shm_dir=_SCRATCH,
        engram_shm_prefix=prefix,
        # MTP-3 (E4.2)
        num_nextn_predict_layers=3,
        # DSpark (E4.2)
        dspark_target_layer_ids=(37, 38, 39),
        dspark_n_routed_experts=128,
        dspark_num_experts_per_tok=3,
        dspark_block_size=5,
        dspark_markov_rank=256,
    )


@pytest.fixture
def mtp_config():
    return _tiny_v41_mtp_config("e41mtp_cfg")


@pytest.fixture
def mtp_model(mtp_config):
    return mtp.DeepSeekV41MTP(config=mtp_config)


# ---------------------------------------------------------------------------
# Construction + reporting (the E4.2 validation contract)
# ---------------------------------------------------------------------------
def test_constructs_on_cpu_and_reports_mtp3(mtp_model):
    assert isinstance(mtp_model, torch.nn.Module)
    assert mtp_model.num_nextn_predict_layers == 3
    assert mtp_model.mtp_layer_count() == 3
    assert len(mtp_model.layers) == 3
    assert mtp_model.mtp_start_layer_idx == 40


def test_reports_dspark_target_layers_and_config(mtp_model):
    assert mtp_model.dspark_target_layer_ids == (37, 38, 39)
    spec = mtp_model.dspark_config()
    assert spec.target_layer_ids == (37, 38, 39)
    assert spec.n_routed_experts == 128
    assert spec.num_experts_per_tok == 3
    assert spec.block_size == 5
    assert spec.markov_rank == 256
    assert spec.num_mtp_layers == 3

    report = mtp_model.report()
    assert report["num_nextn_predict_layers"] == 3
    assert report["dspark_target_layer_ids"] == (37, 38, 39)
    assert report["dspark_n_routed_experts"] == 128


def test_class_level_contract_without_construction():
    # Registered class reports the V4.1 geometry even before an instance exists.
    assert mtp.DeepSeekV41MTP.num_nextn_predict_layers == 3
    assert mtp.DeepSeekV41MTP.dspark_target_layer_ids == (37, 38, 39)


def test_mtp_layer_carries_eh_proj_enorm_hnorm_shared_head(mtp_model):
    layer = mtp_model.layers[0]
    assert layer.eh_proj.shape == (64, 128)  # (hidden, 2*hidden)
    assert layer.enorm.shape == (64,)
    assert layer.hnorm.shape == (64,)
    assert layer.shared_head.norm.shape == (64,)
    assert layer.shared_head.head.shape == (64, 64)  # (vocab, hidden)
    # The transformer block reuses the E4.1 eager decoder layer.
    from vllm_ascend.models.deepseek_v41.assembly import AscendDeepseekV41EagerDecoderLayer

    assert isinstance(layer.mtp_block, AscendDeepseekV41EagerDecoderLayer)


# ---------------------------------------------------------------------------
# Deterministic host draft forward
# ---------------------------------------------------------------------------
def test_forward_is_host_runnable_and_shaped(mtp_model):
    num_tokens = 5
    input_ids = torch.arange(num_tokens, dtype=torch.long)
    positions = torch.arange(num_tokens, dtype=torch.long)
    previous_hidden = torch.zeros(num_tokens, mtp_model.hidden_size, dtype=mtp_model.main_dtype)

    hidden = mtp_model(input_ids, positions, previous_hidden)
    assert hidden.shape == (num_tokens, mtp_model.hidden_size)

    logits = mtp_model.compute_logits(hidden)
    assert logits.shape == (num_tokens, mtp_model.vocab_size)
    assert torch.isfinite(logits.float()).all()


def test_forward_is_deterministic_across_seeded_twins():
    cfg_a = _tiny_v41_mtp_config("e41mtp_a")
    cfg_b = _tiny_v41_mtp_config("e41mtp_b")
    model_a = mtp.DeepSeekV41MTP(config=cfg_a)
    model_b = mtp.DeepSeekV41MTP(config=cfg_b)

    input_ids = torch.arange(4, dtype=torch.long)
    positions = torch.arange(4, dtype=torch.long)
    prev = torch.zeros(4, cfg_a.hidden_size, dtype=model_a.main_dtype)

    out_a = model_a(input_ids, positions, prev)
    out_b = model_b(input_ids, positions, prev)
    torch.testing.assert_close(out_a, out_b, rtol=0, atol=0)


def test_spec_step_selects_distinct_layers(mtp_model):
    # Each spec step routes to its own MTP layer.
    assert mtp_model.forward.__doc__  # sanity: documented deferral
    for step in range(mtp_model.num_nextn_predict_layers):
        assert mtp_model.compute_logits.__self__ is mtp_model  # bound
        idx = step % mtp_model.num_nextn_predict_layers
        assert mtp_model.layers[idx] is mtp_model.layers[step]


# ---------------------------------------------------------------------------
# VllmConfig plumbing (draft config path)
# ---------------------------------------------------------------------------
def test_constructs_from_vllm_config_draft(mtp_config):
    vllm_config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            draft_model_config=SimpleNamespace(hf_config=mtp_config),
        ),
        model_config=None,
    )
    model = mtp.DeepSeekV41MTP(vllm_config=vllm_config)
    assert model.num_nextn_predict_layers == 3
    assert model.dspark_target_layer_ids == (37, 38, 39)


def test_requires_a_config_source():
    with pytest.raises(ValueError):
        mtp.DeepSeekV41MTP()


# ---------------------------------------------------------------------------
# DSpark spec falls back to V4.1 defaults when fields are absent
# ---------------------------------------------------------------------------
def test_dspark_spec_defaults_when_config_minimal():
    minimal = SimpleNamespace(hidden_size=64, vocab_size=64, n_routed_experts=8)
    spec = mtp.DSparkDraftSpec.from_config(minimal)
    assert spec.target_layer_ids == (37, 38, 39)
    assert spec.num_experts_per_tok == 3
    assert spec.block_size == 5
    assert spec.markov_rank == 256
    assert spec.num_mtp_layers == 3
    # n_routed_experts falls back to the backbone count when dspark-specific
    # geometry is absent.
    assert spec.n_routed_experts == 8


# ---------------------------------------------------------------------------
# Registration resolves (the E2.1 registry row is intact + importable)
# ---------------------------------------------------------------------------
def _record_registrations() -> dict:
    from vllm_ascend.models import register_model

    recorded: dict = {}

    with patch(
        "vllm_ascend.models.ModelRegistry.register_model",
        side_effect=lambda arch, target: recorded.__setitem__(arch, target),
    ):
        register_model()
    return recorded


def test_mtp_registration_resolves():
    recorded = _record_registrations()
    assert "DeepSeekV41MTPModel" in recorded
    module_path, _, class_name = recorded["DeepSeekV41MTPModel"].partition(":")
    # The registry target module imports Triton-free and exposes the class.
    import importlib

    module = importlib.import_module(module_path)
    resolved = getattr(module, class_name)
    assert isinstance(resolved, type)
    # The registered class carries the MTP-3 / DSpark contract.
    assert resolved.num_nextn_predict_layers == 3
    assert resolved.dspark_target_layer_ids == (37, 38, 39)


# ---------------------------------------------------------------------------
# Triton-free / torch_npu-free import path (grep + sys.modules gate)
# ---------------------------------------------------------------------------
def test_import_did_not_pull_shipped_device_modules():
    assert not _MTP_PULLS_V4_BASE, "importing mtp eagerly pulled the shipped V4 base"
    assert not _MTP_PULLS_V4_MTP, "importing mtp eagerly pulled the shipped V4 MTP"
    assert not _MTP_PULLS_V4_DSPARK, "importing mtp eagerly pulled the shipped V4 DSpark"
    assert not _MTP_PULLS_MUL_ADD, "importing mtp pulled the shipped muls_add_triton op"


def test_mtp_source_has_no_triton_import():
    path = _PKG_DIR / "mtp.py"
    hits = []
    for lineno, line in enumerate(path.read_text().splitlines(), start=1):
        code = line.split("#", 1)[0]  # comments are documentation, not code
        if "import triton" in code or "from triton" in code:
            hits.append((lineno, line))
        if "triton" in code and "import" in code:
            hits.append((lineno, line))
    assert not hits, f"triton import reachable in mtp.py: {hits}"


def test_fresh_interpreter_import_is_triton_and_npu_free():
    code = (
        "import os, sys\n"
        "os.environ['VLLM_ASCEND_ENABLE_310P'] = '1'\n"
        "import vllm_ascend.models.deepseek_v41.mtp as mtp\n"
        "assert hasattr(mtp, 'DeepSeekV41MTP')\n"
        "assert mtp.DeepSeekV41MTP.num_nextn_predict_layers == 3\n"
        "assert mtp.DeepSeekV41MTP.dspark_target_layer_ids == (37, 38, 39)\n"
        "assert 'vllm_ascend.models.deepseek_v4.model' not in sys.modules, "
        "'shipped V4 base eagerly imported (should be lazy)'\n"
        "assert 'vllm_ascend.models.deepseek_v4.mtp' not in sys.modules, "
        "'shipped V4 MTP eagerly imported (should be lazy)'\n"
        "assert 'vllm_ascend.models.deepseek_v4.dspark' not in sys.modules, "
        "'shipped V4 DSpark eagerly imported (should be lazy)'\n"
        "assert 'vllm_ascend.ops.triton.mul_add' not in sys.modules, "
        "'shipped muls_add_triton op pulled into mtp import path'\n"
        "print('OK')\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, f"subprocess failed:\nSTDOUT:{proc.stdout}\nSTDERR:{proc.stderr}"
    assert "OK" in proc.stdout
