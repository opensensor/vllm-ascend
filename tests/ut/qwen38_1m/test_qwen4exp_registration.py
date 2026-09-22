# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the Ascend Qwen4Exp package + registration (plan T1.2).

Everything runs host-side with NO NPU: a faked NPU platform, mocked
tensor-parallel groups, and meta-device construction. No triton/CUDA import is
allowed from the package on the 310P path.

Run with ``--noconftest`` (the shared tests/ut/conftest.py fails to import on
this host):

    python3 -m pytest -q --noconftest \
        tests/ut/qwen38_1m/test_qwen4exp_registration.py
"""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

_PKG_DIR = Path(__file__).parents[3] / "vllm_ascend" / "models" / "qwen4_exp"

# arch name -> "module_path:ClassName" the Ascend registry must resolve to.
_EXPECTED_REGISTRATIONS = {
    "Qwen4ExpForCausalLM": ("vllm_ascend.models.qwen4_exp.model:AscendQwen4ExpForCausalLM"),
    "Qwen4ExpForConditionalGeneration": ("vllm_ascend.models.qwen4_exp.model:AscendQwen4ExpForConditionalGeneration"),
    "Qwen4ExpMTP": "vllm_ascend.models.qwen4_exp.mtp:AscendQwen4ExpMTP",
}


def _tiny_text_config():
    """A tiny random Qwen4Exp text config (duck-typed, host-safe)."""
    return SimpleNamespace(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        rms_norm_eps=1e-6,
        tie_word_embeddings=False,
        hc_count=2,
        hc_lowrank=16,
        ple_layer_ids=[1],
        ple_embed_dim=64,
        ple_conv_kernel_size=4,
        ngram_size=3,
        heads_per_ngram=2,
        layer_types=["full_attention", "linear_attention"],
    )


def _fake_vllm_config(*, multimodal: bool = False):
    text_config = _tiny_text_config()
    multimodal_config = None
    if multimodal:
        multimodal_config = SimpleNamespace(
            mm_encoder_tp_mode="weights",
            video_pruning_rate=0.0,
            get_video_pruning_spec=lambda: None,
            is_multimodal_pruning_enabled=lambda: False,
        )
    model_config = SimpleNamespace(
        hf_text_config=text_config,
        hf_config=SimpleNamespace(
            text_config=text_config,
            vision_config=(
                SimpleNamespace(
                    deepstack_visual_indexes=[],
                    out_hidden_size=text_config.hidden_size,
                )
                if multimodal
                else None
            ),
        ),
        dtype=torch.float16,
        multimodal_config=multimodal_config,
    )
    return SimpleNamespace(
        model_config=model_config,
        quant_config=None,
        cache_config=SimpleNamespace(
            mamba_cache_mode="align",
            mamba_cache_dtype="auto",
            mamba_ssm_cache_dtype="float32",
            cache_dtype="auto",
        ),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=16),
        speculative_config=None,
        parallel_config=SimpleNamespace(tensor_parallel_size=1),
    )


def _patch_single_rank_tp():
    """Patch tensor-parallel group helpers so vocab layers build on meta."""
    mod = "vllm.model_executor.layers.vocab_parallel_embedding"
    return (
        patch(f"{mod}.get_tensor_model_parallel_rank", return_value=0),
        patch(f"{mod}.get_tensor_model_parallel_world_size", return_value=1),
    )


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def test_all_three_arch_names_register_to_ascend_classes():
    from vllm_ascend.models import register_model

    recorded = {}

    def _record(arch, target):
        recorded[arch] = target

    with patch(
        "vllm_ascend.models.ModelRegistry.register_model",
        side_effect=_record,
    ):
        register_model()

    for arch, target in _EXPECTED_REGISTRATIONS.items():
        assert arch in recorded, f"{arch} was not registered"
        assert recorded[arch] == target, f"{arch} -> {recorded[arch]!r}, expected {target!r}"


# ---------------------------------------------------------------------------
# Import isolation / triton-free package
# ---------------------------------------------------------------------------


def test_package_source_has_no_triton_import():
    pattern_hits = {}
    for path in _PKG_DIR.rglob("*.py"):
        for lineno, line in enumerate(path.read_text().splitlines(), start=1):
            code = line.split("#", 1)[0]
            if "import triton" in code or "from triton" in code:
                pattern_hits.setdefault(path.name, []).append((lineno, line))
            if "triton" in code and "import" in code:
                pattern_hits.setdefault(path.name, []).append((lineno, line))
    assert not pattern_hits, f"triton import reachable in package: {pattern_hits}"


def test_package_imports_without_triton_on_310p_flag(monkeypatch):
    # Simulate the 310P flag path and a faked NPU platform.
    monkeypatch.setenv("VLLM_ASCEND_ENABLE_310P", "1")
    import vllm_ascend.models.qwen4_exp as pkg

    assert hasattr(pkg, "AscendQwen4ExpForCausalLM")
    assert hasattr(pkg, "ASCEND_QWEN4EXP_DTYPE_POLICY")


# ---------------------------------------------------------------------------
# Meta-device construction
# ---------------------------------------------------------------------------


def test_causal_lm_constructs_on_meta_device():
    from vllm_ascend.models.qwen4_exp.model import AscendQwen4ExpForCausalLM

    vllm_config = _fake_vllm_config(multimodal=False)
    p_rank, p_ws = _patch_single_rank_tp()
    with p_rank, p_ws, torch.device("meta"):
        model = AscendQwen4ExpForCausalLM(vllm_config=vllm_config)

    # Authoritative dtype policy is attached and float16-pinned.
    assert model.dtype_policy.main_dtype is torch.float16
    # ParallelLMHead + embedding tie surfaces exist.
    assert hasattr(model, "lm_head")
    assert hasattr(model.model, "embed_tokens")
    # State-cls hook exists (T-later fills the concrete state class).
    assert callable(model.get_model_state_cls)
    # load_weights fused-expert mapping hook exists.
    assert hasattr(model, "get_expert_mapping")
    # Conditional-generation wrappers and pipeline plumbing require this hook.
    intermediate = model.make_empty_intermediate_tensors(2, torch.float16, torch.device("cpu"))
    assert intermediate["hidden_states"].shape == (2, 128)
    assert model.uses_model_owned_mrope is True


def test_qwen4exp_decoder_layers_use_configured_offloader():
    from vllm_ascend.models.qwen4_exp.model import AscendQwen4ExpForCausalLM

    vllm_config = _fake_vllm_config(multimodal=False)
    fake_offloader = MagicMock()
    fake_offloader.wrap_modules.side_effect = lambda modules, prefix="": list(modules)
    p_rank, p_ws = _patch_single_rank_tp()
    with (
        p_rank,
        p_ws,
        patch(
            "vllm_ascend.models.qwen4_exp.model.get_offloader",
            return_value=fake_offloader,
        ),
        torch.device("meta"),
    ):
        model = AscendQwen4ExpForCausalLM(vllm_config=vllm_config)

    fake_offloader.wrap_modules.assert_called_once()
    assert fake_offloader.wrap_modules.call_args.kwargs == {}
    assert len(model.model.layers) == vllm_config.model_config.hf_text_config.num_hidden_layers


def test_conditional_generation_builds_vision_and_qwen4exp_backbone():
    from vllm_ascend.models.qwen4_exp.model import (
        AscendQwen4ExpForConditionalGeneration,
    )

    vllm_config = _fake_vllm_config(multimodal=True)
    fake_visual = torch.nn.Identity()
    fake_language_model = torch.nn.Identity()
    fake_language_model.make_empty_intermediate_tensors = MagicMock()
    with (
        patch(
            "vllm_ascend.models.qwen4_exp.model.cached_tokenizer_from_config",
            return_value=MagicMock(),
        ),
        patch(
            "vllm_ascend.models.qwen4_exp.model.Qwen3_VisionTransformer",
            return_value=fake_visual,
        ),
        patch(
            "vllm_ascend.models.qwen4_exp.model.AscendQwen4ExpForCausalLM",
            return_value=fake_language_model,
        ),
        patch.object(
            AscendQwen4ExpForConditionalGeneration,
            "_mark_tower_model",
            return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock(return_value=False)),
        ),
        patch.object(
            AscendQwen4ExpForConditionalGeneration,
            "_mark_language_model",
            return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock(return_value=False)),
        ),
    ):
        model = AscendQwen4ExpForConditionalGeneration(vllm_config=vllm_config)

    assert model.visual is fake_visual
    assert model.language_model is fake_language_model
    assert model.use_deepstack is False
    assert model.uses_model_owned_mrope is True


def test_conditional_generation_exposes_qwen3vl_placeholders():
    from vllm_ascend.models.qwen4_exp.model import (
        AscendQwen4ExpForConditionalGeneration,
    )

    assert (
        AscendQwen4ExpForConditionalGeneration.get_placeholder_str("image", 0)
        == "<|vision_start|><|image_pad|><|vision_end|>"
    )
    assert (
        AscendQwen4ExpForConditionalGeneration.get_placeholder_str("video", 0)
        == "<|vision_start|><|video_pad|><|vision_end|>"
    )
