# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the DeepSeek V4.1 eager assembly + dummy-weight CPU boot (E4.1).

The shipped ``deepseek_v4`` base that the registered
``AscendDeepseekV41ForCausalLM`` subclasses is ``torch_npu`` / Triton bound and
is NOT importable on this CPU host, so E4.1 ships a *parallel, host-runnable
eager assembly* (:mod:`vllm_ascend.models.deepseek_v41.assembly`) that ties the
six E-wave components (MLA / indexer / W2-MoE / Engram / KV / weight-map) into a
40-layer-shaped decoder stack which constructs + forwards on CPU with dummy
weights. These tests validate:

* construction + forward + greedy token sample on a tiny V4.1 config;
* determinism (two fixed-input forwards are bit-identical);
* the KV-group report matches the E2.2 ``kv`` plan;
* Engram is injected at ``engram_layer_ids=[1, 14]`` and the indexer runs on
  ratio-{1, 2} layers;
* import hygiene: the assembly import path pulls no ``torch_npu`` / shipped V4
  base / ``muls_add_triton`` on the 310P path;
* ``load_weights`` maps a synthetic real-shaped W2 index into the E1.3 fused
  layout (experts), records Engram -> host, loads FP16 by name, and fires the
  E3.4 rejection classes on a malformed artifact.

Like ``test_moe.py`` / ``test_w2_method.py``, the shared UT conftest cannot
import here and the E1.3 W2 method (resolved lazily when the MoE first runs)
needs the ``torch_npu`` device stubs, so this file installs them and MUST run
with ``--noconftest``::

    python3 -m pytest -q --noconftest tests/ut/deepseek_w2/test_deepseekv41_assembly.py
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import tempfile
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Import-hygiene capture FIRST: importing the assembly must not pull the shipped
# ``muls_add_triton`` op, the shipped V4 base, or the heavy 310P W2 method.
# Capture before the torch_npu bootstrap (only needed to *resolve* the E1.3
# method at forward time).
# ---------------------------------------------------------------------------
import vllm_ascend.models.deepseek_v41.assembly as assembly  # noqa: E402

_SHIPPED_BASE = "vllm_ascend.models.deepseek_v4.model"
_MUL_ADD = "vllm_ascend.ops.triton.mul_add"
_W2_DYNAMIC = "vllm_ascend._310p.quantization.methods.w2_dynamic"
_ASSEMBLY_PULLS_SHIPPED_BASE = _SHIPPED_BASE in sys.modules
_ASSEMBLY_PULLS_MUL_ADD = _MUL_ADD in sys.modules
_ASSEMBLY_PULLS_W2_DYNAMIC = _W2_DYNAMIC in sys.modules


# ---------------------------------------------------------------------------
# torch_npu / device stubs the E1.3 method import needs (mirrors test_moe.py).
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

from vllm_ascend.models.deepseek_v41.assembly import (  # noqa: E402
    AscendDeepseekV41EagerForCausalLM,
)
from vllm_ascend.models.deepseek_v41.kv import (  # noqa: E402
    build_deepseekv41_kv_cache_groups,
    build_deepseekv41_layer_plan,
)
from vllm_ascend.models.deepseek_v41.weight_mapping import (  # noqa: E402
    DtypeMismatchError,
    DuplicateTensorError,
    ExtraTensorError,
    MissingTensorError,
    ShapeMismatchError,
    expected_expert_blocks,
    expected_expert_dtype,
    expected_expert_shape,
)

# Host-only shared Engram tables are mmap-backed; keep the .bin files out of the
# repo by defaulting to a temp dir (override via PYTEST_DEEPSEEKV41_SHM).
_SCRATCH = os.environ.get("PYTEST_DEEPSEEKV41_SHM") or tempfile.mkdtemp(prefix="deepseekv41_asm_")


def _tiny_v41_config(
    prefix: str, *, num_hidden_layers: int, engram_layer_ids, n_routed_experts: int
) -> SimpleNamespace:
    """A tiny but shape-faithful V4.1 text config (all dims tile the [32,32] W2 block)."""
    return SimpleNamespace(
        # backbone / vocab
        vocab_size=64,
        hidden_size=64,
        hc_mult=2,
        num_hidden_layers=num_hidden_layers,
        compress_ratios=[i % 3 for i in range(num_hidden_layers)],
        engram_layer_ids=engram_layer_ids,
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
        n_routed_experts=n_routed_experts,
        num_experts_per_tok=3,
        norm_topk_prob=True,
        routed_scaling_factor=1.5,
        moe_intermediate_size=32,
        # Engram (E2.3)
        engram_max_ngram_size=2,
        engram_n_heads=2,
        engram_vocab_size=97,
        engram_compressed_vocab_size=97,
        engram_pad_token_id=0,
        engram_head_dim=32,
        # MTP (E4.2; used only by the load_weights geometry schema here)
        num_nextn_predict_layers=1,
        dspark_n_routed_experts=2,
        # host-only shared Engram table location
        engram_shm_dir=_SCRATCH,
        engram_shm_prefix=prefix,
    )


@pytest.fixture
def boot_config():
    return _tiny_v41_config("e41boot", num_hidden_layers=16, engram_layer_ids=(1, 14), n_routed_experts=8)


@pytest.fixture
def boot_model(boot_config):
    torch.manual_seed(0)
    model = AscendDeepseekV41EagerForCausalLM(config=boot_config)
    yield model
    model.close()


def _input_ids(seq_len: int = 6) -> torch.Tensor:
    return torch.tensor([3, 1, 4, 1, 5, 9, 2, 6][:seq_len], dtype=torch.long)


# ===========================================================================
# Import hygiene (Triton-free / no shipped base / no torch_npu on import)
# ===========================================================================


def test_assembly_import_pulls_no_triton_or_shipped_base():
    assert not _ASSEMBLY_PULLS_SHIPPED_BASE, "assembly import pulled the shipped V4 base (should stay lazy)"
    assert not _ASSEMBLY_PULLS_MUL_ADD, "assembly import pulled muls_add_triton (E3.3 swap target)"
    assert not _ASSEMBLY_PULLS_W2_DYNAMIC, "assembly import eagerly pulled the heavy W2 method"


def test_assembly_source_has_no_triton_import():
    src = Path(assembly.__file__).read_text()
    import_lines = "\n".join(line for line in src.splitlines() if line.strip().startswith(("import ", "from ")))
    assert "triton" not in import_lines
    assert "muls_add_triton" not in import_lines


def test_fresh_interpreter_import_is_clean():
    # A fresh interpreter importing the assembly (no test stubs) must NOT pull
    # torch_npu, the shipped V4 base, muls_add_triton, or the W2 method eagerly.
    code = (
        "import os, sys\n"
        "os.environ['VLLM_ASCEND_ENABLE_310P'] = '1'\n"
        "import vllm_ascend.models.deepseek_v41.assembly as a\n"
        "assert 'torch_npu' not in sys.modules, 'torch_npu pulled by assembly import'\n"
        "assert 'vllm_ascend.models.deepseek_v4.model' not in sys.modules, 'shipped V4 base pulled'\n"
        "assert 'vllm_ascend.ops.triton.mul_add' not in sys.modules, 'muls_add_triton pulled'\n"
        "assert 'vllm_ascend._310p.quantization.methods.w2_dynamic' not in sys.modules, 'W2 method pulled'\n"
        "print('OK')\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, f"subprocess failed:\nSTDOUT:{proc.stdout}\nSTDERR:{proc.stderr}"
    assert "OK" in proc.stdout


# ===========================================================================
# Construction + forward + sample
# ===========================================================================


def test_construct_forward_and_sample_token(boot_model):
    input_ids = _input_ids()
    hidden = boot_model.forward(input_ids)
    assert hidden.shape == (input_ids.shape[0], boot_model.config.hidden_size)
    assert torch.isfinite(hidden).all()

    logits = boot_model.compute_logits(hidden)
    assert logits.shape == (input_ids.shape[0], boot_model.config.vocab_size)

    token = boot_model.sample_token(input_ids)
    assert token.dtype == torch.long
    assert 0 <= int(token) < boot_model.config.vocab_size


def test_two_fixed_seed_forwards_are_identical(boot_model):
    input_ids = _input_ids()
    first = boot_model.forward(input_ids)
    second = boot_model.forward(input_ids)
    assert torch.equal(first, second), "eager assembly forward is not deterministic"
    # The sampled token is likewise stable.
    assert int(boot_model.sample_token(input_ids)) == int(boot_model.sample_token(input_ids))


def test_two_models_same_seed_agree(boot_config):
    # Two independently constructed models under the same torch seed produce the
    # same forward -- the whole graph init is deterministic.
    cfg_a = _tiny_v41_config("e41twin_a", num_hidden_layers=6, engram_layer_ids=(1,), n_routed_experts=8)
    cfg_b = _tiny_v41_config("e41twin_b", num_hidden_layers=6, engram_layer_ids=(1,), n_routed_experts=8)
    torch.manual_seed(0)
    a_model = AscendDeepseekV41EagerForCausalLM(config=cfg_a)
    torch.manual_seed(0)
    b_model = AscendDeepseekV41EagerForCausalLM(config=cfg_b)
    try:
        ids = _input_ids(5)
        assert torch.equal(a_model.forward(ids), b_model.forward(ids))
    finally:
        a_model.close()
        b_model.close()


# ===========================================================================
# Engram injection at [1, 14] + indexer on ratio-{1,2} layers
# ===========================================================================


def test_engram_injected_at_layer_ids(boot_model):
    assert boot_model.model.engram_injected_layers == (1, 14)
    for idx, layer in enumerate(boot_model.model.layers):
        has_engram = layer.engram is not None
        assert has_engram == (idx in (1, 14)), f"layer {idx} engram={has_engram}"
    # The injected sub-blocks carry the correct hash indices (enumerate order).
    assert boot_model.model.layers[1].engram.layer_hash_index == 0
    assert boot_model.model.layers[14].engram.layer_hash_index == 1


def test_indexer_runs_on_ratio_1_2_layers(boot_model):
    boot_model.forward(_input_ids())
    for idx, layer in enumerate(boot_model.model.layers):
        ratio = boot_model.config.compress_ratios[idx]
        if ratio in (1, 2):
            assert layer.indexer is not None and layer.indexer.enabled
            assert layer.last_selection is not None, f"indexer layer {idx} produced no selection"
            assert layer.last_selection.block_indices.shape[0] == _input_ids().shape[0]
        else:
            assert layer.indexer is None


# ===========================================================================
# KV-group report matches the E2.2 kv plan
# ===========================================================================


def test_kv_group_report_matches_e22_plan(boot_model):
    plan = build_deepseekv41_layer_plan(boot_model.config, dtype_policy=boot_model.dtype_policy)
    groups = build_deepseekv41_kv_cache_groups(plan)

    report = boot_model.kv_group_report()
    assert report["num_layers"] == boot_model.config.num_hidden_layers == len(plan)
    # Every layer carries the MLA latent cache.
    assert report["mla_latent_layers"] == len(plan)
    # Ratio-{1,2} layers additionally carry an indexer compressed-history cache.
    expected_indexer = sum(1 for r in boot_model.config.compress_ratios if r in (1, 2))
    assert report["indexer_layers"] == expected_indexer
    assert report["num_groups"] == len(groups)

    # The assembly returns exactly the kv.py groups (identical specs merge).
    got = boot_model.get_kv_cache_groups()
    assert [sorted(g.layer_names) for g in got] == [sorted(g.layer_names) for g in groups]
    assert report["group_block_sizes"] == sorted(g.kv_cache_spec.block_size for g in groups)


# ===========================================================================
# load_weights: synthetic real-shaped W2 index -> E1.3 layout, host, FP16, reject
# ===========================================================================


def _load_model():
    torch.manual_seed(0)
    return AscendDeepseekV41EagerForCausalLM(
        config=_tiny_v41_config("e41load", num_hidden_layers=2, engram_layer_ids=(1,), n_routed_experts=4)
    )


def _expert_stream(geometry: dict) -> list[tuple[str, torch.Tensor]]:
    """Every routed-expert tensor for ``geometry`` at correct shape + dtype."""
    stream: list[tuple[str, torch.Tensor]] = []
    for block, num_experts in expected_expert_blocks(geometry).items():
        for expert in range(num_experts):
            for slot in ("w1", "w2", "w3"):
                for kind in ("codes", "scale"):
                    name = f"{block}.ffn.experts.{expert}.{slot}_{kind}"
                    shape = expected_expert_shape(slot, kind, geometry)
                    if kind == "codes":
                        tensor = torch.randint(0, 256, shape, dtype=torch.uint8)
                    else:
                        tensor = torch.randn(*shape, dtype=torch.float32)
                    stream.append((name, tensor))
    return stream


def _full_stream(model) -> list[tuple[str, torch.Tensor]]:
    geometry = model._expert_geometry()
    stream = _expert_stream(geometry)
    # Engram ~W4 host rows (single shared host copy; never a device param).
    stream.append(("layers.1.engram.embed_codes.p0", torch.zeros(64, 16, dtype=torch.uint8)))
    stream.append(("layers.1.engram.wkv_codes", torch.zeros(32, 16, dtype=torch.uint8)))
    # FP16 by name (loaded) + one FP16 that has no destination (classified only).
    stream.append(("embed.weight", torch.randn(64, 64).half()))
    stream.append(("head.weight", torch.randn(64, 64).half()))
    stream.append(("norm.weight", torch.randn(64).half()))
    stream.append(("layers.0.attn.wq_a.weight", torch.randn(32, 64).half()))
    # Excluded vision tower (text-only deployment).
    stream.append(("vision.blocks.0.attn.wo.weight", torch.randn(64, 64).half()))
    return stream


def test_load_weights_maps_experts_engram_and_fp16():
    model = _load_model()
    try:
        geometry = model._expert_geometry()
        stream = _full_stream(model)
        norm_source = next(t for n, t in stream if n == "norm.weight")

        loaded = model.load_weights(stream)
        report = model.load_report

        # Every routed-expert tensor placed into the E1.3 fused layout.
        num_expert_tensors = sum(6 * n for n in expected_expert_blocks(geometry).values())
        assert report["expert_tensors_placed"] == num_expert_tensors
        # Experts really landed in the fused w13_*/w2_* banks (E1.3 layout).
        bank = model._expert_param_banks["layers.0"]
        assert set(bank) == {"w13_codes", "w13_scale", "w2_codes", "w2_scale"}
        hidden, inter = geometry["hidden_size"], geometry["moe_intermediate_size"]
        assert tuple(bank["w13_codes"].shape) == (geometry["n_routed_experts"], 2 * inter, hidden // 4)
        assert bank["w13_codes"].any() and bank["w2_codes"].any()
        # The MTP block gets its own (dspark) bank -- decoupled from the decoder stack.
        assert "mtp.0" in model._expert_param_banks

        # Engram rows -> shared host copy (recorded, not a device param).
        assert report["engram_host"] == {"layers.1.engram.embed_codes.p0", "layers.1.engram.wkv_codes"}

        # FP16 loaded by name; the norm actually took the streamed values.
        assert report["fp16_loaded"] == {"embed.weight", "head.weight", "norm.weight"}
        assert torch.equal(model.model.norm.detach(), norm_source.to(model.model.norm.dtype))
        assert report["fp16_unplaced"] == {"layers.0.attn.wq_a.weight"}
        assert report["excluded"] == {"vision.blocks.0.attn.wo.weight"}
        assert "layers.0.w13_codes" in loaded
    finally:
        model.close()


def test_load_weights_rejects_wrong_shape():
    model = _load_model()
    try:
        with pytest.raises(ShapeMismatchError):
            model.load_weights([("layers.0.ffn.experts.0.w1_codes", torch.zeros(8, 8, dtype=torch.uint8))])
    finally:
        model.close()


def test_load_weights_rejects_wrong_dtype():
    model = _load_model()
    try:
        shape = expected_expert_shape("w1", "codes", model._expert_geometry())
        # codes must be uint8; hand it float32 instead.
        with pytest.raises(DtypeMismatchError):
            model.load_weights([("layers.0.ffn.experts.0.w1_codes", torch.zeros(*shape, dtype=torch.float32))])
    finally:
        model.close()


def test_load_weights_rejects_missing_experts():
    model = _load_model()
    try:
        with pytest.raises(MissingTensorError):
            model.load_weights(_full_stream(model)[:5])
    finally:
        model.close()


def test_load_weights_rejects_duplicate_and_extra():
    model = _load_model()
    try:
        stream = _full_stream(model)
        with pytest.raises(DuplicateTensorError):
            model.load_weights([*stream, stream[0]])
    finally:
        model.close()

    model = _load_model()
    try:
        geometry = model._expert_geometry()
        stream = _full_stream(model)
        # An expert id beyond the schema is an unexpected (extra) expert tensor.
        extra_expert = geometry["n_routed_experts"]
        for slot in ("w1", "w2", "w3"):
            for kind in ("codes", "scale"):
                shape = expected_expert_shape(slot, kind, geometry)
                tensor = (
                    torch.zeros(*shape, dtype=torch.uint8)
                    if kind == "codes"
                    else torch.zeros(*shape, dtype=torch.float32)
                )
                stream.append((f"layers.0.ffn.experts.{extra_expert}.{slot}_{kind}", tensor))
        with pytest.raises(ExtraTensorError):
            model.load_weights(stream)
    finally:
        model.close()


def test_load_weights_dtype_token_contract():
    # The E3.4 dtype tokens the loader validates against: codes uint8, scale fp32.
    assert expected_expert_dtype("codes") == "U8"
    assert expected_expert_dtype("scale") == "F32"
