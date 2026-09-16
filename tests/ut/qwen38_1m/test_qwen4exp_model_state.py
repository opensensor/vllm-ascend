# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the Ascend 310P Qwen4Exp model state (plan T1.3).

Ports the fork ``Qwen4ExpModelState`` PLE n-gram CONTEXT BUFFER logic onto the
310P ``Ascend310PMambaHybridModelState``. Everything runs host-side with NO NPU.

The ``vllm_ascend._310p.worker.v2.model_state`` module transitively imports the
Ascend attention / fused-MoE stack, which cannot load against this host's
installed vLLM (the same reason the shared ``tests/ut/conftest.py`` fails to
import). We therefore install lightweight stubs for the heavy leaf modules that
``model_state`` imports *before* importing it -- the ported n-gram logic under
test is entirely self-contained in the new state class, and the (mocked) hybrid
base is only reached through ``super()`` calls we patch out.

Run with ``--noconftest`` (the shared conftest fails to import on this host):

    python3 -m pytest -q --noconftest \
        tests/ut/qwen38_1m/test_qwen4exp_model_state.py
"""

import importlib.util
import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


class _MockModule(types.ModuleType):
    """Module whose every missing attribute is a fresh ``MagicMock``."""

    def __getattr__(self, name):
        value = MagicMock()
        setattr(self, name, value)
        return value


def _register(name: str, module: types.ModuleType) -> types.ModuleType:
    module.__spec__ = importlib.util.spec_from_loader(name, loader=None)
    sys.modules.setdefault(name, module)
    return module


def _install_host_stubs() -> None:
    """Install just enough stubs for ``model_state`` to import on CPU."""
    if "torch_npu" in sys.modules and getattr(sys.modules["torch_npu"], "__ascend_qwen4exp_stub__", False):
        return

    triton_runtime = MagicMock()
    triton_runtime.driver.active.utils.get_device_properties.return_value = {
        "num_aic": 8,
        "num_vectorcore": 8,
    }
    sys.modules.setdefault("triton.runtime", triton_runtime)

    torch_npu = _MockModule("torch_npu")
    torch_npu.__path__ = []  # type: ignore[attr-defined]
    torch_npu.__ascend_qwen4exp_stub__ = True  # type: ignore[attr-defined]
    _register("torch_npu", torch_npu)

    # Force the 310P hardware identity so ``is_310p()`` is True.
    build_info = types.ModuleType("vllm_ascend._build_info")
    build_info.__device_type__ = "_310P"  # type: ignore[attr-defined]
    _register("vllm_ascend._build_info", build_info)

    import torch

    try:  # noqa: SIM105
        torch.utils.rename_privateuse1_backend("npu")
    except RuntimeError:
        pass
    torch.npu = MagicMock()  # type: ignore[attr-defined]
    torch.npu.is_available = MagicMock(return_value=False)
    torch.npu.Event = MagicMock

    # Heavy leaf modules ``model_state`` imports directly. The concrete base
    # behaviour is irrelevant here: the state class is exercised via ``__new__``
    # and patched ``super()`` calls, so trivial classes with the prepare hooks
    # are enough to keep the module importable and ``patch.object`` targets live.
    class _StubBase:
        def prepare_inputs(self, input_batch, req_states):  # pragma: no cover
            return {}

        def prepare_dummy_inputs(self, num_reqs, num_tokens):  # pragma: no cover
            return {}

    rope_mod = _MockModule("vllm_ascend._310p.ops.rotary_embedding")
    rope_mod.prepare_mrope_cos_sin_slices_from_runner = MagicMock()
    _register("vllm_ascend._310p.ops.rotary_embedding", rope_mod)

    device_op_mod = _MockModule("vllm_ascend.device.device_op")
    device_op_mod.DeviceOperator = MagicMock()
    _register("vllm_ascend.device.device_op", device_op_mod)

    input_batch_mod = types.ModuleType("vllm_ascend.worker.v2.input_batch")
    input_batch_mod.AscendInputBatch = type("AscendInputBatch", (), {})
    _register("vllm_ascend.worker.v2.input_batch", input_batch_mod)

    default_mod = types.ModuleType("vllm_ascend.worker.v2.model_states.default")
    default_mod.AscendModelState = type("AscendModelState", (_StubBase,), {})
    _register("vllm_ascend.worker.v2.model_states.default", default_mod)

    hybrid_mod = types.ModuleType("vllm_ascend.worker.v2.model_states.mamba_hybrid")
    hybrid_mod.AscendMambaHybridModelState = type("AscendMambaHybridModelState", (_StubBase,), {})
    _register("vllm_ascend.worker.v2.model_states.mamba_hybrid", hybrid_mod)

    sampler_mod = types.ModuleType("vllm_ascend._310p.worker.v2.sampler")
    sampler_mod.Ascend310PSampler = type("Ascend310PSampler", (), {})
    _register("vllm_ascend._310p.worker.v2.sampler", sampler_mod)


def _cleanup_host_stubs() -> None:
    """Drop the ``torch_npu`` stub so sibling tests still see a clean process.

    Other qwen38_1m tests assert ``"torch_npu" not in sys.modules``; the model
    state class and its dependencies have already bound their imports, so the
    entry is safe to remove once importing is done. The remaining stubs (build
    info, leaf modules) are kept: they carry no ``torch_npu`` marker and the
    lazily-imported dispatch still needs the 310P build info to resolve.
    """
    for name in list(sys.modules):
        if name == "torch_npu" or name.startswith("torch_npu."):
            del sys.modules[name]


_install_host_stubs()

import pytest  # noqa: E402
import torch  # noqa: E402

from vllm_ascend._310p.worker.v2 import model_state as model_state_mod  # noqa: E402
from vllm_ascend._310p.worker.v2.model_state import (  # noqa: E402
    Ascend310PQwen4ExpModelState,
)

_cleanup_host_stubs()

_CPU = torch.device("cpu")
_HYBRID_BASE = "vllm_ascend._310p.worker.v2.model_state.Ascend310PMambaHybridModelState"


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


def _make_state(
    *,
    max_num_reqs: int = 4,
    ngram_size: int = 3,
    eos_token_id: int = 0,
    ple_layer_ids=(1,),
    pipeline_parallel_size: int = 1,
) -> Ascend310PQwen4ExpModelState:
    """Build a state instance without running the heavy hybrid ``__init__``."""
    state = Ascend310PQwen4ExpModelState.__new__(Ascend310PQwen4ExpModelState)
    text_config = SimpleNamespace(
        ple_layer_ids=list(ple_layer_ids),
        ngram_size=ngram_size,
        eos_token_id=eos_token_id,
    )
    state.model_config = SimpleNamespace(hf_text_config=text_config)
    state.max_num_reqs = max_num_reqs
    state.device = _CPU
    vllm_config = SimpleNamespace(parallel_config=SimpleNamespace(pipeline_parallel_size=pipeline_parallel_size))
    state._init_ngram_context(vllm_config)
    return state


def _fake_req_states(all_token_ids: torch.Tensor, num_computed_tokens: torch.Tensor):
    return SimpleNamespace(
        all_token_ids=SimpleNamespace(gpu=all_token_ids),
        num_computed_tokens=SimpleNamespace(gpu=num_computed_tokens),
    )


def _fake_input_batch(*, num_reqs: int, idx_mapping: torch.Tensor):
    return SimpleNamespace(num_reqs=num_reqs, idx_mapping=idx_mapping)


def _token_grid(max_num_reqs: int, max_len: int) -> torch.Tensor:
    # Distinctive per-request tokens: request r, position p -> 100*r + p + 1.
    grid = torch.zeros((max_num_reqs, max_len), dtype=torch.int32)
    for r in range(max_num_reqs):
        for p in range(max_len):
            grid[r, p] = 100 * r + p + 1
    return grid


# ---------------------------------------------------------------------------
# Construction / PP enforcement
# ---------------------------------------------------------------------------


def test_pp_greater_than_one_is_rejected():
    with pytest.raises(RuntimeError, match="pipeline_parallel_size=1"):
        _make_state(pipeline_parallel_size=2)


def test_no_ple_layers_disables_ngram():
    state = _make_state(ple_layer_ids=())
    assert state.uses_ngram_embedding is False
    assert state.ngram_context_len == 0


def test_ngram_buffers_have_fixed_shapes():
    state = _make_state(max_num_reqs=4, ngram_size=3)
    # ngram_context_len == ngram_size - 1
    assert state.ngram_context_len == 2
    assert state.ngram_context.shape == (4, 2)
    assert state.ngram_context.dtype == torch.int32
    assert state.ple_query_start_loc.shape == (5,)
    assert torch.equal(state.ngram_context_offsets, torch.tensor([-2, -1]))


def test_ngram_size_one_is_rejected():
    with pytest.raises(ValueError, match="context length >= 1"):
        _make_state(ngram_size=1)


# ---------------------------------------------------------------------------
# N-gram context correctness
# ---------------------------------------------------------------------------


def test_ngram_context_at_chunk_boundary():
    """Context is the (ngram_size-1) tokens preceding num_computed_tokens."""
    state = _make_state(max_num_reqs=4, ngram_size=3, eos_token_id=0)
    tokens = _token_grid(4, 16)
    # req slot 2 has consumed 5 tokens (chunk boundary at 5): context = pos 3,4.
    num_computed = torch.tensor([0, 0, 5, 0], dtype=torch.int64)
    req_states = _fake_req_states(tokens, num_computed)
    input_batch = _fake_input_batch(num_reqs=1, idx_mapping=torch.tensor([2]))

    context = state._prepare_ngram_context(input_batch, req_states)
    # positions 3 and 4 of request slot 2 -> 100*2+4, 100*2+5
    assert context[0].tolist() == [204, 205]
    # unused rows stay EOS-filled (fixed shape for graph capture).
    assert torch.all(context[1:] == 0)


def test_ngram_context_eos_pads_first_tokens():
    """Fewer than (ngram_size-1) preceding tokens -> leading EOS padding."""
    state = _make_state(max_num_reqs=4, ngram_size=3, eos_token_id=7)
    tokens = _token_grid(4, 16)
    # slot 0 consumed exactly 1 token -> [EOS, tokens[0]].
    num_computed = torch.tensor([1, 0, 0, 0], dtype=torch.int64)
    req_states = _fake_req_states(tokens, num_computed)
    input_batch = _fake_input_batch(num_reqs=1, idx_mapping=torch.tensor([0]))

    context = state._prepare_ngram_context(input_batch, req_states)
    assert context[0].tolist() == [7, 1]  # 7 == eos, tokens[0, 0] == 1


def test_ngram_context_rollback_recomputes_from_authoritative_state():
    """After a rejected speculative rollback num_computed_tokens shrinks;
    context must reflect the rolled-back position (recomputed from scratch)."""
    state = _make_state(max_num_reqs=4, ngram_size=3, eos_token_id=0)
    tokens = _token_grid(4, 16)
    input_batch = _fake_input_batch(num_reqs=1, idx_mapping=torch.tensor([1]))

    # Pre-rollback: 8 accepted -> context = pos 6,7.
    req_states = _fake_req_states(tokens, torch.tensor([0, 8, 0, 0], dtype=torch.int64))
    ctx_before = state._prepare_ngram_context(input_batch, req_states).clone()
    assert ctx_before[0].tolist() == [107, 108]

    # Rollback to 6 accepted -> context = pos 4,5. No stale carryover.
    req_states = _fake_req_states(tokens, torch.tensor([0, 6, 0, 0], dtype=torch.int64))
    ctx_after = state._prepare_ngram_context(input_batch, req_states)
    assert ctx_after[0].tolist() == [105, 106]


def test_ngram_context_rollback_to_zero_degenerate_first():
    """S=0 degenerate first step: all context is EOS padding."""
    state = _make_state(max_num_reqs=4, ngram_size=3, eos_token_id=9)
    tokens = _token_grid(4, 16)
    req_states = _fake_req_states(tokens, torch.tensor([0, 0, 0, 0], dtype=torch.int64))
    input_batch = _fake_input_batch(num_reqs=1, idx_mapping=torch.tensor([3]))

    context = state._prepare_ngram_context(input_batch, req_states)
    assert context[0].tolist() == [9, 9]


def test_ngram_context_zero_reqs_returns_all_eos():
    state = _make_state(max_num_reqs=4, ngram_size=3, eos_token_id=5)
    tokens = _token_grid(4, 16)
    req_states = _fake_req_states(tokens, torch.zeros(4, dtype=torch.int64))
    input_batch = _fake_input_batch(num_reqs=0, idx_mapping=torch.empty(0, dtype=torch.int64))

    context = state._prepare_ngram_context(input_batch, req_states)
    assert torch.all(context == 5)


def test_ngram_context_larger_ngram_size():
    """ngram_size=4 -> 3 preceding tokens, with mixed EOS padding at start."""
    state = _make_state(max_num_reqs=4, ngram_size=4, eos_token_id=0)
    tokens = _token_grid(4, 16)
    # slot 0 consumed 2 tokens -> [EOS, tokens[0], tokens[1]].
    req_states = _fake_req_states(tokens, torch.tensor([2, 0, 0, 0], dtype=torch.int64))
    input_batch = _fake_input_batch(num_reqs=1, idx_mapping=torch.tensor([0]))

    context = state._prepare_ngram_context(input_batch, req_states)
    assert context[0].tolist() == [0, 1, 2]


# ---------------------------------------------------------------------------
# ple_query_start_loc
# ---------------------------------------------------------------------------


def test_ple_query_start_loc_pads_trailing_slots():
    state = _make_state(max_num_reqs=4, ngram_size=3)
    # 2 real reqs after padding, cumulative [0, 3, 7]; trailing capacity -> 7.
    input_batch = SimpleNamespace(
        num_reqs_after_padding=2,
        query_start_loc=torch.tensor([0, 3, 7], dtype=torch.int32),
    )
    qsl = state._fill_ple_query_start_loc(input_batch)
    assert qsl.tolist() == [0, 3, 7, 7, 7]


def test_ple_query_start_loc_tolerates_oversized_source_buffer():
    """The 310P input batch keeps query_start_loc as a fixed max+2 buffer."""
    state = _make_state(max_num_reqs=4, ngram_size=3)
    # num_reqs_after_padding=2, source buffer longer than needed, tail == total.
    input_batch = SimpleNamespace(
        num_reqs_after_padding=2,
        query_start_loc=torch.tensor([0, 3, 7, 7, 7, 7], dtype=torch.int32),
    )
    qsl = state._fill_ple_query_start_loc(input_batch)
    assert qsl.tolist() == [0, 3, 7, 7, 7]


# ---------------------------------------------------------------------------
# Integration: prepare_inputs / prepare_dummy_inputs
# ---------------------------------------------------------------------------


def test_prepare_inputs_merges_ngram_and_query_start_loc():
    state = _make_state(max_num_reqs=4, ngram_size=3, eos_token_id=0)
    tokens = _token_grid(4, 16)
    req_states = _fake_req_states(tokens, torch.tensor([0, 5, 0, 0], dtype=torch.int64))
    input_batch = SimpleNamespace(
        num_reqs=1,
        idx_mapping=torch.tensor([1]),
        num_reqs_after_padding=1,
        query_start_loc=torch.tensor([0, 5], dtype=torch.int32),
    )
    with patch(
        f"{_HYBRID_BASE}.prepare_inputs",
        return_value={"positions": torch.zeros(5, dtype=torch.int64)},
    ):
        out = state.prepare_inputs(input_batch, req_states)
    assert "positions" in out  # base 310P output preserved
    assert out["ngram_context"][0].tolist() == [104, 105]
    assert out["query_start_loc"].tolist() == [0, 5, 5, 5, 5]


def test_prepare_inputs_noop_without_ngram():
    state = _make_state(ple_layer_ids=())
    base = {"positions": torch.zeros(3)}
    with patch(f"{_HYBRID_BASE}.prepare_inputs", return_value=base):
        out = state.prepare_inputs(SimpleNamespace(), SimpleNamespace())
    assert out is base
    assert "ngram_context" not in out


def test_prepare_dummy_inputs_builds_fixed_shape_state():
    state = _make_state(max_num_reqs=4, ngram_size=3, eos_token_id=0)
    with patch(
        f"{_HYBRID_BASE}.prepare_dummy_inputs",
        return_value={"positions": torch.zeros(7, dtype=torch.int64)},
    ):
        out = state.prepare_dummy_inputs(num_reqs=3, num_tokens=7)
    # Fixed shapes regardless of active reqs.
    assert out["ngram_context"].shape == (4, 2)
    assert torch.all(out["ngram_context"] == 0)
    qsl = out["query_start_loc"]
    assert qsl.shape == (5,)
    # 7 tokens over 3 reqs -> lens [2, 2, 3] -> cumsum [0,2,4,7], trailing 7.
    assert qsl.tolist() == [0, 2, 4, 7, 7]


def test_prepare_dummy_inputs_noop_without_ngram():
    state = _make_state(ple_layer_ids=())
    base = {"positions": torch.zeros(3)}
    with patch(f"{_HYBRID_BASE}.prepare_dummy_inputs", return_value=base):
        out = state.prepare_dummy_inputs(num_reqs=2, num_tokens=4)
    assert out is base


# ---------------------------------------------------------------------------
# Dispatch registration (model_states/__init__.py)
# ---------------------------------------------------------------------------


def _dispatch_vllm_config(*, is_hybrid=True, ple_layer_ids=(1,)):
    return SimpleNamespace(
        model_config=SimpleNamespace(
            is_hybrid=is_hybrid,
            hf_text_config=SimpleNamespace(ple_layer_ids=list(ple_layer_ids)),
        )
    )


def test_dispatch_routes_ple_hybrid_to_qwen4exp_state_on_310p():
    from vllm_ascend.worker.v2.model_states import init_asecnd_model_state

    sentinel = MagicMock(name="Ascend310PQwen4ExpModelState")
    model = SimpleNamespace()  # no get_model_state_cls
    with (
        patch("vllm_ascend.worker.v2.model_states.is_310p", return_value=True),
        patch.object(model_state_mod, "Ascend310PQwen4ExpModelState", sentinel),
    ):
        result = init_asecnd_model_state(_dispatch_vllm_config(), model, None, _CPU)
    assert result is sentinel.return_value
    sentinel.assert_called_once()


def test_dispatch_tolerates_not_implemented_hook_then_registers():
    """A model may declare get_model_state_cls but defer it (raises)."""
    from vllm_ascend.worker.v2.model_states import init_asecnd_model_state

    def _raises():
        raise NotImplementedError("provided by a later task")

    model = SimpleNamespace(get_model_state_cls=_raises)
    sentinel = MagicMock(name="Ascend310PQwen4ExpModelState")
    with (
        patch("vllm_ascend.worker.v2.model_states.is_310p", return_value=True),
        patch.object(model_state_mod, "Ascend310PQwen4ExpModelState", sentinel),
    ):
        result = init_asecnd_model_state(_dispatch_vllm_config(), model, None, _CPU)
    assert result is sentinel.return_value


def test_dispatch_model_hook_wins_when_it_returns_a_class():
    from vllm_ascend.worker.v2.model_states import init_asecnd_model_state

    winner = MagicMock(name="WinnerState")
    model = SimpleNamespace(get_model_state_cls=lambda: winner)
    with patch("vllm_ascend.worker.v2.model_states.is_310p", return_value=True):
        result = init_asecnd_model_state(_dispatch_vllm_config(), model, None, _CPU)
    assert result is winner.return_value
    winner.assert_called_once()
