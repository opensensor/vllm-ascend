# SPDX-License-Identifier: Apache-2.0
"""The final FP32 mixer must not replace intermediate or draft state mixers."""

import json
from types import MethodType, SimpleNamespace

import pytest
import torch

from tools.glm_perf.instance_bindings import InstanceBindings
from tools.glm_perf.resident_candidates.mhc_final_post import bind_final_mixer


def setup():
    old = lambda self, *args: "original"
    owner = SimpleNamespace()
    owner._forward_method = MethodType(old, owner)
    layer = SimpleNamespace(
        mhc_post_op=owner,
        layer_idx=44,
        num_hidden_layers=45,
        is_mtp_layer=False,
        n=4,
        hidden_size=4096,
        mhc_fused_post_pre_op=SimpleNamespace(use_310p_fp16_mhc_state=True),
        hc_attn_fn=torch.empty(1),
    )
    root = SimpleNamespace(modules=lambda: iter([layer]))
    runner = SimpleNamespace(model=SimpleNamespace(runnable=root))
    return layer, runner


def test_only_final_qualified_prefill_runs_native_and_restoration_is_exact():
    layer, runner = setup()

    class Native:
        final_only = True

        def __init__(self):
            self.configs = {}

        def __call__(self, *args):
            return "native FP32"

    bindings, native = InstanceBindings(), Native()
    old = layer.mhc_post_op._forward_method
    state = {"calls_by_rows": {}}
    assert bind_final_mixer(bindings, runner, native, state) == 44
    for rows in (640, 1280):
        args = (torch.empty(rows, 4096), torch.empty(rows, 4, 4096), torch.empty(rows, 4, 1), torch.empty(rows, 4, 4))
        assert layer.mhc_post_op._forward_method(*args) == "native FP32"
    args = (torch.empty(2, 4096), torch.empty(2, 4, 4096), torch.empty(2, 4, 1), torch.empty(2, 4, 4))
    assert layer.mhc_post_op._forward_method(*args) == "original"
    assert state["calls_by_rows"] == {640: 1, 1280: 1}
    assert len(native.configs) == 2
    bindings.restore()
    assert layer.mhc_post_op._forward_method is old


@pytest.mark.parametrize("problem", ["rounded_helper", "streams", "width", "state", "draft"])
def test_wrong_contract_rejected_before_binding(problem):
    layer, runner = setup()
    native = SimpleNamespace(final_only=problem != "rounded_helper", configs={})
    if problem == "streams":
        layer.n = 8
    elif problem == "width":
        layer.hidden_size = 2048
    elif problem == "state":
        layer.mhc_fused_post_pre_op.use_310p_fp16_mhc_state = False
    elif problem == "draft":
        layer.is_mtp_layer = True
    bindings = InstanceBindings()
    with pytest.raises(ValueError):
        bind_final_mixer(bindings, runner, native, {"calls_by_rows": {}})
    assert not bindings.originals and not native.configs


@pytest.mark.parametrize("problem", ["none", "finite_error", "nan"])
def test_actual_activation_gate_reports_failure_and_returns_reference_without_executor_error(problem):
    layer, runner = setup()

    def reference(self, x, residual, post, comb):
        return torch.einsum("nij,nih->njh", comb, residual) + post * x.unsqueeze(1)

    layer.mhc_post_op._forward_method = MethodType(reference, layer.mhc_post_op)

    class Native:
        final_only = True

        def __init__(self):
            self.configs = {}
            self.calls = 0

        def __call__(self, *args):
            self.calls += 1
            result = reference(None, *args)
            if problem == "finite_error":
                result.add_(1)
            elif problem == "nan":
                result.fill_(float("nan"))
            return result

    native = Native()
    state = {"calls_by_rows": {}, "verify_first_request": True, "actual_gate": None}
    bind_final_mixer(InstanceBindings(), runner, native, state)
    args = (
        torch.ones(640, 4096),
        torch.ones(640, 4, 4096),
        torch.ones(640, 4, 1),
        torch.eye(4).expand(640, 4, 4).contiguous(),
    )
    for _ in range(2):
        assert torch.equal(layer.mhc_post_op._forward_method(*args), reference(None, *args))
    assert state["actual_gate"]["passed"] == (problem == "none")
    assert native.calls == (2 if problem == "none" else 1)
    json.dumps(state, allow_nan=False)
