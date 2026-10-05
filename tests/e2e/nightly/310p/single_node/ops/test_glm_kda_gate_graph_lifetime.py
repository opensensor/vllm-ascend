# SPDX-License-Identifier: Apache-2.0
"""Replay the small MTP graph before the graph captured first has ever run."""

from types import SimpleNamespace

import pytest
import torch
import torch_npu

from vllm_ascend.models.glm5next_w2.kda_310 import (
    _safe_gate,
    _safe_gate_for_layer,
    prepare_kda_gate_weights,
)


@pytest.mark.parametrize("prepared", [False, True])
def test_gate_graphs_do_not_depend_on_replaying_first_capture(prepared):
    torch.npu.set_device(0)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    attn = SimpleNamespace(
        A_log=torch.tensor([0.4, 1.2], device="npu", dtype=torch.float16),
        dt_bias=torch.tensor([-2.0, 0.3, -1.0, 0.5], device="npu", dtype=torch.float16),
        kda_lower_bound=-5.0,
    )
    inputs = {n: torch.linspace(-1, 1, n * 4).reshape(1, n, 2, 2).to("npu") for n in (8, 2)}
    # Warm operator dispatch without creating a per-layer cache.
    for value in inputs.values():
        _safe_gate(value, attn.A_log, attn.dt_bias, attn.kda_lower_bound)
    if prepared:
        prepare_kda_gate_weights(attn)
    torch.npu.synchronize()
    graphs, outputs = {}, {}
    for tokens in (8, 2):
        graphs[tokens] = torch.npu.NPUGraph()
        with torch.npu.graph(graphs[tokens]):
            outputs[tokens] = _safe_gate_for_layer(attn, inputs[tokens])
    for tokens in (2, 2, 8, 2):
        inputs[tokens].add_(0.125)
        graphs[tokens].replay()
        torch.npu.synchronize()
        expected = _safe_gate(inputs[tokens].cpu(), attn.A_log.cpu(), attn.dt_bias.cpu(), attn.kda_lower_bound)
        torch.testing.assert_close(outputs[tokens].cpu(), expected, rtol=1e-5, atol=1e-5)
