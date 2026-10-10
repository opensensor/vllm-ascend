# SPDX-License-Identifier: Apache-2.0
"""Execute the real diagnostic loader with CPU tensors and a device-free shell."""

import ast
import json
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.models.qwen4_exp.shared_partition import shared_expert_range


@pytest.mark.parametrize(
    ("mode", "rank", "size", "start", "stop"),
    [
        ("tp_sharded", 2, 4, 320, 480),
        ("tp_sharded_uneven", 0, 6, 0, 107),
        ("tp_sharded_uneven", 4, 6, 428, 534),
        ("tp_sharded_uneven", 5, 6, 534, 640),
        ("replicated", 5, 6, 0, 640),
    ],
)
def test_loader_uses_actual_shared_span_and_disables_grad(tmp_path, mode, rank, size, start, stop):
    prefix = "model.language_model.layers.0.mlp."
    matrices = {
        "gate.weight": torch.ones(1, 8),
        "shared_expert.gate_proj.weight": torch.arange(640 * 8).view(640, 8).float(),
        "shared_expert.up_proj.weight": torch.arange(640 * 8).view(640, 8).float() + 10000,
        "shared_expert.down_proj.weight": torch.arange(8 * 640).view(8, 640).float(),
        "shared_expert_gate.weight": torch.ones(1, 8),
    }
    config = {"shared_expert_intermediate_size": 640, "ascend_expert_quantization": {}}
    (tmp_path / "config.json").write_text(json.dumps({"text_config": config}))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {prefix + name: "weights.safetensors" for name in matrices}})
    )

    class Layer(torch.nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            assert torch.is_inference_mode_enabled()
            self.has_shared_expert = True
            self.shared_expert_execution = mode
            self.shared_expert_replicated = mode == "replicated"
            self.local_shared_inter = stop - start
            self.gate = torch.nn.Parameter(torch.zeros(1, 8))
            self.shared_gate_up = torch.nn.Parameter(torch.zeros(2 * (stop - start), 8))
            self.shared_down = torch.nn.Parameter(torch.zeros(8, stop - start))
            self.shared_expert_gate = torch.nn.Parameter(torch.zeros(1, 8))

        def npu(self):
            return self

    source = Path(__file__).resolve().parents[3] / "tools/qwen4exp/profile_w4_layer_310.py"
    function = next(
        n for n in ast.parse(source.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "load_layer"
    )
    # Avoid importing the profiler's NPU stack in an offline loader regression.
    namespace = dict(
        torch=torch,
        json=json,
        defaultdict=defaultdict,
        SimpleNamespace=SimpleNamespace,
        Qwen4ExpDtypePolicy=lambda: None,
        W4SparseMoE=Layer,
        EXPERT_NAME=SimpleNamespace(fullmatch=lambda _: None),
        shared_expert_range=shared_expert_range,
        safe_open=lambda *a, **kw: nullcontext(
            SimpleNamespace(get_tensor=lambda name: matrices[name.removeprefix(prefix)])
        ),
    )
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
    layer = namespace["load_layer"](tmp_path, 0, rank, size, "cube_310_int4_a8")
    torch.testing.assert_close(
        layer.shared_gate_up,
        torch.cat(
            [
                matrices["shared_expert.gate_proj.weight"][start:stop],
                matrices["shared_expert.up_proj.weight"][start:stop],
            ]
        ),
    )
    torch.testing.assert_close(layer.shared_down, matrices["shared_expert.down_proj.weight"][:, start:stop])
    torch.testing.assert_close(layer.shared_expert_gate, matrices["shared_expert_gate.weight"])
