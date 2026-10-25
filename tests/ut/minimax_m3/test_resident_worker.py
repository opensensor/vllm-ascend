# SPDX-License-Identifier: Apache-2.0
"""Real resident transaction code with CPU storage and scheduler stand-ins."""

import dataclasses
import json
import types
import uuid
from unittest.mock import Mock

import pytest
import torch

from tools.glm_perf.resident_control import Control
from tools.minimax_m3.resident import candidate_source
from tools.minimax_m3.resident_native import GroupedW4Kernel
from tools.minimax_m3.resident_worker import MiniMaxResidentExtension, storage_identity
from vllm_ascend._310p.ops import minimax_m3
from vllm_ascend._310p.ops.minimax_m3 import PackedM3Experts


@pytest.fixture
def worker(monkeypatch):
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 3)
    monkeypatch.setattr(torch, "npu", types.SimpleNamespace(synchronize=lambda: None), raising=False)
    model = torch.nn.Module()
    model.dense = torch.nn.Linear(256, 256)
    model.experts = PackedM3Experts(256, 256, 4, 2, 0, 2, prefix="model.layers.3.experts")
    worker = MiniMaxResidentExtension()
    worker.model_runner = types.SimpleNamespace(
        model=model,
        vllm_config=types.SimpleNamespace(model_config=types.SimpleNamespace(enforce_eager=True)),
        input_batch=types.SimpleNamespace(req_ids=["finished"], remove_request=Mock()),
        requests={"finished": 1},
        kv_caches=[torch.ones(2, 8)],
        execute_model_state=None,
    )
    return worker


def test_storage_identity_includes_packed_buffers(worker):
    before, count = storage_identity(worker.model_runner.model)
    assert count == 8
    bank = worker.model_runner.model.experts
    bank.w1_codes = bank.w1_codes.clone()
    assert storage_identity(worker.model_runner.model)[0] != before


def test_apply_rollback_keeps_weights_and_clears_cache(worker):
    model = worker.model_runner.model
    before = storage_identity(model)
    source = (
        "def replacements():\n    def project(*args):\n        return 'candidate'\n"
        "    return {'vllm_ascend._310p.ops.minimax_m3:grouped_w4_mm': project}\n"
    )

    original = minimax_m3.grouped_w4_mm
    try:
        control = Control(uuid.uuid4().hex, "direct-target", "test_candidate", source)
        assert "error" not in worker.resident_prepare(json.dumps(dataclasses.asdict(control)))
        assert worker.resident_apply(control.generation)["graphs_dirty"] is False
        assert minimax_m3.grouped_w4_mm() == "candidate"
        assert "error" not in worker.resident_reset()
        assert not worker.model_runner.kv_caches[0].any()
        assert storage_identity(model) == before
        baseline = Control(uuid.uuid4().hex, "direct-target")
        worker.resident_prepare(json.dumps(dataclasses.asdict(baseline)))
        worker.resident_apply(baseline.generation)
        assert minimax_m3.grouped_w4_mm is original
    finally:
        minimax_m3.grouped_w4_mm = original


def test_errors_return_rank_receipts_without_escaping_collective(worker):
    worker.model_runner.execute_model_state = object()
    for result in (worker.resident_reset(), worker.resident_apply("bad"), worker.resident_native_load("bad")):
        assert result["rank"] == 3 and "pending" in result["error"]
    worker.model_runner.execute_model_state = None
    control = Control(uuid.uuid4().hex)
    assert "direct-target" in worker.resident_prepare(json.dumps(dataclasses.asdict(control)))["error"]


def test_composed_module_uses_selected_native_and_patches_callsite_aliases():
    fake = """
def route_experts(): pass
def clipped_swiglu(): pass
def paged_sparse_attention(): pass
def grouped_w4_mm(*args): return 'old'
class PackedM3Experts:
    def forward(self): return grouped_w4_mm(1, 2, 3, 4)
"""
    module = types.ModuleType("test_composition")
    exec(candidate_source(fake, "native_v2"), module.__dict__)
    kernel = lambda *args: "new"
    kernel.max_compact_routes = 4
    replacements = module.replacements({"native_v2": kernel})
    assert replacements["vllm_ascend._310p.ops.minimax_m3:grouped_w4_mm"].max_compact_routes == 4
    assert replacements["vllm_ascend._310p.ops.minimax_m3:PackedM3Experts.forward"](None) == "new"
    assert (
        replacements["vllm_ascend.models.minimax_m3.minimax_m3:paged_sparse_attention"]
        is replacements["vllm_ascend._310p.ops.minimax_m3:paged_sparse_attention"]
    )


@pytest.mark.parametrize(
    "rows,dtype,scale_dtype",
    [(2049, torch.int8, torch.float32), (4, torch.uint8, torch.float32), (4, torch.int8, torch.float16)],
)
def test_native_projection_rejects_unqualified_contract_before_launch(rows, dtype, scale_dtype):
    kernel = GroupedW4Kernel.__new__(GroupedW4Kernel)
    kernel.launch = Mock(side_effect=AssertionError("invalid input reached NPU"))
    with pytest.raises(ValueError, match="NZ256"):
        kernel(
            torch.empty(rows, 256).half(),
            torch.empty(2, 256, 128, dtype=dtype),
            torch.empty(2, 8, 8, dtype=scale_dtype),
            torch.tensor([2, 4]),
        )
    kernel.launch.assert_not_called()
