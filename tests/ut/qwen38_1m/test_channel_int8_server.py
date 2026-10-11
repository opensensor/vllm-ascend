# SPDX-License-Identifier: Apache-2.0
"""Host checks for replacement storage and bounded descriptor ownership."""

import ast
from pathlib import Path

import pytest
import torch
from torch import nn

from tools.qwen4exp.channel_weight_recipe import channel_bank
from tools.qwen4exp.native_channel_int8 import NativeChannelInt8
from vllm_ascend.models.qwen4_exp.w4a8_int4 import pack_native_metadata, pack_native_weight


@pytest.fixture
def definitions():
    # Read the actual host helpers without installing worker/model patches or
    # importing NPU runtime modules into this CPU-only suite.
    source = Path(__file__).resolve().parents[3] / "tools/qwen4exp/channel_int8_server.py"
    tree = ast.parse(source.read_text())
    names = {"CACHED_ROWS", "ServingConfiguration", "ServingProjection", "replace_bank", "stage_bank_on_host"}
    nodes = [
        node
        for node in tree.body
        if getattr(node, "name", None) in names
        or isinstance(node, ast.Assign)
        and any(getattr(t, "id", None) in names for t in node.targets)
    ]
    scope = {
        "torch": torch,
        "nn": nn,
        "channel_bank": channel_bank,
        "NativeChannelInt8": NativeChannelInt8,
        "_capturing": lambda device: False,
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), scope)
    return scope


def test_replacement_drops_w4_storage_and_retains_only_channel_recipe(definitions):
    bank = nn.Module()
    original = torch.randint(-128, 128, (160, 128), dtype=torch.int8)
    packed, total = pack_native_weight(original)
    bank.weight = nn.Parameter(packed.unsqueeze(0), requires_grad=False)
    bank.weight_scale = nn.Parameter(
        pack_native_metadata(torch.full((160, 2), 0.01).half()).unsqueeze(0), requires_grad=False
    )
    bank.weight_offset = nn.Parameter(
        pack_native_metadata(torch.zeros(160, 2).half()).unsqueeze(0), requires_grad=False
    )
    bank.register_buffer("weight_sum", total.unsqueeze(0))
    expected_weight, expected_scale, error = channel_bank(bank)
    old_bytes = sum(t.numel() * t.element_size() for t in bank.parameters()) + total.numel() * total.element_size()
    result = definitions["replace_bank"](bank)
    assert not hasattr(bank, "weight_offset") and not hasattr(bank, "weight_sum")
    assert set(dict(bank.named_parameters())) == {"weight", "weight_scale"}
    assert list(bank.buffers()) == []
    assert bank.weight.shape == (1, 160, 256) and bank.weight_scale.shape == (1, 160)
    torch.testing.assert_close(bank.weight, expected_weight, rtol=0, atol=0)
    torch.testing.assert_close(bank.weight_scale, expected_scale, rtol=0, atol=0)
    assert result == {"w4_bytes": old_bytes, "int8_bytes": 160 * 256 + 160 * 4, "weight_relative_l2": error}


def test_arbitrary_eager_tails_do_not_accumulate_configurations(definitions):
    resource = definitions["ServingProjection"](None, None)
    for rows in range(2000, 3000):
        descriptor = resource.config((rows, 2560, 1280, 85, 1), torch.device("cpu"))
        assert descriptor.tolist()[0] == rows
    assert len(resource._configs) == 1  # Only the admitted 2560-row descriptor.


def test_graph_shape_addresses_remain_stable(definitions, monkeypatch):
    resource = definitions["ServingProjection"](None, None)
    shape = (30, 2560, 1280, 85, 1)
    descriptor = resource.config(shape, torch.device("cpu"))
    monkeypatch.setitem(definitions, "_capturing", lambda device: True)
    assert resource.config(shape, torch.device("cpu")) is descriptor
    with pytest.raises(RuntimeError, match="not admitted"):
        resource.config((31, 2560, 1280, 85, 1), torch.device("cpu"))


def test_host_staging_preserves_loader_geometry_and_final_recipe(definitions):
    bank = nn.Module()
    raw = torch.randint(-128, 128, (160, 128), dtype=torch.int8)
    packed, total = pack_native_weight(raw)
    bank.weight = nn.Parameter(packed.unsqueeze(0), requires_grad=False)
    bank.weight_scale = nn.Parameter(
        pack_native_metadata(torch.full((160, 2), 0.01).half()).unsqueeze(0), requires_grad=False
    )
    bank.weight_offset = nn.Parameter(
        pack_native_metadata(torch.zeros(160, 2).half()).unsqueeze(0), requires_grad=False
    )
    bank.register_buffer("weight_sum", total.unsqueeze(0))
    source = {name: tensor.clone() for name, tensor in (*bank.named_parameters(), *bank.named_buffers())}
    expected_weight, expected_scale, _ = channel_bank(bank)
    old = bank.weight
    destination = definitions["stage_bank_on_host"](bank)
    assert bank.weight is not old and destination == torch.device("cpu")
    for name, tensor in source.items():
        target = getattr(bank, name)
        assert target.shape == tensor.shape and target.dtype == tensor.dtype and target.device.type == "cpu"
        target.copy_(tensor)  # Simulate the original loader filling every operand.
    definitions["replace_bank"](bank, target_device=destination)
    torch.testing.assert_close(bank.weight, expected_weight, rtol=0, atol=0)
    torch.testing.assert_close(bank.weight_scale, expected_scale, rtol=0, atol=0)
