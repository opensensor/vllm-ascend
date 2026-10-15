# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host ownership and changing-boundary gates for shared causal positions."""

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from tools.glm_perf.resident_control import Control, prepare_replacements
from tools.qwen4exp.resident_candidates import shared_qsa_positions as candidate
from vllm_ascend.models.qwen4_exp import model


def metadata(sequence_lengths=(9, 20), query_lengths=(3, 3)):
    return SimpleNamespace(
        seq_lens_cpu=torch.tensor(sequence_lengths, dtype=torch.int32),
        query_lens_cpu=torch.tensor(query_lengths, dtype=torch.int32),
    )


def mrope(tokens):
    return torch.stack((torch.arange(tokens) + 100, torch.zeros(tokens), torch.ones(tokens))).long()


class Capture:
    _capturing = True

    def __init__(self):
        self.callbacks = []

    def add_eager(self, callback):
        self.callbacks.append(callback)
        self._capturing = False
        try:
            callback()
        finally:
            self._capturing = True


def install_capture(monkeypatch, context):
    capture = Capture()
    monkeypatch.setattr(candidate.BreakableCUDAGraphCapture, "current", lambda: capture)
    monkeypatch.setattr(candidate, "get_forward_context", lambda: context)
    fake_utils = ModuleType("vllm_ascend.utils")
    fake_utils.weak_ref_tensor = lambda tensor: tensor
    monkeypatch.setitem(sys.modules, fake_utils.__name__, fake_utils)
    return capture


def select(boundaries, tokens, prefix, cache, positions=None):
    positions = mrope(tokens) if positions is None else positions
    return candidate.step_logical_positions(boundaries, tokens, positions.device, positions, prefix, cache)


@pytest.mark.parametrize("queries", [(3,), (3, 3), (1, 2, 3), (0, 3, 0)])
def test_eager_shares_only_same_metadata_within_one_forward(monkeypatch, queries):
    monkeypatch.setattr(candidate.BreakableCUDAGraphCapture, "current", lambda: None)
    lengths = tuple(23410 + index * 100 for index in range(len(queries)))
    boundaries = metadata(lengths, queries)
    cache = {}
    first = select(boundaries, sum(queries), "qsa0", cache)
    expected = model._host_qsa_logical_positions(boundaries, sum(queries))
    assert torch.equal(first, expected)
    for layer in range(1, 12):
        assert select(boundaries, sum(queries), f"qsa{layer}", cache) is first
    other_group = select(metadata(lengths, queries), sum(queries), "other_group", cache)
    assert other_group is not first
    assert torch.equal(other_group, first)
    boundaries.seq_lens_cpu.add_(10)
    next_step = select(boundaries, sum(queries), "qsa0", {})
    assert next_step is not first
    assert torch.equal(next_step, first + 10)


def test_text_keeps_original_device_view_without_cache(monkeypatch):
    positions = torch.arange(7, 10)
    cache = {}
    monkeypatch.setattr(candidate, "_host_qsa_logical_positions", Mock(side_effect=AssertionError("host work")))
    actual = select(metadata((10,), (3,)), 3, "qsa", cache, positions)
    assert actual.data_ptr() == positions.data_ptr()
    assert cache == {}


def test_capture_shares_twelve_layers_and_refreshes_after_metadata_replacement(monkeypatch):
    boundaries = metadata()
    context = SimpleNamespace(attn_metadata={f"qsa{i}": boundaries for i in range(12)})
    capture = install_capture(monkeypatch, context)
    cache = {}
    outputs = [select(boundaries, 6, prefix, cache) for prefix in context.attn_metadata]
    assert len(capture.callbacks) == 1
    assert all(output is outputs[0] for output in outputs)
    assert torch.equal(outputs[0], torch.tensor([6, 7, 8, 17, 18, 19]))
    for lengths, queries in [((15, 29), (3, 3)), ((262144, 500), (1, 5)), ((8, 40), (3, 3))]:
        current = metadata(lengths, queries)
        context.attn_metadata = {prefix: current for prefix in context.attn_metadata}
        capture.callbacks[0]()
        assert torch.equal(outputs[0], model._host_qsa_logical_positions(current, 6))


def test_equal_lengths_in_different_metadata_groups_get_separate_callbacks(monkeypatch):
    left, right = metadata(), metadata()
    context = SimpleNamespace(attn_metadata={"left": left, "right": right})
    capture = install_capture(monkeypatch, context)
    cache = {}
    outputs = [select(boundaries, 6, prefix, cache) for prefix, boundaries in context.attn_metadata.items()]
    assert len(capture.callbacks) == 2
    assert outputs[0] is not outputs[1]
    right.seq_lens_cpu.add_(100)
    for callback in capture.callbacks:
        callback()
    assert torch.equal(outputs[1], outputs[0] + 100)


def test_replay_rejects_divergent_metadata_for_shared_consumers(monkeypatch):
    boundaries = metadata()
    context = SimpleNamespace(attn_metadata={"a": boundaries, "b": boundaries})
    capture = install_capture(monkeypatch, context)
    cache = {}
    select(boundaries, 6, "a", cache)
    select(boundaries, 6, "b", cache)
    context.attn_metadata["b"] = metadata((10, 25))
    with pytest.raises(RuntimeError, match="identical replay metadata"):
        capture.callbacks[0]()


def test_replay_rejects_too_few_positions(monkeypatch):
    boundaries = metadata()
    context = SimpleNamespace(attn_metadata={"a": boundaries})
    capture = install_capture(monkeypatch, context)
    select(boundaries, 6, "a", {})
    context.attn_metadata["a"] = metadata((20,), (3,))
    with pytest.raises(RuntimeError, match="captured token count"):
        capture.callbacks[0]()


def test_baseline_dispatch_keeps_per_layer_callbacks(monkeypatch):
    boundaries = metadata()
    context = SimpleNamespace(attn_metadata={"a": boundaries, "b": boundaries})
    capture = install_capture(monkeypatch, context)
    monkeypatch.setattr(model, "get_forward_context", lambda: context)
    positions = mrope(6)
    cache = {}
    for prefix in context.attn_metadata:
        model._step_qsa_logical_positions(boundaries, 6, positions.device, positions, prefix, cache)
    assert len(capture.callbacks) == 2
    assert cache == {}


def test_baseline_position_copy_refreshes_stable_buffer_on_cpu(monkeypatch):
    boundaries = metadata((9,), (3,))
    context = SimpleNamespace(attn_metadata={"draft": boundaries})
    capture = install_capture(monkeypatch, context)
    monkeypatch.setattr(model, "get_forward_context", lambda: context)
    monkeypatch.setattr(torch.Tensor, "pin_memory", Mock(side_effect=AssertionError("CPU must not pin")))
    positions = mrope(3)
    output = model._step_qsa_logical_positions(boundaries, 3, positions.device, positions, "draft", {})
    pointer = output.data_ptr()
    assert output.tolist() == [6, 7, 8]
    for length in (2, 262144, 23):
        context.attn_metadata["draft"] = metadata((length,), (3,))
        capture.callbacks[0]()
        assert output.data_ptr() == pointer
        assert output.tolist() == list(range(length - 3, length))


def test_decoder_passes_explicit_forward_cache_to_qsa():
    values = torch.zeros(3, 8)
    mixer = SimpleNamespace(mix=lambda x: (x, None), combine=lambda x, _: x)
    layer = SimpleNamespace(
        ple=None,
        uses_qsa=True,
        attn_hyper_connection=mixer,
        mlp_hyper_connection=mixer,
        attention=Mock(return_value=values),
        mlp=lambda x: x,
    )
    cache, rope_cache = {}, {}
    positions = mrope(3)
    model.AscendQwen4ExpDecoderLayer.forward(
        layer, values, positions, torch.zeros(3).long(), qsa_rope_cache=rope_cache, logical_position_cache=cache
    )
    layer.attention.assert_called_once_with(values, positions, rope_cache, cache)


def test_backbone_creates_one_cache_for_each_forward():
    observed = []

    def layer(hidden, *args, logical_position_cache, **kwargs):
        observed.append(logical_position_cache)
        return hidden

    backbone = SimpleNamespace(
        hc_count=2,
        layers=[layer] * 12,
        _mtp_hidden_buffer=None,
        hyper_connection_mixer=SimpleNamespace(mix=lambda x: (x, None)),
        dtype_policy=SimpleNamespace(main_dtype=torch.float16),
    )
    for _ in range(2):
        model.AscendQwen4ExpModel.forward(backbone, None, mrope(3), inputs_embeds=torch.ones(3, 4).half())
    assert all(cache is observed[0] for cache in observed[:12])
    assert all(cache is observed[12] for cache in observed[12:])
    assert observed[0] is not observed[12]


def test_replacements_are_preparable_by_resident_controller():
    source = Path(candidate.__file__).read_text()
    prepared = prepare_replacements(Control("test", candidate="shared_qsa_positions", source=source))
    assert [item.target for item in prepared] == ["vllm_ascend.models.qwen4_exp.model:_step_qsa_logical_positions"]
