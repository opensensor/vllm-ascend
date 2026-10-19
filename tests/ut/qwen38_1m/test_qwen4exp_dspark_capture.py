# SPDX-License-Identifier: Apache-2.0
"""Test the real QFN capture/forward methods without loading NPU extensions."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def load_method(class_name, method_name):
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/models/qwen4_exp/model.py"
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == method_name)
    namespace = {"torch": torch}
    # Postponed annotations retain the original method body unchanged.
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), method],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[method_name]


@pytest.mark.parametrize("tokens", [1, 3, 7])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_exact_pre_layer_capture_preserves_target_and_mtp(tokens, dtype):
    capture = load_method("AscendQwen4ExpModel", "set_aux_hidden_state_layers")
    forward = load_method("AscendQwen4ExpModel", "forward")
    hidden, streams = 4, 4
    embeds = torch.arange(tokens * hidden).reshape(tokens, hidden).to(dtype) / 16
    step = torch.arange(streams * hidden).reshape(1, -1).to(dtype) / 16
    before_layers = []

    class Layer:
        def __init__(self, index):
            self.index = index

        def __call__(self, value, *args, **kwargs):
            before_layers.append(value.clone())
            # Deliberately overwrite the same storage so captured views fail.
            value.add_(step + (self.index + 1) / 16)
            return value

    model = SimpleNamespace(
        _dspark_aux_enabled=True,
        _aux_hidden_state_layers=(),
        hc_count=streams,
        hidden_size=hidden,
        layers=[Layer(i) for i in range(48)],
        embed_input_ids=lambda ids: embeds.clone(),
        _mtp_hidden_buffer=torch.empty(tokens, streams * hidden, dtype=dtype),
        dtype_policy=SimpleNamespace(main_dtype=dtype),
        hyper_connection_mixer=SimpleNamespace(mix=lambda x: (x[:, :hidden] * 2, None)),
    )
    ids, positions = torch.arange(tokens), torch.arange(tokens)
    baseline = forward(model, ids, positions).clone()
    expected = {
        i: before_layers[i].reshape(tokens, streams, hidden).float().mean(1).to(dtype) for i in [1, 12, 23, 34, 45]
    }
    mtp_before = model._mtp_hidden_buffer.clone()
    # The installed MRv1 adds one to the checkpoint's zero-based target IDs.
    # Non-sorted order also verifies we return features in the requested order.
    runner_indices = (46, 2, 13, 24, 35)
    capture(model, runner_indices)
    target, aux = forward(model, ids, positions)
    torch.testing.assert_close(target, baseline, rtol=0, atol=0)
    torch.testing.assert_close(model._mtp_hidden_buffer, mtp_before, rtol=0, atol=0)
    assert len(aux) == 5
    for index, actual in zip(runner_indices, aux):
        torch.testing.assert_close(actual, expected[index - 1], rtol=0, atol=0)
        assert actual.dtype == dtype and actual.shape == (tokens, hidden)
        assert actual.untyped_storage().data_ptr() != model._mtp_hidden_buffer.untyped_storage().data_ptr()
    saved = [x.clone() for x in aux]
    model._mtp_hidden_buffer.fill_(0)
    forward(model, ids, positions)  # later invocation cannot overwrite captures
    for actual, wanted in zip(aux, saved):
        torch.testing.assert_close(actual, wanted, rtol=0, atol=0)
    capture(model, ())
    assert isinstance(forward(model, ids, positions), torch.Tensor)


@pytest.mark.parametrize("layers", [(0,), (49,), (2, 2), (True,), (1.0,)])
def test_invalid_layers_do_not_replace_existing_capture(layers):
    model = SimpleNamespace(_dspark_aux_enabled=True, _aux_hidden_state_layers=(1,), layers=[None] * 48)
    with pytest.raises(ValueError):
        load_method("AscendQwen4ExpModel", "set_aux_hidden_state_layers")(model, layers)
    assert model._aux_hidden_state_layers == (1,)


def test_non_dspark_capture_is_rejected_instead_of_using_wrong_features():
    model = SimpleNamespace(_dspark_aux_enabled=False, _aux_hidden_state_layers=(), layers=[None] * 48)
    with pytest.raises(NotImplementedError, match="DSpark only"):
        load_method("AscendQwen4ExpModel", "set_aux_hidden_state_layers")(model, (2, 13, 24, 35, 46))


@pytest.mark.parametrize(
    "class_name,attribute",
    [("AscendQwen4ExpForCausalLM", "model"), ("AscendQwen4ExpForConditionalGeneration", "language_model")],
)
def test_wrapper_forwards_capture_contract(class_name, attribute):
    seen = []
    owner = SimpleNamespace(set_aux_hidden_state_layers=seen.append)
    wrapper = SimpleNamespace(**{attribute: owner})
    indices = (2, 13, 24, 35, 46)
    load_method(class_name, "set_aux_hidden_state_layers")(wrapper, indices)
    assert seen == [indices]


def test_default_layers_cannot_silently_replace_checkpoint_contract():
    with pytest.raises(ValueError, match="explicit target_layer_ids"):
        load_method("AscendQwen4ExpForCausalLM", "get_eagle3_default_aux_hidden_state_layers")(None)
