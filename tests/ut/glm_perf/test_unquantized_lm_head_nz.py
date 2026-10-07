# SPDX-License-Identifier: Apache-2.0
"""Exercise the 310P head constructor without importing a worker runtime."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

SOURCE = Path(__file__).resolve().parents[3] / "vllm_ascend/_310p/ops/vocab_parallel_embedding.py"


@pytest.fixture
def head_types():
    class PlainEmbeddingMethod:
        pass

    class SpecializedEmbeddingMethod(PlainEmbeddingMethod):
        pass

    class QuantizedMethod:
        pass

    class HeadBase(torch.nn.Module):
        def __init__(self, *args, quant_config=None, **kwargs):
            super().__init__()
            # The source forwards quant_config positionally to the base.
            config = args[6] if len(args) > 6 else quant_config
            self.quant_method = PlainEmbeddingMethod() if config is None else config.method
            self.disable_tp = kwargs.get("disable_tp", False)

    casts = []

    def prepare(weight):
        casts.append(weight)
        return weight.clone()

    names = {"AscendUnquantizedEmbeddingMethod310", "AscendParallelLMHead310"}
    tree = ast.parse(SOURCE.read_text())
    definitions = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name in names]
    assert len(definitions) == len(names)
    namespace = {
        "torch": torch,
        "F": F,
        "UnquantizedEmbeddingMethod": PlainEmbeddingMethod,
        "AscendParallelLMHead": HeadBase,
        "QuantizationConfig": object,
        "DEFAULT_VOCAB_PADDING_SIZE": 64,
        "maybe_trans_nz": prepare,
    }
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(SOURCE), "exec"), namespace)
    return SimpleNamespace(
        head=namespace["AscendParallelLMHead310"],
        native=namespace["AscendUnquantizedEmbeddingMethod310"],
        plain=PlainEmbeddingMethod,
        specialized=SpecializedEmbeddingMethod,
        quantized=QuantizedMethod,
        casts=casts,
    )


@pytest.mark.parametrize("has_config", [False, True])
def test_plain_head_prepares_nz_even_with_model_quantization(head_types, has_config):
    config = SimpleNamespace(method=head_types.plain()) if has_config else None
    head = head_types.head(32, 16, quant_config=config, disable_tp=True)
    assert type(head.quant_method) is head_types.native
    assert head.disable_tp
    head.weight = torch.nn.Parameter(torch.randn(32, 16), requires_grad=False)
    parameter = head.weight
    head.quant_method.process_weights_after_loading(head)
    assert head.weight is parameter
    assert head.weight_nz.data_ptr() == head.weight.data_ptr()
    assert list(dict(head.named_parameters())) == ["weight"]
    inputs, bias = torch.randn(2, 16), torch.randn(32)
    expected = F.linear(inputs, head.weight, bias)
    for _ in range(3):
        assert torch.equal(head.quant_method.apply(head, inputs, bias), expected)
    assert head_types.casts == [head.weight]


@pytest.mark.parametrize("method_name", ["specialized", "quantized"])
def test_explicit_head_quantization_method_is_preserved(head_types, method_name):
    method = getattr(head_types, method_name)()
    head = head_types.head(32, 16, quant_config=SimpleNamespace(method=method))
    assert head.quant_method is method
    assert not head_types.casts


def test_embedding_lookup_retains_original_weight(head_types):
    weight = torch.randn(32, 16)
    layer = SimpleNamespace(weight=weight)
    head_types.native().process_weights_after_loading(layer)
    assert layer.weight is weight
    assert layer.weight_nz.data_ptr() != weight.data_ptr()
    assert torch.equal(layer.weight_nz, weight)
