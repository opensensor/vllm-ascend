# SPDX-License-Identifier: Apache-2.0
"""The unquantized head keeps one NZ parameter and supports changed-input graphs."""

import pytest
import torch
import torch_npu

from vllm_ascend._310p.ops.vocab_parallel_embedding import (
    AscendParallelLMHead310,
    AscendUnquantizedEmbeddingMethod310,
)


@pytest.mark.parametrize("rows", [1, 2, 8, 640])
def test_unquantized_head_nz_alias_and_graph_replay(rows):
    torch.npu.set_device(0)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    # Construct only the parameter container; distributed initialization is
    # unnecessary for validating this per-rank post-load method.
    head = AscendParallelLMHead310.__new__(AscendParallelLMHead310)
    torch.nn.Module.__init__(head)
    generator = torch.Generator().manual_seed(7302)
    original = torch.randn(256, 512, generator=generator, dtype=torch.float16)
    head.weight = torch.nn.Parameter(original.npu(), requires_grad=False)
    parameter = head.weight
    method = AscendUnquantizedEmbeddingMethod310()
    method.process_weights_after_loading(head)
    assert head.weight is parameter
    assert head.weight_nz.data_ptr() == head.weight.data_ptr()
    assert list(dict(head.named_parameters())) == ["weight"]
    assert torch_npu.get_npu_format(head.weight) == 29
    assert torch.equal(head.weight.cpu(), original)
    inputs = torch.randn(rows, 512, generator=generator, dtype=torch.float16).npu()
    bias = torch.randn(256, generator=generator, dtype=torch.float16).npu()
    reference_weight = original.npu()
    reference = torch.nn.functional.linear(inputs, reference_weight, bias)
    assert torch.equal(method.apply(head, inputs, bias).cpu(), reference.cpu())
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        actual = method.apply(head, inputs, bias)
    inputs.mul_(0.75)
    bias.add_(0.125)
    graph.replay()
    torch.npu.synchronize()
    reference = torch.nn.functional.linear(inputs, reference_weight, bias)
    assert torch.equal(actual.cpu(), reference.cpu())
