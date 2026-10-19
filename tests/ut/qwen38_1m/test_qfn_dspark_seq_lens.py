# SPDX-License-Identifier: Apache-2.0
"""Exercise the real metadata selector without importing an NPU runtime."""

import ast
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch


class CrossAttentionSpec:
    pass


def selector():
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/attention/utils.py"
    tree = ast.parse(path.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_select_seq_lens")
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), function],
        type_ignores=[],
    )
    namespace = {"CrossAttentionSpec": CrossAttentionSpec, "_is_glm_model": lambda cfg: cfg.glm}
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace["_select_seq_lens"]


@pytest.mark.parametrize("arch", ["Qwen4ExpForCausalLM", "Qwen4ExpForConditionalGeneration"])
@pytest.mark.parametrize("canonical", [True, False])
def test_qfn_corrected_host_mirror_and_request_slice(arch, canonical):
    host = torch.tensor([9, 18, 999], dtype=torch.int32)
    device = torch.tensor([99, 99, 99], dtype=torch.int32)
    metadata = NS(
        num_reqs=2, _seq_lens_cpu=host if canonical else None, seq_lens_cpu=host, seq_lens=device, causal=True
    )
    config = NS(
        additional_config={"qfn_dspark_graph_probe": True},
        scheduler_config=NS(async_scheduling=False),
        model_config=NS(architectures=[arch], glm=False),
    )
    spec = NS(parallel_drafting=True, use_dspark=lambda: True)
    result = selector()(metadata, None, spec, config)
    assert result.tolist() == [9, 18]
    assert result.data_ptr() == host.data_ptr()
    # A later rejected-token correction must be observed, never cached.
    host[:2].sub_(1)
    assert selector()(metadata, None, spec, config).tolist() == [8, 17]


@pytest.mark.parametrize("excluded", ["flag", "async", "model", "method", "cross", "missing_config", "draft"])
def test_qfn_opt_in_does_not_change_other_metadata_paths(excluded):
    host = torch.tensor([9, 18], dtype=torch.int32)
    device = torch.tensor([99, 99], dtype=torch.int32)
    metadata = NS(num_reqs=2, _seq_lens_cpu=host, seq_lens_cpu=host, seq_lens=device, causal=excluded != "draft")
    config = NS(
        additional_config={"qfn_dspark_graph_probe": excluded != "flag"},
        scheduler_config=NS(async_scheduling=excluded == "async"),
        model_config=NS(architectures=["Other" if excluded == "model" else "Qwen4ExpForCausalLM"], glm=False),
    )
    if excluded == "missing_config":
        del config.additional_config
    spec = NS(parallel_drafting=True, use_dspark=lambda: excluded != "method")
    cache = CrossAttentionSpec() if excluded == "cross" else None
    assert selector()(metadata, cache, spec, config) is device


def test_glm_existing_cpu_mirror_path_is_preserved():
    host, device = torch.tensor([8]), torch.tensor([99])
    metadata = NS(num_reqs=1, _seq_lens_cpu=host, seq_lens_cpu=host, seq_lens=device)
    config = NS(model_config=NS(glm=True))
    spec = NS(parallel_drafting=True, use_dspark=lambda: True)
    assert selector()(metadata, None, spec, config).data_ptr() == host.data_ptr()
