# SPDX-License-Identifier: Apache-2.0
"""Execute the runner's Mamba reshape on CPU and check shared-page isolation."""

import ast
import logging
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture
def reshape():
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/worker/model_runner_v1.py"
    tree = ast.parse(path.read_text())
    runner = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "NPUModelRunner")
    methods = {node.name: node for node in runner.body if isinstance(node, ast.FunctionDef)}
    method = methods["_reshape_kv_cache_tensors"]
    branch = next(
        node
        for node in ast.walk(method)
        if isinstance(node, ast.If) and ast.unparse(node.test) == "isinstance(current_kv_cache_spec, MambaSpec)"
    )
    # Keep the actual branch, including its continue, in a one-layer loop.
    function = ast.parse(
        "def reshape(self, current_kv_cache_spec, kv_cache_config, kv_cache_raw_tensors):\n"
        '    layer_name = "kda"\n    kv_caches = {}\n'
        '    layer_kv_cache_spec = {"kda": current_kv_cache_spec}\n'
        '    for unused in range(1):\n        pass\n    return kv_caches["kda"]\n'
    ).body[0]
    function.body[3].body = branch.body
    module = ast.Module(
        body=[
            ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
            methods["_adjust_kv_layout"],
            function,
        ],
        type_ignores=[],
    )
    scope = dict(
        torch=torch,
        math=math,
        logger=logging.getLogger(__name__),
        get_dtype_size=lambda dtype: torch.empty((), dtype=dtype).element_size(),
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), scope)
    instance = SimpleNamespace(
        hybrid_with_attn_and_mamba=True,
        max_num_reqs=4,
        model_config=SimpleNamespace(hf_text_config=SimpleNamespace(model_type="glm5_next_text")),
    )
    instance._adjust_kv_layout = lambda *args, **kwargs: scope["_adjust_kv_layout"](instance, *args, **kwargs)
    return lambda spec, blocks, raw: scope["reshape"](instance, spec, SimpleNamespace(num_blocks=blocks), raw)


@pytest.mark.parametrize("component", [0, 1])
@pytest.mark.parametrize("block", [1, 3, 6])
def test_aligned_kda_state_writes_do_not_touch_another_shared_page(reshape, component, block):
    blocks, page_bytes = 8, 256
    raw = torch.zeros(blocks * page_bytes, dtype=torch.int8)
    spec = SimpleNamespace(
        page_size_bytes=page_bytes,
        mamba_cache_mode="align",
        shapes=((4, 6), (2, 4, 4)),
        dtypes=(torch.float16, torch.float16),
    )
    conv, recurrent = reshape(spec, blocks, {"kda": raw, "mla": raw})
    # The other scheduler groups own every page except this KDA block.
    raw.view(blocks, page_bytes).fill_(7)
    (conv, recurrent)[component][block].fill_(2)
    other_pages = torch.arange(blocks) != block
    assert torch.all(raw.view(blocks, page_bytes)[other_pages] == 7)
    for state in (conv, recurrent):
        assert state.stride(0) * state.element_size() == page_bytes


def test_private_live_state_retains_dense_native_layout(reshape):
    blocks = 4
    spec = SimpleNamespace(
        page_size_bytes=112, mamba_cache_mode="none", shapes=((4, 6), (2, 4, 4)), dtypes=(torch.float16, torch.float16)
    )
    raw = torch.zeros(blocks * spec.page_size_bytes, dtype=torch.int8)
    states = reshape(spec, blocks, {"kda": raw})
    assert all(state.is_contiguous() for state in states)
