# SPDX-License-Identifier: Apache-2.0
"""Execute actual input-batch rebuild calls without importing NPU runners."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

RUNNER_PATH = Path(__file__).resolve().parents[3] / "vllm_ascend/_310p/model_runner_310p.py"
REBUILD_METHODS = ("__init__", "may_reinitialize_input_batch")


def input_batch_type():
    """Exercise the real 310P constructor and its forwarding to the base class."""
    path = RUNNER_PATH.with_name("npu_input_batch.py")
    tree = ast.parse(path.read_text())
    batch_class = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "NPUInputBatch310")

    class BaseBatch:
        def __init__(self, *, reasoning_config=None, **kwargs):
            self.reasoning_config = reasoning_config

    namespace = {
        "torch": torch,
        "NPUInputBatch": BaseBatch,
        "KVCacheGroupSpec": object,
        "LogitsProcessors": object,
        "ReasoningConfig": object,
        "MultiGroupBlockTable": lambda **kwargs: object(),
    }
    exec(compile(ast.Module(body=[batch_class], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["NPUInputBatch310"]


def rebuild_input_batch(method_name, config):
    tree = ast.parse(RUNNER_PATH.read_text())
    runner_class = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "NPUModelRunner310"
    )
    method = next(node for node in runner_class.body if isinstance(node, ast.FunctionDef) and node.name == method_name)
    assignments = [
        node
        for node in ast.walk(method)
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "NPUInputBatch"
    ]
    assert len(assignments) == 1
    logitsprocs = object()
    runner = SimpleNamespace(
        max_num_reqs=4,
        model_config=SimpleNamespace(max_model_len=131072, get_vocab_size=lambda: 154880),
        max_encoder_len=0,
        max_num_tokens=640,
        device="cpu",
        pin_memory=False,
        block_size=128,
        cache_config=SimpleNamespace(block_size=128),
        vllm_config=config,
        input_batch=SimpleNamespace(logitsprocs=logitsprocs, reasoning_config=config.__dict__.get("reasoning_config")),
        is_pooling_model=False,
        parallel_config=SimpleNamespace(cp_kv_cache_interleave_size=1),
        kernel_block_sizes=[[128], [0]],
    )
    original_batch = runner.input_batch
    calls = []
    batch_type = input_batch_type()

    def batch_factory(*, reasoning_config=None, **kwargs):
        calls.append(dict(kwargs, reasoning_config=reasoning_config))
        return batch_type(reasoning_config=reasoning_config, **kwargs)

    namespace = {
        "self": runner,
        "NPUInputBatch": batch_factory,
        "max_model_len": 131072,
        "block_sizes": [128, 640],
        "max_num_blocks": [1024, 205],
        "kv_cache_config": SimpleNamespace(kv_cache_groups=[object(), object()]),
    }
    # Run the repository's complete constructor assignment, including all
    # argument evaluation, rather than checking a keyword string in source.
    exec(compile(ast.Module(body=assignments, type_ignores=[]), str(RUNNER_PATH), "exec"), namespace)
    assert runner.input_batch is not original_batch
    assert len(calls) == 1
    assert calls[0]["logitsprocs"] is logitsprocs
    assert calls[0]["max_num_batched_tokens"] == 640
    return runner.input_batch, calls[0]


@pytest.mark.parametrize("method_name", REBUILD_METHODS)
@pytest.mark.parametrize("enabled", (True, False))
def test_rebuild_preserves_reasoning_budget_and_boundary_configuration(method_name, enabled):
    reasoning_config = SimpleNamespace(
        enabled=enabled,
        reasoning_start_token_ids=[154841],
        reasoning_end_token_ids=[154842],
        natural_reasoning_end_token_ids=[154842],
    )
    config = SimpleNamespace(
        speculative_config=SimpleNamespace(num_speculative_tokens=1), reasoning_config=reasoning_config
    )

    batch, call = rebuild_input_batch(method_name, config)

    assert batch.reasoning_config is reasoning_config
    assert call["reasoning_config"] is reasoning_config
    assert call["is_spec_decode"] is True
    assert call["num_speculative_tokens"] == 1


@pytest.mark.parametrize("method_name", REBUILD_METHODS)
@pytest.mark.parametrize("has_config_attribute", (True, False))
def test_rebuild_keeps_no_reasoning_config_compatible(method_name, has_config_attribute):
    config = SimpleNamespace(speculative_config=None)
    if has_config_attribute:
        config.reasoning_config = None

    batch, call = rebuild_input_batch(method_name, config)

    assert batch.reasoning_config is None
    assert call["is_spec_decode"] is False
    assert call["num_speculative_tokens"] == 0
