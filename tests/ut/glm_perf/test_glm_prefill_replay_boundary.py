# SPDX-License-Identifier: Apache-2.0
"""Run the real sparse scheduler patch without loading NPU worker dependencies."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def split():
    path = ROOT / "vllm_ascend/patch/platform/patch_mamba_block_aligned_split.py"
    tree = ast.parse(path.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_mamba_block_aligned_split")
    function.decorator_list = []
    namespace = {
        "_get_sparse_index_kpool": lambda model: 4 if model.sparse else None,
        "_skips_eagle_block_drop": lambda transfer: transfer is None or transfer.is_kv_producer,
        "_original_mamba_block_aligned_split": lambda *args: 73,
    }
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    exec(
        compile(ast.fix_missing_locations(ast.Module(body=[future, function], type_ignores=[])), str(path), "exec"),
        namespace,
    )
    return namespace[function.name]


@pytest.mark.parametrize(
    "prompt,start,budget,expected",
    [
        (6400, 5120, 1280, 640),
        (6400, 5760, 1280, 640),
        (6400, 3840, 1280, 1280),
        (6400, 6400, 2, 2),
        (1280, 0, 1280, 640),
        (12800, 11520, 1280, 640),
        (6529, 5120, 1280, 1280),
        (6529, 6400, 129, 129),
        (128, 0, 128, 128),
    ],
)
def test_larger_glm_chunks_preserve_identical_replay_endpoint(split, prompt, start, budget, expected):
    scheduler = SimpleNamespace(
        vllm_config=SimpleNamespace(kv_transfer_config=None, model_config=SimpleNamespace(sparse=True)),
        block_size=640,
        use_eagle=True,
        cache_config=SimpleNamespace(enable_prefix_caching=True),
    )
    request = SimpleNamespace(num_computed_tokens=start, num_prompt_tokens=prompt, num_tokens=prompt)
    assert split(scheduler, request, budget) == expected


@pytest.mark.parametrize(
    "prefix,consumer,computed,expected", [(False, False, 5120, 1280), (True, True, 5120, 1280), (True, False, 6400, 2)]
)
def test_disabled_cache_consumer_and_decode_preserve_existing_windows(split, prefix, consumer, computed, expected):
    transfer = SimpleNamespace(is_kv_consumer=consumer, is_kv_producer=not consumer)
    scheduler = SimpleNamespace(
        vllm_config=SimpleNamespace(kv_transfer_config=transfer, model_config=SimpleNamespace(sparse=True)),
        block_size=640,
        use_eagle=True,
        cache_config=SimpleNamespace(enable_prefix_caching=prefix),
    )
    request = SimpleNamespace(num_computed_tokens=computed, num_prompt_tokens=6400, num_tokens=6400)
    assert split(scheduler, request, 2 if computed == 6400 else 1280) == expected


def test_local_and_external_cached_tokens_are_included_in_boundary(split):
    scheduler = SimpleNamespace(
        vllm_config=SimpleNamespace(kv_transfer_config=None, model_config=SimpleNamespace(sparse=True)),
        block_size=640,
        use_eagle=True,
        cache_config=SimpleNamespace(enable_prefix_caching=True),
    )
    request = SimpleNamespace(num_computed_tokens=0, num_prompt_tokens=6400, num_tokens=6400)
    assert split(scheduler, request, 1280, num_new_local_computed_tokens=3840, num_external_computed_tokens=1280) == 640
