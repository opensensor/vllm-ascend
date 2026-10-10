# SPDX-License-Identifier: Apache-2.0
"""Exercise the 310P broadcast wrapper without importing NPU dependencies."""

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch


class Tensor:
    device = torch.device("meta")

    def __init__(self, value):
        self.value = value

    def __setitem__(self, key, other):
        self.value = other.value


def wrappers():
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/patch/platform/patch_distributed.py"
    tree = ast.parse(path.read_text())
    tree.body = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))]
    original = Mock()
    subgroup_rank = Mock(return_value=1)

    def gather(output, tensor, *, group):
        for rank, row in enumerate(output):
            row.value = 10 + rank

    distributed = SimpleNamespace(
        broadcast=original,
        distributed_c10d=SimpleNamespace(broadcast=original, all_reduce=Mock()),
        all_reduce=Mock(),
        ReduceOp=torch.distributed.ReduceOp,
        get_group_rank=subgroup_rank,
        get_rank=Mock(return_value=0),
        get_world_size=Mock(return_value=2),
        all_gather=gather,
    )
    namespace = {
        "torch": SimpleNamespace(
            device=torch.device, distributed=distributed, empty_like=lambda _: Tensor(None), int64=torch.int64
        )
    }
    exec(compile(tree, str(path), "exec"), namespace)
    namespace["communication_adaptation_310p"]()
    return distributed, original, subgroup_rank


@pytest.mark.parametrize("async_op", [False, True])
def test_global_broadcast_root_resolves_to_subgroup_rank(async_op):
    distributed, _, subgroup_rank = wrappers()
    group = object()
    tensor = Tensor(-1)
    result = distributed.broadcast(tensor, src=4, group=group, async_op=async_op)
    subgroup_rank.assert_called_once_with(group, 4)
    assert tensor.value == 11
    if async_op:
        result.wait()
    else:
        assert result is None


def test_group_src_is_already_local():
    distributed, _, subgroup_rank = wrappers()
    tensor = Tensor(-1)
    distributed.broadcast(tensor, group=object(), group_src=1)
    subgroup_rank.assert_not_called()
    assert tensor.value == 11


def test_cpu_group_src_is_forwarded_to_original():
    distributed, original, subgroup_rank = wrappers()
    tensor = torch.zeros(1)
    group = object()
    distributed.broadcast(tensor, group=group, group_src=1)
    original.assert_called_once_with(tensor, group_src=1, group=group, async_op=False)
    subgroup_rank.assert_not_called()
