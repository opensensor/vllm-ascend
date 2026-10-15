# SPDX-License-Identifier: Apache-2.0
"""Remove blocking compact-table conversions in the qualified resident runner."""

import hashlib
import inspect
import textwrap
from types import FunctionType

import torch

from vllm_ascend._310p.host_staging import PinnedHostStaging

QUALIFIED_SOURCE_SHA256 = "ab79d707911704f52c65f7e8dcf342792760f3a90d957796ff28fa4489329f21"
PREFIX_COPY = """                block_table.block_table.gpu[:num_reqs].copy_(
                    torch.as_tensor(mapped, device=self.device), non_blocking=True
                )"""
LIVE_COPY = "device_table[:num_reqs].copy_(torch.as_tensor(mapped, device=device_table.device))"


class CompactTableStaging:
    def __init__(self, runner, *, stage_factory=PinnedHostStaging):
        self.runner = runner
        self.enabled = False
        self.copies = 0
        self.stages = {
            index: stage_factory(tuple(table.block_table.gpu.shape), table.block_table.gpu.dtype)
            for index, table in enumerate(runner.input_batch.block_table.block_tables)
            if table.is_mamba_group
        }

    def copy(self, runner, group, mapped, destination):
        if runner is not self.runner:
            raise RuntimeError("compact table staging belongs to another worker")
        source = torch.from_numpy(mapped).to(dtype=destination.dtype)
        self.stages[group].copy_to(source, destination)
        self.copies += 1

    def status(self):
        return dict(enabled=self.enabled, copies=self.copies, groups=sorted(self.stages))


def build_remap(original, resource, *, source=None):
    original = getattr(original, "_qfn_compact_table_base", original)
    source = textwrap.dedent(inspect.getsource(original)) if source is None else source
    if hashlib.sha256(source.encode()).hexdigest() != QUALIFIED_SOURCE_SHA256:
        raise RuntimeError("compact table staging requires the exact qualified remap source")
    if source.count(PREFIX_COPY) != 1 or source.count(LIVE_COPY) != 1:
        raise RuntimeError("qualified compact table copy seams differ")
    body = source.replace(
        PREFIX_COPY, "                _copy(self, group_idx, mapped, block_table.block_table.gpu[:num_reqs])"
    ).replace(LIVE_COPY, "_copy(self, group_idx, mapped, device_table[:num_reqs])")
    factory = "def factory(_copy):\n" + textwrap.indent(body, "    ") + "    return _remap_compact_mamba_block_tables\n"
    namespace = dict(original.__globals__)
    exec(compile(factory, "<qfn-compact-table-staging>", "exec"), namespace)
    compiled = namespace["factory"](resource.copy)
    forward = FunctionType(
        compiled.__code__, original.__globals__, compiled.__name__, compiled.__defaults__, compiled.__closure__
    )

    def remap(runner, *args, **kwargs):
        if resource.enabled:
            return forward(runner, *args, **kwargs)
        return original(runner, *args, **kwargs)

    remap._qfn_compact_table_base = original
    return remap
