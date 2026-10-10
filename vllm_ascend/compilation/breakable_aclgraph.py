#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, fields
from typing import Any

import torch
from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphWrapper
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.forward_context import BatchDescriptor, get_forward_context, is_forward_context_available

from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.compilation.acl_graph import (
    get_draft_graph_params,
    get_draft_graph_prefill_params,
    get_graph_params,
    weak_ref_workspaces,
)


@dataclass(frozen=True)
class _DraftStepBatchDescriptor(BatchDescriptor):
    """Different draft steps bind different persistent attention buffers."""

    draft_step: int = 0


class BreakableACLGraphWrapper(BreakableCUDAGraphWrapper):
    def __init__(
        self,
        runnable: Callable[..., Any],
        vllm_config: VllmConfig,
        use_eagle: bool = False,
        enable_enpu: bool = False,
    ) -> None:
        super().__init__(
            runnable=runnable,
            vllm_config=vllm_config,
        )

        self.use_eagle = use_eagle
        self.enable_enpu = enable_enpu

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if not is_forward_context_available() or not _EXTRA_CTX.is_draft_model:
            return super().__call__(*args, **kwargs)
        context = get_forward_context()
        if context.cudagraph_runtime_mode == CUDAGraphMode.NONE:
            return super().__call__(*args, **kwargs)
        descriptor = context.batch_descriptor
        steps = getattr(context, "draft_attn_metadatas", None)
        if descriptor is None or not steps:
            return super().__call__(*args, **kwargs)
        step = next((index for index, metadata in enumerate(steps) if metadata is context.attn_metadata), None)
        if step is None:
            return super().__call__(*args, **kwargs)
        # Per-model capture encloses one proposal step, not the Python draft
        # loop. Equal token buckets therefore do not imply equal slot-mapping
        # or query-boundary addresses. Give each step its own graph entry.
        context.batch_descriptor = _DraftStepBatchDescriptor(
            **{field.name: getattr(descriptor, field.name) for field in fields(BatchDescriptor)},
            draft_step=step,
        )
        try:
            return super().__call__(*args, **kwargs)
        finally:
            context.batch_descriptor = descriptor

    def _capture(
        self,
        entry: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        forward_context = get_forward_context()
        is_full_capture = forward_context.cudagraph_runtime_mode == CUDAGraphMode.FULL
        if is_full_capture:
            # Ascend FULL graph attention creates task groups and records the
            # mutable graph parameters only while this flag is set.
            forward_context.capturing = True

        output = super()._capture(entry, args, kwargs)

        if is_full_capture:
            # Keep the same workspace lifetime contract as ACLGraphWrapper.
            weak_ref_workspaces(get_graph_params())
            weak_ref_workspaces(get_draft_graph_params())
            weak_ref_workspaces(get_draft_graph_prefill_params())

        return output

    def _replay(
        self,
        entry: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        forward_context = get_forward_context()
        if forward_context.cudagraph_runtime_mode == CUDAGraphMode.FULL:
            # Match ACLGraphWrapper's ordering between async attention
            # parameter updates and the previous/current FULL graph replay.
            is_draft_eagle = _EXTRA_CTX.is_draft_model and self.use_eagle
            if not self.enable_enpu and not is_draft_eagle:
                ready_event = getattr(forward_context, "_ascend_replay_ready_event", None)
                if ready_event is not None:
                    # The owning 310P runner installed an exact completion
                    # boundary for input staging and graph parameter updates.
                    ready_event.synchronize()
                else:
                    torch.npu.current_stream().synchronize()
        super()._replay(entry, args, kwargs)
        return entry.output
