# SPDX-License-Identifier: Apache-2.0
"""Temporary worker instrumentation for the repeated 16.6K prefill fault."""

import functools
import os
from pathlib import Path

import torch
from live_score_extension import GlmScoreResidentExtension
from vllm.forward_context import get_forward_context

from tools.glm_perf.trace_device_ops import DeviceOpTrace
from vllm_ascend.models.glm5next.model import Glm5NextModel
from vllm_ascend.models.glm5next.mtp import Glm5NextMultiTokenPredictor

TRACE_START_TOKENS = 16000
TRACE_STOP_TOKENS = 17920
TRACE_ROOT = Path("/home/matteius/experiments/glm-kpool-live-score-20261005/boundary-isolation")


def traced_forward(original, label):
    @functools.wraps(original)
    def forward(self, *args, **kwargs):
        metadata = get_forward_context().attn_metadata
        lengths = []
        if isinstance(metadata, dict):
            for value in metadata.values():
                if getattr(value, "cache_role", None) != "indexer" or getattr(value, "num_actual_tokens", 0) <= 8:
                    continue
                pools = getattr(value, "seq_lens_cpu", None)
                if pools is not None and pools.device.type == "cpu" and pools.numel():
                    lengths.append(int(pools.max()) * value.compress_ratio)
        if not lengths or not TRACE_START_TOKENS <= max(lengths) <= TRACE_STOP_TOKENS:
            return original(self, *args, **kwargs)
        if torch.npu.is_current_stream_capturing():
            raise RuntimeError("Boundary tracing must not run during graph capture")
        with (TRACE_ROOT / f"ops-{os.getpid()}.jsonl").open("a", buffering=1) as output:
            trace = DeviceOpTrace(output, torch.npu.synchronize)
            trace.emit(event="forward", model=label, max_tokens=max(lengths))
            with trace:
                return original(self, *args, **kwargs)

    return forward


Glm5NextModel.forward = traced_forward(Glm5NextModel.forward, "target")
Glm5NextMultiTokenPredictor.forward = traced_forward(Glm5NextMultiTokenPredictor.forward, "draft")


class GlmBoundaryTraceExtension(GlmScoreResidentExtension):
    pass
