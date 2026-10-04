# SPDX-License-Identifier: Apache-2.0
"""Capture a short 310P trace after an unprofiled timing gate."""


def capture_npu_profile(function, path, torch, torch_npu):
    """Record three calls without mixing profiler overhead into timing."""
    with torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
        record_shapes=True,
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(path)),
    ) as profiler:
        for _ in range(3):
            function()
            profiler.step()
        torch.npu.synchronize()
