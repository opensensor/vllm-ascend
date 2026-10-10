# SPDX-License-Identifier: Apache-2.0
"""Queued isolated-device gates; no library load without both explicit options."""

import subprocess
from pathlib import Path

import pytest

from tools.qwen4exp.build_streaming import verify_bundle
from tools.qwen4exp.thermal_controller import parse_temperatures

STANDALONE_STOP_C = 90


def pytest_addoption(parser):
    parser.addoption("--qwen-streaming-next-bundle", type=Path)
    parser.addoption("--qwen-streaming-next-execute", action="store_true", default=False)


@pytest.fixture(scope="module")
def projection(request):
    bundle = request.config.getoption("--qwen-streaming-next-bundle")
    execute = request.config.getoption("--qwen-streaming-next-execute")
    if bundle is None or not execute:
        pytest.skip("requires verified v2 bundle and explicit isolated NPU execution")
    candidate = verify_bundle(bundle, require_compiled=True)
    if candidate["configuration"].get("projection_variant") != "m32n160_v2":
        pytest.fail("v2 projection bundle required")
    temperatures = parse_temperatures(subprocess.check_output(["npu-smi", "info"], text=True, timeout=3), 6)
    if max(temperatures) >= STANDALONE_STOP_C:
        pytest.fail("standalone thermal limit reached")
    # Worker-isolated imports happen only after explicit admission. The caller
    # supplies ASCEND_RT_VISIBLE_DEVICES for a granted device, never the server.
    import torch
    import torch_npu

    from tools.qwen4exp.native_streaming_next import NativeStreamingProjectionNext

    torch.npu.set_device(0)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    torch.ops.load_library(str(bundle / candidate["bridges"][0]["path"]))
    kernel = getattr(torch.classes, candidate["namespace"]).Kernel
    launch = getattr(torch.ops, candidate["namespace"]).launch
    binary = str(bundle / "binaries/native_streaming_next.bin")
    return NativeStreamingProjectionNext(
        kernel(binary, "qwen_streaming_projection_v2"), launch, kernel(binary, "qwen_streaming_columns_v2")
    )


@pytest.fixture(autouse=True)
def temperature_admission(request):
    if request.config.getoption("--qwen-streaming-next-execute"):
        temps = parse_temperatures(subprocess.check_output(["npu-smi", "info"], text=True, timeout=3), 6)
        if max(temps) >= STANDALONE_STOP_C:
            pytest.fail("standalone thermal limit reached")


@pytest.fixture
def gdn_operators(request):
    bundle = request.config.getoption("--qwen-streaming-next-bundle")
    if bundle is None or not request.config.getoption("--qwen-streaming-next-execute"):
        pytest.skip("requires explicit isolated NPU execution and a verified bundle")
    verify_bundle(bundle, require_compiled=True)
    # The caller supplies the coherent FP32-state OPP and an isolated device.
    import torch
    import torch_npu

    from vllm_ascend.utils import enable_custom_op

    torch.npu.set_device(0)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    enable_custom_op()
    return torch
