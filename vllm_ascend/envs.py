#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# This file is mainly Adapted from vllm-project/vllm/vllm/envs.py
# Copyright 2023 The vLLM team.
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

import os
from collections.abc import Callable
from typing import Any

# The begin-* and end* here are used by the documentation generator
# to extract the used env vars.

# begin-env-vars-definition


def _strict_binary_env(name: str, default: str = "0") -> bool:
    value = os.getenv(name, default)
    if value not in {"0", "1"}:
        raise ValueError(f"{name} must be either '0' or '1', got {value!r}")
    return value == "1"


env_variables: dict[str, Callable[[], Any]] = {
    # max compile thread number for package building. Usually, it is set to
    # the number of CPU cores. If not set, the default value is None, which
    # means all number of CPU cores will be used.
    "MAX_JOBS": lambda: os.getenv("MAX_JOBS", None),
    # The build type of the package. It can be one of the following values:
    # Release, Debug, RelWithDebugInfo. If not set, the default value is Release.
    "CMAKE_BUILD_TYPE": lambda: os.getenv("CMAKE_BUILD_TYPE"),
    # Whether to compile custom kernels. If not set, the default value is True.
    # If set to False, the custom kernels will not be compiled.
    # This configuration option should only be set to False when running UT
    # scenarios in an environment without an NPU. Do not set it to False in
    # other scenarios.
    "COMPILE_CUSTOM_KERNELS": lambda: bool(int(os.getenv("COMPILE_CUSTOM_KERNELS", "1"))),
    # The CXX compiler used for compiling the package. If not set, the default
    # value is None, which means the system default CXX compiler will be used.
    "CXX_COMPILER": lambda: os.getenv("CXX_COMPILER", None),
    # The C compiler used for compiling the package. If not set, the default
    # value is None, which means the system default C compiler will be used.
    "C_COMPILER": lambda: os.getenv("C_COMPILER", None),
    # The version of the Ascend chip. It's used for package building.
    # If not set, we will query chip info through `npu-smi`.
    # Please make sure that the version is correct.
    "SOC_VERSION": lambda: os.getenv("SOC_VERSION", None),
    # If set, vllm-ascend will print verbose logs during compilation
    "VERBOSE": lambda: bool(int(os.getenv("VERBOSE", "0"))),
    # The home path for CANN toolkit. If not set, the default value is
    # /usr/local/Ascend/ascend-toolkit/latest
    "ASCEND_HOME_PATH": lambda: os.getenv("ASCEND_HOME_PATH", None),
    # The path for HCCL library, it's used by pyhccl communicator backend. If
    # not set, the default value is libhccl.so.
    "HCCL_SO_PATH": lambda: os.getenv("HCCL_SO_PATH", None),
    # The version of vllm is installed. This value is used for developers who
    # installed vllm from source locally. In this case, the version of vllm is
    # usually changed. For example, if the version of vllm is "0.9.0", but when
    # it's installed from source, the version of vllm is usually set to "0.9.1".
    # In this case, developers need to set this value to "0.9.0" to make sure
    # that the correct package is installed.
    "VLLM_VERSION": lambda: os.getenv("VLLM_VERSION", None),
    # Whether to anbale dynamic EPLB
    "DYNAMIC_EPLB": lambda: os.getenv("DYNAMIC_EPLB", "false").lower(),
    # Control the aclrtMemcpyBatchAsync compile path for KV cache offloading.
    # "1": force enable, "0": force disable, None: auto-detect from CANN headers.
    "VLLM_ASCEND_ENABLE_BATCH_MEMCPY": lambda: os.getenv("VLLM_ASCEND_ENABLE_BATCH_MEMCPY", None),
    # Emit per-layer KVPool ranged transfer audit events. Default: 0 (disabled).
    # Valid values: 0 or 1. This configuration is not sensitive.
    "VLLM_ASCEND_KVPOOL_RANGE_DEBUG": lambda: _strict_binary_env("VLLM_ASCEND_KVPOOL_RANGE_DEBUG"),
    # Override the Unified Buffer (UB) size in KB for Triton kernel tile sizing.
    # 0 (default): auto-detect from device properties, falling back to 192 KB
    # (safe for Ascend 910B/A3). Set to a positive value to override when
    # auto-detection is unavailable or for debugging UB overflow issues.
    "VLLM_ASCEND_ROPE_UB_SIZE_KB": lambda: int(os.getenv("VLLM_ASCEND_ROPE_UB_SIZE_KB") or 0),
    # Experimental: allow Multi-head Latent Attention (MLA) models (e.g. the
    # DeepSeek-lineage GLM-5.x family) to initialize and run on Ascend 310P.
    # MLA is disabled on 310P by default because the optimized front-end kernels
    # (`mla_preprocess`, `npu_mla_prolog_v3` / MlaPrologV3) are compiled only for
    # the STANDARD hardware family (910B/950/A5) and the fused attention core
    # (`npu_fused_infer_attention_score`) has not been validated on ascend310p1.
    # When set to 1 the three hard 310P MLA guards become a gated fall-through so
    # the decomposed / NoPE MLA path (plain projection matmuls + bmm weight
    # absorption + a 310P-supported attention core) can be attempted. This is a
    # bring-up flag only; leave it at 0 (default) for all production 310P runs
    # until MLA has been verified on hardware. Valid values: 0 or 1.
    "VLLM_ASCEND_310P_ENABLE_MLA": lambda: bool(int(os.getenv("VLLM_ASCEND_310P_ENABLE_MLA", "0"))),
    # Fold each RMSNorm into the per-token activation quant of the W8A8 linear
    # that consumes it, using the 310P npu_(add_)rms_norm_dynamic_quant kernels.
    # Correct but off by default: it is free rather than a win. The fused
    # kernel is 0.97x the cost of npu_add_rms_norm + npu_dynamic_quant at
    # hidden 5120 but 1.6x at hidden 512-2048, and on Qwen3.8-27B prefill
    # measures 983.2 against 984.0 tok/s. Unset installs no patch at all.
    # See tools/310p/README.md. Valid values: "0" (default), "1".
    "VLLM_ASCEND_ENABLE_FUSED_NORM_QUANT": lambda: bool(int(os.getenv("VLLM_ASCEND_ENABLE_FUSED_NORM_QUANT", "0"))),
    # Fraction of the profiled free memory that 310P gives to KV and Mamba
    # cache; the remainder is left for operator workspaces, which are large on
    # this SoC. The 0.5 default is what avoids OOM at the default
    # gpu_memory_utilization, but it also caps the servable context, and
    # raising gpu_memory_utilization cannot compensate because the reserve
    # scales with the budget. Raise it only on a box where the real workspace
    # headroom has been measured. Valid range: (0, 1].
    "VLLM_ASCEND_KV_CACHE_FRACTION": lambda: float(os.getenv("VLLM_ASCEND_KV_CACHE_FRACTION", "0.5")),
    # Experimental: quantize the large FLOAT Qwen GDN qkvz projection weights
    # to per-output-channel INT8 at load time on Ascend 310P. The b/a gate and
    # output projections remain FLOAT to limit accuracy risk.
    # Disabled by default until accuracy and throughput have been validated on
    # 310P hardware. Valid values: 0 or 1. This variable is not sensitive.
    "VLLM_ASCEND_310P_GDN_W8A8": lambda: _strict_binary_env("VLLM_ASCEND_310P_GDN_W8A8"),
    # Emit one CPU-only timing summary for each finished request. Default: 0
    # (disabled); valid values: 0 or 1. This variable is not sensitive.
    # Use only when per-request prefill/decode diagnostics justify log volume.
    "VLLM_ASCEND_LOG_REQUEST_TIMINGS": lambda: _strict_binary_env("VLLM_ASCEND_LOG_REQUEST_TIMINGS"),
}

# end-env-vars-definition


def __getattr__(name: str):
    # lazy evaluation of environment variables
    if name in env_variables:
        return env_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(env_variables.keys())
