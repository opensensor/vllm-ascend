# SPDX-License-Identifier: Apache-2.0
"""Validate complete-layer options before device allocations or worker launch."""


def enabled(spec):
    value = spec.get("stage_transport", "socket")
    if value not in ("socket", "direct-dma"):
        raise ValueError("unknown complete-layer stage transport")
    native = spec.get("npu_native_bf16_round", False)
    if type(native) is not bool or (native and not spec.get("npu_ops_library")):
        raise ValueError("native BF16 rounding requires an explicit operator library")
    return value == "direct-dma"
