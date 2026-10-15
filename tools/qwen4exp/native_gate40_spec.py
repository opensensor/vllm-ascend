# SPDX-License-Identifier: Apache-2.0
"""Immutable build identity; importing this module never imports torch."""

from pathlib import Path

NATIVE_NAME = "qwen_gate40_v1"
BRIDGE_NAMESPACE = "qwen_native_gate40_v1"
PROFILES = ("baseline", "tile40", "stream40")
SYMBOL = "QwenW4A8Int4MatmulV310_f445ca91b09a2d95f71de7992df42bfd_0"
BINARY_BASENAME = "QwenW4A8Int4MatmulV310_f445ca91b09a2d95f71de7992df42bfd.o"


def binary_paths():
    roots = (
        "/srv/ai/src/qwen38-down60-source-20261005/csrc",
        "/srv/ai/src/qwen38-gate-up-tile40-source-20261005/csrc",
        "/srv/ai/src/qwen38-gate-up-stream40-source-20261005/csrc",
    )
    return {
        profile: str(Path(root) / "build/binary/ascend310p/bin/qwen_w4_a8_int4_matmul_v310" / BINARY_BASENAME)
        for profile, root in zip(PROFILES, roots)
    }
