# SPDX-License-Identifier: Apache-2.0
"""Decode-only direct GDN RMSNorm/affine/sigmoid/multiply/cast probe."""

import torch

from tools.qwen4exp.resident_candidates.fused_hc_projection import _is_capturing

MAX_ROWS = 72
HEAD_WIDTH = 128


class DirectGDNNormGate:
    def __init__(self, binary_path):
        self.kernel = torch.classes.qwen_math_next_v1.Kernel(binary_path, "qwen_reduction_probe_v1")
        self.launch = torch.ops.qwen_math_next_v1.launch
        self._geometry = {}

    def __call__(self, x, gamma, gate):
        if x.ndim != 2 or x.shape[1] != HEAD_WIDTH or not 0 < x.shape[0] <= MAX_ROWS:
            raise ValueError("GDN fusion supports 1-72 rows with width 128")
        if x.dtype != torch.float32 or gamma.dtype != torch.float32 or gate.dtype != torch.float16:
            raise ValueError("GDN fusion requires FP32 state/gamma and FP16 gates")
        if gamma.shape != (HEAD_WIDTH,) or gate.shape != x.shape:
            raise ValueError("GDN fusion requires matching gates and one gamma vector")
        if x.device.type != "npu" or gamma.device != x.device or gate.device != x.device:
            raise ValueError("GDN fusion requires contiguous tensors on one NPU")
        if not all(t.is_contiguous() for t in (x, gamma, gate)):
            raise ValueError("GDN fusion requires contiguous inputs")
        key = (x.shape[0], x.device)
        geometry = self._geometry.get(key)
        if geometry is None:
            if _is_capturing(x.device):
                raise RuntimeError("warm up GDN fusion shape before capture")
            geometry = torch.tensor([x.shape[0], HEAD_WIDTH, 3], dtype=torch.int64, device=x.device)
            self._geometry[key] = geometry
        output = torch.empty_like(gate)
        self.launch(self.kernel, [x, gamma, gate, output, geometry], 8)
        return output
