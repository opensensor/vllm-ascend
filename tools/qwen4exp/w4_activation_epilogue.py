# SPDX-License-Identifier: Apache-2.0
"""Exact eager W4 activation epilogue; preserve the existing FP32 SiLU."""

import torch
import torch.nn.functional as F

WIDTH = 640
MIN_ROWS = 2560
MAX_ROWS = 20480
BLOCKS = 8


def _on_npu(device):
    return device.type == "npu"


def _capturing():
    return torch.npu.is_current_stream_capturing()


def _dense_format(value):
    # Lazy device dependency: this module also supports host-only admission tests.
    from torch_npu import get_npu_format

    return get_npu_format(value) == 2


def reference(projected, compute_dtype, params_dtype):
    gate, up = projected.to(compute_dtype).chunk(2, -1)
    return (F.silu(gate) * up).to(params_dtype)


class W4ActivationEpilogue:
    """Owned same-stream launches, with reference-returning serving verification."""

    def __init__(self, kernel, launch):
        self.kernel, self.launch = kernel, launch
        self.configs = {}
        self.enabled = self.verify = True
        self.failed = False
        self.failure = None
        self.calls = self.checked = self.mismatches = self.fallbacks = 0

    def __call__(self, projected, compute_dtype, params_dtype):
        if self.failed:
            raise RuntimeError("W4 activation launch failed; retain native resources and failure frames")
        if (
            not self.enabled
            or compute_dtype != torch.float32
            or params_dtype != torch.float16
            or projected.dtype != torch.float16
            or projected.ndim != 2
            or projected.shape[1] != 2 * WIDTH
            or not MIN_ROWS <= projected.shape[0] <= MAX_ROWS
            or not projected.is_contiguous()
            or not _on_npu(projected.device)
            or torch.is_grad_enabled()
            or _capturing()
            or not _dense_format(projected)
        ):
            self.fallbacks += 1
            return reference(projected, compute_dtype, params_dtype)
        key = (projected.shape[0], projected.device)
        if key not in self.configs:
            # Every admitted row count fits; never evict an in-flight descriptor.
            self.configs[key] = torch.tensor([key[0]], dtype=torch.int64, device=projected.device)
        activated = F.silu(projected[:, :WIDTH].float())
        output = torch.empty((key[0], WIDTH), dtype=torch.float16, device=projected.device)
        try:
            self.launch(self.kernel, [activated, projected, output, self.configs[key]], BLOCKS)
        except BaseException as error:
            self.failed, self.failure = True, error
            raise
        self.calls += 1
        if self.verify:
            expected = reference(projected, compute_dtype, params_dtype)
            # Device synchronization is intentional only during cold shadow gates.
            exact = torch.equal(output.view(torch.uint8), expected.view(torch.uint8))
            self.checked += 1
            if not exact:
                self.mismatches += 1
                self.enabled = False
            return expected
        return output

    def status(self):
        return dict(
            enabled=self.enabled,
            verify=self.verify,
            failed=self.failed,
            calls=self.calls,
            checked=self.checked,
            mismatches=self.mismatches,
            fallbacks=self.fallbacks,
            descriptors=len(self.configs),
            min_rows=MIN_ROWS,
            max_rows=MAX_ROWS,
        )
