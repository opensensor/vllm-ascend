# SPDX-License-Identifier: Apache-2.0
"""Exact eager activation-to-INT4 packing with the retained activation fallback."""

import torch
import torch.nn.functional as F

from tools.qwen4exp import w4_activation_epilogue as epilogue

MIN_ROWS = 2560
MAX_ROWS = 20480
GROUP_SIZE = 128
METADATA_LANES = 8


class W4ActivationPack:
    """Own descriptors and preserve FP16 rounding before activation quantization."""

    def __init__(self, kernel, launch, activation, pack):
        self.kernel, self.launch = kernel, launch
        self.activation, self.pack = activation, pack
        self.configs = {}
        self.enabled = self.verify = True
        self.failed = False
        self.failure = None
        self.calls = self.checked = self.mismatches = self.fallbacks = 0

    def __call__(self, projected, projection, group_ends, compute_dtype, params_dtype):
        if self.failed:
            raise RuntimeError("W4 packing launch failed; retain native resources and failure frames")
        if (
            not self.enabled
            or compute_dtype != torch.float32
            or params_dtype != torch.float16
            or projected.dtype != torch.float16
            or projected.ndim != 2
            or projected.shape[1] != 2 * epilogue.WIDTH
            or not MIN_ROWS <= projected.shape[0] <= MAX_ROWS
            or not projected.is_contiguous()
            or not epilogue._on_npu(projected.device)
            or torch.is_grad_enabled()
            or epilogue._capturing()
            or not epilogue._dense_format(projected)
        ):
            self.fallbacks += 1
            return projection.grouped_linear(self.activation(projected, compute_dtype, params_dtype), group_ends)
        rows, device = projected.shape[0], projected.device
        key = (rows, device)
        if key not in self.configs:
            self.configs[key] = torch.tensor([rows], dtype=torch.int64, device=device)
        activated = F.silu(projected[:, : epilogue.WIDTH].float())
        packed = (
            torch.empty((rows, epilogue.WIDTH // 2), dtype=torch.int8, device=device),
            torch.empty((rows, epilogue.WIDTH // 2), dtype=torch.int8, device=device),
            torch.empty((rows, epilogue.WIDTH // GROUP_SIZE, METADATA_LANES), dtype=torch.float32, device=device),
            torch.empty((rows, epilogue.WIDTH // GROUP_SIZE, METADATA_LANES), dtype=torch.float32, device=device),
        )
        try:
            self.launch(self.kernel, [activated, projected, *packed, self.configs[key]], epilogue.BLOCKS)
        except BaseException as error:
            self.failed, self.failure = True, error
            raise
        self.calls += 1
        if self.verify:
            expected = self.pack(self.activation(projected, compute_dtype, params_dtype))
            # Intentional synchronization for shadow qualification only.
            exact = all(torch.equal(a.view(torch.uint8), b.view(torch.uint8)) for a, b in zip(packed, expected))
            self.checked += 1
            if not exact:
                self.mismatches += 1
                self.enabled = False
            packed = expected
        return projection.native_linear(packed, group_ends)

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
