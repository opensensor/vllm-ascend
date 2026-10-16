# SPDX-License-Identifier: Apache-2.0
"""Exact eager activation-to-INT4 packing with the retained activation fallback."""

import torch
import torch.nn.functional as F

from tools.qwen4exp import native_prefill
from tools.qwen4exp import w4_activation_epilogue as epilogue
from tools.qwen4exp.native_descriptor_admission import descriptors_available
from tools.qwen4exp.resident_candidates import qfn_prefill_schedule

MIN_ROWS = 2560
MAX_ROWS = 20480
GROUP_SIZE = 128
METADATA_LANES = 8
LOCAL_EXPERTS = 128


def local_projection_guard(consumer, resource, native_backend):
    """Admit only the retained local-count consumer, including its cache guard.

    Callers must supply the privately qualified consumer and its resource.
    The resident runner serializes calls between admission and projection.
    """

    def eligible(projection, packed, ends):
        if type(projection).native_linear is not consumer:
            return False
        if not qfn_prefill_schedule.eligible(projection, packed, ends, native_backend):
            return False
        rows, device = packed[0].shape[0], packed[0].device
        if native_prefill._capturing(device):
            return False
        experts, outputs, inputs = projection.weight.shape
        key = ((rows, experts, outputs, inputs * 2, 8, 0, 1), device)
        return descriptors_available(resource, (key,), capacity=native_prefill.MAX_CONFIGS, is_capturing=lambda: False)

    return eligible


class W4ActivationPack:
    """Own descriptors and preserve FP16 rounding before activation quantization."""

    def __init__(self, kernel, launch, activation, pack, *, local_kernel=None, local_projection=None):
        if (local_kernel is None) != (local_projection is None):
            raise ValueError("local packing requires both a kernel and a qualified consumer predicate")
        self.kernel, self.launch = kernel, launch
        self.activation, self.pack = activation, pack
        self.local_kernel, self.local_projection = local_kernel, local_projection
        self.local_routes = local_kernel is not None
        self.configs = {}
        self.enabled = self.verify = True
        self.failed = False
        self.failure = None
        self.calls = self.checked = self.mismatches = self.fallbacks = 0
        self.local_calls = self.checked_capacity_rows = self.checked_local_rows = 0
        self.expert_row_histogram = {}

    def _local_eligible(self, projected, projection, packed, ends):
        return (
            self.local_routes
            and isinstance(ends, torch.Tensor)
            and ends.shape == (LOCAL_EXPERTS,)
            and ends.dtype == torch.int64
            and ends.device == projected.device
            and ends.is_contiguous()
            and self.local_projection(projection, packed, ends)
        )

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
        local = self._local_eligible(projected, projection, packed, group_ends)
        operands = [activated, projected, *packed]
        if local:
            operands.append(group_ends)
        operands.append(self.configs[key])
        try:
            self.launch(self.local_kernel if local else self.kernel, operands, epilogue.BLOCKS)
        except BaseException as error:
            self.failed, self.failure = True, error
            raise
        self.calls += 1
        self.local_calls += int(local)
        if self.verify:
            expected = self.pack(self.activation(projected, compute_dtype, params_dtype))
            # Intentional synchronization for shadow qualification only.
            active = rows
            if local:
                # Cold-only route observations also expose expert scheduling waste.
                ends = group_ends.cpu().tolist()
                active = ends[-1]
                self.checked_capacity_rows += rows
                self.checked_local_rows += active
                for begin, end in zip([0, *ends[:-1]], ends):
                    count = end - begin
                    self.expert_row_histogram[count] = self.expert_row_histogram.get(count, 0) + 1
            exact = all(
                torch.equal(a[:active].view(torch.uint8), b[:active].view(torch.uint8))
                for a, b in zip(packed, expected)
            )
            if local:
                actual_output = projection.native_linear(packed, group_ends)
                expected_output = projection.native_linear(expected, group_ends)
                exact = exact and torch.equal(actual_output.view(torch.uint8), expected_output.view(torch.uint8))
            self.checked += 1
            if not exact:
                self.mismatches += 1
                self.enabled = False
            if local:
                return expected_output
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
            local_routes=self.local_routes,
            local_calls=self.local_calls,
            checked_capacity_rows=self.checked_capacity_rows,
            checked_local_rows=self.checked_local_rows,
            expert_row_histogram=dict(self.expert_row_histogram),
        )
