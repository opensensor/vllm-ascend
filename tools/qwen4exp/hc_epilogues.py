# SPDX-License-Identifier: Apache-2.0
"""Private eager HC fusions; activation and FP32 normalization remain unchanged."""

import torch
import torch.nn.functional as F

WIDTH = 2560
STREAMS = 4
MIN_ROWS = 256
MIN_AFFINE_ROWS = 1024
MAX_ROWS = 2048
# Cover every admitted row count without evicting a device descriptor.
MAX_CONFIGS = MAX_ROWS
BLOCKS = 8


def _on_npu(device):
    return device.type == "npu"


def _capturing():
    return torch.npu.is_current_stream_capturing()


class HCEpilogues:
    """Use an owned native bridge; never admit these calls during graph capture."""

    def __init__(self, affine_kernel, mean_kernel, launch):
        self.affine_kernel, self.mean_kernel, self.launch = affine_kernel, mean_kernel, launch
        self.configs = {}
        self.failed = False
        self.failure = None

    def _config(self, value):
        if self.failed:
            raise RuntimeError("HC epilogue failed; retain native resources and failure frames")
        if (
            value.ndim != 2
            or value.shape[1] != STREAMS * WIDTH
            or not 0 < value.shape[0] <= MAX_ROWS
            or value.dtype != torch.float32
            or not _on_npu(value.device)
            or not value.is_contiguous()
        ):
            raise ValueError("HC epilogues require contiguous [1..2048,10240] FP32 NPU inputs")
        if _capturing():
            raise RuntimeError("HC epilogues are eager-only")
        key = (value.shape[0], value.device)
        if key not in self.configs:
            if len(self.configs) >= MAX_CONFIGS:
                raise RuntimeError("HC epilogue descriptor capacity reached")
            self.configs[key] = torch.tensor([key[0]], device=value.device, dtype=torch.int64)
        return self.configs[key]

    def _launch(self, kernel, tensors):
        try:
            self.launch(kernel, tensors, BLOCKS)
        except BaseException as error:
            self.failed, self.failure = True, error
            raise

    def affine(self, normalized, weight):
        config = self._config(normalized)
        if (
            weight.shape != (STREAMS * WIDTH,)
            or weight.dtype != torch.float32
            or weight.device != normalized.device
            or not weight.is_contiguous()
        ):
            raise ValueError("HC affine requires a contiguous FP32 channel vector on the input device")
        full = torch.empty_like(normalized)
        narrow = torch.empty_like(normalized, dtype=torch.float16)
        self._launch(self.affine_kernel, [normalized, weight, full, narrow, config])
        return full, narrow

    def mean(self, normalized, gate):
        config = self._config(normalized)
        if (
            gate.shape != normalized.shape
            or gate.dtype != torch.float16
            or gate.device != normalized.device
            or not gate.is_contiguous()
        ):
            raise ValueError("HC mean requires matching contiguous FP16 gates on the input device")
        output = torch.empty((normalized.shape[0], WIDTH), dtype=torch.float16, device=normalized.device)
        self._launch(self.mean_kernel, [normalized, gate, output, config])
        return output


def eligible(module, values):
    return (
        not torch.is_grad_enabled()
        and _on_npu(values.device)
        and values.ndim == 2
        and MIN_ROWS <= values.shape[0] <= MAX_ROWS
        and values.shape[1] == STREAMS * WIDTH
        and values.dtype == torch.float16
        and values.is_contiguous()
        and module.hc_count == STREAMS
        and module.hidden_size == WIDTH
        and module.params_dtype == torch.float16
        and module.compute_dtype == torch.float32
        and module.use_combine
        and module.input_mix_weight_down.dtype == module.input_mix_weight_up.dtype == torch.float16
        and module.block_inject_weight.dtype == torch.float16
        and module._rms_unit_weight is not None
    )


def make_mix(owner, original, *, affine=True, mean=True, affine_min_rows=MIN_AFFINE_ROWS, share_operand=False):
    def mix(module, values):
        if not eligible(module, values) or _capturing():
            return original(module, values)
        # Lazy imports keep CPU admission tests independent of model registration.
        import torch_npu

        from vllm_ascend.models.qwen4_exp.model import _linear

        if affine and values.shape[0] >= affine_min_rows:
            module.prepare_norm_affine()
            unit, _ = torch_npu.npu_rms_norm(values.float().view(-1, WIDTH), module._rms_unit_weight, module.eps)
            normalized, operand = owner.affine(unit.view(values.shape), module._hc_norm_affine_cache[1])
            # The owned eager bridge has submitted on this allocation's stream.
            # Do not keep the large RMS intermediate alive through both projections.
            del unit
        else:
            normalized = module._normalize(values)
            operand = normalized.half()
        gate = F.silu(_linear(operand, module.input_mix_weight_down, module.compute_dtype) / module.hc_count)
        gate = torch.sigmoid(_linear(gate, module.input_mix_weight_up, module.compute_dtype))
        if mean:
            mixed = owner.mean(normalized, gate)
        else:
            mixed = (gate.view(-1, STREAMS, WIDTH) * normalized.view(-1, STREAMS, WIDTH)).mean(1).half()
        save_operand = share_operand or getattr(module, "share_projection_operand", False)
        return mixed, (values, operand if save_operand else normalized)

    return mix


class HCEpilogueAdapter:
    """Verification returns the reference and disables on a numerical mismatch."""

    def __init__(self, owner, original, *, share_operand=False):
        self.owner, self.original = owner, original
        self.share_operand = share_operand
        self.candidate = make_mix(owner, original, share_operand=share_operand)
        self.enabled = True
        self.verify = True
        self.calls = self.checked = self.mismatches = 0

        def replacement(module, values):
            if not self.enabled or not eligible(module, values) or _capturing():
                return self.original(module, values)
            self.calls += 1
            actual = self.candidate(module, values)
            if not self.verify:
                return actual
            expected = self.original(module, values)
            self.checked += 1
            # The combine consumer projects in FP16. Sharing that same narrowed
            # operand must preserve its exact values, not the FP32 storage dtype.
            expected_saved = expected[1][1].half() if self.share_operand else expected[1][1]
            if not all(
                torch.equal(left.view(torch.uint8), right.view(torch.uint8))
                for left, right in ((actual[0], expected[0]), (actual[1][1], expected_saved))
            ):
                self.mismatches += 1
                self.enabled = False
            return expected

        self.replacement = replacement

    def status(self):
        return dict(
            enabled=self.enabled,
            verify=self.verify,
            calls=self.calls,
            checked=self.checked,
            mismatches=self.mismatches,
            failed=self.owner.failed,
            min_rows=MIN_ROWS,
            min_affine_rows=MIN_AFFINE_ROWS,
            share_operand=self.share_operand,
        )
