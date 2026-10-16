# SPDX-License-Identifier: Apache-2.0
"""Offline-only paired projection/finalization candidates; no automatic binding.

An unwritten projection tail is private to this pair. Never return it to a
caller that can select the generic finalizer after the producer has run.
The fused reductions have explicit arithmetic orders, not established CANN
parity. Resource construction and CPU tests confer no hardware qualification.
"""

import torch

from tools.qwen4exp import native_prefill, route_finalize
from tools.qwen4exp.native_descriptor_admission import descriptors_available
from tools.qwen4exp.resident_candidates import qfn_prefill_schedule
from tools.qwen4exp.w4_activation_pack import W4ActivationPack

WIDTH = 2560
INPUT_WIDTH = 640
TOP_K = 10
EXPERTS = 128
MIN_ROUTES = 128
MAX_ROUTES = 20480
MAX_CONFIGS = 256
BLOCKS = 8
MODES = ("ordered", "sequential", "pairwise")
ENTRY_POINTS = (
    "qwen_local_route_finalize_v1",
    "qwen_local_route_reduce_sequential_v1",
    "qwen_local_route_reduce_pairwise_v1",
)


def metadata_valid(rows, device, inverse, weights, ends):
    return (
        MIN_ROUTES < rows <= MAX_ROUTES
        and rows % TOP_K == 0
        and isinstance(inverse, torch.Tensor)
        and isinstance(weights, torch.Tensor)
        and isinstance(ends, torch.Tensor)
        and inverse.shape == (rows,)
        and weights.shape == (rows, 1)
        and ends.shape == (EXPERTS,)
        and inverse.dtype == ends.dtype == torch.int64
        and weights.dtype == torch.float32
        and all(t.device == device and t.is_contiguous() for t in (inverse, weights, ends))
    )


class LocalRouteEpilogue:
    """Own one explicitly selected private kernel and append-only descriptors."""

    def __init__(self, kernel, launch, *, mode="ordered"):
        if mode not in MODES:
            raise ValueError("select ordered, sequential or pairwise epilogue")
        self.kernel, self.launch, self.mode = kernel, launch, mode
        self._configs = {}

    def __call__(self, routed, inverse, weights, ends):
        if (
            routed.ndim != 2
            or routed.shape[1] != WIDTH
            or routed.dtype != torch.float16
            or not routed.is_contiguous()
            or not route_finalize._supports_device(routed.device)
            or not metadata_valid(routed.shape[0], routed.device, inverse, weights, ends)
        ):
            raise ValueError("unsupported private local-route epilogue metadata")
        if torch.is_grad_enabled() or route_finalize._is_capturing():
            raise RuntimeError("local-route experiments require eager inference")
        rows, device = routed.shape[0], routed.device
        key = (rows, device)
        if not descriptors_available(self, (key,), capacity=MAX_CONFIGS, is_capturing=lambda: False):
            raise RuntimeError("local-route descriptor capacity reached")
        if key not in self._configs:
            self._configs[key] = torch.tensor([rows], dtype=torch.int64, device=device)
        shape = (rows if self.mode == "ordered" else rows // TOP_K, WIDTH)
        output = torch.empty(shape, dtype=torch.float32, device=device)
        self.launch(self.kernel, [routed, inverse, weights, ends, output, self._configs[key]], BLOCKS)
        return output


class LocalDownEpilogue:
    """Keep the no-zero producer and its local-aware consumer indivisible.

    Preflight both descriptor pools and every metadata contract before producer
    submission. There is no fallback after submission. On failure, retain all
    resources and tensors and forbid reuse. This is an experiment API, with no
    live model patch or loader; a controller must qualify it before integration.
    """

    def __init__(self, projection, epilogue, *, native_backend):
        self.projection, self.epilogue, self.native_backend = projection, epilogue, native_backend
        self.failed = False
        self.failure = None
        self.retained = ()
        self.calls = 0

    def can_run(self, bank, prepared, ends, inverse, weights):
        if self.failed or torch.is_grad_enabled():
            return False
        if not qfn_prefill_schedule.eligible(bank, prepared, ends, self.native_backend):
            return False
        if bank.weight.shape != (EXPERTS, WIDTH, INPUT_WIDTH // 2):
            return False
        rows, device = prepared[0].shape[0], prepared[0].device
        if not metadata_valid(rows, device, inverse, weights, ends) or route_finalize._is_capturing():
            return False
        projection_key = ((rows, EXPERTS, WIDTH, INPUT_WIDTH, 8, 0, 1), device)
        return descriptors_available(
            self.projection, (projection_key,), capacity=native_prefill.MAX_CONFIGS, is_capturing=lambda: False
        ) and descriptors_available(self.epilogue, ((rows, device),), capacity=MAX_CONFIGS, is_capturing=lambda: False)

    def __call__(self, bank, prepared, ends, inverse, weights):
        if self.failed:
            raise RuntimeError("local pair failed; preserve resources and failure frames")
        if not self.can_run(bank, prepared, ends, inverse, weights):
            raise ValueError("local pair not admitted; use the complete retained path before submitting")
        self.retained = (bank, prepared, ends, inverse, weights)
        try:
            routed = self.projection(bank, prepared, ends)
            self.retained += (routed,)
            output = self.epilogue(routed, inverse, weights, ends)
            if self.epilogue.mode == "ordered":
                # Variant 2 preserves the original backend's FP32 reduction.
                output = output.reshape(-1, TOP_K, WIDTH).sum(1)
        except BaseException as error:
            self.failed, self.failure = True, error
            raise
        self.retained = ()
        self.calls += 1
        return output


class _PairedProjection:
    """Private packing consumer; every return value is already reduced."""

    def __init__(self, adapter, bank, inverse, weights):
        self.adapter, self.bank, self.inverse, self.weights = adapter, bank, inverse, weights
        self.local_admitted = False

    def admit(self, prepared, ends):
        self.local_admitted = self.adapter.pair.can_run(self.bank, prepared, ends, self.inverse, self.weights)
        return self.local_admitted

    def native_linear(self, prepared, ends):
        if self.local_admitted:
            # Admission occurred before packing skipped its tail. Do not fall
            # back now, even if a later guard or submission fails.
            return self.adapter.pair(self.bank, prepared, ends, self.inverse, self.weights)
        return self.adapter.finalize(self.bank.native_linear(prepared, ends), self.inverse, self.weights)

    def grouped_linear(self, activated, ends):
        return self.adapter.finalize(self.bank.grouped_linear(activated, ends), self.inverse, self.weights)


def _admit_paired_projection(projection, prepared, ends):
    return isinstance(projection, _PairedProjection) and projection.admit(prepared, ends)


class PackedLocalDownEpilogue:
    """Offline integration seam retaining existing local activation packing.

    Disabled by default. Own a separate pack adapter, borrowing immutable
    kernels and the retained activation fallback. A future model binding would
    replace the down-projection/finalization/sum block with this one call.
    Whole-pipeline shadows compare with the original pack/projection/finalizer
    and backend sum, so a fused reduction cannot validate itself.
    """

    def __init__(self, pair, retained_pack, retained_finalizer):
        if retained_pack.failed or retained_pack.mismatches or retained_pack.local_kernel is None:
            raise ValueError("require the retained healthy local activation pack")
        self.pair, self.retained_pack, self.retained_finalizer = pair, retained_pack, retained_finalizer
        self.packer = W4ActivationPack(
            retained_pack.kernel,
            retained_pack.launch,
            retained_pack.activation,
            retained_pack.pack,
            local_kernel=retained_pack.local_kernel,
            local_projection=_admit_paired_projection,
        )
        # Packed limbs already have their independent gate. The shadow below
        # checks the complete result against the actual retained reduction.
        self.packer.verify = False
        self.enabled, self.verify, self.failed = False, True, False
        self.calls = self.checked = self.mismatches = 0
        self.failure = None
        self.retained = ()

    def finalize(self, routed, inverse, weights):
        return self.retained_finalizer(routed, inverse, weights).reshape(-1, TOP_K, WIDTH).sum(1)

    def baseline(self, projected, bank, ends, inverse, weights, compute_dtype, params_dtype):
        routed = self.retained_pack(projected, bank, ends, compute_dtype, params_dtype)
        return self.finalize(routed, inverse, weights)

    def __call__(self, projected, bank, ends, inverse, weights, compute_dtype, params_dtype):
        if self.failed:
            raise RuntimeError("packed local pair failed; preserve resources and failure frames")
        args = (projected, bank, ends, inverse, weights, compute_dtype, params_dtype)
        if not self.enabled or torch.is_grad_enabled() or route_finalize._is_capturing():
            return self.baseline(*args)
        projection = _PairedProjection(self, bank, inverse, weights)
        self.retained = (projection, args)
        try:
            actual = self.packer(projected, projection, ends, compute_dtype, params_dtype)
            self.calls += 1
            if self.verify:
                expected = self.baseline(*args)
                self.checked += 1
                # Explicit qualification-only synchronization, never a normal
                # serving-path readback or a comparison against our own tree.
                if not torch.equal(actual.view(torch.int32), expected.view(torch.int32)):
                    self.mismatches += 1
                    self.enabled = False
                actual = expected
        except BaseException as error:
            self.failed, self.failure = True, error
            raise
        self.retained = ()
        return actual


def reference_weighted(routed, inverse, weights, active):
    """CPU oracle: synthesize producer +0 without ever selecting its bad tail."""
    if routed.device.type != "cpu" or inverse.device.type != "cpu" or weights.device.type != "cpu":
        raise ValueError("offline reference requires CPU tensors")
    if not 0 <= active <= routed.shape[0] or bool(((inverse < 0) | (inverse >= routed.shape[0])).any()):
        raise ValueError("invalid active boundary or inverse index")
    result = torch.zeros(routed.shape, dtype=torch.float32)
    local = inverse < active
    result[local] = routed[inverse[local]].float()
    return result * weights


def reference_reduce(weighted, *, mode):
    """Explicit FP32 orders; neither is assumed equal to NPU torch.sum."""
    if mode not in ("sequential", "pairwise"):
        raise ValueError("select an explicit fused reduction order")
    if weighted.device.type != "cpu" or weighted.dtype != torch.float32:
        raise ValueError("offline reduction requires CPU FP32 operands")
    values = weighted.reshape(-1, TOP_K, WIDTH)
    if mode == "sequential":
        output = torch.zeros_like(values[:, 0])
        for term in range(TOP_K):
            output = output + values[:, term]
        return output
    terms = list(values.unbind(1))
    while len(terms) > 1:
        terms = [terms[i] + terms[i + 1] if i + 1 < len(terms) else terms[i] for i in range(0, len(terms), 2)]
    return terms[0]


def traffic_bytes(rows, active):
    """Logical payload counts only: excludes metadata, caches and actual DMA."""
    if type(rows) is not int or type(active) is not int or not 0 <= active <= rows <= MAX_ROUTES or rows % TOP_K:
        raise ValueError("invalid route capacity or active count")
    return dict(
        omitted_projection_zero_writes=(rows - active) * WIDTH * 2,
        omitted_epilogue_tail_reads=(rows - active) * WIDTH * 2,
        eliminated_fp32_intermediate=rows * WIDTH * 4,
        eliminated_intermediate_write_and_read=rows * WIDTH * 4 * 2,
        final_fp32_output=rows // TOP_K * WIDTH * 4,
    )
