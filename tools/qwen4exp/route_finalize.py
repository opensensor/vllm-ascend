# SPDX-License-Identifier: Apache-2.0
"""Append-only route finalization prototype; reduction and W4 math unchanged."""

import torch

WIDTH = 2560
TOP_K = 10
MAX_ROUTES = 25600
MAX_CONFIGS = 256
MIN_VALIDATION_CASES = 3


def _supports_device(device):
    return device.type == "npu"


def _is_capturing():
    import torch_npu

    return torch_npu.npu.is_current_stream_capturing()


class RouteFinalizer:
    """Own native resources and immutable shape descriptors across graph replay."""

    def __init__(self, kernel, launch):
        self.kernel = kernel
        self.launch = launch
        self._configs = {}

    def __call__(self, routed, inverse, weights):
        if routed.ndim != 2 or routed.shape[1] != WIDTH:
            raise ValueError("route finalizer requires width 2560")
        rows = routed.shape[0]
        if not 0 < rows <= MAX_ROUTES or rows % TOP_K:
            raise ValueError("route finalizer requires 1..2560 tokens with top-k 10")
        if inverse.shape != (rows,) or weights.shape != (rows, 1):
            raise ValueError("inverse indices and original-order weights must match routed rows")
        if routed.dtype != torch.float16 or inverse.dtype != torch.int64 or weights.dtype != torch.float32:
            raise ValueError("route finalizer requires FP16 output, INT64 inverse, and FP32 weights")
        if not _supports_device(routed.device) or any(t.device != routed.device for t in (inverse, weights)):
            raise ValueError("route finalizer requires tensors on one NPU")
        if not all(t.is_contiguous() for t in (routed, inverse, weights)):
            raise ValueError("route finalizer requires contiguous tensors")
        key = (rows, routed.device)
        config = self._configs.get(key)
        if config is None:
            if _is_capturing():
                raise RuntimeError("warm up each route shape before capture")
            if len(self._configs) >= MAX_CONFIGS:
                raise RuntimeError("route finalizer descriptor capacity reached")
            config = torch.tensor([rows], dtype=torch.int64, device=routed.device)
            self._configs[key] = config
        output = torch.empty(routed.shape, dtype=torch.float32, device=routed.device)
        self.launch(self.kernel, [routed, inverse, weights, output, config], 8)
        return output


def baseline(routed, order, inverse, weights):
    """Exact qualified operations before the unchanged route sum."""
    output = routed.float()
    output *= weights.index_select(0, order)
    return output.index_select(0, inverse)


def simulation(routed, inverse, weights):
    """CPU oracle for the fused operation, never used for serving."""
    return routed.index_select(0, inverse).float() * weights


def validate(resource, device, *, rows=(10, 650, 6400, 20480), repeats=3):
    """Explicit resident-only private-tensor correctness gate; no model mutation.

    Run after native loading, never at import. Require bit-exact weighted rows
    and the original FP32 sum for changing permutations, weights and peer tails.
    At least three cases cover all-peer, mixed local/peer and all-local rows,
    even if repeats is one. Larger repeats add changing mixed permutations.
    Peak memory is bounded to one case; the largest case needs about 0.9 GiB.
    """
    if type(repeats) is not int or not 1 <= repeats <= 5:
        raise ValueError("validation repeats must be within 1..5")
    if not rows or any(type(count) is not int or not 0 < count <= MAX_ROUTES or count % TOP_K for count in rows):
        raise ValueError("validation rows require nonempty supported top-k capacities")
    results = []
    generator = torch.Generator().manual_seed(417)
    for count in rows:
        for step in range(max(MIN_VALIDATION_CASES, repeats)):
            routed = torch.randn(count, WIDTH, generator=generator, dtype=torch.float16)
            # Vary local/peer boundaries, including all-peer output.
            local = 0 if step == 0 else count if step == 2 else count // (step + 1)
            routed[local:] = 0
            order = torch.randperm(count, generator=generator)
            inverse = torch.argsort(order)
            weights = torch.rand(count, 1, generator=generator, dtype=torch.float32)
            # Signed zero, subnormals and large finite values exercise the cast.
            if local:
                routed[0, :4] = torch.tensor([-0.0, 2**-24, -(2**-24), 65504], dtype=torch.float16)
            routed, order, inverse, weights = (t.to(device) for t in (routed, order, inverse, weights))
            expected = baseline(routed, order, inverse, weights)
            actual = resource(routed, inverse, weights)
            weighted_equal = torch.equal(actual.view(torch.int32), expected.view(torch.int32))
            reduced_equal = torch.equal(
                actual.reshape(-1, TOP_K, WIDTH).sum(1).view(torch.int32),
                expected.reshape(-1, TOP_K, WIDTH).sum(1).view(torch.int32),
            )
            results.append(
                {
                    "rows": count,
                    "step": step,
                    "local_rows": local,
                    "weighted_exact": weighted_equal,
                    "sum_exact": reduced_equal,
                }
            )
            del routed, order, inverse, weights, expected, actual
    return {"passed": all(r["weighted_exact"] and r["sum_exact"] for r in results), "cases": results}
