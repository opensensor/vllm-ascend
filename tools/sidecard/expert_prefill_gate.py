# SPDX-License-Identifier: Apache-2.0
"""Real-bank prefill graph qualification before external DMA ownership exists."""

import hashlib
import json

from .layer_weights import MAX_CHUNK, TOP_K, WIDTH


def validate(eager, captured, torch, *, output, device=None):
    report = dict(passed=False, rank=eager.rank, rows=MAX_CHUNK, cases=[])
    try:
        for layer in eager.banks:
            torch.manual_seed(20261012 + layer)
            hidden = torch.randn(MAX_CHUNK, WIDTH).half().to(device or f"npu:{eager.rank}")
            ids = (torch.arange(MAX_CHUNK * TOP_K).reshape(MAX_CHUNK, TOP_K) * 17) % 144 + eager.rank * 144
            weights = torch.full((MAX_CHUNK, TOP_K), 2.5 / TOP_K)
            # A zero-route replay follows active work to detect stale outputs.
            for name in ("local", "mixed_peer", "zero_weights"):
                selected = ids.clone()
                route_weights = weights.clone()
                if name == "mixed_peer":
                    selected[:, ::2] = (selected[:, ::2] + 144) % 288
                if name == "zero_weights":
                    route_weights.zero_()
                values = hidden, selected.to(hidden.device), route_weights.to(hidden.device)
                baseline = eager(layer, *values).cpu()
                candidate = captured(layer, *values).cpu()
                equal = torch.equal(baseline, candidate)
                report["cases"].append(
                    dict(
                        layer=layer,
                        routes=name,
                        bitwise_equal=equal,
                        max_abs_error=float((baseline - candidate).abs().max()),
                        baseline_sha256=hashlib.sha256(baseline.numpy().tobytes()).hexdigest(),
                        captured_sha256=hashlib.sha256(candidate.numpy().tobytes()).hexdigest(),
                    )
                )
                if not equal:
                    raise AssertionError("NPU prefill replay changed real-bank expert outputs")
        report["passed"] = True
        return report
    finally:
        output.write_text(json.dumps(report, indent=2) + "\n")
