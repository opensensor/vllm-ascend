# SPDX-License-Identifier: Apache-2.0
"""Require exact unpadded expert output for every resident layer and route slot."""

import hashlib
import json

SCALES = (0.125, 0.75, 1.0, 3.0)
TOP_K = 8
WIDTH = 4096
EXPERTS_PER_RANK = 144


def validate(graphs, torch, output):
    report = dict(passed=False, rank=graphs.actor.rank, cases=[])
    try:
        device = f"npu:{graphs.actor.rank}"
        for layer in graphs.actor.banks:
            for slot in range(TOP_K):
                for scale in SCALES:
                    torch.manual_seed(22519 + layer * 101 + slot)
                    hidden = torch.randn((1, WIDTH), device=device).half() * scale
                    # Peer contributions remain nonzero in the global input;
                    # each owner must correctly suppress them before reduction.
                    peer = (1 - graphs.actor.rank) * EXPERTS_PER_RANK
                    ids = torch.arange(TOP_K, dtype=torch.int32, device=device)[None] + peer
                    ids[0, slot] = graphs.actor.rank * EXPERTS_PER_RANK + (layer * 13 + slot * 17) % EXPERTS_PER_RANK
                    weights = torch.full((1, TOP_K), 0.137 + slot * 0.013, device=device)
                    expected = graphs(layer, hidden, ids, weights).cpu()
                    actual = graphs(layer, hidden, ids, weights, owned_routes=1).cpu()
                    equal = actual.numpy().tobytes() == expected.numpy().tobytes()
                    report["cases"].append(
                        dict(
                            layer=layer,
                            slot=slot,
                            scale=scale,
                            bitwise_equal=equal,
                            max_abs_error=float((actual - expected).abs().max()),
                            output_sha256=hashlib.sha256(actual.numpy().tobytes()).hexdigest(),
                        )
                    )
                    if not equal or not bool(actual.isfinite().all()):
                        raise AssertionError("single owned contribution changed qualified expert output")
        report["passed"] = True
        return report
    finally:
        output.write_text(json.dumps(report, indent=2) + "\n")
