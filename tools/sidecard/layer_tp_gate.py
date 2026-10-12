# SPDX-License-Identifier: Apache-2.0
"""Real middle-block TP2 projections against complete CPU layer references."""

import json
import time

from .layer_math import TextLayer
from .layer_math_gate import check
from .layer_weights import load_dense

GATE_LAYERS = (18, 19)
GATE_ROWS = (1, 4, 32)


def validate_tp(layers, root, rank, torch, output):
    config = layers[0].config
    weights, _ = load_dense(root, GATE_LAYERS, "cpu", torch)
    reference = {index: TextLayer(index, weights[index], config, "cpu", torch, None, 2048) for index in GATE_LAYERS}
    owned = {layer.index: layer for layer in layers}
    cases = []
    with torch.inference_mode():
        for phase in range(2):
            for rows in GATE_ROWS:
                torch.manual_seed(20261013 + rows + phase)
                hidden = torch.randn(rows, 4096).half() * (0.5 + phase)
                for index in GATE_LAYERS:
                    layer, golden = owned[index], reference[index]
                    layer.reset()
                    golden.reset()
                    device_input = hidden.to(layer.device)
                    for kind in ("attention", "shared_mlp"):
                        start = time.perf_counter()
                        if kind == "attention":
                            actual = layer.kda(device_input) if layer.linear else layer.dsa(device_input)
                            expected = golden.kda(hidden) if golden.linear else golden.dsa(hidden)
                        else:
                            actual = layer.dense_mlp(device_input, "mlp.shared_experts")
                            expected = golden.dense_mlp(hidden, "mlp.shared_experts")
                        actual = actual.float()
                        torch.distributed.all_reduce(actual)
                        result = check(actual, expected, rtol=0.025, atol=0.02, torch=torch)
                        cases.append(
                            dict(
                                layer=index,
                                phase=phase,
                                rows=rows,
                                kind=kind,
                                wall_seconds=time.perf_counter() - start,
                                **result,
                            )
                        )
    for layer in layers:
        layer.reset()
    report = dict(
        passed=all(case["passed"] for case in cases),
        rank=rank,
        cases=cases,
        scope="Real W4 middle-block KDA/DSA and shared MLP; full CPU references versus summed TP2 shards",
    )
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(kind="middle_tp2_gate", rank=rank, passed=report["passed"], cases=len(cases))), flush=True)
    if not report["passed"]:
        raise RuntimeError("real TP2 middle component parity failed")
