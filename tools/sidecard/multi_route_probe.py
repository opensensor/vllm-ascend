# SPDX-License-Identifier: Apache-2.0
"""Explicit exclusive-lane gate for unpadded deterministic native expert decode.

Requires an authenticated candidate build receipt. Does not start a service,
create DMA mappings, repack weights, or time unqualified candidate arithmetic.
"""

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path

from .native_multi_route.contract import EXPERTS, TOP_K, WIDTH, ordered_reduce, route_cases, route_order

ACTIVATION_SCALES = (0.125, 0.75, 1.0, 3.0)


def raw_reference(candidate, layer, hidden, ids, weights):
    """Independently submit each live contribution, then reduce on the CPU.

    The main gate additionally requires bitwise equality to the frozen padded
    path. This check isolates weighting, duplicate multiplicity, and ordering.
    """
    t = candidate.torch
    values = t.full((TOP_K, WIDTH), float("nan"), dtype=t.float16)
    for slot in route_order(ids, weights, candidate.actor.rank):
        route_id = t.tensor([[ids[slot]]], dtype=t.int32, device=hidden.device)
        route_weight = t.ones((1, 1), dtype=t.float32, device=hidden.device)
        value = candidate.native(
            hidden, *candidate.actor.banks[layer], route_weight, route_id, candidate.actor.rank * EXPERTS
        )
        values[slot] = value.cpu()[0].half()
    return ordered_reduce(values, ids, weights, candidate.actor.rank)[None]


def validate(candidate, layer, output):
    t = candidate.torch
    report = dict(
        passed=False, rank=candidate.actor.rank, layer=layer, cases=[], precision="A4/W2-W3-W4 FP16-scale frozen banks"
    )
    try:
        for case_index, case in enumerate(route_cases(candidate.actor.rank, layer)):
            for scale in ACTIVATION_SCALES:
                t.manual_seed(20261014 + layer * 101 + case_index)
                hidden = (t.randn((1, WIDTH), device=f"npu:{candidate.actor.rank}") * scale).half()
                ids = t.tensor([case["ids"]], dtype=t.int32, device=hidden.device)
                weights = t.tensor([case["weights"]], dtype=t.float32, device=hidden.device)
                expected = candidate.actor(layer, hidden, ids, weights).cpu()
                actual = candidate(layer, hidden, ids, weights).cpu()
                repeated = candidate(layer, hidden, ids, weights).cpu()
                equal = actual.numpy().tobytes() == expected.numpy().tobytes()
                repeat_equal = actual.numpy().tobytes() == repeated.numpy().tobytes()
                reference_equal = None
                # Raw one-route reconstruction is deliberately outside timing.
                if scale == ACTIVATION_SCALES[0] and case["name"] in (
                    "owned-8",
                    "duplicate-eight",
                    "duplicate-peer-zero",
                ):
                    reference = raw_reference(candidate, layer, hidden, case["ids"], case["weights"])
                    reference_equal = t.equal(actual, reference)
                report["cases"].append(
                    dict(
                        name=case["name"],
                        scale=scale,
                        owned_contributions=len(route_order(case["ids"], case["weights"], candidate.actor.rank)),
                        bitwise_equal=equal,
                        repeat_equal=repeat_equal,
                        independent_route_reference_equal=reference_equal,
                        finite=bool(actual.isfinite().all()),
                        max_abs_error=float((actual - expected).abs().max()),
                        output_sha256=hashlib.sha256(actual.numpy().tobytes()).hexdigest(),
                    )
                )
                if not equal or not repeat_equal or reference_equal is False or not bool(actual.isfinite().all()):
                    raise AssertionError("multi-route output differs from qualified arithmetic")
        report["passed"] = True
        return report
    finally:
        output.write_text(json.dumps(report, indent=2) + "\n")


def run(config, output, rank):
    if type(rank) is not int or rank not in (0, 1) or os.environ.get("ASCEND_RT_VISIBLE_DEVICES") != "2,3":
        raise ValueError("multi-route qualification requires assigned physical chips 2,3")
    spec = json.loads(config.read_bytes())
    candidate_spec = spec["multi_route_candidate"]
    # Admission stays ahead of device imports and bank allocation.
    from .native_multi_route.adapter import MultiRouteDecode, authenticate_candidate

    authenticate_candidate(Path(candidate_spec["root"]), candidate_spec["manifest_sha256"])
    import torch
    import torch_npu

    from .layer_weights import NPU_LAYERS, NpuExperts

    output.mkdir(exist_ok=False)
    torch.set_num_threads(2)
    torch.npu.set_device(rank)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    layers = []
    with torch.inference_mode():
        for layer in NPU_LAYERS:
            actor = NpuExperts(
                Path(spec["checkpoint"]),
                spec["checkpoint_sha256"],
                rank,
                torch,
                gate_override=spec["native_gate_override"],
                reduce_results=False,
                layers=(layer,),
            )
            candidate = MultiRouteDecode(actor, Path(candidate_spec["root"]), candidate_spec["manifest_sha256"])
            report = validate(candidate, layer, output / f"layer-{layer}.json")
            layers.append(dict(layer=layer, passed=report["passed"], comparisons=len(report["cases"])))
            del candidate, actor
            gc.collect()
            torch.npu.synchronize()
            torch.npu.empty_cache()
    report = dict(
        passed=True,
        rank=rank,
        physical_chip=rank + 2,
        dma_mappings_created=False,
        layers=layers,
        candidate_build_sha256=candidate_spec["manifest_sha256"],
        eligible_r1_owned_contribution_counts=list(range(TOP_K + 1)),
        r1_shape_coverage_percent=100.0,
        production_empirical_coverage_percent=None,
        previous_single_route_observed_percent=7.2,
        performance_validated=False,
        promoted=False,
    )
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rank", type=int, choices=(0, 1), required=True)
    args = parser.parse_args()
    run(args.config, args.output, args.rank)


if __name__ == "__main__":
    main()
