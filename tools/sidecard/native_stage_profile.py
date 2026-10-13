# SPDX-License-Identifier: Apache-2.0
"""Profile frozen Ascend cold-expert stages with production bank geometry."""

import argparse
import hashlib
import json
import os
import statistics
import time
from functools import partial
from pathlib import Path

from .cold_placement import load
from .expert_decode_graph import ExpertDecodeGraphs
from .layer_weights import NPU_LAYERS, NpuExperts

SAMPLES = 7
COLD_ROUTES = (1, 4, 8)


def validate_layers(layers):
    if (
        not layers
        or len(set(layers)) != len(layers)
        or any(type(layer) is not int or layer not in NPU_LAYERS for layer in layers)
    ):
        raise ValueError("native stage profile requires unique supported cold layers")


def run(config, output, layers):
    """Observe only assigned chips; no DMA mapping or serving state is changed."""
    validate_layers(layers)
    if os.environ.get("ASCEND_RT_VISIBLE_DEVICES") != "2,3":
        raise ValueError("native stage profile must preserve the other Ascend card")
    import torch
    import torch_npu

    from tools.glm_perf.fused_moe_profile import measure

    spec = json.loads(config.read_bytes())
    hot = load(spec)
    torch.set_num_threads(2)
    torch.npu.set_device(0)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    cases = []
    with torch.inference_mode():
        for layer in layers:
            actor = NpuExperts(
                Path(spec["checkpoint"]),
                spec["checkpoint_sha256"],
                0,
                torch,
                gate_override=spec["native_gate_override"],
                reduce_results=False,
                layers=(layer,),
            )
            captured = ExpertDecodeGraphs(actor, torch)
            stable_rows = actor.stable_rows
            try:
                actor.stable_rows = 1
                single = ExpertDecodeGraphs(actor, torch)
            finally:
                actor.stable_rows = stable_rows
            owned = [expert for expert in range(144) if expert not in hot[str(layer)]]
            ids = torch.tensor([owned[:8]], dtype=torch.int32, device="npu:0")
            torch.manual_seed(21717 + layer)
            hidden = torch.randn(1, 4096, device="npu:0").half()
            for count in COLD_ROUTES:
                weights = torch.zeros((1, 8), device="npu:0")
                weights[:, :count] = 1 / count

                pipeline = partial(actor, layer, hidden, ids, weights)
                expected_tensor = pipeline().cpu()
                if not bool(expected_tensor.isfinite().all()):
                    raise AssertionError("native stage probe produced non-finite output")
                expected = expected_tensor.numpy().tobytes()
                stage_spans = measure(actor.native, pipeline, warmups=3, samples=SAMPLES)
                actual = captured(layer, hidden, ids, weights).cpu().numpy().tobytes()
                if actual != expected:
                    raise AssertionError("native replay changed stable expert output bytes")
                replay_ms = []
                for _ in range(SAMPLES):
                    begin = time.perf_counter()
                    captured(layer, hidden, ids, weights)
                    torch.npu.synchronize()
                    replay_ms.append((time.perf_counter() - begin) * 1000)
                cases.append(
                    dict(
                        layer=layer,
                        cold_routes=count,
                        bank_experts=144,
                        decode_rows=1,
                        native_padded_rows=actor.stable_rows,
                        replay_bitwise_equal=True,
                        output_sha256=hashlib.sha256(actual).hexdigest(),
                        stages=stage_spans,
                        graph_replay_median_ms=statistics.median(replay_ms),
                        graph_replay_samples_ms=replay_ms,
                    )
                )
                if count == 1:
                    comparison = []
                    single_ms = []
                    for trial in range(SAMPLES):
                        changed = hidden * (0.5 + trial / SAMPLES)
                        expected = captured(layer, changed, ids, weights).cpu()
                        begin = time.perf_counter()
                        observed = single(layer, changed, ids, weights)
                        torch.npu.synchronize()
                        single_ms.append((time.perf_counter() - begin) * 1000)
                        observed = observed.cpu()
                        comparison.append(
                            dict(
                                bitwise_equal=observed.numpy().tobytes() == expected.numpy().tobytes(),
                                max_abs_error=float((observed - expected).abs().max()),
                            )
                        )
                    cases[-1]["single_route_atomic_candidate"] = dict(
                        qualified=all(row["bitwise_equal"] for row in comparison),
                        graph_replay_median_ms=statistics.median(single_ms),
                        samples=single_ms,
                        comparisons=comparison,
                        production_enabled=False,
                    )
                output.write_text(json.dumps(dict(passed=False, cases=cases), indent=2) + "\n")
            captured.close()
            single.close()
            del captured, single, actor, pipeline
            torch.npu.synchronize()
            torch.npu.empty_cache()
    report = dict(
        passed=True,
        physical_npu_device=2,
        dma_mappings_created=False,
        production_speed_validated=False,
        scope="Real 144-expert banks; synthetic cold routes and activations; stage events use eager submission",
        cases=cases,
    )
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(passed=True, cases=len(cases))), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", nargs="+", type=int, default=[18, 24, 31])
    args = parser.parse_args()
    run(args.config, args.output, args.layers)


if __name__ == "__main__":
    main()
