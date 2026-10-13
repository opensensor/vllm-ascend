# SPDX-License-Identifier: Apache-2.0
"""Qualify the one-contribution decode graph without stopping the GLM service."""

import argparse
import gc
import json
import os
from pathlib import Path

from .layer_weights import NPU_LAYERS, NpuExperts
from .single_route_gate import validate
from .single_route_graph import SingleRouteDecodeGraphs


def run(config, output, rank):
    if type(rank) is not int or rank not in (0, 1) or os.environ.get("ASCEND_RT_VISIBLE_DEVICES") != "2,3":
        raise ValueError("single-route qualification requires assigned physical chips 2,3")
    import torch
    import torch_npu

    spec = json.loads(config.read_bytes())
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
            graphs = SingleRouteDecodeGraphs(actor, torch)
            report = validate(graphs, torch, output / f"layer-{layer}.json")
            layers.append(dict(layer=layer, passed=report["passed"], comparisons=len(report["cases"])))
            graphs.close()
            del graphs, actor
            gc.collect()
            torch.npu.synchronize()
            torch.npu.empty_cache()
    report = dict(passed=True, rank=rank, physical_chip=rank + 2, dma_mappings_created=False, layers=layers)
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
