# SPDX-License-Identifier: Apache-2.0
"""Compare complete indexer projection bytes and graph replay on real weights."""

import argparse
import gc
import hashlib
import json
import statistics
import time
from functools import partial
from pathlib import Path

from .indexer_projection_graph import IndexerProjectionGraph, project
from .layer_indexer import PoolIndexer
from .layer_weights import load_dense

LAYERS = (19, 43)
SCALES = (0.0, 0.125, 0.75, 1.0, 3.0)


def timed(torch, operation):
    spans = []
    for _ in range(25):
        torch.cuda.synchronize()
        begin = time.perf_counter()
        operation()
        torch.cuda.synchronize()
        spans.append((time.perf_counter() - begin) * 1e6)
    return statistics.median(spans)


def run(config, output):
    import torch

    spec = json.loads(config.read_bytes())
    report = dict(passed=False, cases=[], measurements=[], full_model_validated=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.cuda.set_device(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    model = json.loads((Path(spec["checkpoint"]) / "config.json").read_bytes())["text_config"]
    try:
        with torch.inference_mode():
            for layer in LAYERS:
                weights, _ = load_dense(spec["checkpoint"], (layer,), "cuda", torch)
                indexer = PoolIndexer(weights[layer], model, "cuda", torch, 32, fused_cuda_rotation=True)
                captured = IndexerProjectionGraph(indexer)
                for repeat in range(4):
                    for scale in SCALES:
                        torch.manual_seed(937 + layer * 11 + repeat)
                        hidden = torch.randn_like(captured.inputs[0]) * scale
                        qr = torch.randn_like(captured.inputs[1]) * scale
                        expected = [x.cpu() for x in project(indexer, hidden, qr)]
                        actual = [x.cpu() for x in captured(hidden, qr)]
                        equal = all(
                            a.numpy().tobytes() == b.numpy().tobytes() for a, b in zip(actual, expected, strict=True)
                        )
                        report["cases"].append(
                            dict(
                                layer=layer,
                                repeat=repeat,
                                scale=scale,
                                all_four_outputs_bitwise_equal=equal,
                                output_sha256=[hashlib.sha256(x.numpy().tobytes()).hexdigest() for x in actual],
                            )
                        )
                        if not equal or not all(bool(x.isfinite().all()) for x in actual):
                            raise AssertionError("captured indexer projection changed qualified math")
                report["measurements"].append(
                    dict(
                        layer=layer,
                        eager_host_microseconds=timed(torch, partial(project, indexer, hidden, qr)),
                        graph_host_microseconds=timed(torch, partial(captured, hidden, qr)),
                    )
                )
                captured.close()
                del captured, indexer, weights, expected, actual, hidden, qr
                gc.collect()
                torch.cuda.empty_cache()
        report["passed"] = True
    except Exception as error:
        report["error"] = dict(type=type(error).__name__, message=str(error))
        raise
    finally:
        output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            dict(passed=report["passed"], comparisons=len(report["cases"]), measurements=report["measurements"])
        ),
        flush=True,
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.config, args.output)


if __name__ == "__main__":
    main()
