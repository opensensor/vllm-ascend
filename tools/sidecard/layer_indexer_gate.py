# SPDX-License-Identifier: Apache-2.0
"""Real-weight sparse selection against the existing GLM kpool CPU operators."""

import argparse
import importlib.util
import json
import os
from pathlib import Path

from .layer_indexer import PoolIndexer
from .layer_weights import load_dense

PREFIX_ROWS = 2048
GATE_ROWS = (1, 4, 32)
LAYER = 19


def run(args):
    import torch

    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    if args.device.startswith("npu"):
        if os.environ.get("ASCEND_RT_VISIBLE_DEVICES") != "2,3":
            raise ValueError("sparse gate must preserve the other Ascend card")
        import torch_npu

        torch.npu.set_device(int(args.device.split(":")[1]))
        torch_npu.npu.set_compile_mode(jit_compile=False)
    torch.set_num_threads(2)
    config = json.loads((args.checkpoint / "config.json").read_bytes())["text_config"]
    weights, _ = load_dense(args.checkpoint, (LAYER,), "cpu", torch)
    weights = weights[LAYER]
    spec = importlib.util.spec_from_file_location("sidecard_kpool_golden", args.reference)
    golden = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(golden)
    device_weights = {name: value.to(args.device) for name, value in weights.items()}
    cases = []
    with torch.inference_mode():
        for phase in range(2):
            torch.manual_seed(20261015 + phase)
            total = PREFIX_ROWS + sum(GATE_ROWS)
            hidden = torch.randn(total, 4096).half() * (0.5 + phase)
            qr = torch.randn(total, 1536).half()
            cache = PoolIndexer(device_weights, config, args.device, torch, 4096)
            for start in range(0, PREFIX_ROWS, 32):
                cache.select(hidden[start : start + 32].to(args.device), qr[start : start + 32].to(args.device), start)
            first = PREFIX_ROWS
            for rows in GATE_ROWS:
                end = first + rows
                actual = cache.select(hidden[first:end].to(args.device), qr[first:end].to(args.device), first).cpu()
                w = {name: value.float() for name, value in weights.items()}
                keys = hidden[:end].float() @ w["self_attn.indexer.wk.weight"].t()
                keys = torch.nn.functional.layer_norm(
                    keys, (128,), w["self_attn.indexer.k_norm.weight"], w["self_attn.indexer.k_norm.bias"], 1e-6
                )
                gates = hidden[:end].float() @ w["self_attn.indexer.index_kpool_compress_gate"].t()
                complete = end // 4 * 4
                pools = golden.compress_kpool(
                    keys[:complete].reshape(-1, 4, 128),
                    gates[:complete].reshape(-1, 4, 128),
                    w["self_attn.indexer.index_kpool_compress_ape"],
                )
                queries = (
                    torch.nn.functional.linear(qr[first:end], weights["self_attn.indexer.wq_b.weight"])
                    .reshape(rows, config["index_n_heads"], 128)
                    .bfloat16()
                )
                head_weights = hidden[first:end].float() @ w["self_attn.indexer.weights_proj.weight"].t()
                head_weights *= (128 * config["index_n_heads"]) ** -0.5
                expected = golden.score_and_select_kpool_tokens(
                    queries, head_weights, pools.half(), torch.arange(first, end), 2048, 4
                )
                equal_sets = all(
                    set(a[a >= 0].tolist()) == set(b[b >= 0].tolist()) for a, b in zip(actual, expected, strict=True)
                )
                causal = all(bool((row[row >= 0] <= first + offset).all()) for offset, row in enumerate(actual))
                pool_error = (cache.pools[: cache.pool_count].float().cpu() - pools.float()).abs().max()
                cases.append(
                    dict(
                        phase=phase,
                        rows=rows,
                        first_position=first,
                        selected_token_sets_equal=equal_sets,
                        causal=causal,
                        pool_max_abs_error=float(pool_error),
                        passed=equal_sets and causal,
                    )
                )
                first = end
    report = dict(
        passed=all(case["passed"] for case in cases),
        device=args.device,
        real_checkpoint_weights=True,
        cases=cases,
        reference=str(args.reference),
        scope="Actual sparse selection beyond 2048 tokens; changed inputs and incomplete pool tails",
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(device=args.device, passed=report["passed"])), flush=True)
    if not report["passed"]:
        raise RuntimeError("real sparse indexer parity failed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "npu:0", "npu:1"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
