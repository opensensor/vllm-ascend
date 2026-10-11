# SPDX-License-Identifier: Apache-2.0
"""CPU torchrun gate for group-local router packets and full head-score sums."""

import argparse
import importlib.util
import json
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist

from .indexer_heads import reduced_score
from .router_owner import RouterExchange


def run(tp_size, output):
    dist.init_process_group("gloo", timeout=timedelta(seconds=60))
    rank, world = dist.get_rank(), dist.get_world_size()
    try:
        if tp_size not in (2, 4) or world % tp_size:
            raise ValueError("complete TP2/TP4 groups required")
        group = None
        for first in range(0, world, tp_size):
            members = list(range(first, first + tp_size))
            current = dist.new_group(members, backend="gloo", timeout=timedelta(seconds=60))
            if rank in members:
                group = current
        stage, local = divmod(rank, tp_size)
        owner = tp_size - 1
        exchange = RouterExchange(
            group=group,
            rank=local,
            size=tp_size,
            owner_rank=owner,
            max_tokens=640,
            top_k=8,
            device="cpu",
            budget_bytes=640 * 8 * 12 * (tp_size + 1),
        )
        path = Path(__file__).resolve().parents[2] / "vllm_ascend/models/glm5next/kpool_ops.py"
        spec = importlib.util.spec_from_file_location("probe_kpool", path)
        kpool = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(kpool)
        score = reduced_score(kpool.score_kpool, group=group)
        heads = 32 // tp_size
        cases = []
        exchange.warmup()
        for rows in (1, 4, 37, 640):
            for step in (0, 1):
                torch.manual_seed(53 + stage * 100 + step)
                ids = torch.arange(rows * 8, dtype=torch.int64).reshape(rows, 8) + 16_777_217 + stage * 10000 + step
                weights = torch.randn(rows, 8)
                exchange.post(rows, (ids, weights) if local == owner else None, overlap=rows > 8)
                q = torch.randn(rows, 32, 128).bfloat16()
                w = torch.randn(rows, 32)
                keys = torch.randn(193, 128).half()
                got_ids, got_weights = exchange.finish()
                assert torch.equal(got_ids, ids) and torch.equal(got_weights, weights)
                part = slice(local * heads, (local + 1) * heads)
                logits = score(q[:, part], w[:, part], keys)
                reference = kpool.score_kpool(q, w, keys)
                torch.testing.assert_close(logits, reference, rtol=1e-4, atol=5e-5)
                agreement = (logits.argmax(1) == reference.argmax(1)).float().mean().item()
                assert agreement == 1
                cases.append(
                    dict(
                        rows=rows,
                        step=step,
                        max_abs=float((logits - reference).abs().max()),
                        argmax_agreement=agreement,
                    )
                )
        output.mkdir(parents=True, exist_ok=True)
        (output / f"rank-{rank}.json").write_text(
            json.dumps(
                dict(
                    rank=rank,
                    stage=stage,
                    local_rank=local,
                    tp_size=tp_size,
                    owner_rank=owner,
                    backend="gloo",
                    hardware_qualified=False,
                    cases=cases,
                ),
                indent=2,
            )
            + "\n"
        )
    finally:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tp-size", type=int, choices=(2, 4), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.tp_size, args.output)


if __name__ == "__main__":
    main()
