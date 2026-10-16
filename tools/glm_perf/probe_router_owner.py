# SPDX-License-Identifier: Apache-2.0
"""Explicit HCCL/Gloo router-packet gate; does not load or qualify a model."""

import argparse
import json
import os
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist

from .router_owner import RouterExchange


def run(backend, tp_size, output, graphs=False):
    if backend not in ("gloo", "hccl") or tp_size not in (2, 4) or graphs and backend != "hccl":
        raise ValueError("TP2/TP4 and HCCL-only graph capture required")
    device = torch.device("cpu")
    if backend == "hccl":
        import torch_npu  # noqa: F401

        device = torch.device("npu", int(os.environ["LOCAL_RANK"]))
        torch.npu.set_device(device)
    dist.init_process_group("gloo", timeout=timedelta(seconds=60))
    rank, world = dist.get_rank(), dist.get_world_size()
    groups = []
    try:
        if world % tp_size:
            raise ValueError("complete local TP groups required")
        for first in range(0, world, tp_size):
            members = list(range(first, first + tp_size))
            current = dist.new_group(members, backend=backend, timeout=timedelta(seconds=60))
            if rank in members:
                group = current
                groups.append(group)
        stage, local = divmod(rank, tp_size)
        owner = tp_size - 1
        exchange = RouterExchange(
            group=group,
            rank=local,
            size=tp_size,
            owner_rank=owner,
            max_tokens=640,
            top_k=8,
            device=device,
            budget_bytes=640 * 8 * 12 * (tp_size + 1),
        )
        exchange.warmup()
        cases = []
        for rows in (1, 4, 37, 640):
            ids_cpu = torch.arange(rows * 8, dtype=torch.int64).reshape(rows, 8) + 16_777_217 + stage * 10000
            weights_cpu = torch.linspace(-1, 1, rows * 8).reshape(rows, 8)
            ids, weights = ids_cpu.to(device), weights_cpu.to(device)

            def transfer(rows=rows, ids=ids, weights=weights):
                exchange.post(rows, (ids, weights) if local == owner else None, overlap=rows > 8)
                return exchange.finish()

            def check(result, ids_cpu=ids_cpu, weights_cpu=weights_cpu):
                if backend == "hccl":
                    torch.npu.synchronize()
                actual = tuple(value.cpu() for value in result)
                if not torch.equal(actual[0], ids_cpu) or not torch.equal(actual[1], weights_cpu):
                    raise AssertionError("router bytes changed or inputs were stale")

            for step in range(2):
                ids_cpu.add_(1)
                weights_cpu.mul_(0.5)
                ids.copy_(ids_cpu)
                weights.copy_(weights_cpu)
                check(transfer())
            graph_parity = None
            if graphs and rows <= 4:
                graph = torch.npu.NPUGraph()
                with torch.npu.graph(graph):
                    captured = transfer()
                ids_cpu.add_(73)
                weights_cpu.neg_()
                ids.copy_(ids_cpu)
                weights.copy_(weights_cpu)
                graph.replay()
                check(captured)
                graph_parity = True
                del graph, captured
            cases.append(dict(tokens=rows, byte_parity=True, graph_parity=graph_parity))
            print(f"Router packet passed: rank={rank}, tokens={rows}, graph={graph_parity}", flush=True)
        output.mkdir(parents=True, exist_ok=True)
        (output / f"rank-{rank}.json").write_text(
            json.dumps(
                dict(
                    backend=backend,
                    rank=rank,
                    stage=stage,
                    local_rank=local,
                    owner_rank=owner,
                    tp_size=tp_size,
                    cases=cases,
                    evidence="synthetic router transport/capture only; real model and timing unqualified",
                ),
                indent=2,
            )
            + "\n"
        )
    finally:
        for group in groups:
            dist.destroy_process_group(group)
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("gloo", "hccl"), default="gloo")
    parser.add_argument("--tp-size", type=int, choices=(2, 4), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--graphs", action="store_true")
    run(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
